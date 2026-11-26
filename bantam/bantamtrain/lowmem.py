"""Low-memory training primitives: 8-bit optimizer state and bf16 weights.

Memory per parameter during training (bytes):

    config                              weights  grads  AdamW state   Muon state
    fp32 everything                        4       4        8             4
    bf16 weights + 8-bit state (here)      2       2        2             1

The optimizers (bantamtrain.optim) update the state in chunks of a few million
elements, so the fp32 working copies of a step are O(chunk), not a multiple of
the largest parameter (Muon still needs fp32 copies of each matrix for
Newton-Schulz).

8-bit state uses blockwise quantization (Dettmers et al., 2021): each block of
`block` consecutive values has its own fp32 scale, so one outlier cannot wipe
out the precision of its neighbours. Two value maps are provided:

  * "sqrt" (default, signed): q = sign(x) * round(sqrt(|x| / absmax) * 127).
    The square-root companding gives small values finer steps than a linear
    int8 map, which matters because most momentum entries are far below the
    block maximum.
  * "log" (non-negative, for Adam's second moment): code 0 is exact zero and
    codes 1..255 cover [min log v, max log v] over the block's positive
    entries, so a block spanning many orders of magnitude keeps ~2 % relative
    precision everywhere (0.45 % for a block spanning one decade). Optional
    stochastic rounding of the codes keeps the stored EMA unbiased when its
    per-step change is far below one code step (Adam with beta2 > 0.95).

bf16 weights without an fp32 master copy lose small updates to rounding
(lr * update is often below bf16's 8-bit mantissa step). Stochastic rounding
(Zamirai et al., 2020) makes the rounding unbiased: the expected weight after
many steps equals the fp32 trajectory. `to_bf16_stochastic` implements it on
the raw bit pattern. The optimizers pass a generator seeded from
(seed, step, parameter), so every DDP rank rounds identically and a resumed
run draws the same noise as an uninterrupted one.
"""
import math

import torch

QBLOCK = 256
LOG_LEVELS = 254            # log map: codes 1..255 = 254 steps over [lo, hi]; 0 = exact zero


def _blocks(x, block):
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, block), pad


def quantize_signed(x, block=QBLOCK, vmap="sqrt"):
    """float tensor -> (int8 codes (n_blocks, block), fp32 absmax (n_blocks,))."""
    b, _ = _blocks(x.float(), block)
    absmax = b.abs().amax(dim=1).clamp_min(1e-30)
    n = b / absmax[:, None]
    if vmap == "sqrt":
        n = n.sign() * n.abs().sqrt()
    q = torch.round(n * 127).to(torch.int8)
    return q, absmax


def dequantize_signed(q, absmax, shape, vmap="sqrt"):
    n = q.float() / 127
    if vmap == "sqrt":
        n = n.sign() * n * n
    return (n * absmax[:, None]).reshape(-1)[:math.prod(shape)].view(shape)


def quantize_log(v, block=QBLOCK, eps=1e-30, stochastic=False, generator=None):
    """non-negative tensor -> (uint8 codes, fp32 lo, fp32 hi) in log space.

    Code 0 is exact zero (also the zero padding of a partial last block), so
    zeros never stretch [lo, hi]. stochastic=True rounds a code up with the
    probability that makes the dequantized value unbiased in linear space,
    drawing from `generator`. (eps is unused, kept for compatibility.)
    """
    b, _ = _blocks(v.float(), block)
    zero = b <= 0
    lv = b.masked_fill(zero, math.inf).log_()                  # zeros -> +inf, masked below
    hi = b.amax(dim=1).log_()
    lo = lv.amin(dim=1)
    empty = ~torch.isfinite(lo)                                 # all-zero block
    lo, hi = lo.masked_fill(empty, 0.0), hi.masked_fill(empty, 0.0)
    span = (hi - lo).clamp_min(1e-12)
    t = lv.sub_(lo[:, None]).mul_((LOG_LEVELS / span)[:, None]).clamp_(0, LOG_LEVELS)
    if stochastic:
        k = t.floor()
        d = (span / LOG_LEVELS)[:, None]
        p_up = torch.expm1((t - k) * d) / torch.expm1(d)       # (v - L) / (U - L)
        t = k.add_(torch.rand(t.shape, generator=generator, device=t.device) < p_up)
    q = t.round_().add_(1).masked_fill_(zero, 0.0).to(torch.uint8)
    return q, lo, hi


def dequantize_log(q, lo, hi, shape, eps=1e-30):
    span = (hi - lo).clamp_min(1e-12)
    v = q.float().sub_(1).mul_((span / LOG_LEVELS)[:, None]).add_(lo[:, None]).exp_()
    return v.masked_fill_(q == 0, 0.0).reshape(-1)[:math.prod(shape)].view(shape)


def _dequantize_log_v1(q, lo, hi, shape, eps=1e-30):
    """The log map of bantamtrain <= 0.2 (codes 0..255, zero padding inside [lo, hi])."""
    span = (hi - lo).clamp_min(1e-12)
    lv = lo[:, None] + q.float() / 255 * span[:, None]
    return (lv.exp() - eps).clamp_min(0).reshape(-1)[:math.prod(shape)].view(shape)


class QState:
    """A quantized optimizer buffer with get()/set() in fp32.

    get/set take an optional element range starting on a block boundary, so an
    optimizer can update a large buffer chunk by chunk.
    """

    _fmt = 1    # QStates pickled by bantamtrain <= 0.2 (old log map) lack the attribute

    def __init__(self, like, kind="signed", block=QBLOCK, vmap="sqrt"):
        self.shape, self.kind, self.block, self.vmap = tuple(like.shape), kind, block, vmap
        self._fmt = 2
        nb, dev = -(-like.numel() // block), like.device
        self.q = torch.zeros(nb, block, device=dev,
                             dtype=torch.int8 if kind == "signed" else torch.uint8)
        if kind == "signed":                    # == quantize_signed(zeros)
            self.s = torch.full((nb,), 1e-30, device=dev)
        else:
            self.lo, self.hi = torch.zeros(nb, device=dev), torch.zeros(nb, device=dev)

    @property
    def numel(self):
        return math.prod(self.shape)

    def _blocks_of(self, start, n):
        assert start % self.block == 0, "QState ranges must start on a block boundary"
        return start // self.block, start // self.block + -(-n // self.block)

    def set(self, x, start=0, stochastic=False, generator=None):
        """Quantize x into elements [start, start + x.numel()).

        stochastic: round log-map codes stochastically (unbiased); signed codes
        are rounded to nearest.
        """
        b0, b1 = self._blocks_of(start, x.numel())
        if self.kind == "signed":
            self.q[b0:b1], self.s[b0:b1] = quantize_signed(x, self.block, self.vmap)
        else:
            if self._fmt != 2 and x.numel() != self.numel:
                self.set(self.get())            # upgrade an old buffer before a partial write
            self.q[b0:b1], self.lo[b0:b1], self.hi[b0:b1] = quantize_log(
                x, self.block, stochastic=stochastic, generator=generator)
            self._fmt = 2

    def get(self, start=0, end=None):
        """fp32 copy of elements [start, end) (default: the whole buffer, shaped)."""
        whole = start == 0 and end is None
        end = self.numel if end is None else end
        b0, b1 = self._blocks_of(start, end - start)
        shape = self.shape if whole else (end - start,)
        if self.kind == "signed":
            return dequantize_signed(self.q[b0:b1], self.s[b0:b1], shape, self.vmap)
        deq = dequantize_log if self._fmt == 2 else _dequantize_log_v1
        return deq(self.q[b0:b1], self.lo[b0:b1], self.hi[b0:b1], shape)

    def nbytes(self):
        n = self.q.numel()
        return n + (self.s.numel() * 4 if self.kind == "signed" else self.lo.numel() * 8)

    def to(self, device):
        """Move codes and (fp32) scales to `device` in place; returns self."""
        self.q = self.q.to(device)
        for k in ("s", "lo", "hi"):
            if hasattr(self, k):
                setattr(self, k, getattr(self, k).to(device, torch.float32))
        return self

    def tensors(self):
        """The buffer as a dict of plain tensors (loadable with weights_only=True)."""
        if self.kind != "signed" and self._fmt != 2:
            self.set(self.get())                # re-encode an old log-map buffer
        d = {"q": self.q}
        if self.kind == "signed":
            d["s"] = self.s
            if self.vmap != "sqrt":
                d["linear"] = torch.ones((), dtype=torch.uint8)
        else:
            d["lo"], d["hi"] = self.lo, self.hi
        return d

    @classmethod
    def from_tensors(cls, shape, d, device=None):
        """Inverse of tensors(): a QState for a buffer of `shape`, moved to `device`."""
        st = cls.__new__(cls)
        st.shape, st.block, st._fmt = tuple(shape), d["q"].shape[1], 2
        st.kind = "signed" if "s" in d else "log"
        st.vmap = "linear" if "linear" in d else "sqrt"
        for k in ("q", "s", "lo", "hi"):
            if k in d:
                setattr(st, k, d[k])
        return st.to(d["q"].device if device is None else device)


def to_bf16_stochastic(x, generator=None):
    """fp32 -> bf16 with stochastic rounding (unbiased), via the bit pattern.

    generator: torch.Generator for the rounding noise (default: global RNG).
    """
    assert x.dtype == torch.float32
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x.device,
                          generator=generator)
    rounded = (bits + noise) & -65536          # keep the top 16 bits
    return rounded.view(torch.float32).to(torch.bfloat16)


def write_param(p, new_fp32, stochastic=True, generator=None):
    """Store an fp32 result into parameter p in p's dtype (bf16 -> stochastic rounding)."""
    if p.dtype == torch.bfloat16 and stochastic:
        p.copy_(to_bf16_stochastic(new_fp32, generator))
    else:
        p.copy_(new_fp32.to(p.dtype))


def load_checkpoint(path, map_location="cpu"):
    """(checkpoint, legacy): torch.load(weights_only=True). Checkpoints written by
    bantamtrain <= 0.2 pickle QState objects (before the rename as zoomtrain.lowmem.QState);
    those load with QState allow-listed (legacy=True), so no arbitrary code runs either way."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True), False
    except Exception as e:  # noqa: BLE001 - UnpicklingError or RuntimeError, by torch version
        err = e
    ser = torch.serialization
    allow = [QState]
    if torch.__version__ >= (2, 6):            # (class, "old.module.Name") entries
        allow.append((QState, "zoomtrain.lowmem.QState"))
    try:
        if hasattr(ser, "safe_globals"):       # torch >= 2.5
            with ser.safe_globals(allow):
                return torch.load(path, map_location=map_location, weights_only=True), True
        ser.add_safe_globals(allow)            # torch 2.4: process-wide (QState is plain data)
        return torch.load(path, map_location=map_location, weights_only=True), True
    except Exception:  # noqa: BLE001
        raise err from None


def state_bytes(optimizer):
    """Bytes held by optimizer state (fp32 tensors and QState buffers)."""
    total = 0
    for o in getattr(optimizer, "optimizers", [optimizer]):
        for st in o.state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
                elif isinstance(v, QState):
                    total += v.nbytes()
    return total
