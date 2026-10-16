"""Low-memory training primitives: 8-bit optimizer state and bf16 weights.

Memory per parameter during training (bytes):

    config                              weights  grads  AdamW state   Muon state
    fp32 everything                        4       4        8             4
    bf16 weights + 8-bit state (here)      2       2        2             1

8-bit state uses blockwise quantization (Dettmers et al., 2021): each block of
`block` consecutive values has its own fp32 scale, so one outlier cannot wipe
out the precision of its neighbours. Two value maps are provided:

  * "sqrt" (default, signed): q = sign(x) * round(sqrt(|x| / absmax) * 127).
    The square-root companding gives small values finer steps than a linear
    int8 map, which matters because most momentum entries are far below the
    block maximum.
  * "log" (non-negative, for Adam's second moment): uint8 over
    [min log v, max log v] in the block, so a block spanning many orders of
    magnitude keeps ~2 % relative precision everywhere.

bf16 weights without an fp32 master copy lose small updates to rounding
(lr * update is often below bf16's 8-bit mantissa step). Stochastic rounding
(Zamirai et al., 2020) makes the rounding unbiased: the expected weight after
many steps equals the fp32 trajectory. `to_bf16_stochastic` implements it on
the raw bit pattern.
"""
import math

import torch

QBLOCK = 256


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


def quantize_log(v, block=QBLOCK, eps=1e-30):
    """non-negative tensor -> (uint8 codes, fp32 lo, fp32 hi) in log space."""
    b, _ = _blocks(v.float(), block)
    lv = (b + eps).log()
    lo = lv.amin(dim=1)
    hi = lv.amax(dim=1)
    span = (hi - lo).clamp_min(1e-12)
    q = torch.round((lv - lo[:, None]) / span[:, None] * 255).to(torch.uint8)
    return q, lo, hi


def dequantize_log(q, lo, hi, shape, eps=1e-30):
    span = (hi - lo).clamp_min(1e-12)
    lv = lo[:, None] + q.float() / 255 * span[:, None]
    return (lv.exp() - eps).clamp_min(0).reshape(-1)[:math.prod(shape)].view(shape)


class QState:
    """A quantized optimizer buffer with get()/set() in fp32."""

    def __init__(self, like, kind="signed", block=QBLOCK, vmap="sqrt"):
        self.shape, self.kind, self.block, self.vmap = tuple(like.shape), kind, block, vmap
        self.set(torch.zeros_like(like, dtype=torch.float32))

    def set(self, x):
        if self.kind == "signed":
            self.q, self.s = quantize_signed(x, self.block, self.vmap)
        else:
            self.q, self.lo, self.hi = quantize_log(x, self.block)

    def get(self):
        if self.kind == "signed":
            return dequantize_signed(self.q, self.s, self.shape, self.vmap)
        return dequantize_log(self.q, self.lo, self.hi, self.shape)

    def nbytes(self):
        n = self.q.numel()
        return n + (self.s.numel() * 4 if self.kind == "signed" else self.lo.numel() * 8)


def to_bf16_stochastic(x):
    """fp32 -> bf16 with stochastic rounding (unbiased), via the bit pattern."""
    assert x.dtype == torch.float32
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x.device)
    rounded = (bits + noise) & -65536          # keep the top 16 bits
    return rounded.view(torch.float32).to(torch.bfloat16)


def write_param(p, new_fp32, stochastic=True):
    """Store an fp32 result into parameter p in p's dtype (bf16 -> stochastic rounding)."""
    if p.dtype == torch.bfloat16 and stochastic:
        p.copy_(to_bf16_stochastic(new_fp32))
    else:
        p.copy_(new_fp32.to(p.dtype))


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
