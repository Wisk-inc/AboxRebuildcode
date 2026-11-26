"""Weight-only quantized layers: keep weights in int8 / int4 in RAM, compute in float.

    from bantamtrain.qweights import quantize_model, load_quantized
    report = quantize_model(model, bits=4, block=64)   # in place
    print(report)                                      # bytes before / after
    model = load_quantized("HuggingFaceTB/SmolLM2-135M", "nf4")   # never holds the float model

Each weight matrix is split into blocks of `block` consecutive input columns;
every block stores one scale and codes:

    bits=8:  int8 in [-127, 127]                           ~1.02 B/param (block 128)
    bits=4:  linear int4 in [-7, 7], two codes per byte    ~0.53 B/param (block 64)
    "nf4":   4-bit NormalFloat (QLoRA, Dettmers 2023): 16 codebook values at the
             quantiles of N(0,1), two codes per byte      ~0.53 B/param (block 64)
NF4 is the right choice for pretrained weights (roughly Gaussian per block);
linear int4 wastes codes on the rarely used extremes. Scales are fp16, or fp32 for a
tensor whose scales leave the fp16 normal range, and keep their dtype through
model.to(dtype). The block is the requested size, or the largest divisor of the row
width between block/2 and block (576 at 128 -> 96), so common widths need no padding.

QLinear multiplies through a custom autograd Function that saves only the codes and
scales: the weight is dequantized in the input's compute dtype (the autocast dtype
under autocast), in row chunks for big layers, in forward and again in backward. A
frozen 4-bit base therefore costs ~0.53 B/param while training too (QLoRA); no weight
gradient is computed. QEmbedding dequantizes only the rows it gathers, and a tied output
head reuses the embedding's codes (stored once). nn.Linear / nn.Embedding subclasses
with their own forward (Gemma's scaled embeddings, OPT's learned positions, routers)
keep their class and forward; only their weight storage is replaced. transformers'
Conv1D (GPT-2) is quantized like a Linear. Norm gains and biases stay in float.
Quality is measured, not assumed: see experiments/weights_eval.py.
"""
import contextlib
import inspect
import json
import os
import struct
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = {8: 127, 4: 7, "nf4": 7}
# NF4 codebook (bitsandbytes create_normal_map): 16 quantiles of N(0,1), 0 exactly representable
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])
FP16_MIN, FP16_MAX = 2.0 ** -14, 65504.0       # fp16 normal range for scales
CHUNK = 1 << 22                                 # weights per (de)quantization chunk


# ----------------------------------------------------------------------------- codes
def _row_chunks(R, width):
    step = max(1, CHUNK // max(width, 1))
    return [(a, min(a + step, R)) for a in range(0, R, step)]


def _blocks(w, pad, block):
    x = w.float().contiguous()
    if pad:
        x = F.pad(x, (0, pad))
    return x.reshape(x.shape[0], -1, block)


def quantize_rows(w, bits, block, scale_dtype=None):
    """(R, C) float -> (codes, scales (R, (C+pad)/block), pad). Codes: int8 (R, nb, block)
    for bits=8, two 4-bit codes per uint8 (R, (C+pad)/2) otherwise. Rows go in chunks, so
    the float temporaries stay small whatever the size of w (bucketize for NF4: no
    (R, C, 16) distance tensor). Scales are fp16, or fp32 when a nonzero scale is outside
    the fp16 normal range; codes are computed against the stored (rounded) scale."""
    w = w.detach()
    R, C = w.shape
    if bits not in QMAX:
        raise ValueError(f"bits must be 8, 4 or 'nf4', got {bits!r}")
    four = bits != 8
    if four and block % 2:
        raise ValueError(f"4-bit codes are packed in pairs: block must be even, got {block}")
    pad = (-C) % block
    nb = (C + pad) // block
    chunks = _row_chunks(R, C + pad)
    s = torch.empty(R, nb, dtype=torch.float32, device=w.device)
    for a, b in chunks:
        s[a:b] = _blocks(w[a:b], pad, block).abs().amax(dim=2)
    if bits != "nf4":
        s /= QMAX[bits]
    if scale_dtype is None:
        nz = s[s > 0]
        ok = nz.numel() == 0 or bool((nz.min() >= FP16_MIN) & (nz.max() <= FP16_MAX))
        scale_dtype = torch.float16 if ok else torch.float32
    s = s.to(scale_dtype)
    inv = s.float().reciprocal().nan_to_num_(0.0, 0.0, 0.0)   # all-zero blocks: scale 0
    if four:
        q = torch.empty(R, (C + pad) // 2, dtype=torch.uint8, device=w.device)
        mids = ((NF4[1:] + NF4[:-1]) / 2).to(w.device) if bits == "nf4" else None
    else:
        q = torch.empty(R, nb, block, dtype=torch.int8, device=w.device)
    for a, b in chunks:
        n = _blocks(w[a:b], pad, block) * inv[a:b, :, None]
        if bits == "nf4":                      # nearest codebook value = bucket between midpoints
            u = torch.bucketize(n, mids, out_int32=True).to(torch.uint8).view(b - a, -1)
        else:
            n = n.round_().clamp_(-QMAX[bits], QMAX[bits])
            if bits == 8:
                q[a:b] = n.to(torch.int8)
                continue
            u = (n + 8).to(torch.uint8).view(b - a, -1)      # 1..15, 0 unused
        q[a:b] = u[:, 0::2] | (u[:, 1::2] << 4)              # two codes per byte
    return q, s, pad


def codebook(bits, device=None):
    """The 16 values of a 4-bit code: NF4, or -8..7 for linear int4 (code 0 unused)."""
    cb = NF4 if bits == "nf4" else torch.arange(-8, 8, dtype=torch.float32)
    return cb.to(device) if device is not None else cb.clone()


def _pair_table(cb):
    """(16,) codebook -> (256, 2) table: byte -> its (low, high) values, on cb's device."""
    b = torch.arange(256, device=cb.device)
    return torch.stack([cb[b & 15], cb[b >> 4]], dim=-1).float()


_TABLES = {}


def _lut(bits, device):
    """Pair table for callers without a codebook buffer: built once per device."""
    t = _TABLES.get((bits, device))
    if t is None:
        t = _TABLES[(bits, device)] = _pair_table(codebook(bits, device))
    return t


def dequantize_rows(q, s, pad, bits, block, dtype=torch.float32, cb=None):
    """Codes + scales -> (R, C) weights in `dtype`. The product is formed in fp32 and
    rounded once; 4-bit codes go through a 256-entry byte -> (low, high) table built from
    the codebook `cb` (a module buffer, already on q's device) with int32 indices."""
    R = q.shape[0]
    if bits == 8:
        w = q.reshape(R, -1, block).float()
    else:
        lut = _lut(bits, q.device) if cb is None else _pair_table(cb)
        w = F.embedding(q.reshape(R, -1).int(), lut).view(R, -1, block)
    w = w.mul_(s.float()[..., None]).view(R, -1)
    if pad:
        w = w[:, :w.shape[1] - pad]
    return w.to(dtype)


def _pick_block(C, block, bits):
    """`block`, or the largest divisor of C in [block/2, block] (no padding); even for 4 bits."""
    block = max(1, min(block, C))
    if C % block:
        for b in range(block, (block + 1) // 2 - 1, -1):
            if C % b == 0 and (bits == 8 or b % 2 == 0):
                block = b
                break
    if bits != 8 and block % 2:
        block += 1
    return block


def autocast_dtype(device_type):
    """The autocast dtype on `device_type`, or None when autocast is off there."""
    try:
        on = torch.is_autocast_enabled(device_type)
    except RuntimeError:                       # device types without autocast (meta)
        return None
    except TypeError:                          # torch < 2.4: per-device functions
        if device_type == "cpu":
            return torch.get_autocast_cpu_dtype() if torch.is_autocast_cpu_enabled() else None
        if device_type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        return None
    return torch.get_autocast_dtype(device_type) if on else None


def _compute_dtype(x):
    return autocast_dtype(x.device.type) or x.dtype


def _dq_chunks(q, s, cb, pad, bits, block, dtype, n_in):
    for a, b in _row_chunks(q.shape[0], n_in):
        yield a, b, dequantize_rows(q[a:b], s[a:b], pad, bits, block, dtype, cb)


class _QMatmul(torch.autograd.Function):
    """y = x @ W^T for a frozen quantized W. Saves only codes/scales (module buffers) and
    rebuilds W in backward; returns no weight gradient (the bitsandbytes MatMul4Bit idea)."""

    @staticmethod
    def forward(ctx, x, q, s, cb, meta):
        pad, bits, block, n_in = meta
        dtype = _compute_dtype(x)
        ctx.meta, ctx.dtype, ctx.x_dtype = meta, dtype, x.dtype
        ctx.save_for_backward(q, s, cb)
        xc = x.to(dtype)
        if q.shape[0] * n_in <= CHUNK:
            return F.linear(xc, dequantize_rows(q, s, pad, bits, block, dtype, cb))
        y = xc.new_empty(*x.shape[:-1], q.shape[0])
        for a, b, w in _dq_chunks(q, s, cb, pad, bits, block, dtype, n_in):
            y[..., a:b] = F.linear(xc, w)
        return y

    @staticmethod
    def backward(ctx, g):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        q, s, cb = ctx.saved_tensors
        pad, bits, block, n_in = ctx.meta
        g = g.to(ctx.dtype)
        gx = None
        for a, b, w in _dq_chunks(q, s, cb, pad, bits, block, ctx.dtype, n_in):
            part = g[..., a:b] @ w
            gx = part if gx is None else gx.add_(part)
        return gx.to(ctx.x_dtype), None, None, None, None


# ----------------------------------------------------------------------------- modules
def is_conv1d(m):
    """transformers' Conv1D (GPT-2 style): weight (in, out), y = x @ W + b."""
    return any(c.__name__ == "Conv1D" and c.__module__.startswith("transformers")
               for c in type(m).__mro__)


class QBase:
    """Quantized weight storage (buffers q, s; codebook cb for 4-bit) shared by QLinear,
    QEmbedding and quantized nn.Linear/nn.Embedding subclasses. `dtype` is the compute
    dtype of the original weight; it follows model.to(dtype), the scales do not."""

    def _set_weight(self, w, bits, block):
        self.bits, self.block, self.dtype = bits, block, w.dtype
        if w.is_meta:                          # placeholder; load_float_weight fills it
            C = w.shape[1]
            pad = (-C) % block
            shape = (w.shape[0], (C + pad) // 2) if bits != 8 else (w.shape[0], (C + pad) // block, block)
            q = torch.empty(shape, dtype=torch.uint8 if bits != 8 else torch.int8, device="meta")
            s = torch.empty(w.shape[0], (C + pad) // block, dtype=torch.float16, device="meta")
        else:
            q, s, pad = quantize_rows(w, bits, block)
        self.pad = pad
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.register_buffer("cb", None if bits == 8 else codebook(bits, q.device), persistent=False)

    @torch.no_grad()
    def load_float_weight(self, w):
        """(Re)quantize from a float (out, in) weight, e.g. one streamed from a checkpoint."""
        q, s, pad = quantize_rows(w, self.bits, self.block)
        if pad != self.pad or q.shape != self.q.shape:
            raise ValueError(f"weight of shape {tuple(w.shape)} does not fit this module")
        self.q, self.s = q, s
        if self.bits != 8:
            self.cb = codebook(self.bits, q.device)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        s = state_dict.get(prefix + "s")       # keep the checkpoint's scale dtype (fp16 or fp32)
        if isinstance(s, torch.Tensor) and self._buffers.get("s") is not None and s.dtype != self.s.dtype:
            self.s = torch.empty_like(self.s, dtype=s.dtype)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def dequantize(self, dtype=None):
        """The full float weight (out, in) in `dtype` (default: the compute dtype)."""
        return dequantize_rows(self.q, self.s, self.pad, self.bits, self.block,
                               dtype or self.dtype, self.cb)

    @property
    def weight(self):
        return self.dequantize()

    @property
    def device(self):
        return self.q.device

    def _n_in(self):
        return self.q.shape[1] * (2 if self.bits != 8 else self.q.shape[2]) - self.pad

    def matmul(self, x):
        """x @ W^T (no bias) without keeping W alive for backward."""
        return _QMatmul.apply(x, self.q, self.s, self.cb,
                              (self.pad, self.bits, self.block, self._n_in()))

    def gather(self, idx):
        """Rows `idx` of W, dequantizing only those rows."""
        flat = idx.reshape(-1)
        w = dequantize_rows(self.q.index_select(0, flat), self.s.index_select(0, flat),
                            self.pad, self.bits, self.block, self.dtype, self.cb)
        return w.reshape(*idx.shape, w.shape[-1])

    def _apply(self, fn, recurse=True):
        try:                                   # follow .to(dtype) / .half() for the compute dtype
            self.dtype = fn(torch.empty(0, dtype=self.dtype)).dtype
        except Exception:
            pass
        keep = {k: self._buffers[k] for k in ("s", "cb") if self._buffers.get(k) is not None}
        super()._apply(fn, recurse)
        for k, t in keep.items():              # scales / codebook keep their dtype
            if self._buffers[k].dtype != t.dtype:
                self._buffers[k] = t.to(self._buffers[k].device)
        return self

    def nbytes(self):
        n = sum(t.numel() * t.element_size() for t in (self.q, self.s))
        b = getattr(self, "bias", None)
        return n + (0 if b is None else b.numel() * b.element_size())

    @property
    def code_bits(self):
        return 4 if self.bits in (4, "nf4") else 8


class QLinear(QBase, nn.Module):
    """Quantized stand-in for nn.Linear or transformers' Conv1D."""

    def __init__(self, linear, bits=8, block=128):
        nn.Module.__init__(self)
        self._from_conv1d = is_conv1d(linear)
        w = linear.weight.t() if self._from_conv1d else linear.weight
        self.out_features, self.in_features = w.shape
        self._set_weight(w.detach(), bits, _pick_block(self.in_features, block, bits))
        b = getattr(linear, "bias", None)
        self.bias = None if b is None else nn.Parameter(b.detach().clone(), requires_grad=False)

    def forward(self, x):
        y = self.matmul(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)


class QEmbedding(QBase, nn.Module):
    """Quantized embedding table: forward dequantizes only the gathered rows."""

    def __init__(self, emb, bits=8, block=128):
        nn.Module.__init__(self)
        self.num_embeddings, self.embedding_dim = emb.weight.shape
        self.padding_idx = getattr(emb, "padding_idx", None)
        self._set_weight(emb.weight.detach(), bits, _pick_block(self.embedding_dim, block, bits))

    def forward(self, idx):
        return self.gather(idx)


class QTiedHead(nn.Module):
    """Output head tied to a QEmbedding: uses the embedding's codes, stores none itself.
    The embedding is referenced, not registered, so state_dict and .to() see it once."""

    def __init__(self, emb, bias=None):
        super().__init__()
        object.__setattr__(self, "emb", emb)
        self.out_features, self.in_features = emb.num_embeddings, emb.embedding_dim
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    q = property(lambda self: self.emb.q)
    s = property(lambda self: self.emb.s)
    dtype = property(lambda self: self.emb.dtype)
    weight = property(lambda self: self.emb.weight)
    device = property(lambda self: self.emb.device)

    def matmul(self, x):
        return self.emb.matmul(x)

    def nbytes(self):
        return 0 if self.bias is None else self.bias.numel() * self.bias.element_size()

    def forward(self, x):
        y = self.emb.matmul(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)


class _QLinearForward(QBase, nn.Linear):
    def forward(self, x):
        return QLinear.forward(self, x)


class _QEmbeddingForward(QBase, nn.Embedding):
    def forward(self, idx):
        return self.gather(idx)


def _scaled_embedding_forward(self, input_ids):
    out = self.gather(input_ids)
    return out * self.embed_scale.to(out.dtype)


def _is_hf_scaled_embedding(cls):
    """forward is exactly `return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)`
    (Gemma*, VaultGemma, MiniCPM3): replace it to avoid a full dequantization for `.dtype`."""
    try:
        src = inspect.getsource(cls.forward)
    except (OSError, TypeError):
        return False
    body = [ln.strip() for ln in src.splitlines()[1:] if ln.strip()]
    body = [ln for ln in body if not (ln.startswith(('"""', "'''", "#")))]
    return body == ["return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)"]


_SWAPPED = {}


def _quantize_subclass(child, kind, bits, block):
    """Keep an nn.Linear/nn.Embedding subclass (and its forward); swap its weight Parameter for
    quantized storage. Its super().forward() then runs quantized; `.weight` dequantizes."""
    cls = type(child)
    key = (cls, kind)
    if key not in _SWAPPED:
        mixin = _QLinearForward if kind == "linear" else _QEmbeddingForward
        ns = {"__module__": __name__}
        if kind == "embedding" and _is_hf_scaled_embedding(cls):
            ns["forward"] = _scaled_embedding_forward
        _SWAPPED[key] = type("Q" + cls.__name__, (cls, mixin), ns)
    w = child._parameters.pop("weight").detach()
    child.__class__ = _SWAPPED[key]
    width = w.shape[1]
    child._set_weight(w, bits, _pick_block(width, block, bits))
    return child


def _kind(m):
    if isinstance(m, (QBase, QTiedHead)) or not isinstance(m, nn.Module):
        return None
    if isinstance(m, nn.Embedding):
        return "embedding"
    if isinstance(m, nn.Linear):
        return "linear"
    if is_conv1d(m):
        return "conv1d"
    return None


def _quantize_module(child, kind, bits, block):
    """A quantized module for `child`, or None (with a warning) when that is unsafe."""
    if kind == "conv1d":
        if type(child).forward is not next(c for c in type(child).__mro__ if c.__name__ == "Conv1D").forward:
            warnings.warn(f"{type(child).__name__} overrides Conv1D.forward; left in float")
            return None
        return QLinear(child, bits, block)
    if kind == "embedding":
        if child.max_norm is not None:
            warnings.warn(f"{type(child).__name__} has max_norm (renormalizes its weight); left in float")
            return None
        if type(child).forward is nn.Embedding.forward:
            return QEmbedding(child, bits, block)
        return _quantize_subclass(child, kind, bits, block)
    if type(child).forward is nn.Linear.forward:
        return QLinear(child, bits, block)
    return _quantize_subclass(child, kind, bits, block)


def model_bytes(model):
    """Bytes held by parameters, buffers and quantized codes (no double counting)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def _head_module(model):
    fn = getattr(model, "get_output_embeddings", None)
    try:
        return fn() if callable(fn) else None
    except (NotImplementedError, AttributeError):
        return None


@torch.no_grad()
def quantize_model(model, bits=8, block=None, min_numel=4096, embed_bits="same",
                   skip_names=()):
    """Replace every nn.Linear / Conv1D / nn.Embedding with >= min_numel weights, in place.
    bits: 8, 4 (linear int4) or "nf4" for the linear layers.
    embed_bits: "same", 8, 4, "nf4" or None (keep embeddings and a tied/untied
    output head in float, as QLoRA does). skip_names: module-name suffixes to leave alone.
    A head tied to the input embedding shares its codes (QTiedHead); subclasses keep their
    forward. Works on a model whose parameters are on the meta device (placeholders)."""
    assert bits in (8, 4, "nf4")
    block = block or (128 if bits == 8 else 64)
    eb = bits if embed_bits == "same" else embed_bits
    eblock = 128 if eb == 8 else 64
    head = _head_module(model)
    before = model_bytes(model)
    work = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (QBase, QTiedHead)):
            continue
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            kind = _kind(child)
            w = child._parameters.get("weight") if kind else None
            if w is None or w.numel() < min_numel or any(full.endswith(sn) for sn in skip_names):
                continue
            work.append((kind != "embedding", mod, cname, full, child, kind, w))
    work.sort(key=lambda t: t[0])                       # embeddings first (tied heads follow)
    emb_ids = {id(t[-1]) for t in work if t[5] == "embedding"}
    qemb, n_q, skipped, tied = {}, 0, [], False
    for _, mod, cname, full, child, kind, w in work:
        if kind != "embedding" and id(w) in emb_ids:    # a head sharing an embedding's weight
            if id(w) in qemb:
                setattr(mod, cname, QTiedHead(qemb[id(w)], getattr(child, "bias", None)))
                tied = True
            continue                                    # (float embedding: the tie is kept)
        is_emb = kind == "embedding" or child is head
        if is_emb and eb is None:
            continue
        new = _quantize_module(child, kind, eb if is_emb else bits, eblock if is_emb else block)
        if new is None:
            skipped.append(full)
            continue
        if kind == "embedding":
            qemb[id(w)] = new
        setattr(mod, cname, new)
        n_q += 1
    after = model_bytes(model)
    return {"bits": bits, "block": block, "modules_quantized": n_q,
            "bytes_before": before, "bytes_after": after, "ratio": before / max(after, 1),
            "tied_head": tied, "skipped": skipped}


# ----------------------------------------------------------------------------- loading
_ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
              "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
              "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
              "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def iter_safetensors(path):
    """Yield (name, tensor) from a .safetensors file one tensor at a time, with plain reads
    (no mmap, so the file's pages never count toward this process's RSS)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        items = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                       key=lambda kv: kv[1]["data_offsets"][0])
        for k, v in items:
            a, b = v["data_offsets"]
            dt = _ST_DTYPES[v["dtype"]]
            f.seek(base + a)
            buf = bytearray(b - a)
            f.readinto(buf)
            t = torch.frombuffer(buf, dtype=dt) if b > a else torch.empty(0, dtype=dt)
            yield k, t.reshape(v["shape"])


@contextlib.contextmanager
def _params_on_meta():
    """Build modules with every parameter on the meta device (buffers stay real)."""
    old = nn.Module.register_parameter

    def register(self, name, param):
        old(self, name, param)
        p = self._parameters.get(name)
        if p is not None and not p.is_meta:
            self._parameters[name] = type(p)(p.to("meta"), requires_grad=p.requires_grad)

    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = old


def _resolve(model_id_or_path, kw):
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download
    patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"]
    if kw.get("trust_remote_code"):
        patterns.append("*.py")                # the repository's own modeling code
    return snapshot_download(model_id_or_path, revision=kw.get("revision"), token=kw.get("token"),
                             cache_dir=kw.get("cache_dir"), allow_patterns=patterns)


@torch.no_grad()
def load_quantized(model_id_or_path, bits, embed_bits=8, dtype=torch.float32,
                   **from_pretrained_kwargs):
    """A Hugging Face causal LM with its weights quantized as quantize_model(model, bits,
    min_numel=1<<16, embed_bits=embed_bits) would, built without ever holding the float model:
    parameters are created on the meta device, the safetensors shards are read tensor by
    tensor, and each weight is quantized as it arrives. Peak RAM is about the quantized size
    plus one float tensor. Extra keywords: block, min_numel, skip_names (as quantize_model);
    revision, token, cache_dir, trust_remote_code, attn_implementation; anything else
    overrides config attributes. The report is in model.quant_report."""
    from transformers import AutoConfig, AutoModelForCausalLM
    kw = dict(from_pretrained_kwargs)
    qkw = {k: kw.pop(k) for k in ("block", "min_numel", "skip_names") if k in kw}
    qkw.setdefault("min_numel", 1 << 16)
    path = _resolve(model_id_or_path, kw)
    for k in ("token", "cache_dir"):
        kw.pop(k, None)
    mkw = {k: kw.pop(k) for k in ("attn_implementation",) if k in kw}
    if "trust_remote_code" in kw:
        mkw["trust_remote_code"] = kw["trust_remote_code"]
    config = AutoConfig.from_pretrained(path, **kw)
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        files = sorted(set(json.load(open(index))["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights in {path}: use from_pretrained + "
                                "quantize_model for other formats")
    with _params_on_meta():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, **mkw)
    report = quantize_model(model, bits, embed_bits=embed_bits, **qkw)

    # every (module, attribute) that should receive each checkpoint key; tied params listed once
    targets, users = {}, {}
    for mname, mod in model.named_modules(remove_duplicate=False):
        pre = mname + "." if mname else ""
        if isinstance(mod, QBase):
            targets.setdefault(pre + "weight", ("q", mod, id(mod)))
        elif isinstance(mod, QTiedHead):                # its codes come with the embedding
            targets.setdefault(pre + "weight", ("tied", None, None))
        for pname, p in list(mod._parameters.items()) + [
                (b, t) for b, t in mod._buffers.items() if b not in mod._non_persistent_buffers_set]:
            if p is None or (isinstance(mod, QBase) and pname in ("q", "s", "cb")):
                continue
            targets.setdefault(pre + pname, ("p", p, id(p)))
            users.setdefault(id(p), []).append((mod, pname))
    prefix = getattr(model, "base_model_prefix", "")
    done, unexpected = set(), []
    for fname in files:
        for key, t in iter_safetensors(os.path.join(path, fname)):
            tgt = targets.get(key)
            if tgt is None and prefix:
                tgt = targets.get(key[len(prefix) + 1:] if key.startswith(prefix + ".")
                                  else f"{prefix}.{key}")
            if tgt is None:
                unexpected.append(key)
                continue
            kind, obj, ident = tgt
            if kind == "q":
                if t.dtype.itemsize >= obj.dtype.itemsize:     # round to dtype first, as
                    t = t.to(obj.dtype)                         # from_pretrained(dtype) does
                obj.load_float_weight(t.t() if getattr(obj, "_from_conv1d", False) else t)
            elif kind == "p":
                new = t.to(dtype) if t.is_floating_point() else t
                if isinstance(obj, nn.Parameter):       # one Parameter for all its (tied) users
                    new = nn.Parameter(new, requires_grad=obj.requires_grad and new.is_floating_point())
                for mod, pname in users[ident]:
                    (mod._parameters if pname in mod._parameters else mod._buffers)[pname] = new
            done.add(ident)
            del t
    missing = sorted(k for k, (kind, obj, ident) in targets.items()
                     if kind != "tied" and ident not in done
                     and (obj.q if kind == "q" else obj).is_meta)
    if missing:
        raise RuntimeError(f"weights missing from the checkpoint: {missing[:8]}")
    if unexpected:
        warnings.warn(f"{len(unexpected)} checkpoint tensors not used, e.g. {unexpected[:4]}")
    report["bytes_after"] = model_bytes(model)
    model.quant_report = report
    try:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(path)
    except Exception:
        pass
    return model.eval()
