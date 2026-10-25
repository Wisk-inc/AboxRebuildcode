"""Weight-only quantized layers: keep weights in int8 / int4 in RAM, compute in float.

    from zoomtrain.qweights import quantize_model
    report = quantize_model(model, bits=4, block=64)   # in place
    print(report)                                      # bytes before / after

Each weight matrix is split into blocks of `block` consecutive input columns;
every block stores one fp16 scale (absmax) and codes:

    bits=8:  int8 in [-127, 127]                           ~1.03 B/param (block 128)
    bits=4:  linear int4 in [-7, 7], two codes per byte    ~0.53 B/param (block 64)
    "nf4":   4-bit NormalFloat (QLoRA, Dettmers 2023): 16 codebook values at the
             quantiles of N(0,1), two codes per byte      ~0.53 B/param (block 64)
NF4 is the right choice for pretrained weights (roughly Gaussian per block);
linear int4 wastes codes on the rarely used extremes.

The forward pass dequantizes the block-scaled codes to the activation dtype
and runs a normal matmul, so the model's output is that of a model whose
weights were rounded once; nothing else changes. Embeddings (including a
tied output head) are quantized the same way, row by row. Norm gains and
biases stay in float (they are a rounding error of the parameter count).
Quality is measured, not assumed: see experiments/weights_eval.py.
"""
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


def quantize_rows(w, bits, block):
    """(R, C) float -> (codes uint8/int8, fp16 scales (R, C/block), pad)."""
    w = w.detach().float()
    R, C = w.shape
    pad = (-C) % block
    if pad:
        w = F.pad(w, (0, pad))
    b = w.view(R, -1, block)
    if bits == "nf4":
        s = b.abs().amax(dim=2).clamp_min(1e-12)           # scale to [-1, 1]
        n = b / s[..., None]
        u = (n[..., None] - NF4.to(n.device)).abs().argmin(dim=-1).to(torch.uint8).view(R, -1)
        q = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
        return q, s.to(torch.float16), pad
    s = (b.abs().amax(dim=2) / QMAX[bits]).clamp_min(1e-12)
    q = torch.round(b / s[..., None]).clamp_(-QMAX[bits], QMAX[bits]).to(torch.int8)
    if bits == 4:
        u = (q + 8).to(torch.uint8).view(R, -1)            # 1..15, 0 unused
        q = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()  # two codes per byte
    return q, s.to(torch.float16), pad


def dequantize_rows(q, s, pad, bits, block, dtype=torch.float32):
    R = q.shape[0]
    if bits == "nf4":
        lo = (q & 0x0F).long()
        hi = (q >> 4).long()
        idx = torch.stack([lo, hi], dim=-1).view(R, -1)
        w = NF4.to(q.device)[idx].to(dtype).view(R, -1, block) * s.to(dtype)[..., None]
        w = w.view(R, -1)
        return w[:, :w.shape[1] - pad] if pad else w
    if bits == 4:
        lo = (q & 0x0F).to(torch.int8) - 8
        hi = (q >> 4).to(torch.int8) - 8
        q = torch.stack([lo, hi], dim=-1).view(R, -1)
    w = q.view(R, -1, block).to(dtype) * s.to(dtype)[..., None]
    w = w.view(R, -1)
    return w[:, :w.shape[1] - pad] if pad else w


class QLinear(nn.Module):
    def __init__(self, linear, bits=8, block=128):
        super().__init__()
        block = min(block, linear.in_features)     # no padding waste on narrow rows
        self.bits, self.block = bits, block
        self.in_features, self.out_features = linear.in_features, linear.out_features
        q, s, self.pad = quantize_rows(linear.weight, bits, block)
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(),
                                                                   requires_grad=False)

    @property
    def weight(self):
        return dequantize_rows(self.q, self.s, self.pad, self.bits, self.block)

    def forward(self, x):
        w = dequantize_rows(self.q, self.s, self.pad, self.bits, self.block, x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    def nbytes(self):
        n = self.q.numel() * self.q.element_size() + self.s.numel() * 2
        return n + (0 if self.bias is None else self.bias.numel() * 4)

    @property
    def code_bits(self):
        return 4 if self.bits in (4, "nf4") else 8


class QEmbedding(nn.Module):
    """Quantized embedding table; `.weight` dequantizes (for a tied output head)."""

    def __init__(self, emb, bits=8, block=128):
        super().__init__()
        block = min(block, emb.weight.shape[1])
        self.bits, self.block = bits, block
        self.num_embeddings, self.embedding_dim = emb.weight.shape
        q, s, self.pad = quantize_rows(emb.weight, bits, block)
        self.register_buffer("q", q)
        self.register_buffer("s", s)

    @property
    def weight(self):
        return dequantize_rows(self.q, self.s, self.pad, self.bits, self.block)

    def forward(self, idx):
        return F.embedding(idx, self.weight)

    def nbytes(self):
        return self.q.numel() * self.q.element_size() + self.s.numel() * 2


def model_bytes(model):
    """Bytes held by parameters, buffers and quantized codes (no double counting)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def quantize_model(model, bits=8, block=None, min_numel=4096, embed_bits="same",
                   skip_names=()):
    """Replace every nn.Linear / nn.Embedding with >= min_numel weights, in place.
    bits: 8, 4 (linear int4) or "nf4" for the linear layers.
    embed_bits: "same", 8, 4, "nf4" or None (keep embeddings and a tied/untied
    output head in float, as QLoRA does). skip_names: module-name suffixes to leave alone."""
    assert bits in (8, 4, "nf4")
    block = block or (128 if bits == 8 else 64)
    eb = bits if embed_bits == "same" else embed_bits
    eblock = 128 if eb == 8 else 64
    head_ids = set()
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out) and get_out() is not None:
        head_ids.add(id(get_out()))
    before = model_bytes(model)
    n_q = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if any(full.endswith(sn) for sn in skip_names):
                continue
            is_head = id(child) in head_ids
            if isinstance(child, nn.Linear) and child.weight.numel() >= min_numel and not is_head:
                setattr(mod, cname, QLinear(child, bits, block))
                n_q += 1
            elif isinstance(child, (nn.Embedding, nn.Linear)) and child.weight.numel() >= min_numel \
                    and (isinstance(child, nn.Embedding) or is_head) and eb is not None:
                if isinstance(child, nn.Embedding):
                    setattr(mod, cname, QEmbedding(child, eb, eblock))
                else:
                    setattr(mod, cname, QLinear(child, eb, eblock))
                n_q += 1
    after = model_bytes(model)
    return {"bits": bits, "block": block, "modules_quantized": n_q,
            "bytes_before": before, "bytes_after": after, "ratio": before / after}
