"""Minimal loader for Karpathy's llama2.c checkpoints (stories260K/15M/42M/110M).

Re-implements llama2.c's model.py forward pass for evaluation only:
RMSNorm, rotary embeddings on *interleaved* (real, imag) pairs, SwiGLU MLP,
grouped-query attention, output head tied to the token embedding. Loads
the original .pt files (dict with 'model' state_dict and 'model_args')
with torch.load(weights_only=True), so no community conversion is involved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim, n, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    ang = torch.outer(torch.arange(n).float(), freqs)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, T, H, D), pairs (2i, 2i+1) are (real, imag)
    xr, xi = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1).flatten(3).type_as(x)


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv, hidden, eps):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = n_heads, n_kv, dim // n_heads
        self.attention = nn.Module()
        self.attention.wq = nn.Linear(dim, n_heads * self.hd, bias=False)
        self.attention.wk = nn.Linear(dim, n_kv * self.hd, bias=False)
        self.attention.wv = nn.Linear(dim, n_kv * self.hd, bias=False)
        self.attention.wo = nn.Linear(n_heads * self.hd, dim, bias=False)
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(dim, hidden, bias=False)
        self.feed_forward.w2 = nn.Linear(hidden, dim, bias=False)
        self.feed_forward.w3 = nn.Linear(dim, hidden, bias=False)
        self.attention_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        a = self.attention
        h = self.attention_norm(x)
        q = a.wq(h).view(B, T, self.n_heads, self.hd)
        k = a.wk(h).view(B, T, self.n_kv, self.hd)
        v = a.wv(h).view(B, T, self.n_kv, self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True)
        x = x + a.wo(y.transpose(1, 2).reshape(B, T, -1))
        f = self.feed_forward
        h = self.ffn_norm(x)
        return x + f.w2(F.silu(f.w1(h)) * f.w3(h))


class Llama2C(nn.Module):
    def __init__(self, args, hidden):
        super().__init__()
        dim, L = args["dim"], args["n_layers"]
        n_heads = args["n_heads"]
        n_kv = args.get("n_kv_heads") or n_heads
        eps = args.get("norm_eps", 1e-5)
        self.max_seq_len = args["max_seq_len"]
        self.tok_embeddings = nn.Embedding(args["vocab_size"], dim)
        self.layers = nn.ModuleList(Block(dim, n_heads, n_kv, hidden, eps) for _ in range(L))
        self.norm = RMSNorm(dim, eps)
        self.output = nn.Linear(dim, args["vocab_size"], bias=False)
        self.output.weight = self.tok_embeddings.weight          # tied, as in llama2.c
        cos, sin = rope_tables(dim // n_heads, self.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_embeddings(idx)
        for blk in self.layers:
            x = blk(x, self.cos[:T], self.sin[:T])
        return self.output(self.norm(x))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("freqs_")}
    hidden = sd["layers.0.feed_forward.w1.weight"].shape[0]
    m = Llama2C(ck["model_args"], hidden)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "output.weight"]   # tied to tok_embeddings
    assert not missing and not unexpected, (missing, unexpected)
    m.eval()
    return m, ck
