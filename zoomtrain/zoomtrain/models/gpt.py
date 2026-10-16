"""Small decoder-only transformer with Zoom Training support.

forward(idx, k):
  idx: (B, T) token ids, T divisible by k.
  k=1 is an ordinary causal LM. For k>1 every k consecutive tokens are merged
  into one position; position p predicts all k tokens of group p+1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..zoom import SlotMerge, SlotUnmerge


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(positions, head_dim, base=10000.0):
    """cos/sin tables for arbitrary (possibly fractional) positions."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                       device=positions.device) / head_dim))
    ang = positions.float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, H, T, Dh)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Block(nn.Module):
    def __init__(self, d, n_head, mlp_mult=4, qk_norm=False):
        super().__init__()
        self.n_head = n_head
        self.ln1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.qkv.weight.muon_split = 3          # orthogonalize q, k, v separately
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = RMSNorm(d)
        self.fc = nn.Linear(d, mlp_mult * d, bias=False)
        self.fc_out = nn.Linear(mlp_mult * d, d, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(d // n_head)
            self.k_norm = RMSNorm(d // n_head)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.fc_out(F.gelu(self.fc(self.ln2(x))))
        return x


class ZoomGPT(nn.Module):
    def __init__(self, vocab, d=256, n_layer=6, n_head=4, zoom_ks=(),
                 gated=True, out_mode="slot", rope_pos="token", qk_norm=False,
                 merge_mode=None, grad_ckpt=False):
        super().__init__()
        assert out_mode in ("slot", "bag")
        assert rope_pos in ("token", "patch")
        self.d, self.n_layer, self.vocab = d, n_layer, vocab
        self.out_mode, self.rope_pos, self.grad_ckpt = out_mode, rope_pos, grad_ckpt
        self.head_dim = d // n_head
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_head, qk_norm=qk_norm) for _ in range(n_layer))
        self.ln_f = RMSNorm(d)
        self.merge = nn.ModuleDict({str(k): SlotMerge(k, d, gated, merge_mode) for k in zoom_ks})
        self.unmerge = nn.ModuleDict(
            {str(k): SlotUnmerge(k, d, gated, merge_mode) for k in zoom_ks} if out_mode == "slot" else {})
        self._rope_cache = {}
        self.apply(self._init)
        for b in self.blocks:  # scaled residual init
            nn.init.normal_(b.proj.weight, std=0.02 / (2 * n_layer) ** 0.5)
            nn.init.normal_(b.fc_out.weight, std=0.02 / (2 * n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def muon_params(self):
        """Hidden weight matrices (Muon); embedding/head/norms/gates stay on AdamW."""
        return [p for b in self.blocks for p in
                (b.qkv.weight, b.proj.weight, b.fc.weight, b.fc_out.weight)]

    def _rope(self, n_pos, k):
        key = (n_pos, k, self.emb.weight.device)
        if key not in self._rope_cache:
            p = torch.arange(n_pos, dtype=torch.float32, device=self.emb.weight.device)
            if self.rope_pos == "token":  # group p sits at its centre, in token units
                p = p * k + (k - 1) / 2
            self._rope_cache[key] = rope_tables(p, self.head_dim)
        return self._rope_cache[key]

    def hidden(self, idx, k=1):
        B, T = idx.shape
        x = self.emb(idx)
        if k > 1:
            x = self.merge[str(k)](x.view(B, T // k, k, self.d))
        cos, sin = self._rope(x.shape[1], k)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.ln_f(x)

    def forward(self, idx, targets, k=1):
        """idx: (B, T); targets: (B, T) where targets[:, i] = token at i + k."""
        B, T = idx.shape
        h = self.hidden(idx, k)                      # (B, T/k, d)
        W = self.emb.weight                          # tied output head
        if k == 1:
            logits = h @ W.t()
            return F.cross_entropy(logits.view(-1, self.vocab), targets.reshape(-1))
        tgt = targets.view(B, T // k, k)
        if self.out_mode == "slot":
            logits = self.unmerge[str(k)](h) @ W.t()   # (B, T/k, k, V)
            return F.cross_entropy(logits.reshape(-1, self.vocab), tgt.reshape(-1))
        logp = F.log_softmax(h @ W.t(), dim=-1)        # (B, T/k, V), one dist per group
        return -logp.gather(-1, tgt).mean()

    def flops_per_step(self, batch, seq, k):
        """Analytic training FLOPs (fwd + bwd = 3x fwd) for one step."""
        n = seq // k
        d, L, V = self.d, self.n_layer, self.vocab
        body = L * (24 * d * d + 2 * n * d)          # per position, causal attention
        head_slots = k if (k > 1 and self.out_mode == "slot") else 1
        head = 2 * d * V * head_slots
        return 3 * batch * n * (body + head)

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=40, eos_id=None):
        for _ in range(n_new):
            h = self.hidden(idx[:, -256:], 1)[:, -1]
            logits = (h @ self.emb.weight.t()) / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx
