"""ViT weather model (6 h step) with Zoom Training support.

The (lat, lon) grid is cut into 2x2-cell patches -> fine token grid. At zoom
level k = f*f, each f x f block of fine tokens is slot-merged into one token,
the transformer runs on 1/k as many tokens, and the result is slot-unmerged
back to the fine token grid. Inputs and targets stay at full resolution in
every stage; only the internal processing resolution changes.

coarse_mode="lowres" instead implements classic progressive resizing: the
fields are area-averaged to a coarser grid and the model is trained on that
coarser problem (the fine-scale data is simply not seen in that stage).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..zoom import SlotMerge, SlotUnmerge


class Block(nn.Module):
    def __init__(self, d, n_head, mlp_mult=4):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.qkv.weight.muon_split = 3
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, mlp_mult * d)
        self.fc_out = nn.Linear(mlp_mult * d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.n_head, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(y.transpose(1, 2).reshape(B, N, D))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class ZoomViT(nn.Module):
    def __init__(self, c_in, c_out, grid=(32, 64), patch=2, d=128, n_layer=6, n_head=4,
                 zoom_ks=(), gated=True, merge_mode=None):
        super().__init__()
        self.c_in, self.c_out, self.patch, self.d, self.n_layer = c_in, c_out, patch, d, n_layer
        self.gh, self.gw = grid[0] // patch, grid[1] // patch    # fine token grid
        self.embed = nn.Linear(c_in * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, self.gh, self.gw, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.decode = nn.Linear(d, c_out * patch * patch)
        nn.init.zeros_(self.decode.weight)
        nn.init.zeros_(self.decode.bias)
        self.merge = nn.ModuleDict({str(k): SlotMerge(k, d, gated, merge_mode) for k in zoom_ks})
        self.unmerge = nn.ModuleDict({str(k): SlotUnmerge(k, d, gated, merge_mode) for k in zoom_ks})

    def muon_params(self):
        return [p for b in self.blocks for p in
                (b.qkv.weight, b.proj.weight, b.fc.weight, b.fc_out.weight)]

    # -- (B, C, H, W) <-> (B, H/p, W/p, C*p*p)
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch
        x = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, H // p, W // p, C * p * p)

    def unpatchify(self, t, C):
        B, h, w, _ = t.shape
        p = self.patch
        t = t.view(B, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5)
        return t.reshape(B, C, h * p, w * p)

    @staticmethod
    def _group(t, f):  # (B, h, w, d) -> (B, h/f * w/f, f*f, d)
        B, h, w, d = t.shape
        t = t.view(B, h // f, f, w // f, f, d).permute(0, 1, 3, 2, 4, 5)
        return t.reshape(B, (h // f) * (w // f), f * f, d)

    @staticmethod
    def _ungroup(t, h, w, f):  # (B, h/f * w/f, f*f, d) -> (B, h, w, d)
        B, _, _, d = t.shape
        t = t.view(B, h // f, w // f, f, f, d).permute(0, 1, 3, 2, 4, 5)
        return t.reshape(B, h, w, d)

    def forward(self, x, k=1):
        """x: (B, c_in, H, W) at full resolution -> (B, c_out, H, W)."""
        tok = self.embed(self.patchify(x)) + self.pos               # (B, gh, gw, d)
        B, h, w, d = tok.shape
        if k > 1:
            f = int(round(k ** 0.5))
            z = self.merge[str(k)](self._group(tok, f))              # (B, N/k, d)
        else:
            z = tok.view(B, h * w, d)
        for blk in self.blocks:
            z = blk(z)
        z = self.ln_f(z)
        if k > 1:
            z = self._ungroup(self.unmerge[str(k)](z), h, w, f)     # (B, gh, gw, d)
        else:
            z = z.view(B, h, w, d)
        return self.unpatchify(self.decode(z), self.c_out)

    def forward_lowres(self, x_coarse, f):
        """Progressive-resizing stage: whole problem on an f-times coarser grid."""
        pos = F.avg_pool2d(self.pos.permute(0, 3, 1, 2), f).permute(0, 2, 3, 1)
        tok = self.embed(self.patchify(x_coarse)) + pos
        B, h, w, d = tok.shape
        z = tok.view(B, h * w, d)
        for blk in self.blocks:
            z = blk(z)
        z = self.ln_f(z).view(B, h, w, d)
        return self.unpatchify(self.decode(z), self.c_out)

    def flops_per_step(self, batch, k):
        n = self.gh * self.gw // k                       # tokens through the body
        d, L = self.d, self.n_layer
        body = n * L * (24 * d * d + 4 * n * d)          # full (non-causal) attention
        io = self.gh * self.gw * 2 * d * self.patch ** 2 * (self.c_in + self.c_out)
        return 3 * batch * (body + io)
