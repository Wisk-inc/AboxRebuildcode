"""Small decoder-only transformer with Zoom Training support.

forward(idx, targets=None, k=1):
  idx: (B, T) token ids, T divisible by k.
  k=1 is an ordinary causal LM. For k>1 every k consecutive tokens are merged
  into one position; position p predicts all k tokens of group p+1.
  Returns the mean cross-entropy (fp32) with targets, the logits without.

generate() decodes with a KV cache. save_pretrained() / from_pretrained() write
and read a self-describing checkpoint (config.json + model.safetensors).
"""
import inspect
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..losses import chunked_cross_entropy
from ..zoom import MODES, SlotMerge, SlotUnmerge


def _is_compiling():
    fn = getattr(torch.compiler, "is_compiling", None)
    if fn is None:                                  # torch < 2.3
        from torch._dynamo import is_compiling as fn
    return fn()


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


class KVCache:
    """Key/value buffers (B, H, max_len, Dh) of one attention layer, for decoding."""

    def __init__(self, max_len):
        self.max_len, self.k, self.v = max_len, None, None

    def update(self, k, v, start):
        """Store k, v (B, H, T, Dh) at slots start..start+T-1; return all keys/values so far."""
        if self.k is None:
            shape = (*k.shape[:2], self.max_len, k.shape[3])
            self.k, self.v = k.new_zeros(shape), v.new_zeros(shape)
        end = start + k.shape[2]
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        return self.k[:, :, :end], self.v[:, :, :end]


class Block(nn.Module):
    def __init__(self, d, n_head, mlp_mult=4, qk_norm=False, mlp="gelu"):
        super().__init__()
        self.n_head = n_head
        self.mlp = mlp
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

    def forward(self, x, cos, sin, kv=None, mask=None):
        """kv=(KVCache, start): decoding; x sits at cache slots start.. and attends to
        every slot before it. mask: optional boolean (B, 1, T, S) mask, True = attend."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv is not None:
            cache, start = kv
            if start and T > 1 and mask is None:   # is_causal aligns top-left: wrong here
                raise ValueError("a multi-token cache update after slot 0 needs a mask")
            k, v = cache.update(k, v, start)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=mask is None and T > 1)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        h = self.fc(self.ln2(x))
        h = F.relu(h).square() if self.mlp == "relu2" else F.gelu(h)
        x = x + self.fc_out(h)
        return x


class ZoomGPT(nn.Module):
    """block_size: the context length the model is trained for (generate's default
    window); it does not change the architecture or the weights."""

    def __init__(self, vocab, d=256, n_layer=6, n_head=4, zoom_ks=(),
                 gated=True, out_mode="slot", rope_pos="token", qk_norm=False,
                 merge_mode=None, grad_ckpt=False, mlp="gelu", softcap=0.0,
                 zero_init_proj=False, ce_chunk=0, block_size=256):
        super().__init__()
        if vocab < 1 or d < 1 or n_layer < 0 or n_head < 1:
            raise ValueError(f"bad sizes: vocab={vocab} d={d} n_layer={n_layer} n_head={n_head}")
        if d % n_head or (d // n_head) % 2:
            raise ValueError(f"d={d} must split into n_head={n_head} heads of even size (RoPE)")
        for name, val, ok in (("out_mode", out_mode, ("slot", "bag")),
                              ("rope_pos", rope_pos, ("token", "patch")),
                              ("mlp", mlp, ("gelu", "relu2")),
                              ("merge_mode", merge_mode or None, (None,) + MODES)):
            if val not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {val!r}")
        if softcap < 0 or ce_chunk < 0 or block_size < 1:
            raise ValueError(f"bad softcap={softcap} / ce_chunk={ce_chunk} / "
                             f"block_size={block_size}")
        zoom_ks = list(zoom_ks)
        self._config = dict(vocab=int(vocab), d=int(d), n_layer=int(n_layer), n_head=int(n_head),
                            zoom_ks=[int(k) for k in zoom_ks], gated=bool(gated),
                            out_mode=out_mode, rope_pos=rope_pos, qk_norm=bool(qk_norm),
                            merge_mode=merge_mode or None, grad_ckpt=bool(grad_ckpt), mlp=mlp,
                            softcap=float(softcap), zero_init_proj=bool(zero_init_proj),
                            ce_chunk=int(ce_chunk), block_size=int(block_size))
        self.d, self.n_layer, self.vocab, self.n_head = d, n_layer, vocab, n_head
        self.out_mode, self.rope_pos, self.grad_ckpt = out_mode, rope_pos, grad_ckpt
        self.softcap = softcap
        self.ce_chunk = ce_chunk
        self.block_size = block_size
        self.head_dim = d // n_head
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_head, qk_norm=qk_norm, mlp=mlp)
                                    for _ in range(n_layer))
        self.ln_f = RMSNorm(d)
        self.merge = nn.ModuleDict({str(k): SlotMerge(k, d, gated, merge_mode) for k in zoom_ks})
        self.unmerge = nn.ModuleDict(
            {str(k): SlotUnmerge(k, d, gated, merge_mode) for k in zoom_ks} if out_mode == "slot" else {})
        self._rope_cache = {}
        self.apply(self._init)
        for b in self.blocks:  # scaled residual init, or zero-init (speedrun)
            if zero_init_proj:
                nn.init.zeros_(b.proj.weight)
                nn.init.zeros_(b.fc_out.weight)
            else:
                nn.init.normal_(b.proj.weight, std=0.02 / (2 * n_layer) ** 0.5)
                nn.init.normal_(b.fc_out.weight, std=0.02 / (2 * n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def muon_params(self):
        """Hidden weight matrices (Muon); embedding/head/norms/gates stay on AdamW."""
        for b in self.blocks:   # re-tag: deepcopy / load_state_dict(assign=True) drop it
            b.qkv.weight.muon_split = 3
        return [p for b in self.blocks for p in
                (b.qkv.weight, b.proj.weight, b.fc.weight, b.fc_out.weight)]

    def muon_splits(self):
        """{parameter name: row blocks}: Muon orthogonalizes fused qkv as q, k, v."""
        return {f"blocks.{i}.qkv.weight": 3 for i in range(len(self.blocks))}

    def config(self):
        """Constructor kwargs: ZoomGPT(**m.config()) rebuilds the architecture."""
        return {**self._config, "zoom_ks": list(self._config["zoom_ks"])}

    def _make_rope(self, n_pos, k, device):
        p = torch.arange(n_pos, dtype=torch.float32, device=device)
        if self.rope_pos == "token":  # group p sits at its centre, in token units
            p = p * k + (k - 1) / 2
        return rope_tables(p, self.head_dim)

    def _rope_table(self, n_pos, k):
        """fp32 (cos, sin) covering positions 0..n_pos-1 at zoom k. One table per
        (k, device), grown to the longest length seen; built outside inference mode,
        so a table first made under torch.inference_mode() still works in training."""
        dev = self.emb.weight.device
        if _is_compiling():        # mutating the cache in a graph would force a recompile
            return self._make_rope(n_pos, k, dev)
        tab = self._rope_cache.get((k, dev))
        if tab is None or tab[0].shape[0] < n_pos:
            with torch.inference_mode(False), torch.no_grad():
                tab = self._make_rope(max(n_pos, self.block_size), k, dev)
            self._rope_cache[(k, dev)] = tab
        return tab

    def _rope(self, n_pos, k):
        cos, sin = self._rope_table(n_pos, k)
        return cos[:n_pos], sin[:n_pos]

    def hidden(self, idx, k=1):
        B, T = idx.shape
        if k != 1:
            if str(k) not in self.merge:
                raise ValueError(f"k={k} is not a zoom level of this model "
                                 f"(zoom_ks={self._config['zoom_ks']})")
            if T % k:
                raise ValueError(f"sequence length {T} is not divisible by k={k}")
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

    def logits(self, h):
        z = h @ self.emb.weight.t()                  # tied output head
        if self.softcap > 0:                         # bounded logits (Gemma-2 / speedrun)
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def forward(self, idx, targets=None, k=1):
        """idx: (B, T); targets: (B, T) where targets[:, i] = token at i + k.

        With targets: the mean cross-entropy, computed in fp32 (also for bf16 weights).
        Without: logits, (B, T, V) for k=1 and for slot mode (row i predicts token
        i + k); (B, T/k, V) for bag mode (one distribution per group)."""
        B, T = idx.shape
        h = self.hidden(idx, k)                      # (B, T/k, d)
        if targets is None:
            if k > 1 and self.out_mode == "slot":
                return self.logits(self.unmerge[str(k)](h)).reshape(B, T, -1)
            return self.logits(h)
        if k == 1:
            if self.ce_chunk:
                return chunked_cross_entropy(h, self.emb.weight, targets, self.ce_chunk,
                                             softcap=self.softcap)
            return F.cross_entropy(self.logits(h).float().view(-1, self.vocab),
                                   targets.reshape(-1))
        tgt = targets.view(B, T // k, k)
        if self.out_mode == "slot":
            hs = self.unmerge[str(k)](h)             # (B, T/k, k, d)
            if self.ce_chunk:
                return chunked_cross_entropy(hs, self.emb.weight, tgt, self.ce_chunk,
                                             softcap=self.softcap)
            z = self.logits(hs).float()              # (B, T/k, k, V)
            return F.cross_entropy(z.reshape(-1, self.vocab), tgt.reshape(-1))
        logp = F.log_softmax(self.logits(h).float(), dim=-1)  # (B, T/k, V), one dist per group
        return -logp.gather(-1, tgt).mean()

    def flops_per_step(self, batch, seq, k, exact=False):
        """Analytic training FLOPs (fwd + bwd = 3x fwd) for one step.

        Default: the budget convention of every recorded run (body + tied head,
        attention charged at the causal half, 2nd per position). exact=True counts
        the matmuls the kernels execute, as torch.utils.flop_counter does: full
        n x n attention (4nd per position; the SDPA backward recomputes the scores,
        2.5x), the merge/unmerge maps of merge_mode="linear", and the forward that
        grad_ckpt (body) and ce_chunk (head) recompute in the backward pass.
        Kernels that skip the masked half (CPU flash attention) do less attention
        work than counted here; the default charge is closer to them."""
        n = seq // k
        d, L, V = self.d, self.n_layer, self.vocab
        head_slots = k if (k > 1 and self.out_mode == "slot") else 1
        head = 2 * d * V * head_slots
        if not exact:
            body = L * (24 * d * d + 2 * n * d)      # per position, causal attention
            return 3 * batch * n * (body + head)
        merge = 0
        if k > 1 and str(k) in self.merge:
            merge += 2 * k * d * d * (self.merge[str(k)].mode == "linear")
            if self.out_mode == "slot":
                merge += 2 * k * d * d * (self.unmerge[str(k)].mode == "linear")
        ck = 1 if self.grad_ckpt else 0
        ce = 1 if self.ce_chunk and (k == 1 or self.out_mode == "slot") else 0
        # checkpoint recomputes qkv, proj, fc (16 d^2) and attention, but stops
        # before fc_out, whose output the backward pass does not need
        per_pos = (L * (3 * 24 + ck * 16) * d * d + (3.5 + ck) * L * 4 * n * d
                   + (3 + ce) * head + 3 * merge)
        return batch * n * per_pos

    # -- inference ---------------------------------------------------------------

    def _decode(self, tok, valid, caches, start):
        """Final hidden states (B, T, d) at k=1 for tok sitting at slots start.. .

        valid: None, or (B, >= start+T) bool marking real (not left-padding) slots;
        RoPE positions then count real tokens per row and pads are masked out.
        caches: one KVCache per layer (incremental decoding) or None."""
        B, T = tok.shape
        cos, sin = self._rope_table(start + T, 1)
        mask = None
        if valid is None:
            cos, sin = cos[start:start + T], sin[start:start + T]
        else:
            v = valid[:, :start + T]
            pos = (v.long().cumsum(-1) - 1).clamp_min(0)[:, start:]
            cos, sin = cos[pos].unsqueeze(1), sin[pos].unsqueeze(1)
            qi = torch.arange(start, start + T, device=tok.device)[:, None]
            ki = torch.arange(start + T, device=tok.device)[None, :]
            mask = ((ki <= qi) & v[:, None, :]) | (ki == qi)   # a pad attends to itself
            mask = mask.unsqueeze(1)
        x = self.emb(tok)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, None if caches is None else (caches[i], start), mask)
        return self.ln_f(x)

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=40, eos_id=None, top_p=None,
                 repetition_penalty=1.0, max_ctx=None, use_cache=True, generator=None):
        """Sample up to n_new tokens per prompt (k=1); returns prompt + new tokens.

        idx: (B, T) LongTensor of equal-length prompts, or a list of token-id lists
        of any lengths (left-padded internally; the result is then a list of lists).
        temperature <= 0: greedy. top_k: None/0 disables, clamped to the vocabulary.
        top_p: nucleus sampling (None/1 disables). repetition_penalty > 1 divides
        positive (multiplies negative) logits of tokens already in the row.
        eos_id: a finished row is padded with eos_id (list results stop at it);
        decoding stops once every row has finished.
        max_ctx: attention window, default self.block_size. The prompt primes a KV
        cache and each step feeds one token at its absolute RoPE position; once
        the context is longer than max_ctx, the last max_ctx tokens are re-encoded
        every step, exactly as a model without a cache sees them (use_cache=False
        always does that)."""
        dev = self.emb.weight.device
        as_list = not torch.is_tensor(idx)
        if as_list:
            rows = [torch.as_tensor(r, dtype=torch.long).reshape(-1) for r in idx]
            if not rows or any(r.numel() == 0 for r in rows):
                raise ValueError("generate: empty prompt (every row needs at least one token)")
            T0 = max(r.numel() for r in rows)
            tok = torch.zeros(len(rows), T0, dtype=torch.long)
            valid = torch.zeros(len(rows), T0, dtype=torch.bool)
            for i, r in enumerate(rows):
                tok[i, T0 - r.numel():], valid[i, T0 - r.numel():] = r, True
            tok, valid = tok.to(dev), (None if valid.all() else valid.to(dev))
        else:
            if idx.dim() != 2 or idx.numel() == 0:
                raise ValueError(f"generate: idx must be a non-empty (B, T) tensor of token "
                                 f"ids, got shape {tuple(idx.shape)}")
            tok, valid, T0 = idx.to(dev), None, idx.shape[1]
        if tok.min() < 0 or tok.max() >= self.vocab:
            raise ValueError(f"generate: token ids must lie in [0, {self.vocab})")
        max_ctx = int(max_ctx or self.block_size)
        if max_ctx < 1 or repetition_penalty <= 0 or (top_k is not None and top_k < 0):
            raise ValueError("generate: need max_ctx >= 1, repetition_penalty > 0, top_k >= 0")
        B, V = tok.shape[0], self.vocab
        rows_i = torch.arange(B, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        seen = None
        if repetition_penalty != 1.0:
            seen = torch.zeros(B, V + 1, device=dev)
            seen.scatter_(1, tok if valid is None else tok.masked_fill(~valid, V), 1.0)
            seen = seen[:, :V].bool()
        caches = None
        for _ in range(n_new):
            L = tok.shape[1]
            if use_cache and L <= max_ctx:
                if caches is None:                   # prime with the whole prompt
                    caches = [KVCache(max_ctx) for _ in self.blocks]
                    h = self._decode(tok, valid, caches, 0)
                else:                                # the newest token, at slot L-1
                    h = self._decode(tok[:, -1:], valid, caches, L - 1)
            else:                                    # sliding window: re-encode it
                h = self._decode(tok[:, -max_ctx:], None if valid is None
                                 else valid[:, -max_ctx:], None, 0)
            logits = self.logits(h[:, -1]).float()   # (B, V)
            if seen is not None:
                logits = torch.where(seen, torch.where(logits > 0, logits / repetition_penalty,
                                                       logits * repetition_penalty), logits)
            if temperature is None or temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(int(top_k), V)).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, -1)
                if top_p is not None and top_p < 1.0:   # smallest set reaching top_p
                    sp, si = probs.sort(-1, descending=True)
                    probs = probs.scatter(-1, si, sp.masked_fill(sp.cumsum(-1) - sp > top_p, 0.0))
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            if eos_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                done |= nxt == eos_id
            tok = torch.cat([tok, nxt[:, None]], 1)
            if valid is not None:
                valid = torch.cat([valid, valid.new_ones(B, 1)], 1)
            if seen is not None:
                seen[rows_i, nxt] = True
            if eos_id is not None and done.all():
                break
        if not as_list:
            return tok
        out = []
        for r, new in zip(rows, tok[:, T0:].tolist()):
            if eos_id is not None and eos_id in new:
                new = new[:new.index(eos_id) + 1]
            out.append(r.tolist() + new)
        return out

    # -- checkpoints ---------------------------------------------------------------

    def save_pretrained(self, out_dir, strip_zoom=True, tokenizer_path=None, eos_id=None):
        """Write out_dir/config.json + model.safetensors (model.pt, a plain tensor dict,
        if safetensors is not installed). strip_zoom drops the merge/unmerge maps,
        which only training at k>1 uses, leaving a plain k=1 model. tokenizer_path:
        a tokenizer.json, or a data dir holding one, copied alongside; eos_id
        (default: the data dir's meta.json) is recorded in config.json."""
        os.makedirs(out_dir, exist_ok=True)
        cfg = self.config()
        sd = self.state_dict()
        if strip_zoom:
            cfg["zoom_ks"] = []
            sd = {n: t for n, t in sd.items() if not n.startswith(("merge.", "unmerge."))}
        sd = {n: t.detach().cpu().contiguous() for n, t in sd.items()}
        info = {"model_type": "zoomgpt",
                "torch_dtype": str(self.emb.weight.dtype).replace("torch.", "")}
        if tokenizer_path:
            src = tokenizer_path
            if os.path.isdir(src):
                src = os.path.join(src, "tokenizer.json")
            dst = os.path.join(out_dir, "tokenizer.json")
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copyfile(src, dst)
            meta = os.path.join(os.path.dirname(os.path.abspath(src)), "meta.json")
            if eos_id is None and os.path.exists(meta):
                with open(meta) as f:
                    eos_id = json.load(f).get("eos_id")
        if eos_id is not None:
            info["eos_id"] = int(eos_id)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({**info, **cfg}, f, indent=1)
        try:
            from safetensors.torch import save_file
        except ImportError:
            torch.save(sd, os.path.join(out_dir, "model.pt"))
        else:
            save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        return out_dir

    @classmethod
    def from_pretrained(cls, path, map_location="cpu"):
        """Rebuild a model written by save_pretrained, in eval mode. Weights keep
        their saved dtype and are loaded into a meta-device skeleton (no second copy)."""
        with open(os.path.join(path, "config.json")) as f:
            raw = json.load(f)
        names = set(inspect.signature(cls.__init__).parameters) - {"self"}
        cfg = {k: v for k, v in raw.items() if k in names}
        st = os.path.join(path, "model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file
            sd = load_file(st, device=str(torch.device(map_location)))
        else:
            sd = torch.load(os.path.join(path, "model.pt"), map_location=map_location,
                            weights_only=True)
        with torch.device("meta"):
            model = cls(**cfg)
        model.load_state_dict(sd, strict=True, assign=True)
        return model.eval()
