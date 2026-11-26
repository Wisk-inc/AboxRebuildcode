"""Optimizers: Muon for hidden weight matrices, AdamW for everything else.

Muon (Jordan et al., 2024) replaces each 2-D weight update with the nearest
semi-orthogonal matrix of its momentum, computed by a few Newton-Schulz
iterations. It is all matmuls, so it runs the same on CPU and GPU, and it
needs no second-moment state (half of AdamW's optimizer memory for those
weights).

Update scaling follows Moonlight (Liu et al., 2025): scale = 0.2*sqrt(max(m, n))
makes the update RMS match AdamW's, so Muon and AdamW share one learning rate
and one weight decay. scale="jordan" uses the original max(1, m/n)**0.5.

Low-memory options (see bantamtrain.lowmem):
  state_bits=8   blockwise 8-bit momentum / Adam moments (4x less state memory)
  bf16 params    updates are computed in fp32 and written back with
                 stochastic rounding, so no fp32 master copy is needed

Muon and AdamWLowMem update each parameter in chunks of STEP_CHUNK elements
(fp32 temporaries are O(chunk)), draw rounding noise from a generator seeded by
(sr_seed, step, parameter index, chunk) so DDP ranks stay identical and resume
is exact, and keep only plain tensors in state_dict() (8-bit buffers as codes
plus fp32 scales), so checkpoints load with torch.load(weights_only=True).
"""
import math
import warnings
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .lowmem import QBLOCK, QState, write_param
from .qweights import QBase, QTiedHead

NS_COEFFS = (3.4445, -4.7750, 2.0315)
# Polar Express (Amsel et al., 2025, Appendix A): per-step minimax coefficients, same
# matmul cost as the quintic iteration. a, b, c of every tuple but the last are divided
# by 1.01, 1.01^3, 1.01^5 for safety; the last (fixed point 1) repeats for further steps,
# and the input is normalized with a 1.01 margin.
_PE = [(8.28721201814563, -23.595886519098837, 17.300387312530933),
       (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
       (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
       (3.3184196573706015, -2.488488024314874, 0.51004894012372),
       (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
       (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
       (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
       (1.875, -1.25, 0.375)]
POLAR_COEFFS = [(a / 1.01, b / 1.01 ** 3, c / 1.01 ** 5) for a, b, c in _PE[:-1]] + [_PE[-1]]

STEP_CHUNK = 1 << 22            # elements per chunk of an optimizer step (multiple of QBLOCK)
ADAMW_FLOPS_PER_PARAM = 12      # moments, bias correction, sqrt, divide, decay, update
MUON_ELEMENTWISE_FLOPS = 10     # momentum, Nesterov blend, scale, decay, update


def _ns_owned_(X, steps, eps, variant):
    """Newton-Schulz on an owned (batch, r, c) tensor with r <= c (any strides).

    Normalizes X in place and alternates between X and one more buffer, so the
    working set is two copies of X plus the (r, r) products.
    """
    if variant == "polar":
        coeffs, margin = (POLAR_COEFFS + [POLAR_COEFFS[-1]] * steps)[:steps], 1.01
    else:
        coeffs, margin = [NS_COEFFS] * steps, 1.0
    X.div_(X.norm(dim=(-2, -1), keepdim=True) * margin + eps)
    Y = torch.empty_like(X)
    for a, b, c in coeffs:
        A = torch.bmm(X, X.mT)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)          # b*A + c*A@A
        torch.baddbmm(X, B, X, beta=a, out=Y)                 # a*X + B@X
        X, Y = Y, X
    return X


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7, variant="quintic"):
    """Approximate U V^T of G = U S V^T (batched over leading dims)."""
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    lead = X.shape[:-2]
    X = X.reshape(-1, *X.shape[-2:])
    if X.dtype == G.dtype or not X.is_contiguous():
        X = X.clone(memory_format=torch.contiguous_format)
    X = _ns_owned_(X, steps, eps, variant).view(*lead, *X.shape[-2:])
    if transpose:
        X = X.mT
    return X.to(G.dtype)


def ns_flops(rows, cols, steps=5, batch=1):
    """FLOPs of newton_schulz on `batch` (rows, cols) matrices."""
    a, b = min(rows, cols), max(rows, cols)
    per_iter = 4 * a * a * b + 2 * a ** 3 + 2 * a * a + 2 * a * b
    return float(batch * (steps * per_iter + 3 * a * b))


_M64 = (1 << 64) - 1


def _seed(*xs):
    """splitmix64 over a tuple of ints: a 63-bit seed, the same on every rank and run."""
    h = 0x9E3779B97F4A7C15
    for x in xs:
        h = ((h ^ (int(x) & _M64)) * 0xBF58476D1CE4E5B9) & _M64
        h = ((h ^ (h >> 31)) * 0x94D049BB133111EB) & _M64
        h ^= h >> 29
    return h & ((1 << 63) - 1)


def _check_loadable(opt, sd):
    """Clear errors (instead of a KeyError at the next step) when a saved optimizer
    state does not fit `opt`."""
    mine, saved = opt.param_groups, sd["param_groups"]
    if [len(g["params"]) for g in saved] != [len(g["params"]) for g in mine]:
        raise ValueError(
            f"{type(opt).__name__}: the checkpoint's parameter groups have sizes "
            f"{[len(g['params']) for g in saved]}, this optimizer's "
            f"{[len(g['params']) for g in mine]}: the model or the optimizer setup "
            "(--optim, LoRA/Muon routing, lr_scale) differs from the checkpoint")

    def bits(gs):
        return [g.get("state_bits", "32 (torch AdamW)") for g in gs]
    if bits(saved) != bits(mine):
        raise ValueError(
            f"{type(opt).__name__}: the checkpoint's optimizer state has state_bits="
            f"{bits(saved)}, this optimizer uses state_bits={bits(mine)}: resume with the "
            "same --state_bits and --param_dtype as the checkpoint")


def _fill_new_keys(opt, before):
    """Hyperparameters added since a checkpoint was written keep this run's values."""
    for g, cur in zip(opt.param_groups, before):
        for k, v in cur.items():
            g.setdefault(k, v)


class _ChunkedOptimizer(Optimizer):
    """Shared machinery of Muon and AdamWLowMem (see the module docstring)."""

    _TAG = 0                     # separates the noise streams of the two optimizers
    _EXTRA = ("chunk",)          # attributes kept when pickling / deep-copying

    def __init__(self, params, defaults, chunk=STEP_CHUNK):
        super().__init__(params, defaults)
        self.chunk = max(QBLOCK, chunk // QBLOCK * QBLOCK)

    def __getstate__(self):
        d = super().__getstate__()
        d.update({k: self.__dict__[k] for k in self._EXTRA if k in self.__dict__})
        return d

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("chunk", STEP_CHUNK)

    def _rng(self, group, p, index, step, chunk_i):
        """Generator for one chunk of one parameter's update at `step`."""
        gens = self.__dict__.setdefault("_gens", {})
        g = gens.get(p.device)
        if g is None:
            g = gens[p.device] = torch.Generator(device=p.device)
        g.manual_seed(_seed(group["sr_seed"], self._TAG, index, step, chunk_i))
        return g

    @staticmethod
    def _new_buf(p, bits, kind):
        if bits == 8:
            return QState(p, kind)
        return torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    @staticmethod
    def _get(buf, s, e):
        """fp32 elements [s, e) of a state buffer (a view of a 32-bit buffer)."""
        return buf.get(s, e) if isinstance(buf, QState) else buf.view(-1)[s:e]

    @staticmethod
    def _put(buf, x, s, stochastic=False, generator=None):
        if isinstance(buf, QState):              # 32-bit buffers were updated in place
            buf.set(x, s, stochastic, generator)

    @staticmethod
    def _write(pc, upd, decay, alpha, stochastic, generator):
        """pc <- pc * decay + alpha * upd in fp32, stored in pc's dtype."""
        if pc.dtype in (torch.float32, torch.float64):
            pc.mul_(decay).add_(upd, alpha=alpha)
        else:
            write_param(pc, pc.float().mul_(decay).add_(upd, alpha=alpha), stochastic, generator)

    def state_dict(self):
        sd = super().state_dict()
        packed = {}
        for i, st in sd["state"].items():
            packed[i] = {}
            for k, v in st.items():
                if isinstance(v, QState):
                    packed[i].update({f"{k}.{n}": t for n, t in v.tensors().items()})
                else:
                    packed[i][k] = v
        sd["state"] = packed
        return sd

    def load_state_dict(self, state_dict):
        """Load a state_dict() (or an older pickled one with QState objects).

        torch's loader would cast fp32 moments to a bf16 parameter's dtype and
        leave 8-bit buffers on the CPU; here every buffer comes back in fp32
        (codes as saved) on its parameter's device.
        """
        saved_groups = state_dict["param_groups"]
        _check_loadable(self, state_dict)
        before = [dict(g) for g in self.param_groups]
        super().load_state_dict({**state_dict, "state": {}})
        _fill_new_keys(self, before)
        saved = state_dict["state"]
        params = [p for g in self.param_groups for p in g["params"]]
        ids = [i for g in saved_groups for i in g["params"]]
        for i, p in zip(ids, params):
            if i in saved:
                self.state[p] = self._unpack(p, saved[i])

    @staticmethod
    def _unpack(p, st):
        out, quant = {}, {}
        for k, v in st.items():
            if isinstance(v, QState):                         # pickled by bantamtrain <= 0.2
                out[k] = QState.from_tensors(p.shape, v.tensors(), p.device)
            elif isinstance(k, str) and "." in k:             # "m.q", "m.s", "v.lo", ...
                name, part = k.rsplit(".", 1)
                quant.setdefault(name, {})[part] = v
            elif torch.is_tensor(v) and v.is_floating_point():
                out[k] = v.to(p.device, torch.float32).contiguous()
            elif torch.is_tensor(v):
                out[k] = v.to(p.device)
            else:
                out[k] = v
        for name, d in quant.items():
            out[name] = QState.from_tensors(p.shape, d, p.device)
        return out


class Muon(_ChunkedOptimizer):
    """Muon for 2-D parameters.

    A parameter can be orthogonalized as `s` independent row blocks of its
    (out, in) matrix (e.g. a fused qkv projection as q, k and v) and can be
    stored transposed, (in, out), like transformers' Conv1D. Both come from
    `splits` ({param: s}) and `transposed` (params), else from the attributes
    p.muon_split / p.muon_transpose.
    """

    _TAG = 1
    _EXTRA = ("chunk", "_split", "_transposed")

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, scale="moonlight", state_bits=32, stochastic=True,
                 ns="quintic", sr_seed=0, splits=None, transposed=(), chunk=STEP_CHUNK):
        assert scale in ("moonlight", "jordan") and state_bits in (32, 8)
        assert ns in ("quintic", "polar")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, scale=scale, state_bits=state_bits,
                        stochastic=stochastic, ns=ns, sr_seed=sr_seed)
        super().__init__(params, defaults, chunk)
        self._split = {p: int(s) for p, s in (splits or {}).items()}
        self._transposed = {p: True for p in transposed}

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("_split", {})
        self.__dict__.setdefault("_transposed", {})

    @staticmethod
    def _scale(m, n, mode):
        if mode == "moonlight":
            return 0.2 * math.sqrt(max(m, n))
        return max(1.0, m / n) ** 0.5

    def layout(self, p):
        """(row blocks, stored transposed) of parameter p."""
        split = self._split.get(p) or getattr(p, "muon_split", 1)
        return split, self._transposed.get(p, getattr(p, "muon_transpose", False))

    def _blocks(self, p):
        split, tr = self.layout(p)
        tr = tr and p.dim() == 2
        out = p.shape[1] if tr else p.shape[0]
        return split, tr, out // split, p.numel() // out

    def param_flops(self, p, group):
        """FLOPs of one update of p: Newton-Schulz per block plus elementwise work."""
        split, _, rows, cols = self._blocks(p)
        return ns_flops(rows, cols, group["ns_steps"], split) + MUON_ELEMENTWISE_FLOPS * p.numel()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = -1
        for group in self.param_groups:
            for p in group["params"]:
                index += 1
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = self._new_buf(p, group["state_bits"], "signed")
                st["step"] = st.get("step", 0) + 1
                w = p if p.is_contiguous() else p.detach().contiguous()
                self._update(w, p.grad, st, group, index, self._blocks(p))
                if w is not p:
                    p.copy_(w)
        return loss

    def _update(self, p, grad, st, group, index, blocks):
        mom, n, C = group["momentum"], p.numel(), self.chunk
        # momentum (8-bit buffers are decoded and re-encoded chunk by chunk) -> G in fp32
        G = torch.empty(p.shape, dtype=torch.float32, device=p.device)
        gf, Gf = grad.reshape(-1), G.view(-1)
        for s in range(0, n, C):
            e = min(s + C, n)
            g = gf[s:e].float()
            buf = self._get(st["buf"], s, e)
            buf.lerp_(g, 1 - mom)
            self._put(st["buf"], buf, s)
            if group["nesterov"]:
                torch.lerp(g, buf, mom, out=Gf[s:e])
            else:
                Gf[s:e] = buf
        # orthogonalize each (rows, cols) block of the (out, in) matrix in its wide orientation
        split, tr, rows, cols = blocks
        M = (G.mT if tr else G.view(G.shape[0], -1)).reshape(split, rows, cols)
        tall = rows > cols
        X = (M.mT if tall else M).to(torch.bfloat16 if p.is_cuda else torch.float32,
                                     memory_format=torch.contiguous_format)
        del G, M                                 # X is a view of G or a copy (G freed)
        U = _ns_owned_(X, group["ns_steps"], 1e-7, group["ns"])
        del X
        U = U.mT if tall else U                  # (split, rows, cols)
        # p <- p * (1 - lr * wd) - lr * scale * U, a few rows at a time
        lr = group["lr"]
        alpha = -lr * self._scale(rows, cols, group["scale"])
        L = p.mT if tr else p.view(p.shape[0], -1)            # (out, in) view of p
        sr = group["stochastic"] and p.dtype == torch.bfloat16
        R, ci = max(1, C // cols), 0
        for j in range(split):
            for r0 in range(0, rows, R):
                r1 = min(r0 + R, rows)
                gen = self._rng(group, p, index, st["step"], ci) if sr else None
                self._write(L[j * rows + r0:j * rows + r1], U[j, r0:r1],
                            1 - lr * group["weight_decay"], alpha, group["stochastic"], gen)
                ci += 1


class AdamWLowMem(_ChunkedOptimizer):
    """AdamW with optional 8-bit moments and bf16 parameters (stochastic rounding).

    Same update as torch.optim.AdamW (decoupled weight decay, bias correction).
    v_rounding: rounding of the 8-bit second-moment codes. Round-to-nearest lets
    v ratchet upward once its per-step change is far below a code step (median
    v8/v32 1.09 at beta2=0.99, 1.2-1.25 at 0.999; unbiased at 0.95);
    "stochastic" rounding keeps it unbiased (median 0.99-1.01) at somewhat more
    per-element noise. "auto" (default): stochastic when betas[1] > 0.95.
    """

    _TAG = 2

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 state_bits=8, stochastic=True, v_rounding="auto", sr_seed=0, chunk=STEP_CHUNK):
        assert state_bits in (32, 8) and v_rounding in ("auto", "nearest", "stochastic")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      state_bits=state_bits, stochastic=stochastic,
                                      v_rounding=v_rounding, sr_seed=sr_seed), chunk)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = -1
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            decay = 1 - lr * group["weight_decay"]
            v_sr = group["v_rounding"] == "stochastic" or (group["v_rounding"] == "auto"
                                                            and b2 > 0.95)
            for p in group["params"]:
                index += 1
                if p.grad is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["m"] = self._new_buf(p, group["state_bits"], "signed")
                    st["v"] = self._new_buf(p, group["state_bits"], "log")
                st["step"] += 1
                t = st["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                v_sr_p = v_sr and isinstance(st["v"], QState)
                sr = group["stochastic"] and p.dtype == torch.bfloat16
                w = p if p.is_contiguous() else p.detach().contiguous()
                pf, gf, n = w.view(-1), p.grad.reshape(-1), p.numel()
                for ci, s in enumerate(range(0, n, self.chunk)):
                    e = min(s + self.chunk, n)
                    gen = self._rng(group, p, index, t, ci) if (sr or v_sr_p) else None
                    g = gf[s:e].float()
                    m, v = self._get(st["m"], s, e), self._get(st["v"], s, e)
                    m.lerp_(g, 1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    self._put(st["m"], m, s)
                    self._put(st["v"], v, s, v_sr_p, gen)
                    upd = (m / bc1).div_((v / bc2).sqrt_().add_(eps))
                    self._write(pf[s:e], upd, decay, -lr, group["stochastic"], gen)
                if w is not p:
                    p.copy_(w)
        return loss


_HOOK_DICTS = ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks",
               "_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks",
               "_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks")


class CombinedOptimizer(Optimizer):
    """Several optimizers over disjoint parameters (Muon + AdamW) as one torch Optimizer.

    param_groups is the concatenation of the sub-optimizers' groups (the same
    dicts), so LR schedulers, `g["lr"] = ...` loops, clip_grad_norm_ and
    torch.amp.GradScaler.unscale_/step(opt) work on it directly (one inf check
    covers every parameter). state_dict() is a regular optimizer state dict over
    all parameters plus the sub-optimizer layout; load_state_dict() also takes the
    per-optimizer list written by bantamtrain <= 0.2.
    """

    def __init__(self, optimizers):
        self.optimizers = [o for o in optimizers if o is not None]
        self.defaults = dict(self.optimizers[0].defaults) if self.optimizers else {}
        for k in _HOOK_DICTS:
            setattr(self, k, OrderedDict())
        self._patch_step_function()

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    @property
    def state(self):
        """Merged view of the sub-optimizers' state (per-parameter dicts are shared)."""
        merged = defaultdict(dict)
        for o in self.optimizers:
            merged.update(o.state)
        return merged

    def __getstate__(self):
        return {"optimizers": self.optimizers, "defaults": self.defaults}

    def __repr__(self):
        return f"CombinedOptimizer({', '.join(repr(o) for o in self.optimizers)})"

    def layout(self):
        return [[type(o).__name__, len(o.param_groups)] for o in self.optimizers]

    def summary(self):
        """[(optimizer class, number of parameters it updates)], e.g. for run logs."""
        return [(type(o).__name__, sum(p.numel() for g in o.param_groups for p in g["params"]))
                for o in self.optimizers]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss

    def add_param_group(self, param_group):
        """Add to the Muon optimizer if param_group["use_muon"], else to the first other one."""
        use_muon = param_group.pop("use_muon", False)
        param_group.setdefault("lr_scale", 1.0)
        for o in self.optimizers:
            if isinstance(o, Muon) == bool(use_muon):
                return o.add_param_group(param_group)
        raise ValueError("no sub-optimizer of that kind to add the group to")

    def state_dict(self):
        state, groups, off = {}, [], 0
        for o in self.optimizers:
            sd = o.state_dict()
            state.update({off + i: s for i, s in sd["state"].items() if isinstance(i, int)})
            groups += [{**g, "params": [off + i for i in g["params"]]} for g in sd["param_groups"]]
            off += sum(len(g["params"]) for g in sd["param_groups"])
        return {"state": state, "param_groups": groups, "optimizers": self.layout()}

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, (list, tuple)):        # bantamtrain <= 0.2
            if len(state_dict) != len(self.optimizers):
                raise ValueError(f"checkpoint holds {len(state_dict)} optimizer states, this "
                                 f"CombinedOptimizer has {len(self.optimizers)} "
                                 f"({[type(o).__name__ for o in self.optimizers]}): resume "
                                 "with the same --optim / --state_bits / --param_dtype")
            subs = list(state_dict)
        else:
            saved = [list(x) for x in state_dict.get("optimizers", [])]
            if saved != self.layout():
                raise ValueError(f"checkpoint optimizers {saved} differ from this run's "
                                 f"{self.layout()}: resume with the same --optim / "
                                 "--state_bits / --param_dtype")
            subs, g0 = [], 0
            for _, ng in saved:
                groups = state_dict["param_groups"][g0:g0 + ng]
                g0 += ng
                local = {i: j for j, i in enumerate(i for g in groups for i in g["params"])}
                subs.append({"state": {local[i]: s for i, s in state_dict["state"].items()
                                       if i in local},
                             "param_groups": [{**g, "params": [local[i] for i in g["params"]]}
                                              for g in groups]})
        for o, sd in zip(self.optimizers, subs):          # validate all before loading any
            _check_loadable(o, sd)
        for o, sd in zip(self.optimizers, subs):
            before = [dict(g) for g in o.param_groups]
            o.load_state_dict(sd)
            _fill_new_keys(o, before)


def optimizer_flops(opt, with_grad_only=False):
    """FLOPs of ONE optimizer step.

    Muon: Newton-Schulz per matrix (per muon_split block) plus ~10 FLOPs/element;
    AdamW-style optimizers ~12 FLOPs/param; SGD 2-6. Frozen parameters (and,
    with with_grad_only, parameters without a gradient this step) count 0.
    Works for CombinedOptimizer, Muon, AdamWLowMem and torch optimizers.
    """
    if hasattr(opt, "optimizers"):
        return float(sum(optimizer_flops(o, with_grad_only) for o in opt.optimizers))
    total = 0.0
    for g in opt.param_groups:
        for p in g["params"]:
            if not p.requires_grad or (with_grad_only and p.grad is None):
                continue
            if isinstance(opt, Muon):
                total += opt.param_flops(p, g)
            elif isinstance(opt, torch.optim.SGD):
                total += (2 + 2 * bool(g.get("momentum")) + 2 * bool(g.get("weight_decay"))) \
                    * p.numel()
            else:
                total += ADAMW_FLOPS_PER_PARAM * p.numel()
    return total


def _is_conv1d(m):
    """transformers' Conv1D (GPT-2 style): a linear layer with weight stored (in, out).
    The name is checked first: .weight of a quantized module is a full dequantization."""
    if type(m).__name__ != "Conv1D":
        return False
    w = getattr(m, "weight", None)
    return isinstance(w, torch.Tensor) and w.dim() == 2


def _is_norm(m):
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._NormBase)
    return isinstance(m, norm_types + ((nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ())) \
        or type(m).__name__.endswith("Norm")


def default_muon_params(model, include_lora=False):
    """Muon candidates for an arbitrary model: trainable 2-D weights of nn.Linear
    (and transformers Conv1D) layers that are not embeddings or the output head
    (and not LoRA adapters unless include_lora). Works for HuggingFace causal LMs.
    Quantized modules (bantamtrain.qweights) are frozen and skipped without touching
    their .weight, which would dequantize the whole matrix."""
    quant = (QBase, QTiedHead)
    skip = {id(m.weight) for m in model.modules()
            if isinstance(m, nn.Embedding) and not isinstance(m, quant)}
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        out = get_out()
        if out is not None and not isinstance(out, quant) and hasattr(out, "weight"):
            skip.add(id(out.weight))
    params, seen = [], set()

    def add(p):
        if p.requires_grad and p.dim() == 2 and id(p) not in skip and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    for name, m in model.named_modules():
        if isinstance(m, quant):
            continue
        if (isinstance(m, nn.Linear) or _is_conv1d(m)) and "lora_" not in name \
                and name.split(".")[-1] not in ("lm_head", "head", "output", "score"):
            add(m.weight)
    if include_lora:
        for name, p in model.named_parameters():
            if "lora_" in name and "magnitude" not in name:
                add(p)
    return params


def no_decay_params(model, decay_embeddings=True):
    """ids of the parameters that take no weight decay, decided by role: tensors
    with fewer than 2 dims (biases, norm scales, gates), every parameter of a norm
    layer or of SlotMerge / SlotUnmerge (gates and linear merge maps), and the
    embedding tables when decay_embeddings=False."""
    ids = {id(p) for p in model.parameters() if p.dim() < 2}
    for m in model.modules():
        if type(m).__name__ in ("SlotMerge", "SlotUnmerge") or _is_norm(m) \
                or (not decay_embeddings and isinstance(m, nn.Embedding)):
            ids.update(id(p) for p in m.parameters(recurse=False))
    return ids


def build_optimizer(model, name, lr, weight_decay, betas=(0.9, 0.95), fused=None,
                    state_bits=32, ns="quintic", muon_lora=False, lr_scale=None, sr_seed=0,
                    decay_embeddings=True):
    """name: 'adamw' or 'muon'.

    Muon takes the 2-D weights the model lists in `model.muon_params()`, or,
    for any other model (e.g. a HuggingFace LM), the hidden nn.Linear / Conv1D
    weights found by `default_muon_params`; row blocks come from
    `model.muon_splits()` ({param name: blocks}) when present, else from
    p.muon_split. Embeddings, output head, norms, gates, biases and (by default)
    LoRA adapters stay on AdamW. Weight decay skips the roles listed in
    `no_decay_params` (embeddings are decayed unless decay_embeddings=False).

    lr_scale: optional callable (param_name, param) -> multiplier. Parameters with
    a multiplier != 1 get their own groups with initial lr = lr * multiplier.
    Every group carries g["lr_scale"], so a schedule sets
    g["lr"] = scheduled_lr * g["lr_scale"].

    state_bits=8, or bf16 parameters, switch AdamW to AdamWLowMem (8-bit
    moments / stochastic-rounding writes); otherwise torch's fused AdamW is used.
    sr_seed seeds the rounding noise (keep it equal on every DDP rank).
    """
    named = dict(model.named_parameters())
    muon_ids, splits, transposed = set(), {}, []
    if name == "muon":
        cands = model.muon_params() if hasattr(model, "muon_params") \
            else default_muon_params(model, include_lora=muon_lora)
        muon_ids = {id(p) for p in cands if p.requires_grad}
        if hasattr(model, "muon_splits"):
            splits = {named[n]: int(s) for n, s in model.muon_splits().items() if n in named}
        transposed = [m.weight for m in model.modules() if _is_conv1d(m)]
    no_decay = no_decay_params(model, decay_embeddings)
    adam = {("decay", 1.0): [], ("no_decay", 1.0): []}     # the first two groups, always
    muon = {1.0: []}
    for n, p in named.items():
        if not p.requires_grad:
            continue
        s = 1.0 if lr_scale is None else float(lr_scale(n, p))
        if id(p) in muon_ids:
            muon.setdefault(s, []).append(p)
        else:
            adam.setdefault(("no_decay" if id(p) in no_decay else "decay", s), []).append(p)
    groups = [{"params": ps, "weight_decay": weight_decay if kind == "decay" else 0.0,
               "lr": lr * s, "lr_scale": s} for (kind, s), ps in adam.items()]
    trainable = [p for p in model.parameters() if p.requires_grad]
    bf16_params = any(p.dtype == torch.bfloat16 for p in trainable)
    if state_bits == 8 or bf16_params:
        adamw = AdamWLowMem(groups, lr=lr, betas=betas, eps=1e-8, state_bits=state_bits,
                            sr_seed=sr_seed)
    else:
        kw = {}
        if fused is None:
            fused = next(model.parameters()).is_cuda
        if fused:
            kw["fused"] = True
        adamw = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **kw)
    muon_opt = None
    if any(muon.values()):
        muon_opt = Muon([{"params": ps, "lr": lr * s, "lr_scale": s} for s, ps in muon.items()],
                        lr=lr, weight_decay=weight_decay, state_bits=state_bits, ns=ns,
                        sr_seed=sr_seed, splits=splits, transposed=transposed)
    elif name == "muon" and trainable:
        warnings.warn("build_optimizer('muon'): no trainable parameter qualifies for Muon "
                      "(e.g. LoRA adapters without muon_lora=True on a frozen base), so every "
                      "parameter is trained by AdamW", stacklevel=2)
    return CombinedOptimizer([adamw, muon_opt])
