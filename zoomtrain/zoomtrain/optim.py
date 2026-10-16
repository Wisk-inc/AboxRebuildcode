"""Optimizers: Muon for hidden weight matrices, AdamW for everything else.

Muon (Jordan et al., 2024) replaces each 2-D weight update with the nearest
semi-orthogonal matrix of its momentum, computed by a few Newton-Schulz
iterations. It is all matmuls, so it runs the same on CPU and GPU, and it
needs no second-moment state (half of AdamW's optimizer memory for those
weights).

Update scaling follows Moonlight (Liu et al., 2025): scale = 0.2*sqrt(max(m, n))
makes the update RMS match AdamW's, so Muon and AdamW share one learning rate
and one weight decay. scale="jordan" uses the original max(1, m/n)**0.5.

Low-memory options (see zoomtrain.lowmem):
  state_bits=8   blockwise 8-bit momentum / Adam moments (4x less state memory)
  bf16 params    updates are computed in fp32 and written back with
                 stochastic rounding, so no fp32 master copy is needed
"""
import math

import torch
from torch.optim import Optimizer

from .lowmem import QState, write_param

NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """Approximate U V^T of G = U S V^T (batched over leading dims)."""
    a, b, c = NS_COEFFS
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon for 2-D parameters.

    A parameter may carry `p.muon_split = s` to be orthogonalized as s
    independent row blocks (e.g. a fused qkv projection as q, k and v).
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, scale="moonlight", state_bits=32, stochastic=True):
        assert scale in ("moonlight", "jordan") and state_bits in (32, 8)
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, scale=scale, state_bits=state_bits,
                        stochastic=stochastic)
        super().__init__(params, defaults)

    @staticmethod
    def _scale(m, n, mode):
        if mode == "moonlight":
            return 0.2 * math.sqrt(max(m, n))
        return max(1.0, m / n) ** 0.5

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = (QState(p, "signed") if group["state_bits"] == 8
                                    else torch.zeros_like(p, dtype=torch.float32))
                q = isinstance(state["buf"], QState)
                buf = state["buf"].get() if q else state["buf"]
                buf.lerp_(g, 1 - group["momentum"])
                if q:
                    state["buf"].set(buf)
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                split = getattr(p, "muon_split", 1)
                rows, cols = g.shape[0] // split, g.shape[1]
                u = newton_schulz(g.reshape(split, rows, cols), group["ns_steps"])
                u = u.reshape(p.shape) * self._scale(rows, cols, group["scale"])
                new = p.float() * (1 - group["lr"] * group["weight_decay"]) - group["lr"] * u
                write_param(p, new, group["stochastic"])
        return loss


class AdamWLowMem(Optimizer):
    """AdamW with optional 8-bit moments and bf16 parameters (stochastic rounding).

    Same update as torch.optim.AdamW (decoupled weight decay, bias correction).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 state_bits=8, stochastic=True):
        assert state_bits in (32, 8)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      state_bits=state_bits, stochastic=stochastic))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    if group["state_bits"] == 8:
                        st["m"], st["v"] = QState(p, "signed"), QState(p, "log")
                    else:
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                q = isinstance(st["m"], QState)
                m = st["m"].get() if q else st["m"]
                v = st["v"].get() if q else st["v"]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if q:
                    st["m"].set(m)
                    st["v"].set(v)
                t = st["step"]
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                new = p.float() * (1 - group["lr"] * group["weight_decay"]) \
                    - group["lr"] * mhat / (vhat.sqrt() + group["eps"])
                write_param(p, new, group["stochastic"])
        return loss


class CombinedOptimizer:
    """Steps several optimizers as one; exposes the union of their param groups."""

    def __init__(self, optimizers):
        self.optimizers = [o for o in optimizers if o is not None]

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)


def build_optimizer(model, name, lr, weight_decay, betas=(0.9, 0.95), fused=None,
                    state_bits=32):
    """name: 'adamw' or 'muon'.

    Muon takes the 2-D weights the model lists in `model.muon_params()`
    (hidden matrices inside the blocks). Embeddings, output head, norms, gates
    and biases stay on AdamW, as recommended for Muon.

    state_bits=8, or bf16 parameters, switch AdamW to AdamWLowMem (8-bit
    moments / stochastic-rounding writes); otherwise torch's fused AdamW is used.
    """
    muon_ids = set()
    if name == "muon":
        muon_ids = {id(p) for p in model.muon_params()}
    muon_p, decay, no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in muon_ids:
            muon_p.append(p)
        elif p.dim() >= 2 and "gate" not in n and ".merge." not in n and ".unmerge." not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    bf16_params = next(model.parameters()).dtype == torch.bfloat16
    if state_bits == 8 or bf16_params:
        adamw = AdamWLowMem(groups, lr=lr, betas=betas, eps=1e-8, state_bits=state_bits)
    else:
        kw = {}
        if fused is None:
            fused = next(model.parameters()).is_cuda
        if fused:
            kw["fused"] = True
        adamw = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **kw)
    muon = (Muon(muon_p, lr=lr, weight_decay=weight_decay, state_bits=state_bits)
            if muon_p else None)
    return CombinedOptimizer([adamw, muon])
