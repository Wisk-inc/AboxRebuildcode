"""Optimizers: Muon for hidden weight matrices, AdamW for everything else.

Muon (Jordan et al., 2024) replaces each 2-D weight update with the nearest
semi-orthogonal matrix of its momentum, computed by a few Newton-Schulz
iterations. It is all matmuls, so it runs the same on CPU and GPU, and it
needs no second-moment state (half of AdamW's optimizer memory for those
weights).

Update scaling follows Moonlight (Liu et al., 2025): scale = 0.2*sqrt(max(m, n))
makes the update RMS match AdamW's, so Muon and AdamW share one learning rate
and one weight decay. scale="jordan" uses the original max(1, m/n)**0.5.
"""
import math

import torch
from torch.optim import Optimizer

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
                 weight_decay=0.0, scale="moonlight"):
        assert scale in ("moonlight", "jordan")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, scale=scale)
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
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                split = getattr(p, "muon_split", 1)
                rows, cols = g.shape[0] // split, g.shape[1]
                u = newton_schulz(g.reshape(split, rows, cols), group["ns_steps"])
                u = u.reshape_as(p) * self._scale(rows, cols, group["scale"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])
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


def build_optimizer(model, name, lr, weight_decay, betas=(0.9, 0.95), fused=None):
    """name: 'adamw' or 'muon'.

    Muon takes the 2-D weights the model lists in `model.muon_params()`
    (hidden matrices inside the blocks). Embeddings, output head, norms, gates
    and biases stay on AdamW, as recommended for Muon.
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
        elif p.dim() >= 2 and "gate" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    kw = {}
    if fused is None:
        fused = next(model.parameters()).is_cuda
    if fused:
        kw["fused"] = True
    adamw = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=lr, betas=betas, eps=1e-8, **kw)
    muon = Muon(muon_p, lr=lr, weight_decay=weight_decay) if muon_p else None
    return CombinedOptimizer([adamw, muon])
