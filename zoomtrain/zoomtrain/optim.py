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
import torch.nn as nn
from torch.optim import Optimizer

from .lowmem import QState, write_param

NS_COEFFS = (3.4445, -4.7750, 2.0315)
# Polar Express (Amsel et al., 2025): per-step minimax coefficients, same matmul
# cost as the quintic iteration; a, b, c of every tuple but the last are divided
# by 1.01, 1.01^3, 1.01^5 for safety, and the input is normalized with a 1.01 margin.
_PE = [(8.28721201814563, -23.595886519098837, 17.300387312530933),
       (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
       (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
       (3.3184196573706015, -2.488488024314874, 0.51004894012372),
       (2.300652019954817, -1.6689039845747493, 0.4188073119525673)]
POLAR_COEFFS = [(a / 1.01, b / 1.01 ** 3, c / 1.01 ** 5) for a, b, c in _PE[:-1]] + [_PE[-1]]


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7, variant="quintic"):
    """Approximate U V^T of G = U S V^T (batched over leading dims)."""
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    if variant == "polar":
        coeffs = (POLAR_COEFFS + [POLAR_COEFFS[-1]] * steps)[:steps]
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)
    else:
        coeffs = [NS_COEFFS] * steps
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in coeffs:
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
                 weight_decay=0.0, scale="moonlight", state_bits=32, stochastic=True,
                 ns="quintic"):
        assert scale in ("moonlight", "jordan") and state_bits in (32, 8)
        assert ns in ("quintic", "polar")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, scale=scale, state_bits=state_bits,
                        stochastic=stochastic, ns=ns)
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
                u = newton_schulz(g.reshape(split, rows, cols), group["ns_steps"],
                                  variant=group["ns"])
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


def default_muon_params(model, include_lora=False):
    """Muon candidates for an arbitrary model: trainable 2-D weights of nn.Linear
    layers that are not embeddings or the output head (and not LoRA adapters
    unless include_lora). Works for HuggingFace causal LMs."""
    skip = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        out = get_out()
        if out is not None and hasattr(out, "weight"):
            skip.add(id(out.weight))
    params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.requires_grad and m.weight.dim() == 2 \
                and id(m.weight) not in skip and "lora_" not in name \
                and not name.split(".")[-1] in ("lm_head", "head", "output", "score"):
            params.append(m.weight)
        elif include_lora and type(m).__name__ == "LoRALinear":
            params += [m.lora_A, m.lora_B]
    return params


def build_optimizer(model, name, lr, weight_decay, betas=(0.9, 0.95), fused=None,
                    state_bits=32, ns="quintic", muon_lora=False):
    """name: 'adamw' or 'muon'.

    Muon takes the 2-D weights the model lists in `model.muon_params()`, or,
    for any other model (e.g. a HuggingFace LM), the hidden nn.Linear weights
    found by `default_muon_params`. Embeddings, output head, norms, gates,
    biases and (by default) LoRA adapters stay on AdamW.

    state_bits=8, or bf16 parameters, switch AdamW to AdamWLowMem (8-bit
    moments / stochastic-rounding writes); otherwise torch's fused AdamW is used.
    """
    muon_ids = set()
    if name == "muon":
        cands = model.muon_params() if hasattr(model, "muon_params") \
            else default_muon_params(model, include_lora=muon_lora)
        muon_ids = {id(p) for p in cands if p.requires_grad}
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    bf16_params = any(p.dtype == torch.bfloat16 for p in trainable)
    if state_bits == 8 or bf16_params:
        adamw = AdamWLowMem(groups, lr=lr, betas=betas, eps=1e-8, state_bits=state_bits)
    else:
        kw = {}
        if fused is None:
            fused = next(model.parameters()).is_cuda
        if fused:
            kw["fused"] = True
        adamw = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **kw)
    muon = (Muon(muon_p, lr=lr, weight_decay=weight_decay, state_bits=state_bits, ns=ns)
            if muon_p else None)
    return CombinedOptimizer([adamw, muon])
