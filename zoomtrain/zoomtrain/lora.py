"""LoRA / QLoRA adapters for any PyTorch model.

    from zoomtrain.lora import apply_lora, lora_state_dict, merge_lora
    apply_lora(model, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"))
    ...train (only adapter weights have requires_grad)...
    torch.save(lora_state_dict(model), "adapter.pt")     # a few MB
    merge_lora(model)                                     # fold W += B A * scale into the base

Each targeted nn.Linear (or an int8/int4 QLinear from zoomtrain.qweights, which
gives QLoRA: frozen quantized base + trainable adapters) becomes

    y = base(x) + (dropout(x) @ A^T @ B^T) * scale,  scale = alpha / r  (or alpha/sqrt(r))

A ~ Kaiming-uniform, B = 0, so the model starts exactly at the base model.
Trainable parameters per layer: r * (d_in + d_out).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qweights import QLinear


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0, rslora=False):
        super().__init__()
        assert isinstance(base, (nn.Linear, QLinear))
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scale = alpha / (math.sqrt(r) if rslora else r)
        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return y + F.linear(F.linear(self.dropout(x), a), b) * self.scale

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into the base weight (float base only) and disable it."""
        if self.merged:
            return
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
        if isinstance(self.base, QLinear):
            raise RuntimeError("merge into a quantized base: dequantize the base first")
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.merged = True


def _match(name, targets):
    leaf = name.split(".")[-1]
    return any(t == leaf or t == name or name.endswith("." + t) for t in targets)


def apply_lora(model, r=8, alpha=16, dropout=0.0, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
               rslora=False, min_numel=0, freeze_rest=True):
    """Wrap every targeted Linear/QLinear in LoRALinear; freeze everything else.

    Returns a report {wrapped, trainable_params, total_params}.
    """
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad_(False)
    wrapped = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, (nn.Linear, QLinear)) and _match(full, targets) \
                    and child.in_features * child.out_features >= min_numel:
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout, rslora))
                wrapped += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"wrapped": wrapped, "r": r, "alpha": alpha, "trainable_params": trainable,
            "total_params": total, "trainable_frac": trainable / max(total, 1)}


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "lora_" in k]
    assert not bad and not unexpected, (bad, unexpected)


def merge_lora(model):
    """Merge every adapter into its base weight; returns the number merged."""
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
            n += 1
    return n


def unwrap_lora(model):
    """Replace each (merged) LoRALinear by its base module."""
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, cname, child.base)
    return model
