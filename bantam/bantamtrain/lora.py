"""LoRA / DoRA / QLoRA adapters for any PyTorch model, PEFT-compatible on disk.

    from bantamtrain.lora import apply_lora, save_peft_adapter, load_peft_adapter, merge_lora
    apply_lora(model, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"))
    # or targets="all-linear" (every Linear / Conv1D / QLinear but the output head)
    ...train (only adapter weights have requires_grad)...
    save_peft_adapter(model, "adapter/", "HuggingFaceTB/SmolLM2-135M")   # PEFT format, a few MB
    merge_lora(model)                                     # fold W += B A * scale into the base

Each targeted nn.Linear, transformers Conv1D (GPT-2) or QLinear from bantamtrain.qweights
(QLoRA: frozen quantized base + trainable adapters) becomes

    y = base(x) + (dropout(x) @ A^T @ B^T) * scale,  scale = alpha / r  (or alpha/sqrt(r))

A ~ Kaiming-uniform, B = 0, so the model starts exactly at the base model. With dora=True
(DoRA, Liu et al. 2024) the adapted weight is m * (W + scale*BA) / ||W + scale*BA||, row
norms detached from the graph as in the paper and in PEFT, with a trainable magnitude m
initialised to ||W||. Adapters are created in fp32 on the base layer's device (an fp32
copy also over a bf16 base; forward casts them to the activation dtype).
save_peft_adapter / load_peft_adapter write and read PEFT's adapter_config.json +
adapter_model.safetensors, so PeftModel.from_pretrained (and vLLM, TGI) load our adapters
and PEFT-trained adapters load here. Trainable parameters per layer: r * (d_in + d_out),
plus d_out for DoRA.
"""
import json
import math
import os
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qweights import CHUNK, QBase, QLinear, dequantize_rows, is_conv1d

ADAPTER_PARAMS = ("lora_A", "lora_B", "lora_magnitude_vector")


def _linear_like(m):
    return isinstance(m, (nn.Linear, QLinear)) or is_conv1d(m)


def _dims(m):
    """(in_features, out_features) of a Linear / Conv1D / quantized linear."""
    if is_conv1d(m):
        return tuple(m.weight.shape)
    return m.in_features, m.out_features


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0, rslora=False, dora=False):
        super().__init__()
        assert _linear_like(base), type(base)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.rslora, self.dora = r, alpha, rslora, dora
        self.scale = alpha / (math.sqrt(r) if rslora else r)
        self.fan_in_fan_out = is_conv1d(base)
        d_in, d_out = _dims(base)
        dev = base.q.device if isinstance(base, QBase) else base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        if dora:
            self.lora_magnitude_vector = nn.Parameter(self.weight_norm(delta=False))

    @property
    def in_features(self):
        return _dims(self.base)[0]

    @property
    def out_features(self):
        return _dims(self.base)[1]

    def _weight_rows(self, a, b):
        """Rows a:b of the base weight as (out, in) fp32."""
        m = self.base
        if isinstance(m, QBase):
            return dequantize_rows(m.q[a:b], m.s[a:b], m.pad, m.bits, m.block, torch.float32, m.cb)
        if self.fan_in_fan_out:
            return m.weight[:, a:b].t().float()
        return m.weight[a:b].float()

    @torch.no_grad()
    def weight_norm(self, delta=True):
        """||W + scale*B A|| per output row (fp32), built in row chunks."""
        d_in, d_out = _dims(self.base)
        out = torch.empty(d_out, device=self.lora_A.device, dtype=torch.float32)
        A = self.lora_A.float()
        step = max(1, CHUNK // max(d_in, 1))
        for a in range(0, d_out, step):
            b = min(a + step, d_out)
            w = self._weight_rows(a, b)
            if delta:
                w = w + (self.lora_B[a:b].float() @ A) * self.scale
            out[a:b] = w.norm(dim=1)
        return out

    def _base_matmul(self, x):
        """x @ W^T without the bias."""
        m = self.base
        if isinstance(m, QBase):
            return m.matmul(x)
        if self.fan_in_fan_out:
            return x @ m.weight.to(x.dtype)
        return F.linear(x, m.weight)

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        xd = self.dropout(x)
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        lora = F.linear(F.linear(xd, a), b) * self.scale
        if not self.dora:
            return y + lora
        mag = (self.lora_magnitude_vector / self.weight_norm()).to(y.dtype)
        if isinstance(self.dropout, nn.Identity) or not self.training:
            bias = getattr(self.base, "bias", None)
            base_out = y if bias is None else y - bias.to(y.dtype)
        else:
            base_out = self._base_matmul(xd)
        return y + (mag - 1) * base_out + mag * lora

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into the base weight (float base only) and disable it."""
        if self.merged:
            return
        if isinstance(self.base, QBase):
            raise RuntimeError("merge into a quantized base: dequantize the base first")
        w = self.base.weight
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale      # (out, in)
        if self.dora:
            wf = (w.t() if self.fan_in_fan_out else w).float() + delta.to(w.device)
            wf *= (self.lora_magnitude_vector.float().to(w.device) / wf.norm(dim=1))[:, None]
            w.copy_(wf.t() if self.fan_in_fan_out else wf)
        else:
            w.add_((delta.t() if self.fan_in_fan_out else delta).to(w.device, w.dtype))
        self.merged = True


def _match(name, targets):
    leaf = name.split(".")[-1]
    return any(t == leaf or t == name or name.endswith("." + t) for t in targets)


def _selector(model, targets, exclude):
    """name, module -> bool for apply_lora's targets / exclude (names, a regex, or "all-linear")."""
    if isinstance(exclude, str):
        exc = lambda n: re.fullmatch(exclude, n) is not None  # noqa: E731
    else:
        excl = tuple(exclude or ())
        exc = lambda n: bool(excl) and _match(n, excl)  # noqa: E731
    if targets == "all-linear":
        fn = getattr(model, "get_output_embeddings", None)
        try:
            head = fn() if callable(fn) else None
        except (NotImplementedError, AttributeError):
            head = None
        return lambda n, m: not exc(n) and m is not head and \
            (head is not None or n.split(".")[-1] not in ("lm_head", "output"))
    if isinstance(targets, str):
        return lambda n, m: not exc(n) and re.fullmatch(targets, n) is not None
    targets = tuple(targets)
    return lambda n, m: not exc(n) and _match(n, targets)


def _wrap(model, select, make, min_numel=0):
    wrapped = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            continue
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if _linear_like(child) and select(full, child):
                d_in, d_out = _dims(child)
                if d_in * d_out >= min_numel:
                    setattr(mod, cname, make(full, child))
                    wrapped += 1
    return wrapped


def _report(model, wrapped, r, alpha, dora):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"wrapped": wrapped, "r": r, "alpha": alpha, "dora": dora, "trainable_params": trainable,
            "total_params": total, "trainable_frac": trainable / max(total, 1)}


def apply_lora(model, r=8, alpha=16, dropout=0.0, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
               rslora=False, min_numel=0, freeze_rest=True, dora=False, exclude=None):
    """Wrap every targeted Linear / Conv1D / QLinear in LoRALinear; freeze everything else.
    targets: module names (leaf or dotted suffix), a regex matched against the full name,
    or "all-linear" (all of them except the output head). exclude: names or a regex.

    Returns a report {wrapped, trainable_params, total_params, ...}.
    """
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad_(False)
    select = _selector(model, targets, exclude)
    wrapped = _wrap(model, select, lambda n, c: LoRALinear(c, r, alpha, dropout, rslora, dora),
                    min_numel)
    return _report(model, wrapped, r, alpha, dora)


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if n.split(".")[-1] in ADAPTER_PARAMS]


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.split(".")[-1] in ADAPTER_PARAMS}


def load_lora_state_dict(model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k.split(".")[-1] in ADAPTER_PARAMS]
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


# ----------------------------------------------------------------------------- PEFT format
PEFT_PREFIX = "base_model.model."


def save_peft_adapter(model, out_dir, base_model_name_or_path=None):
    """Write adapter_config.json + adapter_model.safetensors in PEFT's LoRA format
    (PeftModel.from_pretrained(base, out_dir) reproduces this model's outputs)."""
    from safetensors.torch import save_file
    mods = lora_modules(model)
    if not mods:
        raise ValueError("the model has no LoRA adapters")
    flags = {(m.rslora, m.dora) for _, m in mods}
    if len(flags) > 1:
        raise ValueError("PEFT needs one use_rslora / use_dora setting for all adapters")
    rslora, dora = flags.pop()
    r = Counter(m.r for _, m in mods).most_common(1)[0][0]
    alpha = Counter(m.alpha for _, m in mods).most_common(1)[0][0]
    tensors = {}
    for n, m in mods:
        tensors[f"{PEFT_PREFIX}{n}.lora_A.weight"] = m.lora_A.detach().cpu().contiguous()
        tensors[f"{PEFT_PREFIX}{n}.lora_B.weight"] = m.lora_B.detach().cpu().contiguous()
        if m.dora:
            tensors[f"{PEFT_PREFIX}{n}.lora_magnitude_vector"] = \
                m.lora_magnitude_vector.detach().cpu().contiguous()
    names = [n for n, _ in mods]
    leaves = {n.split(".")[-1] for n in names}
    # leaf names when they select exactly the wrapped modules (PEFT matches any module type by
    # name), else full module names
    same = {n for n, _ in model.named_modules() if n.split(".")[-1] in leaves
            and not any(n.startswith(w + ".") for w in names)}
    target_modules = sorted(leaves) if same == set(names) else names
    p = {m.dropout.p for _, m in mods if isinstance(m.dropout, nn.Dropout)}
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model_name_or_path,
           "r": r, "lora_alpha": alpha, "lora_dropout": max(p) if p else 0.0,
           "target_modules": target_modules, "use_rslora": rslora, "use_dora": dora,
           "fan_in_fan_out": any(m.fan_in_fan_out for _, m in mods), "bias": "none",
           "rank_pattern": {n: m.r for n, m in mods if m.r != r},
           "alpha_pattern": {n: m.alpha for n, m in mods if m.alpha != alpha},
           "modules_to_save": None, "init_lora_weights": True, "inference_mode": True,
           "layers_to_transform": None, "layers_pattern": None, "revision": None}
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def _pattern(patterns, name, default):
    """PEFT's rank/alpha_pattern lookup: first key matching the end of the module name."""
    for k, v in (patterns or {}).items():
        if re.match(rf"(.*\.)?({k})$", name):
            return v
    return default


def load_peft_adapter(model, adapter_dir, freeze_rest=True):
    """Apply LoRA as described by a PEFT adapter_config.json and load its weights into a
    model without adapters (a float or quantized base). The adapters stay trainable (to
    resume training); the rest is frozen with freeze_rest. Returns apply_lora's report."""
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"not a LoRA adapter: peft_type={cfg.get('peft_type')}")
    for k in ("layers_to_transform", "modules_to_save", "trainable_token_indices", "layer_replication"):
        if cfg.get(k):
            raise NotImplementedError(f"PEFT option {k}={cfg[k]!r} is not supported")
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias"):
        raise NotImplementedError("adapters with trained biases are not supported")
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(path):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu",
                        weights_only=True)
    if any(lora_modules(model)):
        raise ValueError("the model already has LoRA adapters")
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad_(False)
    r0, a0 = cfg["r"], cfg["lora_alpha"]
    targets = cfg.get("target_modules") or ()
    select = _selector(model, targets if isinstance(targets, str) else tuple(targets),
                       cfg.get("exclude_modules"))

    def make(name, child):
        r = _pattern(cfg.get("rank_pattern"), name, r0)
        alpha = _pattern(cfg.get("alpha_pattern"), name, a0)
        return LoRALinear(child, r, alpha, cfg.get("lora_dropout", 0.0) or 0.0,
                          bool(cfg.get("use_rslora")), bool(cfg.get("use_dora")))
    wrapped = _wrap(model, select, make)
    params = dict(model.named_parameters())
    seen = set()
    for key, t in sd.items():
        k = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
        k = re.sub(r"\.(lora_A|lora_B)(\.default)?\.weight$", r".\1", k)
        k = re.sub(r"\.lora_magnitude_vector(\.default)?(\.weight)?$", ".lora_magnitude_vector", k)
        p = params.get(k)
        if p is None or p.shape != t.shape:
            raise ValueError(f"adapter tensor {key} {tuple(t.shape)} does not fit the model"
                             f" ({None if p is None else tuple(p.shape)})")
        with torch.no_grad():
            p.copy_(t)
        seen.add(k)
    expected = {n for n, _ in model.named_parameters() if n.split(".")[-1] in ADAPTER_PARAMS}
    if expected - seen:
        raise ValueError(f"adapter weights missing: {sorted(expected - seen)[:6]}")
    return _report(model, wrapped, r0, a0, bool(cfg.get("use_dora")))
