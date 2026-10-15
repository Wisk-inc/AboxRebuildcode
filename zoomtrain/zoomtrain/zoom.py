"""Zoom Training: coarse-to-fine resolution annealing.

The transformer body is the expensive part of training and its cost scales with
the number of positions it processes. Zoom Training keeps *every* input
element and *every* target element, but early in training it merges each group
of K neighbouring elements (K tokens in text, a 2x2 block of patches on a
weather grid) into one position, runs the body on N/K positions, and unmerges
the result back to one prediction per original element. Over training K is
annealed 4 -> 2 -> 1, ending with an ordinary model.

Two small modules make the merge reversible enough to transfer:

* SlotMerge   - gated sum: z = sum_j g_j * x_j. Each slot j inside a group has
                its own per-channel gate g_j, so the order/layout of elements
                inside the group survives the merge (a plain mean would not).
* SlotUnmerge - y_j = h * u_j: one gated copy of the coarse hidden state per
                slot, so each original element gets its own prediction
                instead of a single bag-of-tokens distribution.

Both cost K*d parameters per zoom level and are dropped at K=1.
"""
import math

import torch
import torch.nn as nn


class SlotMerge(nn.Module):
    """(B, N*K, d) grouped as (B, N, K, d) -> (B, N, d) by a slot-gated sum."""

    def __init__(self, k, d, gated=True):
        super().__init__()
        self.k = k
        init = torch.full((k, d), 1.0 / k)
        if gated:
            self.gate = nn.Parameter(init)
        else:
            self.register_buffer("gate", init)

    def forward(self, x):  # x: (..., K, d)
        return (x * self.gate).sum(-2)


class SlotUnmerge(nn.Module):
    """(B, N, d) -> (B, N, K, d): one gated copy per slot."""

    def __init__(self, k, d, gated=True):
        super().__init__()
        self.k = k
        init = torch.ones(k, d)
        if gated:
            self.gate = nn.Parameter(init)
        else:
            self.register_buffer("gate", init)

    def forward(self, h):  # h: (..., d)
        return h.unsqueeze(-2) * self.gate


class ZoomSchedule:
    """Piecewise-constant zoom level as a function of compute spent.

    stages: list of (k, fraction_of_budget); fractions must sum to 1.
    Example: [(4, .35), (2, .25), (1, .40)].
    """

    def __init__(self, stages):
        total = sum(f for _, f in stages)
        assert abs(total - 1.0) < 1e-6, f"stage fractions sum to {total}"
        self.stages = stages

    def k_at(self, frac):
        acc = 0.0
        for k, f in self.stages:
            acc += f
            if frac < acc - 1e-12:
                return k
        return self.stages[-1][0]

    @property
    def ks(self):
        return sorted({k for k, _ in self.stages if k > 1})

    @staticmethod
    def parse(spec):
        """'4:0.35,2:0.25,1:0.4' -> ZoomSchedule; '1' -> baseline."""
        if spec in ("1", "baseline", "", None):
            return ZoomSchedule([(1, 1.0)])
        stages = []
        for part in spec.split(","):
            k, f = part.split(":")
            stages.append((int(k), float(f)))
        return ZoomSchedule(stages)

    def __repr__(self):
        return ",".join(f"{k}:{f:g}" for k, f in self.stages)


def lr_at(frac, peak, warmup=0.05, final=0.1):
    """Linear warmup then cosine decay to final*peak, indexed by compute fraction."""
    if frac < warmup:
        return peak * frac / warmup
    p = min(1.0, (frac - warmup) / (1 - warmup))
    return peak * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * p)))
