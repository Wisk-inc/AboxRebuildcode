"""Wave-2 core review: bf16 stochastic rounding draws new noise every step."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.optim import AdamWLowMem, Muon  # noqa: E402


@pytest.mark.parametrize("cls,kw", [(AdamWLowMem, {"lr": 4e-4, "state_bits": 32}),
                                    (Muon, {"lr": 2e-3})])
def test_bf16_rounding_noise_is_fresh_every_step(cls, kw):
    """A constant gradient moves each weight by ~0.1 bf16 ulp per step. With noise that is
    new every step each weight tracks the fp32 trajectory to a few ulps; noise that
    repeated across steps (a counter seed without the step) would round the same weights
    the same way every time: 10 % of them move every step, the rest never."""
    torch.manual_seed(0)
    g = torch.randn(64, 64)
    pb = torch.nn.Parameter(torch.ones(64, 64, dtype=torch.bfloat16))
    pf = torch.nn.Parameter(torch.ones(64, 64))
    ob, of = cls([pb], **kw), cls([pf], **kw)
    for _ in range(200):
        pb.grad, pf.grad = g.to(torch.bfloat16), g.clone()
        ob.step()
        of.step()
    drift = (1 - pf.detach()).abs().mean().item()             # fp32 trajectory
    dev = pb.detach().float() - pf.detach()
    assert dev.abs().mean().item() < 0.35 * drift             # ~0.2 fresh, ~1.8 repeated
    assert abs(dev.mean().item()) < 0.05 * drift              # unbiased
