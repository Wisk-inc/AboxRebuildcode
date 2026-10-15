import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoomtrain.device import (DutyCycleLimiter, Prefetcher, Runtime,  # noqa: E402
                              find_batch_size, maybe_compile, setup_runtime)
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.optim import build_optimizer  # noqa: E402


def test_setup_runtime_cpu_is_fp32_and_ftz():
    rt = setup_runtime("cpu", "auto", False, 2, 0)
    assert rt.device.type == "cpu" and rt.amp_dtype is None and rt.scaler is None
    # a denormal times 1 must flush to zero once FTZ is on
    x = torch.tensor([1e-40], dtype=torch.float32)
    assert (x * 1.0).item() == 0.0
    assert isinstance(rt, Runtime)


def test_runtime_backward_steps_combined_optimizer():
    rt = setup_runtime("cpu", "fp32", False, 2, 0)
    m = ZoomGPT(64, 16, 1, 2, zoom_ks=[2])
    opt = build_optimizer(m, "muon", 1e-2, 0.0, fused=False)
    x = torch.randint(0, 64, (2, 10))
    before = [p.detach().clone() for p in m.parameters()]
    with rt.autocast():
        loss = m(x[:, :8], x[:, 1:9], k=1)
    opt.zero_grad()
    rt.backward(loss, opt, m.parameters())
    changed = [not torch.equal(a, b) for a, b in zip(before, m.parameters())]
    assert any(changed)


def test_duty_cycle_limiter_idles_proportionally():
    lim = DutyCycleLimiter(0.5)
    t0 = time.perf_counter()
    lim(0.05)                      # a 50 ms step at 50 % duty -> ~50 ms idle
    dt = time.perf_counter() - t0
    assert 0.04 <= dt <= 0.2
    t0 = time.perf_counter()
    DutyCycleLimiter(1.0)(0.05)    # full duty: no idle
    assert time.perf_counter() - t0 < 0.01


def test_find_batch_size_cpu_returns_start():
    assert find_batch_size(lambda b: None, start=8, device=torch.device("cpu")) == 8


def test_prefetcher_delivers_batches_in_background():
    counter = {"n": 0}

    def make():
        counter["n"] += 1
        return counter["n"]

    pf = Prefetcher(make, depth=2)
    got = [pf.next() for _ in range(3)]
    pf.close()
    assert got == [1, 2, 3]


def test_maybe_compile_noop_without_flag():
    rt = setup_runtime("cpu", "fp32", False, 2, 0)
    m = ZoomGPT(64, 16, 1, 2)
    assert maybe_compile(m, rt) is m
