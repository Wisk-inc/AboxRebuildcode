import os
import sys
import time
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.device import (DutyCycleLimiter, Prefetcher, Runtime,  # noqa: E402
                              auto_precision, find_batch_size, maybe_compile, setup_runtime)
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import CombinedOptimizer, Muon, build_optimizer  # noqa: E402


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


# ---- fp16 / GradScaler path (a CPU GradScaler runs the same unscale/step logic as CUDA's)
def _scaled_backward(scale=256.0):
    torch.manual_seed(0)
    m = ZoomGPT(64, 16, 1, 2)
    opt = build_optimizer(m, "muon", 1e-2, 0.1, fused=False)
    rt = Runtime(torch.device("cpu"), None, torch.amp.GradScaler("cpu", init_scale=scale), False)
    x = torch.randint(0, 64, (2, 9))
    rt.backward(m(x[:, :8], x[:, 1:9], k=1), opt, m.parameters(), step=False)
    return m, opt, rt


def test_scaler_overflow_in_one_suboptimizer_skips_the_whole_step():
    for target, bad in (("blocks.0.qkv.weight", float("inf")),    # Muon-owned grad
                        ("ln_f.weight", float("nan"))):            # AdamW-owned grad
        m, opt, rt = _scaled_backward()
        dict(m.named_parameters())[target].grad.view(-1)[0] = bad
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        rt.step(opt, m.parameters())
        changed = [n for n, p in m.named_parameters() if not torch.equal(before[n], p)]
        assert changed == [], (target, changed)
        assert all(not o.state for o in opt.optimizers)            # no AdamW step counted either
        assert rt.scaler.get_scale() == 128.0
        # the next, finite step goes through for every sub-optimizer
        opt.zero_grad()
        x = torch.randint(0, 64, (2, 9))
        rt.backward(m(x[:, :8], x[:, 1:9], k=1), opt, m.parameters())
        assert all(o.state for o in opt.optimizers)
        assert all(torch.isfinite(p).all() for p in m.parameters())


def test_scaler_with_an_empty_suboptimizer_steps():
    m = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 1, bias=False))
    adamw = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))])   # never gets a grad
    muon = Muon([m[0].weight, m[1].weight], lr=0.1)
    opt = CombinedOptimizer([adamw, muon])
    rt = Runtime(torch.device("cpu"), None, torch.amp.GradScaler("cpu"), False)
    before = m[0].weight.detach().clone()
    rt.backward(m(torch.randn(4, 8)).square().mean(), opt, m.parameters())
    assert not torch.equal(before, m[0].weight)


def test_scaler_step_matches_unscaled_step():
    m1, opt1, rt = _scaled_backward(scale=1024.0)
    rt.step(opt1, m1.parameters())
    torch.manual_seed(0)
    m2 = ZoomGPT(64, 16, 1, 2)
    opt2 = build_optimizer(m2, "muon", 1e-2, 0.1, fused=False)
    x = torch.randint(0, 64, (2, 9))
    Runtime(torch.device("cpu"), None, None, False).backward(m2(x[:, :8], x[:, 1:9], k=1), opt2,
                                                             m2.parameters())
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_auto_precision_is_fp16_before_ampere(monkeypatch):
    cuda = torch.device("cuda", 0)
    for cap, want in (((7, 0), "fp16"), ((7, 5), "fp16"), ((8, 0), "bf16"), ((9, 0), "bf16")):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None, c=cap: c)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # emulated bf16
        assert auto_precision(cuda) == want
    assert auto_precision(torch.device("cpu")) == "fp32"
    # setup_runtime picks the scaler path on a T4/V100
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    prec, tf32 = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # GradScaler('cuda') without a GPU warns
            rt = setup_runtime("cuda", "auto", False, None, 0)
    finally:
        torch.set_float32_matmul_precision(prec)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = True
    assert rt.amp_dtype is torch.float16 and rt.scaler is not None


def _fake_cuda(monkeypatch, peak_of, total=100):
    """find_batch_size against a fake GPU: try_step(b) records a peak of peak_of(b) bytes."""
    state = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda d: type("P", (), {"total_memory": total})())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: state["peak"])

    def try_step(b):
        p = peak_of(b)
        if p is None:
            raise torch.OutOfMemoryError("fake OOM")
        state["peak"] = p
    return try_step


def test_find_batch_size_none_when_start_does_not_fit(monkeypatch):
    cuda = torch.device("cuda", 0)
    oom_from_4 = _fake_cuda(monkeypatch, lambda b: None if b >= 4 else 10 * b)
    assert find_batch_size(oom_from_4, start=8, device=cuda) is None
    assert find_batch_size(oom_from_4, start=1, device=cuda) == 2
    over_headroom = _fake_cuda(monkeypatch, lambda b: 95)       # 95 % > 85 % headroom
    assert find_batch_size(over_headroom, start=8, device=cuda) is None
    grows = _fake_cuda(monkeypatch, lambda b: 10 * b)           # 80 fits, 160 does not
    assert find_batch_size(grows, start=1, limit=64, device=cuda) == 8
    assert find_batch_size(grows, start=1, limit=4, device=cuda) == 4


def test_duty_cycle_rejects_nonpositive_targets():
    for bad in (0.0, -0.5, float("nan")):
        with pytest.raises(ValueError):
            DutyCycleLimiter(bad)


def test_prefetcher_propagates_errors():
    n = {"i": 0}

    def make():
        n["i"] += 1
        if n["i"] > 2:
            raise RuntimeError("data source failed")
        return n["i"]

    pf = Prefetcher(make, depth=2)
    assert [pf.next(), pf.next()] == [1, 2]
    for _ in range(2):                       # raised now and on every later call, never a hang
        with pytest.raises(RuntimeError, match="data source failed"):
            pf.next()
    pf.close()


def test_prefetcher_close_stops_a_blocked_worker():
    pf = Prefetcher(lambda: 1, depth=1)
    time.sleep(0.1)                          # worker now blocks on the full queue
    pf.close()
    assert not pf.t.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        pf.next()
