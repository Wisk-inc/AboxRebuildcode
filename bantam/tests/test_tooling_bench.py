"""experiments/bench_1b.py on a tiny model: labels, Chinchilla tokens, build time."""
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import bench_1b  # noqa: E402


def test_bench_reports_params_and_build_time(tmp_path, monkeypatch):
    data = tmp_path / "train.bin"
    np.random.default_rng(0).integers(0, 256, 4096).astype(np.uint16).tofile(data)
    real = bench_1b.build_optimizer
    real(bench_1b.ZoomGPT(256, 64, 1, 2), "muon", 1e-3, 0.1, state_bits=8)   # one-off lazy init

    def slow_optimizer(*a, **kw):                    # make each step clearly slower than the build
        opt = real(*a, **kw)
        step = opt.step

        def slow_step(*sa, **skw):
            time.sleep(0.5)
            return step(*sa, **skw)

        opt.step = slow_step
        return opt

    monkeypatch.setattr(bench_1b, "build_optimizer", slow_optimizer)
    out = bench_1b.main(["--steps", "2", "--threads", "1", "--d", "64", "--layers", "2", "--heads", "2",
                         "--vocab", "256", "--seq", "16", "--data", str(data),
                         "--out", str(tmp_path / "bench.json")])
    assert out["chinchilla_tokens"] == 20 * out["params"]
    assert out["build_seconds"] < 0.5                  # the steps (>= 1.5 s) are not included
    assert (tmp_path / "bench.json").exists()
    with pytest.raises(SystemExit):
        bench_1b.main(["--steps", "0", "--data", str(data), "--out", str(tmp_path / "x.json")])


def test_default_size_is_labelled_correctly():
    import torch
    from bantamtrain.models import ZoomGPT
    with torch.device("meta"):
        n = sum(p.numel() for p in ZoomGPT(32000, 2048, 17, 16).parameters())
    assert 0.9e9 < n < 0.95e9 and "0.92B" in bench_1b.__doc__
