"""bantam weather on a tiny synthetic ERA5-like dataset: splits, metrics, FLOP accounting,
checkpoints (bit-exact resume), --save and forecasts. In-process, CPU, a few seconds each."""
import json
import os
import sys
import time

import numpy as np
import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bantamtrain.cli.weather as W  # noqa: E402
from bantamtrain.data import gridpack  # noqa: E402
from bantamtrain.models import ZoomViT  # noqa: E402
from bantamtrain.optim import build_optimizer  # noqa: E402

CH = ("z500", "t850", "t2m", "u10", "v10", "msl")
BASE = (54000.0, 275.0, 279.0, 0.0, 0.0, 101000.0)
AMP = (3000.0, 15.0, 20.0, 5.0, 4.0, 1300.0)
CPU = torch.device("cpu")
SMALL = ["--d", "16", "--layers", "1", "--heads", "2", "--batch", "4", "--threads", "1",
         "--evals", "2"]


def hours6(start, n):
    return np.datetime64(start, "h") + np.arange(n) * np.timedelta64(6, "h")


def make_era5(path, times=None, test_year=2018, seed=0):
    """meta.json + fields.npy (T, 6, 32, 64) + static.npy, written like prepare-era5 does."""
    os.makedirs(path, exist_ok=True)
    times = hours6("2017-11-01", 320) if times is None else times
    rng = np.random.default_rng(seed)
    T, H, W_ = len(times), 32, 64
    lat, lon = np.linspace(-87.1875, 87.1875, H), np.arange(W_) * 5.625
    t = np.arange(T, dtype=np.float32)[:, None, None]
    la, lo = np.deg2rad(lat)[None, :, None], np.deg2rad(lon)[None, None, :]
    wave = np.cos(la) * np.sin(3 * lo - 0.1 * t) + 0.3 * np.sin(2 * la) * np.cos(0.26 * t)
    x = np.stack([BASE[c] + AMP[c] * (wave + 0.05 * rng.standard_normal((T, H, W_)))
                  for c in range(len(CH))], 1).astype(np.float32)
    n = int((times < np.datetime64(f"{test_year}-01-01", "h")).sum())
    d = x[1:n] - x[:n - 1]
    meta = {"channels": list(CH), "times": [str(s) for s in times], "lat": lat.tolist(),
            "lon": lon.tolist(), "mean": x[:n].mean(axis=(0, 2, 3)).tolist(),
            "std": x[:n].std(axis=(0, 2, 3)).tolist(), "diff_std": d.std(axis=(0, 2, 3)).tolist(),
            "n_train": n, "n_total": T}
    np.save(os.path.join(path, "fields.npy"), x)
    np.save(os.path.join(path, "static.npy"), rng.standard_normal((2, H, W_)).astype(np.float32))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f)
    return x, meta


@pytest.fixture(scope="module")
def era5(tmp_path_factory):
    """Nov 2017 - Jan 2018, 6-hourly: 244 train times (2017), 76 test times (2018)."""
    path = str(tmp_path_factory.mktemp("era5"))
    make_era5(path)
    return path


@pytest.fixture(scope="module")
def era5_3y(tmp_path_factory):
    """Three short 6-hourly segments: 120 times in 2016, 100 in 2017, 80 in 2018."""
    path = str(tmp_path_factory.mktemp("era5_3y"))
    times = np.concatenate([hours6("2016-11-01", 120), hours6("2017-03-01", 100),
                            hours6("2018-01-01", 80)])
    make_era5(path, times)
    return path


@pytest.fixture(autouse=True)
def _restore_torch_state():
    n, rng = torch.get_num_threads(), torch.get_rng_state()
    yield
    torch.set_num_threads(n)
    torch.set_rng_state(rng)


def run(data, out, name, *extra, budget="3e9"):
    W.main(["--data", data, "--out", str(out), "--name", name, "--budget", budget, *SMALL,
            *extra])


def train(data, out, name, *extra, budget="3e9"):
    run(data, out, name, *extra, budget=budget)
    with open(os.path.join(str(out), f"{name}.json")) as f:
        return json.load(f)


# -- argument validation and FLOP accounting


@pytest.mark.parametrize("mode", ["merge", "lowres"])
def test_non_square_zoom_rejected(era5, tmp_path, mode):
    with pytest.raises(SystemExit) as e:
        train(era5, tmp_path, "k2", "--zoom", "2:0.5,1:0.5", "--coarse_mode", mode)
    assert e.value.code == 2


def test_evals_zero_and_ic_coverage(era5, tmp_path):
    log = train(era5, tmp_path, "e0", "--evals", "0")
    assert len(log["evals"]) == 1 and log["evals"][0]["frac"] == 0.0
    assert "rmse_z500_72h" in log["final"]
    ics = log["split"]["test_ics"]                     # default: every 12th time, 00 UTC only
    assert ics["hours"] == [0] and ics["first"] == "2018-01-01T00"
    log = train(era5, tmp_path, "ic14", "--ic_stride", "14", "--max_ics", "0")
    assert log["split"]["test_ics"]["hours"] == [0, 12]


def test_batch_scale_k_and_flops_in_lowres(era5, tmp_path, monkeypatch):
    seen = []
    orig = ZoomViT.forward_lowres

    def rec(self, x, f):
        seen.append(x.shape[0])
        return orig(self, x, f)

    monkeypatch.setattr(ZoomViT, "forward_lowres", rec)
    log = train(era5, tmp_path, "bsk", "--zoom", "4:0.5,1:0.5", "--coarse_mode", "lowres",
                "--batch_scale_k", "1")
    n4, n1 = len(seen), log["steps"] - len(seen)
    assert n4 and set(seen) == {16}                    # batch 4 x k=4 in every coarse step
    assert log["samples"] == 16 * n4 + 4 * n1
    # coarse steps are charged for the coarse patches they embed and decode
    m = ZoomViT(12, 6, (32, 64), 2, 16, 1, 2)
    f4 = m.flops_per_step(16, 4, lowres=True)
    assert f4 < m.flops_per_step(16, 4)
    assert log["model_flops"] == pytest.approx(n4 * f4 + n1 * m.flops_per_step(4, 1), rel=1e-12)


def test_sampler_reshuffles_every_epoch():
    s = W.Sampler(10, 0, CPU)
    first = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    assert torch.equal(s.idx, first)                   # epoch 1 order unchanged
    ep = [torch.cat([s.next(3) for _ in range(3)]).tolist() for _ in range(3)]
    assert s.epochs == 2
    assert all(len(set(e)) == 9 for e in ep)           # no repeats inside an epoch
    assert ep[0] != ep[1] and ep[1] != ep[2]           # a new order every epoch
    s2 = W.Sampler(10, 0, CPU)
    assert [torch.cat([s2.next(3) for _ in range(3)]).tolist() for _ in range(3)] == ep


def test_training_order_changes_between_epochs(era5, tmp_path, monkeypatch):
    seen = []
    orig = W.Sampler.next

    def rec(self, batch):
        t = orig(self, batch)
        seen.append(t.tolist())
        return t

    monkeypatch.setattr(W.Sampler, "next", rec)
    train(era5, tmp_path, "ep", budget="3.3e10")       # ~130 steps of 4 over 243 train times
    per_epoch = 243 // 4
    assert len(seen) > 2 * per_epoch
    assert seen[:per_epoch] != seen[per_epoch:2 * per_epoch]
    assert max(max(b) for b in seen) <= 242            # t + 1 stays in the train years


def test_compile_time_excluded(era5, tmp_path, monkeypatch):
    wrapped = []

    def fake_compile(fn, rt):                          # 1 s "compile" per new input shape
        if not rt.compile:
            return fn
        wrapped.append(getattr(fn, "__name__", type(fn).__name__))
        shapes = set()

        def run(x, *a):
            if (tuple(x.shape), a) not in shapes:
                shapes.add((tuple(x.shape), a))
                time.sleep(1.0)
            return fn(x, *a)
        return run

    monkeypatch.setattr(W, "maybe_compile", fake_compile)
    log = train(era5, tmp_path, "cmp", "--zoom", "4:0.5,1:0.5", "--compile")
    assert log["compile_time"] >= 2.0 and log["train_time"] < 1.0
    log = train(era5, tmp_path, "cmpl", "--zoom", "4:0.5,1:0.5", "--coarse_mode", "lowres",
                "--compile")
    assert "forward_lowres" in wrapped                 # the coarse stage is compiled too
    assert log["compile_time"] >= 2.0


def test_optimizer_flops_match_flop_counter():
    torch.manual_seed(0)
    m = ZoomViT(12, 6, (32, 64), 2, 32, 2, 2, zoom_ks=(4,))
    opt = build_optimizer(m, "muon", 1e-3, 0.05)
    m(torch.randn(2, 12, 32, 64), 4).pow(2).mean().backward()
    muon = opt.optimizers[-1]
    with FlopCounterMode(display=False) as fc:
        muon.step()
    assert W.optimizer_step_flops(muon) == pytest.approx(fc.get_total_flops(), rel=0.02)
    assert W.optimizer_step_flops(opt) > W.optimizer_step_flops(muon)


def test_flops_mode_total(era5, tmp_path):
    small = ["--optim", "muon", "--d", "32", "--batch", "1"]   # optimizer ~7% of a step
    a = train(era5, tmp_path, "fm", *small)
    b = train(era5, tmp_path, "ft", *small, "--flops_mode", "total")
    assert a["opt_flops_per_step"] == b["opt_flops_per_step"] > 0
    assert a["opt_flops"] == pytest.approx(a["steps"] * a["opt_flops_per_step"])
    # model mode: the budget counts model FLOPs only; total mode adds the optimizer's
    assert a["evals"][-1]["flops"] == pytest.approx(a["model_flops"], rel=1e-12)
    assert b["evals"][-1]["flops"] == pytest.approx(b["model_flops"] + b["opt_flops"], rel=1e-12)
    assert b["steps"] < a["steps"]


def test_lr_scale_respected(era5, tmp_path, monkeypatch):
    def build(model, *a, **k):                         # C2: lr_scale 0 freezes the decay group
        opt = build_optimizer(model, *a, **k)
        for g in opt.param_groups:
            g["lr_scale"] = 0.0 if any(p is model.embed.weight for p in g["params"]) else 1.0
        return opt

    monkeypatch.setattr(W, "build_optimizer", build)
    save = str(tmp_path / "m.pt")
    train(era5, tmp_path, "ls", "--save", save)
    trained, _ = W.load_model(save)
    torch.manual_seed(0)
    init = ZoomViT(12, 6, (32, 64), 2, 16, 1, 2)
    assert torch.equal(trained.embed.weight, init.embed.weight)
    assert torch.equal(trained.decode.weight, init.decode.weight)    # zero init, frozen
    assert not torch.equal(trained.decode.bias, init.decode.bias)    # lr_scale 1 group trains


# -- data loading, splits and metrics


def test_stale_gridpack(tmp_path):
    d = str(tmp_path)
    x, meta = make_era5(d, hours6("2017-12-25", 64))
    gridpack.write(os.path.join(d, "fields.gridpack"), x, np.asarray(meta["diff_std"]), chunk=16)
    assert W.ERA5(d, CPU).source == "gridpack"
    time.sleep(0.01)
    x2, _ = make_era5(d, hours6("2017-12-29", 40), seed=1)    # prepare-era5 rerun in place
    ds = W.ERA5(d, CPU)                                       # newer fields.npy wins
    assert ds.source == "npy" and ds.x.shape[0] == 40
    gp = os.path.join(d, "fields.gridpack")
    os.utime(gp, (time.time() + 10, time.time() + 10))       # stale pack that looks newer
    with pytest.raises(ValueError, match="meta.json"):
        W.ERA5(d, CPU)


def test_val_year_split(era5_3y, tmp_path, monkeypatch):
    calls, starts = [], []
    q, nxt = W.quick_eval, W.Sampler.next
    monkeypatch.setattr(W, "quick_eval", lambda m, ds, idx, rt: calls.append(
        (int(idx.min()), int(idx.max()))) or q(m, ds, idx, rt))
    monkeypatch.setattr(W.Sampler, "next", lambda s, b: starts.append(nxt(s, b)) or starts[-1])
    log = train(era5_3y, tmp_path, "v", "--val_year", "2017", "--zoom", "4:0.5,1:0.5")
    sp = log["split"]
    assert sp["eval"] == "val" and sp["norm_stats_rows"] == 120
    assert (sp["train"]["n"], sp["val"]["n"], sp["test"]["n"]) == (120, 100, 80)
    assert sp["val"]["first"] == "2017-03-01T00" and sp["test"]["first"] == "2018-01-01T00"
    assert int(torch.cat(starts).max()) <= 118                 # t + 1 < first 2017 time
    periodic, (fin_lo, fin_hi), (val_lo, val_hi) = calls[:-2], calls[-2], calls[-1]
    assert len(periodic) == 3 and all(120 <= a and b <= 220 - 14 for a, b in periodic)
    assert 220 <= fin_lo and fin_hi <= 300 - 14               # final: test year only
    assert 120 <= val_lo and val_hi <= 220 - 14
    assert log["final_val"]["val_loss"] != log["final"]["val_loss"]
    for key in ("acc_z500_24h", "clim_t2m_72h", "acc_persist_t850_6h"):
        assert key in log["final"] and key in log["final_val"]
    # default: everything before the test year trains, the evals run on the test year
    log = train(era5_3y, tmp_path, "t")
    assert log["split"]["eval"] == "test" and log["split"]["train"]["n"] == 220
    assert "final_val" not in log
    for bad in ("2016", "2018", "2019"):
        with pytest.raises(SystemExit):
            train(era5_3y, tmp_path, "bad", "--val_year", bad)


def test_val_year_normalization_uses_train_rows_only(era5_3y):
    x = np.load(os.path.join(era5_3y, "fields.npy")).astype(np.float64)
    st = W.train_stats(x.astype(np.float32), 120)
    assert np.allclose(st["mean"], x[:120].mean(axis=(0, 2, 3)), rtol=1e-9)
    assert np.allclose(st["std"], x[:120].std(axis=(0, 2, 3)), rtol=1e-9)
    assert np.allclose(st["diff_std"], (x[1:120] - x[:119]).std(axis=(0, 2, 3)), rtol=1e-9)
    ds = W.ERA5(era5_3y, CPU, stats_end=120)
    assert ds.stats_rows == 120
    assert torch.allclose(ds.mean.flatten().double(), torch.tensor(st["mean"]).double(), rtol=1e-6)
    meta_ds = W.ERA5(era5_3y, CPU, stats_end=220)              # = meta n_train: meta's stats
    raw = torch.from_numpy(np.load(os.path.join(era5_3y, "fields.npy")))
    assert torch.equal(meta_ds.x, (raw - meta_ds.mean) / meta_ds.std)


def test_climatology_from_train_years_only(era5):
    ds = W.ERA5(era5, CPU)
    end, ch = ds.n_train, [0, 2]
    clim0 = ds.climatology(end, ch, window=0)
    t = torch.tensor([5, 100])                                 # train times
    for ti in t.tolist():
        same = [i for i in range(end) if ds.doy[i] == ds.doy[ti] and ds.hour[i] == ds.hour[ti]]
        ref = ds.x[same][:, ch].double().mean(0)
        assert torch.allclose(clim0(torch.tensor([ti]))[0].double(), ref, atol=1e-6)
    # a test-year day the train years never reach falls back to that hour's train mean
    tt = torch.tensor([end + 40])
    hour_rows = [i for i in range(end) if ds.hour[i] == ds.hour[end + 40]]
    assert torch.allclose(clim0(tt)[0].double(), ds.x[hour_rows][:, ch].double().mean(0),
                          atol=1e-6)
    clim = ds.climatology(end, ch)
    ds.x[end:] += 100.0                                        # test data must not matter
    clim_b = ds.climatology(end, ch)
    allt = torch.arange(ds.n_total)
    assert torch.equal(clim(allt), clim_b(allt))


def test_lat_acc_and_new_metrics_keep_old_fields(era5):
    w = torch.rand(1, 1, 32, 1) + 0.5
    a = torch.randn(3, 2, 32, 64)
    assert torch.allclose(W.lat_acc(a, a, w), torch.ones(3, 2))
    assert torch.allclose(W.lat_acc(-2 * a, a, w), -torch.ones(3, 2))
    ds = W.ERA5(era5, CPU)
    torch.manual_seed(0)
    m = ZoomViT(12, 6, (32, 64), 2, 16, 1, 2)
    for p in m.decode.parameters():
        torch.nn.init.normal_(p, std=0.01)
    rt = W.setup_runtime("cpu", "fp32", False, None, 0)
    idx = torch.arange(ds.n_train, ds.n_total - 13)
    old = W.full_eval(m, ds, idx[::3], idx[::12], rt)
    new = W.full_eval(m, ds, idx[::3], idx[::12], rt,
                      clim=ds.climatology(ds.n_train, [0, 1, 2]))
    assert all(new[k] == v for k, v in old.items())
    for c in W.REPORT:
        for h in (6, 24, 72):
            assert -1 <= new[f"acc_{c}_{h}h"] <= 1 and -1 <= new[f"acc_persist_{c}_{h}h"] <= 1
            assert new[f"clim_{c}_{h}h"] > 0


# -- checkpoints, --save and forecasts


def test_resume_bit_exact(era5, tmp_path):
    common = ["--zoom", "4:0.5,1:0.5", "--optim", "muon", "--mix_fine", "0.3"]
    full = train(era5, tmp_path, "full", *common, "--save", str(tmp_path / "full.pt"),
                 budget="2e10")
    run(era5, tmp_path, "part", *common, "--ckpt_every", "20", "--stop_step", "70",
        budget="2e10")
    assert not os.path.exists(tmp_path / "part.json")  # stopped early: no final log yet
    ck = tmp_path / "part.ckpt"
    state = torch.load(ck, weights_only=True)          # plain tensors and Python values
    assert state["progress"]["step"] == 70 and state["sampler"]["epochs"] == 1
    part = train(era5, tmp_path, "part", *common, "--resume", str(ck),
                 "--save", str(tmp_path / "part.pt"), budget="2e10")
    assert part["resumed_from"] == str(ck)
    assert full["steps"] == part["steps"] > 70 and full["samples"] == part["samples"]
    assert [t["loss"] for t in full["train"]] == [t["loss"] for t in part["train"]]
    assert [e["val_loss"] for e in full["evals"]] == [e["val_loss"] for e in part["evals"]]
    assert full["final"] == part["final"]
    a, b = W.load_model(str(tmp_path / "full.pt"))[0], W.load_model(str(tmp_path / "part.pt"))[0]
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    with pytest.raises(SystemExit):                    # model options must match
        train(era5, tmp_path, "part", "--zoom", "1", "--optim", "muon", "--resume", str(ck))


def test_save_and_forecast(era5, tmp_path):
    save = str(tmp_path / "wx.pt")
    train(era5, tmp_path, "s", "--zoom", "4:0.5,1:0.5", "--save", save)
    ck = torch.load(save, weights_only=True)
    assert ck["config"]["zoom_ks"] == [4] and ck["channels"] == list(CH)
    model, _ = W.load_model(save)
    ds = W.ERA5(era5, CPU)
    i0 = list(ds.times.astype(str)).index("2018-01-02T06")
    fc = W.forecast(save, era5, "2018-01-02T06", steps=4, out=str(tmp_path / "f.npy"))
    assert fc.shape == (5, 6, 32, 64) and fc.dtype == np.float32
    raw = np.load(os.path.join(era5, "fields.npy"))
    assert np.array_equal(fc[0], raw[i0])
    xn = ds.x[i0:i0 + 1]
    with torch.no_grad():
        for s in range(1, 5):                          # same rollout as full_eval
            xn = xn + model(ds.inputs(xn, torch.tensor([i0 + s - 1])), 1) * ds.res_scale
            ref = (xn * ds.std + ds.mean)[0].numpy()
            assert np.allclose(fc[s], ref, rtol=1e-5, atol=1e-3 * np.abs(ref).max())
    side = json.load(open(tmp_path / "f.json"))
    assert side["valid_times"][-1] == "2018-01-03T06" and len(side["rmse"]["z500"]) == 5
    assert side["rmse"]["z500"][0] == 0.0
    # CLI, past the end of the data: the forcings come from the valid times
    out = str(tmp_path / "cli.npy")
    W.main(["--forecast", "--data", era5, "--load", save, "--init", "2018-01-19T12",
            "--steps", "3", "--forecast_out", out])
    assert np.load(out).shape == (4, 6, 32, 64)
    assert json.load(open(tmp_path / "cli.json"))["rmse_lead_hours"] == [0, 6]
    with pytest.raises(ValueError):
        W.forecast(save, era5, "2019-01-01T00", steps=2)
    W.main(["--forecast", "--load", save, "--init", "2018-01-02T06", "--steps", "4",
            "--out", str(tmp_path)])                   # --data defaults to the training data
    assert np.array_equal(np.load(tmp_path / "forecast_2018-01-02T06.npy"), fc)
