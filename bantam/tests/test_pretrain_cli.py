"""bantam pretrain: argument validation, divergence, run-log fields, checkpoints (in-process, tiny)."""
import argparse
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import optim  # noqa: E402
from bantamtrain.cli import pretrain  # noqa: E402
from bantamtrain.data import tokpack  # noqa: E402
from bantamtrain.device import Runtime  # noqa: E402
from bantamtrain.lowmem import QState  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.zoom import ZoomSchedule  # noqa: E402

V = 64
TINY = ["--d", "16", "--layers", "1", "--heads", "1", "--seq", "32", "--batch", "4",
        "--threads", "1", "--evals", "2", "--eval_seqs", "8", "--final_eval_seqs", "16"]


def make_data(d, n_train=60_000, n_val=3_000, pack=False):
    """A learnable synthetic token stream (next token mostly a function of the last)."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)

    def stream(n):
        t = rng.integers(0, V, n).astype(np.uint16)
        keep = rng.random(n) < 0.7
        for i in range(1, n):
            if keep[i]:
                t[i] = (t[i - 1] * 5 + 3) % V
        return t
    train, val = stream(n_train), stream(n_val)
    json.dump({"vocab_size": V, "eos_id": 0}, open(os.path.join(d, "meta.json"), "w"))
    if pack:
        tokpack.write(os.path.join(d, "train.tokpack"), train, V, chunk=1024)
        tokpack.write(os.path.join(d, "val.tokpack"), val, V, chunk=1024)
    else:
        train.tofile(os.path.join(d, "train.bin"))
        val.tofile(os.path.join(d, "val.bin"))
    return str(d)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_data(tmp_path_factory.mktemp("tok"))


@pytest.fixture
def run(tmp_path):
    """pretrain.main in-process; restores the torch thread count and RNG it changes."""
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    out = str(tmp_path / "out")

    def _run(data, *extra, name="r", budget="5e8"):
        argv = ["--data", data, "--budget", budget, "--name", name, "--out", out] + TINY
        return pretrain.main(argv + [str(a) for a in extra])
    _run.out = out
    yield _run
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def _strict(path):
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.load(open(path), parse_constant=bad)


def test_diverged_run_exits_nonzero_and_is_not_marked_finished(data, run, tmp_path):
    save = str(tmp_path / "m.pt")
    with pytest.raises(SystemExit) as e:
        run(data, "--optim", "adamw", "--lr", "1e6", "--warmup", "0", "--save", save)
    assert e.value.code == pretrain.EXIT_DIVERGED
    log = _strict(os.path.join(run.out, "r.json"))               # strict JSON (no bare NaN)
    assert log["status"] == "diverged" and log["diverged"] is True
    assert "final_val_loss" not in log and "in_progress" not in log
    assert log["train"][-1]["loss"] is None and log["diverged_at_step"] == log["steps"]
    assert not os.path.exists(save)


def test_bad_arguments_fail_before_any_file_is_written(data, run, tmp_path, capsys):
    small_val = make_data(tmp_path / "small", n_val=3_000)        # 93 windows of 32
    for extra, msg in [(["--final_eval_seqs", "2048"], "final_eval_seqs"),
                       (["--zoom", "3:0.5,1:0.5"], "[3]"),        # 32 % 3 != 0
                       (["--zoom", "2:0.5,1:0.4"], "--zoom"),     # fractions do not sum to 1
                       (["--duty", "0"], "--duty"),
                       (["--batch", "4000"], "windows")]:
        with pytest.raises(SystemExit) as e:
            run(small_val, *extra)
        assert e.value.code == 2, extra
        assert msg in capsys.readouterr().err, extra
        assert not os.path.exists(run.out), extra


def test_zoom_level_above_4_trains(data, run):
    assert run(data, "--zoom", "8:0.5,1:0.5", "--stop_step", "3") == 0


def test_run_log_records_sources_flops_and_eval_subsets(tmp_path, run, capsys):
    pack = make_data(tmp_path / "pack", pack=True)
    save = str(tmp_path / "w" / "model.bin")
    assert run(pack, "--optim", "muon", "--save", save, "--save_every_eval", "1") == 0
    log = _strict(os.path.join(run.out, "r.json"))
    assert log["status"] == "finished" and "in_progress" not in log
    src = log["data_source"]
    assert src["train"]["format"] == "tokpack" and src["train"]["tokens"] == 60_000
    assert src["train"]["file"].endswith("train.tokpack") and src["val"]["tokens"] == 3_000
    assert "raw" not in capsys.readouterr().err                   # no raw-.bin warning
    # optimizer FLOPs are logged but (default --flops_mode model) not charged to the budget
    assert log["flops_mode"] == "model" and log["opt_flops_per_step"] > 0
    for e in log["evals"]:
        assert e["total_flops"] == pytest.approx(e["flops"] + e["opt_flops"])
    assert log["opt_flops"] == pytest.approx(log["steps"] * log["opt_flops_per_step"])
    # the final model is also scored on the periodic subset, so curve and finals compare
    last = log["evals"][-1]
    assert last["step"] == log["steps"] and last["frac"] >= 1
    assert log["final_val_loss_eval_seqs"] == last["val_loss"]
    assert log["eval_subset"] == {"periodic_first_seqs": 8, "final_first_seqs": 16, "seq": 32}
    m = pretrain.build_model(log["args"], V)
    m.load_state_dict(torch.load(save, weights_only=True))
    val = pretrain.open_tokens(pack, "val")
    rt = Runtime(torch.device("cpu"), None, None, False)
    args = type("A", (), {"seq": 32})
    assert pretrain.evaluate(m, val, args, rt, 8) == log["final_val_loss_eval_seqs"]
    assert pretrain.evaluate(m, val, args, rt, 16) == log["final_val_loss"]
    # --save_every_eval keeps the extension and gives every eval its own file
    cks = [e["checkpoint"] for e in log["evals"] if "checkpoint" in e]
    assert len(cks) == len(set(cks)) == 2 and all(c.endswith(".bin") for c in cks)


def test_raw_bin_source_is_recorded_and_warned(data, run, capsys):
    assert run(data, "--stop_step", "2") == 0
    assert "raw" in capsys.readouterr().err
    log = json.load(open(os.path.join(run.out, "r.json")))
    assert log["data_source"]["train"]["format"] == "bin"
    assert log["in_progress"] is True                           # stopped early: not finished


def test_flops_mode_total_charges_optimizer_flops(data, run, monkeypatch):
    monkeypatch.setattr(optim, "optimizer_flops", lambda opt: 2e6, raising=False)  # C1
    run(data, name="model")
    run(data, "--flops_mode", "total", name="total")
    a = json.load(open(os.path.join(run.out, "model.json")))
    b = json.load(open(os.path.join(run.out, "total.json")))
    assert a["opt_flops_per_step"] == b["opt_flops_per_step"] == 2e6
    assert b["steps"] < a["steps"]
    assert b["evals"][-1]["flops"] == b["evals"][-1]["total_flops"] >= 5e8
    assert a["evals"][-1]["flops"] == a["model_flops"] >= 5e8


def test_param_group_lr_scale_is_applied(data, run, monkeypatch):
    seen = []

    def build(model, *a, **k):                                    # C2: groups carry lr_scale
        opt = optim.build_optimizer(model, *a, **k)
        for i, g in enumerate(opt.param_groups):
            g["lr_scale"] = 0.5 if i == 0 else 1.0
        seen.append(opt)
        return opt
    monkeypatch.setattr(pretrain, "build_optimizer", build)
    run(data, "--sched", "wsd", "--warmup", "0", "--lr", "0.01", "--stop_step", "2")
    lrs = [g["lr"] for g in seen[0].param_groups]
    assert lrs[0] == pytest.approx(0.005) and all(x == pytest.approx(0.01) for x in lrs[1:])


def test_save_writes_save_pretrained_dir_when_the_model_has_it(data, run, tmp_path, monkeypatch):
    if not hasattr(ZoomGPT, "save_pretrained"):                   # C8 (model group)
        def fake(self, d, strip_zoom=True):
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "config.json"), "w").write("{}")
        monkeypatch.setattr(ZoomGPT, "save_pretrained", fake, raising=False)
    save = str(tmp_path / "m.pt")
    assert run(data, "--save", save) == 0
    log = json.load(open(os.path.join(run.out, "r.json")))
    assert os.path.exists(save)                                   # the plain state_dict stays
    assert log["saved"]["pretrained"] == str(tmp_path / "m_pretrained")
    assert os.path.exists(os.path.join(log["saved"]["pretrained"], "config.json"))


def test_auto_batch_split_and_cpu_noop(data, run, capsys):
    assert pretrain.pick_micro_batch(24, 16) == (12, 2)
    assert pretrain.pick_micro_batch(16, 8) == (8, 2)
    assert pretrain.pick_micro_batch(16, 64) == (16, 1)
    assert pretrain.pick_micro_batch(16, 0) is None and pretrain.pick_micro_batch(16, None) is None
    assert pretrain.pick_micro_batch(16, 8, lambda m, a: m <= 2) == (2, 8)
    a = type("A", (), {"batch": 16, "accum": 1, "batch_scale": 1.5, "batch_scale_k": 0})
    assert pretrain.split_batch(a, [2, 1], 8) == (8, 2)          # 12 x 2 = 24 zoomed sequences
    assert pretrain.split_batch(a, [2, 1], 1) is None            # round(1.5) x 16 != 24
    assert run(data, "--auto_batch", "--stop_step", "1") == 0
    assert "not a CUDA device" in capsys.readouterr().out


def test_auto_batch_probe_does_not_change_the_model_init(monkeypatch):
    # the probe builds and steps throwaway models before the real one is built
    def find(try_step, start, limit, device):
        try_step(1)
        try_step(2)
        return 2
    monkeypatch.setattr(pretrain, "find_batch_size", find)
    a = argparse.Namespace(d=16, layers=1, heads=1, zoom="2:0.5,1:0.5", grad_ckpt=0,
                           param_dtype="bf16", optim="muon", lr=1e-3, wd=0.1, state_bits=32,
                           ns="quintic", seq=32, batch=4, accum=1, batch_scale=0.0,
                           batch_scale_k=0, mix_fine=0.0)
    rt = Runtime(torch.device("cpu"), None, None, False)
    torch.manual_seed(0)
    want = pretrain.build_model(a, V).state_dict()
    torch.manual_seed(0)
    assert pretrain.auto_micro_batch(a, V, ZoomSchedule.parse(a.zoom), rt) == 2
    got = pretrain.build_model(a, V).state_dict()
    assert all(torch.equal(want[k], got[k]) for k in want)


def test_eval_schedule_and_eval_checkpoint_names():
    nxt = pretrain.next_eval_after
    assert nxt(0.0, 10) == pytest.approx(0.1) and nxt(0.1 - 1e-13, 10) == pytest.approx(0.2)
    assert nxt(0.37, 10) == pytest.approx(0.4)                   # one step jumped 3 points
    assert nxt(1.0, 10) > 1.0
    p = pretrain.eval_ckpt_path
    assert p("runs/model.pt", 0.1, 5) == "runs/model_f0.10.pt"   # unchanged naming
    assert p("runs/model.bin", 0.1, 5) == "runs/model_f0.10.bin"
    assert p("runs/model", 0.1, 5) == "runs/model_f0.10.pt"
    assert p("ckpts.pt/model.pt", 0.1, 5) == "ckpts.pt/model_f0.10.pt"
    assert p("m.pt", 0.105, 7, {"m_f0.10.pt"}) == "m_f0.1050_step7.pt"   # 200 evals collide


def test_bf16_model_is_scored_in_fp32_arithmetic():
    torch.manual_seed(0)
    m = ZoomGPT(V, 32, 1, 2).to(torch.bfloat16)
    val = np.random.default_rng(0).integers(0, V, 2000).astype(np.uint16)
    rt = Runtime(torch.device("cpu"), None, None, False)
    args = type("A", (), {"seq": 16})
    got = pretrain.evaluate(m, val, args, rt, 64)
    assert all(p.dtype == torch.bfloat16 for p in m.parameters()) and m.training
    ref = pretrain.evaluate(m.float(), val, args, rt, 64)          # fp32 copy of the bf16 weights
    assert got == ref


# ------------------------------------------------------------------ checkpoints
def _trained(bits=8, dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    m = ZoomGPT(V, 32, 1, 2).to(dtype)
    opt = optim.build_optimizer(m, "muon", 1e-2, 0.1, state_bits=bits, fused=False)
    x = torch.randint(0, V, (2, 17))
    m(x[:, :16], x[:, 1:17], k=1).float().backward()
    opt.step()
    return m, opt


def _step(m, opt, seed=1):
    torch.manual_seed(seed)
    opt.zero_grad()
    x = torch.randint(0, V, (2, 17), device=next(m.parameters()).device)
    m(x[:, :16], x[:, 1:17], k=1).float().backward()
    opt.step()


def test_checkpoint_loads_weights_only_and_restores_scaler_and_8bit_state(tmp_path, data):
    m, opt = _trained(bits=8)
    scaler = torch.amp.GradScaler("cpu", init_scale=512.0)
    rt = Runtime(torch.device("cpu"), None, scaler, False)
    stream = pretrain.ChunkStream(pretrain.open_tokens(data, "train"), 16, 0)
    log = {"args": {"name": "x"}, "evals": []}
    path = str(tmp_path / "c.ckpt")
    pretrain.save_ckpt(path, m, opt, stream, {"flops": 1.0}, log, np.random.default_rng(1), rt)
    ck = torch.load(path, map_location="cpu", weights_only=True)    # no pickle trust needed
    assert ck["scaler"]["scale"] == 512.0
    assert pretrain.load_ckpt(path)["progress"] == {"flops": 1.0}
    m2, opt2 = _trained(bits=8)
    _step(m2, opt2, seed=7)                                       # different state before loading
    m2.load_state_dict(ck["model"])
    pretrain.load_optimizer_state(opt2, ck["opt"])
    assert any(isinstance(v, QState) for o in opt2.optimizers for st in o.state.values()
               for v in st.values())
    _step(m, opt)
    _step(m2, opt2)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    rt2 = Runtime(torch.device("cpu"), None, torch.amp.GradScaler("cpu"), False)
    rt2.scaler.load_state_dict(ck["scaler"])
    assert rt2.scaler.get_scale() == 512.0


def test_optimizer_state_follows_param_device_and_keeps_fp32_state(tmp_path):
    # 8-bit state of a checkpoint loaded on the CPU moves to the params' device ('meta' stands
    # in for a GPU here), so the first step after --resume does not mix devices
    m, opt = _trained(bits=8)
    sd = pretrain._encode_state(opt.state_dict())
    m2 = ZoomGPT(V, 32, 1, 2).to("meta")
    opt2 = optim.build_optimizer(m2, "muon", 1e-2, 0.1, state_bits=8, fused=False)
    pretrain.load_optimizer_state(opt2, sd)
    qs = [v for o in opt2.optimizers for st in o.state.values() for v in st.values()
          if isinstance(v, QState)]
    assert qs and all(q.q.device.type == "meta" for q in qs)
    # fp32 Muon momentum of bf16 params stays fp32 (torch would cast it to bf16)
    m, opt = _trained(bits=32, dtype=torch.bfloat16)
    saved = opt.state_dict()
    m2, opt2 = _trained(bits=32, dtype=torch.bfloat16)
    pretrain.load_optimizer_state(opt2, saved)
    muon = opt2.optimizers[-1]
    bufs = [st["buf"] for st in muon.state.values()]
    assert bufs and all(b.dtype == torch.float32 for b in bufs)
    want = [st["buf"] for st in opt.optimizers[-1].state.values()]
    assert all(torch.equal(a, b) for a, b in zip(bufs, want))


def test_old_checkpoint_with_pickled_qstate_loads_with_a_warning(tmp_path, capsys):
    path = str(tmp_path / "old.ckpt")
    torch.save({"opt": [{"state": {0: {"buf": QState(torch.zeros(4, 4))}}}]}, path)
    with pytest.raises(Exception):
        torch.load(path, weights_only=True)
    ck = pretrain.load_ckpt(path)
    assert isinstance(ck["opt"][0]["state"][0]["buf"], QState)
    assert "old checkpoint format" in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_checkpoint_saves_cuda_rng(tmp_path, data):
    m, opt = _trained(bits=32)
    rt = Runtime(torch.device("cuda", 0), None, None, False)
    stream = pretrain.ChunkStream(pretrain.open_tokens(data, "train"), 16, 0)
    path = str(tmp_path / "c.ckpt")
    pretrain.save_ckpt(path, m, opt, stream, {}, {"args": {}}, np.random.default_rng(1), rt)
    assert torch.equal(torch.load(path, weights_only=True)["cuda_rng"],
                       torch.cuda.get_rng_state(rt.device))


def test_resumed_optimizer_takes_this_runs_hyperparameters():
    m, opt = _trained(bits=32)                                    # wd 0.1
    m2 = ZoomGPT(V, 32, 1, 2)
    opt2 = optim.build_optimizer(m2, "muon", 1e-2, 0.0, state_bits=32, fused=False)   # --wd 0
    pretrain.load_optimizer_state(opt2, opt.state_dict())         # (with --allow_arg_change)
    assert all(g["weight_decay"] == 0.0 for g in opt2.param_groups)
    assert opt2.optimizers[-1].state                              # the state itself is restored
