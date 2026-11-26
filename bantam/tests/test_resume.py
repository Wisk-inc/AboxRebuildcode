"""Resumable checkpoints: stop + resume must reproduce an uninterrupted run exactly.

Runs the installed entry point (`python -m bantamtrain pretrain`) on a synthetic corpus.
Set BANTAM_TINYSTORIES=<tokenized TinyStories dir> to also run the slow real-data check.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = ["--zoom", "2:0.5,1:0.5", "--d", "32", "--layers", "1", "--heads", "1", "--batch", "4",
          "--evals", "2", "--eval_seqs", "8", "--final_eval_seqs", "8", "--threads", "1",
          "--optim", "muon", "--state_bits", "8", "--mix_fine", "0.3"]
SYNTH = ["--seq", "32", "--budget", "3e9"]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    rng = np.random.default_rng(0)
    rng.integers(0, 64, 200_000).astype(np.uint16).tofile(os.path.join(d, "train.bin"))
    rng.integers(0, 64, 5_000).astype(np.uint16).tofile(os.path.join(d, "val.bin"))
    json.dump({"vocab_size": 64, "eos_id": 0}, open(os.path.join(d, "meta.json"), "w"))
    return str(d)


def run(data, name, out, extra, check=True):
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-m", "bantamtrain", "pretrain", "--data", data,
                        "--name", name, "--out", out] + COMMON + extra,
                       cwd=out, env=env, capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stderr[-3000:]
    return r


def _stop_and_resume(data, out, extra):
    run(data, "full", out, extra)
    run(data, "part", out, extra + ["--ckpt_every", "10", "--stop_step", "20"])
    ckpt = os.path.join(out, "part.ckpt")
    assert os.path.exists(ckpt)
    run(data, "part", out, extra + ["--resume", ckpt])
    a = json.load(open(os.path.join(out, "full.json")))
    b = json.load(open(os.path.join(out, "part.json")))
    assert b["resumed_from"].endswith("part.ckpt")
    assert a["steps"] == b["steps"] and a["tokens"] == b["tokens"]
    assert [t["loss"] for t in a["train"]] == [t["loss"] for t in b["train"]]
    assert a["final_val_loss"] == b["final_val_loss"]
    # the eval curve and the FLOP counters continue too (no extra eval at the resume step)
    curve = [(e["step"], e["frac"], e["val_loss"], e["opt_flops"]) for e in a["evals"]]
    assert curve == [(e["step"], e["frac"], e["val_loss"], e["opt_flops"]) for e in b["evals"]]
    assert (a["model_flops"], a["opt_flops"]) == (b["model_flops"], b["opt_flops"])
    return b, ckpt


def test_stop_and_resume_matches_uninterrupted(data, tmp_path):
    b, ckpt = _stop_and_resume(data, str(tmp_path), SYNTH)
    # the log records the invocation that finished the run, the interrupted one in the history
    assert b["args"]["stop_step"] == 0 and b["args"]["resume"] == ckpt
    assert b["resume_history"][0]["args"]["stop_step"] == 20 and b["resume_history"][0]["step"] == 20
    assert b["status"] == "finished" and "in_progress" not in b
    # 8-bit optimizer state is stored as plain tensors: no pickle trust needed to resume
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert ck["progress"]["step"] == 20


def test_resume_refuses_model_or_schedule_changes(data, tmp_path):
    out = str(tmp_path)
    run(data, "p", out, SYNTH + ["--ckpt_every", "5", "--stop_step", "5"])
    ckpt = os.path.join(out, "p.ckpt")
    before = open(os.path.join(out, "p.json")).read()
    r = run(data, "p", out, ["--seq", "64", "--budget", "6e9", "--seed", "1", "--resume", ckpt],
            check=False)
    assert r.returncode == 2
    assert all(f in r.stderr for f in ("'seq'", "'budget'", "'seed'")), r.stderr[-2000:]
    assert open(os.path.join(out, "p.json")).read() == before     # nothing ran
    # other changes only warn, and the new values are recorded
    r = run(data, "p", out, SYNTH + ["--eval_seqs", "4", "--resume", ckpt])
    assert "warning: --resume" in r.stderr and "eval_seqs" in r.stderr
    log = json.load(open(os.path.join(out, "p.json")))
    assert log["args"]["eval_seqs"] == 4 and log["status"] == "finished"


TINYSTORIES = os.environ.get("BANTAM_TINYSTORIES", "")


@pytest.mark.skipif(not os.path.isdir(TINYSTORIES), reason="set BANTAM_TINYSTORIES (slow)")
def test_stop_and_resume_matches_uninterrupted_tinystories(tmp_path):
    _stop_and_resume(TINYSTORIES, str(tmp_path), ["--budget", "6e11"])
