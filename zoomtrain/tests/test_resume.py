"""Resumable checkpoints: stop + resume must reproduce an uninterrupted run exactly."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/home/user/data/tinystories/tok4k"
COMMON = ["--data", DATA, "--zoom", "2:0.5,1:0.5", "--budget", "6e11", "--d", "32", "--layers", "1",
          "--heads", "1", "--batch", "4", "--evals", "2", "--eval_seqs", "8", "--final_eval_seqs", "8",
          "--threads", "1", "--optim", "muon", "--state_bits", "8", "--mix_fine", "0.3"]


def run(name, out, extra):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, os.path.join(ROOT, "llm", "train.py"), "--name", name,
                    "--out", out] + COMMON + extra, check=True, cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


@pytest.mark.skipif(not os.path.exists(DATA), reason="needs the tokenized TinyStories data")
def test_stop_and_resume_matches_uninterrupted(tmp_path):
    out = str(tmp_path)
    run("full", out, [])
    run("part", out, ["--ckpt_every", "10", "--stop_step", "20"])
    assert os.path.exists(os.path.join(out, "part.ckpt"))
    run("part", out, ["--resume", os.path.join(out, "part.ckpt")])
    a = json.load(open(os.path.join(out, "full.json")))
    b = json.load(open(os.path.join(out, "part.json")))
    assert b["resumed_from"].endswith("part.ckpt")
    assert a["steps"] == b["steps"] and a["tokens"] == b["tokens"]
    assert [t["loss"] for t in a["train"]] == [t["loss"] for t in b["train"]]
    assert a["final_val_loss"] == b["final_val_loss"]
