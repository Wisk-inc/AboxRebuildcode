"""bantam pretrain under DDP (2 gloo ranks on CPU): disjoint data, rank-0-only evals, joint NaN stop."""
import json
import os
import socket
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = """
import json, os, sys
sys.path.insert(0, {root!r})
from bantamtrain.cli import pretrain
from bantamtrain.models import gpt

seen, calls = [], [0]
orig_next, orig_fwd = pretrain.ChunkStream.next, gpt.ZoomGPT.forward

def nxt(self, batch):                      # record the windows this rank reads
    c = orig_next(self, batch)
    world, rank = getattr(self, "world", 1), getattr(self, "rank", 0)
    i = self.pos - batch * world + rank * batch
    seen.extend(int(s) for s in self.starts[i:i + batch])
    return c

def fwd(self, idx, targets, k=1):          # optional non-finite loss on one rank only
    loss = orig_fwd(self, idx, targets, k)
    if self.training:
        calls[0] += 1
        if os.environ.get("NAN_ON_RANK") == os.environ["RANK"] and calls[0] == 25:
            loss = loss * float("inf")
    return loss

pretrain.ChunkStream.next, gpt.ZoomGPT.forward = nxt, fwd
try:
    rc = pretrain.main()
finally:
    json.dump(seen, open(os.path.join(os.environ["OUT"], "seen%s.json" % os.environ["RANK"]), "w"))
sys.exit(rc)
"""


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(tmp_path, data, extra=(), env_extra=None):
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(root=ROOT))
    port = str(_port())
    args = [sys.executable, str(driver), "--data", data, "--name", "ddp", "--out", out,
            "--d", "16", "--layers", "1", "--heads", "1", "--seq", "32", "--batch", "4",
            "--accum", "2", "--threads", "1", "--zoom", "2:0.5,1:0.5", "--budget", "1.2e9",
            "--evals", "100", "--eval_seqs", "8", "--final_eval_seqs", "8", *extra]
    procs = []
    for r in range(2):
        env = {**os.environ, "OMP_NUM_THREADS": "1", "WORLD_SIZE": "2", "RANK": str(r),
               "LOCAL_RANK": str(r), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": port, "OUT": out,
               **(env_extra or {})}
        procs.append(subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
    res = []
    try:
        for p in procs:
            o, e = p.communicate(timeout=90)
            res.append((p.returncode, o, e))
    finally:
        for p in procs:
            p.kill()
    return out, res


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    rng = np.random.default_rng(0)
    rng.integers(0, 64, 60_000).astype(np.uint16).tofile(os.path.join(d, "train.bin"))
    rng.integers(0, 64, 3_000).astype(np.uint16).tofile(os.path.join(d, "val.bin"))
    json.dump({"vocab_size": 64}, open(os.path.join(d, "meta.json"), "w"))
    return str(d)


def test_ddp_ranks_shard_the_data_and_only_rank0_evaluates(tmp_path, data):
    out, res = _launch(tmp_path, data)
    for rc, o, e in res:
        assert rc == 0, e[-3000:]
    assert "frac=" in res[0][1] and "frac=" not in res[1][1]      # rank 1 never evaluates
    log = json.load(open(os.path.join(out, "ddp.json")))
    assert log["status"] == "finished" and log["world_size"] == 2
    assert len(log["evals"]) <= log["steps"] + 1                  # one eval per step at most
    seen = [json.load(open(os.path.join(out, f"seen{r}.json"))) for r in range(2)]
    assert len(seen[0]) == len(seen[1]) == log["steps"] * 4 * 2    # batch x accum per step
    assert not set(seen[0]) & set(seen[1])                       # disjoint shards
    assert len(set(seen[0] + seen[1])) == len(seen[0]) + len(seen[1])
    n_windows = (60_000 - 32 - 4) // 32
    assert log["data_epochs"] == pytest.approx(len(seen[0]) * 2 / n_windows)   # global


def test_ddp_nonfinite_loss_on_one_rank_stops_every_rank(tmp_path, data):
    out, res = _launch(tmp_path, data, env_extra={"NAN_ON_RANK": "1"})
    assert [rc for rc, _, _ in res] == [3, 3], [e[-2000:] for _, _, e in res]
    log = json.load(open(os.path.join(out, "ddp.json")))
    assert log["status"] == "diverged" and "final_val_loss" not in log
