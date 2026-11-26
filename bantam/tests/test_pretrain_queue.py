"""experiments/run_queue.py: unfinished runs re-run (resumed), finished ones skipped, paths."""
import importlib.util
import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("run_queue", os.path.join(ROOT, "experiments",
                                                                         "run_queue.py"))
rq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq)


@pytest.fixture
def queue_root(tmp_path, monkeypatch):
    """run_queue with ROOT=tmp_path/root and a fake subprocess.run that records argv."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(rq, "ROOT", str(root))
    calls = []

    def fake_run(argv, cwd, **kw):
        calls.append(argv)
        assert cwd == str(root)
        return subprocess.CompletedProcess(argv, fake_run.rc)
    fake_run.rc = 0
    monkeypatch.setattr(rq.subprocess, "run", fake_run)
    return root, calls, fake_run


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj))


def test_only_finished_runs_are_skipped(queue_root):
    root, calls, _ = queue_root
    cmd = "python3 llm/train.py --data D --budget 1 --name {n} --out results/q"
    cases = {"fresh": None,
             "in_progress": {"evals": [{"frac": 0.0}], "in_progress": True},
             "truncated": '{"args": {"name": "trunc',
             "no_final": {"evals": []},
             "finished": {"final_val_loss": 2.5, "status": "finished"},
             "old_finished": {"final_val_loss": 2.5},                   # pre-0.3 JSON
             "diverged": {"status": "diverged", "diverged": True},
             "old_nan": '{"final_val_loss": NaN}'}                      # pre-0.3 divergence
    for n, obj in cases.items():
        if obj is not None:
            _write(str(root / "results" / "q" / f"{n}.json"), obj)
    got = {n: rq.run(cmd.format(n=n), 1)[1] for n in cases}
    assert got == {"fresh": "ok", "in_progress": "ok", "truncated": "ok", "no_final": "ok",
                   "finished": "skipped", "old_finished": "skipped",
                   "diverged": "skipped (diverged)", "old_nan": "skipped (diverged)"}
    assert all(c[-2:] == ["--threads", "1"] for c in calls)       # no --resume without a ckpt


def test_in_progress_pretrain_run_resumes_from_its_checkpoint(queue_root):
    root, calls, _ = queue_root
    out = root / "results" / "llm"
    _write(str(out / "a.json"), {"in_progress": True})
    _write(str(out / "a.ckpt"), "x")
    name, status, _ = rq.run("bantam pretrain --data D --budget 1 --name a --ckpt_every 50", 2)
    assert status == "ok (resumed)"
    assert calls[-1][-4:] == ["--threads", "2", "--resume", str(out / "a.ckpt")]
    # an explicit --ckpt path is used; weather runs resume too (bantam weather --resume, 0.3)
    _write(str(root / "ck" / "b.ckpt"), "x")
    _write(str(out / "b.json"), {"in_progress": True})
    rq.run("python -m bantamtrain pretrain --name b --ckpt ck/b.ckpt", 1)
    assert calls[-1][-2:] == ["--resume", str(root / "ck" / "b.ckpt")]
    _write(str(root / "results" / "weather" / "w.ckpt"), "x")
    rq.run("python3 weather/train.py --name w", 1)
    assert calls[-1][-2:] == ["--resume", str(root / "results" / "weather" / "w.ckpt")]


def test_result_path_cli_spellings(queue_root):
    root = queue_root[0]
    for cmd, sub in [("python3 llm/train.py --data D --name x", "llm"),
                     ("bantam pretrain --data D --name x", "llm"),
                     ("python3 -m bantamtrain pretrain --name x", "llm"),
                     ("python -m bantamtrain.cli.pretrain --name x", "llm"),
                     ("OMP_NUM_THREADS=1 python3 /abs/bantam/llm/train.py --name x", "llm"),
                     ("python3 weather/train.py --name x", "weather"),
                     ("bantam weather --name x --init results/pretrain.pt", "weather"),
                     ("python3 -m bantamtrain.cli.weather --name=x", "weather"),
                     ("torchrun --nproc_per_node 2 llm/train.py --name x", "llm"),
                     ("torchrun --nproc_per_node=2 -m bantamtrain pretrain --name x", "llm"),
                     ("bantam weather --init pretrain --name x", "weather")]:
        assert rq.result_path(cmd) == ("x", str(root / "results" / sub / "x.json")), cmd
    assert rq.result_path("bantam pretrain --name=foo --out=/abs/o")[1] == "/abs/o/foo.json"
    with pytest.raises(ValueError, match="no --name"):
        rq.result_path("bantam pretrain --data D")


def test_relative_out_is_resolved_against_the_repo_root(queue_root, tmp_path, monkeypatch):
    root, calls, _ = queue_root
    _write(str(root / "results" / "sweep" / "done.json"), {"final_val_loss": 2.0})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)              # e.g. `cd experiments && python run_queue.py ...`
    assert rq.run("python3 llm/train.py --name done --out results/sweep", 1)[1] == "skipped"
    assert calls == []


def test_substitution_uses_whole_names_and_logs_are_per_trainer(queue_root, tmp_path):
    root, calls, fake = queue_root
    q = tmp_path / "q.txt"
    q.write_text("# comment\nD=/data\nZ=2:0.5,1:0.5\nZOOM=4:0.35,2:0.25,1:0.4\n"
                 "python3 llm/train.py --data $D --zoom $ZOOM --name m\n"
                 "python3 llm/train.py --data ${D}/tok --zoom $Z --unknown $NOPE --name m2\n"
                 "python3 weather/train.py --data $D --name m\n")
    assert rq.parse_queue(str(q)) == [
        "python3 llm/train.py --data /data --zoom 4:0.35,2:0.25,1:0.4 --name m",
        "python3 llm/train.py --data /data/tok --zoom 2:0.5,1:0.5 --unknown $NOPE --name m2",
        "python3 weather/train.py --data /data --name m"]
    assert rq.main([str(q), "--threads", "1"]) == 0
    assert os.path.exists(root / "results" / "logs" / "llm" / "m.log")
    assert os.path.exists(root / "results" / "logs" / "weather" / "m.log")
    fake.rc = 1
    _write(str(root / "results" / "llm" / "m.json"), {"final_val_loss": 1.0})
    assert rq.main([str(q), "--threads", "1"]) == 1       # a failed command fails the queue
