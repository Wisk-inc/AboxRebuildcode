"""Run a queue of training commands with a fixed number of CPU workers.

    python experiments/run_queue.py QUEUE.txt --workers 2 --threads 2 [--dry_run]

QUEUE.txt holds one shell command per line (blank lines and # comments are
skipped; NAME=value lines define $NAME / ${NAME} substitutions for later lines,
and a reference is the whole identifier, so $ZOOM never reads as $Z + "OOM").
Each command gets `--threads N` appended, runs from the repo root (a relative
--out is resolved there too) and logs to results/logs/<llm|weather>/<name>.log,
where <name> is taken from its --name argument.

Finished runs (results JSON without "in_progress" and with "final_val_loss" or
"final") are skipped, so a queue can be re-run to fill in gaps; diverged runs
are reported and not re-run. Unfinished runs (killed, crashed, in progress, or a
truncated JSON) run again and resume from their checkpoint (<out>/<name>.ckpt or
--ckpt, written with --ckpt_every) when one exists; a weather run writes no JSON
before it finishes, so a missing JSON next to its checkpoint resumes too.
--dry_run prints what each command would do (skip, run or resume) and runs
nothing. The exit status is 1 if any command failed.
"""
import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM = re.compile(r"(^|/)llm/train\.py$|(^|\.)cli\.pretrain$")
VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _opt(words, flag):
    """Value of `--flag value` or `--flag=value` (the last one wins), else None."""
    val = None
    for i, w in enumerate(words):
        if w == flag and i + 1 < len(words):
            val = words[i + 1]
        elif w.startswith(flag + "="):
            val = w[len(flag) + 1:]
    return val


def trainer(words):
    """'llm' for llm/train.py, `bantam pretrain`, `python -m bantamtrain[.cli.pretrain]` (also
    behind torchrun and its options); 'weather' otherwise (weather/train.py, `bantam weather`;
    the old default). A `pretrain` word counts only as the command after bantam/bantamtrain."""
    for prev, w in zip([""] + words, words):
        if LLM.search(w) or (w == "pretrain" and os.path.basename(prev) in ("bantam", "bantamtrain")):
            return "llm"
    return "weather"


def run_info(cmd):
    """name, trainer ('llm'|'weather'), results JSON and resumable-checkpoint paths of a command."""
    words = shlex.split(cmd)
    name = _opt(words, "--name")
    if not name:
        raise ValueError(f"no --name in: {cmd}")
    sub = trainer(words)
    out = _opt(words, "--out") or os.path.join("results", sub)
    out = out if os.path.isabs(out) else os.path.join(ROOT, out)   # the command runs from ROOT
    ckpt = _opt(words, "--ckpt") or os.path.join(out, f"{name}.ckpt")
    ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    return {"name": name, "sub": sub, "words": words, "json": os.path.join(out, f"{name}.json"),
            "ckpt": ckpt}


def result_path(cmd):
    info = run_info(cmd)
    return info["name"], info["json"]


def run_state(path):
    """'missing' | 'unfinished' | 'diverged' | 'finished' for a results JSON."""
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        return "missing"
    except (OSError, ValueError):
        return "unfinished"                       # truncated or unreadable
    if not isinstance(d, dict):
        return "unfinished"
    if d.get("status") == "diverged" or d.get("diverged"):
        return "diverged"
    if d.get("in_progress"):
        return "unfinished"
    final = d.get("final_val_loss", d.get("final"))
    if final is None:
        return "unfinished"
    if isinstance(final, float) and not math.isfinite(final):
        return "diverged"                         # NaN final from trainers before 0.3
    return "finished"


def plan(cmd, threads):
    """(run_info, state, argv, resume) of a command; argv is None when it is skipped."""
    info = run_info(cmd)
    state = run_state(info["json"])
    if state in ("finished", "diverged"):
        return info, state, None, False
    argv = info["words"] + ["--threads", str(threads)]
    resumable = ("unfinished",) if info["sub"] == "llm" else ("unfinished", "missing")
    resume = (state in resumable and os.path.exists(info["ckpt"])
              and _opt(info["words"], "--resume") is None)
    if resume:
        argv += ["--resume", info["ckpt"]]
    return info, state, argv, resume


def run(cmd, threads):
    info, state, argv, resume = plan(cmd, threads)
    name = info["name"]
    if argv is None:
        return name, "skipped" if state == "finished" else "skipped (diverged)", 0.0
    log = os.path.join(ROOT, "results", "logs", info["sub"], f"{name}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "a" if resume else "w") as f:
        p = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
    return name, status + (" (resumed)" if resume else ""), time.time() - t0


def substitute(line, env):
    """$NAME / ${NAME} -> its value; names not defined stay as written."""
    return VAR.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), line)


def parse_queue(path):
    cmds, env = [], {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Z_][A-Z0-9_]*)=(\S+)", line)
        if m:
            env[m.group(1)] = substitute(m.group(2), env)
            continue
        cmds.append(substitute(line, env))
    return cmds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args(argv)
    cmds = parse_queue(args.queue)
    for c in cmds:
        run_info(c)                               # fail fast on a line without --name
    if args.dry_run:
        for c in cmds:
            info, state, argv, resume = plan(c, args.threads)
            what = "skip" if argv is None else "resume" if resume else "run"
            print(f"{info['name']}: {state} -> {what}"
                  + (f": {shlex.join(argv)}" if argv else ""), flush=True)
        return 0
    print(f"{len(cmds)} commands, {args.workers} workers x {args.threads} threads", flush=True)
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for name, status, dt in ex.map(lambda c: run(c, args.threads), cmds):
            failed += status.startswith("exit")
            print(f"{time.strftime('%H:%M:%S')} {name}: {status} ({dt/60:.1f} min)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
