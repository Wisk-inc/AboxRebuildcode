"""Run a queue of training commands with a fixed number of CPU workers.

    python experiments/run_queue.py QUEUE.txt --workers 2 --threads 2

QUEUE.txt holds one shell command per line (blank lines and # comments are
skipped). Each command gets `--threads N` appended, runs from the repo root
and logs to results/logs/<name>.log where <name> is taken from its --name
argument. Already-finished runs (results JSON present) are skipped, so a
queue can be re-run to fill in gaps.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def result_path(cmd):
    name = re.search(r"--name\s+(\S+)", cmd).group(1)
    out = re.search(r"--out\s+(\S+)", cmd)
    sub = "llm" if "llm/train.py" in cmd else "weather"
    out_dir = out.group(1) if out else os.path.join(ROOT, "results", sub)
    return name, os.path.join(out_dir, f"{name}.json")


def run(cmd, threads):
    name, res = result_path(cmd)
    if os.path.exists(res):
        return name, "skipped", 0.0
    log = os.path.join(ROOT, "results", "logs", f"{name}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(shlex.split(cmd) + ["--threads", str(threads)], cwd=ROOT,
                           stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    return name, "ok" if p.returncode == 0 else f"exit {p.returncode}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cmds = [l.strip() for l in open(args.queue) if l.strip() and not l.startswith("#")]
    print(f"{len(cmds)} commands, {args.workers} workers x {args.threads} threads", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for name, status, dt in ex.map(lambda c: run(c, args.threads), cmds):
            print(f"{time.strftime('%H:%M:%S')} {name}: {status} ({dt/60:.1f} min)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
