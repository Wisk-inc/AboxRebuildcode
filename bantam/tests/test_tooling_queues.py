"""Every command in experiments/queues/* and gpu_smoke.sh parses with the current CLIs."""
import argparse
import glob
import importlib
import inspect
import os
import re
import shlex
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SCRIPTS = {"llm/train.py": "bantamtrain.cli.pretrain", "weather/train.py": "bantamtrain.cli.weather"}
BANTAM = {"pretrain": "bantamtrain.cli.pretrain", "weather": "bantamtrain.cli.weather",
          "finetune": "bantamtrain.finetune"}


class _Parsed(Exception):
    pass


def queue_commands():
    """(file, command tokens) for every command line, with NAME=value substitutions."""
    out = []
    files = sorted(glob.glob(os.path.join(ROOT, "experiments", "queues", "*")))
    for q in files + [os.path.join(ROOT, "experiments", "gpu_smoke.sh")]:
        env = {}
        for line in open(q).read().replace("\\\n", " ").splitlines():
            line = line.strip()
            m = re.fullmatch(r"([A-Z_][A-Z0-9_]*)=(?:\$\{\1:-)?([^}\s]+)\}?", line)
            if m:
                env[m.group(1)] = m.group(2)
                continue
            if not line or line.startswith("#"):
                continue
            for k, v in env.items():
                line = line.replace("${" + k + "}", v).replace("$" + k, v)
            out.append((os.path.relpath(q, ROOT), shlex.split(line)))
    return out


def target(tokens):
    for i, t in enumerate(tokens):
        if t in SCRIPTS:
            return SCRIPTS[t], tokens[i + 1:]
        if t in ("bantam", "bantamtrain") and tokens[i + 1:i + 2] and tokens[i + 1] in BANTAM:
            return BANTAM[tokens[i + 1]], tokens[i + 2:]      # bantam X / python -m bantamtrain X
    return None, None


CMDS = [(q, t) for q, t in queue_commands() if target(t)[0]]


def test_queues_have_commands():
    assert len(CMDS) > 50
    for q, tokens in queue_commands():           # shell drivers point at queue files that exist
        for t in tokens:
            if t.startswith("experiments/queues/"):
                assert os.path.exists(os.path.join(ROOT, t)), (q, t)


@pytest.mark.parametrize("q,tokens", CMDS, ids=[f"{q}:{i}" for i, (q, _) in enumerate(CMDS)])
def test_queue_line_parses(q, tokens, monkeypatch):
    modname, args = target(tokens)
    real = argparse.ArgumentParser.parse_known_args

    def parse_known_args(self, a=None, namespace=None):
        ns, extra = real(self, a, namespace)
        assert not extra, f"{q}: unknown arguments {extra}"
        raise _Parsed(ns)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", parse_known_args)
    monkeypatch.setattr(sys, "argv", ["train.py"] + args)
    mod = importlib.import_module(modname)
    takes_argv = bool(inspect.signature(mod.main).parameters)
    with pytest.raises(_Parsed):
        mod.main(args) if takes_argv else mod.main()
