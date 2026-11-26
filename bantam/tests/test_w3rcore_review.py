"""Wave-3 core review: --recipe never overrides an explicit --batch_scale_k, and a missing
--resume file stays a usage error on a torch without the allow-list APIs."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import lowmem  # noqa: E402
from bantamtrain.cli import pretrain, recipes  # noqa: E402
from test_pretrain_cli import TINY, make_data  # noqa: E402

REQ = ["--data", "d", "--budget", "1e9", "--name", "n"]


def parse(argv):
    args = pretrain.build_parser().parse_args(argv)
    return args, recipes.apply_recipe(args, pretrain.build_parser, argv, "pretrain")


def test_recipe_keeps_an_explicit_batch_scale_k():
    """--batch_scale > 0 overrides --batch_scale_k: the recipe's batch_scale silently
    replaced an explicit --batch_scale_k 1 (batch x 2 instead of x 4 at k=4)."""
    args, note = parse(REQ + ["--recipe", "best", "--zoom", "4:0.5,1:0.5", "--batch_scale_k", "1"])
    assert (args.batch_scale, args.batch_scale_k) == (0.0, 1)
    assert args.recipe_applied == ["optim", "lr", "qk_norm"]
    assert "not set: --batch_scale (it would override --batch_scale_k 1)" in note
    assert pretrain.stage_batch(args.batch, 4, args.batch_scale, args.batch_scale_k) == \
        4 * args.batch
    args, _ = parse(REQ + ["--recipe", "best", "--batch_scale_k", "0"])      # off: no conflict
    assert args.batch_scale == 2.0 and "batch_scale" in args.recipe_applied


def _without_allow_list_api(mp):
    mp.delattr(torch.serialization, "safe_globals", raising=False)
    mp.delattr(torch.serialization, "add_safe_globals", raising=False)


def test_missing_resume_file_without_allow_list_api_is_a_usage_error(tmp_path, monkeypatch,
                                                                     capsys):
    """On a torch with neither safe_globals nor add_safe_globals a missing file became the
    'needs torch >= 2.4' RuntimeError (a traceback instead of the --resume usage error)."""
    with monkeypatch.context() as mp:
        _without_allow_list_api(mp)
        with pytest.raises(FileNotFoundError):
            lowmem.load_checkpoint(str(tmp_path / "missing.ckpt"))

    def load(path, **kw):        # only around the load: torch's lazy imports need the APIs
        with monkeypatch.context() as mp:
            _without_allow_list_api(mp)
            return lowmem.load_checkpoint(path, **kw)
    monkeypatch.setattr(pretrain, "load_checkpoint", load)
    data = make_data(str(tmp_path / "tok"))
    n, rng = torch.get_num_threads(), torch.get_rng_state()
    try:
        with pytest.raises(SystemExit) as e:
            pretrain.main(["--data", data, "--budget", "5e8", "--name", "r", "--out",
                           str(tmp_path / "out"), "--resume", str(tmp_path / "missing.ckpt")]
                          + TINY)
    finally:
        torch.set_num_threads(n)
        torch.set_rng_state(rng)
    assert e.value.code == 2 and "--resume" in capsys.readouterr().err
