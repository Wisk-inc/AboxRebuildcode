"""Wave-3 core: bantam pretrain --ce_chunk, --recipe (pretrain and weather) and
--save_every_eval without --save."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import losses  # noqa: E402
from bantamtrain.cli import pretrain, recipes, weather  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from test_pretrain_cli import TINY, make_data  # noqa: E402

BEST = recipes.RECIPES["pretrain"]["best"]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_data(tmp_path_factory.mktemp("tok"))


@pytest.fixture
def run(tmp_path):
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    out = str(tmp_path / "out")

    def _run(data, *extra, name="r", budget="5e8"):
        argv = ["--data", data, "--budget", budget, "--name", name, "--out", out] + TINY
        return pretrain.main(argv + [str(a) for a in extra])
    _run.out = out
    _run.log = lambda name="r": json.load(open(os.path.join(out, f"{name}.json")))
    yield _run
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def parse(mod, argv):
    """args after --recipe, as main() resolves them."""
    args = mod.build_parser().parse_args(argv)
    note = recipes.apply_recipe(args, mod.build_parser, argv, mod.__name__.split(".")[-1])
    return args, note


REQ = ["--data", "d", "--budget", "1e9", "--name", "n"]


# -- --ce_chunk

def test_ce_chunk_gives_the_same_loss_without_full_logits(data, run, monkeypatch):
    """memory.py's scoreboard assumes chunked CE: --ce_chunk never builds (B, T, V) logits."""
    zoom = ["--zoom", "2:0.5,1:0.5"]                   # k=1 and slot-mode k=2 both chunk
    assert run(data, *zoom, name="full") == 0
    rows = []
    real = losses._chunk_loss
    monkeypatch.setattr(losses, "_chunk_loss", lambda h, *a: rows.append(h.shape[0]) or real(h, *a))

    def no_logits(self, h):
        raise AssertionError("full logits built")
    monkeypatch.setattr(ZoomGPT, "logits", no_logits)
    assert run(data, *zoom, "--ce_chunk", "24", name="chunk") == 0
    a, b = run.log("full"), run.log("chunk")
    assert rows and max(rows) <= 24
    assert a["args"]["ce_chunk"] == 0 and b["args"]["ce_chunk"] == 24
    assert b["final_val_loss"] == pytest.approx(a["final_val_loss"], rel=1e-5)
    assert [t["loss"] for t in b["train"]] == pytest.approx([t["loss"] for t in a["train"]], rel=1e-4)
    assert b["model_flops"] == a["model_flops"]             # the budget is unchanged
    assert b["model_flops_exact"] > a["model_flops_exact"]  # backward recomputes the head
    assert pretrain.build_model(b["args"], 64).ce_chunk == 24
    assert pretrain.build_model(a["args"], 64).ce_chunk == 0


def test_negative_ce_chunk_is_a_usage_error(data, run, capsys):
    with pytest.raises(SystemExit) as e:
        run(data, "--ce_chunk", "-1")
    assert e.value.code == 2 and "ce_chunk" in capsys.readouterr().err
    assert not os.path.exists(run.out)


# -- --save_every_eval

def test_save_every_eval_without_save_is_a_usage_error(data, run, capsys, tmp_path):
    with pytest.raises(SystemExit) as e:
        run(data, "--save_every_eval", "1")
    assert e.value.code == 2 and "--save_every_eval needs --save" in capsys.readouterr().err
    assert not os.path.exists(run.out)
    save = str(tmp_path / "m.pt")
    assert run(data, "--save_every_eval", "1", "--save", save) == 0
    assert any(e.get("checkpoint") for e in run.log()["evals"])


# -- --recipe

def test_recipe_none_changes_nothing():
    plain = vars(parse(pretrain, REQ)[0])
    none, note = parse(pretrain, REQ + ["--recipe", "none"])
    assert note is None and vars(none) == plain and plain["recipe_applied"] == []
    defaults = vars(pretrain.build_parser().parse_args(REQ))
    assert {k: v for k, v in plain.items() if k != "recipe_applied"} == defaults
    wx = vars(parse(weather, REQ)[0])
    assert {k: v for k, v in wx.items() if k != "recipe_applied"} == \
        vars(weather.build_parser().parse_args(REQ))


def test_recipe_best_sets_only_the_options_not_given():
    args, note = parse(pretrain, REQ + ["--recipe", "best"])
    assert {k: getattr(args, k) for k in BEST} == BEST and args.recipe_applied == list(BEST)
    assert note.startswith("--recipe best: --optim muon --lr 0.016")
    # explicit flags win, also abbreviated, as --opt=value and at their default value
    args, note = parse(pretrain, REQ + ["--recipe", "best", "--lr=3e-3", "--qk", "0",
                                        "--batch_scale", "0"])
    assert (args.lr, args.qk_norm, args.batch_scale) == (3e-3, 0, 0.0)
    assert (args.optim, args.zoom) == ("muon", "2:0.33,1:0.67")
    assert args.recipe_applied == ["optim", "zoom"]
    assert "from the command line: --lr --qk_norm --batch_scale" in note
    # the recipe's LR was measured with Muon: an explicit --optim adamw keeps the default LR
    args, note = parse(pretrain, REQ + ["--recipe", "best", "--optim", "adamw"])
    assert (args.optim, args.lr) == ("adamw", 1e-3) and "lr" not in args.recipe_applied
    assert "not set: --lr (measured with --optim muon)" in note
    args, _ = parse(pretrain, REQ + ["--recipe", "best", "--optim", "muon"])
    assert args.lr == 1.6e-2 and args.recipe_applied == ["lr", "qk_norm", "zoom", "batch_scale"]
    args, _ = parse(weather, REQ + ["--recipe", "best", "--zoom", "1"])
    assert (args.optim, args.lr, args.coarse_mode, args.zoom) == ("muon", 1.6e-2, "lowres", "1")
    assert args.recipe_applied == ["optim", "lr", "coarse_mode"]


def test_recipe_help_states_the_evidence(capsys):
    for mod in (pretrain, weather):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args(["--help"])
        text = " ".join(capsys.readouterr().out.split())
        assert "small scale on CPU" in text and "seed" in text
    assert "2.589" in pretrain.RECIPE_HELP and "1.41x" in pretrain.RECIPE_HELP
    assert "0.112" in weather.RECIPE_HELP and "0.124" in weather.RECIPE_HELP


def test_recipe_run_logs_resolved_values_and_resumes(data, run, tmp_path, capsys):
    best = ["--recipe", "best", "--lr", "4e-3"]
    assert run(data, *best, name="full") == 0
    full = run.log("full")
    a = full["args"]
    assert a["recipe"] == "best" and a["recipe_applied"] == ["optim", "qk_norm", "zoom",
                                                               "batch_scale"]
    assert (a["optim"], a["lr"], a["qk_norm"], a["zoom"], a["batch_scale"]) == \
        ("muon", 4e-3, 1, "2:0.33,1:0.67", 2.0)
    assert full["schedule"] == "2:0.33,1:0.67"
    assert "--recipe best: --optim muon --qk_norm 1" in capsys.readouterr().out
    ck = str(tmp_path / "p.ckpt")
    assert run(data, *best, "--stop_step", "5", "--ckpt_every", "5", "--ckpt", ck,
               name="part") == 0
    with pytest.raises(SystemExit) as e:          # without the recipe the schedule would change
        run(data, "--lr", "4e-3", "--resume", ck, "--ckpt", ck, name="part")
    assert e.value.code == 2 and "options differ" in capsys.readouterr().err
    assert run(data, *best, "--resume", ck, "--ckpt", ck, name="part") == 0
    part = run.log("part")
    assert part["steps"] == full["steps"] > 5 and part["final_val_loss"] == full["final_val_loss"]
    assert [t["loss"] for t in part["train"]] == [t["loss"] for t in full["train"]]
    assert part["args"]["recipe_applied"] == a["recipe_applied"]
    assert "recipe_applied" not in pretrain.resume_arg_changes(
        {**a, "recipe_applied": None}, part["args"], pretrain.build_parser().get_default)[1]


def test_weather_recipe_run_resumes(tmp_path):
    from test_weather_cli import make_era5, run as wrun, train
    d = str(tmp_path / "era5")
    make_era5(d)
    n, rng = torch.get_num_threads(), torch.get_rng_state()
    try:
        full = train(d, tmp_path, "full", "--recipe", "best", budget="1e9")
        assert full["args"]["recipe_applied"] == list(recipes.RECIPES["weather"]["best"])
        assert (full["args"]["coarse_mode"], full["schedule"]) == ("lowres", "4:0.6,1:0.4")
        ck = str(tmp_path / "w.ckpt")
        wrun(d, tmp_path, "part", "--recipe", "best", "--ckpt_every", "10", "--stop_step", "10",
             "--ckpt", ck, budget="1e9")
        part = train(d, tmp_path, "part", "--recipe", "best", "--resume", ck, "--ckpt", ck,
                     budget="1e9")
    finally:
        torch.set_num_threads(n)
        torch.set_rng_state(rng)
    assert part["steps"] == full["steps"] > 10 and part["final"] == full["final"]


def test_summarize_pools_recipe_runs_with_explicit_ones():
    """args.recipe / recipe_applied / ce_chunk do not change what is trained: a --recipe best
    run is the same recipe as the explicit command, and a new --recipe none baseline pools
    with logs that predate the flag."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "experiments"))
    import summarize
    old = {"args": {"data": "/x/tok4k", "zoom": "1", "budget": 1e13, "optim": "muon", "lr": 0.016},
           "final_val_loss": 2.7}
    new = {**old, "args": {**old["args"], "recipe": "none", "recipe_applied": [], "ce_chunk": 0}}
    assert summarize.recipe(new) == summarize.recipe(old)
    explicit = {**old, "args": {**old["args"], **BEST, "lr": 1.6e-2}}
    rec = {**old, "args": {**explicit["args"], "recipe": "best", "recipe_applied": list(BEST),
                           "ce_chunk": 2048}}
    assert summarize.recipe(rec) == summarize.recipe(explicit) != summarize.recipe(old)
    assert summarize.recipe(rec, reset=True) == summarize.recipe(old)
