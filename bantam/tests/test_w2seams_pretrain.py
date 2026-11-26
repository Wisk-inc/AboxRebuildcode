"""Wave-2 seams in bantam pretrain: block_size, exact FLOPs, save_pretrained with the
tokenizer, --merge_mode choices, stale packs / uint32 .bin, legacy checkpoint loading."""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import lowmem  # noqa: E402
from bantamtrain.cli import pretrain  # noqa: E402
from bantamtrain.data import tokpack  # noqa: E402
from bantamtrain.lowmem import QState  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from test_pretrain_cli import TINY, V, make_data  # noqa: E402


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


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_data(tmp_path_factory.mktemp("tok"))


def test_build_model_block_size_is_the_training_seq():
    """ZG-4: generate()'s default window is the context the model was trained at."""
    m = pretrain.build_model({"seq": 48, "d": 16, "layers": 1, "heads": 1}, V)
    assert m.block_size == 48 and m.config()["block_size"] == 48
    assert pretrain.build_model({"d": 16, "layers": 1, "heads": 1}, V).block_size == 256


def test_run_log_counts_exact_flops_and_resume_continues_them(data, run, tmp_path):
    """ZG-1 / PRE-13: the kernel-exact FLOPs are logged beside the budget FLOPs."""
    assert run(data) == 0
    log = run.log()
    m = ZoomGPT(V, 16, 1, 1)
    per = m.flops_per_step(4, 32, 1, exact=True)
    assert log["model_flops_exact"] == pytest.approx(log["steps"] * per, rel=1e-12)
    assert log["model_flops_exact"] > log["model_flops"]          # full n x n attention etc.
    assert all("model_flops_exact" in e for e in log["evals"])
    ck = str(tmp_path / "c.ckpt")
    assert run(data, "--stop_step", "3", "--ckpt_every", "3", "--ckpt", ck, name="s") == 0
    assert pretrain.load_ckpt(ck)["progress"]["model_flops_exact"] == pytest.approx(3 * per)
    assert run(data, "--resume", ck, "--ckpt", ck, name="s") == 0
    assert run.log("s")["model_flops_exact"] == pytest.approx(log["model_flops_exact"], rel=1e-12)


def test_save_pretrained_dir_has_the_tokenizer_eos_and_block_size(tmp_path, run):
    """C8: --save writes a directory `bantam generate` can load as is."""
    d = make_data(str(tmp_path / "tok"))
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write('{"version": "1.0"}')
    save = str(tmp_path / "m.pt")
    assert run(d, "--save", save) == 0
    out = run.log()["saved"]["pretrained"]
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["eos_id"] == 0 and cfg["block_size"] == 32
    assert open(os.path.join(out, "tokenizer.json")).read() == '{"version": "1.0"}'
    m = ZoomGPT.from_pretrained(out)
    assert m.block_size == 32


@pytest.mark.parametrize("cli", ["pretrain", "weather"])
def test_merge_mode_typo_is_a_usage_error(cli, capsys):
    """ZG-7: --merge_mode has choices (a typo used to train the mean ablation)."""
    from bantamtrain.cli import weather
    parse = pretrain.main if cli == "pretrain" else weather.main
    with pytest.raises(SystemExit) as e:
        parse(["--data", "x", "--budget", "1", "--name", "n", "--merge_mode", "gated"])
    assert e.value.code == 2 and "invalid choice: 'gated'" in capsys.readouterr().err


def test_stale_tokpack_falls_back_to_the_bin(tmp_path):
    """DATA-X1: a pack from an earlier prepare run is skipped (auto) or refused (tokpack)."""
    d = make_data(str(tmp_path))
    meta = json.load(open(os.path.join(d, "meta.json")))
    json.dump({**meta, "prep_id": "new"}, open(os.path.join(d, "meta.json"), "w"))
    bin_ = np.fromfile(os.path.join(d, "train.bin"), dtype=np.uint16)
    tokpack.write(os.path.join(d, "train.tokpack"), bin_[:5000], V, chunk=1024,
                  extra={"source": {"prep_id": "old"}})
    with pytest.warns(UserWarning, match="prep_id"):
        toks, info = pretrain.open_token_source(d, "train")
    assert info["format"] == "bin" and len(toks) == len(bin_)
    with pytest.raises(tokpack.StalePackError):
        pretrain.open_token_source(d, "train", "tokpack")


def test_bin_is_read_with_the_meta_dtype(tmp_path):
    """DATA-7: vocabularies > 65536 are stored as uint32 (meta.json dtype)."""
    d = str(tmp_path)
    toks = np.array([0, 70000, 5, 65536, 1], dtype=np.uint32)
    toks.tofile(os.path.join(d, "val.bin"))
    json.dump({"vocab_size": 70001, "dtype": "uint32"}, open(os.path.join(d, "meta.json"), "w"))
    got, info = pretrain.open_token_source(d, "val")
    assert info == {"format": "bin", "file": os.path.join(d, "val.bin"), "tokens": 5}
    assert np.asarray(got).tolist() == toks.tolist()


def _old_qstate_ckpt(path):
    torch.save({"opt": [{"state": {0: {"buf": QState(torch.zeros(4, 4))}}}]}, path)


def test_legacy_checkpoint_on_torch_without_safe_globals(tmp_path, monkeypatch, capsys):
    """torch 2.4 has add_safe_globals but not the safe_globals context manager."""
    path = str(tmp_path / "old.ckpt")
    _old_qstate_ckpt(path)
    if not hasattr(torch.serialization, "clear_safe_globals"):
        pytest.skip("needs torch >= 2.5 to restore the allow-list afterwards")
    added = []
    monkeypatch.delattr(torch.serialization, "safe_globals")
    real_add = torch.serialization.add_safe_globals
    before = torch.serialization.get_safe_globals()
    monkeypatch.setattr(torch.serialization, "add_safe_globals",
                        lambda g: added.append(g) or real_add(g))
    try:
        ck = pretrain.load_ckpt(path)
    finally:
        torch.serialization.clear_safe_globals()
        real_add(before)
    assert isinstance(ck["opt"][0]["state"][0]["buf"], QState) and QState in added[0]
    assert "old checkpoint format" in capsys.readouterr().err


def test_pre_rename_zoomtrain_checkpoint_loads(tmp_path, monkeypatch):
    """OPT-3 (3): 8-bit state pickled as zoomtrain.lowmem.QState before the rename."""
    x = torch.linspace(-1, 1, 512).view(16, 32)
    src = QState(x)
    src.set(x)
    mod =types.ModuleType("zoomtrain.lowmem")
    Old = type("QState", (), {"__module__": "zoomtrain.lowmem"})
    mod.QState = Old
    monkeypatch.setitem(sys.modules, "zoomtrain", types.ModuleType("zoomtrain"))
    monkeypatch.setitem(sys.modules, "zoomtrain.lowmem", mod)
    old = Old.__new__(Old)
    old.__dict__.update(vars(src))
    path = str(tmp_path / "zt.ckpt")
    torch.save({"opt": {"state": {0: {"buf": old}}}}, path)
    monkeypatch.delitem(sys.modules, "zoomtrain.lowmem")
    monkeypatch.delitem(sys.modules, "zoomtrain")
    if not torch.__version__ >= (2, 6):
        pytest.skip("renamed safe globals need torch >= 2.6")
    ck, legacy = lowmem.load_checkpoint(path)
    buf = ck["opt"]["state"][0]["buf"]
    assert legacy and isinstance(buf, QState)
    assert torch.equal(buf.get(), src.get())
