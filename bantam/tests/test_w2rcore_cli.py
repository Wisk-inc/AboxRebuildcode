"""Wave-2 core review: bad pretrain/generate arguments fail cleanly (exit 2, nothing written)
and an unknown precision is an error, not a silent fp32 run."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.cli import generate, pretrain  # noqa: E402
from bantamtrain.device import setup_runtime  # noqa: E402

TINY = ["--d", "16", "--layers", "1", "--heads", "2", "--seq", "32", "--batch", "4",
        "--threads", "1", "--evals", "2", "--eval_seqs", "8", "--final_eval_seqs", "8",
        "--budget", "5e8", "--name", "r"]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    rng = np.random.default_rng(0)
    rng.integers(0, 64, 20_000).astype(np.uint16).tofile(os.path.join(d, "train.bin"))
    rng.integers(0, 64, 2_000).astype(np.uint16).tofile(os.path.join(d, "val.bin"))
    json.dump({"vocab_size": 64, "eos_id": 0}, open(os.path.join(d, "meta.json"), "w"))
    return str(d)


@pytest.fixture
def run(tmp_path):
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    out = str(tmp_path / "out")

    def _run(data, *extra):
        return pretrain.main(["--data", data, "--out", out] + TINY + list(extra))
    _run.out = out
    yield _run
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


@pytest.mark.parametrize("extra,msg", [
    (["--d", "18", "--heads", "2"], "n_head=2"),                   # odd head_dim (RoPE)
    (["--heads", "3"], "n_head=3"),                                # d % heads != 0
    (["--zoom", "2:0.5,1:0.5", "--merge_mode", "gated"], "merge_mode"),
    (["--softcap", "-1"], "softcap"),
    (["--precision", "bf16-mixed"], "--precision"),
    (["--resume", "no_such.ckpt"], "--resume"),
])
def test_bad_model_options_exit_2_before_any_file(data, run, capsys, extra, msg):
    with pytest.raises(SystemExit) as e:
        run(data, *extra)
    assert e.value.code == 2
    assert msg in capsys.readouterr().err
    assert not os.path.exists(run.out)


def test_valid_options_still_train(data, run):
    assert run(data, "--zoom", "2:0.5,1:0.5", "--merge_mode", "linear", "--stop_step", "2") == 0


def test_resume_refuses_a_changed_world_size(data, run, capsys):
    """A checkpoint of a 2-rank run resumed on 1 rank would halve the global batch."""
    assert run(data, "--stop_step", "2", "--ckpt_every", "1") == 0
    path = os.path.join(run.out, "r.ckpt")
    ck = torch.load(path, weights_only=True)
    assert ck["log"]["world_size"] == 1
    ck["log"]["world_size"] = 2
    torch.save(ck, path)
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        run(data, "--resume", path)
    assert e.value.code == 2 and "WORLD_SIZE (2 -> 1)" in capsys.readouterr().err
    assert run(data, "--resume", path, "--allow_arg_change", "--stop_step", "3") == 0


@pytest.mark.parametrize("prec", ["bf16-mixed", "BF16", "fp8", "32"])
def test_setup_runtime_rejects_unknown_precision(prec):
    with pytest.raises(ValueError, match="precision"):
        setup_runtime("cpu", prec, False, None, 0)
    assert setup_runtime("cpu", "fp32", False, None, 0).amp_dtype is None


def test_generate_rejects_a_state_dict_or_plain_dir(tmp_path, capsys):
    sd = tmp_path / "model.pt"
    torch.save({"emb.weight": torch.zeros(4, 4)}, sd)
    for model in (str(sd), str(tmp_path)):
        with pytest.raises(SystemExit) as e:
            generate.main(["--model", model, "--prompt", "hi"])
        assert e.value.code == 2
        assert "save_pretrained" in capsys.readouterr().err
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        generate.main(["--model", str(tmp_path), "--prompt", "hi", "--num_samples", "0"])
    assert e.value.code == 2 and "--num_samples" in capsys.readouterr().err
