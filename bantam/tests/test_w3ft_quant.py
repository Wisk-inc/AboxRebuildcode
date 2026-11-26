"""Wave-3 quantization fixes: bantam chat quantizes the base as the adapter was trained
(--quant and --embed_quant from adapter_meta.json / log.json), and WeightPack keeps an
already-quantized (qweights) model's codes and scales exactly."""
import json
import os
import shutil

import pytest
import torch
from test_finetune_fixtures import rows, run_args, save_model_dir, write_jsonl
from torch import nn

from bantamtrain import finetune as ft
from bantamtrain import qweights, weightpack


# ----------------------------------------------------------------------------- chat
@pytest.fixture(scope="module")
def qlora_run(tmp_path_factory):
    """A run trained over an int8 base with float embeddings (--embed_quant none)."""
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("qchat")
    d = save_model_dir(root / "m", "chatml", big=True)
    out = root / "out"
    ft.main(run_args(d, write_jsonl(root / "d.jsonl", rows(24)), out, "--quant", 8, "--embed_quant", "none",
                     "--max_steps", 1))
    return d, str(out)


def _quant_calls(monkeypatch):
    calls, real = [], qweights.quantize_model
    monkeypatch.setattr(qweights, "quantize_model",
                        lambda m, bits, **k: calls.append((bits, k.get("embed_bits"))) or real(m, bits, **k))
    return calls


def test_adapter_meta_records_the_base_quantization(qlora_run):
    with open(os.path.join(qlora_run[1], "adapter_meta.json")) as f:
        meta = json.load(f)
    assert meta["quant"] == "8" and meta["embed_quant"] == "none"


def test_chat_quantizes_the_base_as_in_training(qlora_run, monkeypatch, capsys):
    """chat used float weights unless --quant was given, and then always int8 embeddings,
    whatever --embed_quant the adapter was trained with."""
    from bantamtrain.cli import chat
    d, out = qlora_run
    calls = _quant_calls(monkeypatch)
    base = ["--model", d, "--adapter", out, "--prompt", "hi", "--max_new", "2", "--temperature", "0"]
    for extra, want in (([], (8, None)), (["--embed_quant", "8"], (8, 8)), (["--quant", "nf4"], ("nf4", None)),
                        (["--quant", "4", "--embed_quant", "same"], (4, "same")), (["--quant", "0"], None)):
        calls.clear()
        assert chat.main(base + extra) == 0
        assert calls == ([want] if want else [])
    assert "base quantized as in training: --quant 8 --embed_quant none" in capsys.readouterr().err
    assert chat.trained_quant(os.path.join(out, "adapter.pt")) == ("8", "none")


def test_chat_reads_older_runs_and_unknown_adapters(qlora_run, tmp_path, monkeypatch):
    from bantamtrain.cli import chat
    d, out = qlora_run
    old = tmp_path / "old"                            # adapter_meta.json without embed_quant
    shutil.copytree(out, old)
    with open(old / "adapter_meta.json") as f:
        meta = json.load(f)
    meta.pop("embed_quant")
    with open(old / "adapter_meta.json", "w") as f:
        json.dump(meta, f)
    assert chat.trained_quant(str(old)) == ("8", "none")              # from log.json's args
    os.remove(old / "log.json")
    assert chat.trained_quant(str(old)) == ("8", "8")                 # finetune's default
    assert chat.trained_quant("") == (None, None) and chat.trained_quant(str(tmp_path / "nothing")) == (None, None)
    calls = _quant_calls(monkeypatch)
    assert chat.main(["--model", d, "--prompt", "hi", "--max_new", "2"]) == 0 and calls == []    # no adapter


# ----------------------------------------------------------------------------- weightpack
def _qmodel():
    torch.manual_seed(0)
    m = nn.ModuleDict({"emb": nn.Embedding(4096, 256), "lin": nn.Linear(256, 512), "head": nn.Linear(64, 128)})
    qweights.quantize_model(m, 8, embed_bits=8, skip_names=("head",))
    return m


@pytest.mark.parametrize("bits", [8, 4, 16])
def test_weightpack_keeps_qweights_codes_and_scales_exact(bits, tmp_path):
    """save(bits=8/4) re-quantized the fp16 scale matrices of a qweights-quantized model
    whenever they had >= 4096 elements (the embedding's: 4096 rows x 2 blocks here; SmolLM2's
    49152 x 5), so the loaded model was not the saved one."""
    m = _qmodel()
    sd = m.state_dict()
    assert sd["emb.s"].numel() >= 4096 and sd["emb.s"].dtype == torch.float16
    weightpack.save(tmp_path / "m.wpk", sd, bits=bits)
    back = weightpack.load(tmp_path / "m.wpk")
    for k in ("emb.q", "emb.s", "lin.q", "lin.s") + (("lin.bias",) if bits != 16 else ()):   # 16: bf16 biases
        assert back[k].dtype == sd[k].dtype and torch.equal(back[k], sd[k]), k
    w = sd["head.weight"]                           # an ordinary float matrix: still quantized
    assert (bits == 16 or not torch.equal(back["head.weight"], w)) and torch.allclose(back["head.weight"], w, atol=0.02)
    m2 = _qmodel()
    m2.load_state_dict(back)
    idx = torch.randint(0, 4096, (3, 5))
    assert torch.equal(m2["emb"](idx), m["emb"](idx))
    assert weightpack.qweights_scales(sd) == {"emb.s", "lin.s"}


def test_qweights_scales_needs_integer_codes_beside_it():
    s = torch.randn(64, 80).half()
    assert weightpack.qweights_scales({"a.s": s}) == set()
    assert weightpack.qweights_scales({"a.s": s, "a.q": torch.randn(64, 80)}) == set()          # float q
    assert weightpack.qweights_scales({"a.s": s, "a.q": torch.zeros(32, 80, dtype=torch.int8)}) == set()
    assert weightpack.qweights_scales({"s": s, "q": torch.zeros(64, 40, dtype=torch.uint8)}) == {"s"}
