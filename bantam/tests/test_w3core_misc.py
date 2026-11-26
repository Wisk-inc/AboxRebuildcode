"""Wave-3 core: generate encodes the prompt like prepare-text, from_pretrained follows
config.json's weights_file, legacy checkpoints on a torch without the allow-list APIs,
and user-visible strings that contradicted the docs."""
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bantamtrain  # noqa: E402
from bantamtrain import cli, lowmem, memory  # noqa: E402
from bantamtrain.lowmem import QState  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bos_tokenizer(path):
    """A byte-level BPE whose post-processor adds BOS, like Llama-family tokenizer.json files."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<|eos|>", "<s>"],
                                  show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["Once upon a time there was a cat.", "The dog ran home."] * 20,
                            trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", tok.token_to_id("<s>"))])
    tok.save(str(path))
    return tok


def test_generate_encodes_the_prompt_without_special_tokens(tmp_path, monkeypatch):
    """prepare-text stores documents without BOS (and a literal "<|eos|>" as text)."""
    pytest.importorskip("tokenizers")
    from bantamtrain.cli import generate
    data = tmp_path / "data"
    data.mkdir()
    tok = _bos_tokenizer(data / "tokenizer.json")
    eos, bos = tok.token_to_id("<|eos|>"), tok.token_to_id("<s>")
    assert tok.encode("Once").ids[0] == bos                      # the post-processor adds BOS
    (data / "meta.json").write_text(json.dumps({"vocab_size": tok.get_vocab_size(),
                                                "eos_id": eos}))
    d = ZoomGPT(tok.get_vocab_size(), 16, 1, 2, block_size=32).save_pretrained(
        str(tmp_path / "m"), tokenizer_path=str(data))
    seen = []
    monkeypatch.setattr(ZoomGPT, "generate", lambda self, rows, *a, **k: seen.append(rows[0])
                        or [list(r) for r in rows])
    for prompt in ("Once upon a time", "a cat<|eos|>The dog"):
        assert generate.main(["--model", d, "--prompt", prompt, "--max_new", "2"]) == 0
        ids = seen.pop()
        assert ids[0] == eos and bos not in ids and eos not in ids[1:]
        assert tok.decode(ids[1:]) == prompt
    assert generate.main(["--model", d, "--prompt", "The", "--doc_start", "0"]) == 0
    assert seen.pop() == tok.encode("The", add_special_tokens=False).ids


def test_from_pretrained_follows_weights_file(tmp_path, monkeypatch):
    """A directory saved with safetensors, then re-saved where it is missing (model.pt),
    loaded the stale model.safetensors."""
    pytest.importorskip("safetensors")
    d = str(tmp_path / "m")
    torch.manual_seed(0)
    ZoomGPT(32, 16, 1, 2).save_pretrained(d)
    assert json.load(open(os.path.join(d, "config.json")))["weights_file"] == "model.safetensors"
    torch.manual_seed(1)
    new = ZoomGPT(32, 16, 1, 2)
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "safetensors.torch", None)       # not installed
        new.save_pretrained(d)
    assert json.load(open(os.path.join(d, "config.json")))["weights_file"] == "model.pt"
    assert os.path.exists(os.path.join(d, "model.safetensors"))  # left alone, not deleted
    got = ZoomGPT.from_pretrained(d)
    assert all(torch.equal(a, b) for a, b in zip(got.state_dict().values(),
                                                 new.state_dict().values()))
    # directories written before weights_file: model.safetensors when present
    cfg = json.load(open(os.path.join(d, "config.json")))
    cfg.pop("weights_file")
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    assert not torch.equal(ZoomGPT.from_pretrained(d).emb.weight, new.emb.weight)


def test_legacy_checkpoint_without_any_allow_list_api_names_the_torch_version(tmp_path,
                                                                              monkeypatch):
    """torch < 2.4 has neither safe_globals nor add_safe_globals: a clear error, not the
    bare unpickling error."""
    old = str(tmp_path / "old.ckpt")
    torch.save({"opt": {"state": {0: {"buf": QState(torch.zeros(4, 4))}}}}, old)
    plain = str(tmp_path / "plain.ckpt")
    torch.save({"model": {"w": torch.ones(2)}}, plain)
    monkeypatch.delattr(torch.serialization, "safe_globals", raising=False)
    monkeypatch.delattr(torch.serialization, "add_safe_globals", raising=False)
    with pytest.raises(RuntimeError, match=r"needs torch >= 2\.4 .*this is torch "
                       + re.escape(torch.__version__)):
        lowmem.load_checkpoint(old)
    ck, legacy = lowmem.load_checkpoint(plain)                   # 0.3 checkpoints load anyway
    assert not legacy and torch.equal(ck["model"]["w"], torch.ones(2))


def test_user_visible_strings_match_the_docs():
    with redirect_stdout(io.StringIO()) as f:
        assert cli.main(["--help"]) == 0
    assert "uint16 token files" not in f.getvalue() and "uint32" in f.getvalue()
    rows = [name for name, _ in memory.storage_table(5e9, 1e11)]
    assert not any("lossless" in r for r in rows if "WeightPack 16" in r)
    smoke = open(os.path.join(ROOT, "experiments", "gpu_smoke.sh")).read()
    assert "4:0.35,2:0.25,1:0.4" not in smoke and "--recipe best" in smoke
    assert "more steps per wall-clock second" not in smoke
    assert bantamtrain.__version__ == "0.3.0"
    toml = open(os.path.join(ROOT, "pyproject.toml")).read()
    assert "faster and cheaper" not in toml
