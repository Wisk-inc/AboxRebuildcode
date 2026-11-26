"""`bantam generate`: sample text from a save_pretrained directory."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomGPT  # noqa: E402


def _tiny_tokenizer(path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<|eos|>"], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["Once upon a time there was a cat.", "The dog ran home."] * 20,
                            trainer=trainer)
    tok.save(str(path))
    return tok


def test_cli_generate(tmp_path, capsys):
    pytest.importorskip("tokenizers")
    from bantamtrain.cli.generate import main
    data = tmp_path / "data"
    data.mkdir()
    tok = _tiny_tokenizer(data / "tokenizer.json")
    (data / "meta.json").write_text(json.dumps({"vocab_size": tok.get_vocab_size(),
                                                "eos_id": tok.token_to_id("<|eos|>")}))
    torch.manual_seed(0)
    m = ZoomGPT(tok.get_vocab_size(), 32, 1, 2, zoom_ks=[2], block_size=32)
    d = m.save_pretrained(str(tmp_path / "model"), tokenizer_path=str(data))
    assert main(["--model", d, "--prompt", "Once upon", "--max_new", "5",
                 "--temperature", "0"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Once upon")
    eos = tok.token_to_id("<|eos|>")
    ids = [eos] + tok.encode("Once upon").ids
    new = m.generate([ids], 5, temperature=0, eos_id=eos)[0][len(ids):]
    assert out.rstrip("\n") == "Once upon" + tok.decode([t for t in new if t != eos])
    assert main(["--model", d, "--tokenizer", str(data / "tokenizer.json"), "--prompt", "The",
                 "--max_new", "4", "--num_samples", "2", "--top_p", "0.9", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("-" * 40) == 1 and all(o.startswith("The") for o in out.split("-" * 40 + "\n"))
