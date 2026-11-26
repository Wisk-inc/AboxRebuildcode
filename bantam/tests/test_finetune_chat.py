"""bantam chat: adapters, prompt rendering identical to training, streaming, REPL."""
import io
import json
import os
import shutil
import sys

import pytest
import torch
from test_finetune_fixtures import make_tokenizer, rows, run_args, save_model_dir, write_jsonl

from bantamtrain import finetune as ft
from bantamtrain import lora


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """(model dir, finetune output dir) after a few LoRA steps."""
    root = tmp_path_factory.mktemp("chat")
    d = save_model_dir(root / "m", "chatml")
    out = root / "out"
    ft.main(run_args(d, write_jsonl(root / "d.jsonl", rows(24)), out, "--max_steps", 3, "--lr", 1e-2))
    return d, str(out)


def _reference_reply(d, out, text, max_new):
    """Base + adapter built by hand, prompt as in training, greedy decode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out)
    m = AutoModelForCausalLM.from_pretrained(d).eval()
    with open(os.path.join(out, "adapter_meta.json")) as f:
        meta = json.load(f)
    lora.apply_lora(m, meta["r"], meta["alpha"], 0.0, tuple(meta["modules"]), rslora=meta["rslora"])
    lora.load_lora_state_dict(m, torch.load(os.path.join(out, "adapter.pt"), weights_only=True))
    ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    x = torch.tensor([tok(ids, add_special_tokens=False)["input_ids"]])
    ys = []
    with torch.no_grad():
        for _ in range(max_new):
            nxt = int(m(input_ids=x).logits[0, -1].argmax())
            if nxt in (tok.eos_token_id, m.generation_config.eos_token_id):
                break
            ys.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return tok.decode(ys, skip_special_tokens=True)


def test_chat_prompt_with_a_bantam_adapter_streams_the_greedy_reply(trained, capsys):
    from bantamtrain.cli import chat
    d, out = trained
    capsys.readouterr()
    assert chat.main(["--model", d, "--adapter", out, "--prompt", "what is two", "--max_new", "8",
                      "--temperature", "0"]) == 0
    assert capsys.readouterr().out.strip() == _reference_reply(d, out, "what is two", 8).strip()


@pytest.mark.parametrize("kind", ["chatml", "bpe", "sp"])
def test_chat_prompt_is_exactly_the_training_prompt(kind):
    from bantamtrain.cli import chat
    tok = make_tokenizer(kind)
    ft.resolve_template(tok)
    for ex, fmt in (({"prompt": "what is two", "response": "four"}, "instruction"),
                    ({"messages": [{"role": "user", "content": "what is two"},
                                   {"role": "assistant", "content": "four"}]}, "messages")):
        (ids, labels, _), = ft.tokenize_examples(tok, [ex], 64)
        start = next(i for i, l in enumerate(labels) if l != -100)
        msgs = [{"role": "user", "content": "what is two"}]
        assert chat.prompt_ids(tok, msgs, fmt) == ids[:start]


def test_chat_repl_keeps_and_resets_the_conversation(trained, monkeypatch):
    from bantamtrain.cli import chat
    d, out = trained
    prompts = []
    monkeypatch.setattr(chat, "generate", lambda m, tok, ids, *a, **k: prompts.append(tok.decode(ids)) or "ok")
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\nwhat is two\n/reset\nwhat is three\n/exit\nignored\n"))
    assert chat.main(["--model", d, "--adapter", out, "--system", "be brief"]) == 0
    assert len(prompts) == 3
    assert "hello" in prompts[1] and "ok" in prompts[1]            # history kept
    assert "hello" not in prompts[2] and "be brief" in prompts[2]   # /reset keeps the system message


def test_chat_with_a_quantized_base_and_an_old_run_dir(trained, tmp_path, capsys):
    from bantamtrain.cli import chat
    d, out = trained
    old = tmp_path / "old"                       # a run from before adapter_meta.json: log.json only
    shutil.copytree(out, old)
    os.remove(old / "adapter_meta.json")
    for q in ("0", "8"):
        assert chat.main(["--model", d, "--adapter", str(old), "--quant", q, "--prompt", "hi",
                          "--max_new", "4", "--temperature", "0.7"]) == 0
    with pytest.raises(SystemExit):
        os.remove(old / "log.json")
        chat.main(["--model", d, "--adapter", str(old), "--prompt", "hi"])


def test_chat_peft_dir(trained, tmp_path, capsys):
    from bantamtrain.cli import chat
    d, out = trained
    peft_dir = tmp_path / "peft"
    peft_dir.mkdir()
    if not hasattr(lora, "load_peft_adapter"):
        (peft_dir / "adapter_config.json").write_text("{}")
        with pytest.raises(SystemExit) as e:
            chat.main(["--model", d, "--adapter", str(peft_dir), "--prompt", "hi"])
        assert "load_peft_adapter" in str(e.value)
        return
    if not os.path.exists(os.path.join(out, "adapter_config.json")):
        pytest.skip("finetune wrote no PEFT adapter (no lora.save_peft_adapter)")
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(os.path.join(out, f), peft_dir)
    capsys.readouterr()
    chat.main(["--model", d, "--adapter", str(peft_dir), "--prompt", "what is two", "--max_new", "8",
               "--temperature", "0"])
    a = capsys.readouterr().out
    chat.main(["--model", d, "--adapter", out, "--prompt", "what is two", "--max_new", "8", "--temperature", "0"])
    assert capsys.readouterr().out == a
