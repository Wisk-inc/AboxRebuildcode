"""0.3.0 fresh-eyes walkthrough (the README and GUIDE followed from an installed wheel):
`bantam memory --ce_chunk 0`, transformers' activation-checkpointing API, the finetune data
and optimizer-state lines, quiet low-RAM loading of GPT-2 checkpoints, one-line errors for a
mistyped path, and the dtype in the PEFT loading snippets."""
import json
import os
import re
import sys
import warnings

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain import cli, memory  # noqa: E402


# ----------------------------------------------------------------------------- bantam memory
def test_memory_ce_chunk_zero_builds_the_full_logits():
    """--ce_chunk 0 dropped the loss buffer altogether (rows 2-5 stayed "chunked" with a
    0-token chunk), so the table without chunked CE showed less memory, not more."""
    on = memory.scoreboard()["rows"]
    off = memory.scoreboard(ce_chunk=0)["rows"]
    seq_logits = 2048 * 32000 * memory.LOGIT_BYTES
    assert off[0] == on[0]                                    # row 1 never chunks
    for a, b in zip(on[1:], off[1:]):
        assert b["per_seq_bytes"] == pytest.approx(a["per_seq_bytes"] + seq_logits)
        assert b["fixed_bytes"] == pytest.approx(a["fixed_bytes"] - seq_logits)
        assert b["total_gb"] < 81.6 and b["max_micro"] == a["max_micro"] - 1
    # SCOREBOARD_5B section 3: one sequence less per micro-batch, days unchanged
    assert [r["max_micro"] for r in off[1:3]] == [5, 6]
    assert [r["days"] for r in off] == [r["days"] for r in on]
    assert "chunked CE" in on[1]["config"] and "chunked CE" not in off[1]["config"]


def test_memory_cli_ce_chunk_zero(capsys):
    assert memory.main(["--ce_chunk", "0"]) == 0
    out = capsys.readouterr().out
    row2 = next(line for line in out.splitlines() if line.startswith("2. "))
    assert "chunked CE" not in row2 and row2.split()[-4:-2] == ["yes", "5"]


# ----------------------------------------------------------------------------- finetune
pytest.importorskip("transformers")
from test_finetune_fixtures import rows, run_args, save_model_dir, tiny_llama, write_jsonl  # noqa: E402

from bantamtrain import finetune as ft  # noqa: E402


@pytest.fixture(scope="module")
def chatml_dir(tmp_path_factory):
    return save_model_dir(tmp_path_factory.mktemp("chatml"), "chatml")


def test_enable_grad_ckpt_prefers_activation_checkpointing_enable():
    """transformers 5 deprecates gradient_checkpointing_enable (a FutureWarning, removed in
    v6); --grad_ckpt only looked for that name, so it would silently do nothing in v6."""
    calls = []

    class New:
        def activation_checkpointing_enable(self):
            calls.append("activation")

        def gradient_checkpointing_enable(self):
            calls.append("gradient")

        def enable_input_require_grads(self):
            calls.append("inputs")

    class Old:
        def gradient_checkpointing_enable(self):
            calls.append("gradient")

    assert ft.enable_grad_ckpt(New()) and calls == ["activation", "inputs"]
    calls.clear()
    assert ft.enable_grad_ckpt(Old()) and calls == ["gradient"]
    assert ft.enable_grad_ckpt(torch.nn.Linear(2, 2)) is False

    m = tiny_llama(100)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ft.enable_grad_ckpt(m)
    assert getattr(m, "is_activation_checkpointing", None) or m.is_gradient_checkpointing
    assert not [x for x in w if "gradient_checkpointing_enable" in str(x.message)]


def test_finetune_grad_ckpt_and_the_report_lines(chatml_dir, tmp_path, monkeypatch, capsys):
    """--grad_ckpt goes through enable_grad_ckpt; the data line names --seq's value, not a
    count after "--seq"; the optimizer state (allocated at step 1) is reported at the end
    instead of a "state=0MB" at the start."""
    seen = []
    real = ft.enable_grad_ckpt
    monkeypatch.setattr(ft, "enable_grad_ckpt", lambda m: seen.append(real(m)) or seen[-1])
    rs = rows(24) + [{"prompt": "what is " + "the cat sat " * 40, "response": "ten"},
                     {"text": "no prompt here"}]
    data = write_jsonl(tmp_path / "d.jsonl", rs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ft.main(run_args(chatml_dir, data, tmp_path / "o", "--grad_ckpt", 1, "--max_steps", 2))
    assert seen == [True]
    assert not [x for x in w if "gradient_checkpointing_enable" in str(x.message)]
    out = capsys.readouterr().out
    log = json.load(open(tmp_path / "o" / "log.json"))
    line = next(x for x in out.splitlines() if x.startswith("[finetune] data:"))
    assert line == ft.data_summary(log["data"], 64)
    assert f"kept {log['data']['kept']}/26 rows" in line and "text only: 1" in line
    assert re.search(r"no response token within --seq 64: \d+\)", line)
    start = next(x for x in out.splitlines() if "trainable=" in x)
    assert "state=" not in start
    done = next(x for x in out.splitlines() if x.startswith("[finetune] done:"))
    assert log["optimizer_state_bytes"] > 0
    assert done.endswith(f"optimizer state {log['optimizer_state_bytes'] / 1e6:.2f} MB")


def test_data_summary_wording():
    st = {"rows": 100, "kept": 90, "examples": 92, "skipped": {"text only": 8},
          "no_assistant_turn": 0, "no_target_after_truncation": 2, "truncated": 5,
          "boundary_fallback": 1}
    assert ft.data_summary(st, 512) == (
        "[finetune] data: kept 90/100 rows (dropped: text only: 8, no assistant turn: 0, "
        "no response token within --seq 512: 2); truncated to --seq 512: 5; "
        "prompt not a token prefix: 1")


# ----------------------------------------------------------------------------- low-RAM GPT-2
def test_load_quantized_is_quiet_about_keys_from_pretrained_ignores(tmp_path):
    """GPT-2 checkpoints carry h.N.attn.bias causal-mask buffers, which from_pretrained drops
    (GPT2LMHeadModel._keys_to_ignore_on_load_unexpected); load_quantized warned "12
    checkpoint tensors not used" for them. A really unexpected tensor still warns."""
    from safetensors.torch import save_file
    from transformers import GPT2Config, GPT2LMHeadModel

    from bantamtrain import qweights
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=300, n_embd=128, n_layer=2, n_head=2, n_positions=64))
    sd = {k: v.contiguous() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    sd.update({f"transformer.h.{i}.attn.bias": torch.tril(torch.ones(1, 1, 64, 64)) for i in range(2)})
    d = tmp_path / "g"
    m.config.save_pretrained(d)
    save_file(sd, str(d / "model.safetensors"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        q = qweights.load_quantized(str(d), 8)
    assert q.quant_report["low_ram"] is True
    sd["transformer.extra_unknown"] = torch.zeros(3)
    save_file(sd, str(d / "model.safetensors"))
    with pytest.warns(UserWarning, match=r"1 checkpoint tensors not used.*extra_unknown"):
        qweights.load_quantized(str(d), 8)


# ----------------------------------------------------------------------------- mistyped paths
@pytest.fixture
def keep_runtime():
    """Commands set torch's thread count and flush-to-zero for the process: put them back."""
    n, ftz = torch.get_num_threads(), float(torch.tensor([1e-39]) * 1.0) == 0.0
    yield
    torch.set_num_threads(n)
    torch.set_flush_denormal(ftz)


@pytest.mark.parametrize("argv", [
    ["pack-text", "--data", "{tmp}/nope"],
    ["pack-era5", "--data", "{tmp}/nope"],
    ["weather", "--data", "{tmp}/nope", "--budget", "1e9", "--name", "w", "--out", "{tmp}/o"],
    ["weather", "--forecast", "--load", "{tmp}/nope.pt", "--init", "2018-03-01T00", "--steps", "2"],
    ["prepare-text", "--train", "{tmp}/nope.parquet", "--out", "{tmp}/o"],
])
def test_a_missing_input_path_is_a_one_line_error(argv, tmp_path, capsys, keep_runtime):
    """These raised a FileNotFoundError traceback; pretrain and generate already exit 2 with a
    one-line message."""
    if argv[0] == "prepare-text":
        pytest.importorskip("pyarrow")
    argv = [a.format(tmp=tmp_path) for a in argv]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith(f"bantam {argv[0]}: error: ")
    assert "nope" in err[0] and not (tmp_path / "o").exists()


def test_a_missing_jsonl_or_adapter_is_a_one_line_error(chatml_dir, tmp_path, capsys, keep_runtime):
    assert cli.main(["finetune", *run_args(chatml_dir, str(tmp_path / "nope.jsonl"),
                                           tmp_path / "o")]) == 2
    assert "bantam finetune: error: " in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
    assert cli.main(["chat", "--model", chatml_dir, "--adapter", str(tmp_path / "nope"),
                     "--prompt", "hi", "--max_new", "2"]) == 2
    assert "bantam chat: error: " in capsys.readouterr().err


# ----------------------------------------------------------------------------- docs
def test_peft_snippets_load_the_base_in_fp32(chatml_dir, tmp_path):
    """transformers 5 loads a checkpoint in its stored dtype (bf16 for SmolLM2), so the
    README / GUIDE / PyPI snippet `AutoModelForCausalLM.from_pretrained(id)` gave a bf16
    base whose logits differed from Bantam's by up to 4 on SmolLM2-135M. With the dtype
    given, a bf16-stored base matches Bantam's adapter to 1e-5."""
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    from bantamtrain.cli import chat
    for doc in ("README.md", "docs/GUIDE.md", "packaging/README.md"):
        text = open(os.path.join(ROOT, doc)).read()
        calls = re.findall(r'AutoModelForCausalLM\.from_pretrained\("HuggingFaceTB/SmolLM2-135M"[^)]*\)',
                           text)
        assert calls and all("dtype=torch.float32" in c for c in calls), (doc, calls)
    bf = tmp_path / "bf16"
    AutoModelForCausalLM.from_pretrained(chatml_dir).to(torch.bfloat16).save_pretrained(bf)
    for f in os.listdir(chatml_dir):
        if f.startswith(("tokenizer", "special_tokens", "chat_template")):
            os.link(os.path.join(chatml_dir, f), bf / f)
    data = write_jsonl(tmp_path / "d.jsonl", rows(24))
    ft.main(run_args(str(bf), data, tmp_path / "o", "--max_steps", 2, "--lr", 1e-2))
    x = torch.randint(0, 400, (1, 10))
    base = AutoModelForCausalLM.from_pretrained(str(bf), dtype=torch.float32)
    pm = peft.PeftModel.from_pretrained(base, str(tmp_path / "o")).eval()
    bm = AutoModelForCausalLM.from_pretrained(str(bf), dtype=torch.float32)
    chat.attach_adapter(bm, str(tmp_path / "o" / "adapter.pt"))
    with torch.no_grad():
        assert torch.allclose(pm(input_ids=x).logits, bm.eval()(input_ids=x).logits, atol=1e-5)
