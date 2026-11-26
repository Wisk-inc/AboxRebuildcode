"""finetune.main end to end on tiny offline models: evaluation, outputs, option checks,
default targets, the Muon default, resume, wave-2 flags and 2-process DDP."""
import functools
import inspect
import json
import os
import subprocess
import sys

import pytest
import torch
from test_finetune_fixtures import rows, run_args, save_model_dir, write_jsonl

from bantamtrain import finetune as ft
from bantamtrain import lora, optim, qweights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def chatml_dir(tmp_path_factory):
    return save_model_dir(tmp_path_factory.mktemp("chatml"), "chatml")


@pytest.fixture(scope="module")
def big_dir(tmp_path_factory):
    return save_model_dir(tmp_path_factory.mktemp("big"), "chatml", big=True)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return write_jsonl(tmp_path_factory.mktemp("data") / "d.jsonl", rows(24))


def run(model_dir, data, out, *extra):
    ft.main(run_args(model_dir, data, out, *extra))
    with open(os.path.join(out, "log.json")) as f:
        return json.load(f)


def test_final_eval_runs_with_eval_every_0(chatml_dir, data, tmp_path):
    log = run(chatml_dir, data, tmp_path, "--max_steps", 3, "--lr", 1e-2)
    assert [e["step"] for e in log["evals"]] == [0, 3]
    assert log["train"][-1]["step"] == 3 and log["final_eval_loss"] == log["evals"][-1]["eval_loss"]
    assert log["final_eval_loss"] < log["evals"][0]["eval_loss"]
    for k in ("trainable_params", "train_windows", "padding_fraction", "supervised_tokens", "steps",
              "optimizer_state_bytes", "adapter_bytes", "chat_template"):
        assert k in log                                  # old result fields are still there


def test_eval_loss_ignores_the_prompt_loss_weight_even_at_1(chatml_dir, data, tmp_path):
    # at weight 1 prompt targets look like response targets, so eval must not see them at all
    ev = [run(chatml_dir, data, tmp_path / str(w), "--prompt_loss_weight", w, "--eval_only", 1)["final_eval_loss"]
          for w in (0, 0.3, 1)]
    assert ev[0] == ev[1] == ev[2]
    log = run(chatml_dir, data, tmp_path / "t", "--prompt_loss_weight", 1, "--max_steps", 1)
    assert log["supervised_tokens"] > run(chatml_dir, data, tmp_path / "u", "--max_steps", 1)["supervised_tokens"]


def test_full_finetune_with_quant_is_refused(chatml_dir, data, tmp_path):
    with pytest.raises(SystemExit):
        ft.main(run_args(chatml_dir, data, tmp_path, "--lora", 0, "--quant", 8, "--max_steps", 1))
    assert not os.path.exists(tmp_path / "model")


def test_qlora_with_bf16_params_without_autocast(big_dir, data, tmp_path):
    # k/v projections stay float (< 65536 weights) and meet the int8 embedding's output
    log = run(big_dir, data, tmp_path, "--quant", 8, "--param_dtype", "bf16", "--max_steps", 1)
    assert log["quantized_base_params"] > 0 and log["final_eval_loss"] == log["final_eval_loss"]


def _logits(m, x):
    with torch.no_grad():
        return m(input_ids=x).logits


def test_save_merged_with_quant_merges_into_the_float_base(big_dir, data, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log = run(big_dir, data, tmp_path, "--quant", 8, "--save_merged", 1, "--max_steps", 2, "--lr", 1e-2)
    assert log["merged"] == "float base + adapters"
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged").eval()
    with open(tmp_path / "adapter_meta.json") as f:
        meta = json.load(f)
    ref = AutoModelForCausalLM.from_pretrained(big_dir).eval()
    lora.apply_lora(ref, meta["r"], meta["alpha"], 0.0, tuple(meta["modules"]), rslora=meta["rslora"])
    sd = torch.load(tmp_path / "adapter.pt", weights_only=True)
    assert any(v.abs().sum() > 0 for k, v in sd.items() if "lora_B" in k)
    lora.load_lora_state_dict(ref, sd)
    x = torch.randint(0, 400, (1, 12))
    assert torch.allclose(_logits(merged, x), _logits(ref, x), atol=1e-4)
    assert AutoTokenizer.from_pretrained(tmp_path / "merged").eos_token == "<|im_end|>"


def test_outputs_carry_the_template_and_stop_at_im_end(chatml_dir, data, tmp_path):
    from transformers import AutoTokenizer, GenerationConfig
    run(chatml_dir, data, tmp_path, "--save_merged", 1, "--max_steps", 1)
    tok = AutoTokenizer.from_pretrained(tmp_path)                 # saved next to adapter.pt
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert tok.eos_token == "<|im_end|>" and tok.chat_template
    gen = GenerationConfig.from_pretrained(tmp_path / "merged")
    assert im_end in (gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id])
    run(chatml_dir, data, tmp_path / "full", "--lora", 0, "--param_dtype", "fp32", "--max_steps", 1)
    gen = GenerationConfig.from_pretrained(tmp_path / "full" / "model")
    assert im_end in gen.eos_token_id
    assert AutoTokenizer.from_pretrained(tmp_path / "full" / "model").eos_token == "<|im_end|>"


def test_optim_muon_with_lora_puts_the_adapters_on_muon(chatml_dir, data, tmp_path, capsys):
    log = run(chatml_dir, data, tmp_path / "a", "--optim", "muon", "--max_steps", 1)
    opts = dict(log["optimizers"])
    assert log["args"]["muon_lora"] == 1 and opts["Muon"] == log["trainable_params"]
    log = run(chatml_dir, data, tmp_path / "b", "--optim", "muon", "--muon_lora", 0, "--max_steps", 1)
    assert "Muon" not in dict(log["optimizers"]) and "warning: --optim muon" in capsys.readouterr().out


def _leaf_linears(m):
    return {n.split(".")[-1] for n, mod in m.named_modules()
            if isinstance(mod, torch.nn.Linear) and mod is not m.get_output_embeddings()}


@pytest.mark.parametrize("arch", ["phi3", "opt", "phi", "llama", "gptj"])
def test_default_targets_cover_every_linear_layer(arch):
    import argparse

    import transformers as tf
    kw = dict(vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
              max_position_embeddings=64)
    m = {"phi3": lambda: tf.Phi3ForCausalLM(tf.Phi3Config(**kw, intermediate_size=64, pad_token_id=0)),
         "opt": lambda: tf.OPTForCausalLM(tf.OPTConfig(**kw, ffn_dim=64, word_embed_proj_dim=32)),
         "phi": lambda: tf.PhiForCausalLM(tf.PhiConfig(**kw, intermediate_size=64)),
         "llama": lambda: tf.LlamaForCausalLM(tf.LlamaConfig(**kw, intermediate_size=64)),
         "gptj": lambda: tf.GPTJForCausalLM(tf.GPTJConfig(vocab_size=128, n_positions=64, n_embd=32,
                                                         n_layer=1, n_head=2, rotary_dim=8))}[arch]()
    expect = _leaf_linears(m)
    args = argparse.Namespace(lora=4, lora_alpha=8, lora_dropout=0.0, lora_targets="", dora=0)
    rep = ft._apply_lora(m, args, False)
    assert set(rep["targets"]) == expect and "lm_head" not in rep["targets"]


def test_gpt2_conv1d_layers_are_adapted_or_refused_clearly(tmp_path):
    from test_finetune_fixtures import make_tokenizer
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.pytorch_utils import Conv1D
    tok = make_tokenizer("bpe")
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=32, n_layer=1, n_head=2,
                               bos_token_id=0, eos_token_id=0)).save_pretrained(tmp_path / "g")
    tok.save_pretrained(tmp_path / "g")
    args = run_args(str(tmp_path / "g"), write_jsonl(tmp_path / "d.jsonl", rows(12)), tmp_path / "o", "--max_steps", 1)
    try:                    # Conv1D support lives in bantamtrain.lora (another group's change)
        lora.LoRALinear(Conv1D(6, 4))
    except (AssertionError, TypeError, AttributeError):
        with pytest.raises(SystemExit) as e:
            ft.main(args)
        assert "Conv1D" in str(e.value)
        return
    ft.main(args)
    with open(tmp_path / "o" / "log.json") as f:
        assert {"c_attn", "c_proj", "c_fc"} <= set(json.load(f)["report"]["lora"]["targets"])


def test_resume_matches_an_uninterrupted_run(chatml_dir, data, tmp_path):
    for lr in ("4", "0"):     # LoRA (fp32 adapters, dropout) and full fine-tuning (bf16, stochastic rounding)
        common = ("--lora", lr, "--lora_dropout", 0.1, "--max_steps", 6, "--eval_every", 3, "--lr", 1e-2)
        a = run(chatml_dir, data, tmp_path / f"full{lr}", *common)
        b = run(chatml_dir, data, tmp_path / f"part{lr}", *common, "--save_every", 3, "--stop_step", 3)
        assert b["in_progress"] and len(b["train"]) < len(a["train"])
        ck = tmp_path / f"part{lr}" / "checkpoint"
        st = ft.load_state(str(ck))
        assert st["step"] == 3 and set(st["trainable"]) and st["data"][0]["pos"] > 0
        c = run(chatml_dir, data, tmp_path / f"part{lr}", *common, "--resume", ck)
        assert c["resumed_from"] == str(ck)
        assert [t["loss"] for t in a["train"]] == [t["loss"] for t in c["train"]]
        assert [e["eval_loss"] for e in a["evals"]] == [e["eval_loss"] for e in c["evals"]]
        if lr == "4":
            sa = torch.load(tmp_path / "full4" / "adapter.pt", weights_only=True)
            sc = torch.load(tmp_path / "part4" / "adapter.pt", weights_only=True)
            assert all(torch.equal(sa[k], sc[k]) for k in sa)


def test_resume_refuses_different_options(chatml_dir, tmp_path):
    d = write_jsonl(tmp_path / "d.jsonl", rows(24))
    run(chatml_dir, d, tmp_path, "--pack", "none", "--max_steps", 4, "--save_every", 2, "--stop_step", 2)

    def resume(data, *extra):
        run(chatml_dir, data, tmp_path, "--pack", "none", "--max_steps", 4, *extra, "--resume", tmp_path)
    with pytest.raises(SystemExit, match="seq"):
        resume(d, "--seq", 32)
    with pytest.raises(SystemExit, match="state_bits"):
        resume(d, "--state_bits", 32)
    with pytest.raises(SystemExit, match="jsonl"):
        resume(write_jsonl(tmp_path / "e.jsonl", rows(24, seed=9)))
    with pytest.raises(SystemExit, match="data differ"):                 # same path, new content
        resume(write_jsonl(tmp_path / "d.jsonl", rows(40)))


def test_load_state_without_safe_globals(tmp_path, monkeypatch):
    torch.save({"step": 3, "trainable": {"a": torch.ones(2)}}, tmp_path / "state.pt")
    monkeypatch.delattr(torch.serialization, "safe_globals")      # torch < 2.5
    assert ft.load_state(str(tmp_path))["step"] == 3


# ---- wave-2 flags: each needs a function another group adds; skip until it exists
_OPT = inspect.signature(optim.build_optimizer).parameters
_LORA = inspect.signature(lora.apply_lora).parameters


@pytest.mark.skipif("lr_scale" not in _OPT, reason="build_optimizer has no lr_scale (C2)")
def test_lora_plus_gives_b_matrices_a_larger_lr(chatml_dir, data, tmp_path, monkeypatch):
    seen = []
    real = ft.build_optimizer
    monkeypatch.setattr(ft, "build_optimizer",
                        functools.wraps(real)(lambda *a, **k: seen.append(real(*a, **k)) or seen[-1]))
    run(chatml_dir, data, tmp_path, "--lora_plus", 16, "--max_steps", 1, "--lr", 1e-3)
    scales = {g.get("lr_scale", 1.0) for g in seen[0].param_groups if g["params"]}
    assert scales == {1.0, 16.0}


@pytest.mark.skipif("lr_scale" in _OPT and "dora" in _LORA, reason="both supported")
def test_wave2_flags_fail_clearly_when_unsupported(chatml_dir, data, tmp_path):
    flag = ["--lora_plus", 16] if "lr_scale" not in _OPT else ["--dora", 1]
    with pytest.raises(SystemExit) as e:
        ft.main(run_args(chatml_dir, data, tmp_path, *flag, "--max_steps", 1))
    assert "this build" in str(e.value)


@pytest.mark.skipif("dora" not in _LORA, reason="apply_lora has no dora (C4)")
def test_dora_trains_and_saves(chatml_dir, data, tmp_path):
    log = run(chatml_dir, data, tmp_path, "--dora", 1, "--max_steps", 1)
    assert log["args"]["dora"] == 1 and os.path.exists(tmp_path / "adapter.pt")


@pytest.mark.skipif(not hasattr(lora, "save_peft_adapter"), reason="no lora.save_peft_adapter (C4)")
def test_peft_adapter_is_written_by_default_and_loads_in_peft(chatml_dir, data, tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM
    log = run(chatml_dir, data, tmp_path, "--max_steps", 2, "--lr", 1e-2)
    assert log["peft_adapter"] and os.path.exists(tmp_path / "adapter_config.json")
    x = torch.randint(0, 400, (1, 10))
    pm = peft.PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(chatml_dir), str(tmp_path)).eval()
    from bantamtrain.cli import chat
    bm = AutoModelForCausalLM.from_pretrained(chatml_dir)
    chat.attach_adapter(bm, str(tmp_path / "adapter.pt"))
    assert torch.allclose(_logits(pm, x), _logits(bm.eval(), x), atol=1e-5)


@pytest.mark.skipif(not hasattr(qweights, "load_quantized"), reason="no qweights.load_quantized (C4)")
def test_low_ram_loads_quantized(big_dir, data, tmp_path):
    log = run(big_dir, data, tmp_path, "--quant", 8, "--low_ram", 1, "--max_steps", 1)
    assert log["quantized_base_params"] > 0


@pytest.mark.skipif("exclude" not in _LORA, reason="apply_lora has no all-linear (C4)")
def test_all_linear_string_goes_to_apply_lora(chatml_dir, data, tmp_path):
    log = run(chatml_dir, data, tmp_path, "--lora_targets", "all-linear", "--max_steps", 1)
    assert set(log["report"]["lora"]["targets"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                     "up_proj", "down_proj"}


# ---- 2-process DDP (gloo, CPU)
def _torchrun(args, timeout=150):
    """2-process torchrun of finetune; a hang (the old failure mode) is stopped with its workers
    (torchrun starts each worker in its own session, so SIGTERM torchrun, then sweep by --out)."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": ROOT}
    args = [str(a) for a in args]
    p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                          "-m", "bantamtrain.finetune", *args], cwd=ROOT, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            out, err = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        subprocess.run(["pkill", "-9", "-f", args[args.index("--out") + 1]], check=False)
        pytest.fail(f"torchrun hung for {timeout}s:\n{err[-3000:]}")
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def test_ddp_uneven_shards_finish_in_sync(chatml_dir, tmp_path):
    # 12 examples -> 11 train windows: the old 6/5 shards gave 3 vs 2 steps per epoch and hung
    d = write_jsonl(tmp_path / "d.jsonl", rows(12, seed=1))
    out = tmp_path / "o"
    r = _torchrun(run_args(chatml_dir, d, out, "--pack", "none", "--batch", 1, "--accum", 2, "--epochs", 2,
                           "--lora", 0, "--param_dtype", "bf16", "--lr", 1e-3))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    with open(out / "log.json") as f:
        log = json.load(f)
    # 5 windows per rank at batch 1 x accum 2: 3 steps per epoch, the last with one micro-batch
    assert log["train_windows"] == 10 and log["steps"] == 6 and log["world_size"] == 2
    a, b = log["replica_checksums"]
    assert a == b                     # bf16 stochastic rounding identical on both ranks
    assert os.path.exists(out / "model" / "config.json")


def test_ddp_resume_matches_an_uninterrupted_run(chatml_dir, data, tmp_path):
    # 2 ranks x 4 windows at batch 2 = 2 steps per epoch: step 3 reshuffles each rank's order
    common = ("--lora_dropout", 0.1, "--max_steps", 4, "--eval_every", 2, "--lr", 1e-2)
    logs = []
    for out, extra in ((tmp_path / "a", ()), (tmp_path / "b", ("--save_every", 2, "--stop_step", 2)),
                       (tmp_path / "b", ("--resume", tmp_path / "b"))):
        r = _torchrun(run_args(chatml_dir, data, out, *common, *extra))
        assert r.returncode == 0, r.stderr[-3000:]
        with open(out / "log.json") as f:
            logs.append(json.load(f))
    a, b, c = logs
    st = ft.load_state(str(tmp_path / "b"))
    assert b["in_progress"] and len(st["data"]) == 2 and all("torch_rng" in d for d in st["data"])
    assert [t["loss"] for t in a["train"]] == [t["loss"] for t in c["train"]]
    assert [e["eval_loss"] for e in a["evals"]] == [e["eval_loss"] for e in c["evals"]]
    sa, sc = (torch.load(p / "adapter.pt", weights_only=True) for p in (tmp_path / "a", tmp_path / "b"))
    assert all(torch.equal(sa[k], sc[k]) for k in sa)


def test_ddp_eval_only_exits_cleanly_on_every_rank(chatml_dir, data, tmp_path):
    r = _torchrun(run_args(chatml_dir, data, tmp_path, "--eval_only", 1))
    assert r.returncode == 0, r.stderr[-3000:]
    with open(tmp_path / "log.json") as f:
        assert json.load(f)["final_eval_loss"] > 0
