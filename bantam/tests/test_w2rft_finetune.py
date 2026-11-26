"""Wave-2 end-to-end review of fine-tuning: packed windows under every attention
implementation, and other interactions between finetune, lora, qweights and losses."""
import pytest
import torch
import torch.nn.functional as F

from bantamtrain import finetune as ft

pytest.importorskip("transformers")


def _tiny(name, attn=None):
    import transformers as tf
    torch.manual_seed(0)
    kw = dict(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
              initializer_range=0.2)             # large enough that attention patterns matter
    if name == "gemma2":                          # eager by default (SDPA has no attention softcap)
        cfg = tf.Gemma2Config(**kw, head_dim=8, query_pre_attn_scalar=8, final_logit_softcapping=30.0)
    elif name == "gpt2":
        cfg = tf.GPT2Config(vocab_size=200, n_positions=64, n_embd=32, n_layer=2, n_head=4,
                            initializer_range=0.2)
    else:
        cfg = tf.LlamaConfig(**kw)
    if attn:
        cfg._attn_implementation = attn
    return tf.AutoModelForCausalLM.from_config(cfg).eval()


@pytest.mark.parametrize("name,attn", [("llama", None), ("llama", "eager"), ("gemma2", None),
                                       ("gpt2", "eager")])
def test_packed_window_equals_examples_alone_for_every_attention(name, attn):
    """A packed window (block-diagonal mask) gives every example its unpacked loss. Eager
    attention adds the prepared 4-D mask to the scores, so the bool mask used to mask
    nothing there (Gemma-2's default: packed examples saw each other and the future)."""
    m = _tiny(name, attn)
    torch.manual_seed(1)
    a, b, c = (torch.randint(1, 200, (k,)).tolist() for k in (9, 6, 4))
    items = [(e, e, [1.0] * len(e)) for e in (a, b, c)]
    windows = ft.pack(items, 20, 0, "bfd")
    assert len(windows) == 1 and max(windows[0][3]) == 2           # all three in one window
    x, y, w, g = (torch.tensor([v]) for v in windows[0])
    with torch.no_grad():
        s, n = ft.lm_loss_sum(m, x[:, :-1], y[:, 1:], 0, w[:, 1:], g[:, :-1])
        s_chunked, _ = ft.lm_loss_sum(m, x[:, :-1], y[:, 1:], 8, w[:, 1:], g[:, :-1])
        ref = sum(F.cross_entropy(m(input_ids=torch.tensor([e])).logits[0, :-1].float(),
                                  torch.tensor(e[1:]), reduction="sum") for e in (a, b, c))
    assert int(n) == 8 + 5 + 3
    assert abs(s.item() - ref.item()) < 1e-4 and abs(s_chunked.item() - ref.item()) < 1e-4
    # token weights (--prompt_loss_weight): chunked == full == the weighted per-example sum
    wt = [[0.5 if i % 3 else 1.0 for i in range(len(e))] for e in (a, b, c)]
    (x, y, w, g), = ft.pack([(e, e, v) for e, v in zip((a, b, c), wt)], 20, 0, "bfd")
    x, y, w, g = (torch.tensor([v]) for v in (x, y, w, g))
    with torch.no_grad():
        full = ft.lm_loss_sum(m, x[:, :-1], y[:, 1:], 0, w[:, 1:], g[:, :-1])
        chunked = ft.lm_loss_sum(m, x[:, :-1], y[:, 1:], 8, w[:, 1:], g[:, :-1])
        ref = sum((F.cross_entropy(m(input_ids=torch.tensor([e])).logits[0, :-1].float(), torch.tensor(e[1:]),
                                   reduction="none") * torch.tensor(v[1:])).sum() for e, v in zip((a, b, c), wt))
    assert abs(full[1].item() - sum(sum(v[1:]) for v in wt)) < 1e-5 and abs(chunked[1] - full[1]) < 1e-5
    assert abs(full[0].item() - ref.item()) < 1e-4 and abs(chunked[0].item() - ref.item()) < 1e-4


def test_model_attention_mask_forms():
    m = _tiny("llama", "eager")
    mask, _ = ft.segment_mask(torch.tensor([[0, 0, 1, -1]]))
    add = ft.model_attention_mask(m, mask)
    assert add.dtype == torch.float32 and (add == 0).eq(mask).all() and add.min() < -1e30
    with torch.autocast("cpu", dtype=torch.bfloat16):          # in the dtype of the scores
        assert ft.model_attention_mask(m, mask).dtype == torch.bfloat16
    assert ft.model_attention_mask(_tiny("llama", "sdpa"), mask) is mask


@pytest.mark.parametrize("tied", [True, False])
def test_split_head_does_not_dequantize_a_quantized_head(tied, monkeypatch):
    """_split_head runs on every loss call; `hasattr(head, "weight")` evaluated the quantized
    head's .weight property, a dequantization of the whole vocab table each time."""
    import transformers as tf

    from bantamtrain import qweights
    torch.manual_seed(0)
    m = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=64,
                                           num_hidden_layers=1, num_attention_heads=4,
                                           tie_word_embeddings=tied))
    qweights.quantize_model(m, 8, min_numel=1 << 16, embed_bits=8)
    assert type(m.lm_head).__name__ == ("QTiedHead" if tied else "QLinear")
    calls = []
    real = qweights.QBase.dequantize
    monkeypatch.setattr(qweights.QBase, "dequantize", lambda self, *a: calls.append(1) or real(self, *a))
    base, head = ft._split_head(m)
    assert head is m.lm_head and base is m.model and not calls


def test_max_examples_picks_the_same_rows_without_loading_the_whole_dataset(tmp_path, monkeypatch):
    """--dataset with --max_examples materialized every row as a dict before keeping a few
    (GBs for a 1M-row SFT set); the selection itself must not change."""
    import argparse
    import json
    import sys
    import types
    rs = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(500)]
    with open(tmp_path / "d.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rs)
    seen = []

    class DS:                                   # the parts of datasets.Dataset load_examples uses
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            for r in self.rows:
                seen.append(r)
                yield r

        def select(self, idx):
            return DS([self.rows[i] for i in idx])
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *a, **k: DS(rs)))
    kw = dict(config="", split="train", max_examples=7, seed=3, messages_col="messages",
              prompt_col="prompt", response_col="response")
    hub = ft.load_examples(argparse.Namespace(jsonl="", dataset="x", **kw))
    assert len(seen) == 7
    assert hub == ft.load_examples(argparse.Namespace(jsonl=str(tmp_path / "d.jsonl"), dataset="", **kw))


def test_lora_targets_list_with_spaces_wraps_every_named_layer():
    """--lora_targets "q_proj, v_proj" silently adapted q_proj only (" v_proj" matched nothing)."""
    import argparse
    m = _tiny("llama")
    args = argparse.Namespace(lora=4, lora_alpha=8, lora_dropout=0.0, lora_targets="q_proj, v_proj ", dora=0)
    rep = ft._apply_lora(m, args, False)
    assert rep["targets"] == ["q_proj", "v_proj"] and rep["wrapped"] == 4


def test_lora_plus_ratio_survives_the_lr_schedule(tmp_path, monkeypatch):
    """The schedule rewrites every group's lr each step: the B groups must keep lr x ratio (the
    existing test only checked the lr_scale tags build_optimizer sets)."""
    import functools
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finetune_fixtures import rows, run_args, save_model_dir, write_jsonl
    d = save_model_dir(tmp_path / "m", "chatml")
    seen, real = [], ft.build_optimizer
    monkeypatch.setattr(ft, "build_optimizer",
                        functools.wraps(real)(lambda *a, **k: seen.append(real(*a, **k)) or seen[-1]))
    ft.main(run_args(d, write_jsonl(tmp_path / "d.jsonl", rows(24)), tmp_path / "o", "--lora_plus", 16,
                     "--max_steps", 3, "--lr", 1e-3))
    last = json.load(open(tmp_path / "o" / "log.json"))["train"][-1]["lr"]
    groups = [g for g in seen[0].param_groups if g["params"]]
    assert {g["lr_scale"] for g in groups} == {1.0, 16.0}
    for g in groups:
        assert abs(g["lr"] - last * g["lr_scale"]) < 1e-12
