"""finetune data path: prompt/response boundary, BOS, conversation shapes, dataset formats,
truncation, packing and step planning."""
import argparse
import random
import time

import pytest
from test_finetune_fixtures import make_tokenizer, rows, run_args, save_model_dir, write_jsonl

from bantamtrain import finetune as ft


def supervised(tok, ex, seq=64, **kw):
    (ids, labels, w), = ft.tokenize_examples(tok, [ex], seq, **kw)
    return tok.decode([t for t, l, ww in zip(ids, labels, w) if l != -100 and ww == 1.0])


def msg(*turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


def test_plain_messages_supervise_first_response_token():
    tok = make_tokenizer("bpe")
    assert ft.resolve_template(tok) == "plain"
    # the space before the answer merges into its first token ('Ġyes'): it must be a target
    assert supervised(tok, msg(("user", "is the sky blue?"), ("assistant", "yes"))) == " yes<|endoftext|>"
    out = supervised(tok, msg(("user", "capital of france?"), ("assistant", "paris is the capital")))
    assert out == " paris is the capital<|endoftext|>"
    assert supervised(tok, {"prompt": "is the sky blue?", "response": "yes"}) == "yes<|endoftext|>"


def test_nonprefix_generation_prompt_supervises_whole_answer():
    tok = make_tokenizer("chatml")
    tok.chat_template = ("{% for m in messages %}{% if m['role'] == 'user' %}"
                         "{{ '<|im_start|>user\n' + m['content'] + '<|im_end|>' }}"
                         "{% else %}{{ '<|im_start|>assistant\n' + m['content'] + '<|im_end|>' }}"
                         "{% endif %}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}")
    st = {}
    out = supervised(tok, {"prompt": "what is two", "response": "four five six seven"}, stats=st)
    assert out == "four five six seven<|im_end|>" and st["boundary_fallback"] == 1


def test_template_that_differs_before_the_answer_supervises_only_the_answer():
    tok = make_tokenizer("chatml")       # the generation prompt changes the header, not just the end
    tok.chat_template = ("{% if add_generation_prompt %}{{ 'two ' }}{% else %}{{ 'one ' }}{% endif %}"
                         "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + "
                         "'<|im_end|>' }}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    assert supervised(tok, {"prompt": "what is two", "response": "four five"}) == "four five<|im_end|>"


def test_conversation_ending_in_user_turn_is_trimmed():
    tok = make_tokenizer("chatml")
    ft.resolve_template(tok)
    ex = msg(("user", "what is two"), ("assistant", "four"), ("user", "red green blue"))
    out = supervised(tok, ex)
    assert "red" not in out and "green" not in out and out.startswith("four<|im_end|>")
    st = {}
    assert ft.tokenize_examples(tok, [msg(("user", "hi"))], 64, stats=st) == []
    assert st["no_assistant_turn"] == 1


def test_plain_path_adds_bos_when_the_tokenizer_does():
    sp, bpe = make_tokenizer("sp"), make_tokenizer("bpe")
    assert ft.resolve_template(sp) == "plain"
    (ids, labels, _), = ft.tokenize_examples(sp, [{"prompt": "what is two", "response": "yes"}], 64)
    assert ids[0] == sp.bos_token_id and ids.count(sp.bos_token_id) == 1 and labels[0] == -100
    assert sp.decode([t for t, l in zip(ids, labels) if l != -100]) == "yes</s>"
    (ids, _, _), = ft.tokenize_examples(sp, [msg(("user", "what is two"), ("assistant", "yes"))], 64)
    assert ids[0] == sp.bos_token_id
    (ids, _, _), = ft.tokenize_examples(bpe, [{"prompt": "what is two", "response": "yes"}], 64)
    assert ids[0] != bpe.bos_token_id                   # GPT-2 style tokenizers add no BOS


def test_sharegpt_and_role_aliases_are_loaded(tmp_path):
    rs = [{"conversations": [{"from": "human", "value": "what is two"}, {"from": "gpt", "value": "four"}]},
          {"text": "just text"}, {"prompt": "p", "response": "r"},
          {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                        {"role": "assistant", "content": "hello"}]}]
    a = argparse.Namespace(jsonl=write_jsonl(tmp_path / "d.jsonl", rs), max_examples=0, seed=0,
                           messages_col="messages", prompt_col="prompt", response_col="response")
    st = {}
    ex = ft.load_examples(a, st)
    assert ex[0] == {"messages": [{"role": "user", "content": "what is two"},
                                  {"role": "assistant", "content": "four"}]}
    assert ex[2]["messages"][0]["content"] == "hi" and len(ex) == 3
    assert st["skipped"] == {"text only": 1}


@pytest.mark.parametrize("data", [[{"text": "just text"}] * 5, [{"prompt": "p", "response": "r"}]])
def test_unusable_datasets_fail_with_a_clear_message(tmp_path, data):
    d = save_model_dir(tmp_path / "m")
    with pytest.raises(SystemExit) as e:
        ft.main(run_args(d, write_jsonl(tmp_path / "d.jsonl", data), tmp_path / "o", "--max_steps", 1))
    assert "usable" in str(e.value)


def test_prompt_loss_weight_applied_after_tokenizing_matches_tokenize_examples():
    tok = make_tokenizer("chatml")
    ft.resolve_template(tok)
    ex = rows(6) + rows(3, kind="msg")
    base = ft.tokenize_examples(tok, ex, 64)
    assert ft.with_prompt_loss(base, 0.0) is base
    for w in (0.3, 1.0):
        assert ft.with_prompt_loss(base, w) == ft.tokenize_examples(tok, ex, 64, w)


def test_wrapped_mode_keeps_long_examples_whole():
    tok = make_tokenizer("chatml")
    ft.resolve_template(tok)
    ex = {"prompt": "what is two", "response": " ".join(["red green blue"] * 10)}
    st = {}
    assert len(ft.tokenize_examples(tok, [ex], 16, stats=st)[0][0]) == 16 and st["truncated"] == 1
    items = ft.tokenize_examples(tok, [ex], 16, truncate=False)
    ids, labels, _ = items[0]
    assert len(ids) > 40
    w = ft.pack(items, 16, tok.pad_token_id, "wrapped")
    targets = [t for x in w for t in x[1][1:] if t != -100]
    assert targets == [t for t in labels[1:] if t != -100]           # each once, EOS included
    assert tok.convert_tokens_to_ids("<|im_end|>") in targets[-2:]   # ...<|im_end|>\n


def test_padding_fraction_counts_padding_not_pad_ids():
    eos = 7                                    # pad == eos, and every example ends with eos
    items = [([1, 2, 3, eos], [1, 2, 3, eos], [1.0] * 4), ([4, 5, eos], [4, 5, eos], [1.0] * 3)]
    w = ft.pack(items, 9, eos, "bfd")          # one bin of capacity 10: 7 real tokens, 3 pad
    assert ft.padding_fraction(w) == pytest.approx(0.3)
    assert sum(x[0].count(eos) for x in w) / 10 == pytest.approx(0.5)    # what was logged before


def _bfd_reference(items, seq, pad_id):
    """The original O(n^2) best-fit decreasing."""
    n = seq + 1
    items = [(i[:n], ([-100] + l[1:])[:n], w[:n]) for i, l, w in items if len(i) >= 2]
    bins = []
    for i, l, w in sorted(items, key=lambda t: -len(t[0])):
        best, best_left = None, None
        for b in bins:
            left = n - b["len"]
            if len(i) <= left and (best_left is None or left < best_left):
                best, best_left = b, left
        if best is None:
            best = {"len": 0, "x": [], "y": [], "w": [], "g": []}
            bins.append(best)
        best["g"] += [best["g"][-1] + 1 if best["g"] else 0] * len(i)
        best["x"] += i
        best["y"] += l
        best["w"] += w
        best["len"] += len(i)
    return [ft._pad(b["x"], b["y"], b["w"], b["g"], n, pad_id) for b in bins]


def test_fast_bfd_matches_the_reference_and_scales():
    r = random.Random(0)
    for _ in range(200):
        seq = r.randint(4, 40)
        items = []
        for _ in range(r.randint(0, 30)):
            k = r.randint(1, seq + 5)
            ids = [r.randint(1, 50) for _ in range(k)]
            items.append((ids, ids, [1.0] * k))
        assert ft.pack(items, seq, 0, "bfd") == _bfd_reference(items, seq, 0)
    items = [([1] * k, [1] * k, [1.0] * k) for k in (r.randint(20, 400) for _ in range(40000))]
    t = time.perf_counter()
    w = ft.pack(items, 512, 0, "bfd")
    assert time.perf_counter() - t < 10                    # the O(n^2) version took ~60 s here
    assert sum(1 for x in w for g in x[3] if g >= 0) == sum(len(i) for i, _, _ in items)


def test_plan_steps_is_the_same_on_every_rank():
    # 15 windows on 2 ranks used to be 8/7 -> 2 vs 1 steps at batch 4: a hang at the last step
    per, steps = ft.plan_steps(15, 2, 4, 1, 1.0, 0)
    assert per == 7 and steps == 1
    shards = [list(range(15))[:per * 2][r::2] for r in range(2)]
    assert len(shards[0]) == len(shards[1]) == per
    assert ft.plan_steps(39, 2, 2, 1, 1.0, 0) == (19, 9)
    assert ft.plan_steps(10, 1, 4, 1, 2.0, 0) == (10, 4) and ft.plan_steps(10, 1, 4, 1, 2.0, 7)[1] == 7


def test_existing_pipeline_still_masks_and_packs():
    tok = make_tokenizer("bpe")
    items = ft.tokenize_examples(tok, rows(6) + rows(4, kind="msg"), 64)
    assert len(items) == 10
    for mode in ("bfd", "wrapped", "none"):
        assert all(len(x[0]) == 17 for x in ft.pack(items, 16, 0, mode))
