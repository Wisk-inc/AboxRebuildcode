"""Offline fixtures shared by the test_finetune_* files: tokenizers trained in memory
(ByteLevel BPE like GPT-2, the same with ChatML tokens like SmolLM2 bases, and a
SentencePiece-style one that prepends BOS like Llama-2), tiny random HF models saved to
tmp dirs, and JSONL data. Nothing touches the network."""
import functools
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORDS = ("the cat sat on a mat and dog ran fast slow blue red green what is who where when why "
         "how answer question paris london two three four five six seven eight nine ten yes no "
         "maybe hello hi capital of france sky").split()


@functools.lru_cache(None)
def _tokenizer_json(kind):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    r = random.Random(0)
    corpus = [" ".join(r.choice(WORDS) for _ in range(20)) for _ in range(300)]
    corpus += ["### Instruction:\nfoo\n\n### Response:\nbar", "user: a\nassistant: b"]
    sp = kind == "sp"
    specials = {"bpe": ["<|endoftext|>"], "chatml": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                "sp": ["<unk>", "<s>", "</s>"]}[kind]
    tk = Tokenizer(models.BPE(unk_token="<unk>" if sp else None))
    if sp:
        tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first")
        tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="first")
    else:
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=450, special_tokens=specials,
        initial_alphabet=[] if sp else pre_tokenizers.ByteLevel.alphabet()))
    if sp:
        tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    return tk.to_str()


def make_tokenizer(kind="chatml"):
    """kind: "bpe" (plain format), "chatml" (ChatML gets installed), "sp" (adds BOS)."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer.from_str(_tokenizer_json(kind))
    if kind == "sp":
        return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                       unk_token="<unk>")
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", bos_token="<|endoftext|>")


def tiny_llama(vocab, hidden=64, inter=128, heads=4, kv=2, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=2,
                      num_attention_heads=heads, num_key_value_heads=kv, max_position_embeddings=256,
                      tie_word_embeddings=False, bos_token_id=0, eos_token_id=0)
    return LlamaForCausalLM(cfg)


def save_model_dir(path, kind="chatml", big=False):
    """Tokenizer + tiny random Llama in `path`. big=True makes every q/o/mlp projection,
    the embeddings and the head >= 65536 weights, so `--quant` quantizes them (k/v stay
    float, as on small GQA models)."""
    tok = make_tokenizer(kind)
    m = tiny_llama(len(tok), 256, 512, 4, 1) if big else tiny_llama(len(tok))
    m.config.bos_token_id, m.config.eos_token_id = tok.bos_token_id, tok.eos_token_id
    m.generation_config.bos_token_id, m.generation_config.eos_token_id = tok.bos_token_id, tok.eos_token_id
    m.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def rows(n, seed=0, kind="pr"):
    r = random.Random(seed)

    def text(k):
        return " ".join(r.choice(WORDS) for _ in range(k))
    out = []
    for _ in range(n):
        p, a = "what is " + text(r.randint(2, 8)), text(r.randint(1, 8))
        out.append({"prompt": p, "response": a} if kind == "pr" else
                   {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": a}]})
    return out


def write_jsonl(path, rs):
    with open(path, "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    return str(path)


def run_args(model_dir, data, out, *extra):
    """Common fast CPU arguments for finetune.main."""
    return ["--model", model_dir, "--jsonl", data, "--out", str(out), "--seq", "64", "--batch", "2",
            "--threads", "1", "--eval_every", "0", "--lora", "4", "--lora_dropout", "0",
            *map(str, extra)]


def test_fixture_tokenizers_behave_like_their_models():
    from bantamtrain import finetune as ft
    assert ft.resolve_template(make_tokenizer("bpe")) == "plain"
    assert ft.resolve_template(make_tokenizer("chatml")) == "chatml-installed"
    sp = make_tokenizer("sp")
    assert sp("what is two")["input_ids"][0] == sp.bos_token_id          # prepends BOS like Llama-2
    assert make_tokenizer("bpe")("what")["input_ids"][0] != 0
