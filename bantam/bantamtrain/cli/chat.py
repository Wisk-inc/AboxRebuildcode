"""Chat with a Hugging Face causal LM, optionally with a fine-tuned adapter.

    bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly --prompt "Name three colours"
    bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly --quant 4    # interactive

--adapter takes a `bantam finetune` output dir (adapter.pt with adapter_meta.json, or an
older run's log.json), an adapter.pt file, or a PEFT adapter dir (adapter_config.json).
The prompt is rendered as in training: with the tokenizer saved next to the adapter
(chat template, eos) when there is one, else the model's template, else the plain
format. Tokens are streamed to stdout. In the REPL, /reset clears the conversation and
/exit (or end of input) quits.
"""
import argparse
import json
import os
import sys

import torch


def _has(d, name):
    return os.path.isfile(os.path.join(d, name))


def load_tokenizer(model, adapter=""):
    """Tokenizer saved with the adapter (training's template and eos) or the model's."""
    from transformers import AutoTokenizer

    from ..finetune import resolve_template
    d = adapter if adapter and os.path.isdir(adapter) else os.path.dirname(adapter or "")
    tok = AutoTokenizer.from_pretrained(d if d and _has(d, "tokenizer_config.json") else model)
    resolve_template(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _bantam_meta(d, sd):
    """r / alpha / rslora / dora / module names of a Bantam adapter.pt."""
    mods = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith((".lora_A", ".lora_B"))})
    if not mods:
        raise SystemExit("[chat] adapter.pt has no lora_A / lora_B tensors")
    meta = {"r": sd[mods[0] + ".lora_A"].shape[0], "modules": mods}
    if _has(d, "adapter_meta.json"):
        with open(os.path.join(d, "adapter_meta.json")) as f:
            m = json.load(f)
        meta.update(alpha=m["alpha"], rslora=m.get("rslora", False), dora=m.get("dora", False))
    elif _has(d, "log.json"):                      # runs from before adapter_meta.json
        with open(os.path.join(d, "log.json")) as f:
            a = json.load(f)["args"]
        rs = a.get("rslora", -1)
        meta.update(alpha=a["lora_alpha"], rslora=(meta["r"] >= 64) if rs < 0 else bool(rs),
                    dora=bool(a.get("dora", 0)))
    else:
        raise SystemExit(f"[chat] {d}: adapter.pt without adapter_meta.json or log.json; "
                         "pass --lora_alpha (and --rslora 1 if it was trained with rsLoRA)")
    return meta


def attach_adapter(model, path, lora_alpha=None, rslora=None):
    """Load a Bantam adapter.pt (or its dir) or a PEFT adapter dir into `model`."""
    from .. import lora
    if os.path.isdir(path) and not _has(path, "adapter.pt"):
        if not _has(path, "adapter_config.json"):
            raise SystemExit(f"[chat] {path}: no adapter.pt or adapter_config.json")
        load = getattr(lora, "load_peft_adapter", None)
        if load is None:
            raise SystemExit("[chat] this bantamtrain cannot load PEFT adapters "
                             "(bantamtrain.lora.load_peft_adapter is missing). Load it with "
                             "peft.PeftModel.from_pretrained(model, dir).merge_and_unload(), or pass "
                             "the adapter.pt that `bantam finetune` writes")
        load(model, path)
        return "peft"
    f = os.path.join(path, "adapter.pt") if os.path.isdir(path) else path
    sd = torch.load(f, map_location="cpu", weights_only=True)
    meta = {"rslora": False, "dora": False}
    if lora_alpha is not None:
        mods = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith((".lora_A", ".lora_B"))})
        meta.update(r=sd[mods[0] + ".lora_A"].shape[0], modules=mods, alpha=lora_alpha)
    else:
        meta = _bantam_meta(os.path.dirname(f), sd)
    if rslora is not None:
        meta["rslora"] = bool(rslora)
    kw = {"rslora": meta["rslora"]}
    if meta.get("dora") or any(k.endswith(".lora_magnitude_vector") for k in sd):   # also with --lora_alpha
        kw["dora"] = True
    lora.apply_lora(model, meta["r"], meta["alpha"], 0.0, tuple(meta["modules"]), **kw)
    lora.load_lora_state_dict(model, sd)
    return "bantam"


def prompt_ids(tok, msgs, plain_format="instruction"):
    """Token ids of the generation prompt, rendered exactly as finetune renders a prompt.
    Without a chat template, "instruction" answers the last user message on its own
    (### Instruction / ### Response, as for prompt/response data) and "messages" keeps
    the conversation (role: text lines)."""
    from ..finetune import bos_ids, render
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    elif plain_format == "instruction":
        text = render(tok, {"prompt": msgs[-1]["content"], "response": ""})[0]
    else:
        text = render(tok, {"messages": msgs + [{"role": "assistant", "content": ""}]})[0]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bos = bos_ids(tok)
    return bos + ids if bos and not text.startswith(tok.bos_token) else ids


def eos_ids(model, tok):
    ids = [tok.eos_token_id]
    g = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids += g if isinstance(g, list) else [g]
    return [i for n, i in enumerate(ids) if i is not None and i not in ids[:n]]


@torch.no_grad()
def generate(model, tok, ids, max_new=256, temperature=0.7, top_p=0.95, stream=True):
    """Sample (greedy at temperature 0) up to max_new tokens after `ids`; streams the text
    to stdout and returns it."""
    from transformers import TextStreamer
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    kw = {"max_new_tokens": max_new, "eos_token_id": eos_ids(model, tok),
          "pad_token_id": tok.pad_token_id, "attention_mask": torch.ones_like(x)}
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    out = model.generate(x, streamer=streamer, **kw)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True)


def load_model(args):
    from transformers import AutoModelForCausalLM

    from ..finetune import _match_embedding_dtype
    from ..qweights import quantize_model
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if args.device == "auto" else args.device))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(
        args.dtype, torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.quant != "0":
        quantize_model(model, {"8": 8, "4": 4, "nf4": "nf4"}[args.quant], min_numel=1 << 16, embed_bits=8)
        if dtype == torch.bfloat16:
            _match_embedding_dtype(model, dtype)
    if args.adapter:
        attach_adapter(model, args.adapter, args.lora_alpha, None if args.rslora < 0 else args.rslora)
    return model.to(dev).eval()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bantam chat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local dir (the base model)")
    ap.add_argument("--adapter", default="", help="finetune output dir, adapter.pt, or PEFT adapter dir")
    ap.add_argument("--quant", default="0", choices=["0", "8", "4", "nf4"], help="quantize the base")
    ap.add_argument("--prompt", default="", help="answer this and exit (default: interactive)")
    ap.add_argument("--system", default="", help="system message")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--plain_format", default="instruction", choices=["instruction", "messages"],
                    help="prompt format for models without a chat template")
    ap.add_argument("--lora_alpha", type=float, default=None,
                    help="alpha of an adapter.pt that has no adapter_meta.json / log.json")
    ap.add_argument("--rslora", type=int, default=-1, help="override the adapter's rsLoRA setting")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    tok = load_tokenizer(args.model, args.adapter)
    model = load_model(args)
    msgs = [{"role": "system", "content": args.system}] if args.system else []

    def turn(text):
        msgs.append({"role": "user", "content": text})
        reply = generate(model, tok, prompt_ids(tok, msgs, args.plain_format), args.max_new,
                         args.temperature, args.top_p)
        msgs.append({"role": "assistant", "content": reply})
        return reply

    if args.prompt:
        turn(args.prompt)
        return 0
    interactive = sys.stdin.isatty()
    if interactive:
        print("bantam chat: /reset clears the conversation, /exit quits", flush=True)
    while True:
        if interactive:
            print("> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line or line.strip() in ("/exit", "/quit"):
            return 0
        if line.strip() == "/reset":
            del msgs[1 if args.system else 0:]
            continue
        if line.strip():
            turn(line.strip())


if __name__ == "__main__":
    sys.exit(main())
