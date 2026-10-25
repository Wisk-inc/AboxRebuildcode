"""Fine-tune any HuggingFace causal LM with the Bantam stack.

    python -m zoomtrain.finetune --model HuggingFaceTB/SmolLM2-135M \
        --dataset databricks/databricks-dolly-15k --max_examples 2000 \
        --lora 16 --quant 4 --optim muon --state_bits 8 --epochs 1 --out runs/dolly

What it gives you
  * LoRA / QLoRA: frozen base (optionally int8/int4 blockwise, zoomtrain.qweights)
    + trainable low-rank adapters; or full fine-tuning (--lora 0).
  * Muon (hidden matrices) + 8-bit optimizer state + bf16 weights with
    stochastic rounding; chunked cross-entropy so a 32k-128k vocabulary never
    materializes the full logits; activation checkpointing; grad accumulation.
  * SFT data from a HuggingFace dataset or a local JSONL: prompt/response
    columns, Dolly-style instruction/context/response, or chat `messages`.
    Uses the tokenizer's chat template when it has one; the loss is computed
    on response tokens only; examples are packed into fixed windows.
  * Multi-GPU with torchrun (DistributedDataParallel), resumable JSON logs,
    adapter-only checkpoints (MBs) and an optional merged save_pretrained().
"""
import torch

torch.set_flush_denormal(True)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

from .device import DutyCycleLimiter, describe, setup_runtime  # noqa: E402
from .lora import apply_lora, lora_state_dict, merge_lora, unwrap_lora  # noqa: E402
from .losses import chunked_cross_entropy  # noqa: E402
from .lowmem import state_bytes  # noqa: E402
from .optim import build_optimizer  # noqa: E402
from .qweights import model_bytes, quantize_model  # noqa: E402
from .zoom import lr_at  # noqa: E402

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "c_attn", "c_proj", "c_fc", "query_key_value", "dense", "dense_h_to_4h",
                   "dense_4h_to_h", "qkv", "proj", "fc", "fc_out", "wq", "wk", "wv", "wo",
                   "w1", "w2", "w3")


# ----------------------------------------------------------------------------- data
def load_examples(args):
    """Return a list of dicts with either {'prompt','response'} or {'messages'}."""
    rows = []
    if args.jsonl:
        with open(args.jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, args.config or None, split=args.split)
        rows = [dict(r) for r in ds]
    if args.max_examples:
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.max_examples]
    out = []
    for r in rows:
        if args.messages_col in r and r[args.messages_col]:
            out.append({"messages": r[args.messages_col]})
            continue
        resp = r.get(args.response_col) or r.get("output") or r.get("completion")
        prompt = r.get(args.prompt_col) or r.get("instruction") or r.get("question")
        if not resp or not prompt:
            continue
        ctx = r.get("context") or r.get("input")
        if ctx:
            prompt = f"{prompt}\n\n{ctx}"
        out.append({"prompt": prompt, "response": resp})
    return out


def render(tok, ex):
    """-> (prompt_text, full_text) using the chat template when available."""
    has_tpl = getattr(tok, "chat_template", None) is not None
    if "messages" in ex:
        msgs = ex["messages"]
        if has_tpl:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(msgs, tokenize=False)
        else:
            prompt = "".join(f"{m['role']}: {m['content']}\n" for m in msgs[:-1]) + "assistant: "
            full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
        return prompt, full
    if has_tpl:
        msgs = [{"role": "user", "content": ex["prompt"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": ex["response"]}],
                                       tokenize=False)
    else:
        prompt = f"### Instruction:\n{ex['prompt']}\n\n### Response:\n"
        full = prompt + ex["response"] + (tok.eos_token or "")
    return prompt, full


def tokenize_examples(tok, examples, seq):
    """-> list of (ids, labels) with labels = -100 on the prompt part."""
    items = []
    for ex in examples:
        prompt, full = render(tok, ex)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        if not f_ids[:len(p_ids)] == p_ids:          # template quirk: fall back to lengths
            p_len = min(len(p_ids), len(f_ids) - 1)
        else:
            p_len = len(p_ids)
        f_ids = f_ids[:seq]
        if p_len >= len(f_ids):
            continue
        labels = [-100] * p_len + f_ids[p_len:]
        items.append((f_ids, labels))
    return items


def pack(items, seq, pad_id):
    """Concatenate examples into windows of `seq` tokens (+1 for the shifted target)."""
    ids, labels = [], []
    for i, l in items:
        ids += i
        labels += l
    windows = []
    for s in range(0, len(ids) - 1, seq):
        x = ids[s:s + seq + 1]
        y = labels[s:s + seq + 1]
        if len(x) < 2:
            break
        if len(x) < seq + 1:
            x = x + [pad_id] * (seq + 1 - len(x))
            y = y + [-100] * (seq + 1 - len(y))
        windows.append((x, y))
    return windows


# ----------------------------------------------------------------------------- model
def lm_loss(model, x, y, ce_chunk):
    """Loss over response tokens; chunked head when the model exposes it."""
    base = getattr(model, "model", None) or getattr(model, "base_model", None)
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if ce_chunk and base is not None and head is not None and hasattr(head, "weight"):
        h = base(input_ids=x).last_hidden_state
        bias = getattr(head, "bias", None)
        return chunked_cross_entropy(h, head.weight, y, ce_chunk, bias=bias)
    logits = model(input_ids=x).logits.float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=-100)


@torch.no_grad()
def evaluate(model, windows, rt, batch, ce_chunk):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(windows), batch):
        xs = torch.tensor([w[0] for w in windows[i:i + batch]], device=rt.device)
        ys = torch.tensor([w[1] for w in windows[i:i + batch]], device=rt.device)
        with rt.autocast():
            loss = lm_loss(model, xs[:, :-1], ys[:, 1:], ce_chunk)
        nv = int((ys[:, 1:] != -100).sum())
        tot += loss.float().item() * nv
        n += nv
    model.train()
    return tot / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--dataset", default="", help="HF dataset id")
    ap.add_argument("--config", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--jsonl", default="", help="local JSONL instead of --dataset")
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--response_col", default="response")
    ap.add_argument("--messages_col", default="messages")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--eval_frac", type=float, default=0.05)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--sched", default="wsd", choices=["cosine", "wsd"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon_lora", type=int, default=0, help="also put LoRA matrices on Muon")
    ap.add_argument("--ns", default="quintic", choices=["quintic", "polar"])
    ap.add_argument("--state_bits", type=int, default=8, choices=[32, 8])
    ap.add_argument("--param_dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--lora", type=int, default=16, help="LoRA rank (0 = full fine-tuning)")
    ap.add_argument("--lora_alpha", type=float, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_targets", default="", help="comma list; default: all projections")
    ap.add_argument("--rslora", type=int, default=0)
    ap.add_argument("--quant", type=int, default=0, choices=[0, 8, 4], help="quantize the frozen base")
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--ce_chunk", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_merged", type=int, default=0, help="also save_pretrained() the merged model")
    args = ap.parse_args(argv)

    # ---- distributed (torchrun) ----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    is_main = rank == 0
    rt = setup_runtime(args.device, args.precision, False, args.threads, args.seed + rank)
    if rt.device.type == "cpu":
        rt.amp_dtype = None
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pdt = args.param_dtype
    if pdt == "auto":
        pdt = "bf16" if (rt.device.type == "cuda" or args.lora == 0) else "fp32"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if pdt == "bf16" else torch.float32)
    base_bytes_fp = model_bytes(model)
    report = {}
    if args.quant:
        report["quant"] = quantize_model(model, args.quant, min_numel=1 << 16)
    if args.lora:
        targets = tuple(t for t in args.lora_targets.split(",") if t) or DEFAULT_TARGETS
        report["lora"] = apply_lora(model, args.lora, args.lora_alpha, args.lora_dropout,
                                    targets, rslora=bool(args.rslora))
        assert report["lora"]["wrapped"] > 0, "no LoRA targets matched; use --lora_targets"
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(rt.device)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = build_optimizer(model, args.optim, args.lr, args.wd, state_bits=args.state_bits,
                          ns=args.ns, muon_lora=bool(args.muon_lora))
    fwd_model = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        fwd_model = DDP(model, device_ids=[rt.device.index] if rt.device.type == "cuda" else None)

    # ---- data ----
    examples = load_examples(args)
    items = tokenize_examples(tok, examples, args.seq)
    random.Random(args.seed).shuffle(items)
    n_eval = max(1, int(len(items) * args.eval_frac))
    eval_w = pack(items[:n_eval], args.seq, tok.pad_token_id)
    train_w = pack(items[n_eval:], args.seq, tok.pad_token_id)
    train_w = train_w[rank::world]
    steps_per_epoch = max(1, len(train_w) // (args.batch * args.accum))
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    limiter = DutyCycleLimiter(args.duty, rt.device)

    log = {"args": vars(args), "report": report, "trainable_params": trainable,
           "total_params": sum(p.numel() for p in model.parameters()),
           "model_bytes_before_quant": base_bytes_fp, "model_bytes": model_bytes(model),
           "optimizer_state_bytes": state_bytes(opt), "train_windows": len(train_w) * world,
           "eval_windows": len(eval_w), "examples": len(items), "steps": total_steps,
           "world_size": world, "runtime": describe(rt), "evals": [], "train": []}
    if is_main:
        print(f"[finetune] {args.model} trainable={trainable/1e6:.2f}M "
              f"model={log['model_bytes']/1e6:.0f}MB state={log['optimizer_state_bytes']/1e6:.0f}MB "
              f"windows={len(train_w)*world} steps={total_steps} {describe(rt)}", flush=True)
        ev = evaluate(model, eval_w, rt, args.batch, args.ce_chunk)
        log["evals"].append({"step": 0, "eval_loss": ev})
        print(f"  step 0 eval={ev:.4f}", flush=True)

    rng = random.Random(args.seed + rank)
    order = list(range(len(train_w)))
    rng.shuffle(order)
    pos, t_train = 0, 0.0
    for step in range(1, total_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step / total_steps, args.lr, warmup=args.warmup, sched=args.sched)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for a in range(args.accum):
            if pos + args.batch > len(order):
                rng.shuffle(order)
                pos = 0
            idx = order[pos:pos + args.batch]
            pos += args.batch
            xs = torch.tensor([train_w[i][0] for i in idx], device=rt.device)
            ys = torch.tensor([train_w[i][1] for i in idx], device=rt.device)
            with rt.autocast():
                loss = lm_loss(fwd_model, xs[:, :-1], ys[:, 1:], args.ce_chunk) / args.accum
            rt.backward(loss, opt, [p for p in model.parameters() if p.requires_grad], step=False)
            loss_acc += loss.float().item()
        rt.step(opt, [p for p in model.parameters() if p.requires_grad])
        dt = time.perf_counter() - t0
        t_train += dt
        limiter(dt)
        if is_main and (step % 10 == 0 or step == 1):
            log["train"].append({"step": step, "loss": loss_acc, "time": t_train})
            print(f"  step {step}/{total_steps} loss={loss_acc:.4f} {dt:.1f}s", flush=True)
        if is_main and args.eval_every and (step % args.eval_every == 0 or step == total_steps):
            ev = evaluate(model, eval_w, rt, args.batch, args.ce_chunk)
            log["evals"].append({"step": step, "eval_loss": ev, "time": t_train})
            print(f"  step {step} eval={ev:.4f}", flush=True)
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump(log, f, indent=1)

    if is_main:
        log["train_time"] = t_train
        log["optimizer_state_bytes"] = state_bytes(opt)      # allocated lazily at step 1
        log["final_eval_loss"] = log["evals"][-1]["eval_loss"]
        if args.lora:
            torch.save(lora_state_dict(model), os.path.join(args.out, "adapter.pt"))
            log["adapter_bytes"] = os.path.getsize(os.path.join(args.out, "adapter.pt"))
        else:
            model.save_pretrained(os.path.join(args.out, "model"))
        if args.save_merged and args.lora and not args.quant:
            merge_lora(model)
            unwrap_lora(model)
            model.save_pretrained(os.path.join(args.out, "merged"))
            tok.save_pretrained(os.path.join(args.out, "merged"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        print(f"[finetune] done: eval {log['evals'][0]['eval_loss']:.4f} -> "
              f"{log['final_eval_loss']:.4f} in {t_train:.0f}s", flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
