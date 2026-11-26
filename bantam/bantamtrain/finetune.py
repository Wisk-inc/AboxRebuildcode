"""Fine-tune any HuggingFace causal LM with the Bantam stack.

    python -m bantamtrain.finetune --model HuggingFaceTB/SmolLM2-135M \
        --dataset databricks/databricks-dolly-15k --max_examples 2000 \
        --lora 16 --quant 4 --optim muon --state_bits 8 --epochs 1 --out runs/dolly

What it gives you
  * LoRA / QLoRA: frozen base (optionally int8/int4 blockwise, bantamtrain.qweights)
    + trainable low-rank adapters on every linear layer; or full fine-tuning (--lora 0).
  * Muon (hidden matrices; the LoRA matrices when the base is frozen) + 8-bit optimizer
    state + bf16 weights with stochastic rounding; chunked cross-entropy so a 32k-128k
    vocabulary never materializes the full logits; activation checkpointing; grad
    accumulation with the loss averaged over every supervised token of the step.
  * SFT data from a HuggingFace dataset or a local JSONL: prompt/response
    columns, Dolly-style instruction/context/response, chat `messages` or ShareGPT
    `conversations`. Uses the tokenizer's chat template when it has one; the loss is
    computed on response tokens only; examples are packed into fixed windows with a
    block-diagonal mask (packed examples never see each other).
  * Multi-GPU with torchrun (DistributedDataParallel), periodic resumable checkpoints
    (--save_every, --resume), adapter-only outputs (MBs, plus the tokenizer) and an
    optional merged save_pretrained().
"""
import torch

torch.set_flush_denormal(True)

import argparse  # noqa: E402
import bisect  # noqa: E402
import contextlib  # noqa: E402
import heapq  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import lora as _lora  # noqa: E402
from . import qweights as _qw  # noqa: E402
from .device import DutyCycleLimiter, describe, setup_runtime  # noqa: E402
from .lora import apply_lora, load_lora_state_dict, lora_state_dict, merge_lora, unwrap_lora  # noqa: E402
from .losses import chunked_cross_entropy  # noqa: E402
from .lowmem import state_bytes  # noqa: E402
from .optim import build_optimizer  # noqa: E402
from .qweights import dequantize_rows, model_bytes, quantize_model  # noqa: E402
from .zoom import lr_at  # noqa: E402

# The pre-0.3 default target list (exact leaf names). The default is now "all-linear".
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "c_attn", "c_proj", "c_fc", "query_key_value", "dense", "dense_h_to_4h",
                   "dense_4h_to_h", "qkv", "proj", "fc", "fc_out", "wq", "wk", "wv", "wo",
                   "w1", "w2", "w3")
_CE_ARGS = inspect.signature(chunked_cross_entropy).parameters     # newer builds: scale, token_weights


# ----------------------------------------------------------------------------- data
_ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
          "chatgpt": "assistant", "bing": "assistant", "bard": "assistant", "model": "assistant",
          "system": "system"}


def _norm_messages(msgs):
    """{role, content} or ShareGPT {from, value} turns -> [{role, content}]; None if malformed."""
    if not isinstance(msgs, list):
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role", m.get("from")), m.get("content", m.get("value"))
        if isinstance(content, list):                     # content parts: keep the text ones
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role is None or not isinstance(content, str):
            return None
        d = {k: v for k, v in m.items() if k not in ("from", "value")}
        d["role"], d["content"] = _ROLES.get(str(role).lower(), role), content
        out.append(d)
    return out


def load_examples(args, stats=None):
    """Return a list of dicts with either {'prompt','response'} or {'messages'}.
    `stats` (a dict) receives the row count and the skipped rows by reason."""
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
    out, skipped = [], {}
    for r in rows:
        msgs = r.get(args.messages_col) or r.get("conversations")
        if msgs:
            msgs = _norm_messages(msgs)
            if msgs:
                out.append({"messages": msgs})
                continue
            reason = "malformed messages"
        else:
            resp = r.get(args.response_col) or r.get("output") or r.get("completion")
            prompt = r.get(args.prompt_col) or r.get("instruction") or r.get("question")
            if resp and prompt:
                ctx = r.get("context") or r.get("input")
                if ctx:
                    prompt = f"{prompt}\n\n{ctx}"
                out.append({"prompt": prompt, "response": resp})
                continue
            reason = "text only" if r.get("text") else "no prompt/response or messages"
        skipped[reason] = skipped.get(reason, 0) + 1
    if stats is not None:
        stats.update(rows=len(rows), usable=len(out), skipped=skipped)
    return out


CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + "
          "message['content'] + '<|im_end|>' + '\n' }}{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")


def resolve_template(tok):
    """Base checkpoints have no chat template: install ChatML when the vocab has its
    tokens (SmolLM2, Qwen2.5 bases) and align eos with the end-of-turn token."""
    if getattr(tok, "chat_template", None):
        return "model"
    if "<|im_end|>" in tok.get_vocab():
        tok.chat_template = CHATML
        tok.eos_token = "<|im_end|>"
        return "chatml-installed"
    return "plain"


def to_messages(ex):
    if "messages" in ex:
        return ex["messages"]
    return [{"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]}]


def trim_to_assistant(msgs):
    """Drop the turns after the last assistant message ([] when there is none), so a
    conversation that ends with a user turn never trains on the user's text."""
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            return msgs[:i + 1]
    return []


def render(tok, ex):
    """-> (prompt_text, full_text) using the chat template when available. The final
    message must be the assistant's (see trim_to_assistant)."""
    has_tpl = getattr(tok, "chat_template", None) is not None
    if "messages" in ex:
        msgs = ex["messages"]
        if has_tpl:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(msgs, tokenize=False)
        else:   # the space before the answer belongs to the answer: BPE merges it ('Ġyes')
            prompt = "".join(f"{m['role']}: {m['content']}\n" for m in msgs[:-1]) + "assistant:"
            full = prompt + " " + msgs[-1]["content"] + (tok.eos_token or "")
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


_GENERATION = re.compile(r"\{%-?\s*generation\s*-?%\}")


def _assistant_mask(tok, msgs):
    """Use the template's {% generation %} blocks when it has them; else None."""
    if not _GENERATION.search(str(getattr(tok, "chat_template", "") or "")):
        return None
    try:
        enc = tok.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True,
                                      add_generation_prompt=False)
    except Exception:
        return None
    mask = enc.get("assistant_masks")
    if not mask or sum(mask) == 0:
        return None
    return list(enc["input_ids"]), mask


def bos_ids(tok):
    """[bos] when the tokenizer prepends BOS by itself (Llama-2 / Mistral style bases) and
    the text is rendered by us (plain format or installed ChatML); [] otherwise."""
    tpl = getattr(tok, "chat_template", None)
    if (tpl and tpl != CHATML) or tok.bos_token_id is None:
        return []
    try:
        ids = tok("a")["input_ids"]
    except Exception:
        return []
    return [tok.bos_token_id] if ids[:1] == [tok.bos_token_id] else []


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _content_start(tok, full, content):
    """Token index of the first token overlapping the last occurrence of `content` in
    `full` (character offsets; fast tokenizers only), or 0 when unknown."""
    c = full.rfind(content) if isinstance(content, str) and content else -1
    if c < 0:
        return 0
    try:
        offs = tok(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    except Exception:
        return 0
    return next((i for i, (_, e) in enumerate(offs) if e > c), 0)


def tokenize_examples(tok, examples, seq, prompt_loss_weight=0.0, truncate=True, stats=None):
    """-> list of (ids, labels, weights): labels = -100 on prompt tokens unless
    prompt_loss_weight > 0, in which case prompt tokens get that weight and
    response tokens weight 1. Multi-turn chats get every assistant turn supervised
    when the template marks generation spans; otherwise the final turn only.

    The prompt/response boundary is the longest common token prefix of the rendered
    prompt and the full text, so the supervised tokens are exactly what generation
    after the prompt has to produce. Trailing non-assistant turns are dropped. With
    truncate=False (--pack wrapped) long examples are kept whole. `stats` (a dict)
    receives the kept / dropped / truncated / boundary-fallback counts."""
    st = {"examples": len(examples), "kept": 0, "no_assistant_turn": 0, "truncated": 0,
          "no_target_after_truncation": 0, "boundary_fallback": 0}
    bos = bos_ids(tok)
    items = []
    for ex in examples:
        if "messages" in ex:
            msgs = trim_to_assistant(ex["messages"])
            if not msgs:
                st["no_assistant_turn"] += 1
                continue
            ex = {"messages": msgs}
        am = _assistant_mask(tok, to_messages(ex))
        if am is not None:
            f_ids, mask = am
            sup = [bool(m) for m in mask]
        else:
            prompt, full = render(tok, ex)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            off = 0
            if bos and not prompt.startswith(tok.bos_token):
                p_ids, f_ids, off = bos + p_ids, bos + f_ids, len(bos)
            p_len = _common_prefix(p_ids, f_ids)
            if p_len < len(p_ids):         # the prompt is not a token prefix of the full text
                st["boundary_fallback"] += 1
                c = _content_start(tok, full, to_messages(ex)[-1]["content"])
                p_len = max(p_len, c + off if c else 0)
            sup = [i >= p_len for i in range(len(f_ids))]
        if truncate and len(f_ids) > seq:
            st["truncated"] += 1
            f_ids, sup = f_ids[:seq], sup[:seq]
        if not any(sup):
            st["no_target_after_truncation"] += 1
            continue
        labels = [t if s_ or prompt_loss_weight > 0 else -100 for t, s_ in zip(f_ids, sup)]
        weights = [1.0 if s_ else prompt_loss_weight for s_ in sup]
        items.append((f_ids, labels, weights))
    st["kept"] = len(items)
    if stats is not None:
        stats.update(st)
    return items


def with_prompt_loss(items, weight):
    """Response-only items (tokenize_examples with prompt_loss_weight=0) -> the same items
    with every prompt token supervised at `weight`, as tokenize_examples(..., weight) gives."""
    if not weight:
        return items
    return [(ids, list(ids), [1.0 if l != -100 else weight for l in labels]) for ids, labels, _ in items]


def _pad(x, y, w, g, n, pad_id):
    k = n - len(x)
    return x + [pad_id] * k, y + [-100] * k, w + [0.0] * k, g + [-1] * k


def pack(items, seq, pad_id, mode="bfd"):
    """Group tokenized examples into windows of seq+1 tokens.

    mode="bfd":     best-fit-decreasing bin packing, examples never split (TRL default)
    mode="wrapped": concatenate everything and cut every seq tokens (may split examples)
    mode="none":    one example per window, padded
    Each window is (ids, labels, weights, segments) of length seq+1 (inputs = [:-1],
    targets = [1:]). `segments` numbers the examples inside the window (-1 = padding);
    lm_loss turns it into a block-diagonal causal mask with per-example positions, so a
    packed example never attends to its neighbours. The first token of every example
    is never a target (it would be predicted from the previous example).
    """
    n = seq + 1
    items = [(i, [-100] + l[1:], w) for i, l, w in items if len(i) >= 2]
    if mode != "wrapped":                               # one example per bin: truncate to fit
        items = [(i[:n], l[:n], w[:n]) for i, l, w in items]
    if mode == "wrapped":
        ids, labels, ws, gs = [], [], [], []
        for e, (i, l, w) in enumerate(items):
            ids += i
            labels += l
            ws += w
            gs += [e] * len(i)
        out = []
        for s in range(0, len(ids) - 1, seq):
            x, y, w, g = ids[s:s + n], labels[s:s + n], ws[s:s + n], gs[s:s + n]
            out.append(_pad(x, y, w, [v - g[0] for v in g], n, pad_id))
        return out
    if mode == "none":
        return [_pad(i, l, w, [0] * len(i), n, pad_id) for i, l, w in items]
    # best-fit decreasing in O(n log n): the tightest bin is the smallest free capacity
    # >= len (bisect over the distinct capacities), ties go to the oldest bin (a heap)
    bins, caps, by_cap = [], [], {}
    for i, l, w in sorted(items, key=lambda t: -len(t[0])):
        k = bisect.bisect_left(caps, len(i))
        if k < len(caps):
            c = caps[k]
            b = heapq.heappop(by_cap[c])
            if not by_cap[c]:
                del by_cap[c]
                caps.pop(k)
        else:
            c, b = n, len(bins)
            bins.append({"x": [], "y": [], "w": [], "g": []})
        best = bins[b]
        best["g"] += [best["g"][-1] + 1 if best["g"] else 0] * len(i)
        best["x"] += i
        best["y"] += l
        best["w"] += w
        c -= len(i)
        if c > 0:
            if c not in by_cap:
                bisect.insort(caps, c)
                by_cap[c] = []
            heapq.heappush(by_cap[c], b)
    return [_pad(b["x"], b["y"], b["w"], b["g"], n, pad_id) for b in bins]


def padding_fraction(windows):
    """Share of window positions that are padding (segment id -1), whatever the pad id."""
    return sum(w[3].count(-1) for w in windows) / max(1, sum(len(w[3]) for w in windows))


def plan_steps(n_windows, world, batch, accum, epochs, max_steps):
    """-> (windows per rank, total optimizer steps), the same on every rank: the windows
    are cut to a multiple of `world` (drop_last) before sharding."""
    per = n_windows // world
    steps_per_epoch = max(1, per // (batch * accum))
    return per, max_steps or max(1, int(steps_per_epoch * epochs))


def segment_mask(seg):
    """(B, T) example ids (-1 = padding) -> (B, 1, T, T) bool block-diagonal causal mask
    and (B, T) position ids restarting at 0 for every example; None, None when every
    row is a single unpadded example (keeps the plain causal fast path)."""
    if bool((seg == seg[:, :1]).all()):
        return None, None
    B, T = seg.shape
    idx = torch.arange(T, device=seg.device).expand(B, T)
    start = torch.ones_like(seg, dtype=torch.bool)
    start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    pos = idx - torch.where(start, idx, torch.zeros_like(idx)).cummax(dim=1).values
    causal = torch.ones(T, T, dtype=torch.bool, device=seg.device).tril()
    return ((seg[:, :, None] == seg[:, None, :]) & causal)[:, None], pos


# ----------------------------------------------------------------------------- model
def head_transform(model):
    """(scale, softcap) the model's own forward applies to the output head's logits:
    Gemma-2/3 final_logit_softcapping, Cohere logit_scale, Granite / MiniCPM3
    1/logits_scaling, Falcon-H1 lm_head_multiplier; (1.0, 0.0) for a plain head."""
    cfg = getattr(model, "config", None)
    if cfg is not None and callable(getattr(cfg, "get_text_config", None)):
        cfg = cfg.get_text_config()

    def get(k):
        v = getattr(cfg, k, None)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v else None

    scale = (get("logit_scale") or 1.0) * (get("lm_head_multiplier") or 1.0) / (get("logits_scaling") or 1.0)
    return scale, get("final_logit_softcapping") or 0.0


def _split_head(model):
    """(decoder, output head) when the logits are head(decoder(x).last_hidden_state)."""
    base = getattr(model, "model", None) or getattr(model, "base_model", None)
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if base is None or base is model or head is None or not hasattr(head, "weight"):
        return None, None
    return base, head


def _compute_dtype(h):
    """The dtype a linear layer on `h` computes in: the autocast dtype when autocast is on."""
    t = h.device.type
    if hasattr(torch, "get_autocast_dtype"):                       # torch >= 2.4
        on, dt = torch.is_autocast_enabled(t), torch.get_autocast_dtype(t)
    elif t == "cpu":
        on, dt = torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()
    else:
        on, dt = torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()
    return dt if on else h.dtype


def _head_weight(head, dtype):
    """The head weight in the compute dtype. A quantized head is dequantized straight to
    that dtype instead of fp32: the chunked loss keeps it alive until backward."""
    if all(hasattr(head, a) for a in ("q", "s", "pad", "bits", "block")):
        return dequantize_rows(head.q, head.s, head.pad, head.bits, head.block, dtype)
    return head.weight


def _transformed_logits(h, head, scale, softcap):
    z = F.linear(h, _head_weight(head, _compute_dtype(h)), getattr(head, "bias", None)).float() * scale
    return softcap * torch.tanh(z / softcap) if softcap > 0 else z


@torch.no_grad()
def _chunk_ok(model, base, head):
    """Once per model: does head(decoder(x)) with head_transform reproduce the model's own
    logits? If not (an architecture with a post-head step this code does not know), the
    loss uses the model's full logits instead of the chunked head."""
    ok = model.__dict__.get("_bantam_chunk_ok")
    if ok is None:
        was = model.training
        model.eval()
        try:
            dev = next(model.parameters()).device
            vocab = getattr(head, "out_features", None) or getattr(head, "num_embeddings", None) \
                or head.weight.shape[0]
            x = (torch.arange(8, device=dev) * 7 % vocab)[None]
            ref = model(input_ids=x).logits.float()
            got = _transformed_logits(base(input_ids=x).last_hidden_state, head, *head_transform(model))
            ok = got.shape == ref.shape and bool((got - ref).abs().max() <= 2e-2 * (1 + ref.abs().max()))
        except Exception:
            ok = False
        finally:
            model.train(was)
        if not ok:
            print(f"[finetune] {type(model).__name__}: output head is not a linear map plus a known "
                  "transform; using the model's full logits (--ce_chunk has no effect)", flush=True)
        model.__dict__["_bantam_chunk_ok"] = ok
    return ok


def _token_weights(y, w):
    """-> (per-target weights, or None when every target has weight 1; normaliser = the
    number of targets or their total weight)."""
    valid = y != -100
    if w is not None and bool(((w != 1.0) & valid).any()):
        w = w.float() * valid
        return w, w.sum()
    return None, valid.sum().float()


def lm_loss_sum(model, x, y, ce_chunk, w=None, seg=None):
    """-> (summed CE over supervised targets, weighted by w; its normaliser). Uses the
    chunked head (never the full logits) when the model is decoder + linear head; the
    model's own logit transform (softcap / scale) is applied either way. `seg` (B, T)
    keeps packed examples apart. Targets that are all ignored give 0, never NaN."""
    kw = {}
    if seg is not None:
        mask, pos = segment_mask(seg)
        if mask is not None:
            kw = {"attention_mask": mask, "position_ids": pos}
    tw, denom = _token_weights(y, w)
    base, head = _split_head(model) if ce_chunk else (None, None)
    if head is not None and (tw is None or "token_weights" in _CE_ARGS) and _chunk_ok(model, base, head):
        h = base(input_ids=x, **kw).last_hidden_state
        scale, softcap = head_transform(model)
        bias = getattr(head, "bias", None)
        if scale != 1.0 and "scale" not in _CE_ARGS:      # scale*(hW^T + b) = (scale*h)W^T + scale*b
            h, bias, scale = h * scale, None if bias is None else bias * scale, 1.0
        opt = {"softcap": softcap} if softcap else {}
        if scale != 1.0:
            opt["scale"] = scale
        if tw is not None:              # unit total weight: the returned mean times denom is the sum
            opt["token_weights"] = tw / denom
        mean = chunked_cross_entropy(h, _head_weight(head, _compute_dtype(h)), y, ce_chunk,
                                     bias=bias, **opt)
        return mean * denom, denom
    logits = model(input_ids=x, **kw).logits.float()
    per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                          ignore_index=-100, reduction="none")
    return ((per * tw.reshape(-1)).sum() if tw is not None else per.sum()), denom


def lm_loss(model, x, y, ce_chunk, w=None, seg=None):
    """Mean CE over supervised tokens (optionally weighted); see lm_loss_sum."""
    s, n = lm_loss_sum(model, x, y, ce_chunk, w, seg)
    return s / n.clamp_min(1)


class LossModule(nn.Module):
    """Model + loss in one forward, so DDP wraps the chunked head as well (calling the
    inner model directly would skip DDP's gradient hooks). Returns the summed loss."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, w=None, seg=None, ce_chunk=2048):
        return lm_loss_sum(self.model, x, y, ce_chunk, w, seg)[0]


def wrap_ddp(model, rt):
    """DDP around LossModule(model). Buffers (quantized base codes, rotary tables) are
    frozen and identical on every rank, so they are not re-broadcast every forward."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    params = inspect.signature(DDP.__init__).parameters
    kw = {"forward_sync_buffers": False} if "forward_sync_buffers" in params else {"broadcast_buffers": False}
    return DDP(LossModule(model), device_ids=[rt.device.index] if rt.device.type == "cuda" else None,
               gradient_as_bucket_view=True, **kw)


def _batch(windows, idx, device):
    return [torch.tensor([windows[i][k] for i in idx], device=device) for k in range(4)]


def accumulate(fwd, mbs, rt, ce_chunk):
    """Forward + backward over one optimizer step's micro-batches [(ids, labels, weights,
    segments)] through `fwd` (LossModule, or DDP around it). The step's loss is the sum of
    token losses over the supervised tokens of the whole step (every micro-batch, every
    rank) divided by their number, so --batch 2 --accum 2 equals --batch 4; under DDP the
    gradients are all-reduced once, after the last micro-batch.
    -> (mean token loss of the step, number of supervised targets)."""
    ddp = hasattr(fwd, "no_sync")
    world = dist.get_world_size() if ddp else 1
    n_tok = sum(_token_weights(ys[:, 1:], ws[:, 1:])[1].double() for _, ys, ws, _ in mbs)
    if ddp:
        dist.all_reduce(n_tok)
    norm = world / n_tok.clamp_min(1e-12).float()       # DDP averages gradients over ranks
    loss_sum = torch.zeros((), dtype=torch.float64, device=rt.device)
    for a, (xs, ys, ws, gs) in enumerate(mbs):
        with fwd.no_sync() if ddp and a < len(mbs) - 1 else contextlib.nullcontext():
            with rt.autocast():
                s = fwd(xs[:, :-1], ys[:, 1:], ws[:, 1:], gs[:, :-1], ce_chunk)
            rt.backward(s * norm, None, None, step=False)
        loss_sum += s.detach().double()
    if ddp:
        dist.all_reduce(loss_sum)
    return ((loss_sum / n_tok).item() if n_tok > 0 else 0.0), n_tok.item()


@torch.no_grad()
def evaluate(model, windows, rt, batch, ce_chunk, world=1, rank=0):
    """Mean CE over response tokens only: targets of weight != 1 (prompt tokens under
    0 < --prompt_loss_weight < 1) are excluded, and main() builds its eval windows
    response-only for any weight. With world > 1 every rank must call it: each scores
    windows[rank::world] and the sums are all-reduced. NaN for an empty set."""
    was = model.training
    model.eval()
    tot = torch.zeros(2, dtype=torch.float64, device=rt.device)
    mine = windows[rank::world] if world > 1 else windows
    for i in range(0, len(mine), batch):
        xs, ys, ws, gs = _batch(mine, range(i, min(i + batch, len(mine))), rt.device)
        y = ys.masked_fill(ws != 1.0, -100)[:, 1:]
        if not bool((y != -100).any()):
            continue
        with rt.autocast():
            s, n = lm_loss_sum(model, xs[:, :-1], y, ce_chunk, seg=gs[:, :-1])
        tot += torch.stack([s.double(), n.double()])
    if world > 1:
        dist.all_reduce(tot)
    model.train(was)
    return (tot[0] / tot[1]).item() if tot[1] > 0 else float("nan")


def linear_targets(model):
    """Full names of every linear layer (nn.Linear, QLinear, transformers Conv1D) except
    the output head: the "all-linear" target set, as in PEFT."""
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return tuple(n for n, m in model.named_modules()
                 if n and m is not out and (isinstance(m, nn.Linear) or type(m).__name__ in ("QLinear", "Conv1D")))


def lora_module_names(model):
    return [n for n, m in model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "base")]


def _apply_lora(model, args, rslora):
    """apply_lora with the CLI options; "all-linear" goes to apply_lora when it knows it
    (it then also takes `exclude`), else every linear layer is listed by name."""
    params = inspect.signature(apply_lora).parameters
    kw = {"rslora": rslora}
    if args.dora:
        if "dora" not in params:
            raise SystemExit("[finetune] --dora needs bantamtrain.lora.apply_lora(dora=...), "
                             "which this build does not have")
        kw["dora"] = True
    spec = args.lora_targets.strip()
    if spec in ("", "all-linear"):
        targets = "all-linear" if "exclude" in params else linear_targets(model)
    else:
        targets = tuple(t for t in spec.split(",") if t)
    rep = apply_lora(model, args.lora, args.lora_alpha, args.lora_dropout, targets, **kw)
    rep["targets"] = sorted({n.split(".")[-1] for n in lora_module_names(model)})
    return rep


def _match_embedding_dtype(model, dtype):
    """QEmbedding returns fp32; in a bf16 model without autocast that activation meets bf16
    Linear weights and fails. Cast its output to the model's float dtype."""
    def hook(mod, inp, out):
        return out.to(dtype) if out.is_floating_point() and out.dtype != dtype else out
    for m in model.modules():
        if type(m).__name__ == "QEmbedding":
            m.register_forward_hook(hook)


def _quantized_params(model):
    return sum(m.q.numel() * (2 if getattr(m, "bits", 8) in (4, "nf4") else 1)
               for m in model.modules() if hasattr(m, "q") and hasattr(m, "s"))


def _fix_eos(model, tok, template):
    """ChatML installed on a base checkpoint: generation must stop at <|im_end|>, the
    turn terminator the model was trained on, as well as at the original eos."""
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "pad_token_id", None) is None:
        gc.pad_token_id = tok.pad_token_id
    if template != "chatml-installed":
        return
    eos, old = tok.eos_token_id, model.config.eos_token_id
    ids = [eos] + [i for i in (old if isinstance(old, list) else [old]) if i is not None and i != eos]
    model.config.eos_token_id = eos
    if gc is not None:
        gc.eos_token_id = ids if len(ids) > 1 else eos


def _save_merged_from_float(model_id, dtype, sd, meta, tok, template, out_dir):
    """QLoRA merge: reload the float base, apply the trained adapters, merge, save."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    kw = {"rslora": meta["rslora"]}
    if meta.get("dora"):
        kw["dora"] = True
    apply_lora(m, meta["r"], meta["alpha"], 0.0, tuple(meta["modules"]), **kw)
    load_lora_state_dict(m, sd)
    merge_lora(m)
    unwrap_lora(m)
    _fix_eos(m, tok, template)
    m.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


# ----------------------------------------------------------------------------- checkpoints
_RESUME_KEYS = ("model", "dataset", "config", "split", "jsonl", "eval_jsonl", "max_examples", "prompt_col",
                "response_col", "messages_col", "lora", "lora_alpha", "rslora", "dora", "lora_plus", "quant",
                "embed_quant", "param_dtype", "seq", "pack", "batch", "accum", "seed", "lr", "epochs",
                "max_steps", "prompt_loss_weight", "optim", "muon_lora", "state_bits", "sched", "warmup",
                "eval_frac")


def _state_path(path):
    for p in (path, os.path.join(path, "state.pt"), os.path.join(path, "checkpoint", "state.pt")):
        if os.path.isfile(p):
            return p
    raise SystemExit(f"[finetune] --resume: no checkpoint at {path}")


def save_state(out, model, opt, rt, step, data, log, times, args):
    """Atomically write <out>/checkpoint/state.pt: trainable params (adapters, or the whole
    model for --lora 0), optimizer state, step, every rank's data order / position and
    shuffle / torch / CUDA RNG states (`data`), and the log so far."""
    d = os.path.join(out, "checkpoint")
    os.makedirs(d, exist_ok=True)
    state = {"version": 1, "step": step, "args": vars(args), "log": log, "times": list(times),
             "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
             "opt": opt.state_dict(), "data": data,
             "scaler": rt.scaler.state_dict() if rt.scaler is not None else None}
    tmp = os.path.join(d, "state.pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(d, "state.pt"))


def load_state(path):
    """torch.load(weights_only=True); 8-bit optimizer state objects (lowmem.QState) are
    allowlisted for optimizers whose state_dict still holds them."""
    from . import lowmem
    safe = [c for c in (getattr(lowmem, "QState", None),) if c is not None]
    allow = getattr(torch.serialization, "safe_globals", None)       # torch >= 2.5
    with allow(safe) if allow is not None else contextlib.nullcontext():
        return torch.load(_state_path(path), map_location="cpu", weights_only=True)


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
    ap.add_argument("--eval_jsonl", default="", help="held-out JSONL (same columns); overrides --eval_frac")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--pack", default="bfd", choices=["bfd", "wrapped", "none"])
    ap.add_argument("--prompt_loss_weight", type=float, default=0.0,
                    help="0 = loss on response tokens only (default); 1 = train on everything")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--sched", default="wsd", choices=["cosine", "wsd"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon_lora", type=int, default=-1,
                    help="put the LoRA matrices on Muon; -1: auto (on with --optim muon and --lora > 0, "
                         "where the frozen base leaves Muon nothing else)")
    ap.add_argument("--ns", default="quintic", choices=["quintic", "polar"])
    ap.add_argument("--state_bits", type=int, default=8, choices=[32, 8])
    ap.add_argument("--param_dtype", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--lora", type=int, default=16, help="LoRA rank (0 = full fine-tuning)")
    ap.add_argument("--lora_alpha", type=float, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", default="",
                    help="comma list of module names, or all-linear (default): every linear layer "
                         "except the output head")
    ap.add_argument("--rslora", type=int, default=-1, help="-1: auto (on when r >= 64)")
    ap.add_argument("--dora", type=int, default=0, help="DoRA adapters (needs bantamtrain.lora support)")
    ap.add_argument("--lora_plus", type=float, default=0.0,
                    help="LoRA+: the B matrices get lr x this ratio (e.g. 16); 0 = off")
    ap.add_argument("--quant", default="0", choices=["0", "8", "4", "nf4"],
                    help="quantize the frozen base: 8, 4 (linear int4) or nf4 (QLoRA codebook)")
    ap.add_argument("--embed_quant", default="8", choices=["same", "8", "none"],
                    help="embeddings / output head: int8 (default), same as --quant, or float")
    ap.add_argument("--low_ram", type=int, default=0,
                    help="with --quant: load and quantize tensor by tensor (qweights.load_quantized)")
    ap.add_argument("--eval_only", type=int, default=0, help="evaluate the held-out loss and exit")
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--ce_chunk", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=50, help="0: evaluate only at step 0 and the end")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_merged", type=int, default=0,
                    help="also save_pretrained() the merged model to <out>/merged (with --quant the "
                         "adapters are merged into the reloaded float base)")
    ap.add_argument("--save_peft", type=int, default=-1,
                    help="also write a PEFT adapter (adapter_config.json + adapter_model.safetensors); "
                         "-1: auto (on when bantamtrain.lora.save_peft_adapter exists)")
    ap.add_argument("--save_every", type=int, default=0,
                    help="write a resumable checkpoint to <out>/checkpoint every N steps")
    ap.add_argument("--resume", default="", help="resume from a checkpoint dir (or its state.pt)")
    ap.add_argument("--stop_step", type=int, default=0, help="stop after this many steps (tests)")
    args = ap.parse_args(argv)
    quant = {"0": 0, "8": 8, "4": 4, "nf4": "nf4"}[str(args.quant)]
    if args.lora == 0 and quant:
        ap.error("--quant quantizes the frozen base for QLoRA and needs --lora > 0 "
                 "(full fine-tuning of a quantized model is not supported)")
    if (args.dora or args.lora_plus) and not args.lora:
        ap.error("--dora / --lora_plus need --lora > 0")
    if args.resume and args.eval_only:
        ap.error("--resume continues training; --eval_only does not train")

    # ---- distributed (torchrun) ----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    is_main = rank == 0
    # the same seed on every rank: identical model init and stochastic rounding
    rt = setup_runtime(args.device, args.precision, False, args.threads, args.seed)
    if rt.device.type == "cpu":
        rt.amp_dtype = None
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    template = resolve_template(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pdt = args.param_dtype
    if pdt == "auto":
        pdt = "bf16" if (rt.device.type == "cuda" or args.lora == 0) else "fp32"
    dtype = torch.bfloat16 if pdt == "bf16" else torch.float32
    report = {}
    eq = {"same": "same", "8": 8, "none": None}[args.embed_quant]
    load_quantized = getattr(_qw, "load_quantized", None)
    if quant and args.low_ram and load_quantized is not None:
        res = load_quantized(args.model, quant, embed_bits=eq, dtype=dtype)
        model, rq = res if isinstance(res, tuple) else (res, None)
        report["quant"] = rq or {"bits": quant, "low_ram": True, "bytes_after": model_bytes(model)}
        base_bytes_fp = (model_bytes(model) - sum(m.q.numel() * m.q.element_size() + m.s.numel() * 2
                                                  for m in model.modules() if hasattr(m, "q") and hasattr(m, "s"))
                         + _quantized_params(model) * dtype.itemsize)
    else:
        if args.low_ram and is_main:
            print("[finetune] note: --low_ram " + ("needs --quant" if not quant else
                  "needs bantamtrain.qweights.load_quantized; loading normally"), flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        base_bytes_fp = model_bytes(model)
        if quant:
            report["quant"] = quantize_model(model, quant, min_numel=1 << 16, embed_bits=eq)
    if quant and dtype == torch.bfloat16:
        _match_embedding_dtype(model, dtype)
    rslora = (args.lora >= 64) if args.rslora < 0 else bool(args.rslora)
    if args.lora:
        report["lora"] = _apply_lora(model, args, rslora)
        if not report["lora"]["wrapped"]:
            raise SystemExit("[finetune] no LoRA target matched; check --lora_targets (GPT-2 style "
                             "Conv1D layers need a bantamtrain.lora with Conv1D support)")
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
    params = [p for p in model.parameters() if p.requires_grad]

    opt, optimizers = None, []
    if not args.eval_only:
        if args.muon_lora < 0:
            args.muon_lora = int(args.optim == "muon" and args.lora > 0)
            if args.muon_lora and is_main:
                print("[finetune] --optim muon with LoRA: the base is frozen, so Muon runs on the LoRA "
                      "matrices (--muon_lora 1); pass --muon_lora 0 for AdamW only", flush=True)
        elif args.optim == "muon" and args.lora and not args.muon_lora and is_main:
            print("[finetune] warning: --optim muon with a frozen base and --muon_lora 0 puts no "
                  "parameter on Muon; every adapter uses AdamW", flush=True)
        okw = {"state_bits": args.state_bits, "ns": args.ns, "muon_lora": bool(args.muon_lora)}
        bparams = inspect.signature(build_optimizer).parameters
        if "sr_seed" in bparams:
            okw["sr_seed"] = args.seed
        if args.lora_plus:
            if "lr_scale" not in bparams:
                raise SystemExit("[finetune] --lora_plus needs build_optimizer(lr_scale=...), "
                                 "which this build does not have")
            ratio = args.lora_plus
            okw["lr_scale"] = lambda n, p: ratio if "lora_B" in n else 1.0
        opt = build_optimizer(model, args.optim, args.lr, args.wd, **okw)
        optimizers = [[type(o).__name__, sum(p.numel() for g in o.param_groups for p in g["params"])]
                      for o in getattr(opt, "optimizers", [opt])]
    fwd = wrap_ddp(model, rt) if (world > 1 and not args.eval_only) else LossModule(model)

    # ---- data ----
    dstats = {}
    examples = load_examples(args, dstats)
    if not examples:
        raise SystemExit(f"[finetune] no usable rows in {args.jsonl or args.dataset} "
                         f"({dstats['rows']} rows, skipped {dstats['skipped']}): expected prompt/response "
                         "(or instruction/output, question/completion), chat `messages` or ShareGPT "
                         "`conversations` columns")
    trunc = args.pack != "wrapped"
    # response-only labels; --prompt_loss_weight is applied to the training split below, so the
    # eval windows score response tokens only (at weight 1 prompt and response look the same)
    items = tokenize_examples(tok, examples, args.seq, 0.0, trunc, dstats)
    if not items:
        raise SystemExit(f"[finetune] no example has a supervised token after tokenization: {dstats}")
    dropped = dstats["examples"] - dstats["kept"]
    if is_main and (dropped or dstats["truncated"] or dstats["skipped"] or dstats["boundary_fallback"]):
        print(f"[finetune] data: kept {dstats['kept']}/{dstats['rows']} rows (skipped {dstats['skipped']}, "
              f"no assistant turn {dstats['no_assistant_turn']}, no target left after truncation to --seq "
              f"{dstats['no_target_after_truncation']}); truncated {dstats['truncated']}; prompt not a token "
              f"prefix {dstats['boundary_fallback']}", flush=True)
    random.Random(args.seed).shuffle(items)
    if args.eval_jsonl:
        ev_args = argparse.Namespace(**{**vars(args), "jsonl": args.eval_jsonl, "max_examples": 0})
        eval_items, n_eval = tokenize_examples(tok, load_examples(ev_args), args.seq, truncate=trunc), 0
    else:
        n_eval = max(1, int(len(items) * args.eval_frac))
        eval_items = items[:n_eval]
        if len(items) <= n_eval and not args.eval_only:
            raise SystemExit(f"[finetune] {len(items)} usable example(s): too few to hold out "
                             f"{n_eval} for evaluation and train on the rest (add data or pass --eval_jsonl)")
    eval_w = pack(eval_items, args.seq, tok.pad_token_id, args.pack)
    train_w = pack(with_prompt_loss(items[n_eval:], args.prompt_loss_weight), args.seq, tok.pad_token_id,
                   args.pack)
    if not eval_w:
        raise SystemExit("[finetune] the evaluation set is empty")
    n_sup = sum(1 for w in train_w for t in w[1][1:] if t != -100)
    pad_frac = padding_fraction(train_w)
    per_rank, total_steps = plan_steps(len(train_w), world, args.batch, args.accum, args.epochs,
                                       args.max_steps)
    if per_rank == 0 and not args.eval_only:
        raise SystemExit(f"[finetune] {len(train_w)} training window(s) for {world} ranks")
    train_w = train_w[:per_rank * world][rank::world]
    if world > 1:            # belt and braces: a rank with fewer steps would hang the others
        t = torch.tensor(total_steps, device=rt.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        total_steps = int(t)
    limiter = DutyCycleLimiter(args.duty, rt.device)

    n_quant = _quantized_params(model)
    scale, softcap = head_transform(model)
    log = {"args": vars(args), "report": report, "trainable_params": trainable,
           "total_params": sum(p.numel() for p in model.parameters()) + n_quant,
           "quantized_base_params": n_quant,
           "model_bytes_before_quant": base_bytes_fp, "model_bytes": model_bytes(model),
           "optimizer_state_bytes": state_bytes(opt) if opt is not None else 0,
           "optimizers": optimizers, "train_windows": per_rank * world,
           "eval_windows": len(eval_w), "examples": len(items), "steps": total_steps,
           "chat_template": template, "supervised_tokens": n_sup, "padding_fraction": pad_frac,
           "data": dstats, "logit_transform": {"scale": scale, "softcap": softcap},
           "world_size": world, "runtime": describe(rt), "evals": [], "train": []}
    log_path = os.path.join(args.out, "log.json")

    def write_log():
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)

    rng = random.Random(args.seed + rank)                 # rank offsets only for the data order
    order = list(range(len(train_w)))
    rng.shuffle(order)
    pos, t_train, cpu_train, start = 0, 0.0, 0.0, 0
    if args.resume:
        st = load_state(args.resume)
        bad = [k for k in _RESUME_KEYS if st["args"].get(k) != getattr(args, k)]
        if bad or len(st["data"]) != world:
            raise SystemExit(f"[finetune] --resume: options differ from the checkpoint: {bad or 'world size'}")
        if len(st["data"][rank]["order"]) != len(train_w):
            raise SystemExit(f"[finetune] --resume: {len(train_w)} training windows per rank, the checkpoint "
                             f"has {len(st['data'][rank]['order'])}: the data differ")
        sd = st["trainable"]
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(sd) != names:
            raise SystemExit("[finetune] --resume: the checkpoint's trainable parameters do not match")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in sd:
                    p.copy_(sd[n])
        opt.load_state_dict(st["opt"])
        if rt.scaler is not None and st["scaler"] is not None:
            rt.scaler.load_state_dict(st["scaler"])
        own = st["data"][rank]
        order, pos, rs = own["order"], own["pos"], own["rng"]
        rng.setstate((rs[0], tuple(rs[1]), rs[2]))
        torch.set_rng_state(own["torch_rng"])
        if own["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(own["cuda_rng"])        # this rank's device
        start, (t_train, cpu_train) = st["step"], st["times"]
        log = {**st["log"], "args": vars(args), "resumed_from": args.resume}
    if is_main:
        print(f"[finetune] {args.model} trainable={trainable/1e6:.2f}M "
              f"model={log['model_bytes']/1e6:.0f}MB state={log['optimizer_state_bytes']/1e6:.0f}MB "
              f"windows={per_rank * world} steps={total_steps} {describe(rt)}"
              + (f" resumed at step {start}" if start else ""), flush=True)
    if not start:
        ev = evaluate(model, eval_w, rt, args.batch, args.ce_chunk, world, rank)
        log["evals"].append({"step": 0, "eval_loss": ev})
        log["chunked_ce"] = bool(model.__dict__.get("_bantam_chunk_ok")) and bool(args.ce_chunk)
        if is_main:
            print(f"  step 0 eval={ev:.4f}", flush=True)
    if args.eval_only:
        if is_main:
            log["final_eval_loss"] = log["evals"][-1]["eval_loss"]
            write_log()
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    def data_state():
        mine = {"order": order, "pos": pos, "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
        if world == 1:
            return [mine]
        allst = [None] * world
        dist.all_gather_object(allst, mine)
        return allst

    for step in range(start + 1, total_steps + 1):
        lr = lr_at(step / total_steps, args.lr, warmup=args.warmup, sched=args.sched)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        t0, c0 = time.perf_counter(), time.process_time()
        opt.zero_grad(set_to_none=True)
        mbs = []
        for _ in range(args.accum):
            if pos + args.batch > len(order):
                rng.shuffle(order)
                pos = 0
            mbs.append(_batch(train_w, order[pos:pos + args.batch], rt.device))
            pos += args.batch
        loss_v, n_tok = accumulate(fwd, mbs, rt, args.ce_chunk)
        rt.step(opt, params)
        dt = time.perf_counter() - t0
        t_train += dt
        cpu_train += time.process_time() - c0
        limiter(dt)
        if is_main and (step % 10 == 0 or step == 1 or step == total_steps):
            log["train"].append({"step": step, "loss": loss_v, "time": t_train, "lr": lr,
                                 "targets": n_tok})
            print(f"  step {step}/{total_steps} loss={loss_v:.4f} {dt:.1f}s", flush=True)
        if (args.eval_every and step % args.eval_every == 0) or step == total_steps:
            ev = evaluate(model, eval_w, rt, args.batch, args.ce_chunk, world, rank)
            log["evals"].append({"step": step, "eval_loss": ev, "time": t_train})
            if is_main:
                print(f"  step {step} eval={ev:.4f}", flush=True)
                write_log()
        stop = bool(args.stop_step and step >= args.stop_step and step < total_steps)
        if args.save_every and step < total_steps and (step % args.save_every == 0 or stop):
            ds = data_state()
            if is_main:
                save_state(args.out, model, opt, rt, step, ds, log, (t_train, cpu_train), args)
        if stop:
            if is_main:
                print(f"  stopped at step {step} (--stop_step)", flush=True)
                log["in_progress"] = True
                write_log()
            if world > 1:
                dist.barrier()
                dist.destroy_process_group()
            return

    if world > 1:          # replicas must be identical (same seed, synchronized updates)
        ck = torch.stack([p.detach().double().sum() for p in params]).sum().reshape(1)
        cks = [torch.zeros_like(ck) for _ in range(world)]
        dist.all_gather(cks, ck)
        log["replica_checksums"] = [c.item() for c in cks]
    if is_main:
        log.pop("in_progress", None)
        log["train_time"] = t_train
        log["train_cpu_time"] = cpu_train
        log["optimizer_state_bytes"] = state_bytes(opt)      # allocated lazily at step 1
        log["final_eval_loss"] = log["evals"][-1]["eval_loss"]
        _fix_eos(model, tok, template)
        tok.save_pretrained(args.out)            # the chat template and eos the adapters were trained with
        if args.lora:
            sd = lora_state_dict(model)
            sd.update({n: p.detach().cpu() for n, p in model.named_parameters()
                       if p.requires_grad and n not in sd})
            torch.save(sd, os.path.join(args.out, "adapter.pt"))
            log["adapter_bytes"] = os.path.getsize(os.path.join(args.out, "adapter.pt"))
            meta = {"format": "bantam-lora", "base_model": args.model, "r": args.lora,
                    "alpha": args.lora_alpha, "rslora": rslora, "dora": bool(args.dora),
                    "modules": lora_module_names(model), "quant": args.quant, "chat_template": template}
            with open(os.path.join(args.out, "adapter_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            save_peft = getattr(_lora, "save_peft_adapter", None)
            if args.save_peft and save_peft is not None:
                save_peft(model, args.out, args.model)
                log["peft_adapter"] = True
            elif args.save_peft > 0:
                print("[finetune] note: --save_peft needs bantamtrain.lora.save_peft_adapter; "
                      "only adapter.pt was written", flush=True)
        else:
            model.save_pretrained(os.path.join(args.out, "model"))
            tok.save_pretrained(os.path.join(args.out, "model"))
        if args.save_merged and not args.lora:
            print("[finetune] note: --save_merged with --lora 0: the full model is in <out>/model", flush=True)
        elif args.save_merged and quant:
            del fwd, opt, params
            model = None                         # free the quantized model before loading the float one
            _save_merged_from_float(args.model, dtype, sd, meta, tok, template,
                                    os.path.join(args.out, "merged"))
            log["merged"] = "float base + adapters"
        elif args.save_merged:
            merge_lora(model)
            unwrap_lora(model)
            model.save_pretrained(os.path.join(args.out, "merged"))
            tok.save_pretrained(os.path.join(args.out, "merged"))
            log["merged"] = "base + adapters"
        write_log()
        print(f"[finetune] done: eval {log['evals'][0]['eval_loss']:.4f} -> "
              f"{log['final_eval_loss']:.4f} in {t_train:.0f}s", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
