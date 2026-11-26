"""Compute-matched LLM training: Zoom Training x {AdamW, Muon} on CPU or GPU.

Every run gets the same FLOP budget and reads the same pre-tokenized chunks in
the same order: one seeded permutation of the windows, of which DDP ranks read
disjoint slices. Baseline: --zoom 1. Zoom Training: e.g. --zoom 2:0.33,1:0.67
(zoom level : fraction of the FLOP budget). --recipe best sets the best settings
measured so far (small scale, CPU) for every option the command line does not give.
Validation is always plain next-token loss at k=1 (the deployable model) in the
arithmetic of an fp32-weight run (a bf16-weight model is scored as an fp32
copy). Periodic evals use the first --eval_seqs val windows, the final eval the
first --final_eval_seqs; the final model is also scored on the periodic subset
(final_val_loss_eval_seqs) so curves and finals can be compared. The budget counts
model_flops (ZoomGPT.flops_per_step: causal-half attention, the FLOP convention of
every recorded run; plus opt_flops with --flops_mode total); model_flops_exact, the
matmuls the kernels execute (flops_per_step(exact=True)), is logged beside it.

    bantam pretrain --data DIR --budget 6.8e13 --recipe best --name my_run \
        [--device cuda --precision bf16 --compile]

Exit status: 0 finished (or stopped by --stop_step), 2 bad arguments,
3 diverged (non-finite loss: the JSON says "status": "diverged", no --save).
"""
import torch

torch.set_flush_denormal(True)  # before any parallel op (see bantamtrain.device)

import argparse  # noqa: E402
import copy  # noqa: E402
from contextlib import nullcontext  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from ..data import open_tokens as open_tokens_any  # noqa: E402
from ..data.tokpack import TokPack  # noqa: E402
from ..device import (PRECISIONS, DutyCycleLimiter, describe,  # noqa: E402
                      find_batch_size, maybe_compile, setup_runtime)
from ..models import ZoomGPT  # noqa: E402
from ..lowmem import QState, load_checkpoint, state_bytes  # noqa: E402
from ..optim import build_optimizer  # noqa: E402
from ..zoom import ZoomSchedule, lr_at  # noqa: E402
from .recipes import NAMES as RECIPE_NAMES, apply_recipe  # noqa: E402

KMAX = 4            # windows hold seq + max(KMAX, largest zoom level) tokens
EXIT_DIVERGED = 3


def open_token_source(data_dir, split, fmt="auto"):
    """(tokens, info): <split>.tokpack if present (fmt auto) else the raw <split>.bin (of
    meta.json's dtype, uint16 by default). fmt auto skips a pack left over from an earlier
    prepare run (bantamtrain.data.open_tokens); fmt tokpack raises StalePackError on one.
    info = {"format": "tokpack"|"bin", "file", "tokens"} for the run log."""
    if fmt == "tokpack":
        kind, tokens = "tokpack", TokPack(os.path.join(data_dir, f"{split}.tokpack"))
    else:
        tokens, kind = open_tokens_any(data_dir, split, prefer_pack=fmt == "auto")
    path = os.path.join(data_dir, f"{split}.{kind}")
    return tokens, {"format": kind, "file": os.path.abspath(path), "tokens": int(len(tokens))}


def open_tokens(data_dir, split, fmt="auto"):
    return open_token_source(data_dir, split, fmt)[0]


class ChunkStream:
    """Shuffled, non-overlapping windows of the token stream, consumed in order.

    One permutation per epoch (seed, then [seed, epoch]: reshuffled on every
    wrap). A step's batch*world windows are the next ones in that order and
    rank r reads its own slice, so DDP ranks never overlap and pos/epochs are
    global. A wrap starts a new epoch rather than return a short batch.
    """

    def __init__(self, tokens, seq, seed, rank=0, world=1, kmax=KMAX):
        self.data, self.seq, self.seed = tokens, seq, seed
        self.rank, self.world = rank, world
        self.pad = max(KMAX, kmax)            # targets reach k tokens past the window
        self.n = (len(self.data) - seq - self.pad) // seq
        if self.n < 1:
            raise ValueError(f"{len(self.data)} tokens hold no window of {seq}+{self.pad}")
        self.set_position(0, 0)

    def _perm(self, epoch):
        rng = np.random.default_rng(self.seed if epoch == 0 else [self.seed, epoch])
        return rng.permutation(self.n) * self.seq

    def set_position(self, pos, epochs):
        """Global position (as saved in a checkpoint)."""
        self.starts = self._perm(epochs)
        self.pos, self.epochs = pos, epochs

    def next(self, batch):
        need = batch * self.world
        if need > self.n:
            raise ValueError(f"{self.n} windows of {self.seq} tokens cannot fill a batch of {need}")
        if self.pos + need > self.n:
            self.set_position(0, self.epochs + 1)
        i = self.pos + self.rank * batch
        self.pos += need
        chunk = np.stack([self.data[s:s + self.seq + self.pad]
                          for s in self.starts[i:i + batch]]).astype(np.int64)
        return torch.from_numpy(chunk)

    @property
    def data_epochs(self):
        return self.epochs + self.pos / self.n


@torch.no_grad()
def evaluate(model, val, args, rt, n_seqs):
    """Mean k=1 loss over the first n_seqs windows of val, with the arithmetic of an fp32-weight
    run (fp32 on CPU, the run's autocast on GPU): a bf16/fp16-weight model is scored as an fp32
    copy (exactly its rounded weights)."""
    m = model
    if any(p.dtype in (torch.bfloat16, torch.float16) for p in model.parameters()):
        m = copy.deepcopy(model).float()
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, n_seqs, 64):
        rows = [val[s * args.seq:s * args.seq + args.seq + 1]
                for s in range(i, min(i + 64, n_seqs))]
        c = rt.to(torch.from_numpy(np.stack(rows).astype(np.int64)))
        with rt.autocast():
            tot += m(c[:, :-1], c[:, 1:], k=1).float().item() * c.shape[0]
        n += c.shape[0]
    model.train()
    return tot / n


MODEL_FIELDS = ("d", "layers", "heads", "zoom", "gated", "out_mode", "rope_pos", "qk_norm",
                "merge_mode", "mlp", "softcap", "zero_init_proj")
# --resume refuses changes to these (model, data order, LR / zoom / FLOP schedule) unless
# --allow_arg_change; batch and accum are compared as batch*accum; other changes warn
RESUME_FIELDS = MODEL_FIELDS + ("seq", "seed", "budget", "batch_scale", "batch_scale_k", "optim",
                                "state_bits", "param_dtype", "lr", "sched", "warmup", "decay_frac",
                                "lr_restart", "mix_fine", "wd", "ns", "evals", "flops_mode")
RESUME_IGNORE = ("resume", "stop_step", "ckpt_every", "ckpt", "threads", "allow_arg_change",
                 "auto_batch", "batch", "accum",
                 "recipe_applied")       # the values --recipe set are compared as options


def build_model(a, vocab, grad_ckpt=False):
    """ZoomGPT from an argparse namespace / run-log args (shared with the eval scripts).
    block_size = --seq: generate()'s default window is the trained context (not a weight);
    ce_chunk (memory only, same loss) is 0 for logs that predate it."""
    g = lambda k, default: getattr(a, k, default) if not isinstance(a, dict) else a.get(k, default)  # noqa: E731
    return ZoomGPT(vocab, g("d", 256), g("layers", 6), g("heads", 4),
                   zoom_ks=ZoomSchedule.parse(g("zoom", "1")).ks, gated=bool(g("gated", 1)),
                   out_mode=g("out_mode", "slot"), rope_pos=g("rope_pos", "token"),
                   qk_norm=bool(g("qk_norm", 0)), merge_mode=g("merge_mode", "") or None,
                   grad_ckpt=grad_ckpt, mlp=g("mlp", "gelu"), softcap=float(g("softcap", 0.0)),
                   zero_init_proj=bool(g("zero_init_proj", 0)),
                   ce_chunk=int(g("ce_chunk", 0) or 0), block_size=int(g("seq", 256)))


def stage_batch(batch, k, batch_scale=0.0, batch_scale_k=0):
    """Sequences per micro-batch at zoom level k."""
    if k > 1 and batch_scale > 0:
        return int(round(batch * batch_scale))
    return batch * (k if batch_scale_k else 1)


def next_eval_after(frac, evals):
    """Next eval point after an eval at frac (skips points that one step jumped over)."""
    return (math.floor(frac * evals + 1e-9) + 1) / evals


def eval_ckpt_path(save, frac, step, used=()):
    """--save_every_eval file: <root>_f<frac><ext> (.pt if --save has no extension),
    with the step added when that name was already used in this run."""
    root, ext = os.path.splitext(save)
    ext = ext or ".pt"
    ck = f"{root}_f{frac:.2f}{ext}"
    return ck if ck not in used else f"{root}_f{frac:.4f}_step{step}{ext}"


# ------------------------------------------------------------------ checkpoints
_QTAG = "__bantam_qstate__"
OPT_HYPER = ("weight_decay", "ns", "state_bits", "lr_scale")   # --wd, --ns, --state_bits, C2


def _encode_state(o):
    """QState objects -> tagged dicts of tensors, so checkpoints load with weights_only=True."""
    if isinstance(o, QState):
        return {_QTAG: dict(vars(o))}
    if isinstance(o, dict):
        return {k: _encode_state(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_encode_state(v) for v in o)
    return o


def _decode_state(o):
    if isinstance(o, dict):
        if set(o) == {_QTAG}:
            q = QState.__new__(QState)
            q.__dict__.update(o[_QTAG])
            return q
        return {k: _decode_state(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_decode_state(v) for v in o)
    return o


def load_optimizer_state(opt, saved):
    """opt.load_state_dict, robust for optimizers that keep QState or fp32 state for bf16
    params: QState buffers move to their param's device and saved fp32 state stays fp32
    (torch casts floating state to the param dtype). The hyperparameters set from the args
    (OPT_HYPER) come from this run's optimizer; the schedule sets lr every step anyway."""
    saved = _decode_state(saved)
    fresh = [{k: g[k] for k in OPT_HYPER if k in g} for g in opt.param_groups]
    opt.load_state_dict(saved)
    for g, f in zip(opt.param_groups, fresh):
        g.update(f)
    for o, sd in zip(getattr(opt, "optimizers", [opt]), saved if isinstance(saved, list) else [saved]):
        params = [p for g in o.param_groups for p in g["params"]]
        ids = [i for g in sd["param_groups"] for i in g["params"]]
        for i, p in zip(ids, params):
            st, old = o.state.get(p, {}), sd["state"].get(i, {})
            for key, v in st.items():
                if isinstance(v, QState):
                    for a in ("q", "s", "lo", "hi"):
                        if hasattr(v, a):
                            setattr(v, a, getattr(v, a).to(p.device))
                elif (torch.is_tensor(v) and torch.is_tensor(old.get(key))
                      and old[key].is_floating_point() and v.dtype != old[key].dtype):
                    st[key] = old[key].to(device=v.device, dtype=old[key].dtype)


def save_ckpt(path, model, opt, stream, progress, log, mix_rng, rt=None):
    """Atomic resumable checkpoint (plain tensors and containers: weights_only-loadable)."""
    ck = {"model": model.state_dict(), "opt": _encode_state(opt.state_dict()),
          "stream": {"pos": stream.pos, "epochs": stream.epochs},
          "progress": progress, "log": log, "rng": torch.get_rng_state(),
          "mix_rng": mix_rng.bit_generator.state, "args": log["args"]}
    if rt is not None and rt.device.type == "cuda":
        ck["cuda_rng"] = torch.cuda.get_rng_state(rt.device)   # ranks share the seed
    if rt is not None and rt.scaler is not None:
        ck["scaler"] = rt.scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_ckpt(path):
    """torch.load(weights_only=True). Checkpoints written before 0.3 pickled QState objects:
    those load with QState allow-listed (still no arbitrary code), with a warning."""
    ck, legacy = load_checkpoint(path)
    if legacy:
        print(f"warning: {path}: old checkpoint format (pickled 8-bit optimizer state), "
              "loaded with QState allow-listed", file=sys.stderr, flush=True)
    return ck


def resume_arg_changes(old, new, default):
    """(refused, warned): options that differ between a checkpoint's args and this run's.
    Options the checkpoint predates compare against default(name)."""
    refused, warned = [], []
    for k, v in new.items():
        if k not in RESUME_IGNORE and old.get(k, default(k)) != v:
            (refused if k in RESUME_FIELDS else warned).append(k)
    ob, oa = old.get("batch"), old.get("accum", 1)
    if ob is not None and ob * oa != new["batch"] * new["accum"]:
        refused.append("batch*accum")
    elif ob is not None and (ob, oa) != (new["batch"], new["accum"]):
        warned.append("batch/accum split")
    return refused, warned


# ------------------------------------------------------------------ logging
def _json_safe(o):
    """Strict JSON: non-finite floats become null."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def write_json(path, log):
    """Atomic: a kill mid-write never leaves a truncated run JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_json_safe(log), f, indent=1)
    os.replace(tmp, path)


def optimizer_step_flops(opt):
    """FLOPs of one optimizer step: bantamtrain.optim.optimizer_flops when available, else
    an estimate (Muon: Newton-Schulz matmuls per row block + ~10/elem; AdamW: ~12/param)."""
    from .. import optim
    if hasattr(optim, "optimizer_flops"):
        return float(optim.optimizer_flops(opt))
    total = 0.0
    for o in getattr(opt, "optimizers", [opt]):
        muon = type(o).__name__ == "Muon"
        for g in o.param_groups:
            for p in g["params"]:
                if not p.requires_grad:
                    continue
                if muon and p.dim() == 2:
                    s = getattr(p, "muon_split", 1)
                    a, b = sorted((p.shape[0] // s, p.shape[1]))
                    total += s * g.get("ns_steps", 5) * (4 * a * a * b + 2 * a ** 3) + 10 * p.numel()
                else:
                    total += 12 * p.numel()
    return total


def pick_micro_batch(target, fits, ok=lambda micro, accum: True):
    """Largest micro-batch <= fits that divides target (and passes ok) -> (micro, accum)."""
    for micro in range(min(target, fits or 0), 0, -1):
        if target % micro == 0 and ok(micro, target // micro):
            return micro, target // micro
    return None


def split_batch(args, ks, fits):
    """(micro, accum) for --auto_batch: micro <= fits, and the same number of sequences per
    optimizer step as --batch x --accum at every zoom level ks (and k=1)."""
    def per_step(b, a):
        return [stage_batch(b, k, args.batch_scale, args.batch_scale_k) * a for k in ks + [1]]
    want = per_step(args.batch, args.accum)
    return pick_micro_batch(args.batch * args.accum, fits, lambda m, a: per_step(m, a) == want)


def auto_micro_batch(args, vocab, sched, rt):
    """--auto_batch probe: the largest power-of-two micro-batch <= batch*accum whose
    fwd/bwd/step fits under the GPU memory headroom at every zoom stage, run on a throwaway
    model and optimizer with the RNG states restored afterwards (None: not even 1 fits)."""
    ks = sorted({k for k, _ in sched.stages} | ({1} if args.mix_fine > 0 else set()))
    probe_rt = copy.copy(rt)
    probe_rt.scaler = None                    # leave the real loss scale untouched

    def try_step(b):
        m = build_model(args, vocab, bool(args.grad_ckpt)).to(rt.device)
        if args.param_dtype == "bf16":
            m = m.to(torch.bfloat16)
        o = build_optimizer(m, args.optim, args.lr, args.wd, state_bits=args.state_bits, ns=args.ns)
        for k in ks:
            nb = stage_batch(b, k, args.batch_scale, args.batch_scale_k)
            x = torch.randint(0, vocab, (nb, args.seq + k), device=rt.device)
            with probe_rt.autocast():
                loss = m(x[:, :args.seq], x[:, k:args.seq + k], k=k)
            probe_rt.backward(loss, o, m.parameters())
            o.zero_grad(set_to_none=True)

    # the probe models' init and stochastic rounding draw from the global RNGs: without the
    # fork the real model would start from other weights than the same run without --auto_batch
    with torch.random.fork_rng(devices=[rt.device] if rt.device.type == "cuda" else []):
        return find_batch_size(try_step, start=1, limit=args.batch * args.accum, device=rt.device)


# ------------------------------------------------------------------ main
RECIPE_HELP = (
    "best: the best settings measured so far, for each of these options that the command line "
    "does not give: --optim muon --lr 1.6e-2 --qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2 "
    "(the LR only with --optim muon; --batch_scale, which overrides --batch_scale_k, not with "
    "a --batch_scale_k 1). Measured at small scale on CPU only (TinyStories, "
    "5.8M-param ZoomGPT, d=256, 6.8e13 FLOPs): final val loss 2.589 (2 seeds) vs 2.737 for "
    "the Muon baseline (3 seeds), which needs 1.41x the compute to match; not tested on "
    "larger models or GPUs. none (default): no change. The log records args.recipe, the "
    "resolved values and args.recipe_applied (the options it set)")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train/val .bin or .tokpack")
    ap.add_argument("--data_format", default="auto", choices=["auto", "bin", "tokpack"],
                    help="auto: <split>.tokpack if present, else the raw <split>.bin")
    ap.add_argument("--recipe", default="none", choices=RECIPE_NAMES, help=RECIPE_HELP)
    ap.add_argument("--zoom", default="1")
    ap.add_argument("--budget", type=float, required=True, help="training FLOPs")
    ap.add_argument("--flops_mode", default="model", choices=["model", "total"],
                    help="what --budget counts: model fwd+bwd FLOPs, or those plus optimizer "
                         "FLOPs (opt_flops / total_flops are logged either way)")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--state_bits", type=int, default=32, choices=[32, 8],
                    help="8: blockwise 8-bit optimizer state")
    ap.add_argument("--param_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="bf16: weights in bf16, updates written with stochastic rounding")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="activation checkpointing per block")
    ap.add_argument("--ce_chunk", type=int, default=0,
                    help="chunked cross-entropy: the loss is computed over this many positions "
                         "at a time (logits recomputed in backward), so the (batch, seq, vocab) "
                         "logits are never materialized; same loss up to fp32 summation order. "
                         "0 = off (default); e.g. 2048. Not used by --out_mode bag at k > 1")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--auto_batch", action="store_true",
                    help="GPU: split batch*accum into the largest micro-batch that fits in memory "
                         "(same effective batch, schedule and data order)")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch_scale_k", type=int, default=0,
                    help="use batch*k sequences at zoom k (constant positions per step)")
    ap.add_argument("--batch_scale", type=float, default=0.0,
                    help="multiply the batch by this while zoomed (k>1); overrides --batch_scale_k")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--decay_frac", type=float, default=0.45, help="wsd: fraction spent decaying")
    ap.add_argument("--lr_restart", type=int, default=0,
                    help="restart warmup+cosine at every zoom stage switch")
    ap.add_argument("--mix_fine", type=float, default=0.0,
                    help="in coarse stages, run a k=1 step with this probability")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--gated", type=int, default=1)
    ap.add_argument("--merge_mode", default="", choices=["", "gate", "mean", "linear"],
                    help="gate|mean|linear (overrides --gated)")
    ap.add_argument("--out_mode", default="slot", choices=["slot", "bag"])
    ap.add_argument("--rope_pos", default="token", choices=["token", "patch"])
    ap.add_argument("--qk_norm", type=int, default=0)
    ap.add_argument("--mlp", default="gelu", choices=["gelu", "relu2"])
    ap.add_argument("--softcap", type=float, default=0.0, help="logit softcap (0 = off; 15 typical)")
    ap.add_argument("--zero_init_proj", type=int, default=0)
    ap.add_argument("--ns", default="quintic", choices=["quintic", "polar"], help="Muon orthogonalizer")
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--eval_seqs", type=int, default=256,
                    help="periodic evals (the evals[] curve) score the FIRST this-many val windows "
                         "of --seq tokens; the final model is also scored on them "
                         "(final_val_loss_eval_seqs)")
    ap.add_argument("--final_eval_seqs", type=int, default=2048,
                    help="final_val_loss scores the first this-many val windows (a larger set "
                         "than the curve's: compare it with finals, not with evals[])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto", choices=PRECISIONS,
                    help="GPU autocast (CPU runs are always fp32)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--duty", type=float, default=1.0, help="cap average device load (0 < duty <= 1)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join("results", "llm"))
    ap.add_argument("--save", default="",
                    help="final weights (state_dict), plus <root>_pretrained/ via "
                         "model.save_pretrained (with the --data tokenizer.json and eos_id, "
                         "ready for `bantam generate`)")
    ap.add_argument("--save_every_eval", type=int, default=0,
                    help="also save <save root>_f<frac><ext> at every evaluation (compute curve; "
                         "needs --save)")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="write a resumable checkpoint (model+optimizer+data position) every N steps")
    ap.add_argument("--ckpt", default="", help="resumable checkpoint path (default <out>/<name>.ckpt)")
    ap.add_argument("--resume", default="", help="resume from this resumable checkpoint")
    ap.add_argument("--allow_arg_change", action="store_true",
                    help="--resume even if model / data-order / schedule options differ")
    ap.add_argument("--stop_step", type=int, default=0, help="stop after this many steps (tests)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    recipe_note = apply_recipe(args, build_parser, argv, "pretrain")
    args.ckpt = args.ckpt or os.path.join(args.out, f"{args.name}.ckpt")

    # ---- validate before any compute is spent or any file is written
    if not args.duty > 0:
        ap.error(f"--duty must be in (0, 1], got {args.duty}")
    for flag in ("budget", "evals", "batch", "accum", "seq", "eval_seqs", "final_eval_seqs"):
        if not getattr(args, flag) > 0:
            ap.error(f"--{flag} must be positive")
    if args.save_every_eval and not args.save:
        ap.error("--save_every_eval needs --save (the eval checkpoints are named after it)")
    try:
        sched = ZoomSchedule.parse(args.zoom)
    except (ValueError, AssertionError) as e:
        ap.error(f"--zoom {args.zoom!r}: {e}")
    ks = [k for k, _ in sched.stages]
    bad = [k for k in ks if k < 1 or args.seq % k]
    if bad:
        ap.error(f"--zoom {args.zoom}: zoom level(s) {bad} must be >= 1 and divide --seq {args.seq}")
    kmax = max(ks)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    try:
        train, train_src = open_token_source(args.data, "train", args.data_format)
        val, val_src = open_token_source(args.data, "val", args.data_format)
        meta = json.load(open(os.path.join(args.data, "meta.json")))
    except OSError as e:
        ap.error(f"--data {args.data}: {e}")
    try:                                     # --d/--heads/--merge_mode/...: a skeleton, no memory
        with torch.device("meta"):
            build_model(args, meta["vocab_size"])
    except ValueError as e:
        ap.error(f"model options: {e}")
    max_seqs = (len(val) - 1) // args.seq
    if max(args.eval_seqs, args.final_eval_seqs) > max_seqs:
        ap.error(f"the val split holds {max_seqs} windows of --seq {args.seq}: lower --eval_seqs "
                 f"({args.eval_seqs}) / --final_eval_seqs ({args.final_eval_seqs}) to <= {max_seqs}")
    n_windows = (len(train) - args.seq - max(KMAX, kmax)) // args.seq
    max_batch = max(stage_batch(args.batch, k, args.batch_scale, args.batch_scale_k)
                    for k in ks + [1]) * world
    if max_batch > n_windows:
        ap.error(f"the train split holds {n_windows} windows of --seq {args.seq}, fewer than one "
                 f"step's batch ({max_batch} sequences over {world} rank(s))")
    try:
        resume = load_ckpt(args.resume) if args.resume else None
    except OSError as e:
        ap.error(f"--resume {args.resume}: {e}")
    if resume is not None:
        refused, warned = resume_arg_changes(resume["args"], vars(args), ap.get_default)
        old_src = resume["log"].get("data_source", {}).get("train", {})
        if old_src.get("tokens", train_src["tokens"]) != train_src["tokens"]:
            refused.append(f"train data ({old_src['tokens']} -> {train_src['tokens']} tokens)")
        old_world = resume["log"].get("world_size", 1)    # global batch = batch*accum*world
        if old_world != world:
            refused.append(f"WORLD_SIZE ({old_world} -> {world})")
        if refused and not args.allow_arg_change:
            ap.error(f"--resume: options differ from the checkpoint: {refused} (they change the "
                     "model, data order or schedule; --allow_arg_change overrides)")
        if (refused or warned) and is_main:
            print(f"warning: --resume: options differ from the checkpoint, using this run's: "
                  f"{refused + warned}", file=sys.stderr, flush=True)

    os.makedirs(args.out, exist_ok=True)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    # multi-GPU / multi-process via torchrun: one data order shared by all ranks, each reading
    # a disjoint slice of every step's windows; DDP averages gradients, so the global batch is
    # batch x accum x WORLD_SIZE. Every rank seeds the model init with the same --seed.
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rt = setup_runtime(args.device, args.precision, args.compile, args.threads, args.seed)
    if rt.device.type == "cpu":
        rt.amp_dtype, rt.scaler = None, None  # CPU: fp32, as in every recorded run
    auto = None
    if args.auto_batch and rt.device.type == "cuda":
        fits = auto_micro_batch(args, meta["vocab_size"], sched, rt)
        if world > 1:                            # every rank must use the same split
            t = torch.tensor([fits or 0], device=rt.device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            fits = int(t.item())
        pick = split_batch(args, ks, fits)
        if pick is None:
            raise SystemExit("--auto_batch: not even a micro-batch of 1 fits in GPU memory")
        auto = {"requested": [args.batch, args.accum], "largest_fit": fits,
                "micro": pick[0], "accum": pick[1]}
        args.batch, args.accum = pick
        if is_main:
            print(f"  --auto_batch: micro-batch {args.batch} x accum {args.accum}", flush=True)
    elif args.auto_batch and is_main:
        print("  --auto_batch: not a CUDA device, keeping --batch/--accum", flush=True)
    model = build_model(args, meta["vocab_size"], bool(args.grad_ckpt)).to(rt.device)
    if args.param_dtype == "bf16":
        model = model.to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    params = list(model.parameters())
    opt = build_optimizer(model, args.optim, args.lr, args.wd, state_bits=args.state_bits,
                          ns=args.ns)
    opt_step_flops = optimizer_step_flops(opt) * world   # every rank runs the optimizer
    ddp = None
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp = DDP(model, device_ids=[rt.device.index] if rt.device.type == "cuda" else None,
                  find_unused_parameters=bool(sched.ks))   # merge gates idle at other k
        fwd = maybe_compile(ddp, rt)
    else:
        fwd = maybe_compile(model, rt)
    stream = ChunkStream(train, args.seq, args.seed, rank=rank, world=world, kmax=kmax)
    limiter = DutyCycleLimiter(args.duty, rt.device)
    data_source = {"train": train_src, "val": val_src}
    stats = os.path.join(args.data, "pack_stats.json")
    if train_src["format"] == "tokpack" and os.path.exists(stats):
        data_source["dedup_params"] = json.load(open(stats)).get("dedup", {}).get("params")
    if train_src["format"] == "bin" and args.data_format == "auto" and is_main:
        print(f"warning: training on the raw {train_src['file']} (no train.tokpack): runs on the "
              "deduplicated pack (bantam pack-text) read a different corpus in a different order",
              file=sys.stderr, flush=True)

    run_info = {"world_size": world, "data_source": data_source, "flops_mode": args.flops_mode,
                "opt_flops_per_step": opt_step_flops,
                "eval_subset": {"periodic_first_seqs": args.eval_seqs,
                                "final_first_seqs": args.final_eval_seqs, "seq": args.seq}}
    if auto:
        run_info["auto_batch"] = auto
    log = {"args": vars(args), "schedule": repr(sched), "params": n_params, "evals": [],
           "train": [], "runtime": describe(rt), **run_info}
    if is_main:
        print(f"[{args.name}] params={n_params/1e6:.2f}M zoom={sched} optim={args.optim} "
              f"budget={args.budget:.3e} world={world} {describe(rt)}", flush=True)
        if recipe_note:
            print(f"  {recipe_note}", flush=True)

    model_flops, opt_flops, tokens, step, train_time = 0.0, 0.0, 0, 0, 0.0
    exact_flops = 0.0     # flops_per_step(exact=True): the matmuls the kernels run (logged only)
    next_eval = 0.0
    mix_rng = np.random.default_rng(args.seed + 1)
    if resume is not None:
        model.load_state_dict(resume["model"])
        load_optimizer_state(opt, resume["opt"])
        stream.set_position(resume["stream"]["pos"], resume["stream"]["epochs"])
        pr = resume["progress"]
        tokens, step, train_time, next_eval = (pr["tokens"], pr["step"], pr["train_time"],
                                               pr["next_eval"])
        model_flops = pr.get("model_flops", pr["flops"])    # before 0.3: model FLOPs only
        opt_flops = pr.get("opt_flops", step * opt_step_flops)
        exact_flops = pr.get("model_flops_exact")          # None: checkpoint predates it
        log = resume["log"]
        log.setdefault("resume_history", []).append({"step": step, "args": log["args"]})
        log.update(run_info, args=vars(args), resumed_from=args.resume)
        torch.set_rng_state(resume["rng"])
        if resume.get("cuda_rng") is not None and rt.device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"], rt.device)
        if resume.get("scaler") and rt.scaler is not None:
            rt.scaler.load_state_dict(resume["scaler"])
        mix_rng.bit_generator.state = resume["mix_rng"]

    def budget_flops():
        return model_flops + (opt_flops if args.flops_mode == "total" else 0.0)

    def progress():
        return {"flops": budget_flops(), "model_flops": model_flops, "opt_flops": opt_flops,
                "model_flops_exact": exact_flops, "tokens": tokens, "step": step,
                "train_time": train_time, "next_eval": next_eval}

    if resume is not None and is_main:
        print(f"  resumed at step {step} (frac={budget_flops()/args.budget:.3f})", flush=True)
    json_path = os.path.join(args.out, f"{args.name}.json")
    used_ckpts = {e["checkpoint"] for e in log["evals"] if e.get("checkpoint")}
    diverged = stopped = False
    while True:
        flops = budget_flops()
        frac = flops / args.budget
        if frac >= next_eval - 1e-12:
            next_eval = next_eval_after(frac, args.evals)    # identical on every rank
            if is_main:
                vl = evaluate(model, val, args, rt, args.eval_seqs)
                log["evals"].append({"frac": frac, "flops": flops, "tokens": tokens, "step": step,
                                     "train_time": train_time, "val_loss": vl,
                                     "opt_flops": opt_flops, "total_flops": model_flops + opt_flops,
                                     "model_flops_exact": exact_flops})
                print(f"  frac={frac:.3f} step={step} tok={tokens/1e6:.2f}M "
                      f"t={train_time:.0f}s val={vl:.4f}", flush=True)
                if args.save and args.save_every_eval and step > 0:
                    ck = eval_ckpt_path(args.save, frac, step, used_ckpts)
                    used_ckpts.add(ck)
                    torch.save(model.state_dict(), ck)
                    log["evals"][-1]["checkpoint"] = ck
                log["in_progress"] = True
                write_json(json_path, log)
        if frac >= 1.0:
            break
        k, s0, s1 = sched.stage_at(frac)
        if k > 1 and args.mix_fine > 0 and mix_rng.random() < args.mix_fine:
            k = 1
        batch = stage_batch(args.batch, k, args.batch_scale, args.batch_scale_k)
        lr = lr_at(frac, args.lr, warmup=args.warmup, stage=(s0, s1) if args.lr_restart else None,
                   sched=args.sched, decay_frac=args.decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        lsum = None
        for a in range(args.accum):
            chunk = rt.to(stream.next(batch))
            x, y = chunk[:, :args.seq], chunk[:, k:args.seq + k]
            sync = ddp.no_sync() if (ddp is not None and a < args.accum - 1) else nullcontext()
            with sync:
                with rt.autocast():
                    loss = fwd(x, y, k=k) / args.accum
                rt.backward(loss, opt, params, step=False)
            lsum = loss.detach() if lsum is None else lsum + loss.detach()
        rt.step(opt, params)
        if ddp is not None:     # mean loss over ranks; a non-finite loss on any rank stops all
            t = torch.stack([lsum.float(), (~torch.isfinite(lsum)).float()])
            dist.all_reduce(t)
            lv, nonfinite = t[0].item() / world, t[1].item() > 0
        else:
            lv = lsum.item()
            nonfinite = not math.isfinite(lv)
        dt = time.perf_counter() - t0
        train_time += dt
        limiter(dt)
        model_flops += model.flops_per_step(batch, args.seq, k) * args.accum * world
        if exact_flops is not None:
            exact_flops += model.flops_per_step(batch, args.seq, k, exact=True) * args.accum * world
        opt_flops += opt_step_flops
        tokens += batch * args.seq * args.accum * world
        step += 1
        if step % 25 == 0 or nonfinite:
            log["train"].append({"step": step, "frac": frac, "k": k, "loss": lv,
                                 "train_time": train_time})
        if nonfinite:
            if is_main:
                print(f"non-finite loss at step {step}, abort", file=sys.stderr, flush=True)
            diverged = True
            break
        if args.ckpt_every and step % args.ckpt_every == 0 and is_main:
            save_ckpt(args.ckpt, model, opt, stream, progress(), log, mix_rng, rt)
        if args.stop_step and step >= args.stop_step:
            if is_main:
                print(f"  stopped at step {step} (--stop_step)", flush=True)
                if args.ckpt_every:
                    save_ckpt(args.ckpt, model, opt, stream, progress(), log, mix_rng, rt)
            stopped = True
            break

    if world > 1:
        dist.barrier()
        if not is_main or stopped:
            dist.destroy_process_group()
            if diverged:
                sys.exit(EXIT_DIVERGED)
            return 0
    if stopped:
        return 0
    log.pop("in_progress", None)
    log["train_time"] = train_time
    log["tokens"] = tokens
    log["steps"] = step
    log["data_epochs"] = stream.data_epochs
    log.update(model_flops=model_flops, opt_flops=opt_flops, total_flops=model_flops + opt_flops,
               model_flops_exact=exact_flops)
    if diverged:
        log.update(status="diverged", diverged=True, diverged_at_step=step)
        write_json(json_path, log)
        print(f"[{args.name}] diverged at step {step}: no final eval, no --save", file=sys.stderr,
              flush=True)
        if world > 1:
            dist.destroy_process_group()
        sys.exit(EXIT_DIVERGED)
    log["status"] = "finished"
    if args.save:   # before the final eval, so a crash there cannot lose the model
        torch.save(model.state_dict(), args.save)
        log["saved"] = {"state_dict": args.save}
        if callable(getattr(model, "save_pretrained", None)):
            d = os.path.splitext(args.save)[0] + "_pretrained"
            tok = os.path.join(args.data, "tokenizer.json")   # + eos_id from meta.json
            try:
                model.save_pretrained(d, **({"tokenizer_path": args.data} if os.path.exists(tok)
                                            else {}))
                log["saved"]["pretrained"] = d
            except Exception as e:  # noqa: BLE001 - the state_dict is saved already
                print(f"warning: save_pretrained({d}) failed: {e}", file=sys.stderr, flush=True)
    log["final_val_loss"] = evaluate(model, val, args, rt, args.final_eval_seqs)
    last = log["evals"][-1] if log["evals"] else None
    log["final_val_loss_eval_seqs"] = (last["val_loss"] if last and last["step"] == step
                                       else evaluate(model, val, args, rt, args.eval_seqs))
    log["optimizer_state_bytes"] = state_bytes(opt)
    log["param_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    write_json(json_path, log)
    print(f"[{args.name}] final val={log['final_val_loss']:.4f} tokens={tokens/1e6:.1f}M "
          f"steps={step} train_time={train_time:.0f}s", flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
