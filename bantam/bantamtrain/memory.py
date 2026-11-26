"""Memory and time scoreboard for single-GPU LLM training.

    python -m bantamtrain.memory                 # 5B model on one RTX PRO 6000 Blackwell
    python -m bantamtrain.memory --tokens 2e10   # a 20B-token run
    python -m bantamtrain.memory --markdown      # table for docs
    python -m bantamtrain.memory --finetune --params 7e9 --gpu_gb 24   # full / LoRA / QLoRA

Each row adds one technique to the row above and reports static memory
(weights + grads + optimizer state), total memory, whether it fits (with
activation recompute when that is what makes it fit, costed at 8ND), the largest
micro-batch, training FLOPs and wall-clock days to reach the same loss as the
first row that fits, trained on --tokens tokens.

Units: GPU memory is binary: a "96 GB" card has 96 GiB (nvidia-smi shows MiB), so
every memory column is GiB. Usable = --gpu_gb GiB x --headroom (0.85: the peak
device.find_batch_size keeps to) minus --overhead_gb (CUDA context, workspaces).

Assumptions (defaults; every one is a flag):
  GPU       RTX PRO 6000 Blackwell Workstation: 96 GiB, 503.8 TFLOPS dense BF16
            (NVIDIA RTX Blackwell PRO architecture whitepaper, Appendix A).
  MFU       0.40 of dense BF16 peak (conservative). nanochat on one RTX PRO 6000
            measured ~297 TFLOPS = 59 % at 600 W and ~40 % at 300 W
            (karpathy/nanochat #288, d20 = 3.49 GFLOP/token); 2 cards ~50 % (#28).
  FLOPs     6*N*D with N including both embedding matrices and no attention-score
            term. nanochat's MFU counts 6*(N - V*d) + 12*L*d*T per token; the
            difference (a few %) is printed so days can be rescaled.
  FP8       x1.20 throughput for FP8 linears (Transformer Engine current
            scaling is supported on SM 12.0); reports range 20-30 %. Not
            measured here.
  Muon      reaches AdamW's loss with 0.52x the FLOPs (Liu et al. 2025,
            "Muon is Scalable for LLM Training", compute-optimal 1-16B runs).
            Newton-Schulz costs ns_steps * (4m^2n + 2m^3) FLOPs per hidden matrix
            per optimizer step, charged per --global_batch_tokens tokens (~3 % at
            0.5M tokens per step for 5B). Measured here at 5.8M params: Muon at ~40 %
            of AdamW's compute (~2.4x; Newton-Schulz FLOPs not in that budget) beat
            AdamW at full compute; not used in the table.
  bf16 + SR bf16 weights/grads with stochastic rounding match fp32-master mixed
            precision up to 6.7B (Ozkara et al. 2025); 8-bit Adam states match
            32-bit up to 1.5B (Dettmers et al. 2021).
  Power     450 W cap keeps ~95 % throughput on this card in a ResNet-50 training
            sweep (naveen.ing); for LLM training the only datapoint is nanochat at
            300 W: -32 % throughput, +36 % tokens per watt (#288).
Activation memory, measured with measure_saved_bytes() on this repo's ZoomGPT in
eager mode, in bytes per layer per token per d_model: 48.7 under AMP (fp32 params +
bf16 autocast: the residual stream stays fp32, so recompute keeps 4 B per element
of every block input) and 40.7 with bf16 params (2 B block inputs). Korthikanti et
al.'s 34 assumes fused kernels (--act_factor 0.7). Unchunked cross-entropy keeps
~14 B per logit (bf16 logits, fp32 log-softmax, fp32 grad); chunked CE keeps that
for --ce_chunk tokens at a time. Full recompute stores the block inputs, recomputes
one layer at a time for every sequence of the micro-batch, and costs one extra
forward (8ND instead of 6ND; Megatron measured +30-40 % wall-clock).
The fine-tune table uses constants measured on HF LlamaForCausalLM (HF_ACT),
including LoRA dropout, and charges QLoRA for what bantamtrain.qweights actually
keeps for backward (qlora_probe()).
"""
import argparse
import functools
import warnings

GB = 1e9          # disk and file sizes (storage table)
GIB = 2 ** 30     # GPU memory
# saved-for-backward bytes per layer per token per d_model (measure_saved_bytes)
ZOOM_ACT = {"amp": 48.7, "bf16": 40.7}            # ZoomGPT eager, 4x GELU MLP
HF_ACT = {"amp": 56.0, "bf16": 46.0,              # HF Llama (MLP 2.69 d), full fine-tune
          "lora": 43.0, "lora_dropout": 25.0,     # bf16 base, LoRA on all 7 projections
          "lora_fp32": 49.0, "lora_fp32_dropout": 29.0}   # same with an fp32 residual stream
LOGIT_BYTES = 14  # per logit element: bf16 logits + fp32 log-softmax + fp32 grad


def default_width(params):
    """A conventional d_model for a parameter count (GPT-2/Llama-style aspect ratios)."""
    for limit, d in ((2e7, 384), (6e7, 512), (2e8, 768), (5e8, 1024), (1.5e9, 2048),
                     (4e9, 3072), (1e10, 4096), (2e10, 5120)):
        if params <= limit:
            return d
    return 8192


def model_dims(params, d, vocab, tied=False):
    """Layers for a target parameter count: 12*L*d^2 + 2*V*d (V*d when the head is tied).
    Warns when no model of this width and vocabulary is within 10 % of the target."""
    d = d or default_width(params)
    emb = (1 if tied else 2) * vocab * d
    L = max(1, round((params - emb) / (12 * d * d)))
    N = 12 * L * d * d + emb
    if abs(N - params) > 0.1 * params:
        warnings.warn(f"--params {params:.3g}: the closest model with d={d}, vocab={vocab} has "
                      f"{N:.3g} params ({L} layers); pass a smaller --d or --vocab", stacklevel=2)
    return L, N


def capacity_bytes(gpu_gb, headroom=1.0):
    """Usable bytes of a card sold as `gpu_gb` GB (binary, like nvidia-smi's MiB)."""
    return gpu_gb * GIB * headroom


def muon_ns_flops(d, L, ns_steps=5, mlp_mult=4):
    """Newton-Schulz FLOPs of one Muon step on a GPT block stack (fused qkv split in 3),
    counted as bantamtrain.optim.ns_flops / optimizer_flops count them."""
    from .optim import ns_flops
    per_layer = (ns_flops(d, d, ns_steps, batch=4)               # q, k, v blocks + proj
                 + ns_flops(mlp_mult * d, d, ns_steps) + ns_flops(d, mlp_mult * d, ns_steps))
    return L * per_layer


def measure_saved_bytes(fn, exclude=()):
    """Bytes of the distinct storages autograd saves for backward while fn() runs,
    skipping the storages of the tensors in `exclude` (parameters, buffers). Returns
    (bytes, [(shape, dtype, storage bytes)]) so callers can filter, e.g. weight casts."""
    import torch
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        if st.data_ptr() not in skip:
            seen[st.data_ptr()] = (tuple(t.shape), t.dtype, st.nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return sum(v[2] for v in seen.values()), list(seen.values())


@functools.lru_cache(maxsize=None)
def qlora_probe():
    """What bantamtrain.qweights keeps, measured on a tiny layer under bf16 autocast (as
    on a GPU): bytes per quantized param a QLinear saves for backward when its input
    needs a gradient (every layer after the first LoRA one), and the bytes per element
    QEmbedding returns for a bf16 table (4 = an fp32 residual stream)."""
    try:
        import torch
        from .qweights import QEmbedding, QLinear
        q = QLinear(torch.nn.Linear(64, 96, bias=False), "nf4", 64)
        x = torch.randn(5, 64, requires_grad=True)

        def run():
            with torch.autocast("cpu", dtype=torch.bfloat16):
                q(x).float().sum()

        _, saved = measure_saved_bytes(run, [x] + list(q.buffers()))
        w = sum(b for shape, _, b in saved if sorted(shape) == [64, 96])
        emb = QEmbedding(torch.nn.Embedding(32, 16).to(torch.bfloat16), 8, 16)
        with torch.no_grad():
            out = emb(torch.tensor([0, 1])).element_size()
        return {"dequant_bytes_per_param": w / (64 * 96), "embed_out_bytes": out, "measured": True}
    except Exception:  # noqa: BLE001 - fall back to what the code kept when this was written
        return {"dequant_bytes_per_param": 2.0, "embed_out_bytes": 4, "measured": False}


def scoreboard(params=5e9, d=0, vocab=32000, seq=2048, gpu_gb=96.0, peak_tflops=503.8,
               mfu=0.40, tokens=1e11, fp8_speedup=1.20, muon_flops=0.52, power_tput=0.95,
               overhead_gb=3.0, hidden_frac=None, act_factor=1.0, headroom=0.85, ce_chunk=2048,
               tied=False, global_batch_tokens=524288, ns_steps=5):
    d = d or default_width(params)
    L, N = model_dims(params, d, vocab, tied)
    if hidden_frac is None:
        hidden_frac = 12 * L * d * d / N
    cap = capacity_bytes(gpu_gb, headroom)
    logits_seq = seq * vocab * LOGIT_BYTES             # unchunked loss, per sequence
    logits_chunk = ce_chunk * vocab * LOGIT_BYTES      # chunked cross-entropy, once
    ns_step = muon_ns_flops(d, L, ns_steps)

    def row(name, w, g, state, act, chunked, tok_mult=1.0, speed=1.0, muon=False, note=""):
        coef = ZOOM_ACT[act] * act_factor               # bytes per layer*token*d
        resid = 4 if act == "amp" else 2                # checkpointed block inputs
        static = N * (w + g + state)
        fixed = static + overhead_gb * GIB + (logits_chunk if chunked else 0)
        for ckpt in (False, True):                      # recompute only when needed to fit
            act_seq = (resid * L + coef) * seq * d if ckpt else coef * L * seq * d
            per_seq = act_seq + (0 if chunked else logits_seq)
            max_micro = int((cap - fixed) // per_seq) if cap > fixed else 0
            if max_micro >= 1:
                break
        fits = max_micro >= 1
        micro = max(1, min(max_micro, 8))
        flops = (8 if ckpt else 6) * N * tokens * tok_mult
        if muon:
            flops += ns_step * tokens * tok_mult / global_batch_tokens
        days = flops / (peak_tflops * 1e12 * mfu * speed) / 86400 if fits else None
        if ckpt and fits:
            name += " (+ act. ckpt to fit)"
        note = note + ("; " if note else "") + (
            "does not fit, even with activation recompute" if not fits else
            "needs activation recompute" if ckpt else "fits without recompute")
        need = fixed + micro * per_seq
        return {"config": name, "bytes_per_param": w + g + state, "ckpt": ckpt,
                "static_gb": static / GIB, "act_gb": micro * per_seq / GIB,
                "total_gb": need / GIB, "fits": fits, "max_micro": max_micro,
                "margin_gb": (cap - need) / GIB, "fixed_bytes": fixed, "per_seq_bytes": per_seq,
                "flops": flops, "days": days, "note": note}

    muon_state = hidden_frac * 1 + (1 - hidden_frac) * 2      # 8-bit momentum / 8-bit Adam
    rows = [
        row("1. Standard: AMP bf16, fp32 master + fp32 AdamW", 4 + 2, 4, 8, "amp", False,
            note="fp32 weights + bf16 autocast copies"),
        row("2. + bf16 weights (stochastic rounding) + 8-bit AdamW + chunked CE",
            2, 2, 2, "bf16", True, note="no fp32 master"),
        row("3. + Muon (8-bit momentum)", 2, 2, muon_state, "bf16", True, tok_mult=muon_flops,
            muon=True, note=f"same loss with {muon_flops:.2f}x the FLOPs + Newton-Schulz"),
        row("4. + FP8 linears", 2, 2, muon_state, "bf16", True, tok_mult=muon_flops,
            speed=fp8_speedup, muon=True, note=f"x{fp8_speedup:.2f} throughput (literature)"),
        row("5. + 450 W power cap (75 % power)", 2, 2, muon_state, "bf16", True,
            tok_mult=muon_flops, speed=fp8_speedup * power_tput, muon=True,
            note=f"x{power_tput:.2f} throughput, 25 % less power"),
    ]
    fpt = 6 * N
    return {"layers": L, "params": N, "d": d, "rows": rows, "gpu_gb": gpu_gb,
            "headroom": headroom, "capacity_gb": cap / GIB, "overhead_gb": overhead_gb,
            "flops_per_token": fpt,
            "flops_per_token_nanochat": 6 * (N - vocab * d) + 12 * L * d * seq,
            "muon_ns_flops_per_step": ns_step, "global_batch_tokens": global_batch_tokens}


def finetune_table(N, d, L, seq, micro, gpu_gb, lora_r=16, overhead_gb=2.0, act_factor=1.0,
                   lora_dropout=0.05, vocab=32000, ce_chunk=2048, headroom=0.85, tied=False,
                   probe=None):
    """Memory for fine-tuning an N-param HF Llama-style base model on one GPU: full FT
    vs LoRA vs QLoRA, without and with activation checkpointing.

    LoRA params ~ 2 * r * (sum over the 7 projections of (d_in + d_out)) per layer
    ~ r * L * (4*2d + 3*(d + 2.7d)) for a Llama block; adapters keep fp32 state.
    QLoRA: linears int4 (0.53 B/param), embeddings and head int8 (finetune.py's
    --embed_quant 8: 1.03 B/param), plus what qlora_probe() measures qweights keeps.
    """
    probe = probe or qlora_probe()
    cap = capacity_bytes(gpu_gb, headroom)
    tok = seq * micro
    lora_params = lora_r * L * (4 * 2 * d + 3 * (d + 2.7 * d))
    emb = (1 if tied else 2) * vocab * d
    lin = N - emb                                       # quantized linears under QLoRA
    loss = (ce_chunk if ce_chunk else tok) * vocab * LOGIT_BYTES
    drop = lora_dropout > 0
    q_resid = probe["embed_out_bytes"]                  # 4 when QEmbedding returns fp32
    lora_coef = HF_ACT["lora"] + drop * HF_ACT["lora_dropout"]
    qlora_coef = lora_coef if q_resid <= 2 else HF_ACT["lora_fp32"] + drop * HF_ACT["lora_fp32_dropout"]
    lora_static = 16 * lora_params                      # fp32 adapters + grads + AdamW
    specs = [  # name, static bytes, act coef, checkpointed input bytes, saved weights per layer
        ("full fine-tune: fp32 master + AdamW (standard)", (4 + 2 + 4 + 8) * N, HF_ACT["amp"], 4, 0, N),
        ("full fine-tune: bf16 weights(SR) + 8-bit Muon", (2 + 2 + 1.1) * N, HF_ACT["bf16"], 2, 0, N),
        (f"LoRA r={lora_r}: bf16 base frozen + fp32 adapters/AdamW", 2 * N + lora_static,
         lora_coef, 2, 0, lora_params),
        (f"QLoRA r={lora_r}: int4 base (bantamtrain.qweights) + adapters",
         0.53 * lin + 1.03 * emb + lora_static, qlora_coef, q_resid,
         probe["dequant_bytes_per_param"] * lin / L, lora_params),
    ]
    rows = []
    for name, static, coef, resid, saved_w, trainable in specs:
        coef *= act_factor
        for ck in ("no", "yes"):
            if ck == "no":
                a = coef * d * L * tok + saved_w * L
            else:                                       # stored inputs + one layer recomputed
                a = resid * d * L * tok + coef * d * tok + saved_w
            total = static + a + loss + overhead_gb * GIB
            rows.append({"config": name, "ckpt": ck, "trainable": trainable,
                         "static_gb": static / GIB, "act_gb": (a + loss) / GIB,
                         "total_gb": total / GIB, "fits": total <= cap, "capacity_gb": cap / GIB})
    return rows


def fmt_finetune(rows, N, gpu_gb):
    cap = rows[0].get("capacity_gb", gpu_gb) if rows else gpu_gb
    hdr = f"fine-tuning a {N/1e9:.1f}B base model on one {gpu_gb:g} GiB GPU ({cap:.1f} GiB usable)"
    out = [hdr, f"{'configuration':60s} {'act.ckpt':>8s} {'trainable':>10s} {'static':>7s} {'act':>6s} {'total':>6s} fit"]
    out.append("-" * len(out[1]))
    for r in rows:
        out.append(f"{r['config']:60s} {r['ckpt']:>8s} {r['trainable']/1e6:9.1f}M {r['static_gb']:6.1f}G "
                   f"{r['act_gb']:5.1f}G {r['total_gb']:5.1f}G {'yes' if r['fits'] else 'NO'}")
    return "\n".join(out)


def storage_table(N, tokens, bits_per_token=5.97):
    """Disk/RAM sizes (decimal GB) for weights, resumable checkpoints and the token corpus."""
    src = ("measured on 117M TinyStories tokens, 4k vocab" if bits_per_token == 5.97
           else "--bits_per_token")
    return [
        ("weights fp32 (RAM or disk)", 4 * N),
        ("weights bf16", 2 * N),
        ("weights bf16, lossless (WeightPack 16: byte-shuffle + zstd, ~71 %)", 1.42 * N),
        ("weights int8 (qweights / WeightPack 8)", 1.03 * N),
        ("weights int4 (qweights / WeightPack 4)", 0.53 * N),
        ("resumable ckpt: fp32 weights + fp32 AdamW", 12 * N),
        ("resumable ckpt: bf16 weights + 8-bit Muon", 3.1 * N),
        ("corpus: raw UTF-8 text (~4 B/token)", 4 * tokens),
        ("corpus: uint16 token ids", 2 * tokens),
        (f"corpus: TokPack v2 ({bits_per_token:.2f} bits/token, {src})",
         bits_per_token / 8 * tokens),
    ]


def fmt(sb, tokens, markdown=False):
    ref = next((r for r in sb["rows"] if r["fits"]), None)
    base = ref["days"] if ref else None

    def days(r):
        return "n/a" if r["days"] is None else f"{r['days']:.0f}"

    def speedup(r):
        return f"{base / r['days']:.2f}x" if base and r["days"] else "n/a"

    lines = []
    if markdown:
        lines.append("| configuration | bytes/param | static GiB | total GiB | fits | max micro-batch | days | speedup |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sb["rows"]:
            lines.append(f"| {r['config']} | {r['bytes_per_param']:.1f} | {r['static_gb']:.0f} | "
                         f"{r['total_gb']:.0f} | {'yes' if r['fits'] else 'no'} | {r['max_micro']} | "
                         f"{days(r)} | {speedup(r)} |")
        return "\n".join(lines)
    ref_name = "nothing fits" if ref is None else (
        "standard recipe" if ref is sb["rows"][0] else f"row {ref['config'].split('.')[0]}")
    lines.append(f"model: {sb['params']/1e9:.2f}B params ({sb['layers']} layers), "
                 f"reference = {ref_name} on {tokens:.1e} tokens")
    if "capacity_gb" in sb:
        extra = sb["flops_per_token_nanochat"] / sb["flops_per_token"] - 1
        lines.append(f"GPU {sb['gpu_gb']:g} GiB x {sb['headroom']:.2f} headroom = "
                     f"{sb['capacity_gb']:.1f} GiB usable, {sb['overhead_gb']:g} GiB overhead; "
                     f"FLOPs = 6*N*D (nanochat's MFU convention counts {extra:+.1%})")
    w = max([72] + [len(r["config"]) for r in sb["rows"]])
    hdr = f"{'configuration':{w}s} {'B/p':>5s} {'static':>7s} {'total':>6s} fit {'maxB':>5s} {'days':>6s} {'speedup':>7s}"
    lines += [hdr, "-" * len(hdr)]
    for r in sb["rows"]:
        lines.append(f"{r['config']:{w}s} {r['bytes_per_param']:5.1f} {r['static_gb']:6.0f}G "
                     f"{r['total_gb']:5.0f}G {'yes' if r['fits'] else 'NO ':3s} {r['max_micro']:5d} "
                     f"{days(r):>6s} {speedup(r):>7s}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Single-GPU memory / time scoreboard.")
    ap.add_argument("--params", type=float, default=5e9)
    ap.add_argument("--d", type=int, default=0, help="d_model (0 = pick from --params)")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--tied", action="store_true", help="output head tied to the embedding")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--gpu_gb", type=float, default=96,
                    help="GPU memory as sold, in GiB (a 24 GB card = 24; nvidia-smi MiB / 1024)")
    ap.add_argument("--headroom", type=float, default=0.85,
                    help="usable fraction of GPU memory (device.find_batch_size keeps 0.85)")
    ap.add_argument("--overhead_gb", type=float, default=None,
                    help="CUDA context + workspaces in GiB (default 3, fine-tune table 2)")
    ap.add_argument("--peak_tflops", type=float, default=503.8)
    ap.add_argument("--mfu", type=float, default=0.40)
    ap.add_argument("--tokens", type=float, default=1e11)
    ap.add_argument("--fp8_speedup", type=float, default=1.20)
    ap.add_argument("--muon_flops", type=float, default=0.52)
    ap.add_argument("--global_batch_tokens", type=float, default=524288,
                    help="tokens per optimizer step (Muon Newton-Schulz cost is per step)")
    ap.add_argument("--ns_steps", type=int, default=5, help="Newton-Schulz iterations per step")
    ap.add_argument("--power_tput", type=float, default=0.95)
    ap.add_argument("--act_factor", type=float, default=1.0,
                    help="scale the measured activation constants (0.7 ~ fused/compiled kernels)")
    ap.add_argument("--ce_chunk", type=int, default=2048,
                    help="chunked cross-entropy tokens per chunk (fine-tune table: 0 = unchunked)")
    ap.add_argument("--bits_per_token", type=float, default=5.97,
                    help="TokPack size (5.97 measured on TinyStories with a 4k vocab; a 32k-vocab "
                         "web corpus needs roughly 7.3-9)")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--finetune", action="store_true",
                    help="print the fine-tuning memory table (full / LoRA / QLoRA) instead")
    ap.add_argument("--micro", type=int, default=4, help="micro-batch for the fine-tune table")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                    help="LoRA dropout (bantam finetune default 0.05; each adapter keeps its mask)")
    a = ap.parse_args(argv)
    if a.finetune:
        d = a.d or default_width(a.params)
        L, N = model_dims(a.params, d, a.vocab, a.tied)
        oh = 2.0 if a.overhead_gb is None else a.overhead_gb
        print(fmt_finetune(finetune_table(N, d, L, a.seq, a.micro, a.gpu_gb, a.lora_r, oh,
                                          a.act_factor, a.lora_dropout, a.vocab, a.ce_chunk,
                                          a.headroom, a.tied), N, a.gpu_gb))
        return 0
    sb = scoreboard(a.params, a.d, a.vocab, a.seq, a.gpu_gb, a.peak_tflops, a.mfu, a.tokens,
                    a.fp8_speedup, a.muon_flops, a.power_tput,
                    overhead_gb=3.0 if a.overhead_gb is None else a.overhead_gb,
                    act_factor=a.act_factor, headroom=a.headroom, ce_chunk=a.ce_chunk,
                    tied=a.tied, global_batch_tokens=a.global_batch_tokens, ns_steps=a.ns_steps)
    print(fmt(sb, a.tokens, a.markdown))
    print()
    for name, b in storage_table(sb["params"], a.tokens, a.bits_per_token):
        print(f"{name:66s} {b / GB:8.1f} GB")
    return 0


if __name__ == "__main__":
    main()
