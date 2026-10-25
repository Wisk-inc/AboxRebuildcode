"""Memory and time scoreboard for single-GPU LLM training.

    python -m zoomtrain.memory                 # 5B model on one RTX PRO 6000 Blackwell
    python -m zoomtrain.memory --tokens 2e10   # a 20B-token run
    python -m zoomtrain.memory --markdown      # table for docs

Each row adds one technique to the row above and reports static memory
(weights + grads + optimizer state), activation memory, whether it fits,
the largest micro-batch, training FLOPs and wall-clock days to reach the
same loss as the standard recipe trained on --tokens tokens.

Assumptions (defaults; every one is a flag):
  GPU       RTX PRO 6000 Blackwell Workstation: 96 GB, 503.8 TFLOPS dense BF16
            (NVIDIA RTX Blackwell PRO architecture whitepaper, Appendix A).
  MFU       0.40 of dense BF16 peak (conservative). nanochat on one RTX PRO 6000
            measured ~297 TFLOPS = 59 % at 600 W and ~40 % at 300 W
            (karpathy/nanochat #288, d20 = 3.49 GFLOP/token); 2 cards ~50 % (#28).
  FP8       x1.20 throughput for FP8 linears (Transformer Engine current
            scaling is supported on SM 12.0); reports range 20-30 %. Not
            measured here.
  Muon      reaches AdamW's loss with 0.52x the FLOPs (Liu et al. 2025,
            "Muon is Scalable for LLM Training", compute-optimal 1-16B runs).
            Measured here at 5.8M params: Muon at 1/3 of the compute beat AdamW
            at full compute (>= 3x), not used in the table.
  bf16 + SR bf16 weights/grads with stochastic rounding match fp32-master mixed
            precision up to 6.7B (Ozkara et al. 2025); 8-bit Adam states match
            32-bit up to 1.5B (Dettmers et al. 2021).
  Power     450 W cap keeps ~95 % throughput on this card in a ResNet-50 training
            sweep (naveen.ing); for LLM training the only datapoint is nanochat at
            300 W: -32 % throughput, +36 % tokens per watt (#288).
Activation memory (bf16, FlashAttention): 34*T*d bytes per layer per sequence
(Korthikanti et al. 2022; llm.c GPT-2 1.6B measured within ~1 %). HF-style
Llama code measured ~1.6x higher (LOMO) -> --act_factor 1.6. Full checkpointing
keeps 2*T*d per layer plus one layer's activations and costs one extra forward
(8ND instead of 6ND; Megatron measured +30-40 % wall-clock).
"""
import argparse

GB = 1e9


def default_width(params):
    """A conventional d_model for a parameter count (GPT-2/Llama-style aspect ratios)."""
    for limit, d in ((2e7, 384), (6e7, 512), (2e8, 768), (5e8, 1024), (1.5e9, 2048),
                     (4e9, 3072), (1e10, 4096), (2e10, 5120)):
        if params <= limit:
            return d
    return 8192


def model_dims(params, d, vocab):
    """Layers for a target parameter count: 12*L*d^2 + 2*V*d (untied embeddings)."""
    d = d or default_width(params)
    L = max(1, round((params - 2 * vocab * d) / (12 * d * d)))
    return L, 12 * L * d * d + 2 * vocab * d


def scoreboard(params=5e9, d=0, vocab=32000, seq=2048, gpu_gb=96.0, peak_tflops=503.8,
               mfu=0.40, tokens=1e11, fp8_speedup=1.20, muon_flops=0.52, power_tput=0.95,
               overhead_gb=3.0, hidden_frac=None, act_factor=1.0):
    d = d or default_width(params)
    L, N = model_dims(params, d, vocab)
    if hidden_frac is None:
        hidden_frac = 12 * L * d * d / N
    act_seq = 34 * seq * d * L * act_factor           # bytes per sequence, no recompute
    act_seq_ckpt = 2 * seq * d * L                    # stored block inputs
    act_layer = 34 * seq * d * act_factor             # one layer recomputed at a time
    logits_seq = seq * vocab * 6                      # bf16 logits + fp32 loss buffer
    logits_chunk = 1024 * vocab * 6                   # chunked cross-entropy

    def row(name, w, g, state, ckpt, chunked, tok_mult=1.0, speed=1.0, note=""):
        static = N * (w + g + state)
        per_seq = (act_seq_ckpt if ckpt else act_seq) + (0 if chunked else logits_seq)
        fixed = static + overhead_gb * GB + (act_layer if ckpt else 0) + \
            (logits_chunk if chunked else 0)
        max_micro = int((gpu_gb * GB - fixed) // per_seq) if gpu_gb * GB > fixed else 0
        micro = max(1, min(max_micro, 8))
        flops = (8 if ckpt else 6) * N * tokens * tok_mult
        secs = flops / (peak_tflops * 1e12 * mfu * speed)
        return {"config": name, "bytes_per_param": w + g + state,
                "static_gb": static / GB, "act_gb": micro * per_seq / GB,
                "total_gb": (fixed + micro * per_seq) / GB, "fits": max_micro >= 1,
                "max_micro": max_micro, "flops": flops, "days": secs / 86400, "note": note}

    muon_state = hidden_frac * 1 + (1 - hidden_frac) * 2      # 8-bit momentum / 8-bit Adam
    rows = []
    r = row("1. Standard: AMP bf16, fp32 master + fp32 AdamW", 4 + 2, 4, 8, False, False,
            note="fp32 weights + bf16 autocast copies")
    if not r["fits"]:
        r = row("1. Standard: AMP bf16, fp32 master + fp32 AdamW (+ act. ckpt to fit)",
                4 + 2, 4, 8, True, False, note="does not fit without activation checkpointing")
    rows.append(r)
    rows.append(row("2. + bf16 weights (stochastic rounding) + 8-bit AdamW + chunked CE",
                    2, 2, 2, False, True, note="no fp32 master; fits without recompute"))
    rows.append(row("3. + Muon (8-bit momentum)", 2, 2, muon_state, False, True,
                    tok_mult=muon_flops, note=f"same loss with {muon_flops:.2f}x the FLOPs"))
    rows.append(row("4. + FP8 linears", 2, 2, muon_state, False, True, tok_mult=muon_flops,
                    speed=fp8_speedup, note=f"x{fp8_speedup:.2f} throughput (literature)"))
    rows.append(row("5. + 450 W power cap (75 % power)", 2, 2, muon_state, False, True,
                    tok_mult=muon_flops, speed=fp8_speedup * power_tput,
                    note=f"x{power_tput:.2f} throughput, 25 % less power"))
    return {"layers": L, "params": N, "d": d, "rows": rows}


def storage_table(N, tokens, bits_per_token=5.97):
    """Disk/RAM sizes for weights, resumable checkpoints and the token corpus."""
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
        (f"corpus: TokPack v2 ({bits_per_token:.2f} bits/token, measured on 117M TinyStories tokens)",
         bits_per_token / 8 * tokens),
    ]


def fmt(sb, tokens, markdown=False):
    base = sb["rows"][0]["days"]
    lines = []
    if markdown:
        lines.append("| configuration | bytes/param | static GB | total GB | fits | max micro-batch | days | speedup |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sb["rows"]:
            lines.append(f"| {r['config']} | {r['bytes_per_param']:.1f} | {r['static_gb']:.0f} | "
                         f"{r['total_gb']:.0f} | {'yes' if r['fits'] else 'no'} | {r['max_micro']} | "
                         f"{r['days']:.0f} | {base / r['days']:.2f}x |")
        return "\n".join(lines)
    lines.append(f"model: {sb['params']/1e9:.2f}B params ({sb['layers']} layers), "
                 f"reference = standard recipe on {tokens:.1e} tokens")
    hdr = f"{'configuration':72s} {'B/p':>5s} {'static':>7s} {'total':>6s} fit {'maxB':>5s} {'days':>6s} {'speedup':>7s}"
    lines += [hdr, "-" * len(hdr)]
    for r in sb["rows"]:
        lines.append(f"{r['config']:72s} {r['bytes_per_param']:5.1f} {r['static_gb']:6.0f}G "
                     f"{r['total_gb']:5.0f}G {'yes' if r['fits'] else 'NO ':3s} {r['max_micro']:5d} "
                     f"{r['days']:6.0f} {base / r['days']:6.2f}x")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=5e9)
    ap.add_argument("--d", type=int, default=0, help="d_model (0 = pick from --params)")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--gpu_gb", type=float, default=96)
    ap.add_argument("--peak_tflops", type=float, default=503.8)
    ap.add_argument("--mfu", type=float, default=0.40)
    ap.add_argument("--tokens", type=float, default=1e11)
    ap.add_argument("--fp8_speedup", type=float, default=1.20)
    ap.add_argument("--muon_flops", type=float, default=0.52)
    ap.add_argument("--power_tput", type=float, default=0.95)
    ap.add_argument("--act_factor", type=float, default=1.0,
                    help="activation memory vs the 34*T*d formula (1.6 for HF Llama code)")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    sb = scoreboard(a.params, a.d, a.vocab, a.seq, a.gpu_gb, a.peak_tflops, a.mfu, a.tokens,
                    a.fp8_speedup, a.muon_flops, a.power_tput, act_factor=a.act_factor)
    print(fmt(sb, a.tokens, a.markdown))
    print()
    for name, b in storage_table(sb["params"], a.tokens):
        print(f"{name:66s} {b / GB:8.1f} GB")


if __name__ == "__main__":
    main()
