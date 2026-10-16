"""Memory and time estimator for single-GPU transformer training.

    python -m zoomtrain.memory --params 5e9 --d 4096 --layers 32 --vocab 32000 \
        --seq 2048 --micro 8 --gpu_gb 96 --tflops 250 --mfu 0.4 --tokens 1e11

Prints, for several training configurations, the bytes/parameter breakdown,
activation memory with/without checkpointing and chunked cross-entropy, the
largest micro-batch that fits, and the wall-clock for the given token count.

Formulas (bf16 activations, FlashAttention so no T^2 term):
  activations stored per layer, no recompute:   34 * B*T*d bytes
                                                (Korthikanti et al., 2022)
  with full checkpointing: 2 * B*T*d per layer (block input) + one layer's
                                                34 * B*T*d transient
  logits: B*T*V*4 (fp32 loss)  or  B*chunk*V*4 with chunked cross-entropy
  training FLOPs: 6*N*D, +2*N*D with full recompute
"""
import argparse
import math

GB = 1024 ** 3


def config_table(N, d, L, V, T, B, gpu_gb, tflops, mfu, tokens, chunk=1024,
                 muon_mult=1.0, overhead_gb=2.5, hidden_frac=0.9):
    rows = []
    configs = [
        ("fp32 weights + AdamW fp32 (naive)", 4, 4, "adamw", 4, False, False),
        ("bf16 autocast, fp32 master + AdamW fp32", 6, 4, "adamw", 4, False, False),
        ("bf16 weights(SR) + AdamW 8-bit, ckpt, chunked CE", 2, 2, "adamw", 1, True, True),
        ("bf16 weights(SR) + Muon 8-bit, ckpt, chunked CE", 2, 2, "muon", 1, True, True),
        ("bf16 weights(SR) + Muon 8-bit, no ckpt, chunked CE", 2, 2, "muon", 1, False, True),
    ]
    for name, wb, gb_, opt, sb, ckpt, chunked in configs:
        if opt == "adamw":
            state = N * 2 * sb
        else:  # Muon: momentum only on hidden matrices, AdamW on the rest
            state = N * hidden_frac * sb + N * (1 - hidden_frac) * 2 * sb
        static = N * (wb + gb_) + state
        per_tok_layer = 34 * d
        if ckpt:
            act = L * 2 * d * B * T + per_tok_layer * B * T
        else:
            act = L * per_tok_layer * B * T
        logits = (B * chunk if chunked else B * T) * V * 4 + B * T * 2 * d * (0 if chunked else 1)
        total = static + act + logits + overhead_gb * GB
        # largest micro-batch that fits
        per_b = (L * 2 * d * T + per_tok_layer * T if ckpt else L * per_tok_layer * T) \
            + (chunk if chunked else T) * V * 4
        max_b = int((gpu_gb * GB - static - overhead_gb * GB) // per_b)
        flops = (8 if ckpt else 6) * N * tokens
        secs = flops / (tflops * 1e12 * mfu)
        rows.append({"config": name, "static_gb": static / GB, "act_gb": act / GB,
                     "logits_gb": logits / GB, "total_gb": total / GB,
                     "fits": total <= gpu_gb * GB, "max_micro": max(0, max_b),
                     "days": secs / 86400, "days_muon_adj": secs / 86400 /
                     (muon_mult if opt == "muon" else 1.0)})
    return rows


def fmt(rows):
    hdr = f"{'config':52s} {'static':>7s} {'act':>6s} {'logit':>6s} {'total':>7s} fits {'maxB':>5s} {'days':>7s} {'days(x)':>8s}"
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['config']:52s} {r['static_gb']:7.1f} {r['act_gb']:6.1f} {r['logits_gb']:6.1f} "
                   f"{r['total_gb']:7.1f} {'yes' if r['fits'] else 'NO ':4s} {r['max_micro']:5d} "
                   f"{r['days']:7.1f} {r['days_muon_adj']:8.1f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=5e9)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--gpu_gb", type=float, default=96)
    ap.add_argument("--tflops", type=float, default=250, help="dense bf16 peak")
    ap.add_argument("--mfu", type=float, default=0.4)
    ap.add_argument("--tokens", type=float, default=1e11)
    ap.add_argument("--muon_mult", type=float, default=2.0,
                    help="compute-to-target multiplier credited to Muon in the last column")
    args = ap.parse_args()
    rows = config_table(args.params, args.d, args.layers, args.vocab, args.seq, args.micro,
                        args.gpu_gb, args.tflops, args.mfu, args.tokens, muon_mult=args.muon_mult)
    print(f"N={args.params:.2e} params, d={args.d}, L={args.layers}, V={args.vocab}, "
          f"T={args.seq}, micro-batch={args.micro}, GPU={args.gpu_gb:.0f} GB, "
          f"{args.tflops:.0f} TFLOPS x MFU {args.mfu:.2f}, {args.tokens:.2e} tokens")
    print(fmt(rows))
    print("days(x): wall-clock after dividing Muon rows by --muon_mult "
          "(measured 2.5x at 5.8M params here; ~2x reported at 1-16B).")


if __name__ == "__main__":
    main()
