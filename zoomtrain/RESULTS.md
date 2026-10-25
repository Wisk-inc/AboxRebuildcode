# Results

Everything below was measured in this repository on a 4-core CPU sandbox
(no GPU) on real data: TinyStories (HuggingFace `roneneldan/TinyStories`, V1
split, 4096-token BPE trained here) and ERA5 5.625° (WeatherBench2). Every run's
JSON log is under `results/`; `experiments/summarize.py` regenerates the tables.
Sections marked *pending* are still running.

Model for all LLM numbers: 5.77M-parameter decoder (d=256, 6 layers, 4 heads,
context 256). "Equal FLOPs" means the same analytic training-FLOP budget
(6.8e13 unless stated; ~450 baseline steps of 16×256 tokens). Validation loss
is next-token loss in nats on 2,048 held-out sequences, always at k=1.

## 1. Optimizer: Muon vs AdamW (equal FLOPs, tuned LRs)

| optimizer | best LR (sweep) | val loss at 6.8e13 | notes |
|---|---|---|---|
| AdamW | 1e-3 | **3.332** | 5e-4: 4.23, 2e-3: 4.09 at 1/3 budget |
| Muon (hidden matrices) + AdamW (rest) | 1.6e-2 | **2.734 / 2.727 / 2.750** (3 seeds) | 8e-3: 3.23, 3.2e-2: 3.21, 6.4e-2: 3.52 at 1/3 budget |

Muon's validation curve passes AdamW's *final* loss at ~40 % of the budget
(≈2.5× less compute to the same loss). Muon is Jordan et al. 2024; the
implementation, 8-bit momentum and bf16 updates are this repo's.

## 2. Zoom Training (the new method)

Zoom merges `k` neighbouring tokens into one position for the early part of
training and anneals back to `k = 1`. Main matrix at equal FLOPs, Muon:

| variant | val loss | vs baseline 2.737 ± 0.011 |
|---|---|---|
| **k=2 for the first 50 %, batch ×2 while zoomed** | **2.690 / 2.684 / 2.682** (3 seeds, mean **2.685 ± 0.004**) | better on every seed |
| k=2 for the first 50 %, same batch | 2.723 | ≈ baseline |
| k=2 first 50 % + per-stage LR restart | 2.760 | worse |
| k=2 first 50 % + restart + batch ×2 | 2.787 | worse |
| original 4→2→1 (35/25/40 %) | 2.875 / 2.872 / 2.874 (3 seeds) | worse |
| 4→2→1 with bag-of-tokens loss (Shao et al. style) | 2.860 | worse |
| 4→2→1 with plain mean merge (no gates) | 2.943 | worse |
| 4→2→1 with RoPE positions by patch | 2.857 | worse |
| 4→2→1 with linear (d×d) merge | *pending* | |
| 4:50 % → 1 | 2.947 | worse |
| 4:60 %, 2:20 %, 1:20 % | 3.022 | worse |
| 4:20 %, 2:20 %, 1:60 % | 2.800 | worse |
| 4→2→1 with batch ×k while zoomed | 2.730 | ≈ baseline |
| interleaved k=1 steps / LR restarts on 4→2→1 | *pending* | |

**Compute-to-target.** Baseline runs completed at larger budgets (each with
its own full LR schedule): 1.0× → 2.734, 1.25× → 2.637, 1.5× → 2.563,
2× → 2.461. Interpolating, the baseline needs **1.12× the compute** to reach
the Zoom result (≈ **11 % saved**). Modest, but real and consistent across
seeds. Why the original design lost: during k=4 stages the body drifts away
from k=1 geometry (its k=1 loss *rises* mid-stage) and the fine stage only gets
the LR tail. The k=2 / half-budget / doubled-batch variant avoids both.
`results/llm/zoom_compute_to_target.json`.

Weather (2×2 patch merge on the ERA5 ViT): *pending* (matrix running).

## 3. Low-memory training (same model, matched steps)

| configuration | val loss | optimizer state | weights |
|---|---|---|---|
| AdamW fp32 state (100 steps) | 4.4824 | 46.2 MB | 23.1 MB |
| AdamW 8-bit state | 4.4792 | 11.8 MB | 23.1 MB |
| Muon fp32 state (100 steps) | 3.5719 | 27.3 MB | 23.1 MB |
| Muon 8-bit state | 3.5812 | 7.0 MB | 23.1 MB |
| Muon 8-bit + bf16 weights (stochastic rounding) | 3.5752 | 7.0 MB | 11.5 MB |
| Muon fp32 + activation recompute + grad accumulation ×2 | 3.5719 (identical) | | |
| **full budget:** Muon 8-bit + bf16 weights | **2.728** (baseline 2.737 ± 0.011) | 7.0 MB | 11.5 MB |

Weights + state: 50.4 MB → 18.5 MB (2.7×) for no measurable quality change.

## 4. Storing trained weights (fully trained 5.8M model, val loss 2.7365)

| format | bytes/param | val loss | delta |
|---|---|---|---|
| fp32 in RAM | 4.00 | 2.7365 | — |
| bf16 in RAM | 2.00 | 2.7365 | +0.0000 |
| int8 weight-only (block 128) | 1.02 | 2.7367 | +0.0002 |
| int4 weight-only (block 64) | 0.53 | 2.7623 | +0.026 |
| WeightPack file, fp32 + zstd | 3.55 | 2.7365 | 0 |
| WeightPack file, bf16 lossless (byte-shuffle + zstd) | 1.39 | 2.7365 | 0 |
| WeightPack file, int8 | 0.94 | 2.7366 | +0.0001 |
| WeightPack file, int4 | 0.46 | 2.7624 | +0.026 |

For a pretrained 135M model (SmolLM2) the picture changes: int8 is still free
(+0.003) but 4-bit costs +0.18 (NF4, embeddings at int8) to +0.52 (linear
int4 everywhere) — see §7.

## 5. Storing data

| store | input | size | ratio | exact? |
|---|---|---|---|---|
| TokPack v2 (rank remap + LEB128 + zstd-19 with 1 MB dictionary, 16K-token chunks) | 116.8M TinyStories tokens as uint16 (233.5 MB) | **87.2 MB**, 5.97 bits/token | **2.68×** | yes, bit-exact; 13M tokens/s streaming |
| TokPack v1 (byte-shuffle) | same | 124.6 MB | 1.87× | yes (the shuffle hurt; replaced) |
| GridPack (int16 with error ≤ 5 % of the 6 h change std, temporal delta, zstd) | ERA5 2010-2018, 6 channels, float32 646 MB | **123 MB**, 6.1 bits/value | **5.2×** | bounded error; float16 would overflow on pressure |
| MinHash dedup (8-gram, 64 perms, 8 bands, J ≥ 0.8) | TinyStories V1 train shard | 21,800 exact duplicates removed (4 %), 0 near-duplicates (verified with injected edits) | | |

## 6. Against published TinyStories models

Scored on **1,000 validation stories that appear in neither TinyStories V1
train (30 % of the official validation set does) nor llama2.c's training
shards (69 % does)**, in bits per byte so different tokenizers compare
fairly. Published compute from each model's config / README
(`experiments/published_compute.json`).

| model | params | training FLOPs | bits/byte |
|---|---|---|---|
| **Bantam 5.8M, one completed CPU run** | 5.8M | **6.4e13** | **1.029** |
| llama2.c stories260K (Karpathy) | 0.26M | 1.0e16 | 1.097 |
| TinyStories-1M (Eldan & Li) | 3.7M | ~2.1e17 | 0.873 |
| TinyStories-3M | 8.3M | ~4.7e17 | 0.719 |
| TinyStories-8M | 19.7M | ~8.8e17 | 0.651 |
| TinyStories-28M | 52M | ~1.9e18 | 0.586 |
| llama2.c stories15M | 15M | 3.6e18 | 0.577 |
| TinyStories-33M | 68M | ~3.9e18 | 0.578 |
| llama2.c stories42M | 42M | 5.4e18 | 0.516 |
| llama2.c stories110M | 110M | 1.7e19 | 0.480 |

Bantam beats stories260K with 157× less compute (partly a model-size effect);
it has not reached TinyStories-1M, which used ~3,000× more compute. A 10×
run is in progress (its mid-schedule checkpoints score worse than completed
runs at the same compute and are drawn hollow in
`results/plots/published_bpb_vs_compute.png`).

## 7. Fine-tuning a real HuggingFace model (SmolLM2-135M, Dolly-15k, CPU)

Rank-16 LoRA on all projections, 40 steps of batch 2×2, seq 256, Muon with
8-bit state, response-only loss (`results/finetune/`):

| frozen base | held-out loss before → after | model RAM | output |
|---|---|---|---|
| fp32 (LoRA) | 2.871 → 2.792 | 538 MB | 19.7 MB adapter |
| int4 linear everywhere (QLoRA) | 3.389 → 3.178 | 106 MB | 19.7 MB adapter |

Base-format cost measured before training on 100 held-out examples:
fp32 2.702; int8 2.704 (+0.003, 200 MB); NF4 linears + int8 embeddings 2.878
(+0.18, 140 MB); linear int4 + int8 embeddings 3.035; NF4 everything 3.042;
linear int4 everything 3.222. Recommendation encoded in the defaults: int8
for small bases, NF4 for 7B+ (where the QLoRA paper finds it near-lossless;
not measurable here).

## 8. Upgrade ablations (WSD schedule, ReLU², softcap, zero-init, Polar Express, QK-norm, Zoom batch ×4, 2:33 %, 2:65 %, bundles)

*pending* — queue `experiments/queues/upgrades.txt`.

## 9. Runtime findings

- **Flush-to-zero on CPU**: denormal softmax gradients made a
  (8192×4096)·(4096×256) matmul 15–23× slower (897 → 39 ms). One line
  (`torch.set_flush_denormal(True)` before the first parallel op) fixes it.
- **Threads**: 2 jobs × 2 threads gave 24 % more total throughput than 1 job ×
  4 threads on this CPU (160 vs 129 GFLOP/s).
- **DDP**: two CPU processes trained one Zoom model together (gloo); losses
  decreased normally; `world_size` recorded in the log.
- **Stop/resume**: bit-identical to an uninterrupted run (test).

## Limitations

- One model size (5.8M) for all LLM pretraining claims; one fine-tuning
  model (135M). Gains at larger scale are literature-backed, not measured.
- The LR was tuned on the baseline and reused for Zoom (favours the baseline).
- Nothing ran on a GPU; the GPU code paths are untested beyond CPU logic.
- Published-model compute figures are reconstructed from configs and READMEs;
  ranges are given where the step count was not published.
