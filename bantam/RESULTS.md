# Bantam 0.3.0: experimental results

This is the full report behind the numbers in the README. Every number here
comes from a JSON log under `results/` (or, where noted, from a stats file next
to the data). `python3 experiments/summarize.py --plots` regenerates the
tables in `results/SUMMARY.md`, `results/summary.json` and the plots in
`results/plots/`.

**Where it ran.** Every run in this report ran on one CPU-only sandbox: 4 cores,
16 GB RAM, PyTorch on CPU, usually 1 or 2 threads per run with other runs
sharing the machine. **Nothing has been run on a GPU yet.** Speed and memory
claims for GPUs are therefore not made here. The 5B scoreboard
([`docs/SCOREBOARD_5B.md`](docs/SCOREBOARD_5B.md)) is arithmetic, not a
measurement.

**How to read the tables.**
- `n` is the number of seeds. "±" is the sample standard deviation over seeds.
  Where one seed was run twice (a replicate), the two runs are averaged first,
  so `n` counts distinct seeds.
- `n = 1` marks a single run. Treat a single-seed difference smaller than
  about 0.02 nats (LLM) as noise: the baseline's seed spread is 0.012.
- LLM losses are validation cross-entropy in nats per token. Lower is better.
- Weather losses are one-step MSE on normalized fields. Lower is better.

## Contents

1. [Headline numbers](#1-headline-numbers)
2. [Setup](#2-setup)
3. [Optimizer: Muon vs AdamW](#3-optimizer-muon-vs-adamw)
4. [Learning-rate sweeps](#4-learning-rate-sweeps)
5. [Zoom schedule ablations](#5-zoom-schedule-ablations)
6. [Upgrade ablations, and QK-norm with Zoom](#6-upgrade-ablations-and-qk-norm-with-zoom)
7. [Compute-to-target and long-budget baselines](#7-compute-to-target-and-long-budget-baselines)
8. [Wall-clock on this CPU](#8-wall-clock-on-this-cpu)
9. [Low-memory training](#9-low-memory-training)
10. [Storing trained weights](#10-storing-trained-weights)
11. [Storing data](#11-storing-data)
12. [Against published TinyStories models](#12-against-published-tinystories-models)
13. [Weather (ERA5)](#13-weather-era5)
14. [Fine-tuning SmolLM2-135M](#14-fine-tuning-smollm2-135m)
15. [Fine-tuning race: Hugging Face Trainer + PEFT vs Bantam](#15-fine-tuning-race-hugging-face-trainer--peft-vs-bantam)
16. [What changed since 0.2](#16-what-changed-since-02)
17. [Open questions and what has not been measured](#17-open-questions-and-what-has-not-been-measured)
18. [Reproducing](#18-reproducing)

## 1. Headline numbers

Small language model: TinyStories, 5.77M parameters, a fixed budget of 6.8e13
model FLOPs per run (section 2.3).

| result | val loss | n | baseline compute needed to match |
|---|---|---|---|
| Plain Muon baseline | 2.7370 ± 0.0118 | 3 | 1.00x |
| AdamW baseline (tuned LR 1e-3) | 3.3318 | 1 | worse than the Muon baseline |
| Zoom, k=2 for the first 50 % of compute, batch x2 while merged | 2.6851 ± 0.0041 | 3 | 1.12x |
| Zoom, k=2 for the first 33 %, batch x2 | 2.6795 ± 0.0053 | 2 | 1.14x |
| QK-norm (a known trick, not Bantam's) | 2.6385 ± 0.0029 | 2 | 1.24x |
| QK-norm + Zoom k=2 33 % batch x2 | 2.5894 ± 0.0057 | 2 | 1.41x |

- **Muon is the biggest single win** at this scale. AdamW is a single seed.
- **Zoom saves about 11 % of compute** (1.12x) on 3 seeds. The gain is real but
  small, and smaller than what the prior art reports (section 5.1).
- **QK-norm beats Zoom alone.** Zoom's gain (about -0.05 nats) is the same with
  and without QK-norm, so the two stack.
- Most other upgrades did not help (section 6).

Weather: ERA5 at 5.625°, a 1.26M-parameter ViT, 5.5e13 model FLOPs per run.

| result | one-step MSE | n | z500 RMSE at 24 h / 72 h (m²/s²) |
|---|---|---|---|
| Muon baseline | 0.1240 ± 0.0006 | 2 | 217 / 539 (seed 0) |
| Muon + Zoom merge (k=4 for 60 %) | 0.1588 ± 0.0005 | 2 | 253 / 600 (seed 0) |
| Muon + low-resolution warm-up (4x coarser grid for 60 %) | 0.1119 | 1 | 194 / 511 |
| AdamW baseline | 0.2292 | 1 | 346 / 737 |
| Persistence (tomorrow = today) | | | 569 / 907 |

- **Zoom made weather worse under Muon** (+28 % MSE). This is a negative result.
- Low-resolution warm-up (established practice) helped by about 10 %, or about
  6-7 % after charging its extra optimizer compute (section 13.4).
- **The weather numbers are on the test year, which was also used to choose the
  learning rate** (section 13.5).

Published models (1,000 clean TinyStories validation stories, bits per byte,
each family scored in its own training format):

- Bantam 5.8M scores **1.029 BPB** with 6.4e13 training FLOPs. llama2.c
  stories260K scores 1.097 with 1.0e16 FLOPs. Bantam beats it with
  **at least 157x less compute** (6.4e13 is an upper bound on what Bantam
  needs, read off its one completed run). This is mostly a model-size effect:
  stories260K has 22x fewer parameters and is capacity-limited.
- **Bantam is behind every TinyStories model.** TinyStories-1M scores 0.822,
  0.207 BPB better, with about 3,350x more compute.

## 2. Setup

### 2.1 Machine and software

- CPU-only sandbox, 4 cores, 16 GB RAM, PyTorch CPU build. No GPU.
- Most runs used 1 or 2 threads, with 2-4 runs sharing the machine. Wall-clock
  times are therefore noisy (section 8).
- Every LLM pretraining run in this report except `save_muon_base_repro` was
  produced by code from before the 0.3 changes (their logs lack 0.3 fields
  such as `data_source` and `opt_flops`). 0.3 keeps the default FLOP budget
  charge and the fp32 evaluation unchanged. Muon's Newton-Schulz now differs
  at fp32 rounding level (a 0.3 retrain of `save_muon_base` gave 2.7352 vs
  2.7365). The bf16, 8-bit and Polar Express numerics changed more; sections 6
  and 9 say which rows need a rerun.

### 2.2 LLM data

- **Corpus:** TinyStories V1 (`roneneldan/TinyStories`), training shard and the
  official validation split.
- **Tokenizer:** a 4,096-token byte-level BPE trained here (`bantam prepare-text`).
- **Training stream:** 116,765,743 tokens after `bantam pack-text` deduplication,
  stored as TokPack. Every LLM run's `data_epochs` matches this stream.
- **Deduplication:** 21,800 exact duplicate documents were dropped from the
  training split (4.1 % of its 529,875 documents). Per the 0.3 audit, 1,550
  of these were training copies of validation stories and the rest were
  duplicates inside the training split. MinHash (8-gram, 64 permutations,
  8 bands, Jaccard ≥ 0.8) found 0 near-duplicates. So no validation story
  appears verbatim in the training stream. The raw `train.bin` written by 0.2 `prepare-text` did still contain
  those 1,550 stories. 0.3 `prepare-text` drops them too (DATA-1).
- **Validation:** 4,884,029 tokens (21,990 stories).

### 2.3 LLM model and budget

- **Model:** ZoomGPT, a GPT-style decoder: d=256, 6 layers, 4 heads, RoPE,
  GELU MLP, output head tied to the embedding, context 256 tokens.
  5,770,496 parameters (gate parameters add a few thousand when Zoom is on).
- **Batch:** 16 sequences of 256 tokens (4,096 tokens per step) unless a Zoom
  schedule scales it.
- **Budget:** 6.8e13 model FLOPs for every main run. The plain baseline uses
  them in 450 steps (1.84M tokens, 0.016 epochs). Runs stop when the budget is
  spent, so a Zoom run takes a different number of steps.
- **Optimizers:** Muon (Jordan et al. 2024) on the hidden weight matrices with
  AdamW on embeddings, head and norms, LR 1.6e-2, cosine schedule, 5 % warm-up,
  weight decay 0.1. AdamW-only runs use LR 1e-3.

### 2.4 FLOP accounting: what the budget counts and what it leaves out

The budget charges `ZoomGPT.flops_per_step()` in its default mode:
forward + backward = 3x forward, with the forward counting the block matmuls
(24·d² per position per layer), causal attention at half the n x n score
matrix (2·n·d per position per layer) and the output head (2·d·V per output
position). The head runs at full resolution, so a Zoom step still pays for
every token's logits.

Not charged:
- **Optimizer FLOPs.** Muon's Newton-Schulz iteration costs about 3.0e10 FLOPs
  per optimizer step for this model. That is **20 % of the model FLOPs of a
  baseline step** (`save_muon_base_repro`: `opt_flops` 1.37e13 against
  `model_flops` 6.81e13). It is a per-step cost, so a run that takes more
  steps gets more of it for free. The k=2 batch-x2 Zoom runs take fewer steps
  than the baseline (422-432 vs 450), so this favours the baseline slightly.
  The losing 4→2→1 runs took 808-1,073 steps and got 1.8-2.4x the baseline's
  optimizer compute, and still lost. AdamW's update costs about 12 FLOPs per
  parameter per step, which is negligible. 0.3 logs `opt_flops` and offers
  `--flops_mode total` to charge it; the committed runs predate that.
- **The rest of attention.** Counting attention as `torch.utils.flop_counter`
  does (the full n x n score matrix, recomputed by the SDPA backward;
  `flops_per_step(..., exact=True)`, logged by 0.3 as `model_flops_exact`)
  gives 8.5 % more than the budget charge for a k=1 step and 3.7 % more for a
  k=2 batch-x2 step. Kernels that skip the masked half (CPU flash attention)
  do less work than that count. Either way, the default convention does not
  over-credit Zoom; if anything it slightly under-credits it.
- Elementwise work (norms, the gate/mean merges, the optimizer's elementwise
  terms), the embedding lookup, and evaluation.

### 2.5 LLM evaluation protocol

- Loss is always measured at k=1 (no merging), in fp32 arithmetic.
- **Final val loss** (`final_val_loss`): the first 2,048 contiguous windows of
  256 tokens of the validation stream (524,288 tokens, 10.7 % of it). This is
  the number in every main table.
- **Other eval sizes.** LR sweeps: the first 512 windows. Low-memory 100-step
  runs: the first 1,024 windows. Training curves (`evals[]` and the plots): the
  first 256 windows. The early windows are easier, so the 256-window curve reads
  0.27-0.30 nats lower than the 2,048-window final loss of the same weights.
  Compare numbers only within one eval size. Each table states its size.
- 0.3 also scores the final model on the curve subset
  (`final_val_loss_eval_seqs`) so the two can be matched.

### 2.6 Weather data, model and evaluation

- **Data:** ERA5 regridded to 64 x 32 (5.625°), 6-hourly, 2010-2018, six
  channels (z500, t850, t2m, u10, v10, msl), plus static fields and time
  forcings. Train: 2010-2017 (11,688 steps). Test: 2018 (1,460 steps).
- **Model:** ZoomViT, d=128, 6 layers, 4 heads, 2x2-cell patches (512 tokens),
  1,264,792 parameters. It predicts the 6-hour change.
- **Budget:** 5.5e13 model FLOPs, batch 16, weight decay 0.05.
- **Metrics.** "One-step MSE": normalized one-step MSE on every third 2018 time
  step. "RMSE": latitude-weighted RMSE in physical units from 96 rollouts
  started at 00 UTC every 3 days from 2018-01-01 to 2018-10-13, at +6 h, +24 h
  and +72 h. Persistence is scored on the same initial conditions.
- **Caveat:** the committed runs have no separate validation year (WX-3,
  section 13.5).

### 2.7 What Zoom is

Zoom training merges each group of k neighbouring tokens into one position
for an early part of training, runs the transformer on the shorter sequence,
unmerges back to every token for the loss, and then finishes at k=1. The merge
is a learned per-slot gate.

This is a learned-gate variant of **patch-level training** (Shao, Meng and
Zhou, ICLR 2025), which averages K=4 token embeddings for 2/3 of the data and
reports about half the training cost at 370M-2.7B parameters. It is also close
to Hourglass and Funnel-Transformer shortening, learned dynamic pooling, and
sequence-length warm-up. Bantam's design differences are the learned gate (vs
averaging), per-token unmerge (vs a next-patch bag loss), and k=2 with a larger
batch while merged. See [`RESEARCH.md`](RESEARCH.md) and
[`docs/METHOD.md`](docs/METHOD.md).

## 3. Optimizer: Muon vs AdamW

Equal model FLOPs (6.8e13), TinyStories, final val loss on 2,048 windows.

| optimizer | LR (best in sweep) | val loss | n |
|---|---|---|---|
| AdamW | 1e-3 | 3.3318 | 1 |
| Muon (hidden matrices) + AdamW (embeddings, head, norms) | 1.6e-2 | 2.7370 ± 0.0118 | 3 |

How much compute does Muon save? Two definitions, both measured on the same
256-window curve metric:
- **Mid-curve crossing:** Muon's curve passes AdamW's final curve value (3.065)
  at 38-39 % of the budget on all three seeds.
- **Separate short run:** a Muon run with its own full schedule at 1/3 of the
  budget (2.3e13, `sweep_muon_lr1.6e-2`) reached 3.030, below AdamW's 3.065 at
  the full budget.

Both statements ignore Muon's uncharged Newton-Schulz FLOPs (about 20 % of
model FLOPs here, section 2.4). Charging them, the 1/3-budget Muon run used
about 0.41x AdamW's total compute and still won.

Two caveats:
- **Two things change at once.** In the Muon runs, the AdamW part (embeddings,
  tied head and norms: 1.05M of 5.77M parameters) also runs at LR 1.6e-2. That
  is 16x the tuned AdamW-only LR. The 0.2 claim that Moonlight-style RMS
  matching lets both share one LR is not supported by the sweep: Muon at 1e-3
  scores 3.937 vs 3.176 at 1.6e-2 (section 4). No ablation with a separate LR
  for the AdamW part has been run.
- AdamW is one seed, at the LR that was best among three (5e-4, 1e-3, 2e-3).
  Independent benchmarks find that carefully tuned AdamW shrinks Muon's lead
  (about 1.4x at 0.1B and 1.1x at 1.2B in Wen et al. 2025).

## 4. Learning-rate sweeps

### 4.1 LLM (2.3e13 FLOPs = 1/3 of the main budget, 152 steps, 1 seed each, final val on 512 windows)

| LR | AdamW | Muon |
|---|---|---|
| 5e-4 | 4.2252 | 4.2265 |
| 1e-3 | **3.9607** | 3.9369 |
| 2e-3 | 4.0903 | 3.6919 |
| 4e-3 | | 3.4402 |
| 8e-3 | | 3.2302 |
| 1.6e-2 | | **3.1759** |
| 3.2e-2 | | 3.2078 |
| 6.4e-2 | | 3.5223 |

Both optima are bracketed. The LR was tuned on the baseline and reused for
every Zoom and upgrade run, which favours the baseline.

### 4.2 Weather (1.8e13 FLOPs, 186 steps, 1 seed each, one-step MSE on 2018)

| LR | AdamW | Muon |
|---|---|---|
| 5e-4 | 0.4543 | 0.4496 |
| 1e-3 | 0.4143 | 0.3686 |
| 2e-3 | 0.3928 | 0.3201 |
| 4e-3 | **0.3911** | 0.2929 |
| 8e-3 | | 0.2672 |
| 1.6e-2 | | **0.2331** |
| 3.2e-2 | | 0.2567 |

- The main weather runs use Muon 1.6e-2 and AdamW 2e-3. AdamW's best value is
  at the top of its sweep (4e-3, within 0.002 of 2e-3), so AdamW may be
  under-tuned. 8e-3 was not tried.
- These LRs were chosen on the 2018 test year (section 13.5).

## 5. Zoom schedule ablations

Muon, LR 1.6e-2, 6.8e13 model FLOPs, final val on 2,048 windows. Baseline
2.7370 ± 0.0118 (n=3, 450 steps). "Compute" is the baseline compute needed to
reach the same loss (section 7). `<= 1.00x` means no better than the baseline.

| variant | flags | steps | val loss | n | compute |
|---|---|---|---|---|---|
| **k=2 for 50 %, batch x2 while merged** | `--zoom 2:0.5,1:0.5 --batch_scale_k 1` | 422 | **2.6851 ± 0.0041** | 3 | **1.12x** |
| **k=2 for 33 %, batch x2** | `--zoom 2:0.33,1:0.67 --batch_scale 2` | 432 | **2.6795 ± 0.0053** | 2 | **1.14x** |
| k=2 for 65 %, batch x2 | `--zoom 2:0.65,1:0.35 --batch_scale 2` | 414 | 2.7135 | 1 | 1.05x |
| k=2 for 50 %, batch x4 | `--zoom 2:0.5,1:0.5 --batch_scale 4` | 323 | 2.7272 | 1 | 1.02x |
| k=2 for 50 %, same batch | `--zoom 2:0.5,1:0.5` | 620 | 2.7233 | 1 | 1.03x |
| k=2 for 50 %, LR restart at the switch | `... --lr_restart 1` | 620 | 2.7603 | 1 | <= 1.00x |
| k=2 for 50 %, LR restart, batch x2 | `... --lr_restart 1 --batch_scale_k 1` | 422 | 2.7865 | 1 | <= 1.00x |
| 4→2→1 (35/25/40 %), the 0.1 design | `--zoom 4:0.35,2:0.25,1:0.4` | 808 | 2.8736 ± 0.0015 | 3 | <= 1.00x |
| 4→2→1, batch x k while merged | `... --batch_scale_k 1` | 386 | 2.7298 | 1 | 1.02x |
| 4→2→1, bag-of-tokens loss (Shao et al. style) | `... --out_mode bag` | 1,073 | 2.8603 | 1 | <= 1.00x |
| 4→2→1, plain mean merge (no gates) | `... --gated 0` | 808 | 2.9425 | 1 | <= 1.00x |
| 4→2→1, RoPE positions by patch | `... --rope_pos patch` | 808 | 2.8574 | 1 | <= 1.00x |
| k=4 for 50 % → 1 | `--zoom 4:0.5,1:0.5` | 840 | 2.9468 | 1 | <= 1.00x |
| 4:60 %, 2:20 %, 1:20 % | `--zoom 4:0.6,2:0.2,1:0.2` | 985 | 3.0220 | 1 | <= 1.00x |
| 4:20 %, 2:20 %, 1:60 % | `--zoom 4:0.2,2:0.2,1:0.6` | 674 | 2.7999 | 1 | <= 1.00x |

With AdamW (LR 1e-3, 1 seed each), the 4→2→1 schedule also lost: 3.5014 vs
3.3318.

**What made Zoom work here:** k=2 only (every k=4 schedule lost), and a larger
batch during the merged phase. Without the larger batch, the cheap k=2 steps
turn into many small optimizer steps (620 instead of 450). With batch x2 the
run takes slightly fewer steps than the baseline (422) and each merged step
sees twice the tokens. Restarting the LR at the switch hurt. The 0.1 design
(4→2→1 without batch scaling) is clearly worse than the baseline on all three
seeds.

Not run (queued in `experiments/queues/followup.txt`): the linear d x d merge
(`--merge_mode linear`), interleaved k=1 steps (`--mix_fine`), k=2 for 30 %,
and extra seeds of the same-batch k=2 schedule.

### 5.1 How this compares with prior art

Patch-level training (Shao et al.) reports about 0.5x training cost at
370M-2.7B parameters on the Pile. Bantam's best measured gain is 1.12-1.14x
(about 11-12 % saved) on one 5.8M-parameter model on an easy dataset. Shao et
al. also found that learned projections at the input and output hurt the later
token-level model, and that their advantage shrank with size: test loss was
better than token-level training at 370M and slightly worse at 2.7B. Whether
Bantam's learned gate helps or hurts at larger scale is unknown. The closest
untested competitor is sequence-length warm-up at equal FLOPs.

## 6. Upgrade ablations, and QK-norm with Zoom

Each row changes one thing from the plain Muon baseline (2.7370 ± 0.0118, n=3).
Same budget, LR and eval as section 5. These are other people's ideas that
Bantam implements: QK-norm (Henry et al. 2020; Dehghani et al. 2023), logit
softcap, zero-init output projections, ReLU² (Primer), WSD schedules, and Polar
Express (Amsel et al.).

| change | flag | val loss | n | Δ vs baseline | compute |
|---|---|---|---|---|---|
| **QK-norm** | `--qk_norm 1` | **2.6385 ± 0.0029** | 2 | **-0.098** | **1.24x** |
| Logit softcap 15 | `--softcap 15` | 2.7264 | 1 | -0.011 | 1.02x |
| Zero-init output projections | `--zero_init_proj 1` | 2.7317 | 1 | -0.005 | 1.01x |
| Softcap 15 + zero-init | both | 2.7277 | 1 | -0.009 | 1.02x |
| Polar Express orthogonalizer | `--ns polar` | 2.7547 | 1 | +0.018 | <= 1.00x |
| WSD schedule (decay over the last 45 %) | `--sched wsd` | 2.7725 | 1 | +0.036 | <= 1.00x |
| ReLU² MLP | `--mlp relu2` | 2.8288 | 1 | +0.092 | <= 1.00x |
| Softcap + zero-init + Zoom k=2 50 % batch x2 | | 2.6908 | 1 | -0.046 | 1.11x |

- **QK-norm is the strongest single change measured.** It matches what the
  modded-nanogpt speedrun found (record #5). It should be the default baseline
  for future Zoom work.
- **WSD and ReLU² were worse** at this scale. WSD used the LR tuned for cosine.
- Softcap and zero-init are within noise of the baseline.
- Polar Express was run with the 0.2 coefficients. 0.3 corrected them
  (OPT-8). At 5 steps the change is small (the singular-value range moves from
  [0.846, 1.124] to [0.840, 1.124]), but this row should be rerun.
- Softcap + zero-init added nothing on top of Zoom (2.6908 vs 2.6896 for Zoom
  alone on seed 0).
- The WSD, ReLU², softcap and zero-init runs were made earlier in the project,
  with an older build of the code (their logs lack fields later runs have) and
  before the environment was rebuilt. Their `data_epochs` matches the same
  116.77M-token deduplicated stream as every other run. Treat them as
  indicative.

### 6.1 Do QK-norm and Zoom stack? (2 x 2, seeds 0 and 1)

Zoom here is `--zoom 2:0.33,1:0.67 --batch_scale 2`. The matched baseline is
the two `up_base` runs from the same batch of experiments.

| | no Zoom | Zoom | Zoom effect |
|---|---|---|---|
| no QK-norm | 2.7307 ± 0.0081 | 2.6795 ± 0.0053 | **-0.051** |
| QK-norm | 2.6385 ± 0.0029 | 2.5894 ± 0.0057 | **-0.049** |
| QK-norm effect | -0.092 | -0.090 | |

The effects add up: Zoom gives about -0.05 nats with and without QK-norm, and
QK-norm about -0.09 with and without Zoom. Per seed, the Zoom effect was
-0.053 / -0.049 without QK-norm and -0.051 / -0.047 with it.
The plain baseline needs 1.41x the compute to match QK-norm + Zoom (1.42x,
1.39x per seed). That figure is measured against the plain baseline curve; no
QK-norm baseline curve at larger budgets exists, so how much compute Zoom
saves *on top of*
QK-norm is not known in FLOP terms.

## 7. Compute-to-target and long-budget baselines

**Question:** how much more compute does the plain baseline need to reach a
run's final loss?

### 7.1 Baseline curve (Muon, plain, each point a separate run with its own full cosine schedule)

| budget | FLOPs | steps | val loss | n | train time |
|---|---|---|---|---|---|
| 1.0x | 6.8e13 | 450 | 2.7370 ± 0.0118 | 3 | 790-1,067 s (1-2 threads) |
| 1.25x | 8.5e13 | 562 | 2.6366 | 1 | 1,708 s |
| 1.5x | 1.02e14 | 674 | 2.5634 | 1 | 2,091 s |
| 2x | 1.36e14 | 899 | 2.4607 | 1 | 2,399 s |

A 10x run (6.8e14, `results/published/long_muon_10x.json`) was killed at 30 %
and is not used anywhere.

### 7.2 Method

`experiments/summarize.py` computes it from completed runs only:
- The baseline for a run is the completed runs with the same optimizer, LR,
  model, data and eval set, and every Zoom and upgrade option at its default.
  AdamW runs are therefore compared with AdamW baselines.
- Final losses are averaged over seeds at each budget (replicates first).
- The FLOPs to reach the target are found by linear interpolation in
  log-FLOPs between neighbouring budgets. Curve points are never mixed with
  final losses.

### 7.3 Results

| run | val loss | n | compute (per seed) |
|---|---|---|---|
| Zoom k=2 50 % batch x2 | 2.6851 | 3 | **1.12x** (1.11, 1.12, 1.13) |
| Zoom k=2 33 % batch x2 | 2.6795 | 2 | **1.14x** (1.13, 1.15) |
| QK-norm | 2.6385 | 2 | **1.24x** (1.25, 1.24) |
| QK-norm + Zoom k=2 33 % batch x2 | 2.5894 | 2 | **1.41x** (1.42, 1.39) |

**Uncertainty.** The baseline points above 1x are single runs. An audit
re-derivation of the 1.12x figure (CLM-14) found 1.11-1.14x depending on which
seed anchors the 1x point and on a ±1 sd shift of the 1.25x point. So "about
11 % saved" is the right precision. The equal-FLOP loss gap itself is
consistent: every Zoom seed (2.682-2.690) beat every baseline seed
(2.726-2.750).

## 8. Wall-clock on this CPU

The runs shared a 4-core machine, so the same baseline took 790-1,067 s.
Within that noise:

| comparison (equal FLOPs) | Zoom | baseline |
|---|---|---|
| 4→2→1, same batch, 2 threads, seeds 0-2 | 1,087 / 944 / 957 s | 942 / 790 / 807 s |
| k=2 50 % batch x2 (the 1.12x recipe), seed 0, 2 threads | 1,484 s | 942 s |
| k=2 50 % batch x2, seeds 1-2, 1 thread | 1,275 / 1,274 s | 1,067 / 1,039 s (1-thread seeds 0-1) |
| k=2 33 % batch x2, 1 thread, seeds 0-1 | 967 / 1,068 s | 1,067 / 1,039 s |
| QK-norm + Zoom vs QK-norm, 1 thread | 1,082 / 1,111 s | 1,041 / 1,069 s |
| weather, Muon, k=4 merge 60 %, 1 thread | 1,061 / 1,087 s | 700 / 817 s |
| weather, AdamW, k=4 merge 60 %, 1 thread | 1,632 s | 1,513 s |

- **There is no wall-clock gain from Zoom on this CPU.** Without batch scaling,
  Zoom was 15-20 % slower at equal FLOPs on the LLM, 33-52 % slower on weather
  with Muon and 8 % slower with AdamW. Coarse steps cost more wall-clock per
  charged FLOP because the optimizer step is a fixed per-step cost (about
  0.3 s for Muon on the LLM). With batch x2 the result was mixed. The 33 %
  schedule ran about as fast as its baselines. The 50 % schedule behind the
  1.12x headline was slower: +19-23 % at 1 thread and +58 % for seed 0 at
  2 threads. The shared machine makes these timings noisy, but none of them
  shows a gain. The 0.2 claim that coarse steps "run faster per FLOP, so the
  wall-clock gain is larger than the FLOP gain" was wrong.
- GPU wall-clock has not been measured.

## 9. Low-memory training

Same model and data as section 2. Muon LR 1.6e-2, AdamW LR 1e-3, seed 0,
1 thread. Optimizer state and weights in MB (10^6 bytes).

| configuration | steps | eval windows | val loss | optimizer state | weights | train time |
|---|---|---|---|---|---|---|
| AdamW, 32-bit state | 100 | 1,024 | 4.4824 | 46.2 MB | 23.1 MB | 230 s |
| AdamW, 8-bit state | 100 | 1,024 | 4.4792 | 11.8 MB | 23.1 MB | 279 s |
| Muon, 32-bit state | 100 | 1,024 | 3.5719 | 27.3 MB | 23.1 MB | 266 s |
| Muon, 8-bit state | 100 | 1,024 | 3.5812 | 6.9 MB | 23.1 MB | 271 s |
| Muon, 8-bit + bf16 weights (stochastic rounding) | 100 | 1,024 | 3.5752 * | 6.9 MB | 11.5 MB | 718 s |
| Muon, 32-bit + activation recompute + grad accumulation x2 | 100 | 1,024 | 3.5719 (same to 1e-8) | 27.3 MB | 23.1 MB | 383 s |
| **full budget:** Muon, 32-bit (`save_muon_base`) | 450 | 2,048 | 2.7365 | 27.3 MB | 23.1 MB | 3,072 s |
| **full budget:** Muon, 8-bit + bf16 weights | 450 | 2,048 | 2.7280 * | 6.9 MB | 11.5 MB | 5,039 s |

\* **Needs a rerun.** These two bf16 runs were scored by the 0.2 evaluator,
which computed the whole loss in bf16. No checkpoint was saved, so they cannot
be re-scored in fp32. Measured on comparable checkpoints, that scoring error
is noise-like, not a fixed offset: about ±0.001 at 2,048 windows and ±0.004 at
1,024 windows (`results/llm_lowmem/bf16_eval_bias.json`).

- **8-bit state + bf16 weights at the full budget: 2.728 vs 2.737** for the
  fp32 baseline (3-seed mean; 2.7365 for the same seed). Within the scoring
  error and the seed spread, there is no measurable quality change.
- **Weights + optimizer state: 50.4 MB → 18.5 MB (2.7x smaller).** Not 3x, as
  0.2 said.
- 8-bit state is 3.9x smaller than 32-bit state for both optimizers.
- **bf16 with stochastic rounding is 1.6-2.7x slower on this CPU** (718 vs 271 s;
  5,039 vs 3,072 s), because CPU bf16 matmuls and the rounding writes are
  emulated. GPU speed is unmeasured.
- Recompute + accumulation gives the same loss (3.571913 both times; the two
  differ by 1.5e-8, so not bit for bit) at +44 % time. It also halves the
  micro-batch.
- 0.3 changed the 8-bit numerics (OPT-4: code 0 is now exact zero and the
  log map uses positive entries only) and the stochastic-rounding generator
  (OPT-1: counter-based, rank- and resume-safe). The whole low-memory set
  should be rerun with 0.3 before these rows are quoted as 0.3 results.
- 8-bit Muon momentum is also published prior art (Gupta et al. 2025, parity up
  to 2.7B), as are 8-bit Adam (Dettmers et al. 2022) and bf16 stochastic
  rounding (Zamirai et al. 2020; Ozkara et al. 2025).

## 10. Storing trained weights

The fully trained 5.8M model `save_muon_base` (fp32-trained, 2,048-window val
loss 2.73648), 1 seed. `results/llm_lowmem/save_muon_base_weights.json`.

| format | bytes/param | val loss | Δ |
|---|---|---|---|
| fp32 in RAM | 4.00 | 2.73648 | 0 |
| bf16 in RAM (round to nearest) | 2.00 | 2.73650 | +0.00002 |
| int8 weight-only, block 128 (`qweights`) | 1.02 | 2.73666 | +0.00018 |
| int4 weight-only, block 64 (`qweights`) | 0.53 | 2.76234 | +0.026 |
| `torch.save` fp32 file | 4.00 | | |
| WeightPack 32 (fp32 + zstd, exact) | 3.55 | 2.73648 | 0 |
| WeightPack 16 (bf16, byte shuffle + zstd) | 1.39 | 2.73650 | +0.00002 |
| WeightPack 8 | 0.94 | 2.73661 | +0.00013 |
| WeightPack 4 | 0.46 | 2.76241 | +0.026 |

- WeightPack 16 rounds fp32 weights to bf16, so it is **not lossless for an
  fp32 model**: its loss change equals bf16 rounding in RAM. It is bit-exact
  only for weights that are already bf16 (and 0.3 keeps fp16 tensors exact).
- On this small model int8 is free and int4 costs +0.026. On a pretrained 135M
  model 4-bit costs much more (section 14.2).
- 0.3 changed the WeightPack format (a checksummed header, absmax/127 int8
  scales, small tensors stored exactly at 8 and 4 bits). The 0.2 weights are
  gone, so these sizes were not re-measured with 0.3.

## 11. Storing data

Sizes in MB (10^6 bytes). The TinyStories and ERA5 statistics come from
`pack_stats.json` files written next to the prepared data
(`/home/user/data/tinystories/tok4k/`, `/home/user/data/era5_64x32/`); they are
not committed under `results/`.

| store | input | size | ratio | exact? |
|---|---|---|---|---|
| TokPack v2: rank remap + LEB128 + zstd-19 with a shared dictionary, 16K-token chunks | 116.77M training tokens as uint16 (233.5 MB) | **87.2 MB**, 5.97 bits/token | 2.68x | yes, bit-exact |
| same, validation split | 4.88M tokens (9.8 MB) | 4.0 MB, 6.52 bits/token | 2.45x | yes |
| GridPack: int16 temporal deltas, error ≤ 5 % of each channel's 6-hour change std, zstd | ERA5 2010-2018, 6 channels, float32 646.3 MB | **123.4 MB**, 6.11 bits/value | 5.24x | bounded error (z500: max 11.4 m²/s²) |

- TokPack streams at about 24M tokens/s on one CPU process (`pack_stats.json`).
- The v1 TokPack format (byte shuffle) compressed worse and is now read-only.
  Its size was never recorded in a committed file, so no number is given.
- float16 would overflow on mean sea-level pressure (in Pa), which is why
  GridPack uses scaled int16.
- 0.3's streaming pack-text reproduced these files byte for byte on a copy of
  the data (same tokens, payload and 21,800 dropped documents) and adds a
  document index of 0.76 MB.
- Deduplication on TinyStories: section 2.2. Near-duplicate detection was also
  checked on a crafted corpus during the 0.3 review (30 one-word near copies:
  all 30 dropped). That check is not a committed result.

## 12. Against published TinyStories models

### 12.1 Protocol

- **Stories:** the first 1,000 of the 4,194 TinyStories V1 validation stories
  that appear in neither TinyStories V1 train (which contains 30 % of the
  validation stories verbatim) nor llama2.c's training shards (68.6 %).
  756,822 bytes. `experiments/clean_val_indices.json`, pinned dataset revision.
- **Metric:** bits per byte (BPB) over the original UTF-8 bytes, so different
  tokenizers compare fairly. Context capped at 512 tokens.
- **Boundaries (new in 0.3):** each family is scored in its own training
  format, and the separator before a story is context, never scored. Bantam:
  `[eos] story [eos]`. llama2.c: `[BOS] story [BOS]`. TinyStories (roneneldan)
  models: `<|endoftext|>\n story \n<|endoftext|>`.
- **Compute:** training FLOPs as 6·N·D from each model's config and README
  (`experiments/published_compute.json`). Ranges where the step count was not
  published.
- Published models were trained to convergence over many epochs. This compares
  quality per unit of training compute, not final quality.

### 12.2 Results

Bantam's row is `save_muon_base_repro`: the recorded `save_muon_base` command
retrained with 0.3 on the same data stream (the 0.2 weights are gone).
Final val 2.7352 vs 2.7365 for the 0.2 run.

| model | params | training FLOPs | BPB 0.2 | **BPB 0.3** | change |
|---|---|---|---|---|---|
| **Bantam 5.8M (one CPU run)** | 5.77M | **6.4e13** | 1.0291 | **1.0288** | retrain noise |
| llama2.c stories260K | 0.26M | 1.0e16 | 1.0966 | 1.0966 | identical |
| TinyStories-1M | 3.75M | ~2.1e17 | 0.8729 | **0.8223** | -0.051 |
| TinyStories-3M | 8.28M | ~4.7e17 | 0.7189 | **0.6767** | -0.042 |
| TinyStories-8M | 19.7M | ~8.8e17 (to 2.0e18) | 0.6511 | **0.5808** | -0.070 |
| TinyStories-28M | 52.0M | ~1.9e18 (to 4.3e18) | 0.5855 | **0.5377** | -0.048 |
| llama2.c stories15M | 15.2M | 3.6e18 | 0.5768 | 0.5768 | identical |
| TinyStories-33M | 68.5M | ~3.9e18 | 0.5784 | **0.5416** | -0.037 |
| llama2.c stories42M | 41.7M | 5.4e18 | 0.5157 | 0.5157 | identical |
| llama2.c stories110M | 110M | 1.7e19 | 0.4798 | not re-scored | |

Plot: `results/plots/published_bpb_vs_compute.png`.

- **Bantam vs stories260K:** better BPB with **at least 157x less compute**.
  6.4e13 is an upper bound on the compute Bantam needs to match it (read off
  one completed run), so 157x is a lower bound on the ratio. It is mostly a
  model-size effect: stories260K has 22x fewer parameters (and a 512-token
  vocabulary), so it is capacity-limited and most of its compute could not
  improve it.
- **Bantam is behind every TinyStories model.** The gap to TinyStories-1M is
  0.207 BPB; TinyStories-1M used about 3,350x more compute. No run reaches any
  model of 3.7M parameters or more.
- stories110M was not re-scored (a 438 MB download over the rerun's budget).
  Its format and code path did not change, and stories260K/15M/42M reproduce
  their 0.2 scores bit for bit.

### 12.3 Why the TinyStories numbers moved (EXP-1)

0.2 fed every model `[eos] story [eos]` and scored all of it. The TinyStories
models never saw that separator: they were trained on the V1 text layout with
`<|endoftext|>\n` between stories. Under the 0.2 convention their end-of-story
token cost 11.5-12.7 nats per story; in their own format it costs 0.4-0.7. The
extra `\n` in the prefix also helps the story tokens a little (0.014-0.048 BPB
with the end token excluded). In total the 0.2 comparison overstated their BPB
by 0.037-0.070, which is 19-37 nats per story. That made Bantam look closer
than it is. Re-scoring with the legacy format reproduces the 0.2 numbers bit
for bit, so the whole shift comes from the convention, not from code or data
drift. Consequences:
- Bantam's gap to TinyStories-1M grew from 0.156 to 0.207 BPB.
- In the 0.2 table TinyStories-33M (0.578) sat level with stories15M (0.577).
  Now TS-28M (0.538) and TS-33M (0.542) are clearly ahead of stories15M
  (0.577), and TS-8M (0.581) is level with it.
- The 0.2 plot drew mid-schedule points of the unfinished 10x run; they are gone.
- Scoring without any end-of-story token (`--no_end`) changes no conclusion
  (`compare_clean1000_noend.json`).

## 13. Weather (ERA5)

### 13.1 Main matrix

5.5e13 model FLOPs, batch 16, 1 thread per run. RMSE is latitude-weighted, in
physical units (z500 in m²/s², temperatures in K), averaged over 96 rollouts
from 2018 (section 2.6).

| run | optimizer, LR | schedule | steps | epochs | one-step MSE | n | z500 24 h | z500 72 h | t850 24 h | t2m 24 h |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | Muon 1.6e-2 | k=1 | 567 | 0.78 | **0.1240 ± 0.0006** | 2 | 217.4 / 217.6 | 538.7 / 551.2 | 1.16 / 1.15 | 1.03 / 1.03 |
| Zoom merge (gated) | Muon 1.6e-2 | `4:0.6,1:0.4` | 2,128 | 2.91 | 0.1588 ± 0.0005 | 2 | 252.7 / 257.7 | 600.5 / 623.2 | 1.31 / 1.31 | 1.18 / 1.18 |
| Zoom merge, shorter | Muon 1.6e-2 | `4:0.4,1:0.6` | 1,607 | 2.20 | 0.1289 | 1 | 219.1 | 548.1 | 1.17 | 1.07 |
| Zoom, mean merge | Muon 1.6e-2 | `4:0.6,1:0.4 --gated 0` | 2,128 | 2.91 | 0.1276 | 1 | 206.1 | 526.6 | 1.17 | 1.07 |
| Zoom 16→4→1 | Muon 1.6e-2 | `16:0.3,4:0.3,1:0.4` | 5,081 | 6.96 | 0.1799 | 1 | 275.4 | 633.7 | 1.42 | 1.25 |
| **low-res warm-up** | Muon 1.6e-2 | `4:0.6,1:0.4 --coarse_mode lowres` | 2,128 | 2.91 | **0.1119** | 1 | **194.3** | **511.2** | 1.10 | 1.01 |
| baseline, 2x budget | Muon 1.6e-2 | k=1, 1.1e14 | 1,133 | 1.55 | 0.0938 | 1 | 171.8 | 461.3 | 0.99 | 0.92 |
| baseline | AdamW 2e-3 | k=1 | 567 | 0.78 | 0.2292 | 1 | 346.2 | 737.2 | 1.60 | 1.32 |
| Zoom merge | AdamW 2e-3 | `4:0.6,1:0.4` | 2,128 | 2.91 | 0.1800 | 1 | 281.8 | 646.8 | 1.42 | 1.23 |
| low-res warm-up | AdamW 2e-3 | `4:0.6,1:0.4 --coarse_mode lowres` | 2,128 | 2.91 | 0.1582 | 1 | 256.6 | 601.1 | 1.33 | 1.17 |
| persistence | | | | | | | 569.1 | 906.6 | 2.70 | 1.72 |

"k=4" on the weather model merges each 2 x 2 block of tokens (512 → 128
tokens). Low-res warm-up instead averages the fields to a 2x coarser grid in
each direction (also 128 tokens) and trains on that coarser problem.

### 13.2 Zoom hurts weather under Muon (negative result)

- With Muon, every gated Zoom merge was worse than the baseline: 0.159 for 60 %
  of compute at k=4 (+28 %), 0.129 for 40 % (+4 %), 0.180 for 16→4→1 (+45 %).
- The plain mean merge (0.128) did better than the learned gate (0.159).
- Zoom only helped with AdamW (0.180 vs 0.229), and every AdamW run lost
  clearly to the Muon baseline. That pattern fits Zoom acting as a stabilizer
  for a weak AdamW setup rather than a compute saver.
- Zoom on weather does not pay off. Low-res warm-up is the better choice.

### 13.3 Low-resolution warm-up helps (prior art)

Training on a 4x coarser grid for the first 60 % of compute gave 0.112 vs
0.124 one-step MSE (about 10 % lower; 1 seed) and z500 RMSE of 194 vs 217 at
24 h and 511 vs 539 at 72 h (seed 0 baseline). With AdamW it also helped
(0.158 vs 0.229). This is progressive resizing, an established technique
(e.g. Composer, EfficientNetV2, and low-to-high-resolution transfer in weather
models such as FengWu-GHR). Bantam implements it; it did not invent it.

### 13.4 Uncharged optimizer compute in the weather comparison (WX-2)

Muon's Newton-Schulz step costs about 3.8e9 FLOPs per step for this model and
is not in the budget. The baseline takes 567 steps (2.2e12 uncharged, 3.9 % of
the budget). The Zoom and low-res runs take 2,128 steps (8.1e12, 14.7 %).
So under Muon the coarse runs got about 10 % more total compute than the
baseline.
- Zoom lost anyway, so this makes the negative result stronger.
- For low-res warm-up, interpolating the baseline in log-FLOPs between the 1x
  (0.124) and 2x (0.094) runs puts a baseline with 10 % more compute at about
  0.120. Low-res (0.112) still wins, by about 6-7 % rather than 10 %.
- With AdamW the optimizer cost is negligible.
- In 0.2 the low-res coarse steps were also overcharged by about 2 % for patch
  embedding and decoding (WX-11), which slightly disfavoured low-res.

### 13.5 The test year was also used for model selection (WX-3)

The committed weather runs have no separate validation year. The learning
rates (section 4.2) and the ablations were chosen on 2018, the same year the
table reports. The RMSE values are therefore not clean held-out numbers. The
effect is probably small (it mostly concerns the LR choice), but it is real.
0.3 adds `--val_year Y` to `bantam weather` (train on years before Y, choose
on Y, report the test year only at the end) and `prepare-era5 --val_year`.
The committed runs predate it and have not been rerun.

### 13.6 Other caveats and context

- The Zoom and low-res runs saw the training data 2.9 times; the baseline saw
  it 0.78 times. In 0.2, runs longer than one epoch replayed the first epoch's
  order (fixed in 0.3, WX-14).
- The model beats persistence clearly at every lead (z500 at 24 h: 217 vs 569).
- For orientation only (different test period and initial conditions):
  WeatherBench (Rasp et al. 2020) lists 3-day z500 RMSE at 5.625° of 936 for
  persistence and 489 for IFS T42; the CMIP-pretrained ResNet of Rasp and
  Thuerey (2021) reaches 268.
  Bantam's 1.3M-parameter ViT is at 511-551 at the 5.5e13 budget (461 at 2x).
  A proper comparison needs WeatherBench's test years, metrics and
  climatology. 0.3 logs latitude-weighted ACC against a train-years
  climatology for new runs; the committed runs predate it.

## 14. Fine-tuning SmolLM2-135M

All fine-tuning results so far are on `HuggingFaceTB/SmolLM2-135M` with
Dolly-15k on the CPU, loss on response tokens only. **They were produced by
the 0.2 code, before packed examples got block-diagonal masks** (added in
commit 60759d3, the first change after 0.2): in a packed window, each example
could attend to the examples packed before it. All formats in a table share
that effect, so comparisons within a table hold, but the absolute losses are
not clean. The race in
section 15 uses the 0.3 masks and scores examples one at a time.

### 14.1 LoRA and QLoRA (0.2 runs, 1 seed)

Settings: LoRA rank 16, alpha 32, no dropout, on all 7 projections (210
modules, 4.88M trainable parameters); 40 steps of 2 x 2 accumulation at seq
256 (160 sequences); LR 2e-4, WSD schedule; 355 Dolly examples. Held-out loss
on 5 % of the examples (8 packed windows, about 17 examples).

**Optimizer correction:** these runs passed `--optim muon`, but `--muon_lora`
was off in 0.2, so no parameter went to Muon. They trained with **8-bit
AdamW**. 0.2's "Muon with 8-bit state" label was wrong. In 0.3,
`--optim muon` with LoRA puts the LoRA matrices on Muon by default
(`--muon_lora -1` = auto).

| frozen base | held-out loss before → after | base RAM | with fp32 adapters | adapter file | train time |
|---|---|---|---|---|---|
| fp32 (LoRA) | 2.871 → 2.792 | 538.1 MB | 557.6 MB | 19.7 MB | 353 s |
| int4 linears **and** int4 embeddings (QLoRA) | 3.389 → 3.178 | 86.6 MB | 106.2 MB | 19.7 MB | 459 s |

- QLoRA training recovered 0.21 of the 0.52 the int4 base lost, but **the
  QLoRA model (3.178) is still worse than the untouched float base (2.871)**.
- That run predates `--embed_quant`: it quantized the embeddings to int4 too,
  the worst format in section 14.2. Today `--quant 4` and `--quant nf4` keep
  int8 embeddings by default. The recommended NF4 + int8-embedding QLoRA has
  not been trained yet; the race (section 15) trains it.

### 14.2 What quantizing the frozen base costs (before any training)

`--eval_only`, the same 355 examples, held-out set = 25 % = **88 examples**
(43 packed windows), deterministic. Base RAM excludes the 19.5 MB of fp32 LoRA
adapters that every row carried; the last column includes them.

| base format | eval loss | Δ | base RAM | with adapters |
|---|---|---|---|---|
| fp32 (no quantization) | 2.7018 | 0 | 538.1 MB | 557.6 MB |
| int8 linears + int8 embeddings | 2.7043 | +0.003 | 180.9 MB | 200.4 MB |
| NF4 linears, float embeddings | 2.8752 | +0.173 | 169.8 MB | 189.3 MB |
| **NF4 linears + int8 embeddings (default for `--quant nf4`)** | 2.8777 | +0.176 | 120.4 MB | 140.0 MB |
| linear int4 + int8 embeddings | 3.0353 | +0.333 | 120.4 MB | 140.0 MB |
| NF4 everything | 3.0422 | +0.340 | 86.6 MB | 106.2 MB |
| linear int4 everything | 3.2215 | +0.520 | 86.6 MB | 106.2 MB |

- **int8 is nearly free** (+0.003) for a 135M model. **4-bit is not**: +0.18
  with NF4 and int8 embeddings, +0.52 with plain int4 everywhere. NF4 beats
  linear int4 by 0.16-0.18.
- So for a small base, `--quant 8` is the better choice; `--quant` defaults to
  no quantization. The QLoRA paper finds NF4 close to lossless at 7B and
  above; that was not measurable here.
- 0.2 stored a tied output head as a second quantized copy of the embedding
  table (QNT-6). 0.3 stores it once, so these RAM figures overstate 0.3 for the
  rows with quantized embeddings. An end-to-end check of 0.3 measured 105 MB
  (NF4) and 157 MB (int8) under the same convention as the 557.6 MB float
  figure. That check is not a committed result file.
- `results/finetune/quantization_quality.json` still describes its eval set as
  "100 held-out examples"; the logs show 88.
- Pure-PyTorch NF4 QLoRA is not unique: torchtune does the same through
  torchao, and bitsandbytes is the reference implementation (Dettmers et al.
  2023).

## 15. Fine-tuning race: Hugging Face Trainer + PEFT vs Bantam

**Question:** on the same model, data, adapters and budget, how do Bantam's
fine-tuning loop and recipe compare with the standard Hugging Face stack in
speed, memory and quality, on this CPU?

### 15.1 Protocol (`experiments/race_finetune.py`)

Shared by every arm:
- `HuggingFaceTB/SmolLM2-135M` in fp32 on the CPU, SDPA attention, `--threads`
  the same everywhere.
- Dolly-15k from the local cache. Row indices shuffled with `Random(1234)`:
  the first 800 train, the next 200 are the judge set. Rows with no response
  token within 512 tokens are skipped (33 rows). Prompt = instruction, a blank
  line, then the context.
- ChatML template, loss on response tokens only.
- LoRA r=16, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down projections
  (210 modules, 4,884,480 trainable parameters) in every arm.
- LR 2e-4, 3 % warm-up then cosine to 10 % of the peak, AdamW betas
  (0.9, 0.95), no weight decay, gradient clipping 1.0, seq 512,
  micro-batch 2 x accumulation 2, 1 epoch, seed 1234.

Arms (each in its own process):

| arm | what it runs |
|---|---|
| `base` | no training; the judge's reference |
| `hf_peft` | transformers Trainer + peft `LoraConfig`: one example per row, dynamic padding, torch AdamW (fp32 state), `cosine_with_min_lr` |
| `bantam_adamw` | `bantam finetune --optim adamw --state_bits 32 --pack bfd`: the same torch AdamW as `hf_peft`, so the difference is the training loop (packing, chunked cross-entropy) |
| `bantam` | Bantam's recipe: `--optim muon` (Muon on the LoRA matrices), 8-bit optimizer state, best-fit-decreasing packing with block-diagonal masks, chunked cross-entropy (chunk 2048), flush-to-zero |
| `bantam_nf4` | the `bantam` arm on an NF4 base with int8 embeddings (`--quant nf4 --low_ram 1`); judged on its NF4 base and also on the float base |

Measurements:
- **Speed:** real (non-pad) trained tokens per second of training-only wall
  time.
- **Memory:** peak RSS of the arm's process tree, sampled every 0.2 s.
- **Quality (the judge):** a separate process per arm. Pooled per-token
  cross-entropy over the response tokens of the 200 held-out examples, one
  example at a time, no packing, fp32 logits from the model's own forward.
  Bootstrap 95 % confidence intervals and paired differences against
  `hf_peft`. The `bantam` and `bantam_adamw` arms are judged through their
  PEFT export loaded with `peft`; `bantam_nf4` through
  `qweights.load_quantized` plus its adapter, and also on the float base.
  Plus one greedy answer to each of three fixed prompts.

Fairness checks built into the harness (the values in brackets are from the
32-example smoke run; the full run's values are in 15.2):
- Token parity: both sides must tokenize the training examples identically
  (smoke: 32 of 32, 0 mismatches).
- The same LoRA modules and parameter count in every arm.
- Bantam's own final eval must match the judge on the same examples (smoke:
  to about 1e-7), and Bantam's step-0 eval must match the `base` judge
  (smoke: to 3e-8).

Known asymmetries (by design, reported rather than hidden):
- Packing puts about 1,024 real tokens in each Bantam micro-batch, so Bantam
  takes about 3.4x fewer, larger optimizer steps than HF at the same LR
  (58 vs 200).
- Bantam's "1 epoch" drops the last packed windows that do not fill a whole
  optimizer step (3 of 235 windows: 61,257 of 61,972 supervised targets,
  about 1.2 %). HF trains every example. Speed counts only tokens actually
  trained.
- Importing `bantamtrain.finetune` turns on flush-to-zero process-wide. The HF
  arm never imports it and runs with flush-to-zero off (opt in with
  `--hf_flush_denormal 1`).
- The `bantam` arm uses `--optim muon` and a cosine schedule; the CLI
  defaults are `--optim adamw` and `--sched wsd`.
- At this shape chunked cross-entropy saves nothing (a 2,048-token chunk is
  larger than a micro-batch).

Command (resumable; `--force <arm>|all` to redo):

```bash
python3 experiments/race_finetune.py --threads 4
```

A 32-example smoke run (`results/race/smoke/`) only validates the harness; its
numbers are not a comparison.

### 15.2 Results

Follow-up arms (same protocol, overrides in brackets): `bantam_adamw_steps`,
`bantam_muon_steps` and `bantam_nf4_adamw_steps` [micro-batch 1 x accumulation 1,
so one packed window (about 3.4 examples) per optimizer step: 235 steps, close to
`hf_peft`'s 200 steps of 4 examples], `bantam_muon_steps_lr1e-3` and
`bantam_adamw_steps_lr1e-3` [the same, LR 1e-3].

Full run, 4 threads, nothing else running on the machine (`results/race/summary.json`, `results/race_hf_ftz/` for the flush-to-zero control):

| arm | optimizer steps | train min | real tok/s | peak RSS GB | judge loss | vs `hf_peft` (paired 95 % CI) |
|---|---:|---:|---:|---:|---:|---|
| `base` (no training) | - | - | - | - | 2.645 | +0.385 (+0.326 to +0.466) |
| `hf_peft` | 200 | 13.2 | 151 | 4.17 | 2.260 | reference |
| `bantam_adamw` (2 x 2 windows per step) | 58 | 9.9 | 199 | 5.16 | 2.316 | +0.056 (+0.046 to +0.068) |
| `bantam` (Muon LoRA, 2 x 2) | 58 | 9.3 | 213 | 5.28 | 2.535 | +0.275 (+0.231 to +0.338) |
| `bantam_nf4` (Muon LoRA, NF4 base, 2 x 2) | 58 | 9.2 | 216 | 4.61 | 2.694 | +0.434 (+0.390 to +0.492) |
| `bantam_adamw_steps` (1 window per step) | 235 | 9.7 | 206 | 2.57 | **2.259** | **-0.001 (-0.005 to +0.003)** |
| `bantam_muon_steps` | 235 | 9.3 | 216 | 2.56 | 2.303 | +0.044 (+0.035 to +0.054) |
| `bantam_muon_steps_lr1e-3` | 235 | 9.0 | 223 | 2.57 | 2.241 | -0.019 (-0.027 to -0.012) |
| `bantam_adamw_steps_lr1e-3` | 235 | 8.9 | 226 | 2.59 | 2.228 | -0.031 (-0.040 to -0.025) |
| `bantam_nf4_adamw_steps` (NF4 base, AdamW) | 235 | 9.4 | 212 | 2.12 | 2.368 | +0.109 (+0.100 to +0.119) |

What it shows:
- **Same optimizer, comparable steps** (`bantam_adamw_steps` vs `hf_peft`): the same
  held-out loss (2.259 vs 2.260; the paired interval straddles 0), **1.36x faster**
  (9.7 vs 13.2 min of training; 206 vs 151 real tokens/s) and **38 % lower peak
  RSS** (2.57 vs 4.17 GB). The speed comes from packing (no padding tokens are
  computed) and Bantam's loop. Flush-to-zero is not the reason: `hf_peft` with
  flush-to-zero on ran at 147 tokens/s (`results/race_hf_ftz/`).
- **Packing changes the optimizer.** At the protocol's 2 x 2 windows per step a
  packed window holds about 3.4 examples, so Bantam made 58 optimizer steps instead
  of 200 and lost 0.056; its two full 512-token rows per micro-batch also peaked at
  5.16 GB, more than `hf_peft`. Match the number of optimizer steps (here one window
  per step) or raise the learning rate.
- **Muon on LoRA did not beat AdamW.** At LR 2e-4: Muon 2.303 vs AdamW 2.259. At
  LR 1e-3: Muon 2.241 vs AdamW 2.228. The best loss (2.228) came from the higher
  learning rate, not from Muon, and `hf_peft` was not run at 1e-3, so that gap is
  not a Bantam advantage.
- **QLoRA (NF4 base)** cut peak RSS to 2.12 GB (49 % below `hf_peft`) and cost
  0.109 in held-out loss; the same adapters on the float base score 2.302. 4-bit
  costs real quality on a 135M model.

Limits: one run per arm (two AdamW runs that differ only in LR took 9.7 and 8.9
min, so wall time varies by about 9 %), fp32 on a 4-core CPU, a 135M model. GPU
tools with fused kernels (FlashAttention varlen packing, Liger, Unsloth) are not in
this race. Fairness checks of the full run: token parity 800 of 800; the same 210
LoRA modules (4,884,480 trainable parameters) in every arm; the one-window-per-step
arms train all 61,972 supervised tokens, like `hf_peft` (the 2 x 2 arms 61,257);
Bantam's own final eval matches the judge to about 1e-7 and its step-0 eval matches
the `base` judge to about 1e-8. Full table, commands and greedy answers:
`results/race/summary.md`.

## 16. What changed since 0.2

Every claim below was wrong, stale or missing a caveat in the 0.2 docs, and is
corrected in this report. Audit finding ids are in brackets.

**Names and commands**
- The package is `bantamtrain` and the command is `bantam <command>` (or
  `python -m bantamtrain <command>`). Commands that use the pre-rename package
  name no longer work [CLM-3, EXPV-2, FT-19, PKG-11, MEM-7].

**LLM pretraining**
- Muon vs AdamW: the AdamW part of every Muon run uses a 16x higher LR than
  AdamW-only runs, so the comparison changes two things at once. The claim
  that both parts can share one LR is not supported. The two compute figures
  ("~1/3" and "~40 %") are now each defined: a separate 1/3-budget run, and a
  mid-curve crossing at 38-39 % (section 3) [CLM-5].
- Zoom: the 0.1 design (4→2→1) lost on all seeds; the measured recipe is k=2
  with a larger batch. The upgrade ablations are now reported, including the
  two that were worse than baseline (WSD, ReLU²) and the 2-seed QK-norm/Zoom
  factorial [CLM-9].
- Compute-to-target 1.12x: now computed by `summarize.py` with a stated method
  (log-FLOP interpolation, seed-averaged, completed runs only) and an
  uncertainty of about 1.11-1.14x; the baseline curve above 1x is single-seed
  [CLM-14].
- Muon's Newton-Schulz FLOPs (20 % of model FLOPs at this size) are not in the
  budget. 0.3 logs them; the effect on each comparison is stated (sections 2.4
  and 13.4).
- Not every validation loss is "on 2,048 held-out sequences": each table now
  states its eval size, and curves on 256 windows read 0.27-0.30 nats lower
  [CLM-15].
- The training stream excluded validation stories, but the raw 0.2 `train.bin`
  did not; 0.3 `prepare-text` now drops them [DATA-1].
- No wall-clock gain from Zoom on this CPU. The claim that coarse steps run
  faster per FLOP was wrong [CLM-12, WX-8, ZG-2].

**Low-memory training and storage**
- Weights + state shrink 2.7x, not 3x [CLM-17, OPT-12, MEM-14].
- The bf16 rows were scored with bf16 arithmetic; they carry a ±0.001 / ±0.004
  uncertainty and are marked for rerun.
- bf16 with stochastic rounding is 1.6-2.7x slower on this CPU [CLM-12].
- WeightPack 16 is bf16 rounding (+0.00002 from fp32), not "lossless" or
  "bit-exact"; it is exact only for bf16 weights [DOC-1].
- Unsourced runtime numbers were removed: the flush-to-zero timing, the
  thread-scaling GFLOP/s, the DDP training claim (no log had world_size > 1;
  0.3 tests DDP with 2-process CPU tests instead), the "verified with injected
  edits" MinHash claim and the v1 TokPack size. The remaining data-store
  numbers now cite their `pack_stats.json` [CLM-21].
- The TokPack description is the v2 format (rank remap + LEB128 + zstd with a
  dictionary) and the weather ViT has 1.26M parameters [CLM-20].

**Published comparison**
- TinyStories models are now scored in their own boundary format. They gain
  0.037-0.070 BPB, Bantam's gap to TS-1M grows from 0.156 to 0.207, and the
  TS-33M vs stories15M ordering changed [EXP-1].
- "157x less compute than stories260K" is an upper bound and mostly a
  model-size effect [CLM-6].
- The scores use 1,000 of the 4,194 clean stories, not 4,194; the validation
  file is now downloaded at a pinned revision [EXP-11].

**Weather**
- 0.2 said "pending". The full matrix is now reported, including the negative
  Zoom result under Muon and the low-res warm-up result [CLM-7].
- The test year was used for LR selection; there was no validation year [WX-3].
- Coarse arms got about 10 % more uncharged optimizer compute under Muon [WX-2].
- `results/SUMMARY.md` was stale (missing the best weather sweep runs and the
  weather section) [WX-16]. It must be regenerated whenever runs are added.

**Fine-tuning**
- The "Muon with 8-bit state" LoRA runs used 8-bit AdamW [FT-14, CLM-4].
- The quantization table's eval set is 88 examples, not 100. RAM is now given
  both without and with adapters, consistently [CLM-16, FT-20].
- The QLoRA run (3.39 → 3.18) ends worse than the untouched float base
  (2.87), and used int4 embeddings, which the current defaults no longer do
  [PKG-X1].
- All committed fine-tuning results predate block-diagonal masks for packed
  examples; the race uses them.

**5B scoreboard** (details in [`docs/SCOREBOARD_5B.md`](docs/SCOREBOARD_5B.md))
- GPU memory is in GiB, file sizes in GB. With a 0.85 safety margin the
  standard recipe does not fit, and the Muon row's speedup over the first row
  that fits is 1.87x, not 2.56x [MEM-14].
- FP8 has no code path in Bantam, and `bantam pretrain` has no chunked
  cross-entropy flag; the scoreboard no longer claims the trainer "exposes
  every row" [GPU-16, MEM-8].

Findings that concern only other files (README, METHOD, the packaging README)
belong to those files: CLM-19 (0.3 wires the batch finder into
`bantam pretrain --auto_batch`, GPU only), DATA-14, ZG-11 and PKG-13.

## 17. Open questions and what has not been measured

**Not measured at all**
- Anything on a GPU: speed, memory, MFU, `--compile`, bf16 autocast on CUDA,
  8-bit state on CUDA. The 5B scoreboard is arithmetic.
- Any model larger than 5.8M parameters for pretraining claims, or other than
  SmolLM2-135M for fine-tuning.
- FP8 training (no code path exists).
- Multi-GPU training beyond 2-process CPU tests of DDP.

**Open questions**
- Does Zoom's gain survive at 124M+ parameters, against sequence-length
  warm-up and Shao-style averaging, on a QK-norm baseline, with 3+ seeds?
  Prior art suggests the gain shrinks with scale.
- How much of Muon's lead remains with a separately tuned LR for its AdamW
  part, and against a more carefully tuned AdamW?
- Single-seed rows: every upgrade except QK-norm and QK-norm + Zoom; every
  Zoom variant except the two headline schedules and the 4→2→1 design (3
  seeds); all AdamW runs; every weather run except the Muon baseline and the
  Muon k=4 merge (2 seeds each); the long-budget baselines; all fine-tuning.
- Rerun with 0.3: the low-memory set (bf16 scoring, OPT-4 8-bit map, new
  stochastic rounding), Polar Express (OPT-8), the weather LR sweeps on a
  separate `--val_year`, and the four older upgrade runs.
- A baseline curve at larger budgets with QK-norm on, to express the stacked
  Zoom gain in FLOPs.
- Weather in WeatherBench terms (test years, ACC, climatology), and an
  AdamW sweep that goes above 4e-3.
- The NF4 + int8-embedding QLoRA quality after training (the race measures it
  on CPU).

## 18. Reproducing

All LLM commands assume the prepared data in `$D`
(`bantam prepare-text` then `bantam pack-text`). The committed runs used data
prepared by 0.2. 0.3 `prepare-text` drops validation copies before it trains
the BPE, so its tokenizer differs slightly. Pass `--drop_val_overlap 0` to
`prepare-text` to rebuild the 0.2 files byte for byte; `pack-text` drops the
copies from the training stream either way. `python3 llm/train.py` is the
same trainer as `bantam pretrain`. Queue files run with
`python3 experiments/run_queue.py <queue> --workers 2 --threads 2`, which skips
finished runs and resumes unfinished ones from their checkpoint when one exists.

| results | queue file |
|---|---|
| LLM LR sweeps | `experiments/queues/llm_sweep.txt`, `llm_sweep2.txt`, `sweep3.txt` |
| main matrix and Zoom ablations | `llm_main.txt`, `followup.txt`, `zoom_confirm.txt` |
| long-budget baselines | `compute_to_target.txt` |
| upgrade ablations | `upgrades.txt` |
| low-memory | `lowmem.txt`, `lowmem2.txt` |
| published comparison | `published_cmp.txt` plus `experiments/compare_published.py` |
| weather sweeps and matrix | `weather_sweep.txt`, `weather_sweep4.txt`, `sweep3.txt`, `weather_main.txt` |

The two bundle lines in `upgrades.txt` (`up_bundle_base_s0`,
`up_bundle_zoom_s0`) now also pass `--sched wsd --mlp relu2 --ns polar`. The
committed bundle runs used only `--softcap 15 --zero_init_proj 1` (see the
`args` in their JSON logs), which is what section 6 reports. Running the queue
as it stands produces a different bundle.

Most runs of the 2-seed QK-norm/Zoom factorial (section 6.1) have no queue
file. Seed 1 is shown; seed 0 is the same with `--seed 0` and `_s0` names:

```bash
D=/path/to/tok4k
bantam pretrain --data $D --budget 6.8e13 --zoom 1 --optim muon --lr 1.6e-2 --threads 1 --seed 1 --name up_base_s1
bantam pretrain --data $D --budget 6.8e13 --zoom 1 --optim muon --lr 1.6e-2 --qk_norm 1 --threads 1 --seed 1 --name up_qknorm_s1
bantam pretrain --data $D --budget 6.8e13 --zoom 2:0.33,1:0.67 --batch_scale 2 --optim muon --lr 1.6e-2 --threads 1 --seed 1 --name up_zoom_2-33_bs2_s1
bantam pretrain --data $D --budget 6.8e13 --zoom 2:0.33,1:0.67 --batch_scale 2 --optim muon --lr 1.6e-2 --qk_norm 1 --threads 1 --seed 1 --name up_qknorm_zoom_s1
```

The published comparison (section 12), from the repository root:

```bash
python3 experiments/compare_published.py --stories 1000 --ctx_cap 512 --threads 1 \
    --hf roneneldan/TinyStories-1M roneneldan/TinyStories-3M roneneldan/TinyStories-8M \
         roneneldan/TinyStories-28M roneneldan/TinyStories-33M \
    --out results/published/compare_clean1000.json
python3 experiments/plot_published.py results/published/compare_clean1000.json
```

Add `--ours <run.json>:<weights.pt>` for a Bantam checkpoint and
`--l2c <checkpoint.pt>:<tokenizer.model>` for llama2.c models. The clean story
set is the default.

Then regenerate the tables and plots:

```bash
python3 experiments/summarize.py --plots
```
