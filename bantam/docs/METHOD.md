# How Bantam works

This file explains each technique in Bantam 0.3.0: what the code does, why,
and what it costs. It is written for an engineer who wants to read or change
the code. The literature behind each idea is in [RESEARCH.md](../RESEARCH.md).
The full result tables are in [RESULTS.md](../RESULTS.md) and
[results/SUMMARY.md](../results/SUMMARY.md). How Bantam compares with other
tools is in [COMPARISON.md](COMPARISON.md).

**Conditions for every number in this file.** All of them were measured on one
CPU-only sandbox: 4 cores, 16 GB RAM, PyTorch 2.15 for CPU. Nothing has been
run on a GPU. The language-model numbers come from TinyStories with a 4,096-token
BPE vocabulary (about 117M training tokens after deduplication), a 5.8M-parameter
model (d=256, 6 layers, 4 heads, context 256) and a fixed budget of 6.8e13 model
FLOPs. Loss is validation cross-entropy in nats per token (lower is better).
"n" is the number of seeds; "± x" is the sample standard deviation over seeds.
A result with n=1 is a single run and can be off by about one baseline standard
deviation (0.012).

Most of the ideas here are other people's. Bantam implements them in one small
pure-PyTorch package. Zoom training is the one piece that is Bantam's own
design, and it is a variant of earlier work (patch-level training, Hourglass,
sequence-length warm-up; see §1.10).

Contents

1. [Zoom training](#1-zoom-training)
2. [Muon](#2-muon)
3. [8-bit optimizer state](#3-8-bit-optimizer-state)
4. [bf16 weights with stochastic rounding](#4-bf16-weights-with-stochastic-rounding)
5. [Quantized frozen weights (int8, int4, NF4)](#5-quantized-frozen-weights-int8-int4-nf4)
6. [LoRA, DoRA, LoRA+, rsLoRA](#6-lora-dora-lora-rslora)
7. [Chunked cross-entropy](#7-chunked-cross-entropy)
8. [Packed fine-tuning data](#8-packed-fine-tuning-data)
9. [Data storage: TokPack, dedup, GridPack](#9-data-storage-tokpack-dedup-gridpack)
10. [Weather model: ZoomViT and low-resolution warm-up](#10-weather-model-zoomvit-and-low-resolution-warm-up)
11. [Model options for the small GPT](#11-model-options-for-the-small-gpt)
12. [Checkpoints, resume and saved models](#12-checkpoints-resume-and-saved-models)
13. [Runtime details](#13-runtime-details)
14. [What has not been measured](#14-what-has-not-been-measured)
15. [Where the code lives](#15-where-the-code-lives)

---

## 1. Zoom training

### 1.1 The idea

The expensive part of a transformer is the stack of blocks (attention and
MLP). Its cost grows with the number of positions it processes. Zoom training
makes early training cheaper by letting the blocks run on fewer, coarser
positions:

- Early in training, every group of `k` neighbouring tokens is merged into one
  position. The blocks run on `T/k` positions instead of `T`.
- The output is split back into `k` per-token states, so every token still gets
  its own prediction and its own loss.
- Later, `k` drops to 1. The model ends as an ordinary transformer.

Every input token and every target token is used in every stage. Only the
internal resolution of the blocks changes.

This is not a new idea. The closest prior work is *patch-level training*
(Shao, Meng and Zhou, ICLR 2025): average K token embeddings into one patch,
train on the shorter sequence, then switch to tokens. Zoom differs in three
details (a learned merge, a per-token split at the output, and the schedule
that worked here); see §1.10 and [RESEARCH.md §2.3](../RESEARCH.md).

### 1.2 The data path

```
token ids (B, T)
   |  embedding (tied with the output head)
   v
x  (B, T, d)  --view-->  (B, T/k, k, d)
   |  SlotMerge[k]          z_p = merge(x_p1 ... x_pk)
   v
z  (B, T/k, d)             RoPE angle of group p = p*k + (k-1)/2
   |  L causal transformer blocks over the T/k groups      <- the part that gets cheaper
   v
h  (B, T/k, d)
   |  SlotUnmerge[k]        y_pj = unmerge_j(h_p),  j = 1..k
   v
y  (B, T/k, k, d) --> tied head --> logits (B, T, V) --> cross-entropy against w[k : T+k]
```

At `k = 1` the merge and unmerge are skipped and the model is a plain causal
language model (`ZoomGPT.forward(idx, targets, k=1)`).

### 1.3 Three merge modes

`--merge_mode` (or the older `--gated 1/0`) picks how a group is merged and
split. The code is `SlotMerge` and `SlotUnmerge` in `bantamtrain/zoom.py`.

| mode | merge | split | trainable parameters per zoom level | notes |
|---|---|---|---|---|
| `gate` (default) | `z = Σ_j g_j ⊙ x_j`, one vector `g_j ∈ R^d` per slot, init `1/k` | `y_j = h ⊙ u_j`, init `u_j = 1` | `2·k·d` (1,024 at k=2, d=256) | The per-slot gates let the merged vector keep which slot a token was in ("the cat sat" vs "sat the cat"). |
| `mean` (`--gated 0`) | fixed `1/k` average | fixed copy, `y_j = h` | 0 | All `k` slots of a group get the same vector, so the same logits. The loss is then the "bag of next tokens" loss of patch-level training. This is the closest Bantam gets to Shao et al. |
| `linear` | `z = Σ_j W_j x_j`, `W_j ∈ R^{d×d}`, init `I/k` | `y_j = U_j h`, init `U_j = I` | `2·k·d²` | A learned patch embedding and un-embedding. Its FLOPs are charged since 0.3. No committed run uses it. |

`--out_mode bag` is a fourth variant: it skips the split and scores one
distribution per group against all `k` next tokens (Shao's loss). It works
with any merge mode; the one committed bag run used the gated merge.

Measured (Muon, 4:0.35,2:0.25,1:0.4 schedule, n=1 except where stated):
gate 2.874 ± 0.002 (n=3), mean 2.943, bag 2.860. All three lost to the
baseline (2.737 ± 0.012, n=3) under that schedule, which is the schedule that
does not work (§1.5). The mean-vs-gate comparison has not been repeated under
the schedule that does work.

### 1.4 Targets, and why the merged model cannot see the future

The training loop reads windows of `T + max(4, kmax)` tokens and passes

```
x = window[0 : T]          (input)
y = window[k : T + k]      (targets: shifted by k, not by 1)
```

Slot `j` of group `p` predicts token `p·k + k + j`, which is a token of the
*next* group. Here is `k = 2`, `T = 8`:

```
window:      t0 t1 | t2 t3 | t4 t5 | t6 t7 | t8 t9
groups:        g0      g1      g2      g3
g0 sees t0..t1   -> predicts t2, t3
g1 sees t0..t3   -> predicts t4, t5
g2 sees t0..t5   -> predicts t6, t7
g3 sees t0..t7   -> predicts t8, t9
```

Attention is causal over groups. Group `p` sees groups `0..p`, which hold
tokens up to `p·k + k − 1`. Its targets start at `p·k + k`. So no slot can see
the token it predicts or any later token. A naive per-token split with the
usual shift of 1 would leak: slot 0 of group `p` would predict token `p·k + 1`,
which is inside its own merged input. Hourglass transformers handle the same
problem with an explicit shift (Nawrot et al., 2022). Patch-level training
avoids it by predicting the next patch, as Zoom does.

The cost of this design: during the merged phase the model predicts 1 to `k`
tokens ahead, not always 1. That is a multi-token-prediction objective, and it
is a harder task than next-token prediction. Bantam evaluates at `k = 1`
only, and only the final `k = 1` loss is reported.

RoPE positions: with `--rope_pos token` (default) group `p` sits at angle
position `p·k + (k−1)/2`, the centre of its tokens, so the positional geometry
matches what the model sees at `k = 1`. `--rope_pos patch` uses the group index
`p` instead. One run (4:0.35,2:0.25,1:0.4) gave 2.857 with patch positions vs
2.874 ± 0.002 (n=3) with token positions. That is a single-seed difference of
0.017, about 1.4 baseline standard deviations, under the schedule that lost to
the baseline. It was not repeated.

### 1.5 Schedules

`--zoom` takes a list of `k:fraction` stages. The fractions are shares of the
FLOP budget and must sum to 1, for example `2:0.5,1:0.5`. `--zoom 1` is the
baseline. `ZoomSchedule` raises an error for a bad schedule and warns when the
last stage is not `k = 1` (the model would never train at the resolution it is
evaluated at). Every `k` must divide `--seq`.

The learning-rate schedule (`--sched cosine` or `wsd`) runs over the whole
budget and does not restart at a stage switch. `--lr_restart 1` restarts
warm-up and decay inside every stage; it was worse (2:0.5,1:0.5, seed 0: 2.760
with restart vs 2.723 without; with batch ×2: 2.787 vs 2.690). `--mix_fine p` runs a
`k = 1` step with probability `p` during coarse stages; no committed run uses
it.

Measured schedules (Muon, n=1 unless stated; compute-to-match explained in
§1.8):

| `--zoom` | batch during merged phase | final val loss | n | baseline compute to match |
|---|---|---|---|---|
| `1` (baseline) | 16 | 2.737 ± 0.012 | 3 | 1.00x |
| `4:0.35,2:0.25,1:0.4` (the 0.1/0.2 design) | 16 | 2.874 ± 0.002 | 3 | worse than baseline |
| `4:0.35,2:0.25,1:0.4` | × k (`--batch_scale_k 1`) | 2.730 | 1 | 1.02x |
| `4:0.2,2:0.2,1:0.6` | 16 | 2.800 | 1 | worse |
| `4:0.5,1:0.5` | 16 | 2.947 | 1 | worse |
| `4:0.6,2:0.2,1:0.2` | 16 | 3.022 | 1 | worse |
| `2:0.5,1:0.5` | 16 | 2.723 | 1 | 1.03x |
| `2:0.5,1:0.5` | × 2 (`--batch_scale_k 1`) | **2.685 ± 0.004** | 3 | **1.12x** |
| `2:0.5,1:0.5` | × 4 (`--batch_scale 4`) | 2.727 | 1 | 1.02x |
| `2:0.33,1:0.67` | × 2 (`--batch_scale 2`) | **2.680 ± 0.005** | 2 | **1.14x** |
| `2:0.65,1:0.35` | × 2 | 2.713 | 1 | 1.05x |

What made Zoom work at this scale: `k = 2` only, and a larger batch during the
merged phase. Every `k = 4` schedule at a fixed batch lost; with batch × k the
`4 → 2 → 1` staircase only tied the baseline (2.730, n=1). The schedule the old
docs presented as "the method" (`4 → 2 → 1`, fixed batch) loses on all three
seeds.

### 1.6 Batch scaling during the merged phase

`--batch_scale_k 1` uses `batch × k` sequences per step while zoomed, so the
number of positions per step stays constant. `--batch_scale s` multiplies the
batch by `s` while `k > 1` and overrides `--batch_scale_k`. The function is
`stage_batch` in `bantamtrain/cli/pretrain.py`.

Why it matters is not fully understood. Two plausible reasons, not tested
separately:

1. At a fixed batch, a `k = 2` step costs about 0.57x the FLOPs of a `k = 1`
   step (§1.7), so the same budget buys more, smaller steps. The 4→2→1 schedule
   took 808 steps against the baseline's 450. A `k = 2` step sees half the
   positions of a `k = 1` step (a `k = 4` step a quarter), so its gradient is
   noisier.
2. Muon's per-step cost is fixed and is not charged to the budget (§1.7). More
   steps also mean more uncharged optimizer work and more wall-clock.

With batch × 2 at `k = 2`, the `2:0.5,1:0.5` run takes 422 steps, close to the
baseline's 450. Batch × 4 was worse (2.727), so "bigger is better" does not
hold either.

### 1.7 FLOP accounting

Every run stops at a fixed FLOP budget (`--budget`). Each step is charged the
analytic training FLOPs of `ZoomGPT.flops_per_step(batch, seq, k)`
(3 × forward):

```
n    = seq / k                       positions through the blocks
s    = k  in slot mode with k > 1, else 1      (head evaluations per group)
step = 3 · batch · n · [ L·(24·d² + 2·n·d)  +  2·d·V·s  +  m ]
m    = 2·k·d² for each linear-mode merge or unmerge map, else 0
```

What this counts and what it does not:

- **The head is not made cheaper.** In slot mode the tied head runs once per
  token at every `k`. At d=256 and V=4,096 the head is about 17% of a baseline
  position's FLOPs. So one token costs 0.57x at `k = 2` and 0.37x at `k = 4`,
  not 0.5x and 0.25x (main config, seq 256).
- **Attention is charged at the causal half** (`2·n·d` per position per layer).
  SDPA computes the full `n × n` product, so the default undercounts attention.
- **Gate and mean merges are not charged.** They are elementwise and tiny.
  Linear merges are charged (since 0.3).
- **Optimizer FLOPs are not charged by default.** Muon's Newton-Schulz costs
  3.03e10 FLOPs per step for the main config, independent of `k` and batch. That
  is 20% of a charged `k = 1` step at batch 16, 35% of a `k = 2` step and 55%
  of a `k = 4` step. In the one 0.3 run that logs it
  (`results/published/save_muon_base_repro.json`), optimizer FLOPs are 1.37e13
  next to 6.81e13 model FLOPs. Runs with more steps get more uncharged compute:
  the 4→2→1 schedule received about 13% more real compute than its baseline and
  still lost. The winning `k = 2`, batch × 2 recipe takes slightly fewer steps
  than the baseline (422 vs 450), so this bias runs slightly against it.
  `--flops_mode total` charges optimizer FLOPs to the budget. Every run logs
  `opt_flops` and `total_flops` from 0.3 on; the committed Zoom runs predate
  that logging.
- **`exact=True`** counts the matmuls the kernels actually run: full `n × n`
  attention, the SDPA backward recompute (2.5x the attention forward), and the
  forward passes that `--grad_ckpt` and chunked cross-entropy recompute. At
  `k = 1` in the main config it is 1.085x the default count. `bantam pretrain`
  logs it as `model_flops_exact`; it is not used for the budget.

Wall-clock is not the same as FLOPs on this CPU. The old docs said coarse steps
run faster per FLOP. The logs show the opposite. At a fixed batch, `k = 2` steps
took 1.16-1.25x and `k = 4` steps 1.34-1.57x the wall-clock per charged FLOP
of the `k = 1` steps in the same run. With batch × k the cost per FLOP is about
the same as at `k = 1`. The 4→2→1 Muon runs took 1087, 944 and 957 s against
942, 790 and 807 s for their baselines. On weather (AdamW, 1 thread each) the
Zoom run took 1632 s against 1513 s. The machine was shared, so treat these
timings as rough. No wall-clock speedup from Zoom has been shown.

### 1.8 How "compute to match" is computed

`experiments/summarize.py` reads the final losses of plain Muon baselines at
1x, 1.25x, 1.5x and 2x the budget (the longer ones are single runs). It
averages seeds at each budget, interpolates linearly in log-FLOPs, and reports
how many baseline FLOPs reach a run's final loss. The 1.12x figure for the
3-seed Zoom recipe ranges from 1.11x to 1.13x across its seeds. A ±1 sd change
in the single-seed 1.25x baseline would move it between about 1.11x and
1.14x.

The evaluation set is the first 2,048 validation windows of 256 tokens. Early
windows are easier than later ones, so losses from different eval sizes are not
comparable (the periodic `evals[]` curve uses the first 256 windows and reads
0.27-0.30 nats lower than the 2,048-window final of the same run). Every
full-budget final loss in this file uses the 2,048-window set. Two tables are
exceptions and say so: the learning-rate sweep (§2.3, 512 windows) and the
short low-memory runs (§3, 1,024 windows). Every run's logged `data_epochs`
matches the same 116.8M-token deduplicated TokPack stream, which excludes the
1,550 training copies of validation stories (§9.2).

### 1.9 After training: the zoom modules

During training the merge and unmerge parameters stay in the model's
`state_dict` (3,072 extra parameters for levels 2 and 4 at d=256). They are
unused at `k = 1`. `--save PATH` writes the raw `state_dict` (with them) and
`<root>_pretrained/` through `ZoomGPT.save_pretrained(strip_zoom=True)`, which
drops them and writes a plain `k = 1` model (§12.3). Only the stripped export
is "an ordinary model with no extra parameters".

### 1.10 What Zoom borrows and what is new

Known ideas that Zoom reuses:

- Coarse token granularity first, then tokens, keeping all weights: patch-level
  training (Shao et al., 2025).
- Shortening a sequence inside the network and upsampling for per-token output:
  Hourglass (Nawrot et al., 2022), Funnel-Transformer (Dai et al., 2020).
- Cheap positions early in training: sequence-length warm-up (Li et al., 2022),
  progressive resizing in vision, dataset decomposition.
- A learned merge decision: dynamic token pooling (Nawrot et al., 2023).

Bantam's own choices: a learned per-slot gate instead of averaging, a
per-token split at the output instead of a bag loss, `k = 2`, a larger batch in
the merged phase, and running it with Muon. These are design variants worth an
ablation, not a new method.

Size of the effect: about 11-12% less compute for the same loss (the baseline
needs 1.12-1.14x the compute to match), at 5.8M parameters on TinyStories.
Patch-level training reports about 2x at 370M-2.7B parameters. The gain stacks with QK-norm: QK-norm alone gives 2.639 ± 0.003
(n=2, 1.24x), QK-norm plus `2:0.33,1:0.67` with batch × 2 gives 2.589 ± 0.006
(n=2, 1.41x). The Zoom gain (about −0.05) is the same with and without
QK-norm. On the weather model Zoom did not pay off under Muon (§10). Zoom has
not been tested at larger scale or on a GPU.

Commands for the two best recipes (data prepared as in the README):

```bash
bantam pretrain --data DATA/tok4k --budget 6.8e13 --optim muon --lr 1.6e-2 \
    --zoom 2:0.5,1:0.5 --batch_scale_k 1 --name zoom_k2 --out results/mine

bantam pretrain --data DATA/tok4k --budget 6.8e13 --optim muon --lr 1.6e-2 \
    --qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2 --name qk_zoom --out results/mine
```

---

## 2. Muon

Muon (Keller Jordan, 2024) is an optimizer for the 2-D weight matrices inside
the blocks. It keeps a momentum buffer, and each step replaces the update with
the nearest semi-orthogonal matrix: it keeps the directions of the momentum
matrix and sets all its singular values to about 1. Bantam implements it in
`bantamtrain/optim.py` (`Muon`, `newton_schulz`, `build_optimizer`).

### 2.1 One Muon step

For each parameter `W` (shape `out × in`) with gradient `G`:

```
buf  <- lerp(buf, G, 1 - 0.95)                 momentum 0.95
M    <- lerp(G, buf, 0.95)                     Nesterov (default)
U    <- NewtonSchulz(M)                        ≈ U_svd · V_svdᵀ of M
W    <- W · (1 - lr·wd)  -  lr · scale · U     decoupled weight decay
scale = 0.2 · sqrt(max(rows, cols))            "moonlight" scaling (default)
```

The moonlight scale (Liu et al., Moonshot AI, 2025) is meant to give the update
roughly the RMS of an AdamW update. `scale="jordan"` (library only, not a CLI
flag) uses the original `sqrt(max(1, rows/cols))`.

Newton-Schulz runs in fp32 on CPU and bf16 on CUDA. The matrix is first divided
by its Frobenius norm and transposed so that rows ≤ cols. Each iteration is

```
A = X Xᵀ
X <- a·X + (b·A + c·A²)·X
```

using `baddbmm`, with two buffers the size of `X`.

- `--ns quintic` (default): five iterations with the fixed coefficients
  `(a, b, c) = (3.4445, −4.7750, 2.0315)` from the Muon reference code. The
  reference code describes the output singular values as roughly 0.5 to 1.5;
  exact orthogonality is not the goal.
- `--ns polar`: Polar Express (Amsel et al., 2025) uses a different
  coefficient triple at each iteration. Bantam uses the paper's 8 tuples
  (listed in [RESEARCH.md §2.1](../RESEARCH.md)), divides `a, b, c` of the first
  seven by 1.01, 1.01³, 1.01⁵ for safety, repeats the 8th tuple
  `(1.875, −1.25, 0.375)` for any further steps, and normalizes the input with
  a 1.01 margin. 0.2 kept only five tuples and repeated the fifth, which does
  not converge to 1 (audit OPT-8; fixed). With the default 5 steps the change
  is small: in a hand check on random Gaussian matrices the largest output
  singular value was 1.124; how close the smallest get to 1 depends on the
  input's conditioning (about 0.87 for a 256 × 1,024 matrix, below 0.2 for a
  square one).

Measured: `--ns polar` gave 2.755 vs 2.737 ± 0.012 for quintic (n=1). It did
not help at this scale. That run used the 0.2 coefficient list (before the
OPT-8 fix), so it should be rerun before it is quoted as a 0.3 result.

### 2.2 Which parameters Muon gets

- **Bantam's own models** (`ZoomGPT`, `ZoomViT`) list their Muon parameters in
  `muon_params()`: `qkv`, `proj`, `fc` and `fc_out` of every block. The fused
  `qkv` weight is orthogonalized as three separate `d × d` blocks (q, k, v),
  given by `muon_splits()`.
- **Hugging Face models** go through `default_muon_params`: every trainable 2-D
  `nn.Linear` weight and transformers `Conv1D` weight (GPT-2, stored
  transposed), except embeddings, the output head and quantized modules.
- **Everything else stays on AdamW**: embeddings, the (tied) head, norms,
  biases, zoom gates, DoRA magnitudes.
- **LoRA**: with `--optim muon` and `--lora > 0`, the LoRA `A` and `B`
  matrices go to Muon (`--muon_lora -1`, auto). This is new in 0.3. In 0.2
  `--optim muon` with LoRA silently trained the adapters with 8-bit AdamW, so
  the old "Muon with 8-bit state" fine-tuning results were really 8-bit AdamW
  results. No learning rate has been tuned for Muon on LoRA here.

### 2.3 Learning rate: one LR for both halves

`build_optimizer` gives the Muon part and the AdamW part the same learning rate
and weight decay. The old docs said Moonlight scaling makes that safe. The
sweep here says otherwise. At 2.3e13 FLOPs (one third of the budget, n=1 each,
scored on the first 512 validation windows, so not comparable with the
2,048-window finals):

| lr | AdamW only | Muon (+ AdamW part at the same lr) |
|---|---|---|
| 5e-4 | 4.225 | 4.226 |
| 1e-3 | **3.961** | 3.937 |
| 2e-3 | 4.090 | 3.692 |
| 4e-3 | (not run) | 3.440 |
| 8e-3 | (not run) | 3.230 |
| 1.6e-2 | (not run) | **3.176** |
| 3.2e-2 | (not run) | 3.208 |
| 6.4e-2 | (not run) | 3.522 |

The best LRs differ by 16x (AdamW 1e-3, Muon 1.6e-2). In every Muon run the
AdamW part (the embedding and tied head plus norms: 1.05M of 5.77M parameters)
also runs at 1.6e-2. So "Muon 2.737 vs AdamW 3.332 at equal FLOPs" changes two
things at once: the optimizer of the block matrices and the LR of the
embedding/head. At the same LR (1e-3) the two were about equal. An ablation with
AdamW-part LR 1e-3 under Muon has not been run. Muon is still the largest
single improvement measured here, but how much of it is Muon itself is not
known.

### 2.4 Cost

`ns_flops(rows, cols, steps)` gives the Newton-Schulz cost:
`steps · (4a²b + 2a³ + 2a² + 2ab) + 3ab` with `a = min(rows, cols)`,
`b = max(...)`. Elementwise work adds about 10 FLOPs per element. For the main
config that is 3.03e10 FLOPs per step, 20% of a baseline step (§1.7). The
q/k/v split costs more than one fused matrix would (the `a³` term).
Updates are applied in chunks of 4M elements (`STEP_CHUNK`), so the fp32
temporaries stay small, but Newton-Schulz still needs fp32 copies of each whole
matrix.

---

## 3. 8-bit optimizer state

`--state_bits 8` stores optimizer state in 8 bits (`bantamtrain/lowmem.py`,
`QState`). It follows the blockwise idea of Dettmers et al. (2022, the
bitsandbytes 8-bit optimizers): every block of 256 consecutive values has its
own fp32 scale, so one outlier cannot ruin the precision of the whole tensor.

Two value maps:

- **Momentum (signed, int8):** `q = sign(x)·round(sqrt(|x|/absmax)·127)`. The
  square root gives small values finer steps than a linear map. Most momentum
  entries are far below their block's maximum. Muon's momentum and Adam's first
  moment use this map.
- **Adam's second moment (non-negative, uint8, log map):** code 0 is exact zero;
  codes 1-255 cover `[log min, log max]` of the block's positive entries in
  254 steps. About 2% relative precision when a block spans many orders of
  magnitude, 0.45% for one decade. When `beta2 > 0.95` the codes are rounded
  stochastically so the stored average stays unbiased; Bantam's default betas
  are (0.9, 0.95), so CLI runs round to nearest.

The 0.2 log map padded partial blocks with zeros and let them stretch the
range, which cost about 25x precision on every tensor smaller than 256 elements
(norm weights, biases, gates) and on the tail block of others (audit OPT-4;
fixed in 0.3). That changes 8-bit AdamW numerics, so the committed 8-bit runs
used the old map.

Bytes per parameter during training:

| configuration | weights | grads | AdamW state | Muon state |
|---|---|---|---|---|
| fp32 everything | 4 | 4 | 8 | 4 |
| bf16 weights + 8-bit state | 2 | 2 | ~2 | ~1 |

Measured (5.8M model; n=1 each; 0.2 numerics; the 1.5e13 runs are scored on
the first 1,024 validation windows, the 6.8e13 run on 2,048):

| run | budget | 32-bit state | 8-bit state |
|---|---|---|---|
| Muon, fp32 weights | 1.5e13 | 3.572 | 3.581 |
| AdamW, fp32 weights | 1.5e13 | 4.482 | 4.479 |
| Muon, 8-bit state + bf16 weights | 1.5e13 | 3.572 (fp32 weights) | 3.575 |
| Muon, 8-bit state + bf16 weights | 6.8e13 | 2.737 ± 0.012 (fp32 weights, n=3) | 2.728 |

Both bf16 rows were scored with 0.2's bf16 cross-entropy and have no saved
checkpoint, so they cannot be re-scored. Their evaluation error is about
±0.001 (full budget) and ±0.004 (short run)
(`results/llm_lowmem/bf16_eval_bias.json`); both are marked for a rerun.
Weights plus optimizer state went from 50.4 MB to 18.5 MB (2.7x) for this
model. The bf16 path ran 1.6-2.7x slower in wall-clock on this CPU (5039 s vs
3072 s at full budget; 718 s vs 271 s for the short runs, 1 thread each). The
cause was not profiled.

The same techniques are available elsewhere: bitsandbytes ships fused CUDA
8-bit optimizers, and torchao ships pure-PyTorch AdamW8bit/4bit/FP8 with a
`bf16_stochastic_round` flag. 8-bit Muon momentum is studied in Gupta et al.
(2025). Bantam's version is portable and readable; it is not faster.

---

## 4. bf16 weights with stochastic rounding

`--param_dtype bf16` keeps the weights in bf16 with no fp32 master copy. The
optimizer computes each update in fp32 and writes the result back with
stochastic rounding (`lowmem.to_bf16_stochastic`).

Why: bf16 stores 7 mantissa bits (8 with the implicit leading bit). A weight
near 1.0 has a step of about 0.008.
A typical update `lr × u` is far smaller, so round-to-nearest would drop it and
the weight would never move. Stochastic rounding rounds up with probability
equal to the remainder, so the expected stored value equals the fp32 value and
small updates add up over many steps (Gupta et al., 2015; Zamirai et al.,
2020).

How, on the bit pattern of each fp32 value:

```
bits    = float32 as int32
noise   = uniform integer in [0, 2^16)
rounded = (bits + noise) & 0xFFFF0000        keep the top 16 bits = bf16
```

**Counter-based noise, the same on every rank.** 0.2 drew the noise from the
global RNG. 0.3 seeds a `torch.Generator` for every chunk of every update from
a hash (splitmix64) of five integers:

```
(sr_seed, optimizer tag, parameter index, per-parameter step, chunk index)
```

Consequences:

- Every DDP rank rounds identically, so replicas stay bit-identical without
  broadcasting weights.
- A resumed run draws exactly the noise an uninterrupted run would.
- Training noise does not depend on anything else that uses the global RNG.
- Changing the chunk size changes the noise (not its distribution).

New bf16 runs are therefore not bit-identical to 0.2 runs. The idea of deriving
random numbers from counters instead of a sequential stream is the
counter-based RNG approach (Salmon et al., 2011); Bantam uses a hash-derived
seed per chunk rather than a Philox stream.

Evaluation of a bf16 model now runs in fp32 on an fp32 copy of the rounded
weights, and the training loss is computed in fp32. In 0.2 the whole evaluation
ran in bf16 arithmetic. That added rounding noise to the reported losses, not a
fixed offset: measured on bf16-rounded copies of an fp32 checkpoint it was
+0.0007 at 2,048 windows and up to ±0.007 on smaller window sets
(`results/llm_lowmem/bf16_eval_bias.json`; audits ZG-3 and CLM-18). Rounding
the weights themselves to bf16 changed the loss by less than 1e-4.

---

## 5. Quantized frozen weights (int8, int4, NF4)

`bantamtrain/qweights.py` stores frozen weights in 8 or 4 bits and computes in
float. It is the base-model half of QLoRA (Dettmers et al., 2023), written in
pure PyTorch with no bitsandbytes dependency.

| format | codes | default block | bytes per parameter |
|---|---|---|---|
| `8` | int8 in [−127, 127] | 128 | ~1.02 |
| `4` | linear int4 in [−7, 7], two per byte | 64 | ~0.53 |
| `nf4` | 4-bit NormalFloat: 16 code values at quantiles of N(0, 1) (the bitsandbytes table), two per byte | 64 | ~0.53 |

Each block of consecutive input columns has one scale (fp16, or fp32 when a
tensor's scales leave the fp16 normal range). When a row width is not a
multiple of the block, the largest divisor between `block/2` and `block` is
used (width 576 with block 128 uses 96), so common widths need no padding.
Unlike the QLoRA paper, the scales are not themselves quantized ("double
quantization").

**`QLinear` saves only the codes.** It multiplies through a custom autograd
function (`_QMatmul`). Forward dequantizes the weight in row chunks, in the
input's compute dtype, multiplies, and frees it. Backward dequantizes again to
compute the input gradient. No weight gradient is computed, and the float
weight is never kept between forward and backward. So a 4-bit frozen base
costs about 0.53 B/param during training too.

Other pieces:

- `QEmbedding` dequantizes only the rows it gathers.
- A tied output head (`QTiedHead`) reuses the embedding's codes, stored once.
- The chunked loss (§7) receives the quantized head itself and dequantizes it
  block by block per chunk, so no full float vocabulary table exists.
- `load_quantized` (`--low_ram 1`) builds the model on the meta device and
  quantizes each safetensors tensor as it is read, so the float model is never
  in RAM at once.
- `--embed_quant 8` (default) keeps embeddings and the head at int8 whatever
  `--quant` is; `same` uses the `--quant` format; `none` keeps them in float.
- Linear subclasses with their own forward (Gemma's scaled embeddings, OPT's
  learned positions) keep their class; only the weight storage changes.

Measured quality on SmolLM2-135M, before any training: held-out loss on 88
Dolly examples (one run, CPU). The windows were packed before the block-diagonal
mask existed (§8.4), so the deltas include cross-example attention. "Model MB"
includes 19.5 MB of fp32 LoRA adapters.

| linears | embeddings/head | loss change vs fp32 | model MB |
|---|---|---|---|
| fp32 | fp32 (nothing quantized) | 0 (2.702) | 558 |
| int8 | int8 | +0.003 | 200 |
| NF4 | int8 (default with `--quant nf4`) | +0.18 | 140 |
| NF4 | float | +0.17 | 189 |
| linear int4 | int8 | +0.33 | 140 |
| NF4 | NF4 | +0.34 | 106 |
| linear int4 | int4 | +0.52 | 106 |

NF4 is the better 4-bit format for pretrained weights; int8 is nearly free.
A 4-bit base costs real quality at 135M parameters. In the one committed QLoRA
run (int4 everything, 40 CPU steps, 8 eval windows of about 17 examples, n=1)
the loss went from 3.389 to 3.178, which is still worse than the untouched
float base (2.871; LoRA on the float base reached 2.792). Those two runs also
used 8-bit AdamW, not Muon (§2.2).

---

## 6. LoRA, DoRA, LoRA+, rsLoRA

`bantamtrain/lora.py` wraps linear layers with low-rank adapters (Hu et al.,
2021). Each targeted `nn.Linear`, transformers `Conv1D` or `QLinear` becomes

```
y = base(x) + (dropout(x) · Aᵀ · Bᵀ) · scale
A: r × d_in   (Kaiming-uniform init)
B: d_out × r  (zeros, so training starts exactly at the base model)
scale = alpha / r,  or alpha / sqrt(r) with rsLoRA
```

The base is frozen. Adapters are created in fp32 on the base layer's device.
Trainable parameters per layer: `r · (d_in + d_out)`. SmolLM2-135M with r=16 on
all seven projection types has 210 adapted modules and 4,884,480 trainable
parameters.

Flags of `bantam finetune`:

| flag | default | what it does | source |
|---|---|---|---|
| `--lora R` | 16 | rank; `--lora 0` = full fine-tuning | Hu et al., 2021 |
| `--lora_alpha` | 32 | scale numerator | |
| `--lora_dropout` | 0.05 | dropout on the adapter input | |
| `--lora_targets` | all-linear | every linear layer except the output head; or a comma list of module names | |
| `--rslora` | −1 (auto: on when r ≥ 64) | `scale = alpha/sqrt(r)` | Kalajdzievski, 2023 |
| `--dora 1` | off | weight = `m · (W + scale·BA) / ‖W + scale·BA‖` per output row, with a trainable magnitude `m` initialised to `‖W‖`; the norm is detached from the graph, as in the paper and PEFT | Liu et al., 2024 |
| `--lora_plus RATIO` | 0 (off) | the `B` matrices get `lr × RATIO` | Hayou et al., 2024 |
| `--quant 8/4/nf4` | 0 | quantize the frozen base (§5) | Dettmers et al., 2023 |
| `--optim muon` | adamw | Muon on the `A`/`B` matrices (§2.2) | |

The 0.2 default target list was a set of leaf names that missed some
architectures (Phi-3 `qkv_proj`/`gate_up_proj`, OPT and Phi `fc1`/`fc2`, GPT-J
`fc_in`). 0.3 defaults to all linear layers.

**PEFT interop.** `save_peft_adapter` and `load_peft_adapter` write and read
PEFT's `adapter_config.json` plus `adapter_model.safetensors`. `bantam finetune`
writes one by default (`--save_peft -1`), so `PeftModel.from_pretrained` loads
Bantam adapters and PEFT adapters load into Bantam. Mixed ranks are written as
`rank_pattern`/`alpha_pattern`. PEFT options Bantam does not support
(`layers_to_transform`, `modules_to_save`, trained biases) raise an error
instead of loading wrongly. `merge_lora` folds `W += scale·BA` into the base;
`--save_merged 1` writes a merged model (with `--quant`, into the reloaded
float base).

Example:

```bash
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --lora 16 --quant nf4 --optim muon --out runs/dolly
```

---

## 7. Chunked cross-entropy

For a large vocabulary the logits are the largest activation. At 16K tokens per
step and a 128K vocabulary they take 8 GB in fp32. `chunked_cross_entropy`
(`bantamtrain/losses.py`) never builds them all at once:

```
for each chunk of `chunk` positions (default 2048):
    logits = softcap(scale · (h_chunk · Wᵀ + b))      in fp32
    loss  += sum of cross-entropy over the chunk      (weighted if token_weights)
    -- torch.utils.checkpoint: the chunk's logits are freed and recomputed in backward
loss /= number (or total weight) of supervised targets
```

Peak loss memory falls from `tokens × V × 4` bytes to `chunk × V × 4`. The
result matches `F.cross_entropy` up to summation order. The price is one extra
head matmul per chunk in backward.

Details:

- `softcap` (Gemma-2 style `c·tanh(z/c)`) and `scale` (Cohere `logit_scale`,
  Granite `1/logits_scaling`, Falcon-H1 `lm_head_multiplier`) are applied
  inside the chunk. `bantam finetune` reads them from the model config, so the
  chunked loss trains the same objective as the model's own forward. In 0.2 the
  chunked path skipped them for these families (audit FT-1; fixed).
- `token_weights` gives a weighted mean, used for `--prompt_loss_weight`.
- `weight` can be a quantized head; each chunk dequantizes it (§5).
- A loss with nothing supervised returns 0, never NaN.

Where it is used: `bantam finetune --ce_chunk 2048` (default; 0 turns it off),
and `ZoomGPT(ce_chunk=...)` from Python. `bantam pretrain` has no `--ce_chunk`
flag yet, so pretraining always builds the full logits (fine for a 4K
vocabulary).

When it does not help: if a micro-batch holds fewer tokens than one chunk,
there is nothing to save (the fine-tuning race uses about 1,024 tokens per
micro-batch, below 2,048). On GPUs, fused kernels (Liger's fused linear
cross-entropy, Apple's Cut Cross-Entropy) do the same job faster and with less
memory; see [COMPARISON.md](COMPARISON.md).

---

## 8. Packed fine-tuning data

### 8.1 From examples to tokens

`bantam finetune` reads a Hugging Face dataset (`--dataset`) or a local JSONL
(`--jsonl`). It accepts prompt/response columns, Dolly-style
instruction/context/response, chat `messages`, or ShareGPT `conversations`. It
uses the tokenizer's chat template when there is one. A base checkpoint without
a template gets ChatML when its vocabulary has the `<|im_end|>` token (SmolLM2,
Qwen2.5 bases), otherwise a plain format. The loss covers response tokens
only by default; `--prompt_loss_weight w` gives prompt tokens weight `w`.

### 8.2 Packing

`--pack` groups examples into windows of `--seq + 1` tokens:

- `bfd` (default): best-fit-decreasing bin packing. Examples are sorted by
  length and each goes into the fullest window it fits. Examples are never
  split (one longer than the window is truncated to it). It runs in
  O(n log n).
- `wrapped`: concatenate everything and cut every `--seq` tokens (examples can
  be split across windows).
- `none`: one example per window, padded.

Each window carries segment ids: 0, 1, 2, ... for its examples, −1 for padding.
The first token of each example is never a target, so no example is asked to
predict its own start from the previous example.

### 8.3 Block-diagonal attention

`segment_mask` turns the segment ids into a causal mask that only allows
attention inside an example, and into position ids that restart at 0 for each
example:

```
window:      A0 A1 A2 A3 | B0 B1 B2 | C0 C1 | pad pad
segments:     0  0  0  0    1  1  1    2  2    -1  -1
positions:    0  1  2  3    0  1  2    0  1

mask (row may attend to column):
        A0 A1 A2 A3 B0 B1 B2 C0 C1
   A0    x
   A1    x  x
   A2    x  x  x
   A3    x  x  x  x
   B0                x
   B1                x  x
   B2                x  x  x
   C0                         x
   C1                         x  x
```

So a packed example is computed as if it were alone. A wave-2 check compared
Bantam's packed eval loss on SmolLM2-135M with an independent computation
(plain transformers + `peft.PeftModel`, one example at a time, no packing): they
differed by 2.9e-7. The fine-tuning race repeats this cross-check on its own
held-out set ([COMPARISON.md](COMPARISON.md) §4). When every row is a single unpadded example,
no mask is built and the plain causal fast path is used.

The mask is passed in the form the model's attention implementation expects:
a boolean mask for SDPA, and an additive float mask (0 or the dtype's minimum)
for eager and flex attention, which add the mask to the scores.

### 8.4 The leak in 0.2

- Early 0.2 builds packed examples with no mask at all: an example could attend
  to the examples before it in its window. The committed quantization-quality
  runs (§5) were made that way.
- Late 0.2 added the block-diagonal mask, but always as a boolean tensor.
  SDPA honours that. Eager and flex attention add the mask to the scores, so a
  boolean mask added 0 or 1 and masked nothing, not even future tokens. That hit
  models that load with eager attention, such as Gemma-2.
- 0.3 passes an additive float mask to non-SDPA implementations. SDPA models
  (Llama, SmolLM2, Qwen, GPT-2) are unchanged.

### 8.5 Known limits

- A prepared 4-D mask overrides sliding-window attention. Layers with a window
  (Gemma-3 above 512/1024 tokens, Mistral/Phi-3/Starcoder2 above their window,
  Gemma-2 above 4096) lose it when `--seq` exceeds the window. The default
  `--seq 512` is unaffected.
- `flash_attention_*` implementations cannot take a 4-D mask. The CLI never
  selects them.
- A dense `T × T` mask is slower than FlashAttention's variable-length packing
  (used by TRL padding-free and Axolotl multipack). That matters on GPUs.
- "1 epoch" drops the last windows that do not fill a whole optimizer step (up
  to `batch × accum − 1` windows; 3 of 235 in the race protocol).

### 8.6 Loss normalisation

With `--accum > 1` or several processes, the loss is the sum of token losses
over the whole step divided by the number of supervised tokens in the step. In
0.2 it was the mean of per-micro-batch means, which over-weights short
micro-batches. The two agree for `--accum 1` on one process.

---

## 9. Data storage: TokPack, dedup, GridPack

### 9.1 TokPack (token streams)

`bantam pack-text` writes `train.tokpack` and `val.tokpack` next to the raw
`.bin` files (`bantamtrain/data/tokpack.py`). Format v2:

1. **Frequency-rank remap.** Each token id becomes the rank of its corpus
   frequency, so common tokens become small numbers.
2. **LEB128 varint.** Ranks below 128 take one byte, below 16,384 two, below
   2²¹ three.
3. **zstd level 19 per chunk** of 16,384 tokens (`--chunk`), every chunk using
   one shared zstd dictionary trained on a sample of the corpus and stored in
   the file.

The dictionary lets small chunks (cheap random access) compress like large
ones. On TinyStories with the 4,096-token vocabulary this gives about 6
bits/token (0.75 bytes/token, against 2 bytes for uint16). That figure was
measured by hand during development; no `pack_stats.json` for TinyStories is
committed under `results/`. Decoding is exact: `decode(encode(x)) == x`.

Also in the file: an optional document index (zstd-compressed document lengths,
about 1.15 bytes per document) and a `source` record of which prepared data the
pack was made from. Readers raise `StalePackError` when the pack no longer
matches `meta.json`; `bantam pretrain` then falls back to the `.bin` with a
warning.

Format v1 (rank remap, byte-plane shuffle, zstd) is still readable but no
longer written. The byte shuffle made the files larger (8.55 vs 7.86 bits/token
for plain zstd on uint16), because it splits each symbol across two byte planes
and breaks zstd's matches. The 0.2 METHOD.md still described v1.

### 9.2 Deduplication

`bantamtrain/data/dedup.py`, run by `bantam pack-text --dedup 1` (default),
works on the token stream with documents separated by EOS:

- **Exact:** a 64-bit hash of each document's tokens.
- **Near:** MinHash over token 8-gram shingles (`--ngram 8`), 64 permutations
  (`--num_perm`), LSH with 8 bands of 8 rows (`--bands`). Candidate pairs that
  share a bucket are confirmed when their estimated Jaccard similarity is at
  least 0.8 (`--threshold`), then clustered with union-find; one document per
  cluster is kept.
- **Train vs validation:** training documents identical to a validation
  document are dropped by default (`--drop_val_overlap 1`, in both
  `prepare-text` and `pack-text`).

On TinyStories this drops 21,800 documents: 20,250 exact duplicates inside the
training split and 1,550 training copies of validation stories. No near
duplicates were found. These counts come from the pack's `pack_stats.json`,
which lives with the prepared data, not under `results/`, plus the 0.3 audit's
recount. `dedup_stream` gives the same result out of core (about 14 bytes of
RAM per document).

The raw 0.2 `train.bin` still contained the validation copies (audit DATA-1).
The published runs read the deduplicated TokPack, so they are not affected,
but a run that falls back to the raw `.bin` trains on a different corpus.
`bantam pretrain` warns when that happens.

MinHash deduplication follows Lee et al. (2022) and FineWeb's pipeline
(5-grams, 112 hashes); Bantam's parameters are smaller and suited to small
corpora.

### 9.3 GridPack (weather fields)

`bantam pack-era5` writes `fields.gridpack` (`bantamtrain/data/gridpack.py`).
float16 is a poor fit for weather fields: at z500 ≈ 5.4e4 m²/s² its step is 32
m²/s², about the size of a 6-hour forecast error, and surface pressure (≈ 1e5
Pa) does not fit at all. GridPack instead:

1. Quantizes each channel to int16 around its mean, with one scale per channel,
   chosen so the error is at most `err_frac` (default 0.05) of that channel's
   6-hour-change standard deviation.
2. Stores, inside each chunk of 64 time steps (`--chunk`), the difference from
   the previous step. Consecutive 6-hour states differ by about 10
   quantization steps, so the deltas are small integers.
3. Byte-shuffles the deltas and compresses each chunk with zstd level 10
   (`--level`).

The error bound `|x̂ − x| ≤ err_frac · diff_std` is checked on every value, in
float64, when the pack is written. Constant channels round-trip exactly.

---

## 10. Weather model: ZoomViT and low-resolution warm-up

### 10.1 The model and the task

`bantam weather` trains a 6-hour forecaster on ERA5 at 5.625° (a 64 × 32 grid,
from WeatherBench 2) with six channels: z500, t850, t2m, u10, v10, msl.

- **Input:** the current state (normalized per channel) plus six forcing
  channels: two static fields, local solar time (sin, cos) and day of year
  (sin, cos).
- **Target:** the 6-hour change, divided by each channel's standard deviation
  of 6-hour changes (a residual target).
- **Loss:** latitude-weighted MSE over all channels. A persistence forecast
  (predict no change) scores roughly 1.0 on this scale.
- **Model:** `ZoomViT` (`bantamtrain/models/vit_weather.py`). Each 2 × 2 cell
  patch is one token, so 512 tokens. d=128, 6 layers, 4 heads, a learned
  position embedding, 1.26M parameters. The output layer is zero-initialised,
  so training starts at persistence.
- **Metrics:** latitude-weighted RMSE in physical units at 6, 24 and 72 hours
  (autoregressive rollout), next to persistence. 0.3 adds latitude-weighted
  anomaly correlation (ACC) against a training-years climatology, and a
  climatology RMSE baseline.

### 10.2 Zoom on a grid

At zoom level `k = f²`, each `f × f` block of tokens is merged into one token
with the same `SlotMerge`/`SlotUnmerge` as the language model, the blocks run
on `512/k` tokens, and the result is split back to every token. The position
embedding is added before the merge. Attention is not causal here, so the
causality issue of §1.4 does not arise. Levels must be perfect squares (4, 16)
whose root divides the 16 × 32 token grid.

### 10.3 Low-resolution warm-up (`--coarse_mode lowres`)

This is progressive resizing, an established technique. During a coarse stage
the input fields and targets are area-averaged by `f` (so `--zoom 4:0.6,1:0.4`
trains on a 32 × 16 grid for 60% of the compute), the position embedding is
average-pooled the same way, and the same weights are trained on that smaller
problem. Then training continues at full resolution. The fine-scale detail is
not seen during the coarse stage. FLOPs for coarse steps charge embedding and
decoding for the coarse patches only (a 0.3 fix, WX-11). The committed lowres
runs predate that fix: they were charged like merge runs (2,128 steps each, as
the merge runs), so they were slightly over-charged (about 2% per coarse step
at `k = 4`). The bias runs against low-res warm-up.

### 10.4 What was measured

ERA5 64 × 32, train 2010-2017, evaluated on 2018. FLOP-matched at 5.5e13.
Muon lr 1.6e-2, AdamW lr 2e-3, batch 16 (no batch scaling in any weather Zoom
run). Val MSE is the normalized one-step loss; z500 RMSE is in m²/s² (mean of
the seeds; the two Muon baseline seeds give 539 and 551 at 72 h).

| run | optimizer | val MSE | n | z500 RMSE 24 h | z500 RMSE 72 h |
|---|---|---|---|---|---|
| baseline | Muon | **0.1240 ± 0.0006** | 2 | 217.5 | 545 |
| Zoom merge, k=4 for 60% | Muon | 0.1588 ± 0.0005 | 2 | 255 | 612 |
| Zoom merge, k=4 for 40% | Muon | 0.129 | 1 | 219 | 548 |
| Zoom, mean merge, k=4 for 60% | Muon | 0.128 | 1 | 206 | 527 |
| Zoom 16:0.3,4:0.3,1:0.4 | Muon | 0.180 | 1 | 275 | 634 |
| low-res warm-up, 4:0.6 | Muon | **0.112** | 1 | **194** | **511** |
| baseline at 2x compute | Muon | 0.094 | 1 | 172 | 461 |
| baseline | AdamW | 0.229 | 1 | 346 | 737 |
| Zoom merge, k=4 for 60% | AdamW | 0.180 | 1 | 282 | 647 |
| low-res warm-up, 4:0.6 | AdamW | 0.158 | 1 | 257 | 601 |
| persistence | - | - | - | 569 | 907 |

Reading it:

- Muon is again much better than AdamW (0.124 vs 0.229), with the same LR
  caveat as §2.3.
- **Zoom merging does not pay off on weather under Muon.** Every merge schedule
  has a worse val MSE than the baseline. (The mean-merge run had a lower z500
  RMSE, 206 vs 217.5 at 24 h, but a worse val MSE; one seed.) With AdamW Zoom beats the AdamW
  baseline, but all AdamW runs lose to the Muon baseline; the AdamW gain most
  likely reflects an under-tuned AdamW, not a compute saving.
- **Low-resolution warm-up helped** with both optimizers: about 10% lower MSE
  than the Muon baseline (1 seed), z500 24-hour RMSE 194 vs 217.5.
- The model beats persistence clearly (217.5 vs 569 at 24 h). It is a small demo
  at the coarsest WeatherBench resolution, not a competitive forecaster.

Two caveats. First, the learning rates were chosen on 2018, the same year the
results are reported on; the committed runs had no separate validation year
(audit WX-3). New runs can hold out a validation year with `--val_year` (train
before it, evaluate periodically on it, touch the test year only at the end).
Second, no wall-clock gain has been shown. Coarse stages take many more,
cheaper steps (2,128 steps against 567 for the baseline) and each step has a
fixed cost. Timings on the shared machine were too noisy to rank the runs:
the AdamW and Muon baselines ran the same number of steps in 1513 s and
700-817 s.

```bash
bantam weather --data DATA/era5_64x32 --budget 5.5e13 --optim muon --lr 1.6e-2 \
    --zoom 4:0.6,1:0.4 --coarse_mode lowres --val_year 2017 --name lowres_v \
    --out results/mine
```

---

## 11. Model options for the small GPT

`ZoomGPT` (`bantamtrain/models/gpt.py`) is a pre-norm decoder: RMSNorm, fused
qkv projection, RoPE, SDPA attention, a 4x MLP, tied embedding and head, no
biases. Residual projections are initialised with std `0.02/sqrt(2L)`. Options
borrowed from the literature and from the modded-nanogpt speedrun, with their
measured effect here (Muon, 6.8e13 FLOPs, vs baseline 2.737 ± 0.012):

| flag | what it does | result | n | from |
|---|---|---|---|---|
| `--qk_norm 1` | RMSNorm on q and k per head before RoPE | **2.639 ± 0.003 (1.24x)** | 2 | Henry et al. 2020; Dehghani et al. 2023 |
| `--softcap 15` | logits `15·tanh(z/15)` | 2.726 (1.02x) | 1* | Gemma 2; modded-nanogpt |
| `--zero_init_proj 1` | zero-init attention and MLP output projections | 2.732 (1.01x) | 1* | modded-nanogpt |
| `--softcap 15 --zero_init_proj 1` | both | 2.728 (1.02x) | 1 | |
| `--mlp relu2` | ReLU² instead of GELU | 2.829 (worse) | 1* | So et al. 2021 (Primer) |
| `--sched wsd` | warm-up, flat, linear decay over the last 45% | 2.773 (worse) | 1* | Hägele et al. 2024 |
| `--ns polar` | Polar Express orthogonalizer (§2.1) | 2.755 (worse) | 1† | Amsel et al. 2025 |
| `--softcap 15 --zero_init_proj 1 --zoom 2:0.5,1:0.5 --batch_scale 2` | both, plus Zoom | 2.691 (1.11x) | 1 | |

\* made earlier in the project with an older build of the code, before the
environment was rebuilt (their logs lack fields later runs have). Their
`data_epochs` matches the same 116.8M-token deduplicated stream as every other
run. Indicative only.
† run with the 0.2 Polar Express coefficients (before the OPT-8 fix); to be
rerun.

Softcap plus zero-init added nothing on top of Zoom: 2.691 vs 2.690 for the
same Zoom recipe alone on seed 0.

QK-norm is the one that clearly helped, and it beat Zoom alone. It is a known
trick, not Bantam's. It is not the default yet (`--qk_norm 0`), so the
published baselines do not use it.

`generate()` decodes with a KV cache (greedy, temperature, top-k, top-p,
repetition penalty, per-row EOS). In a hand-timed development check on the
main config with 1 thread (not committed as a result file) it produced the
same tokens as the no-cache reference at 351 vs 59 tokens/s.

---

## 12. Checkpoints, resume and saved models

### 12.1 Language-model pretraining

- `--ckpt_every N` writes a resumable checkpoint every N steps to `--ckpt`
  (default `<out>/<name>.ckpt`). `--resume PATH` continues from one.
  `--stop_step N` stops early (for tests and for splitting a run).
- A checkpoint holds the weights, the optimizer state, the data stream position
  and epoch, budget progress (model, optimizer and exact FLOPs, tokens, step,
  time, next eval), the log so far, the torch RNG, the Zoom mixing RNG, and on
  CUDA the CUDA RNG and GradScaler state.
- It is written to a temporary file and renamed, so a crash never leaves a
  half-written checkpoint.
- It contains only plain tensors and containers. 8-bit state is stored as codes
  plus fp32 scales. So it loads with `torch.load(weights_only=True)`. Pre-0.3
  checkpoints that pickled `QState` objects load with that one class
  allow-listed, with a warning. Those written before the package was renamed
  need PyTorch 2.6 or newer for this.
- Resume refuses changes that would change the run: model options, sequence
  length, seed, budget, zoom and batch scaling, optimizer, state bits, weight
  dtype, LR and schedule options, weight decay, orthogonalizer, eval count,
  FLOP mode, `batch × accum`, the world size and the training data size.
  `--allow_arg_change` overrides. Changing only the split between `--batch`
  and `--accum` is allowed with a warning.
- Resume on CPU is bit-exact (tested: the train losses, every eval point and the
  FLOP counters match an uninterrupted run).

### 12.2 Weather and fine-tuning

- `bantam weather` has the same `--ckpt_every`, `--ckpt`, `--resume` and
  `--stop_step`, and refuses changes to the model, budget, FLOP mode, batch,
  batch scaling, LR and schedule options, orthogonalizer, `--mix_fine`, weight
  decay and seed. `--save PATH` writes the weights with their config and
  normalization statistics. `bantam weather --forecast --load PATH --init
  2018-03-01T00 --steps 12` rolls a saved model forward and writes the
  forecast in physical units.
- `bantam finetune --save_every N` writes `<out>/checkpoint/state.pt`
  (trainable parameters, optimizer, step, every rank's data order and RNG,
  GradScaler, log). `--resume DIR` continues and refuses changed model, data,
  schedule, weight-decay, orthogonalizer or dropout options and a different
  world size. Outputs are `adapter.pt` plus `adapter_meta.json`, a PEFT
  adapter, and the tokenizer, all a few MB; `--save_merged 1` adds a merged
  model.

### 12.3 Saved language models

`bantam pretrain --save PATH` writes, before the final evaluation:

- `PATH`: the raw `state_dict` (with the zoom modules).
- `<root>_pretrained/`: `config.json` (constructor arguments, dtype, EOS id),
  `model.safetensors` (or `model.pt` without safetensors), and the data
  directory's `tokenizer.json`. The zoom modules are stripped.

`bantam generate --model <root>_pretrained --prompt "Once upon a time"` samples
from it. `ZoomGPT.from_pretrained` rebuilds the model on the meta device and
loads the weights in place (no second copy).

A run whose loss becomes non-finite stops at once, writes its log with status
`diverged`, saves no model and exits with code 3.

---

## 13. Runtime details

- **Flush-to-zero on CPU.** Denormal numbers in gradients slow CPU matmuls
  sharply. `bantam pretrain`, `weather` and `finetune` call
  `torch.set_flush_denormal(True)` at start-up. A hand-timed check on this CPU
  found a 15-23x slowdown from denormals in one large matmul; that measurement
  is not committed as a result file. Note: importing `bantamtrain.finetune`
  turns flush-to-zero on for the whole process.
- **CPU precision.** Bantam never uses autocast on CPU, so a default CPU run
  is fp32 throughout. With `--param_dtype bf16` the weights are bf16, so the
  forward and backward run in bf16 arithmetic (the loss and the evaluation are
  computed in fp32). On the sandbox that was 1.6-2.7x slower than fp32 (§3).
  The run logs do not record the CPU model; the CPU of the 0.3 sandbox reports
  `avx512_bf16` and `amx_bf16`, the 2026-10 development box did not.
- **GPU path (untested here).** Precision `auto` picks bf16 autocast on compute
  capability 8.0 and above, fp16 with a GradScaler below. TF32, fused AdamW,
  `--compile` and SDPA are used when available. `bantam pretrain --auto_batch`
  splits `batch × accum` into the largest micro-batch that keeps peak memory
  under 85% of the card. `--duty` caps the average load. None of this has run on
  a GPU.
- **Several processes.** `torchrun` starts DDP. Ranks share one data order and
  read disjoint slices; the global batch is `batch × accum × WORLD_SIZE`. This
  is tested with CPU processes only.

---

## 14. What has not been measured

- Anything on a GPU: speed, memory, `--auto_batch`, mixed precision, compile,
  multi-GPU training.
- The 5B-on-one-96 GB-GPU scoreboard ([SCOREBOARD_5B.md](SCOREBOARD_5B.md) and
  `bantam memory`) is arithmetic, not a measurement. Bantam has no FP8 code
  path, so the scoreboard's FP8 row is a literature-based estimate, and the
  chunked loss its later rows assume cannot be switched on from
  `bantam pretrain`.
- Zoom above 5.8M parameters, on other data, against sequence-length warm-up at
  equal FLOPs, or in wall-clock.
- Mean merge and bag loss under the `k = 2`, batch × 2 recipe.
- Muon with a separate (AdamW-tuned) LR for the embedding/head.
- The Polar Express, 8-bit and bf16 runs with 0.3 numerics (the committed
  ones used the 0.2 coefficient list, log map and bf16 evaluation).
- A tuned LR for Muon on LoRA matrices.
- Weather Zoom with batch scaling, and weather results on a held-out
  validation year.
- Fine-tuning speed against other tools: the fine-tuning race (Hugging Face
  Trainer + PEFT vs Bantam on SmolLM2-135M) is described in
  [COMPARISON.md](COMPARISON.md).

---

## 15. Where the code lives

| technique | code |
|---|---|
| Zoom merge/split, schedules, LR schedule | `bantamtrain/zoom.py` |
| Small GPT, FLOP count, generate, save/load | `bantamtrain/models/gpt.py` |
| Weather ViT | `bantamtrain/models/vit_weather.py` |
| Muon, Polar Express, AdamWLowMem, optimizer FLOPs, parameter routing | `bantamtrain/optim.py` |
| 8-bit state, bf16 stochastic rounding, checkpoint loader | `bantamtrain/lowmem.py` |
| int8/int4/NF4 frozen weights, streaming loader | `bantamtrain/qweights.py` |
| LoRA, DoRA, PEFT save/load | `bantamtrain/lora.py` |
| Chunked cross-entropy | `bantamtrain/losses.py` |
| Fine-tuning: data, packing, masks, training loop | `bantamtrain/finetune.py` |
| Pretraining and weather training loops | `bantamtrain/cli/pretrain.py`, `bantamtrain/cli/weather.py` |
| TokPack, dedup, GridPack | `bantamtrain/data/tokpack.py`, `bantamtrain/data/dedup.py`, `bantamtrain/data/gridpack.py` |
| Runtime (FTZ, precision, batch finder, duty cycle) | `bantamtrain/device.py` |
| Memory scoreboard | `bantamtrain/memory.py` |
| Weight files (WeightPack) | `bantamtrain/weightpack.py` |
| Result tables, compute-to-target | `experiments/summarize.py` |
