# Zoom Training — method

## 1. The problem it attacks

Training cost for a transformer is dominated by the body (attention + MLP blocks),
and the body's cost is linear in the number of *positions* it processes
(quadratic for the attention part). For a language model every token is a
position; for a weather ViT every spatial patch is a position. The usual way to
cut cost is to process fewer positions: fewer tokens (data subsampling),
shorter sequences, or coarser grids (lower resolution). All of these **discard
information**: a token or a grid cell the model never sees cannot shape its
weights.

Zoom Training keeps every input element and every target element in every
stage of training. What changes over training is only the *internal
processing resolution* of the body.

## 2. The mechanism

Let the input be a sequence of `N` elements (`N` tokens, or `N` fine patches
of a grid) with embeddings `x_1 … x_N ∈ R^d`. At zoom level `k`:

1. **SlotMerge.** Group neighbouring elements into `N/k` groups of `k`
   (consecutive tokens; `√k × √k` blocks of patches). Each group `p` is merged
   into one position

       z_p = Σ_{j=1..k} g_j ⊙ x_{p,j}            g_j ∈ R^d  (learned, init 1/k)

   The per-slot gate `g_j` is what distinguishes this from averaging: the
   merged vector keeps enough information about *which slot* each element
   occupied that the model can tell "the cat sat" from "sat the cat". Cost:
   `k·d` parameters per zoom level.

2. **Body.** The unchanged transformer body runs on the `N/k` merged
   positions. For a causal LM, RoPE positions are placed at the group's centre
   in token units (`p·k + (k−1)/2`), so the positional geometry the body sees
   is the same one it will see at `k = 1`. For the ViT the learned positional
   embedding is added *before* merging, so it is merged with the content.

3. **SlotUnmerge.** Each coarse hidden state `h_p` is expanded back to `k`
   per-slot states

       y_{p,j} = h_p ⊙ u_j                       u_j ∈ R^d  (learned, init 1)

   and each `y_{p,j}` goes through the ordinary output head. For the LM, group
   `p` predicts *all k tokens of group p+1* (slot `j` predicts token
   `p·k + k + j`) — a multi-token-prediction objective whose supervision covers
   every token of the corpus. For the ViT every fine patch gets its own
   decoded prediction, so the loss is the full-resolution loss.

4. **Annealing.** `k` follows a schedule indexed by the fraction of the FLOP
   budget spent, e.g. `4 : 35 % → 2 : 25 % → 1 : 40 %`. At `k = 1` the merge and
   unmerge modules are bypassed and the model is a completely ordinary
   transformer — no extra parameters or inference cost ship with it.

### Why this should help

* **Tokens per FLOP.** At `k = 4` the body processes 4× more raw tokens per
  unit of compute (the output head, which is per-token, is not reduced in
  `slot` mode). Early training mostly learns unigram/bigram statistics,
  embeddings and the gross shape of attention, which do not need
  token-level resolution; cheap coarse steps buy that knowledge for less.
* **Nothing discarded.** Unlike data subsampling or low-resolution training,
  every token and every grid cell contributes a gradient in every stage, and
  the target distribution is the real one. The gates make the merge reversible
  enough for the fine stage to inherit, rather than relearn, the coarse
  features.
* **Smooth hand-off.** The body's weights, optimizer state and learning-rate
  schedule are continuous across stage changes; only the tiny gate modules
  switch.

### Relation to prior work

* *Patch-level training* (Shao et al., 2024) trains an LLM on averaged
  token embeddings predicting a bag of next tokens, then switches to token
  level. Zoom Training generalises it with slot-gated merge/unmerge (slot
  identity is preserved, every target token gets its own prediction), a
  multi-level compute-indexed anneal (4 → 2 → 1 rather than one switch), and
  the same operator applied to spatial grids for weather models. Both the
  `bag` loss and plain mean are kept as ablations (`--out_mode bag`,
  `--gated 0`).
* *Progressive resizing / low-resolution pretraining* (vision, FourCastNet-style
  coarse-then-fine) trains on *downsampled* data. That is the `--coarse_mode
  lowres` ablation in the weather trainer: it is the natural baseline Zoom
  should beat, because it loses the fine-scale data.
* *Multi-token prediction* (Gloeckle et al., 2024) is what the slot-unmerge
  head reduces to at the end of the coarse stage.

## 3. Compute accounting

Every run has a FLOP budget; a step at zoom `k` is charged its analytic
training FLOPs (3 × forward):

    LM:   3 · B · (N/k) · [ L · (24 d² + 2 (N/k) d) + 2 d V · k ]
    ViT:  3 · B · [ (N/k) · L · (24 d² + 4 (N/k) d) + N · 2 d p² (C_in + C_out) ]

The zoom and baseline runs stop at the same FLOP total and are evaluated
identically at `k = 1`. Wall-clock is reported too; coarse steps actually run
*faster per FLOP* on CPU (better matmul shapes), so the wall-clock gain is
larger than the FLOP gain.

## 4. What is measured

* **LLM:** TinyStories (HuggingFace `roneneldan/TinyStories`), 4 096-token BPE,
  5.8 M-parameter decoder (d = 256, 6 layers). Metric: validation next-token
  loss at `k = 1` on 2 048 held-out sequences. Baseline vs Zoom at equal FLOPs,
  for AdamW and for Muon; ablations (gates, bag loss, RoPE placement,
  schedule); and *compute-to-target*: how many baseline FLOPs reach the Zoom
  run's loss.
* **Weather:** ERA5 5.625° (WeatherBench2, 64 × 32), six channels (z500, t850,
  t2m, u10, v10, msl), 6-hour residual prediction with a 1 M-parameter ViT
  (2 × 2 patches, d = 128, 6 layers). Metric: latitude-weighted RMSE in
  physical units at 6 h, 24 h and 72 h (autoregressive rollout) on 2018, vs
  persistence. Baseline vs Zoom (2 × 2 token merge) vs progressive resizing.

## 5. The rest of the toolkit

Zoom Training is the compute lever. The other levers in the package:

| Lever | What it does | Where |
|---|---|---|
| Muon | Orthogonalised momentum for the body's weight matrices, AdamW elsewhere; same LR/WD thanks to Moonlight RMS matching | `zoomtrain/optim.py` |
| FTZ on CPU | Denormal gradients made CPU matmuls ~15× slower; flushing them is free | `zoomtrain/device.py` |
| GPU path | bf16/fp16 autocast, TF32, fused AdamW, `torch.compile`, SDPA; memory-headroom batch finder and a duty-cycle limiter so the card is not pinned at 100 % | `zoomtrain/device.py` |
| TokPack | Random-access compressed token store: frequency-rank remap → byte-shuffle → zstd, exact round-trip | `zoomtrain/data/tokpack.py` |
| Dedup | Exact + MinHash-LSH near-duplicate removal on the token stream, vectorised | `zoomtrain/data/dedup.py` |
| GridPack | Error-bounded int16 quantisation (≤ 5 % of the 6 h change std) + temporal delta + zstd for gridded fields; avoids float16's precision cliff on geopotential/pressure | `zoomtrain/data/gridpack.py` |
