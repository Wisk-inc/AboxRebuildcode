# zoomtrain: research foundation

*Status: 2026-10-15. Audience: engineers implementing zoomtrain (LLM and ERA5 weather training on tiny compute).*

**How to read this.** A verifier pass checked every number here against its primary source. Where the verifier corrected a number, only the corrected value appears. Claims it could not confirm are marked **UNVERIFIABLE**, and claims it refuted are left out. "Inference" marks our own reasoning, not a sourced result. "Local" means measured on the project's 4-core dev box and re-measured by the verifier. That box has `avx512f` and `avx512_vnni` but no `avx512_bf16`, so CPU training there uses fp32. Citations [n] link to §7.

## 1. Executive summary

| # | Technique | Domain | Expected gain (evidence) | Effort |
|---|---|---|---|---|
| 1 | Muon on hidden 2-D matrices; AdamW on embeddings/head/norms/scalars | LLM, ViT | ~1.3-1.4x fewer steps than a *tuned* AdamW at 0.1B, ~1.1x at 1.2B [[1](#ref-1)]. Speedrun record 31.4 -> 24.9 min (1.26x) [[2](#ref-2)] | ~40 LoC |
| 2 | **Zoom / coarse-to-fine**: merge K tokens or 2x2 patches early, then anneal to K=1 | both | LLM: 0.5x compute at equal PPL, 370M-2.7B (K=4, λ=2/3) [[3](#ref-3)]; TST: 1.8x (3B), 2.58x (10B-A1B) [[4](#ref-4)]. Grids: WeatherNext 3 runs 85% of steps at 1° [[5](#ref-5)]; Multigrid 4.5x wall-clock at +0.8 pt [[6](#ref-6)] | 20-100 LoC |
| 3 | Budget-matched LR: no warmup (with zero-init), flat, then linear decay over the last 40-55% to a 0.1-0.15 floor; or constant + 5% 1-sqrt cooldown | both | ResNet-18 at a 10% budget: 62.3 vs 59.5 top-1 vs step decay [[7](#ref-7)]. Weather Swin: constant + cooldown beats cosine at every stop point [[8](#ref-8)] | 5 LoC |
| 4 | Speedrun bundle: QK-norm + ReLU² + zero-init output projections, softcap 15, value residual | LLM | Record 5: -32% wall-clock [[9](#ref-9)]; softcap 15: -6.7% steps [[10](#ref-10)]; value residual: equal loss with 20.3% fewer tokens [[11](#ref-11)] | ~30 LoC |
| 5 | Per-source MinHash dedup + fastText top-10-30% filter + ≤4 epochs | LLM | DCLM 1B-1x Core 30.2 vs 27.5 unfiltered [[12](#ref-12)]; FineWeb-Edu needs "almost 10x" fewer tokens to 33.6% MMLU [[13](#ref-13)]; ≤4 epochs ≈ fresh data [[14](#ref-14)] | CPU minutes |
| 6 | Token store: uint16 in RAM + one numpy gather; rank-remap + LEB128 + zstd at rest; drop byte-shuffle | LLM | Local: 139-156 Mtok/s vs 10.1 for per-batch memmap; 0.751 B/tok vs 2.0 (2.66x). TokPack's byte-shuffle costs 14.5-25% [[15](#ref-15), [16](#ref-16)] | ~60 LoC |
| 7 | Residual target (Δ / std of Δt-differences) + lat × level × variable weighted loss | weather | Free; the model starts at persistence skill. Standard in GraphCast, Stormer, ArchesWeather, Otter [[17](#ref-17), [18](#ref-18)] | ~20 LoC |
| 8 | Sparse-Reslim: random 25% of tokens through the middle blocks, scatter-add the residual delta | weather ViT | 2.46x faster at 1.0°, 2.2-2.4x less memory, Z500 120 h RMSE -11.6%. Measured only at ≥8k tokens [[19](#ref-19)] | ~10 LoC |
| 9 | Late, short rollout fine-tune (K=2->4) at 100-1000x lower LR, one Δt per sequence, or pushforward | weather | ArchesWeather: 20k of 320k steps [[20](#ref-20)]. A naive 6 h iterative CNN reached Z500 1114 at 3 d, worse than climatology [[21](#ref-21)] | ~30 LoC |
| 10 | Precision hygiene. GPU: bf16 autocast + TF32 `high` + SDPA `is_causal` (no mask tensor) + `compile(dynamic=False)`. CPU: fp32 unless `avx512_bf16`/AMX. Always normalize in fp32 before a 16-bit cast | both | TF32: 0.11 -> 0.016 s per large A100 GEMM [[22](#ref-22), [23](#ref-23)], GPT-2 124M 1000 -> 333 ms/step (Karpathy's GPT-2 reproduction); compile: +43% average training [[24](#ref-24)]; CPU bf16 without hardware support is 3-4x slower (oneDNN docs), 6.4x measured on an i7-12700K [[25](#ref-25), [26](#ref-26)] | 10 LoC |

Cross-cutting rule: report a speedup as compute (FLOPs or wall-clock) to a fixed loss, against a tuned baseline, over ≥5 seeds. Per-run val-loss σ at 124M is 0.0014-0.0019 nats [[2](#ref-2)], larger than many individual tricks.

## 2. LLM training

### 2.1 Muon

**Reference algorithm** [[27](#ref-27), [28](#ref-28)]
- Nesterov momentum 0.95, then a 5-step quintic Newton-Schulz iteration with (a,b,c) = (3.4445, -4.7750, 2.0315).
- Normalize X/(‖X‖_F + 1e-7) and transpose if rows > cols. Run NS in bf16 on GPU.
- Output singular values are roughly U(0.5, 1.5) (muon.py docstring). The blog's tuning target is [0.7, 1.3].
- Overhead ≈ T·m/B (T NS steps, width m, B tokens per batch): 0.7% at m=768, B=524,288.
- Routing: Muon for `p.ndim ≥ 2` inside blocks, with Q, K, V orthogonalized separately and conv kernels flattened to [out, in·k·k]. AdamW for embeddings, lm_head, norms, biases and scalars.
- `None` grads are replaced by zeros, so momentum still decays.

```python
def ns5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()   # fp32 on CPU (no bf16 fast path here)
    X = X / (X.norm() + eps)
    tall = X.size(-2) > X.size(-1)
    if tall: X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X
# buf.lerp_(g, 1-mom); u = ns5(g.lerp(buf, mom)); u *= max(1, rows/cols)**0.5   # 'original' scale
# p.mul_(1 - lr*wd); p.add_(u, alpha=-lr)
# cubic5 (10 matmuls instead of 15): X = a*X + b*(A@X) with (a,b) in
#   (3.3656576,-3.3420992),(2.5744352,-1.4957376),(2.5368962,-1.4312570),(2.4418906,-1.2764040),(2.2230472,-0.9630650)
```

**Update-scale conventions (the most common bug).** `torch.optim.Muon` (PyTorch 2.9+) [[29](#ref-29)] has three `adjust_lr_fn` options:
- `original`: scale sqrt(max(1, A/B)), lr ≈ 0.02-0.05.
- `match_rms_adamw`: scale 0.2·sqrt(max(A, B)); reuse the AdamW lr and wd [[30](#ref-30)].
- `spectral_unclamped`: scale sqrt(A/B).

They convert as lr_original = 0.2·lr_matchrms·sqrt(min(m, n)). At d=768, 0.02 corresponds to 3.6e-3. Log the update RMS (target ~0.2, inside AdamW's 0.2-0.4) and store all matrices in one [out, in] orientation. Moonlight uses decoupled wd 0.1 and reached AdamW's loss with 52% of the FLOPs against its own AdamW baseline [[30](#ref-30)].

**Honest expectations**
- Against a well-tuned AdamW [[1](#ref-1)], matrix optimizers give 1.4x at 0.1B and 1.1x at 1.2B. Muon is fastest at 1-4x Chinchilla and is overtaken by SOAP/Kron at 8-16x. Only its lr is scale-sensitive. The tuned AdamW at 130M used lr 0.008, wd 0.1, warmup 2000, betas (0.9, 0.98), eps 1e-10.
- Essential AI [[31](#ref-31)]: 10-15% fewer tokens and a larger critical batch, so use the largest batch that fits.
- muP-style transfer [[32](#ref-32)] (per-layer lr ∝ sqrt(d_out/d_in), independent wd ∝ 1/width) keeps ~1.4x from 190M to 1.4B. Scaling both lr and wd as 1/width drops the gain below 1.1x.
- The blog's "1.35x" speedrun figure is the author's own statement. The record table ratio is 1.26x.

**Cheaper or better orthogonalizers**
- Polar Express [[33](#ref-33)] uses per-step minimax coefficients at the same matmul cost. The first five tuples are (8.28721201814563, -23.595886519098837, 17.300387312530933), (4.107059111542203, -2.9478499167379106, 0.5448431082926601), (3.9486908534822946, -2.908902115962949, 0.5518191394370137), (3.3184196573706015, -2.488488024314874, 0.51004894012372), (2.300652019954817, -1.6689039845747493, 0.4188073119525673). Divide a, b, c by 1.01, 1.01³, 1.01⁵ for every tuple except the last; the speedrun uses 1.02. Normalize X/(‖X‖·1.01 + 1e-7).
  - GPT-Small val loss at 1B tokens: 3.639 (Jordan) -> 3.588. Best lr was 0.005-0.01 at GPT-2 Small; 0.02 was best only for GPT-2 Large. The paper does not state "2x faster than NS".
- cubic5 [[34](#ref-34)]: 10 matmuls instead of 15, within ~1e-3 loss (GPT-2 Small, 5 steps: 3.2811 vs 3.2822). Four steps are about as good as five.
- Periodic Row-wise Muon [[35](#ref-35)]: full NS every K=3 steps, otherwise 0.15 × row-normalized momentum. -47..-54% optimizer time and -16..-24% step time, within 0.5% of Muon at 1.3-4B.

**Add-ons**
- NorMuon [[36](#ref-36)]: per-row second moment (β2 0.95) plus a 0.2·sqrt(mn)/‖Ô‖_F rescale. Saves 21.7% of steps vs Muon's 10.4% at 1.1B, at +2.9% step time.
- Weight decay:
  - Muon-SW (wd ∝ η_t²/η_max): 21.7-29.4% fewer steps [[37](#ref-37)].
  - Hyperball norm constraint: 20-30% token-equivalent at ≤1.2B [[38](#ref-38)].
  - Speedrun cautious WD [[2](#ref-2)]: mask = (update·param ≥ 0) and decay lr_t·1.2 folded into the update. Net shrink is ≈ lr_t·param_lr·(lr_t·1.2); a naive `p.addcmul_(p, mask, -lr*wd)` is ~33x too strong.
- ViT-B/16 [[39](#ref-39)]: Muon lr 1e-3 vs AdamW 3e-4 gives 81.20 vs 66.18 IN-100 top-1 with heavy augmentation, but only 0.77-0.90x throughput on one H100.

**Skip below ~1B:** QK-Clip (K2 needed τ=100 at 9B active; logits exceeded 1000 [[40](#ref-40)]) and Dion (14.5% slower than Muon on one node [[41](#ref-41)]).

### 2.2 Speedrun architecture tricks

GPT-2 124M to 3.28 FineWeb val loss on 8xH100 went from 45 min to 0.665 min (record 92, Aug 2026): 67.7x in wall-clock and ~31x fewer tokens (10.26B -> ~0.33B) [[2](#ref-2)]. Records need a one-sample t-test at p<0.01. Port the transferable class of tricks; FP8, FlashAttention-3, FlexAttention and comms tricks are tied to the hardware. On 2x4090 the per-trick percentages carried over [[42](#ref-42)]: Muon -39.7%, doc-masked 32k sequences -36.4%, RoPE + ReLU² + trapezoid -7.6%, softcap -5.9%.

| Trick | Spec | Measured |
|---|---|---|
| QK-norm + ReLU² + zero-init `c_proj` | Parameter-free RMSNorm on q, k per head. Record 5 applies it after RoPE; norm-then-RoPE from record 14 | 22.3 -> 15.2 min. QK-norm mainly allows a higher LR (record 11 doubled it) [[9](#ref-9)] |
| Fixed attention scale | 0.12 instead of head_dim^-0.5 | -20 steps, zero overhead [[2](#ref-2)] |
| Logit softcap 15 | `30*sigmoid(logits/7.5)` in fp32 | 1490 -> 1390 steps; the README notes it helps "in the small-scale regime" [[10](#ref-10)] |
| Value residual / value embeddings | v = (1-λ)v + λ·v1, λ init 0.5 (record 9). Embedding tables: record 14, 4.66 -> 4.41 min; record 17 uses first-3 + last-3 layers | ResFormer: 16.1% fewer params or 20.3% fewer tokens at equal loss [[11](#ref-11)] |
| Skip / mixing scalars | Record 45: one sigmoid-gated skip 4 -> 7 (logit init -1.5), mixing λ init 1.1 | Removing the skip costs +0.0025 [[2](#ref-2)] |
| Embed/head tying | Untie at 2/3 of training: copy lm_head weights and Adam state into embed | Record 53: -80 steps together with MTP [[43](#ref-43)] |
| LR schedule | No warmup; cooldown fraction 0.4 -> 0.45 -> 0.55; floor 0.1 (record 19), 0.15 (record 72) | -1.2 s from the floor change [[44](#ref-44)]. "The 3.28 bar is set almost entirely by the last 200 steps" [[45](#ref-45)] |
| Momentum | Warmup 0.85 -> 0.95 over 300 steps; 50-step cooldown since record 39 | ~1% [[2](#ref-2)] |
| Batch ramp | 131k -> 262k -> 393k tokens/step in thirds; LR ×1.0/1.51/1.93 = (batch ratio)^0.6, *not* sqrt | -65 steps, -1.4% [[46](#ref-46)] |
| Window / sequence-length warmup | 3-stage window staircase; seq 896 for the first third, then 2048. A middle stage below 1920 missed the target | Record 13: 5.03 -> 4.66 min; record 72: -1.2 s [[44](#ref-44)] |
| Gates | Zero-init Linear(12 -> n_heads) sigmoid gate on the attention output | 1750 -> 1695 iterations [[2](#ref-2)] |
| Annealed parameter-free MTP | Same logits scored on x_{t+1..t+3} with weights [1, .5, .25->0] -> [1, .5->0] -> [1] | Removing it costs +0.0066 val loss, the largest single contributor in record 53 [[47](#ref-47)] |

Pitfalls:
- Every shape change retraces `torch.compile`. A 3-stage curriculum raised σ from 0.0014 to 0.0023 [[45](#ref-45)].
- Slow-state optimizers (AdEMAMix, schedule-free, LAWA) all lost to the hand-tuned linear cooldown at ~1500 steps [[45](#ref-45)].
- FP8 lm_head (record 19) is Hopper/Ada only; nanochat measured ~17% more tok/s but only ~5% net [[48](#ref-48)].

### 2.3 Coarse-to-fine pretraining and Zoom Training

**What Shao et al. did** [[3](#ref-3), [49](#ref-49)]
1. Input: mean of K=4 consecutive token embeddings into one position, E(p_i) = (1/K)Σ_k E(x_{iK+k}), with positions indexed over patches.
2. Output: the *unchanged* transformer and single LM head emit one logits vector per patch, scored by CE against each of the K next-patch tokens (sum ÷ K). This is a bag loss with no order inside the patch.
3. Schedule: patch level for the first λ=2/3 of the data. Then load weights only, start a fresh AdamW and a fresh cosine schedule, and train token level on the remaining 1/3.

Cost is λ/K + (1-λ) = 0.5. On the Pile (360B tokens), PPL was 10.7 vs 10.9 (370M), 9.1 vs 9.2 (780M), 8.2 vs 8.2 (1.3B) and 7.2 vs 7.1 (2.7B). Zero-shot averages improved at all four sizes.
- λ sweep (370M): 1/2 gives 0.625x cost at PPL 10.6; 4/5 gives 0.4x at PPL 11.0.
- The core change is ~10 lines.
- Phase-1 throughput was 3.78x (3.50x at constant tokens/step). The end-to-end 0.523x is an estimate derived from that, not a measured run.

**Failure modes** [[3](#ref-3)]
- K=2 ≈ K=4. K≥8 degrades, and K=8 on twice the data gives no extra benefit. With a 128k vocabulary, K=2 beats K=4.
- λ > 3/4 causes a "rapid decline".
- Input + output projections cut patch PPL from 159 to 87 but *raised* final token PPL from 11.46 to 12.33.
- Patch initialization was worse than the baseline at 45B tokens (2.553 vs 2.526) and better only from 180B (2.413 vs 2.423).
- No (K, λ) scaling law exists.

**Token-Superposition Training (TST)** [[4](#ref-4)] is an independent replication. It uses the same mean-pooled input and a multi-hot bag CE (labels shifted left by s-1). Differences from Shao: one continued WSD schedule (optimizer-state handling is not discussed) and a smaller coarse fraction, r ≈ 0.3 of equal-FLOP steps.
- 270M: 3.212 -> 3.142. 600M: 3.019 -> 2.943. 3B: baseline loss at 1.8x less B200 time. 10B-A1B (s=16, r=0.25): 2.58x less compute, MMLU 39.0 vs 37.4.
- Robust over s ∈ [4, 8], r ∈ [0.2, 0.4] (270M grid 3.139-3.148, best at s=7).
- r=0.9 is worse than baseline (3.222) and r=1.0 gives 5.63.
- Re-initializing embedding and head at the switch: 2.938 vs baseline 2.808 (3B).

**Neighbouring levers**
- Head-based MTP hurts at ≤0.6B (MBPP 0.3B: 1.8 -> 1.0 with n=4) [[50](#ref-50)]. NTP<->MTP curricula never beat NTP at 1.3-3B [[51](#ref-51)].
- Sequence-length warmup: 2.2x fewer tokens and 3.7x less wall-clock for GPT-2 117M. Use token-based LR decay; it pays mainly at ≥1k context [[52](#ref-52)].
- Dataset Decomposition: -20% step time at 8k context, ~0 at ≤512 [[53](#ref-53)].
- Random-LTD: 25.6% wall-clock at 1.3B, 14.3% at 350M; needs ≥24 layers and ≥2k context [[54](#ref-54)].
- Hourglass: 1.87x steps/s at equal bpc, but needs 2 full-resolution layers at each end. Without them bpc is 1.460 vs 1.128 [[55](#ref-55)].
- Learned chunking (H-Net and successors) is unproven at toy scale [[56](#ref-56)].

**How Zoom differs, and what is not yet evidenced**

| Aspect | Shao / TST | Zoom |
|---|---|---|
| Merge | Plain mean of K embeddings; position inside the patch is lost | Slot-gated merge (learned per-slot gates) |
| Output | One logits vector, bag loss over the K next tokens | Slot-gated unmerge: K slot-specific states, ordered targets |
| Schedule | One switch, indexed by data fraction (λ=2/3) or step fraction (r=0.3) | Staircase K = 4 -> 2 -> 1, indexed by consumed compute |
| Domain | Text | Text, and 2x2 patch merge on ERA5 grids |

No published work tests slot-gated merge/unmerge or a 3-stage K anneal for LMs. The closest evidence is:
1. Shao and TST for a 2-stage K -> 1.
2. Record 53's 3-phase annealed MTP, the only published multi-stage anneal, which helps [[47](#ref-47)].
3. Progressive resolution for grids (§3.2).

The main risk is that extra merge/unmerge parameters repeat Shao's projection ablation, which hurt the final model.

**Design constraints for Zoom (derived from the evidence)**
1. Keep coarse compute ≤ ~1/3. Shao spends (2/3)/4 ÷ 0.5 = 1/3 of compute at K=4; TST spends 30%. Untested starting point: K=4 for the first ~20% of FLOPs, K=2 until ~33%, then K=1.
2. K_max = 4, or 2 for vocabularies ≥100k.
3. *Inference:* initialize the gates so that Zoom at K=1 is bit-identical to the token model. Share tokenizer, embedding and head, and never re-initialize at a switch.
4. Never select models or stop early on coarse-phase loss (patch PPL ~150 is normal).
5. At small data, budget for parity at ~0.5-0.6x compute rather than a loss gain.
6. A/B optimizer reset vs continuation at each switch.
7. Pre-tokenize; streaming loaders erase the speedup [[49](#ref-49)].
8. Log FLOPs. *Inference:* the LM head also sees 1/K positions, which matters most for a 5M model with a 4-16K vocabulary.
9. For grids: spend the final ≥25% of compute at the native grid, PI-resize the patch embedding, interpolate the position embedding, keep the batch size fixed, and cut the LR 2-7x at the switch (§3.2).

### 2.4 Data deduplication and selection

| Method | Setting | Measured effect | CPU feasibility |
|---|---|---|---|
| ExactSubstr / NearDup [[57](#ref-57), [58](#ref-58)] | ≥50-token spans (suffix array); 5-gram MinHash, Jaccard 0.8 | Memorized 50-token outputs 1.926% -> 0.138%; 4.6% of C4 val has near-duplicates in train | The dataset must fit in RAM; the suffix array costs ~8x the corpus on *disk* |
| MinHash, FineWeb [[13](#ref-13)] | 5-grams, 14 bands × 8 rows (112 hashes, ~0.75 Jaccard), **per dump** (global gives "little improvement" over none) | Filters: punctuation-line ratio <0.12 removes 10.1%; duplicated-line-char ratio **>0.01** removes 12.5% (the main text's 0.1 is wrong); short-line ratio >0.67. ~22% removed, ~+1% aggregate | Seconds to minutes |
| Repetition law [[14](#ref-14)] | D' = U + U·R*(1 - e^{-(ep-1)/R*}), R* = 15.39 | ≤4 epochs ≈ fresh data; useful up to ~16 | Free |
| fastText quality filter [[12](#ref-12), [59](#ref-59)] | OH-2.5 + ELI5 vs CC, `wordNgrams=2`, keep top 10% | Core 30.2 vs 27.5 unfiltered, 29.0 perplexity filter, 27.1 SemDeDup (1B-1x). At 7B: top 10% 41.0 > 15% 39.8 > 20% 38.7. PreSelect 1B/30B: 33.4 vs DCLM 31.2 vs random@300B 31.3 | Fastest high-value filter |
| FineWeb-Edu ≥3 [[13](#ref-13), [60](#ref-60)] | Linear head on a 109M encoder, Llama-3-70B labels | Keeps ~1.3T of 15T tokens; MMLU 33 -> 37, ARC 46 -> 57 (350B tokens) | Encoder slow on 4 cores; use fastText distillates |
| DSIR [[61](#ref-61)] | Hashed uni+bigrams, 10k buckets, Gumbel top-k | +2.05 GLUE over random. KL reduction correlates r=0.82 with downstream score | Pure CPU |
| RHO-1 [[62](#ref-62)] | Keep top 60-70% of tokens by excess loss vs a reference | Self-reference 23.9 vs 21.5 (24.8 for the intersection variant). <50% kept hurts (20.7) | One scoring pass |
| DoReMi [[63](#ref-63)] | Group-DRO proxy, η=1, c=1e-3, averaged α | 8B: +6.5 pts, baseline reached in 2.6x fewer steps, 8% overhead; ~+2% at equal size | Needs ≥4 domains |
| Perplexity pruning [[64](#ref-64), [65](#ref-65)] | Reference ≤1/5 of model size; keep high/mid 50% | 1.31x/1.45x (Pile 1B/3B), 1.29x/1.14x (Dolma); test PPL gets *worse* (7.83 -> 8.51) | Yin & Rush (a fine-tuning study): compute-optimal only at ≥5x size ratio |
| Curriculum [[66](#ref-66), [67](#ref-67)] | zlib ratio / MTLD / Flesch, 10 bins, linear pacing, constant LR | 29.8-45.3% fewer steps to the baseline peak; +3.5% as warmup. Gone under cosine decay; recovered (+1.64%) with a moderate final LR + checkpoint averaging | Microseconds/doc |

Small-data notes:
- TinyStories: 1M-33M-param models become fluent in ≤30 h on one V100 [[68](#ref-68)].
- Richer text beats simplified text: Gutenberg beats CHILDES/TinyStories at every size tested, and the mixed BabyLM set beats Gutenberg at 18-44M [[69](#ref-69)].
- WRAP synthetic rephrasing is ~3x faster to train on but costs ~25k A100-h per 85B tokens to generate [[70](#ref-70)].
- Strong filters remove ~90% of tokens, so plan epochs accordingly [[71](#ref-71)].
- Skip at tiny scale: SemDeDup, JEST (contrastive only [[72](#ref-72)]), online RHO-LOSS (≈3.3x forward overhead [[73](#ref-73)]).

### 2.5 Storage and loading

- **Format** [[74](#ref-74), [75](#ref-75)]: a flat little-endian uint16 stream with the llm.c header (256 × int32: magic 20240520, version 1, N) and 100M-token shards.
  - Keep vocab ≤65,535 (Megatron switches dtypes at 65,500).
  - Store a uint64 doc-end index for re-shuffling and BOS alignment.
  - modded-nanogpt reads pre-shuffled shards sequentially (4-thread `preadv` into a pinned uint16 buffer), sends inputs as int32 and targets as int64, and does no runtime shuffle.
- **Loader, local, 32×256 batches** [[16](#ref-16), [76](#ref-76)]:
  - In-RAM numpy fancy-gather: 139-156 Mtok/s.
  - nanoGPT per-batch memmap: 10.1 Mtok/s.
  - Sequential slab: 803 Mtok/s.
  - Torch advanced indexing on int16: 1.0 Mtok/s.
  - When a memmap exceeds free RAM it drops from 60 to 3 batches/s (forum report, 90 GB float file) [[77](#ref-77)].
- **Compression, local, 30M TinyStories tokens, 4k vocab, bytes/token** [[16](#ref-16), [15](#ref-15), [78](#ref-78)]:

| uint16 | zstd-3 | zstd-10 | zstd-19 | xz-6 | rank+LEB128 (no codec) | rank+LEB128+zstd-10 | current TokPack (rank+byte-shuffle+zstd-10) |
|---|---|---|---|---|---|---|---|
| 2.000 | 0.989 | 0.819 | 0.710 | **0.660** (decodes at 37-40 MB/s) | 1.362 | **0.751** | 0.938 |

- What the numbers mean:
  - Byte-shuffling a uint16 stream *hurts* (zstd-10: 1.042 vs 0.819) because it splits symbols and breaks LZ matches. Keep byte-shuffle for float grids only.
  - Order-0 entropy is 8.66 bits/token and order-1 is 5.17.
  - zstd decodes at ~170-250 Mtok/s per core (load-dependent), so decompression is free next to any trainer here.
- **Chunks:**
  - Independently decodable chunk size vs B/tok: 4k tokens 1.138, 64k 1.068, 1M 1.000.
  - Random windows read from compressed chunks run at 0.31 Mtok/s (26 ms/batch). Consume chunks sequentially, or inflate everything to RAM at startup.
  - Inflating the 124.6 MB pack takes 0.58 s of zstd but 3.0-3.2 s in TokPack's numpy unshuffle and gather.
- **Web text:**
  - About 4.5 text bytes per BPE token.
  - zstd on 16-bit BPE ids gives 2.31 vs 2.60 bits/char on raw C4 (-11%) [[79](#ref-79)].
  - *Inference:* expect ~1.5-1.7x over uint16 on web text, not the 2.4x seen on TinyStories.
- **Vocabulary** [[80](#ref-80)]:
  - Fitted laws: N_v = 0.20·C^0.42, N_nv = 0.08·C^0.50.
  - Predicted V ≈ 6.7k at 33M non-embedding params (d=512) and ≈11k at 100M (d=768). The 302M optimum is 16k (10k if data-scarce, 24k if over-trained).
  - At 5M params the law gives ≈2.8k, but this is extrapolated outside the fitted 33M-1.13B range.
  - The claim that compressed bytes/char stays flat across vocab sizes (local sweep) is **UNVERIFIABLE** (not re-run).
  - Plain BPE saturates at ~4.45 B/token even at 200k; superword tokenizers only matter at ≥50k [[81](#ref-81)].
- **Pre-tokenize once:** HF BPE-4096 tokenizes at 0.76 Mtok/s on 4 cores [[16](#ref-16)]; Apertus spent ~60 node-hours on tokenization alone [[82](#ref-82)]. MosaicML MDS and LitData solve remote-storage problems; copy their ideas (fixed chunks, offset table, zstd) rather than the libraries [[83](#ref-83), [84](#ref-84)].

### 2.6 GPU and CPU efficiency

- **AMP** [[85](#ref-85)]:
  - bf16 needs no GradScaler. fp16 GradScaler defaults: 65536 / ×2 / ×0.5 / 2000.
  - Autocast keeps softmax, log_softmax, layer_norm and cross_entropy in fp32.
  - On GeForce cards, FP16/BF16 tensor cores with FP32 accumulate run at half rate (4090 spec: 165.2 vs 330.3 dense TFLOPS). *Inference:* expect ~2x from AMP there, not ~3x.
- **TF32** is off for matmuls by default. `torch.set_float32_matmul_precision('high')` takes an A100 GEMM from 0.11 s to 0.016 s [[22](#ref-22), [23](#ref-23)]. In Karpathy's GPT-2 reproduction it took GPT-2 124M from 1000 to 333 ms/step (300 with bf16).
- **`torch.compile`** [[24](#ref-24), [86](#ref-86)]:
  - +43% average training on A100 (+51% AMP, +21% fp32), less on a 3090.
  - After 8 recompiles it silently falls back to eager. Use fixed (B, T), `drop_last=True` and a tensor LR, and check with `TORCH_LOGS=recompiles`.
- **SDPA** [[87](#ref-87)]:
  - Flash needs sm80-sm121, fp16/bf16, head_dim ≤256 and `attn_mask=None`; any mask tensor sends it to mem-efficient.
  - On sm86/89/120/121, training rejects head_dim in (192, 224], and (224, 256] when dropout > 0.
  - The math path is ~38x slower (87.4 ms vs 2.28 ms).
- **Fused AdamW**: +3% in the HF benchmark [[88](#ref-88)], but it can lose to foreach (A40, PT 2.2: 68 vs 85 TFLOPS) [[89](#ref-89)]. Benchmark both.
- **Activation checkpointing**:
  - Full recompute costs 30-40%. Megatron's selective recompute is 29% faster than full (54.2 vs 42.1% MFU) [[90](#ref-90)].
  - PyTorch SAC that saves matmuls + SDPA cuts activation memory ~50% [[91](#ref-91)].
  - Prefer a smaller micro-batch + accumulation first.
- **CUDA graphs** (`reduce-overhead`) help launch-bound small models. They need static shapes and no per-step `.item()`, and each captured launch costs 64 KB of device memory [[92](#ref-92)]. Pinned memory: pin via `DataLoader(pin_memory=True)`; manual `pin_memory().to()` is slower than a plain copy (568 vs 391 ms) [[93](#ref-93)].
- **Max batch** [[94](#ref-94)]:
  - Set `expandable_segments:True`.
  - Accelerate's search starts at 128 and now multiplies by 0.9 on OOM. Accept the largest batch whose `max_memory_reserved` is ≤85% of VRAM.
  - Pad the vocab to a multiple of 128 (~3%).
- **Power**:
  - Capping at 70-80% of the default keeps 93-99% throughput: V100 at 150 W gave -13.7% energy for +6.8% time [[95](#ref-95)]; at 200 W (80%) it ran <5% slower while saving 10-20% energy [[96](#ref-96)].
  - Below ~50% of TDP throughput collapses. Capping is reactive and adds variability; frequency locking is steadier [[97](#ref-97)].
  - Throttling does not show in utilization. Log `clocks_event_reasons` and read the device's "GPU Slowdown Temp" [[98](#ref-98)].
  - MIG exists only on datacenter / RTX PRO cards [[99](#ref-99)].
- **CPU (4 cores)**:
  - Threads = physical cores (PyTorch's default) [[100](#ref-100)]. Use `OMP_PROC_BIND=CLOSE`, `OMP_PLACES=cores`, `num_workers=0`, `set_num_interop_threads(1)` and preload jemalloc/tcmalloc [[101](#ref-101)].
  - Call `set_flush_denormal(True)`; convolutions on denormal weights ran up to 30x slower [[102](#ref-102)].
  - Use bf16 autocast only with `avx512_bf16`/AMX. Measured slowdown without them: 6.4x on an i7-12700K [[26](#ref-26)]; oneDNN states bf16 is 3-4x slower without hardware acceleration [[25](#ref-25)]. Gains with hardware support: 1.4-2.2x on Cooper Lake [[103](#ref-103)].

## 3. Weather (ERA5)

### 3.1 Training recipes

| Model | Grid / size | Compute | Loss | Target / normalization | Rollout fine-tune |
|---|---|---|---|---|---|
| GraphCast [[17](#ref-17)] | 0.25°, 36.7M GNN | 32 TPUv4 × ~4 weeks (Otter converts this to 1344 A100-days) | MSE × cell area × level weight (∝ pressure, unit mean) × surface weights (2T 1.0; 10u/10v/msl/tp 0.1) × 1/Var(Δ6h) | Residual / std of 6 h differences; inputs z-scored per variable-level | 11k steps, 2 -> 12 AR steps (+1 per 1000), fixed LR 3e-7 (peak was 1e-3), backprop through the full unroll |
| Pangu [[104](#ref-104)] | 0.25°, 4 models (1/3/6/24 h) | 192 V100 × ~15-16 days *per model* | L1, surface term ×0.25 | Absolute z-scored state | None; hierarchical temporal aggregation. Not converged at 100 epochs |
| FourCastNet [[105](#ref-105)] | 0.25° AFNO | 64 A100 × 16 h (~43 A100-days) | Loss form not stated in the paper | Per-channel z-score | 2-step fine-tune, 50 epochs from 1e-4 |
| SFNO [[106](#ref-106)] | 0.25°, 433M | 4 h on 8 DGX per stage (GPU type unstated) | Per-channel relative L2 with spherical quadrature | Per-channel | n_steps from 2, 5 epochs each at 1e-5. Stable for a 1,450-step rollout |
| ClimaX [[107](#ref-107)] | 5.625° / 1.40625° ViT, patch 2/4 | 80 V100, fp16 pretrain | Lat-weighted MSE | z-score; lead time conditioned on U[6,168] h (direct) | None (fine-tune LR 5e-7) |
| Stormer [[18](#ref-18)] | 1.40625° ViT-L, patch 2 | 128 A100 × <24 h | Lat-weighted MSE × pressure weights (T2m 1, others 0.1) | Residual with per-Δt stats, Δt ∈ {6, 12, 24} h | K=4 at 5e-6, then K=8 at 5e-7, 20 epochs each |
| ArchesWeather-M [[20](#ref-20), [108](#ref-108)] | 1.5°, 89M 3D-Swin with column attention | 2 A100 × 2.5 days | Density-weighted MSE | 24 h residual, unit variance | 20k steps: K=2 (8k), 3 (8k), 4 (4k). Last 50k pretrain steps on 2007-2018 |
| Aurora [[109](#ref-109)] | 0.25°, 1.3B | 32 A100 × ~2.5 weeks | MAE, no lat weighting; per-variable weights | Absolute | LoRA + pushforward + replay buffer (200/GPU), 13k steps at 5e-5 |
| FuXi [[110](#ref-110)] | 0.25° | 8 A100 × 30 h + ~2 days per cascade member | Lat-weighted L1 | z-score | 2 -> 12 AR steps at 1e-7; 3-model cascade |
| FengWu [[111](#ref-111)] | 0.25° | 32 A100 × 17 days | Gaussian NLL (learned channel weights) | z-score | CPU replay buffer |
| NeuralGCM-2.8° [[112](#ref-112)] | 14.5M hybrid | 16 TPUv4 × 1 day | Filtered MSE + spectrum + bias; (1+τ/24)^-1/2 lead weight | Rescaled by std of 24 h differences | Unroll curriculum 12 -> 72 h |
| GenCast [[113](#ref-113)] | 0.25° diffusion, 12 h step | 32 TPUv5 × ~5 days | EDM-weighted MSE × GraphCast weights | Residual / std_diff (σ_data=1) | None. 2M steps at 1°, then 64k at 0.25° |
| Otter [[114](#ref-114)] | 1.5° Swin U-Net | <3.5 A100-days (XL ~10) | Σ_v sqrt(lat-weighted MSE); 2T ×10; levels ∝ p | Residual | 1 epoch on 2007+, horizon growing 4 -> 12 steps. Muon (2e-4, momentum 0.95, wd 0.15) |
| KAI-α [[115](#ref-115)] | 2.5°, ~7M CNN, daily step | 1 L40s × 12 h | Lat-weighted L2 | z-score | None |
| ResNet [[116](#ref-116)] | 5.625°, 19×128 (param count **UNVERIFIABLE**) | ~1 day on one RTX 2080 | Lat-weighted MSE | z-score; direct per lead | None (CMIP pretrain, fine-tune at 5e-7) |

Common optimizer: AdamW, betas (0.9, 0.95). Global batch is usually 16-32 (1 per accelerator), but not always: Pangu used 192, ClimaX 128 per device. Scaling laws [[117](#ref-117), [8](#ref-8)]:
- L(D) data exponents are 0.34-0.51. At fixed compute, train longer on more data rather than enlarge the model, and go wider rather than deeper.
- Compute-optimal allocation: N* ∝ C^0.59, samples ∝ C^0.41.
- Peak LR falls from 1.5e-3 (≤15M params) to 5e-4 (≥115M).
- Dense models use small GPUs far better: GraphCast reaches 1.03% of H100 peak vs Aurora's 37.2%.

### 3.2 Multi-resolution and progressive training: evidence

**Precedents**
- Multigrid: cycling resolution with batch size plus a final full-resolution stage gave 4.5x wall-clock at +0.8 pt top-1 [[6](#ref-6)].
- FixRes: train at 128, fine-tune the head + BN at 224 to 77.1, matching train-at-224 [[118](#ref-118)].
- EfficientNetV2: progressive resizing gives 84.3 with fixed regularization and 85.1 when regularization grows with input size [[119](#ref-119)]. A uniform linear growth schedule hurt DeiT-S (74.1 -> 72.6) [[120](#ref-120)].

**Patch size**
- FlexiViT: PI-resize keeps accuracy flat across patch sizes; bilinear resize collapses it (52.4 -> 12.8 at /16) [[121](#ref-121)].
- Anagnostidis: a scaling-law-driven patch-size schedule saves up to 60% FLOPs [[122](#ref-122)].
- Stormer: every smaller patch lowered RMSE, so staying coarse permanently costs accuracy [[18](#ref-18)].
- FastDINOv2 switch ablation (low-res/full-res split): 75/25 78.1%, 85/15 76.8%, all low-res 73.9%. Varying the batch size across stages hurt [[123](#ref-123)].

**Production weather**
- WeatherNext 3: 500k of 591k steps at 1°. Fine-stage LR 4.95e-4 -> 7.07e-5; message sums scaled by 1/4 and 1/25 at each refinement [[5](#ref-5)].
- PARADIS v2: 150k steps at 1° (20 GPU-days), then 100k at 0.25° (160 GPU-days). v3 dropped this for native 0.25° I/O with a 0.5° latent [[124](#ref-124)].
- GenCast: 97% of steps at 1° [[113](#ref-113)].
- None of these publish a from-scratch full-resolution ablation.

**Operators and token reduction**
- Zero-shot super-resolution aliases above the training Nyquist (Navier-Stokes losses vary 10x across grids). Mixing resolutions in training fixes it [[125](#ref-125)].
- iFNO mode + resolution growth: 27% faster, 0.961 vs 0.988 error [[126](#ref-126)].
- Sparse-Reslim keep ratio at 1.0° (Z500): r=0.10 gives 801.5 at 2.91x, r=0.25 gives 768.3 at 2.46x, dense is 868.7. A random selector (768.3) beat learned (779.8) and top-k (784.1) selectors [[19](#ref-19)].
- ToMe: ~1.5x faster DeiT-S training, but small gains at 512 tokens [[127](#ref-127)].

### 3.3 WeatherBench at 64×32 (5.625°): reference numbers

**Conventions**
- Latitude weight w_j = cos(lat_j)/mean(cos). On the pole-free 64×32 grid this is analytically identical to WB2's sine-difference cell areas (min 0.0770, max 1.5683).
- WB1 RMSE takes the sqrt per forecast, then averages; WB2 takes the sqrt outside the time-space mean. The gap is 0.3-0.8% for persistence but 1.4/2.5/3.7% (Z500/T850/T2M) for annual-mean climatology [[21](#ref-21), [128](#ref-128)].
- WB1 split: train 1979-2015, val 2016, test 2017-18.
- WB2 protocol: test 2020, 00/12 UTC inits, 6 h lead steps. WB2's own GraphCast was trained on 1979-2019, and its 850 hPa wind headline is the vector RMSE.

"Computed" rows were computed by our agents from the WB2 zarr [[129](#ref-129)] and reproduced by the verifier. They use the WB1 convention, all 6-hourly inits, test 2017-18.

**Z500 RMSE (m²/s²)**

| Method | 6 h | 24 h | 72 h | 120 h |
|---|---|---|---|---|
| Persistence (computed; test 2020 00/12 UTC: 219/576/924/1019) | 218 | 571 | 901 | 998 |
| Persistence, WB1 paper (bilinear regrid, hourly inits) [[21](#ref-21)] | – | – | 936 | 1033 |
| Climatology: day-of-year × hour, 1990-2019 (computed; 2020: 798) | 780 | 780 | 780 | 780 |
| Climatology: annual mean (computed) / WB1 / WB1 weekly | 1041 / 1075 / 816 | | | |
| Linear regression, direct [[21](#ref-21)] | – | – | 693 | 783 |
| CNN, direct, 314k params [[21](#ref-21)] | – | – | 626 | 757 |
| CNN, iterative 6 h [[21](#ref-21)] | – | – | 1114 | 1559 |
| ResNet 19×128, ERA5 only [[116](#ref-116)] | – | – | 314 | 561 |
| ResNet 19×128 + CMIP6 pretrain [[116](#ref-116)] | – | – | 268 | 523 |
| ResNet 28×128, direct [[107](#ref-107)] | 47.0 | 86.6 | 305.2 | 614.2 |
| ClimaX ~108M + CMIP6 [[107](#ref-107)] | 62.7 | 96.2 | 244.1 | 440.4 |
| ClimODE 2.8M (train 2006-15) [[130](#ref-130)] | 102.9 | 193.4 | – | – |
| CoDiCast (train 2006-15) [[131](#ref-131)] | 73.1 | 186.5 | – | – |
| IFS regridded to 5.625° (2018) [[107](#ref-107)] | 26.9 | 51.0 | 152.2 | 331.5 |

**T850 RMSE (K)**

| Method | 6 h | 24 h | 72 h | 120 h |
|---|---|---|---|---|
| Persistence (computed; 2020: 1.17/2.71/4.00/4.27) | 1.19 | 2.69 | 3.91 | 4.18 |
| Persistence, WB1 paper | – | – | 4.23 | 4.56 |
| Climatology: day-of-year × hour / annual (computed); WB1 / weekly | 3.17 / 5.21; 5.51 / 3.50 | | | |
| LR direct / CNN direct [[21](#ref-21)] | – | – | 3.19 / 2.87 | 3.44 / 3.37 |
| ResNet 19×128 ERA5 only / + CMIP6 [[116](#ref-116)] | – | – | 1.79 / 1.65 | 2.82 / 2.52 |
| ResNet 28×128 [[107](#ref-107)] | 0.70 | 1.26 | 1.90 | 2.86 |
| ClimaX [[107](#ref-107)] | 0.88 | 1.11 | 1.59 | 2.23 |
| ClimODE / CoDiCast [[130](#ref-130), [131](#ref-131)] | 1.16 / 1.02 | 1.55 / 1.52 | – | – |
| IFS at 5.625° [[107](#ref-107)] | 0.69 | 0.87 | 1.34 | 2.01 |

**T2M RMSE (K)**

| Method | 6 h | 24 h | 72 h | 120 h |
|---|---|---|---|---|
| Persistence (computed; 2020: 2.62/1.73/2.62/2.90) | 2.51 | **1.73** | 2.59 | 2.86 |
| Persistence, ClimateLearn (hourly inits) [[132](#ref-132)] | 2.76 | 2.13 | 2.99 | 3.26 |
| Climatology: day-of-year × hour / annual (computed); WB1 / weekly [[116](#ref-116)] | 2.33 / 5.66; 6.07 / 3.19 | | | |
| ResNet 19×128 ERA5 only / + CMIP6 [[116](#ref-116)] | – | – | 1.53 / 1.42 | 2.32 / 2.03 |
| ResNet 28×128 (ClimateLearn) [[132](#ref-132)] | 0.72 | 0.94 | 1.50 | 2.20 |
| ViT D=128, depth 8 (ClimateLearn) [[132](#ref-132)] | 0.78 | 1.09 | 1.71 | 2.38 |
| ClimaX [[107](#ref-107)] | 0.95 | 1.10 | 1.43 | 1.83 |
| ClimODE / CoDiCast [[130](#ref-130), [131](#ref-131)] | 1.21 / 0.95 | 1.40 / 1.45 | – | – |
| IFS at 5.625° [[107](#ref-107)] | 0.97 | 1.02 | 1.30 | 1.72 |

**Reading notes**
- T2M persistence is non-monotonic (diurnal cycle). Judge T2M against the day-of-year × hour climatology and 24 h persistence, and feed TOA solar radiation as an input.
- Retrained transformers on 10 years can look worse than persistence: VarteX's ClimaX scored 667 at 6 h, but it was trained at LR 5e-7 and is under-trained [[133](#ref-133)]. Use ClimODE's retrained ClimaX (247.5 at 6 h) as the fair small-data number.
- ClimateLearn: ResNet-iterative degrades fastest. Continuous lead conditioning trails direct models at 6 h-1 d but matches them at 3-5 d.

**Residual-normalization constants** (computed, 2017-2020)

| Field | 6 h Δ std (lat-weighted) | 12 h Δ std | 24 h Δ std | Field std |
|---|---|---|---|---|
| Z500 | 227.6 m²/s² (218.8) | 385.7 | 635.3 | 3371 |
| T850 | 1.24 K (1.19) | 2.05 | 2.97 | 15.3 |
| T2M | 2.35 K (2.54) | 3.26 | 2.10 | 20.7 |

The means of the differences are ≈0. Per-cell 6 h std spans an order of magnitude, so use one scalar per variable-level, as GraphCast and Stormer do.

### 3.4 ERA5 compression

- **Bit information** (Klöwer et al.) [[134](#ref-134)]:
  - Real information is the mutual information of each bit position between neighbouring grid points. Keep the mantissa bits that hold 99% of it; most variables need <7 bits.
  - Bit rounding + zstd (level 10, ≥100 MB/s) gives a 17x geometric mean at 99% (6x at 100%). These factors are **vs float64**; roughly halve them for a float32 baseline. Water vapour: 7 bits at 99%, 39x.
  - Bitshuffle gave no notable gain after rounding.
- **`numcodecs.BitRound`** is integer round-to-nearest, ties-to-even, with max relative error 2^-(k+1): k=5 gives 1.56%, k=7 gives 0.39% [[135](#ref-135)].
  - xbitinfo example keepbits: z 10, u 3, v 2; NetCDF 8.0 MB -> 532 KB [[136](#ref-136)]. Its `get_bitinformation` defaults to the Julia backend; pass `implementation='python'`.
  - *Inference:* compute keepbits on per-level anomalies so geopotential does not need 10 bits.
- **Error-bounded compressors**:
  - Range-relative ε=1e-3, float64 baseline [[137](#ref-137)]: T: SZ3.1 32.3x, SPERR 30.2x, ZFP 5.41x. Z: SZ3.1 80.0x, ZFP 7.09x.
  - On 1.7 TB of float32 ERA5, SZ3.1 reaches 198.8/36.1/11.7x at ε = 1e-2/1e-3/1e-4, compresses at 349 MB/s and decompresses at 1385 MB/s. Learned predictors (CRA5, GraphCast, Aurora) do not beat it [[138](#ref-138)].
  - EBCC: ~35x at 0.5% error (SSIM >0.99995) and ~100x while staying within ERA5 natural variability (SZ3 ~50x). Encoding runs at only 1.4 MB/s [[139](#ref-139)].
- **Safe error budget**: a forecast model trained on CRA5-decoded data (z500 RMSE 11.7 m²/s², t850 0.34 K) was "almost on par" (figure only) [[140](#ref-140)].
- **Validation gate** (ClimateBenchPress [[141](#ref-141)]):
  - Known failures: SZ3 and ZFP silently fill NaNs; ZFP and SPERR do not preserve zeros; bit rounding zeroes subnormals; SZ3/EBCC "relative" bounds are per-chunk-range.
  - Check max|x - x̂|, NaN masks and zero masks using the deployed chunk shape.
  - Bit rounding + Zstd/PCO (~10x) is the robust default.
- **Lossless back end**: pcodec 6.89x vs Blosc+Zstd 2.85x on numeric columns [[142](#ref-142)]. WB2 itself uses Blosc-LZ4 clevel 5 with byte-shuffle, chosen for read speed.
- **16-bit pitfalls**:
  - fp16 max is 65,504, so raw z overflows at ≥~430 hPa (the 400-50 hPa levels) and sp/msl overflow too [[143](#ref-143)].
  - bf16 steps: 512 Pa for msl, 1024 m²/s² for z50, 1 K for T at 250 K.
  - After per-level standardization the relative step is 0.049% for fp16 and 0.39% for bf16, so store fp16 or per-channel int16, not bf16.
  - Per-file int16 packing gives stratospheric q (~3e-6) only ~7 distinct values (step 4.6e-7 when one scale spans all levels).
  - New CDS NetCDF is float32 per ECMWF's spec [[144](#ref-144)]; a forum report of float64 output contradicts it [[145](#ref-145)].
- **Layout**:
  - WB2 64×32 store: 93,544 6-hourly steps (1959-2023), 13 levels, chunks (100, 13, 64, 32) = 10.65 MB float32 [[146](#ref-146)]. One random timestep decodes 100.
  - 240×121 store: chunks (8, 13, 240, 121). ARCO-ERA5 puts all 37 levels in one chunk [[147](#ref-147), [148](#ref-148)].
  - Curate once into a time-major array. Size per channel is 0.77 GB float32 / 0.38 GB int16 for the full record (×0.69 from 1979); a 69-channel int16 sample is 283 KB.
- **Loading**: rechunking to time:1 plus GPU-side decode gave ≥~17x training throughput on one GPU [[149](#ref-149)]. Tuning workers and prefetch cut batch wait from 1.3 s to 7 ms; set prefetch_factor = ceil(t_load/t_step) [[150](#ref-150)].

### 3.5 GPU and mixed precision for weather models

- **GraphCast** [[151](#ref-151), [17](#ref-17)]:
  - fp32 params, bf16 activations. Normalization, residual-target subtraction and x_t + Δ·σ run in fp32 (Bfloat16Cast sits inside InputsAndResiduals).
  - Its loss is actually computed in bf16, with only the batch mean in fp32; an fp32 loss is the safer choice.
  - Per-step remat doubles the backward pass and adds ~50% step time.
- **Aurora** [[152](#ref-152)]: autocasts only the Swin backbone; encoder, decoder and unnormalization stay fp32. A pure-bf16-weights mode "caused serious issues for gradient computation". The v1.5 fp16 variant clamps attention logits in an unfused kernel.
- **FFT**: `torch.fft` has no bf16 path, and fp16 needs CUDA SM53+ with power-of-2 lengths [[153](#ref-153)]. Naive AMP made FNOs output NaN; fixes were fp32 transforms or a tanh pre-activation. bf16 gave significantly higher neural-operator error than fp16 [[154](#ref-154)].
- **Prithvi WxC**: clamp σ to [1e-4, 1e4] (anomalies reached 1e-26); input and output layers in fp32 [[155](#ref-155)].
- **Attention**:
  - On CUDA, flash rejects any `attn_mask`, so window masks run on mem-efficient/cuDNN kernels [[87](#ref-87)].
  - Pangu's Earth-specific bias is 40.4M of 64.2M params. Relative bias cuts parameters ~40-46% at equal val loss [[156](#ref-156)].
  - FlashBias on a Pangu reproduction (inference timing): 98.0 s / 26.6 GB -> 76.8 s / 12.2 GB [[157](#ref-157)].
  - *Inference:* at 64×32 with patch 2 (512 tokens), maskless global SDPA is flash-eligible and simpler than windows.
- **Rollout stability**:
  - Two-year rollouts [[158](#ref-158)]: FourCastNet blows up in 8-9 days, AIFS Z500 in 16 days, GraphCast in 274-470 days; Aurora, SFNO and Pangu stay stable >730 days. A 21M AuroraS lost the seasonal cycle.
  - Keep one Δt per rollout; randomizing it per step destabilized Stormer [[18](#ref-18)].
- **Blurring**: deterministic MSE models keep 24% (Stormer) to 47% (ArchesWeather-Mx4) of energy at the highest wavenumber. ArchesWeatherGen (flow matching) keeps 0.96 and Mosaic (fair CRPS, N=2 members) 0.97 [[159](#ref-159)].
  - FouRKS adds a spectral penalty: λ=0.8 at k ≥ 40 of 180 longitudes, ≈0.44 Nyquist [[160](#ref-160)]. *Inference:* that is k ≈ 14 on 64 longitudes.
- **Latitude weighting**: use sine-difference weights with explicit ±π/2 bounds, because cos(lat) zeroes pole rows on pole-inclusive grids [[161](#ref-161)]. Lat-weighting the loss raised 1-step 2-day Z500 RMSE by 40% *without* channel weights, but only by 4.6% with them; A/B it [[162](#ref-162)].

## 4. Problems and fixes

| Domain | Problem | Known fix | Src |
|---|---|---|---|
| LLM | Muon LR copied across update-scale conventions (10-100x off); transposed storage flips the "tall" rule | Pick one `adjust_lr_fn`; lr_orig = 0.2·lr_rms·sqrt(min(m,n)); log update RMS ≈0.2; fixed [out,in] orientation | [[29](#ref-29)] |
| LLM | Claimed 2x Muon gains vanish against a tuned AdamW | Expect 1.3-1.4x at 0.1B; tune AdamW first; report FLOP-matched curves | [[1](#ref-1)] |
| LLM | NS overhead at small batch or small models (ViT-B ran at 0.77x) | Larger batch, cubic5, Periodic Row-wise Muon (K=3), batched NS | [[34](#ref-34), [35](#ref-35), [39](#ref-39)] |
| LLM | Weight-norm drift under Muon | Decoupled wd 0.1, or schedule-scaled wd; QK-norm | [[30](#ref-30), [37](#ref-37)] |
| LLM | Run-to-run noise (σ ~0.0015 nats) hides tricks | ≥5-8 seeds, t-tests, steps-to-target | [[2](#ref-2), [45](#ref-45)] |
| LLM | Shape changes retrace `torch.compile` (σ 0.0014 -> 0.0023) | ≤3 stages; warm each shape; skip compile on CPU | [[45](#ref-45), [44](#ref-44)] |
| LLM | Value embeddings / untied head dwarf a 5M model | Parameter-free value residual; small vocab; tie until 2/3 | [[11](#ref-11), [43](#ref-43)] |
| LLM | Slow-state optimizers fail in short runs | Muon + trapezoid with linear cooldown | [[45](#ref-45)] |
| Zoom | Coarse-phase loss does not predict final quality; added projections hurt | Keep architecture, embedding and head; never select on phase-1 loss | [[3](#ref-3)] |
| Zoom | Too much coarse training collapses (λ>3/4; r=0.9 worse than baseline) | ≤~1/3 of compute coarse; ≥2/3 at K=1 | [[3](#ref-3), [4](#ref-4)] |
| Zoom | Re-initializing embedding/head at the switch erases the gain | Share them; same tokenizer | [[4](#ref-4)] |
| Zoom | Phase-1 speedup below K (3.5-3.78x of 4x) | Pre-tokenize and pack; no streaming loader | [[3](#ref-3), [49](#ref-49)] |
| LLM | Head-based MTP hurts ≤0.6B | Parameter-free weighted MTP annealed to NTP | [[50](#ref-50), [47](#ref-47)] |
| LLM | Length warmup with step-based LR decay decays too fast | Token-based decay; start at 8, multiples of 8 | [[52](#ref-52)] |
| Data | Global dedup removes good repeated content | Per-source MinHash + classifier filter | [[13](#ref-13)] |
| Data | Val PPL moves the wrong way under good pruning | Select on downstream proxies or KL reduction | [[64](#ref-64), [61](#ref-61)] |
| Data | Curriculum gain lost under cosine decay | Constant LR phase + moderate final LR / checkpoint averaging | [[67](#ref-67)] |
| Data | Eval contamination (4.6% of C4 val) | Dedup train against every eval set | [[57](#ref-57)] |
| Storage | memmap random windows are slow and collapse above RAM | RAM gather, or sequential slabs from pre-shuffled shards | [[77](#ref-77), [75](#ref-75)] |
| Storage | Byte-shuffle before zstd on token ids (+14.5-25% size, 5x slower inflate) | rank-remap + LEB128 + zstd | [[15](#ref-15), [16](#ref-16)] |
| Storage | Per-sample reads from compressed chunks (0.31 Mtok/s) | Chunk-sequential stream or inflate to RAM | [[16](#ref-16)] |
| Storage | uint16 overflow above 65,535 vocab | Assert ids < 2^16; small vocabularies are optimal anyway | [[74](#ref-74), [80](#ref-80)] |
| GPU | Silent eager fallback after 8 recompiles | Fixed (B,T), `drop_last`, tensor LR, `TORCH_LOGS=recompiles` | [[86](#ref-86)] |
| GPU | Any mask tensor disables flash SDPA | `is_causal=True, attn_mask=None` | [[87](#ref-87)] |
| CPU | bf16 without hardware support is several times slower; denormals cost up to 30x | Gate on `avx512_bf16`/`amx_bf16`; `set_flush_denormal(True)` | [[25](#ref-25), [102](#ref-102)] |
| GPU | Throttling invisible in utilization | Log `clocks_event_reasons`; cap at 70-80% | [[98](#ref-98), [95](#ref-95)] |
| Weather | fp16 overflow of raw z/sp/msl; tiny σ blows up gradients | Normalize per variable-level in fp32 first; clamp σ ∈ [1e-4, 1e4]; fp32 input/output layers | [[143](#ref-143), [155](#ref-155)] |
| Weather | bf16 residual add loses sub-K increments | Add x_t + Δ·σ in fp32 | [[151](#ref-151)] |
| Weather | FFT layers under autocast give NaN or no bf16 path | Disable autocast, fp32 transforms | [[153](#ref-153), [154](#ref-154)] |
| Weather | Rollout divergence (iterative CNN worse than climatology by 3 d) | Residual targets; late K=2 -> 4 curriculum at ≤1e-2× LR; pushforward; fixed Δt | [[21](#ref-21), [18](#ref-18), [109](#ref-109)] |
| Weather | MSE blurring (24-47% energy kept at highest k) | Fair CRPS N=2, spectral loss, or keep multi-step fine-tune short | [[159](#ref-159), [160](#ref-160)] |
| Weather | cos(lat) zeroes pole rows; wrong broadcast axis | Sine-difference weights; unit-test row 0 > 0 | [[161](#ref-161)] |
| Weather | Baselines differ across regrid and convention (936 vs 901) | Recompute persistence/climatology on your exact split | [[128](#ref-128)] |
| Weather | WB2 / ARCO chunks force 37-100x read amplification | Curate once to time-major (1, C, H, W) or a memmap | [[146](#ref-146), [148](#ref-148)] |
| Weather | Lossy codecs fill NaNs, drop zeros, use chunk-relative bounds | Post-decode validation gate on deployed chunks | [[141](#ref-141)] |
| Weather | Zero-shot super-resolution aliasing | Train on mixed resolutions; end at the deployment grid | [[125](#ref-125), [118](#ref-118)] |
| Weather | Bilinear patch-embed resize collapses accuracy | PI-resize | [[121](#ref-121)] |
| Weather | Bias tables block flash and add 40-63% params | Coordinate embeddings / relative bias; maskless SDPA | [[156](#ref-156), [157](#ref-157)] |

## 5. Ranked by speedup per effort

**(a) ~5M-param LLM, 4 CPU cores, minutes**
1. Data path: pre-tokenized uint16 in RAM with one numpy gather (~14x vs memmap), `num_workers=0`, 4 pinned OpenMP threads, `set_flush_denormal(True)`, fp32 (this box has no `avx512_bf16`) [[16](#ref-16), [101](#ref-101)].
2. Budget-matched schedule: no warmup with zero-init projections, then a linear warmdown over the last 40-55% to a 0.1-0.15 floor [[7](#ref-7), [2](#ref-2)].
3. Muon with NS in fp32 (cubic5) and Nesterov momentum warming 0.85 -> 0.95; expect ~1.3x [[1](#ref-1), [34](#ref-34)].
4. QK-norm + ReLU² (no erf on CPU) + zero-init + softcap 15 + value residual [[9](#ref-9), [10](#ref-10), [11](#ref-11)].
5. Vocab ~3-4k: the law extrapolates to ~2.8k, so the repo's 4096 default is reasonable [[80](#ref-80)]. Use TinyStoriesV2 or a fastText-filtered, deduplicated corpus for 2-4 epochs [[14](#ref-14)].
6. Zoom K=4 -> 2 -> 1 with ≤1/3 of FLOPs coarse. At this data scale expect parity at ~0.5-0.6x compute, not a loss gain [[3](#ref-3), [4](#ref-4)].
7. Annealed parameter-free MTP in the K=1 phase; batch ramp ×1/×2/×3 with LR multipliers 1.0/1.51/1.93 [[47](#ref-47), [46](#ref-46)].
8. Skip: FP8, flash kernels, `torch.compile` (unless the run exceeds ~10 min), MTP heads, Random-LTD, SemDeDup, JEST, checkpointing.

**(b) ~100M-param LLM, one GPU**
1. bf16 autocast + TF32 `high` + SDPA `is_causal` + `compile(dynamic=False)` on fixed shapes: ~3x from TF32 and ~1.4x from compile on A100, less on GeForce [[22](#ref-22), [24](#ref-24)].
2. Data: fastText top 10-30% (or FineWeb-Edu ≥3) + per-source MinHash + ≤4 epochs (several-x token efficiency) [[12](#ref-12), [13](#ref-13)].
3. Zoom / patch-level training (Shao K=4, λ=2/3, or TST s=6, r=0.3): 0.5x compute at equal loss, verified from 270M to 2.7B [[3](#ref-3), [4](#ref-4)].
4. Muon + Polar Express (or NorMuon) + cautious or scheduled wd: 1.3-1.4x [[33](#ref-33), [36](#ref-36)].
5. Speedrun bundle: trapezoid LR with floor, batch ramp, window/sequence warmup, value embeddings (vocab 8-16k), tie embeddings until 2/3, annealed MTP [[2](#ref-2), [43](#ref-43)].
6. llm.c-format pre-shuffled shards, pinned uint16, int32 inputs [[75](#ref-75)].
7. Max-batch search to ≤85% VRAM with `expandable_segments`; checkpointing only on OOM [[94](#ref-94), [91](#ref-91)].
8. Power cap at 70-80% of TDP: 10-20% less energy for <5-7% time [[96](#ref-96), [95](#ref-95)].

**(c) ~1M-param ViT on 64×32 ERA5, CPU**
1. Convert WB2 64×32 once into a normalized time-major int16/fp16 memmap holding only the channels you use; no loader workers [[146](#ref-146)].
2. Compute persistence and climatology first (§3.3) as unit tests and skill denominators.
3. Residual target with Δ std (Z500 ~228, T850 1.24, T2M 2.35), added in fp32; GraphCast variable and level weights; A/B latitude weighting [[17](#ref-17), [162](#ref-162)].
4. Input stack: t, t-6h, t-12h + TOA solar + land-sea mask, orography and latitude [[116](#ref-116), [132](#ref-132)].
5. AdamW at 1e-3 to 1.5e-3, 500-step warmup, constant LR + 5% 1-sqrt cooldown (or linear to 0) [[8](#ref-8), [7](#ref-7)].
6. A small ViT (ClimateLearn D=128, depth 8, patch 2: T2M 0.78 K at 6 h vs ResNet 0.72) with maskless SDPA, or a periodic-padded CNN for short leads [[132](#ref-132)].
7. Zoom on the grid: first ~60-75% of compute at 32×16 or patch 4 (PI-resized), final ≥25% at 64×32 [[123](#ref-123), [121](#ref-121)]. Or Sparse-Reslim r=0.25, but verify it at 512 tokens [[19](#ref-19)].
8. Muon on qkv/proj/mlp (ViT-B used Muon lr 1e-3 vs AdamW 3e-4) with Periodic Row-wise NS [[39](#ref-39), [35](#ref-35), [114](#ref-114)].
9. Direct models for 1-2 leads, or a short K=2 -> 4 rollout fine-tune at 100x lower LR with fixed Δt; log the high-wavenumber energy ratio [[18](#ref-18), [158](#ref-158)].
10. Skip: checkpointing, bf16 on this CPU, ToMe (<2k tokens), half-precision FFTs.

## 6. Further reading / not yet evaluated

The verifiers flagged these as missing. None of their claims are used above.
- **Optimizers:**
  - Cautious Weight Decay (arXiv 2510.12402).
  - Modular Duality (2410.21265).
  - Scion (2502.07529).
  - CANS (2506.10935).
  - MuonBP (2510.16981).
  - Optimizer benchmark (2509.01440): AdEMAMix and MARS beat Muon at small batch.
  - 8-bit Muon states (2509.23106).
- **Speedrun:**
  - Hashed n-gram embeddings (records 62/92) and Over-Tokenized Transformer (2501.16975).
  - Depth/width pruning (records 30/35/90).
- **Token reduction:**
  - EfficientTrain++ (2405.08768), PatchDropout (2208.07220), NaViT (2307.06304), FLIP (2212.00794).
  - Mixture-of-Depths (2404.02258).
  - BLT (2412.09871).
  - DeepSeek-V3 MTP (2412.19437).
- **Data:**
  - RegMix (2407.01492), CLIMB (2504.13161).
  - DataDecide (2504.11393).
  - Filter × epoch interaction (2404.07177).
  - FineWeb2 rehydration (2506.20920).
  - Best-fit packing (2404.10830).
  - Frequency-ordered tokenization (2602.22958); Stream VByte (1709.08990).
- **GPU:**
  - FlexAttention (2412.05496).
  - Liger fused linear-CE (2410.10989); Cut Cross-Entropy (2411.09009).
  - Zeus power tuning.
  - WSD cooldown origin (2405.18392).
- **Weather:**
  - Keisler (2202.07575), design-space study (2410.07472), DLWP-HPX (2311.06253).
  - AMSE loss (2501.19374), almost-fair CRPS (2412.15832).
  - U-Cast, which uses Muon on ERA5 (2604.09041).
  - Pushforward (2202.03376), PDE-Refiner (2308.05732).
  - ChaosBench (2402.00712), WeatherBench-X.
- **Compression:**
  - Neural ERA5 compression (2210.12538, 2407.11666).
  - netCDF-C built-in BitRound.
  - ERA5 GRIB `bitsPerValue` caps.

## 7. References

<a id="ref-1"></a>[1] Fantastic Pretraining Optimizers and Where to Find Them (2025). https://arxiv.org/abs/2509.02046  
<a id="ref-2"></a>[2] KellerJordan/modded-nanogpt: README and record history (2026). https://github.com/KellerJordan/modded-nanogpt  
<a id="ref-3"></a>[3] Beyond Next Token Prediction: Patch-Level Training for Large Language Models (Shao, Meng, Zhou; ICLR 2025) (2024). https://arxiv.org/abs/2407.12665  
<a id="ref-4"></a>[4] Efficient Pre-Training with Token Superposition (Nous Research) (2026). https://arxiv.org/abs/2605.06546  
<a id="ref-5"></a>[5] WeatherNext 3: Increasing resolution and performance of global weather models with raw observations (2026). https://arxiv.org/abs/2609.03582  
<a id="ref-6"></a>[6] A Multigrid Method for Efficiently Training Video Models (CVPR 2020) (2020). https://arxiv.org/abs/1912.00998  
<a id="ref-7"></a>[7] Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints (ICLR 2020) (2020). https://arxiv.org/abs/1905.04753  
<a id="ref-8"></a>[8] On Neural Scaling Laws for Weather Emulation through Continual Training (2026). https://arxiv.org/abs/2603.25687  
<a id="ref-9"></a>[9] modded-nanogpt record 5 train_gpt2.py (QK-norm, ReLU², zero-init) (2024). https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/train_gpt2.py  
<a id="ref-10"></a>[10] modded-nanogpt record 18 (SoftCap 30 -> 15) (2025). https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-04_SoftCap  
<a id="ref-11"></a>[11] Value Residual Learning (ResFormer) (2024). https://arxiv.org/abs/2410.17897  
<a id="ref-12"></a>[12] DataComp-LM: In search of the next generation of training sets for language models (NeurIPS 2024) (2024). https://arxiv.org/abs/2406.11794  
<a id="ref-13"></a>[13] The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (2024). https://arxiv.org/abs/2406.17557  
<a id="ref-14"></a>[14] Scaling Data-Constrained Language Models (NeurIPS 2023) (2023). https://arxiv.org/abs/2305.16264  
<a id="ref-15"></a>[15] Token-Native Storage: Read and Write in your Agent's Language (2026). https://arxiv.org/abs/2608.02376  
<a id="ref-16"></a>[16] Local measurements on the zoomtrain dev box (TokPack, loaders, tokenizer), re-measured by the verifier (2026). file:///home/user/AboxRebuildcode/zoomtrain/zoomtrain/data/tokpack.py  
<a id="ref-17"></a>[17] GraphCast: Learning skillful medium-range global weather forecasting (Science 2023) (2022). https://arxiv.org/abs/2212.12794  
<a id="ref-18"></a>[18] Scaling transformer neural networks for skillful and reliable medium-range weather forecasting (Stormer, NeurIPS 2024) (2023). https://arxiv.org/abs/2312.03876  
<a id="ref-19"></a>[19] Less Tokens, Better Forecasts: Sparse Residual Routing for Efficient Weather Prediction (Sparse-Reslim, ECCV 2026) (2026). https://arxiv.org/abs/2607.02829  
<a id="ref-20"></a>[20] ArchesWeather: An efficient AI weather forecasting model at 1.5° resolution (2024). https://arxiv.org/abs/2405.14527  
<a id="ref-21"></a>[21] WeatherBench: A benchmark dataset for data-driven weather forecasting (2020). https://arxiv.org/abs/2002.00469  
<a id="ref-22"></a>[22] PyTorch docs: torch.set_float32_matmul_precision (2026). https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html  
<a id="ref-23"></a>[23] PyTorch docs: CUDA semantics (TensorFloat-32 on Ampere and later) (2026). https://docs.pytorch.org/docs/2.14/notes/cuda.html  
<a id="ref-24"></a>[24] PyTorch 2.0 overview: torch.compile benchmarks (2023). https://pytorch.org/get-started/pytorch-2.0/  
<a id="ref-25"></a>[25] oneDNN Developer Guide: Data Types (2026). https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html  
<a id="ref-26"></a>[26] ChimeraX ticket #17555: bfloat16 autocast slowdown on i7-12700K (accessed 2026). https://www.rbvi.ucsf.edu/trac/ChimeraX/ticket/17555  
<a id="ref-27"></a>[27] Muon: An optimizer for hidden layers in neural networks (K. Jordan) (2024). https://kellerjordan.github.io/posts/muon/  
<a id="ref-28"></a>[28] KellerJordan/Muon (muon.py) (2024). https://github.com/KellerJordan/Muon  
<a id="ref-29"></a>[29] torch.optim.Muon, PyTorch documentation (2025). https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html  
<a id="ref-30"></a>[30] Muon is Scalable for LLM Training (Moonshot AI) (2025). https://arxiv.org/abs/2502.16982  
<a id="ref-31"></a>[31] Practical Efficiency of Muon for Pretraining (Essential AI) (2025). https://arxiv.org/abs/2505.02222  
<a id="ref-32"></a>[32] Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales (2025). https://arxiv.org/abs/2512.05620  
<a id="ref-33"></a>[33] The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm (2025). https://arxiv.org/abs/2505.16932  
<a id="ref-34"></a>[34] How Much Orthogonalization Does Muon Need? (2026). https://arxiv.org/abs/2606.00371  
<a id="ref-35"></a>[35] Scaling Muon for Diffusion Transformers (2026). https://arxiv.org/abs/2608.20818  
<a id="ref-36"></a>[36] NorMuon: Making Muon more efficient and scalable (2025). https://arxiv.org/abs/2510.05491  
<a id="ref-37"></a>[37] Scale Weight Decay and Train Better (Muon-SW) (2026). https://arxiv.org/abs/2607.23777  
<a id="ref-38"></a>[38] Fantastic Pretraining Optimizers II: Hyperball Optimization (2026). https://arxiv.org/abs/2606.16899  
<a id="ref-39"></a>[39] Muon in Vision Transformers: Optimizer-Recipe Interactions and Gradient Spectra (2026). https://arxiv.org/abs/2605.24770  
<a id="ref-40"></a>[40] Kimi K2: Open Agentic Intelligence (2025). https://arxiv.org/abs/2507.20534  
<a id="ref-41"></a>[41] Dion: Distributed Orthonormalized Updates (2025). https://arxiv.org/abs/2504.05295  
<a id="ref-42"></a>[42] NanoGPT speedrun worklog on 2x RTX 4090 (T. Romero) (accessed 2026). https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/  
<a id="ref-43"></a>[43] modded-nanogpt record 53 (MultiTokenPrediction; untie at 2/3) (2025). https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-12-22_MultiTokenPrediction  
<a id="ref-44"></a>[44] modded-nanogpt PR #224 (record 72: sequence-length schedule, LR floor 0.15) (2026). https://github.com/KellerJordan/modded-nanogpt/pull/224  
<a id="ref-45"></a>[45] hunting for headroom on modded-nanoGPT (WR #82), djdumpling (2026). https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html  
<a id="ref-46"></a>[46] modded-nanogpt record 46 (BatchSizeSchedule) (2025). https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-11-29_BatchSizeSchedule  
<a id="ref-47"></a>[47] modded-nanogpt PR #178: multi-token prediction and untie LM head at 2/3 (record 53) (2025). https://github.com/KellerJordan/modded-nanogpt/pull/178  
<a id="ref-48"></a>[48] modded-nanogpt record 19 (FP8 lm_head) (2025). https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-13_Fp8LmHead  
<a id="ref-49"></a>[49] shaochenze/PatchTrain (reference code for patch-level training) (2024). https://github.com/shaochenze/PatchTrain  
<a id="ref-50"></a>[50] Better & Faster Large Language Models via Multi-token Prediction (2024). https://arxiv.org/abs/2404.19737  
<a id="ref-51"></a>[51] Pre-Training Curriculum for Multi-Token Prediction in Language Models (2025). https://arxiv.org/abs/2505.22757  
<a id="ref-52"></a>[52] The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models (NeurIPS 2022) (2022). https://arxiv.org/abs/2108.06084  
<a id="ref-53"></a>[53] Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum (NeurIPS 2024) (2024). https://arxiv.org/abs/2405.13226  
<a id="ref-54"></a>[54] Random-LTD: Random and Layerwise Token Dropping Brings Efficient Pre-training for Large-scale Transformers (2022). https://arxiv.org/abs/2211.11586  
<a id="ref-55"></a>[55] Hierarchical Transformers Are More Efficient Language Models (Hourglass) (2021). https://arxiv.org/abs/2110.13711  
<a id="ref-56"></a>[56] Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net) (2025). https://arxiv.org/abs/2507.07955  
<a id="ref-57"></a>[57] Deduplicating Training Data Makes Language Models Better (ACL 2022) (2021). https://arxiv.org/abs/2107.06499  
<a id="ref-58"></a>[58] google-research/deduplicate-text-datasets (2021). https://github.com/google-research/deduplicate-text-datasets  
<a id="ref-59"></a>[59] PreSelect (Shum et al., ICML 2025) (2025). https://arxiv.org/abs/2503.00808  
<a id="ref-60"></a>[60] HuggingFaceFW/fineweb-edu-classifier model card (2024). https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier  
<a id="ref-61"></a>[61] Data Selection for Language Models via Importance Resampling (NeurIPS 2023) (2023). https://arxiv.org/abs/2302.03169  
<a id="ref-62"></a>[62] Rho-1: Not All Tokens Are What You Need (NeurIPS 2024) (2024). https://arxiv.org/abs/2404.07965  
<a id="ref-63"></a>[63] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining (NeurIPS 2023) (2023). https://arxiv.org/abs/2305.10429  
<a id="ref-64"></a>[64] Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models (ICLR 2025) (2024). https://arxiv.org/abs/2405.20541  
<a id="ref-65"></a>[65] Compute-Constrained Data Selection (2024). https://arxiv.org/abs/2410.16208  
<a id="ref-66"></a>[66] Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning (2025). https://arxiv.org/abs/2506.11300  
<a id="ref-67"></a>[67] How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining (ICLR 2026) (2025). https://arxiv.org/abs/2511.18903  
<a id="ref-68"></a>[68] TinyStories: How Small Can Language Models Be and Still Speak Coherent English? (2023). https://arxiv.org/abs/2305.07759  
<a id="ref-69"></a>[69] What Should Baby Models Read? (2024). https://arxiv.org/abs/2411.06672  
<a id="ref-70"></a>[70] Rephrasing the Web (WRAP) (2024). https://arxiv.org/abs/2401.16380  
<a id="ref-71"></a>[71] Nemotron-CC (NVIDIA, arXiv 2412.02595) (2024). https://arxiv.org/abs/2412.02595  
<a id="ref-72"></a>[72] Data curation via joint example selection further accelerates multimodal learning (JEST) (2024). https://arxiv.org/abs/2406.17711  
<a id="ref-73"></a>[73] Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (RHO-LOSS) (2022). https://arxiv.org/abs/2206.07137  
<a id="ref-74"></a>[74] karpathy/llm.c dev/data/data_common.py (2024). https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py  
<a id="ref-75"></a>[75] modded-nanogpt track_1_short/data.py (2025). https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/data.py  
<a id="ref-76"></a>[76] karpathy/nanoGPT train.py (get_batch) (2023). https://github.com/karpathy/nanoGPT/blob/master/train.py  
<a id="ref-77"></a>[77] PyTorch forum: Numpy memmap throttles with Dataloader when available RAM less than file size (accessed 2026). https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274  
<a id="ref-78"></a>[78] facebook/zstd README (benchmarks) (accessed 2026). https://github.com/facebook/zstd  
<a id="ref-79"></a>[79] An Information-Theoretic Perspective on LLM Tokenizers (2026). https://arxiv.org/abs/2601.09039  
<a id="ref-80"></a>[80] Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies (NeurIPS 2024) (2024). https://arxiv.org/abs/2407.13623  
<a id="ref-81"></a>[81] SuperBPE: Space Travel for Language Models (2025). https://arxiv.org/abs/2503.13423  
<a id="ref-82"></a>[82] An Engineering Journey Training Large Language Models at Scale on Alps: The Apertus Experience (2026). https://arxiv.org/abs/2604.12973  
<a id="ref-83"></a>[83] MosaicML Streaming docs: dataset conversion (MDSWriter) (2024). https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html  
<a id="ref-84"></a>[84] Lightning-AI/litdata (2025). https://github.com/Lightning-AI/litdata  
<a id="ref-85"></a>[85] PyTorch docs: Automatic Mixed Precision package (torch.amp) (2026). https://docs.pytorch.org/docs/stable/amp.html  
<a id="ref-86"></a>[86] PyTorch docs: torch.compile programming model, Recompilation (2025). https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html  
<a id="ref-87"></a>[87] PyTorch source: aten/src/ATen/native/transformers/cuda/sdp_utils.cpp (SDPA backend selection) (2026). https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp  
<a id="ref-88"></a>[88] HF transformers issue #22141: fused torch AdamW benchmarks (2023). https://github.com/huggingface/transformers/issues/22141  
<a id="ref-89"></a>[89] PyTorch issue #121857: fused AdamW slower than foreach (2024). https://github.com/pytorch/pytorch/issues/121857  
<a id="ref-90"></a>[90] Reducing Activation Recomputation in Large Transformer Models (2022). https://arxiv.org/abs/2205.05198  
<a id="ref-91"></a>[91] Current and New Activation Checkpointing Techniques in PyTorch (PyTorch blog) (2025). https://pytorch.org/blog/activation-checkpointing-techniques/  
<a id="ref-92"></a>[92] PyTorch docs: CUDAGraph Trees (2026). https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html  
<a id="ref-93"></a>[93] PyTorch tutorial: A guide on good usage of non_blocking and pin_memory() (2024). https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html  
<a id="ref-94"></a>[94] HF Accelerate utilities reference (find_executable_batch_size) (2025). https://huggingface.co/docs/accelerate/main/en/package_reference/utilities  
<a id="ref-95"></a>[95] Great Power, Great Responsibility: Recommendations for Reducing Energy for Training Language Models (2022). https://arxiv.org/abs/2205.09646  
<a id="ref-96"></a>[96] Sustainable Supercomputing for AI: GPU Power Capping at HPC Scale (2024). https://arxiv.org/abs/2402.18593  
<a id="ref-97"></a>[97] Characterizing Power Management Opportunities for LLMs in the Cloud (ASPLOS '24) (2024). https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf  
<a id="ref-98"></a>[98] NVIDIA nvidia-smi documentation (2025). https://docs.nvidia.com/deploy/nvidia-smi/index.html  
<a id="ref-99"></a>[99] NVIDIA Multi-Instance GPU User Guide: Supported GPUs (2025). https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html  
<a id="ref-100"></a>[100] PyTorch source: c10/core/thread_pool.cpp (2025). https://github.com/pytorch/pytorch/blob/main/c10/core/thread_pool.cpp  
<a id="ref-101"></a>[101] PyTorch Performance Tuning Guide (2025). https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html  
<a id="ref-102"></a>[102] PyTorch forum: Conv2D is very slow on trained weights (vs random weights) (accessed 2026). https://discuss.pytorch.org/t/conv2d-is-very-slow-on-trained-weights-vs-random-weights/43377  
<a id="ref-103"></a>[103] Empowering PyTorch on Intel Xeon Scalable processors with Bfloat16 (PyTorch blog) (2023). https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16  
<a id="ref-104"></a>[104] Pangu-Weather (Bi et al., arXiv 2211.02556; Nature 2023): official pseudocode (2023). https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py  
<a id="ref-105"></a>[105] FourCastNet (Pathak et al., arXiv 2202.11214) (2022). https://arxiv.org/abs/2202.11214  
<a id="ref-106"></a>[106] Spherical Fourier Neural Operators (ICML 2023) (2023). https://arxiv.org/abs/2306.03838  
<a id="ref-107"></a>[107] ClimaX: A foundation model for weather and climate (ICML 2023) (2023). https://arxiv.org/abs/2301.10343  
<a id="ref-108"></a>[108] ArchesWeather & ArchesWeatherGen: a deterministic and generative model for efficient ML weather forecasting (2024). https://arxiv.org/abs/2412.12971  
<a id="ref-109"></a>[109] Aurora: A Foundation Model for the Earth System (Nature 2025) (2024). https://arxiv.org/abs/2405.13063  
<a id="ref-110"></a>[110] FuXi: a cascade machine learning forecasting system for 15-day global weather forecast (2023). https://arxiv.org/abs/2306.12873  
<a id="ref-111"></a>[111] FengWu (Chen et al., arXiv 2304.02948) (2023). https://arxiv.org/abs/2304.02948  
<a id="ref-112"></a>[112] Neural General Circulation Models for Weather and Climate (Nature 2024) (2023). https://arxiv.org/abs/2311.07222  
<a id="ref-113"></a>[113] GenCast: Diffusion-based ensemble forecasting for medium-range weather (Nature 2024) (2023). https://arxiv.org/abs/2312.15796  
<a id="ref-114"></a>[114] Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting (2026). https://arxiv.org/abs/2606.26421  
<a id="ref-115"></a>[115] Modernizing CNN-based Weather Forecast Model towards Higher Computational Efficiency (KAI-α) (2025). https://arxiv.org/abs/2507.10893  
<a id="ref-116"></a>[116] Data-driven medium-range weather prediction with a ResNet pretrained on climate simulations (JAMES 2021) (2020). https://arxiv.org/abs/2008.08626  
<a id="ref-117"></a>[117] Scaling Laws of Global Weather Models (ICML 2026) (2026). https://arxiv.org/abs/2602.22962  
<a id="ref-118"></a>[118] Fixing the train-test resolution discrepancy (NeurIPS 2019) (2019). https://arxiv.org/abs/1906.06423  
<a id="ref-119"></a>[119] EfficientNetV2: Smaller Models and Faster Training (ICML 2021) (2021). https://arxiv.org/abs/2104.00298  
<a id="ref-120"></a>[120] AutoProg-One / AutoProg-Zero: automated progressive learning (arXiv 2410.00350) (2024). https://arxiv.org/abs/2410.00350  
<a id="ref-121"></a>[121] FlexiViT: One Model for All Patch Sizes (CVPR 2023) (2023). https://arxiv.org/abs/2212.08013  
<a id="ref-122"></a>[122] Navigating Scaling Laws: Compute Optimality in Adaptive Model Training (ICML 2024) (2024). https://arxiv.org/abs/2311.03233  
<a id="ref-123"></a>[123] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed (2025). https://arxiv.org/abs/2507.03779  
<a id="ref-124"></a>[124] PARADIS (arXiv 2601.21151, v2: coarse-to-fine recipe) (2026). https://arxiv.org/abs/2601.21151v2  
<a id="ref-125"></a>[125] The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators (ICLR 2026) (2025). https://arxiv.org/abs/2510.06646  
<a id="ref-126"></a>[126] Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs (iFNO, TMLR 2024) (2022). https://arxiv.org/abs/2211.15188  
<a id="ref-127"></a>[127] Token Merging: Your ViT But Faster (ICLR 2023) (2023). https://arxiv.org/abs/2210.09461  
<a id="ref-128"></a>[128] WeatherBench 2: A benchmark for the next generation of data-driven global weather models (2024). https://arxiv.org/abs/2308.15560  
<a id="ref-129"></a>[129] WB2 ERA5 64x32 zarr and 1990-2019 6h 64x32 climatology (own computation, 2026-10-15) (2026). gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr  
<a id="ref-130"></a>[130] ClimODE: Climate and Weather Forecasting with Physics-informed Neural ODEs (ICLR 2024) (2024). https://arxiv.org/abs/2404.10024  
<a id="ref-131"></a>[131] CoDiCast: diffusion-based weather forecasting (arXiv 2409.05975) (2024). https://arxiv.org/abs/2409.05975  
<a id="ref-132"></a>[132] ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling (2023). https://arxiv.org/abs/2307.01909  
<a id="ref-133"></a>[133] VarteX (arXiv 2406.19615) (2024). https://arxiv.org/abs/2406.19615  
<a id="ref-134"></a>[134] Compressing atmospheric data into its real information content (Klöwer et al., Nature Computational Science 1, 713) (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC10766530/  
<a id="ref-135"></a>[135] numcodecs BitRound codec (2024). https://numcodecs.readthedocs.io/en/stable/filter/bitround.html  
<a id="ref-136"></a>[136] xbitinfo quick-start (2024). https://xbitinfo.readthedocs.io/en/latest/quick-start.html  
<a id="ref-137"></a>[137] LLMComp: A Language Modeling Paradigm for Error-Bounded Scientific Data Compression (2025). https://arxiv.org/abs/2510.23632  
<a id="ref-138"></a>[138] Can Deep Neural Networks Improve Compression of Very Large Scientific Data? (2026). https://arxiv.org/abs/2606.14353  
<a id="ref-139"></a>[139] Error bounded compression for weather and climate applications (EBCC) (2025). https://arxiv.org/abs/2510.22265  
<a id="ref-140"></a>[140] CRA5: Extreme Compression of ERA5 for Portable Global Climate and Weather Research via an Efficient Variational Transformer (2024). https://arxiv.org/abs/2405.03376  
<a id="ref-141"></a>[141] ClimateBenchPress (v1.0): a benchmark for lossy compression of climate data (GMD 19, 5933) (2026). https://gmd.copernicus.org/articles/19/5933/2026/  
<a id="ref-142"></a>[142] pcodec benchmark results (2025). https://github.com/pcodec/pcodec/blob/main/docs/benchmark_results.md  
<a id="ref-143"></a>[143] Half-precision floating-point format (IEEE 754 binary16), Wikipedia (accessed 2026). https://en.wikipedia.org/wiki/Half-precision_floating-point_format  
<a id="ref-144"></a>[144] ECMWF CKB: GRIB to netCDF conversion on new CDS and ADS systems (2024). https://confluence.ecmwf.int/spaces/CKB/pages/450760890/GRIB+to+netCDF+conversion+on+new+CDS+and+ADS+systems  
<a id="ref-145"></a>[145] ERA5 netcdf encoding since the new CDSAPI release (ECMWF forum) (2024). https://forum.ecmwf.int/t/era5-netcdf-encoding-since-the-new-cdsapi-release/8146  
<a id="ref-146"></a>[146] WeatherBench 2 Data Guide (2024). https://weatherbench2.readthedocs.io/en/latest/data-guide.html  
<a id="ref-147"></a>[147] google-research/arco-era5 (Analysis-Ready, Cloud Optimized ERA5) (accessed 2026). https://github.com/google-research/arco-era5  
<a id="ref-148"></a>[148] NVIDIA PhysicsNeMo: Unified Recipe for Training Global Weather Forecasting Models (2025). https://docs.nvidia.com/physicsnemo/26.08/physicsnemo/examples/weather/unified_recipe/README.html  
<a id="ref-149"></a>[149] Accelerating AI/ML Workflows in Earth Sciences with GPU-Native Xarray, Zarr, DALI, and nvcomp (2025). https://xarray.dev/blog/gpu-pipeline  
<a id="ref-150"></a>[150] Cloud native data loaders for ML using Zarr and Xarray (Earthmover blog) (accessed 2026). https://www.earthmover.io/blog/cloud-native-dataloader/  
<a id="ref-151"></a>[151] google-deepmind/weathernext: weathernext/utils/casting.py (GraphCast bf16 wrapper) (2023). https://raw.githubusercontent.com/google-deepmind/weathernext/master/weathernext/utils/casting.py  
<a id="ref-152"></a>[152] microsoft/aurora: aurora/model/aurora.py (2024). https://github.com/microsoft/aurora/blob/main/aurora/model/aurora.py  
<a id="ref-153"></a>[153] PyTorch docs: torch.fft.fft2 (2026). https://docs.pytorch.org/docs/2.14/generated/torch.fft.fft2.html  
<a id="ref-154"></a>[154] Guaranteed Approximation Bounds for Mixed-Precision Neural Operators (2023). https://arxiv.org/abs/2307.15034  
<a id="ref-155"></a>[155] Prithvi WxC (NASA/IBM weather and climate foundation model) (2024). https://arxiv.org/abs/2409.13598  
<a id="ref-156"></a>[156] Pangu-Weather reproduction and ablation study (Geosci. Model Dev. 17, 8873) (2024). https://gmd.copernicus.org/articles/17/8873/2024/  
<a id="ref-157"></a>[157] FlashBias: Fast Computation of Attention with Bias (2025). https://arxiv.org/abs/2505.12044  
<a id="ref-158"></a>[158] Can AI Weather Models Predict Beyond Two Weeks? (long-rollout benchmark) (2026). https://arxiv.org/html/2605.30184v1  
<a id="ref-159"></a>[159] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models (Mosaic) (2026). https://arxiv.org/abs/2604.16429  
<a id="ref-160"></a>[160] Challenges of learning multi-scale dynamics with AI weather models (FouRKS) (2023). https://arxiv.org/abs/2304.07029  
<a id="ref-161"></a>[161] google-research/weatherbench2: metrics.py (2024). https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py  
<a id="ref-162"></a>[162] Analyzing and Exploring Training Recipes for Large-Scale Transformer-Based Weather Prediction (2024). https://arxiv.org/abs/2404.19630  
