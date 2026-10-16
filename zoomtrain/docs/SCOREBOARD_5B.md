# Scoreboard: training a 5B-parameter LLM on one RTX PRO 6000 Blackwell (96 GB)

**Question.** How long does a 5B model take to train on one RTX PRO 6000
Blackwell with the standard PyTorch recipe, compared with the Bantam stack,
and how much memory and storage does each need?

**Short answer.** To reach the same model quality, the full software stack is
about **2.6× faster** (234 → 91 days for a 100B-token-quality run; 47 → 18
days for 20B), or **~3× with FP8** matmuls. Weights, gradients and optimizer
state shrink from **92 GB to 26 GB**. A resumable checkpoint shrinks from
61 GB to 16 GB, and the training corpus from 200 GB of token ids to 75 GB.

> **How these numbers were made.** The 5B rows are *estimates* from
> `python -m zoomtrain.memory`. They combine three things: published, fact-checked
> figures for the GPU and for each technique (sources below and in
> [`../RESEARCH.md`](../RESEARCH.md)), the standard activation-memory formula,
> and measurements of each technique made in this repo at small scale. This
> machine has no GPU, so nothing was run on an RTX PRO 6000 here.
> `experiments/gpu_smoke.sh` is the first thing to run on the real card.

## 1. Time and memory to the same model quality

Model: 5.09B parameters (d = 4096, 24 layers, 32k vocabulary, untied
embeddings), sequence length 2048. The reference is the standard recipe
trained on **100B tokens**, about Chinchilla-optimal for 5B. Each row adds one
technique to the row above and is costed at the compute it needs to reach the
same loss.

| configuration | bytes/param | weights+grads+state | fits in 96 GB? | max micro-batch | days | speedup |
|---|---|---|---|---|---|---|
| 1. Standard: AMP bf16, fp32 master weights + fp32 AdamW | 18 | 92 GB | only with activation recompute | 1 | **234** | 1.00× |
| 2. + bf16 weights (stochastic rounding) + 8-bit AdamW + chunked cross-entropy | 6 | 31 GB | yes, no recompute | 9 | 176 | 1.33× |
| 3. + Muon optimizer (8-bit momentum) | 5.1 | 26 GB | yes | 9 | **91** | **2.56×** |
| 4. + FP8 matmuls (Transformer Engine) | 5.1 | 26 GB | yes | 9 | **76** | **3.08×** |
| 5. + 450 W power cap (25 % less power) | 5.1 | 26 GB | yes | 9 | 80 | 2.92× |

For a 20B-token run (continued pretraining, or a smaller data budget) the
days are 47 → 35 → 18 → 15 → 16.

**What each step buys, in plain words**
- **Step 2:** stops keeping a 4-byte fp32 copy of every weight and stores
  optimizer state in 8 bits. The memory freed is enough to drop activation
  recomputation, which was costing one extra forward pass (8ND → 6ND FLOPs).
- **Step 3:** Muon reaches AdamW's loss with about half the compute. It also
  keeps one state buffer instead of two.
- **Step 4:** FP8 tensor cores run the big matmuls faster.
- **Step 5:** gives up ~5 % speed for 25 % less power and heat. Use a hardware
  cap (`nvidia-smi -pl 450`); `--duty` is the software fallback.

## 2. Evidence behind every row

| technique | what the literature reports (fact-checked) | what this repo measured (5.8M-param LM, TinyStories) |
|---|---|---|
| bf16 weights + stochastic rounding | Matches fp32-master mixed precision on validation perplexity up to 6.7B, with 1.07-1.54× throughput (Ozkara et al. 2025). Nearest rounding without SR is clearly worse | Muon 8-bit + bf16 weights: loss **3.5752 vs 3.5719** fp32 (+0.003); weights 23.1 → 11.5 MB |
| 8-bit optimizer state | 8-bit Adam matches 32-bit perplexity up to 1.5B, state 8 → 2 B/param (Dettmers et al. 2021) | AdamW 8-bit **4.4792 vs 4.4824**; Muon 8-bit **3.5812 vs 3.5719**; state memory **3.9× smaller** |
| activation recompute | +30-40 % wall-clock in Megatron (theory +33 %) | identical loss with recompute + grad accumulation (**3.5719 vs 3.5719**) |
| Muon | AdamW's loss with **52 % of the FLOPs**, compute-optimal 1-16B (Moonshot 2025). Against a heavily tuned AdamW, ~1.4× at 0.1B and ~1.1× at 1.2B (RESEARCH.md §2.1) | Equal compute: **2.737 ± 0.011 vs 3.332** (3 seeds); Muon at 1/3 of the compute beat AdamW at full compute. The table credits only the published ~1.9× |
| FP8 matmuls | Transformer Engine FP8 per-tensor and block scaling are supported on SM 12.0; nanochat reports 20-30 % speedups. No memory saving | not measurable on CPU |
| 450 W cap | ResNet-50 training on this card keeps 94.8 % throughput at 450 W. For LLM training, nanochat at 300 W lost 32 % throughput and gained 36 % tokens per watt | not measurable on CPU |
| GPU throughput | dense BF16 peak 503.8 TFLOPS. Single-card nanochat measured ~297 TFLOPS (59 %) | the table assumes a conservative **40 %** for every row |

**Sensitivity**
- **Activation memory 1.6× the formula** (HF-style Llama code): the efficient
  stack still fits without recompute (micro-batch 6), and the standard recipe
  still does not.
- **MFU:** if utilisation is 59 % instead of 40 %, every row is 1.5× faster
  and the speedup ratios are unchanged.

## 3. Storage for the same 5B model

| item | standard | Bantam | how |
|---|---|---|---|
| weights for inference (RAM or disk) | 20.4 GB fp32 / 10.2 GB bf16 | **7.2 GB** lossless; **5.2 GB** int8; **2.7 GB** int4 | WeightPack 16 (byte-shuffle + zstd, bit-exact); `qweights` int8 / int4 weight-only blocks |
| resumable training checkpoint | 61.1 GB (fp32 weights + AdamW) | **15.8 GB** | bf16 weights + 8-bit Muon state |
| 100B-token training corpus | 200 GB uint16 / ~400 GB raw text | **74.6 GB** | TokPack v2: rank remap + LEB128 + zstd with dictionary; 5.97 bits/token measured on 116.8M TinyStories tokens, bit-exact |

Int8 / int4 quality on a trained model is measured by
`experiments/weights_eval.py`; results are in `RESULTS.md` once that run finishes.

## 4. What is *not* in the table

- **Zoom Training.** The original 4→2→1 schedule lost to the baseline. One
  variant (k = 2 for the first half, with the batch doubled while zoomed) beat
  it on one seed (2.690 vs 2.737 ± 0.011). Confirmation seeds are still
  running. Until they agree, it gets no credit here.
- **Deduplication.** TinyStories had 4 % exact duplicates and no near-duplicates.
  The gain depends on the dataset, so none is claimed.
- **Multi-GPU.** Everything here is one card.

## 5. Reproduce / adapt

```bash
python -m zoomtrain.memory                          # the table above
python -m zoomtrain.memory --tokens 2e10            # 20B-token run
python -m zoomtrain.memory --params 1e9 --d 2048    # other sizes
python -m zoomtrain.memory --mfu 0.59 --act_factor 1.6 --markdown
```

On the GPU, `llm/train.py` exposes every row: `--param_dtype bf16
--state_bits 8 --optim muon --device cuda --precision bf16 --compile`, plus
`--grad_ckpt 1` / `--accum N` when memory is tight and `--duty 0.75` as the
software power fallback.
