# Scoreboard: a 5B-parameter LLM on one RTX PRO 6000 Blackwell (96 GiB)

> **This is arithmetic, not a measurement.** Every number below comes from
> `bantam memory` (`bantamtrain/memory.py`, Bantam 0.3.0): published figures
> for the GPU and for each technique, an activation-memory model whose
> constants were measured on Bantam's own model on a CPU, and standard FLOP
> accounting. Bantam has never run on this card or on any GPU. Section 7 lists
> what would have to be measured to confirm it.

**Question.** Can a 5B-parameter model be trained from scratch on one
RTX PRO 6000 Blackwell (96 GiB), how long would it take, and how much memory
and storage does each recipe need?

**Short answer (by the tool's arithmetic).**
- The standard PyTorch recipe (AMP with fp32 master weights and fp32 AdamW)
  keeps 16-18 bytes per parameter of model state (76-85 GiB for 5.09B
  parameters), so its activations fit only if they are recomputed. In the
  tool's 18-byte accounting it **does not fit** with a 15 % safety margin even
  then, and just fits with no margin. A 16-byte variant would just fit with
  recompute and a 15 % margin: one 2048-token sequence, with 0.7 GiB to spare
  (section 1.1).
- Bantam's memory stack (bf16 weights with stochastic rounding, 8-bit
  optimizer state, chunked cross-entropy) keeps **6 bytes per parameter
  (28 GiB)** and fits without recompute.
- Adding Muon (8-bit momentum, about 5.1 bytes per parameter, 24 GiB) cuts the
  estimated time for a 100B-token run from **176 to 94 days, 1.87x**. That
  factor is the published Muon compute claim, not a Bantam measurement.
- The 0.2 version of this page said 2.56x (234 → 91 days). It mixed GB and
  GiB and compared against a standard recipe that, with the margin the code
  itself uses, does not fit. Section 1 shows both framings.
- None of this is distinctive. ZeRO-Offload trained more than 13B parameters
  on one 32 GB V100 in 2021, and torchao ships 8-bit and 4-bit AdamW with bf16
  stochastic rounding (section 5).

**Units.** GPU memory is in **GiB** (2^30 bytes). GPU memory chips come in
binary sizes, so a "96 GB" card has 96 GiB of physical memory; `nvidia-smi`
reports it in MiB, a little below 96 x 1024 because the driver reserves some.
If you prefer the decimal reading (96 x 10^9 bytes = 89.4 GiB), pass
`--gpu_gb 89.4` (section 3). File and disk sizes (section 4) are in **GB**
(10^9 bytes).

## 1. Time and memory to the same model quality

Model: 5.09B parameters (d=4096, 24 layers, 32k vocabulary, untied
embeddings), sequence length 2048, trained on **100B tokens** (about
Chinchilla-optimal for 5B). Each row adds one technique to the row above and
is costed at the compute it needs to reach the same loss. "Usable" memory is
96 GiB x 0.85 headroom = 81.6 GiB, minus 3 GiB of CUDA context and workspaces.

```bash
bantam memory --markdown
```

| configuration | bytes/param | static GiB | total GiB | fits | max micro-batch | days | speedup |
|---|---|---|---|---|---|---|---|
| 1. Standard: AMP bf16, fp32 master + fp32 AdamW | 18.0 | 85 | 90 | no | 0 | n/a | n/a |
| 2. + bf16 weights (stochastic rounding) + 8-bit AdamW + chunked CE | 6.0 | 28 | 78 | yes | 6 | 176 | 1.00x |
| 3. + Muon (8-bit momentum) | 5.1 | 24 | 81 | yes | 7 | 94 | 1.87x |
| 4. + FP8 linears (**no code path in Bantam**) | 5.1 | 24 | 81 | yes | 7 | 78 | 2.24x |
| 5. + 450 W power cap (75 % power) | 5.1 | 24 | 81 | yes | 7 | 83 | 2.13x |

- *static* = weights + gradients + optimizer state. *total* adds activations
  for the largest micro-batch that fits (capped at 8), the loss buffer and the
  3 GiB overhead. Speedups are against the first row that fits (row 2).
- Row 1 needs 85.4 GiB of static memory alone, more than the 81.6 GiB usable.
- **Row 4 is a literature estimate only.** Bantam has no FP8 training code
  (no Transformer Engine or torchao float8 path).
- **Row 5** gives up about 5 % speed for 25 % less power. Use a hardware cap
  (`nvidia-smi -pl 450`); `--duty` is the software fallback.

### 1.1 The same table with no safety margin

```bash
bantam memory --headroom 1
```

| configuration | static GiB | total GiB | max micro-batch | days | speedup |
|---|---|---|---|---|---|
| 1. Standard (+ activation recompute to fit) | 85 | 94 | 3 | 234 | 1.00x |
| 2. + bf16 weights + 8-bit AdamW + chunked CE | 28 | 93 | 8 | 176 | 1.33x |
| 3. + Muon (8-bit momentum) | 24 | 89 | 8 | 94 | 2.49x |
| 4. + FP8 linears (no code path) | 24 | 89 | 8 | 78 | 2.98x |
| 5. + 450 W power cap | 24 | 89 | 8 | 83 | 2.84x |

Using 100 % of the card is not realistic (allocator fragmentation, spikes), but
this is the framing closest to the 0.2 page. Here the standard recipe fits only
with full activation recompute, which costs one extra forward pass (8·N·D
instead of 6·N·D FLOPs). Row 2's 1.33x is exactly that saving; row 3's further
1.87x is the Muon credit.

**A fairer standard recipe.** The tool charges row 1 at 18 bytes/param:
fp32 weights (4), the bf16 weight copies autocast keeps during the forward
pass (2), fp32 gradients (4) and AdamW's two fp32 moments (8). The ZeRO paper's
accounting for mixed-precision Adam is 16 bytes/param (16-bit weights and
gradients, fp32 master copy and moments): 81.5 GB = 75.9 GiB. By the tool's own
per-sequence constants, that variant would leave about 2.7 GiB for activations
at 85 % headroom. One 2048-token sequence with full recompute needs about
2.0 GiB (1.1 GiB of stored block inputs and one recomputed layer, 0.9 GiB for
the unchunked loss), so it fits with 0.7 GiB to spare. It would then take
about 234 days, like row 1 in section 1.1. Whether the standard recipe fits at
all therefore comes down to less than 1 GiB, which only a measurement on the
card can settle. (This paragraph is hand arithmetic with the tool's constants;
the tool does not model the 16-byte variant.)

### 1.2 A 20B-token run

```bash
bantam memory --tokens 2e10
```

Days for rows 1-5: n/a, 35, 19, 16, 17 (85 % headroom). With `--headroom 1`:
47, 35, 19, 16, 17.

## 2. How the numbers are made

Every assumption is a flag of `bantam memory` (`bantam memory --help`).

| quantity | default | source |
|---|---|---|
| GPU memory | 96 GiB (`--gpu_gb`) | NVIDIA spec |
| usable fraction | 0.85 (`--headroom`) | the peak that `device.find_batch_size` keeps to |
| CUDA context and workspaces | 3 GiB (`--overhead_gb`) | assumption |
| dense BF16 peak | 503.8 TFLOPS (`--peak_tflops`) | NVIDIA RTX Blackwell PRO architecture whitepaper |
| utilisation (MFU) | 0.40 (`--mfu`) | conservative; nanochat measured about 59 % on one card at 600 W |
| training FLOPs | 6·N·D, embeddings included, no attention term | standard; nanochat's MFU convention counts +5.3 % for this model |
| activation recompute | 8·N·D | one extra forward pass |
| Muon compute to the same loss | 0.52x AdamW's (`--muon_flops`) | Liu et al. 2025 (Moonshot): about 2x compute efficiency in compute-optimal scaling-law runs |
| Muon Newton-Schulz cost | 4.9e14 FLOPs per optimizer step at 524,288 tokens per step (`--global_batch_tokens`), about +3 % | computed like `bantamtrain.optim.ns_flops` |
| FP8 speed-up | x1.20 (`--fp8_speedup`) | reports of 20-30 %; not implemented in Bantam |
| power cap throughput | x0.95 (`--power_tput`) | ResNet-50 sweep on this card; one LLM data point (nanochat at 300 W) lost 32 % |
| activations, AMP | 48.7 bytes per layer · token · d_model | measured on Bantam's ZoomGPT, eager mode, CPU |
| activations, bf16 weights | 40.7 bytes per layer · token · d_model | same |
| logits (unchunked loss) | 14 bytes per element | bf16 logits + fp32 log-softmax + fp32 gradient |
| chunked cross-entropy | 2048 tokens per chunk (`--ce_chunk`) | `bantamtrain.losses.chunked_cross_entropy` |

Bytes per parameter by row: row 1 = 4 + 2 + 4 + 8 = 18; row 2 = bf16 weights 2
+ bf16 gradients 2 + 8-bit AdamW moments 2 = 6; rows 3-5 = 2 + 2 + 8-bit Muon
momentum on the hidden matrices (1 byte) and 8-bit AdamW elsewhere (2 bytes),
about 5.1 on average.

The activation constants were measured with `memory.measure_saved_bytes()` on
the 5.8M-parameter ZoomGPT on a CPU. Korthikanti et al.'s fused-kernel figure
of 34 bytes corresponds to `--act_factor 0.7`.

## 3. Sensitivity

| change | flag | effect |
|---|---|---|
| higher utilisation (59 %) | `--mfu 0.59` | rows 2/3: 119 / 64 days; ratios unchanged |
| fused or compiled kernels | `--act_factor 0.7` | max micro-batch 9 (row 2) and 10 (rows 3-5); days unchanged |
| activations 1.6x the measured constants | `--act_factor 1.6` | max micro-batch 4 in rows 2-5; row 1 still does not fit |
| decimal reading of "96 GB" | `--gpu_gb 89.4` | max micro-batch 5 (row 2) and 6 (rows 3-5); row 1 does not fit |
| Muon gain like a tuned-AdamW benchmark at 1.2B (Wen et al. 2025: about 1.1x) | `--muon_flops 0.9` | row 3: 163 days, 1.08x over row 2 |
| no chunked cross-entropy (`bantam pretrain` without `--ce_chunk`, its default) | `--ce_chunk 0` | one sequence less per micro-batch (row 2: 5, row 3: 6); days unchanged |

The Muon row carries most of the claimed speed-up, and it rests entirely on
the literature. Wen et al. 2025 find that against a carefully tuned AdamW,
matrix optimizers give about 1.4x at 0.1B parameters and only about 1.1x at
1.2B. With that figure the Muon row is 1.08x, not 1.87x.

## 4. Storage for the same 5B model (decimal GB)

| item | standard | Bantam | how |
|---|---|---|---|
| weights for inference | 20.4 GB fp32; 10.2 GB bf16 | **7.2 GB** bf16 + byte shuffle + zstd; **5.2 GB** int8; **2.7 GB** int4 | WeightPack 16 / 8 / 4, `qweights` int8 / int4 weight-only blocks |
| resumable training checkpoint | 61.1 GB (fp32 weights + fp32 AdamW) | **15.8 GB** | bf16 weights + 8-bit Muon state |
| 100B-token training corpus | 200 GB uint16 token ids; ~400 GB raw text | **74.6 GB** at 5.97 bits/token | TokPack v2: rank remap + LEB128 + zstd with a shared dictionary, exact |

- WeightPack 16 stores bf16. It is exact for bf16-trained weights, as in rows
  2-5. For fp32 weights it is bf16 rounding (a +0.00002 loss change on the
  5.8M model), not lossless. The tool's 1.42 bytes/param is close to the 1.39
  measured on that 5.8M model; it has not been measured at 5B.
- int8 / int4 cost +0.0002 / +0.026 nats on the trained 5.8M model, but +0.003 /
  +0.17-0.52 (4-bit, depending on NF4 vs int4 and the embedding format) on the
  pretrained SmolLM2-135M ([`../RESULTS.md`](../RESULTS.md)
  sections 10 and 14.2).
- 5.97 bits/token was measured on 117M TinyStories tokens with a 4,096-token
  vocabulary. A 32k-vocabulary web corpus needs roughly 7.3-9 bits/token,
  that is 91-112 GB for 100B tokens (`--bits_per_token`).

## 5. Fair comparison with other ways to fit a large model on one GPU

The table above compares Bantam's stack with plain PyTorch AMP + AdamW. That
is not the strongest baseline. These tools already do the same or more
(claims as each project or paper reports them; sources in
[`../RESEARCH.md`](../RESEARCH.md)):

| tool / method | what it does | published claim (their conditions) | relation to Bantam |
|---|---|---|---|
| Standard AMP + AdamW | fp32 master weights and moments | 16 B/param model state (ZeRO paper); LLaMA-Factory's rule of thumb is about 18 GB per billion parameters for full AMP training | the reference rows here |
| DeepSpeed ZeRO-Offload (2021) | optimizer state and step on the CPU | more than 13B parameters on one 32 GB V100; 40 TFLOPS for 10B | already beats "5B on one card" with enough host RAM. Bantam's difference is no offload and no PCIe traffic, which is unmeasured |
| torchao low-bit optimizers | AdamW8bit / 4bit / FP8 state, `bf16_stochastic_round`, CPU offload; pure PyTorch + compile | Llama2-7B full-bf16 fine-tune on an A100: 51.6 GB (fused AdamW) → 39.1 GB (8-bit) → 33.2 GB (4-bit) | the closest existing equivalent of rows 2-3. Bantam adds 8-bit **Muon** and this estimator |
| bitsandbytes 8-bit optimizers (Dettmers et al. 2022) | blockwise 8-bit Adam state, CUDA kernels | 32-bit quality, about 75 % less optimizer memory | the idea behind Bantam's 8-bit state; Bantam's pure-PyTorch version is expected to be slower without fused kernels |
| 8-bit Muon (Gupta et al. 2025) | blockwise 8-bit Muon momentum | parity with full-precision Muon up to 2.7B | published prior art for row 3's state format |
| GaLore (Zhao et al. 2024) | low-rank gradient projection | LLaMA-7B pretraining on a 24 GB RTX 4090 (8-bit, layer-wise updates) | a larger model in a quarter of the memory; uses a projection rather than a full-rank optimizer |
| APOLLO (Zhu et al. 2025) | SGD-like memory with estimated scaling | LLaMA-7B from scratch under 12 GB with quantization | far more memory-efficient than Bantam's approach |
| Adam-mini (Zhang et al. 2025) | one LR per Hessian block | about 50 % less optimizer memory | similar saving to 8-bit Adam, without quantization |
| FSDP / ZeRO-3 | shard everything across GPUs | multi-GPU | Bantam has no sharding and cannot scale out |
| Fused kernels (Liger, Cut Cross-Entropy, FlashAttention) | fused loss, norms, attention | Liger: +20 % throughput, -60 % memory (8 x A100); CCE: Gemma 2 2B loss memory 24 GB → 1 MB | Bantam's chunked cross-entropy is the same idea in plain PyTorch: it saves memory, not time |

**Honest position.** Fitting 5B on one 96 GiB card is plausible by arithmetic
and not unusual. Bantam's possible contribution is a single, readable,
pure-PyTorch package that combines bf16 + stochastic rounding, 8-bit state
and Muon, plus this estimator. Fitting is also not the same as being
practical: about 94-176 days for 100B tokens at 40 % utilisation. The realistic
single-card use is fine-tuning or continued pretraining on a few billion
tokens.

## 6. What Bantam's trainer can run today

`bantam pretrain` (or `python3 llm/train.py`) has these flags for the rows. The
command shows the flags for a 5B shape (d=4096, 24 layers, 524,288 tokens per
step) and a budget of about 100B tokens (3.1e21 FLOPs in the trainer's
convention, about 190,000 steps). It is not a tuned recipe, and the LR has not
been tested at this size. ZoomGPT ties the output head to the embedding, so
with a 32k vocabulary this shape has about 4.96B parameters, not the untied
5.09B of section 1:

```bash
bantam pretrain --data DIR --budget 3.1e21 --zoom 1 --optim muon --lr 1.6e-2 \
    --param_dtype bf16 --state_bits 8 --ce_chunk 2048 --device cuda --precision bf16 --compile \
    --d 4096 --layers 24 --heads 32 --seq 2048 --batch 4 --accum 64 \
    --name gpu_5b --out results/gpu
```

- `--param_dtype bf16` (stochastic rounding), `--state_bits 8`, `--ce_chunk 2048`
  (chunked cross-entropy), `--optim muon`, `--precision bf16`, `--compile`: rows 2-3.
- `--grad_ckpt 1` (activation recompute) and `--accum N` when memory is tight.
- `--auto_batch` (GPU only) splits `--batch x --accum` into the largest
  power-of-two micro-batch whose peak memory stays under 85 % of the card,
  keeping the same effective batch, schedule and data order.
- `--duty 0.75`: the software power fallback for row 5.

Gaps:
- **Chunked cross-entropy is off by default.** Rows 2-5 assume it; pass
  `--ce_chunk 2048` as above. Without it pretraining builds the full logits and
  the micro-batch is one sequence smaller (section 3; `bantam memory --ce_chunk 0`).
  Its memory saving has not been measured on a GPU.
- **No FP8 path** (row 4).
- **No peak-GPU-memory logging.** The run log has no
  `torch.cuda.max_memory_allocated` field yet, which a measurement needs.
- The vocabulary comes from the data: a 32k-vocabulary 5B run needs data
  prepared with `bantam prepare-text --tokenizer NAME_OR_PATH` (or a 32k BPE).
- No sharding (FSDP / ZeRO), so one card only.

## 7. What would have to be measured on a real GPU

1. **Peak memory per row**: `torch.cuda.max_memory_allocated` and
   `max_memory_reserved` at the micro-batch the tool predicts, first on a
   ~1B model (where every row fits) to check the constants, then at 5B.
2. **Activation constants with GPU kernels** (SDPA flash attention,
   `--compile`), which set `--act_factor`.
3. **Whether row 1 fits at all**, with 16- and 18-byte standard recipes.
4. **Throughput and MFU per row** (tokens/s), including the 8-bit
   dequantize/update transient, bf16 stochastic rounding and Newton-Schulz on
   the GPU. On this CPU, bf16 + stochastic rounding was 1.6-2.6x slower than
   fp32, so the GPU cost must not be assumed.
5. **Loss parity at scale**: bf16 + stochastic rounding + 8-bit state against
   fp32-master AdamW, and Muon's compute multiplier against a tuned AdamW at
   124M parameters or more.
6. FP8 (after it is implemented), and LLM throughput under a 450 W cap.
7. CUDA context and allocator overhead (3 GiB assumed).
8. WeightPack and TokPack ratios on bf16-trained 5B weights and on a
   32k-vocabulary corpus.

`experiments/gpu_smoke.sh` is only a first check that the CUDA, bf16, compile
and duty-cycle paths run: a Muon baseline at LR 1.6e-2 and `--recipe best` at
batch 64 (not a tuned batch). It is not a benchmark and records no memory
numbers.

## 8. What is not in the table

- **Zoom training.** On the 5.8M CPU model, the k=2 batch-x2 schedule saved
  about 11 % of compute (1.12x, 3 seeds), and the plain baseline needed 1.41x
  the compute to match QK-norm + Zoom (2 seeds). It gets no credit here: one
  tiny model, a gain smaller than the prior art reports (patch-level training: about 0.5x cost at
  370M-2.7B, with the advantage shrinking with size), no wall-clock gain on
  the CPU, and a loss on weather under Muon.
- **QK-norm** (1.24x at 5.8M, 2 seeds) is also uncredited; its effect at 5B is
  a literature question.
- **Deduplication.** TinyStories had 4 % exact duplicates and no
  near-duplicates. The gain depends on the dataset.
- **Multi-GPU.** Everything here is one card.

## 9. Evidence behind the rows

| technique | literature | measured in this repo (5.8M-parameter LM, TinyStories, CPU) |
|---|---|---|
| bf16 weights + stochastic rounding | matches fp32-master mixed precision up to 6.7B (Ozkara et al. 2025); prior art Zamirai et al. 2020 | 8-bit Muon + bf16 weights at full budget: 2.728 vs 2.7365 (fp32, same seed); scored with 0.2 bf16 arithmetic, ±0.001, marked for rerun |
| 8-bit optimizer state | 8-bit Adam matches 32-bit up to 1.5B (Dettmers et al.) | 100 steps: AdamW 4.4792 vs 4.4824; Muon 3.5812 vs 3.5719; state 3.9x smaller. 0.3 changed the 8-bit map (OPT-4); rerun pending |
| activation recompute | +30-40 % wall-clock in Megatron (theory +33 %) | the same loss (3.5719, to 1e-8) with recompute + accumulation x2; +44 % time on the CPU |
| Muon | 0.52x AdamW's FLOPs in compute-optimal scaling-law runs (Moonshot); 1.4x at 0.1B and 1.1x at 1.2B against tuned AdamW (Wen et al.) | equal model FLOPs: 2.737 ± 0.012 (3 seeds) vs AdamW 3.332 (1 seed). The AdamW part of the Muon runs used a 16x higher LR, and Newton-Schulz adds an uncharged 20 % of model FLOPs here. The table credits only the published 0.52x |
| chunked cross-entropy | torchtune: -15.8 % memory on Llama 3.2 3B | on by default in `bantam finetune`; `bantam pretrain --ce_chunk` (off by default), the same loss up to fp32 summation order; no committed memory measurement |
| FP8 | Transformer Engine supports SM 12.0; reports of 20-30 % speed-up | not implemented |
| 450 W cap | ResNet-50 keeps 94.8 % throughput at 450 W on this card | not measurable on a CPU |
| GPU throughput | 503.8 TFLOPS dense BF16 peak; nanochat ~297 TFLOPS (59 %) on one card | the table assumes 40 % |

## 10. Reproduce and adapt

```bash
bantam memory                                   # section 1 (text)
bantam memory --markdown                        # section 1 (markdown)
bantam memory --headroom 1                      # section 1.1
bantam memory --tokens 2e10                     # section 1.2
bantam memory --mfu 0.59 --act_factor 0.7       # section 3
bantam memory --ce_chunk 0                      # section 3, no chunked cross-entropy
bantam memory --params 1e9 --d 2048             # other sizes
bantam memory --finetune --params 7e9 --gpu_gb 24   # full / LoRA / QLoRA fine-tuning table
```

`python -m bantamtrain memory ...` and `python -m bantamtrain.memory ...` are
equivalent.

## What changed since 0.2

- The commands use `bantam memory` (or `python -m bantamtrain.memory`); the
  0.2 commands used the pre-rename package name, which no longer exists.
- GPU memory is GiB throughout and the default headroom is 0.85. The standard
  recipe no longer fits at the default, so the headline is 1.87x against the
  first row that fits (2.49x with no margin), not 2.56x.
- Activation constants are measured (48.7 / 40.7 bytes per layer · token ·
  d_model instead of 34), logits cost 14 bytes per element, the
  chunked-CE chunk is 2048 tokens, and Muon's Newton-Schulz FLOPs are charged.
  Max micro-batches are now 6-7 (default), 4 (`--act_factor 1.6`), not 9 and 6.
- The FP8 row is marked as having no code path, and the page no longer claims
  that the trainer "exposes every row". Pretraining had no chunked
  cross-entropy flag; 0.3.0 adds `--ce_chunk` (off by default), and the section 6
  command passes it.
- WeightPack 16 is described as bf16 (exact only for bf16 weights), not as
  lossless for any model.
- The Zoom confirmation seeds are done (3 seeds, 1.12x); int8 / int4 quality is
  reported in `RESULTS.md`; AdamW in the Muon comparison is one seed, not three.
- A fair-comparison section (ZeRO-Offload, torchao, GaLore and others) and a
  list of GPU measurements were added.
