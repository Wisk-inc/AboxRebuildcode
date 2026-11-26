# How good is Bantam compared with other tools?

*Status: Bantam 0.3.0, 2026-11-26.*

Every Bantam number here was measured on one CPU-only sandbox (4 cores, 16 GB
RAM) with small models: a 5.8M-parameter TinyStories language model, a
1.26M-parameter ERA5 weather model, and LoRA fine-tuning of SmolLM2-135M.
Nothing has been run on a GPU. Every number for another tool or paper is that
project's own published claim, under its own conditions, with a link. None of
those claims was reproduced here. "n" is the number of seeds.

How Bantam's techniques work is in [METHOD.md](METHOD.md). The literature
behind them is in [RESEARCH.md](../RESEARCH.md). Bantam's full results are in
[RESULTS.md](../RESULTS.md).

---

## 1. Short answer

- **Bantam is a small, readable, pure-PyTorch toolkit for research, teaching
  and limited hardware.** It is not a replacement for Unsloth, Axolotl, TRL or
  torchtune on GPUs.
- **Its speed and memory on GPUs are unknown.** Every serious competitor
  publishes GPU numbers. Unsloth reports 2x speed and more than 70% less VRAM
  against Hugging Face + FlashAttention-2 on H100 and Blackwell GPUs. Liger
  Kernel reports +20% throughput and −60% memory on 8×A100. Bantam has no GPU
  number at all, so the honest statement is "not measured", not "good" or
  "bad". A first CPU comparison with Hugging Face Trainer + PEFT is running
  now (§4).
- **Almost everything in Bantam is known and already shipped elsewhere.**
  Muon, Polar Express, 8-bit optimizer state, bf16 stochastic rounding, QLoRA,
  DoRA, LoRA+, packing and chunked cross-entropy all exist in other tools,
  most with fused GPU kernels.
- **Zoom training, Bantam's own idea, is a variant of patch-level training.**
  At 5.8M parameters it saves about 11-12% of compute for the same loss: the
  baseline needs 1.12-1.14x the compute to match (n=2-3).
  The closest prior work reports about 2x at 370M-2.7B parameters. A known trick,
  QK-norm, helped more (1.24x, n=2). The two stack: QK-norm plus Zoom reached
  1.41x against the plain baseline (n=2). On weather, Zoom hurt under Muon.
- **Bantam's small language model is behind every TinyStories model** in the
  published comparison (§3). It beats only the 0.26M-parameter llama2.c
  stories260K, which is 22x smaller.

---

## 2. Tool by tool

### 2.1 Fine-tuning tools

| Tool | What it does | Published claims (their conditions) | How Bantam compares |
|---|---|---|---|
| **Unsloth** ([benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks), [repo](https://github.com/unslothai/unsloth)) | Hand-written Triton kernels and manual autograd for attention, MLP, head, norms, RoPE; LoRA, QLoRA, full fine-tuning, RL (GRPO, DPO), FP8 | Llama 3.1 8B and Llama 3.3 70B: 2x speed, >70% / >75% less VRAM vs HF + FA2 (H100 and Blackwell; Alpaca, batch 2, grad-accum 4, r=32, QLoRA on all linear layers) | Same feature space for HF causal LMs. Bantam has no kernels; on a GPU expect it to be slower than HF + FA2, let alone Unsloth, until measured. Bantam's edge is portability: no custom kernels or compiled code of its own |
| **Liger Kernel** ([repo](https://github.com/linkedin/Liger-Kernel), [paper](https://arxiv.org/abs/2410.10989)) | Triton kernels: RMSNorm, RoPE, SwiGLU, fused linear cross-entropy, chunked DPO/ORPO/GRPO losses; plugs into HF Trainer, TRL, Axolotl, LLaMA-Factory | +20% throughput, −60% memory (LLaMA-3-8B, batch 8, bf16, AdamW, gradient checkpointing, FSDP1 on 8×A100) | Bantam's chunked cross-entropy is the same idea in plain PyTorch: it saves memory, not time. On CUDA, Liger is the better choice |
| **Cut Cross-Entropy** ([paper](https://arxiv.org/abs/2411.09009)) | A kernel that computes the loss without materializing the logits | Gemma 2 2B loss memory 24 GB → 1 MB; 1.5-10x larger batches at no speed or convergence cost | Same goal as Bantam's chunked loss, far better on GPU |
| **TRL + PEFT** ([PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora), [TRL memory guide](https://huggingface.co/docs/trl/reducing_memory_usage), [packing with FA2](https://huggingface.co/blog/packing-with-FA2)) | The reference Hugging Face stack: LoRA, DoRA, LoRA+, rsLoRA, PiSSA, LoftQ; QLoRA through bitsandbytes; SFT, DPO, GRPO; best-fit packing; padding-free batching with FA2 | Packing with FA2 position ids: 1.4x throughput ([IBM](https://research.ibm.com/blog/hugging-face-training-flash-attention), [paper](https://arxiv.org/abs/2407.09105)) | Bantam reads and writes PEFT adapter files, its strongest interoperability point. Its block-diagonal mask gives the same isolation as TRL padding-free, but a dense mask in SDPA cannot use FlashAttention's packed kernels. The race opponent (§4) is plain transformers Trainer + PEFT, without TRL |
| **Axolotl** ([docs](https://docs.axolotl.ai/), [multipack](https://docs.axolotl.ai/docs/multipack.html)) | YAML-driven trainer: full, LoRA, QLoRA, GPTQ, QAT, DPO/KTO/ORPO, GRPO; multipacking; FA 2/3/4, Flex, xFormers; Liger and CCE; sequence parallelism; FSDP2, DeepSpeed, multi-node; distributed Muon for FSDP2 pretraining | Few headline numbers | A superset of Bantam's fine-tuning features plus scale-out and RL. It already ships distributed Muon |
| **LLaMA-Factory** ([repo](https://github.com/hiyouga/LLaMA-Factory), [ACL 2024 demo](https://aclanthology.org/2024.acl-demos.38)) | Web UI and CLI over 100+ models; optimizers GaLore, BAdam, APOLLO, Adam-mini, Muon; DoRA, LoRA+, PiSSA, LoftQ; 2-8-bit QLoRA through many back-ends; FA2, Unsloth, Liger | Mostly integrations; its own table: 4-bit QLoRA ~6 GB for 7B | Covers every Bantam fine-tuning and optimizer feature, Muon included. Bantam is smaller and easier to read, not more capable |
| **torchtune** ([repo](https://github.com/pytorch/torchtune), [QLoRA tutorial](https://pytorch.org/torchtune/0.1/tutorials/qlora_finetune.html), [status issue](https://github.com/meta-pytorch/torchtune/issues/2883)) | PyTorch-native recipes; QLoRA through torchao's NF4 tensor (no bitsandbytes); chunked cross-entropy, compile, activation offloading | Llama 3.2 3B on an A100 (batch 2, seq 4096, bf16), step by step: chunked CE −15.8% memory, +2% tokens/s; QLoRA ends at 4.6 GiB. The repository says it is no longer actively maintained | The closest in philosophy (pure PyTorch, hackable). Its bitsandbytes-free QLoRA means Bantam's pure-PyTorch NF4 is not unique. Its wind-down leaves room for a maintained, hackable PyTorch-native trainer |
| **LitGPT** ([repo](https://github.com/Lightning-AI/litgpt)) | 20+ LLMs implemented from scratch; pretrain, fine-tune and deploy CLI; FSDP, 4-bit, LoRA | No headline benchmark found | Same "from scratch, readable" ethos, plus multi-GPU and deployment |
| **bitsandbytes QLoRA / NF4** ([paper](https://arxiv.org/abs/2305.14314)) | NF4 4-bit base weights, double quantization, paged optimizers | 65B fine-tune on one 48 GB GPU at 16-bit-equivalent quality | Bantam re-implements NF4/int4/int8 bases in PyTorch (no double quantization). Its NF4 costs +0.18 loss on SmolLM2-135M before training (88 examples, n=1); not checked against bitsandbytes outputs |
| **DoRA** ([paper](https://arxiv.org/abs/2402.09353)), **LoRA+** ([paper](https://arxiv.org/abs/2402.12354)) | Magnitude/direction split; a larger LR for the B matrices | DoRA +3.4 / +1.0 over LoRA (LLaMA-7B / 13B, commonsense reasoning); LoRA+ 1-2% better and up to ~2x faster | Both are in PEFT and LLaMA-Factory too; in Bantam they are table stakes |

### 2.2 Optimizers and memory

| Tool / method | What it does | Published claims (their conditions) | How Bantam compares |
|---|---|---|---|
| **Muon** ([blog](https://kellerjordan.github.io/posts/muon/)) | Nesterov momentum on hidden 2-D weights, orthogonalized by a quintic Newton-Schulz iteration; AdamW for embeddings, head, scalars | ~1.35x to 3.28 val loss in the NanoGPT speedrun; optimizer FLOP overhead under 1% at that scale | Same algorithm and split. At Bantam's tiny batch (4,096 tokens per step) Newton-Schulz costs about 20% of a step's model FLOPs, not 1%, and Bantam does not charge it to the budget. Bantam's Muon-vs-AdamW gap (2.737 vs 3.332, AdamW n=1) is confounded by a 16x higher LR on the embedding/head ([METHOD.md §2.3](METHOD.md)) |
| **Muon at scale (Moonshot)** ([paper](https://arxiv.org/abs/2502.16982)) | Muon plus AdamW-style weight decay and update-RMS matching | ~2x compute efficiency vs AdamW (about 52% of the FLOPs at compute-optimal) | Bantam uses the same weight decay and RMS-matching scale |
| **Tuned-baseline benchmark** ([Wen et al.](https://arxiv.org/abs/2509.02046)) | Ten optimizers, carefully tuned, 0.1B-1.2B | Matrix optimizers ~1.4x over tuned AdamW at 0.1B, ~1.1x at 1.2B; weak baselines can inflate speedups by up to 2x | A warning for Bantam's own comparisons: its AdamW baselines were tuned over only three (LLM) or four (weather) LRs, one seed each |
| **Polar Express** ([paper](https://arxiv.org/abs/2505.16932)) | Minimax-optimal per-step polynomial for the polar factor | Lower val loss than Newton-Schulz Muon for GPT-2 on 1-10B FineWeb tokens; used in modded-nanogpt since record #38 | Implemented (all 8 coefficient tuples). No gain at 5.8M (2.755 vs 2.737, n=1; that run used the 0.2 coefficient list and is to be rerun). NorMuon and faster Gram Newton-Schulz variants ([blog](https://tridao.me/blog/2026/gram-newton-schulz/)) are not implemented |
| **8-bit Muon** ([paper](https://arxiv.org/abs/2509.23106)) | Blockwise 8-bit Muon momentum | Parity with full-precision Muon up to 2.7B; at 2.7B the optimizer state is ~62% smaller than 32-bit Muon (AdamW part in 32 bits) and ~75% smaller with everything in 8 bits | Same idea as Bantam's `--state_bits 8` with Muon; this paper is the stronger evidence |
| **bitsandbytes 8-bit optimizers** ([paper](https://arxiv.org/abs/2110.02861), [docs](https://huggingface.co/docs/bitsandbytes/v0.43.0/optimizers)) | Blockwise dynamic quantization of Adam state, CUDA kernels | 32-bit-equivalent quality; ~75% optimizer-memory saving | Bantam re-implements the idea in plain PyTorch. Short CPU runs at 5.8M: 8-bit within noise of 32-bit (n=1). Expect it to be slower without fused kernels |
| **torchao low-bit optimizers** ([README](https://github.com/pytorch/ao/tree/main/torchao/optim)) | Pure-PyTorch (+ compile) AdamW8bit, AdamW4bit, AdamWFp8; `bf16_stochastic_round`; a CPU-offload optimizer | Llama2-7B full-bf16 fine-tune on an A100: fused AdamW 51.6 GB / 3,200 tok/s; 8-bit 39.1 GB, 4-bit 33.2 GB at 2,900 tok/s | **The closest existing equivalent of Bantam's optimizer-memory stack.** Bantam's differences: 8-bit Muon in the same package and a memory estimator |
| **Revisiting BFloat16 Training** ([paper](https://arxiv.org/abs/2010.06192)) | Stochastic rounding or Kahan summation for pure-bf16 weight updates | Up to 7% absolute accuracy recovered for 16-bit training; Kahan closes the remaining gap at 2x weight memory | Prior art for Bantam's bf16 stochastic rounding. Bantam has no Kahan option |
| **ZeRO / ZeRO-Offload** ([ZeRO](https://arxiv.org/abs/1910.02054), [Offload](https://arxiv.org/abs/2101.06840)) | Shard model state across GPUs; or keep optimizer state and step on the CPU | Offload: more than 13B parameters on one GPU (one 32 GB V100) | On one large card with enough host RAM this already exceeds 5B. Bantam has no sharding and no offload |
| **GaLore** ([paper](https://arxiv.org/abs/2403.03507)), **APOLLO** ([paper](https://arxiv.org/abs/2412.05270)), **Adam-mini** ([paper](https://arxiv.org/abs/2406.16793)) | Low-rank gradient projection; SGD-like memory with estimated scaling; one LR per Hessian block | GaLore: LLaMA-7B pretraining on a 24 GB RTX 4090; APOLLO: 7B from scratch under 12 GB; Adam-mini: ~50% less optimizer memory | All fit larger models in less memory than Bantam's approach |
| **Memory estimators** ([Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator), [LLaMA-Factory table](https://github.com/hiyouga/LLaMA-Factory)) | Rule-of-thumb calculators | Accelerate: load size and "Adam training" size | `bantam memory` is more detailed (8-bit state, Muon, stochastic rounding, chunked loss), but its activation constants were measured on CPU and it has never been checked against a real GPU |

**The "5B parameters on one 96 GB GPU" scoreboard** ([SCOREBOARD_5B.md](SCOREBOARD_5B.md))
is arithmetic, not a measurement. Model state for standard mixed-precision
AdamW is about 16 bytes per parameter in the
[ZeRO accounting](https://arxiv.org/abs/1910.02054), 80 GB for 5B. Bantam's
bf16 weights with 8-bit state need about 6 bytes per parameter, about 30 GB.
With activations, `bantam memory` says the standard recipe does not fit a 96 GB
[RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
and Bantam's rows do. That is plausible but not distinctive: ZeRO-Offload
trained more than 13B on one 32 GB V100 in 2021, and torchao already ships the
same optimizer-memory techniques. The 0.2 docs gave a 2.56x speedup for this
scoreboard using GB and GiB inconsistently; the corrected figure is about
1.9x, and it is an estimate built on published Muon speedups, not on a Bantam
measurement.

### 2.3 Pretraining speed-ups (Zoom's neighbours)

| Method | What it does | Published claims (their conditions) | How Bantam compares |
|---|---|---|---|
| **Patch-level training** ([Shao et al., ICLR 2025](https://arxiv.org/abs/2407.12665)) | Average K token embeddings into one patch, predict the next patch's tokens with a bag loss, then switch to tokens | ~0.5x total training cost with no quality loss, 370M-2.7B on the Pile (K=4, 2/3 of the data at patch level) | **Closest prior art for Zoom.** Bantam's Zoom uses a learned gate, a per-token split and k=2; at 5.8M the baseline needs 1.12x (n=3) to 1.14x (n=2) the compute to match it. The paper found learned input/output projections hurt its final model and its gain shrinking with size |
| **Hourglass** ([paper](https://arxiv.org/abs/2110.13711)), **Funnel** ([paper](https://arxiv.org/abs/2006.03236)), **dynamic token pooling** ([paper](https://arxiv.org/abs/2211.09761)) | Shorten the sequence inside the network and upsample; learned pooling | Hourglass beats a Transformer at equal compute on enwik8; pooling faster and more accurate than fixed pooling | Zoom is Hourglass-style shortening used only during an early phase. Hourglass is also the reference for causal shifting when tokens are merged |
| **Sequence-length warm-up** ([paper](https://arxiv.org/abs/2108.06084), [Composer card](https://docs.mosaicml.com/projects/composer/en/v0.3.1/method_cards/seq_len_warmup.html)) | Short sequences first, then longer | GPT-3 125M stable with 8x batch and 40x LR; Composer ~1.5x to the same loss (GPT-2 52M) | The main competitor for Zoom. Not compared at equal FLOPs yet |
| **Dataset Decomposition** ([paper](https://arxiv.org/abs/2405.13226)) | Length buckets and a variable-length curriculum | Up to 6x faster to target accuracy | A strong baseline Zoom has not been tested against |
| **modded-nanogpt** ([repo](https://github.com/KellerJordan/modded-nanogpt)) | Community speedrun: GPT-2 small to 3.28 FineWeb val loss on 8×H100 | Record about 40 s (record #92) vs the 45 min llm.c baseline; stack includes Muon (#3), QK-norm (#5), window warm-up (#13), Polar Express (#38), NorMuon (#41), batch-size schedule (#46), sequence-length schedule (#72) | The reference benchmark for any "train faster" claim. Bantam's QK-norm gain agrees in direction with record #5, which adopted QK-norm. Zoom would have to beat the existing sequence and window schedules here; Bantam cannot run this track without GPUs |
| **nanoGPT** ([repo](https://github.com/karpathy/nanoGPT)) | Minimal GPT-2 training code | GPT-2 124M on OpenWebText to ~2.85 val loss in ~4 days on 8×A100-40GB | Same genre as Bantam's small trainer; a baseline, not a competitor |
| **QK-norm** ([Henry et al.](https://arxiv.org/abs/2010.04245), [ViT-22B](https://arxiv.org/abs/2302.05442), [Wortsman et al.](https://arxiv.org/abs/2309.14322)) | Normalize q and k before the dot product | Prevents attention-logit divergence; reduces LR sensitivity | Bantam: 2.639 ± 0.003 vs 2.737 ± 0.012 (n=2 vs 3), 1.24x. It should become the default baseline |

### 2.4 Data tools

| Tool | What it does | Published claims | How Bantam compares |
|---|---|---|---|
| **Megatron indexed dataset** ([docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.datasets.indexed_dataset.html)); nanoGPT/llm.c uint16 `.bin` | Raw token arrays plus an index, memory-mapped, no compression | Format only | TokPack (rank remap + varint + zstd dictionary) is smaller on disk (about 6 bits/token on TinyStories, hand-measured, not committed) but gives up zero-copy random access. Not benchmarked against uint16 + zstd |
| **MosaicML Streaming** ([docs](https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/basic_dataset_conversion.html)) | Sharded columnar format with zstd, streaming from object stores, deterministic resumption | Format and feature claims | MDS already offers compressed, sharded, resumable streaming; TokPack is single-file and in-RAM |
| **datatrove MinHash** ([FineWeb paper](https://arxiv.org/abs/2406.17557), [datatrove](https://github.com/huggingface/datatrove)) | Four-stage MinHash pipeline used for FineWeb | 5-grams, 112 hashes; per-snapshot dedup | Bantam's MinHash suits small corpora; datatrove is the scale reference |
| **Climate-data compression** ([Klöwer et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10766530/)) | Bit-rounding to the real information content plus lossless coding | ~17x vs 64-bit floats, keeping 99% of the information | GridPack is lossy with a per-value error bound (5% of the 6 h change sd); not compared with bit-rounding + zstd |

### 2.5 Weather models

Bantam's weather model is a demo at the coarsest WeatherBench resolution
(5.625°). Its validation metric is a normalized MSE and its RMSEs use 2018 as
both the test year and the year the LRs were tuned on, so they are not
comparable with published scores.

| Model | Published claims | How Bantam compares |
|---|---|---|
| **WeatherBench** baselines at 5.625° ([paper](https://arxiv.org/abs/2002.00469)) | z500 RMSE at 3 / 5 days: persistence 936 / 1033, weekly climatology 816, direct CNN 626 / 757, IFS T42 489 / 743, operational IFS 154 / 334 m²/s²; a ResNet with CMIP pretraining ([Rasp & Thuerey](https://arxiv.org/abs/2008.08626)) 268 at 3 days | Bantam reports 6/24/72 h on a different split. Its 72 h z500 RMSE (539 and 551 for the two Muon baseline seeds, 511 with low-res warm-up, 1 seed) is better than the direct CNN at 3 days (626) and worse than IFS T42 (489), the CMIP-pretrained ResNet (268) and operational IFS (154). The split and conventions differ, so this is not a ranking |
| **ClimaX** ([paper](https://arxiv.org/abs/2301.10343)) | ViT pretrained on CMIP6 at 5.625° and 1.40625°; up to 80 V100s | The closest published analogue of Bantam's 64×32 ViT and the right comparison |
| **Stormer** ([paper](https://arxiv.org/abs/2312.03876)) | Smaller patches consistently improve accuracy; under 24 h on 128 A100s | Predicts a quality cost for merging tokens on a ViT, which matches Bantam's Muon result |
| **GraphCast** ([paper](https://arxiv.org/abs/2212.12794)), **Pangu** ([paper](https://www.nature.com/articles/s41586-023-06185-3)), **FourCastNet** ([paper](https://arxiv.org/abs/2202.11214)), **Aurora** ([paper](https://arxiv.org/abs/2405.13063)) | 0.25° global models trained on tens to hundreds of accelerators for days to weeks | Not in the same league; Bantam's weather code is for experiments on tricks like low-res warm-up |

### 2.6 Feature matrix

"n/v" = not verified for this file.

| Feature | Bantam | Unsloth | Axolotl | TRL + PEFT | LLaMA-Factory | torchtune |
|---|---|---|---|---|---|---|
| LoRA / QLoRA | yes | yes | yes | yes | yes | yes |
| QLoRA without bitsandbytes | yes (pure PyTorch) | n/v | n/v | no (bitsandbytes) | several back-ends | yes (torchao NF4) |
| DoRA / LoRA+ | yes / yes | n/v | n/v | yes / yes | yes / yes | n/v |
| Fused GPU kernels | **no** | yes (Triton) | via Liger / CCE | via Liger | via Liger / Unsloth | compile |
| Packing | block-diagonal mask | n/v | multipack (FA varlen) | padding-free + FA2 | n/v | yes |
| Chunked or fused loss | chunked | CCE | Liger / CCE | Liger | Liger | chunked |
| Multi-GPU (FSDP / DeepSpeed) | **no** (DDP only, tested on CPU) | "supported" | yes | yes | n/v | yes |
| RL / preference (DPO, GRPO) | **no** | yes | yes | yes | n/v | n/v |
| Muon | yes (incl. on LoRA) | n/v | yes (distributed, pretraining) | n/v | yes | n/v |
| Runs on CPU without custom kernels | yes | n/v | n/v | n/v | n/v | n/v |
| Published GPU benchmarks | **none** | yes | few | some | own claims | yes |

---

## 3. Bantam's language model next to published TinyStories models

1,000 clean TinyStories V1 validation stories, bits per byte (lower is
better), every model family scored in its own training boundary convention.
Compute is 6·N·D training FLOPs. (The 0.2 comparison scored the roneneldan
models in a convention that cost them about 25-37 nats per story; fixed in 0.3.)

| model | parameters | training FLOPs | bits per byte |
|---|---|---|---|
| **Bantam ZoomGPT (plain Muon baseline)** | 5.8M | 6.4e13 | **1.029** |
| llama2.c stories260K | 0.26M | 1.0e16 | 1.097 |
| TinyStories-1M | 3.75M | ~2.1e17 | 0.822 |
| TinyStories-3M | 8.28M | ~4.7e17 | 0.677 |
| TinyStories-8M | 19.7M | ~8.8e17 | 0.581 |
| llama2.c stories15M | 15.2M | 3.6e18 | 0.577 |
| TinyStories-28M | 52.0M | ~1.9e18 | 0.538 |
| TinyStories-33M | 68.5M | ~3.9e18 | 0.542 |
| llama2.c stories42M | 41.7M | 5.4e18 | 0.516 |
| llama2.c stories110M | 110M | 1.7e19 | 0.480 (not re-scored in 0.3) |

The training compute of TinyStories-8M and -28M is uncertain (8.8e17-2.0e18 and
1.9e18-4.3e18 in `results/published/compare_clean1000_table.json`); the table
shows the low end. Bantam's row is the 0.3 retrain of the plain Muon baseline
(`save_muon_base_repro`, final val loss 2.7352, one run).

- Bantam reaches stories260K's quality with at least 157x less compute. That
  is an upper-bound match from one completed run, and mostly a model-size
  effect: stories260K has 22x fewer parameters and a 512-token vocabulary, and
  is limited by its capacity, so most of its compute could not help it.
- Bantam is behind every TinyStories model. Its gap to TinyStories-1M is
  0.207 bits per byte; TinyStories-1M used about 3,350x more compute.
- Bantam's run is tiny (under 0.02 epochs of TinyStories). The table says
  where it stands, not how it would do with more compute.

---

## 4. The fine-tuning race: Bantam vs Hugging Face Trainer + PEFT

This is the first direct speed and quality comparison between Bantam and a
standard tool. It runs on the same 4-core CPU, so it says nothing about GPUs.
The harness is `experiments/race_finetune.py`
(`python3 experiments/race_finetune.py --threads 4`).

**Shared by every arm**
- SmolLM2-135M in fp32 on CPU; Dolly-15k, split with `Random(1234)` into 800
  training and 200 held-out ("judge") examples (rows with no response token
  within 512 tokens are skipped: 33); ChatML format; loss on response tokens
  only.
- LoRA r=16, α=32, dropout 0.05 on q, k, v, o, gate, up and down projections
  (210 modules, 4,884,480 trainable parameters, the same in every arm).
- LR 2e-4, sequence 512, micro-batch 2 × gradient accumulation 2, 1 epoch,
  seed 1234, the same thread count everywhere, one model process at a time.

**Arms**
- `base`: no training; judged only, as a reference.
- `hf_peft`: transformers Trainer + peft `LoraConfig`, one example per row with
  dynamic padding, torch AdamW (fp32 state), cosine to 10% of the peak LR, 3%
  warm-up, betas (0.9, 0.95), gradient clip 1.0, no weight decay, SDPA.
  Flush-to-zero is off (as in a stock install).
- `bantam_adamw`: `bantam finetune --optim adamw --state_bits 32 --pack bfd`
  with a cosine schedule: the same torch AdamW as `hf_peft`, plus Bantam's
  packing and chunked loss. This is the like-for-like tooling comparison.
- `bantam`: the Bantam recipe, `--optim muon --pack bfd`: Muon on the LoRA
  matrices, 8-bit state, packing, chunked loss, flush-to-zero, cosine schedule
  (the CLI default is `--sched wsd`; the protocol sets cosine in every arm).
  This arm changes the optimizer too.
- `bantam_nf4`: the `bantam` arm on an NF4-quantized base
  (`--quant nf4 --low_ram 1`, int8 embeddings), judged on its NF4 base and
  also on the float base.

**What is measured**
- Training speed: real (non-padding) trained tokens divided by training-only
  wall time.
- Memory: peak resident memory of the process tree, sampled every 0.2 s.
- Quality: pooled per-token cross-entropy on the response tokens of the 200
  held-out examples, one example at a time in fp32, with bootstrap 95%
  confidence intervals and paired differences against `hf_peft`. Bantam arms
  are judged through their exported PEFT adapters loaded by peft.
- Three fixed prompts with one greedy answer each, for reading.

**Fairness checks built into the harness**
- Token parity: the same token ids in every arm (0 mismatches on all 800
  examples in a dry run).
- Bantam's own eval loss and the judge agree to about 1e-7 on the same
  examples, which also checks that packed examples do not see each other.

**Known asymmetries**
- With 2 rows per micro-batch in every arm, a packed Bantam row holds about
  1,024 real tokens, so Bantam takes about 3.4x fewer, larger optimizer steps
  (58 vs 200) at the same LR.
- Hugging Face's first update has LR 0 (warm-up starts at 0).
- Bantam's "1 epoch" skips the last windows that do not fill a whole optimizer
  step (3 of 235 windows, about 1.2% of supervised tokens); tokens/s counts only
  tokens actually trained.
- The chunked loss saves nothing at this shape (1,024 tokens per micro-batch is
  below the 2,048-token chunk).
- The machine is shared; timing varied by about 10% between identical smoke
  runs, while quality was bit-identical.

**Result.** With the same optimizer and a comparable number of optimizer steps,
Bantam matched Hugging Face Trainer + PEFT's held-out loss (2.259 vs 2.260) 1.36x
faster with 38 % less peak RSS (2.57 vs 4.17 GB), on a 4-core CPU in fp32, one run
per arm. Most of the gain is packing: Hugging Face's dynamic padding computes padding
tokens, and Bantam's packed windows do not. TRL and Axolotl also pack (with
FlashAttention varlen on GPUs), so against them this advantage would shrink or
vanish; Unsloth and Liger add fused kernels that Bantam does not have. Muon on LoRA
did not beat AdamW. Full numbers: `RESULTS.md` section 15.2.

---

## 5. What Bantam is good for today

- **Reading and changing training code.** One small package, pure PyTorch, no
  custom kernels or compiled code of its own (core dependencies: torch, numpy,
  zstandard). Muon, Polar Express, 8-bit state, bf16 stochastic
  rounding, LoRA/QLoRA/DoRA/LoRA+, packing with exact per-example masks,
  chunked loss and compressed token storage are each a few hundred lines.
- **Running on what you have.** Everything works on a CPU, and in principle on
  any PyTorch back-end. That suits teaching, small-scale research and
  prototyping.
- **Testing training-efficiency ideas at small scale.** Runs are FLOP-matched,
  log optimizer FLOPs and exact FLOPs, resume bit-exactly on CPU, and
  `experiments/summarize.py` turns them into compute-to-target tables.
- **Muon on LoRA matrices.** An under-explored niche; the evidence so far is
  thin ([Can Muon fine-tune Adam-pretrained models?](https://arxiv.org/abs/2605.10468)
  reports that LoRA narrows the penalty of switching to Muon;
  [another study](https://arxiv.org/abs/2610.22650) finds Muon competitive
  with AdamW for full SFT up to 14B). Bantam has not tuned it.
- **Moving adapters between tools.** Bantam reads and writes PEFT adapter
  files.

## 6. Where other tools win

- **Speed and memory on GPUs:** Unsloth (Triton kernels), Liger and Cut
  Cross-Entropy (fused losses), FlashAttention variable-length packing (TRL,
  Axolotl). Bantam has none of these and no GPU numbers.
- **Breadth:** Axolotl, LLaMA-Factory and TRL add DPO and GRPO, many
  quantization back-ends, and Liger/CCE integration.
- **Scale-out:** FSDP2, DeepSpeed ZeRO, multi-node. Bantam has single-node DDP
  only, tested with CPU processes.
- **Memory-efficient full training:** GaLore, APOLLO, Adam-mini and
  ZeRO-Offload fit larger models in less memory.
- **Optimizer-memory techniques in plain PyTorch:** torchao already ships
  8-bit/4-bit/FP8 AdamW with bf16 stochastic rounding.
- **Maintenance and testing at scale:** the established tools are used and
  tested on real GPUs by many people.

## 7. What is novel and what is known

Known, implemented by Bantam (credit to the original authors): Muon (Jordan),
the Moonlight update scale and weight decay (Moonshot AI), Polar Express
(Amsel et al.), QK-norm (Henry et al.; Dehghani et al.), blockwise 8-bit
optimizer state (Dettmers et al.), 8-bit Muon (Gupta et al.), bf16 stochastic
rounding (Gupta et al. 2015; Zamirai et al.), LoRA (Hu et al.), QLoRA and NF4
(Dettmers et al.), DoRA (Liu et al.), LoRA+ (Hayou et al.), rsLoRA
(Kalajdzievski), packing without cross-contamination (Krell et al.), chunked
cross-entropy (as in torchtune, Liger, CCE), MinHash deduplication, and
low-resolution warm-up (progressive resizing). See
[RESEARCH.md](../RESEARCH.md) for citations.

Bantam's own:
- **Zoom training**, a learned-gate variant of patch-level training. The
  specific choices (a per-slot learned gate, a per-token split, k=2, a larger
  batch in the merged phase, used with Muon) are worth an ablation and a short
  write-up if they hold at larger scale. They are not a new paradigm. The
  honest claim is: "a learned-merge variant of patch-level training with a
  small CPU-scale gain on TinyStories, untested at scale and on GPU".
- **Engineering in one package:** counter-seeded stochastic rounding that keeps
  data-parallel ranks bit-identical, an optimizer-state format that loads with
  `weights_only=True`, and FLOP-matched experiment tooling. Useful, not
  research contributions.

Negative results worth knowing: the 4→2→1 Zoom staircase Bantam started with
lost to the baseline; Zoom merging hurt the weather model under Muon; ReLU²,
WSD, Polar Express, softcap and zero-init did not help at 5.8M parameters.

## 8. What Bantam needs to be competitive

In priority order:

1. **GPU benchmarks using the competitors' own protocols.** For example
   Unsloth's (Llama-3.1-8B QLoRA, r=32, all linear layers, Alpaca, batch 2,
   grad-accum 4) and torchtune's (Llama-3.2-3B, seq 4096, batch 2, A100).
   Report tokens/s, peak `torch.cuda.max_memory_allocated` and loss parity
   against HF + FA2, Unsloth and torchtune. Measure the 5B scoreboard directly.
2. **Kernels by integration, not reinvention.** Optional FlashAttention
   variable-length packing (or FlexAttention block masks), Liger or CCE for the
   loss, `torch.compile` for the 8-bit update path. Keep the pure-PyTorch path
   as the fallback on CPU and other back-ends; that portability is Bantam's
   real edge.
3. **A rigorous Zoom study.** Zoom against Shao-style averaging,
   sequence-length warm-up and Dataset Decomposition, at equal FLOPs, on a
   QK-norm baseline, with 3 or more seeds, at 124M parameters or more (ideally
   as a modded-nanogpt submission). Report whether the gain persists or
   shrinks with scale, and report wall-clock as well as FLOPs.
4. **Tuned baselines.** Re-tune AdamW (LR, warm-up, weight decay) and give the
   AdamW part of Muon runs its own LR before claiming optimizer or Zoom gains.
   Make QK-norm the default baseline.
5. **Scale-out and post-training.** FSDP2 (or ZeRO) and at least DPO, or a
   clear statement that Bantam is a single-node research kit.
6. **Weather in WeatherBench terms.** Latitude-weighted RMSE and ACC for z500
   and t850 at 3 and 5 days at 5.625°, on a held-out year (`--val_year` now
   exists), against the WeatherBench table and ClimaX.
7. **Data tools.** A bytes/token and decode-speed table for TokPack against
   uint16 + zstd, MosaicML Streaming and Megatron `.bin`, committed as a result
   file.

---

## Sources

Every external claim above links to its source. The underlying research notes
(with the full source list, dated 2026-11-26) were checked against the primary
pages where possible; a few figures (torchao's optimizer table, Unsloth's
benchmark setup, the patch-level training abstract) were re-checked for this
file. Bantam's numbers come from the JSON files under `results/` and from
`results/SUMMARY.md`.
