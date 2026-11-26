# Bantam

**Train and fine-tune small language models and small weather models on limited
compute, in plain PyTorch.**

`pip install bantamtrain` · `import bantamtrain` · command `bantam` · version 0.3.0 ·
MIT license · Made by Nathan from Jamaica.

Bantam is a small, readable toolkit. The core needs only PyTorch, NumPy and
zstandard. One command, `bantam`, covers the whole path:

- **pretrain** a small GPT on a Hugging Face text dataset (or local parquet files) at
  a fixed compute budget;
- **fine-tune** a Hugging Face causal language model with LoRA, QLoRA (a 4-bit NF4
  base, in pure PyTorch, no bitsandbytes) or DoRA, and export a standard PEFT adapter;
- **forecast weather**: train a small 6-hour ERA5 forecaster;
- **plan**: estimate memory before you rent a GPU (`bantam memory`) and print an
  environment report (`bantam doctor`).

**Who it is for.** People who want to see how training works, researchers who test
training ideas at small scale, and anyone with a CPU or one GPU (the GPU paths are
untested).
**Who it is not for (yet).** If you need the fastest GPU fine-tuning, multi-node
training or RL/DPO, use [Unsloth](https://github.com/unslothai/unsloth),
[Axolotl](https://docs.axolotl.ai/), [TRL](https://github.com/huggingface/trl) +
[PEFT](https://github.com/huggingface/peft) or
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Bantam has no fused GPU
kernels and no GPU benchmarks.

**Where every number comes from.** All results below were measured in one CPU-only
sandbox: 4 cores, 16 GB RAM, PyTorch 2.15 CPU build. **Nothing has been run on a GPU
yet.** The models are small: 5.8M parameters for pretraining, 135M for fine-tuning,
1.26M for weather. `n` is the number of seeds, `±` is the sample standard deviation
over seeds, and n = 1 means a single run. Every run's JSON log is in
[`results/`](results/); [`RESULTS.md`](RESULTS.md) has the full tables.

## Contents

- [Results at a glance](#results-at-a-glance)
- [Install](#install)
- [Quickstart: pretrain a small language model](#quickstart-pretrain-a-small-language-model)
- [Quickstart: fine-tune a Hugging Face model](#quickstart-fine-tune-a-hugging-face-model)
- [Quickstart: a small weather model](#quickstart-a-small-weather-model)
- [Check your setup and plan memory](#check-your-setup-and-plan-memory)
- [Stop and resume](#stop-and-resume)
- [Running on a GPU (should work, untested)](#running-on-a-gpu-should-work-untested)
- [The fine-tuning race](#the-fine-tuning-race)
- [Use it from Python](#use-it-from-python)
- [Credit: ideas Bantam implements](#credit-ideas-bantam-implements)
- [How Bantam compares with other tools](#how-bantam-compares-with-other-tools)
- [Limitations and what has not been measured](#limitations-and-what-has-not-been-measured)
- [Reproduce the experiments](#reproduce-the-experiments)
- [Project layout](#project-layout)
- [Documentation](#documentation)

## Results at a glance

| area | headline (conditions below each table) | n |
|---|---|---|
| LLM pretraining | The plain Muon baseline needs **1.41x the compute** to match QK-norm + Zoom's loss (2.589 vs 2.737 at equal FLOPs, so about 29 % less compute for that loss). Zoom alone: 1.12x. QK-norm alone: 1.24x | 2-3 |
| LLM optimizer | Muon 2.737 vs AdamW 3.332 validation loss at equal FLOPs: the biggest single win at this size (confounded: the AdamW-side learning rate also differs 16x, see below) | 3 vs 1 |
| Weather | Muon beats AdamW again (0.124 vs 0.229 normalized MSE). Zoom **hurt** under Muon (0.159). A low-resolution warm-up helped (0.112, about 10 % lower MSE, one seed) | 1-2 |
| Published models | **Behind every TinyStories model** (1.029 vs 0.822 bits/byte for TinyStories-1M). Beats llama2.c stories260K, a 22x smaller model, with at least 157x less compute | 1 run |
| Low memory | 8-bit Muon state + bf16 weights: loss 2.728 vs 2.737, weights + optimizer state 2.7x smaller, 1.6x slower on this CPU | 1 vs 3 |
| Fine-tuning race | Same optimizer and comparable steps as Hugging Face Trainer + PEFT on SmolLM2-135M: **the same held-out loss (2.259 vs 2.260), 1.36x faster, 38 % less peak RAM** (CPU). Muon on LoRA did not beat AdamW. See [the race](#the-fine-tuning-race) | 1 per arm |
| 5B on one 96 GB GPU | fits by arithmetic only (about 6 vs 16-18 bytes per parameter); **not measured** | |

### LLM pretraining

Setup for every row: TinyStories (one shard, a 4,096-token BPE, about 117M training
tokens after deduplication), a GPT with 5.8M parameters (d = 256, 6 layers, 4 heads,
context 256), the same budget of 6.8e13 model FLOPs (Muon's own optimizer FLOPs are
not charged), Muon at learning rate 1.6e-2 unless stated. Score: validation loss in
nats per token on the first 2,048 validation windows (lower is better).

**Compute-to-target** answers one question: how much more compute does the plain
Muon baseline need to reach the same loss? It is read off baseline runs at 1x,
1.25x, 1.5x and 2x the budget (each with its own full LR schedule; interpolation in
log-FLOPs). The baseline runs above 1x are single seeds, so treat the second decimal
as noise.

| recipe | val loss | n | compute-to-target |
|---|---|---|---|
| AdamW baseline (tuned LR 1e-3) | 3.332 | 1 | not on the Muon curve |
| **Muon baseline** | **2.737 ± 0.012** | 3 | 1.00x (reference) |
| Zoom: k=2 for the first 50 % of compute, batch x2 while merged | 2.685 ± 0.004 | 3 | **1.12x** (per seed 1.11-1.13x) |
| Zoom: k=2 for the first 33 %, batch x2 | 2.680 ± 0.005 | 2 | 1.14x |
| QK-norm (`--qk_norm 1`; a known trick) | 2.639 ± 0.003 | 2 | 1.24x |
| **QK-norm + Zoom (k=2 for 33 %, batch x2)** | **2.589 ± 0.006** | 2 | **1.41x** (per seed 1.39-1.42x) |

What this says, plainly:

- **Muon is the biggest single win at this size.** One caveat: in Muon runs, the
  parameters that stay on AdamW (embeddings, the tied output head and the norms:
  1.05M of 5.77M) also get learning rate 1.6e-2. That is 16x the tuned AdamW-only
  rate, so the comparison changes two things at once. The AdamW side is one seed.
- **Zoom gives a small gain that repeats across seeds.** "Zoom" merges each pair of
  neighbouring tokens into one position for the first part of training, then trains
  at full resolution. It is a learned-gate variant of patch-level training
  ([Shao et al., ICLR 2025](https://arxiv.org/abs/2407.12665)), which reports about
  half the training cost at 370M-2.7B parameters. Bantam's gain (1.12-1.14x at 5.8M
  parameters) is real but smaller than that prior art reports.
- **A known trick beat it.** QK-norm alone did better than Zoom alone. Zoom's
  improvement (about -0.05 loss) is the same with and without QK-norm, so the two
  stack. The 1.41x is measured against the plain baseline. There is no QK-norm
  baseline at larger budgets yet, so Zoom's share on top of QK-norm cannot be stated
  in compute.
- **What made Zoom work:** k=2 only (not 4), and a larger batch while merged
  (`--batch_scale 2`, or `--batch_scale_k 1` for batch x k).
- **What did not help** (n = 1 each, against 2.737): WSD schedule 2.773, ReLU² MLP
  2.829, Polar Express orthogonalizer 2.755, logit softcap 15 2.726 (1.02x),
  zero-init projections 2.732 (1.01x), schedules with k=4 (2.80-3.02 without batch
  scaling, 1.02x at best with it), and restarting the LR at the zoom switch 2.760.
  The original 0.1 schedule (k=4, then 2, then 1, no batch scaling) lost on all 3
  seeds: 2.874. Two ablations of that original schedule did no better: a plain mean
  merge instead of the learned gate 2.943, and a bag-of-tokens loss 2.860. The WSD,
  ReLU², softcap and zero-init runs read the older raw token files, whose training
  split still held copies of some validation stories, so treat them as indicative.

### Weather

Setup: ERA5 from WeatherBench2 at 5.625° (a 64 x 32 grid, 6 channels, 6-hour step), a
ViT with 1.26M parameters (d = 128, 6 layers), 5.5e13 FLOPs per run, trained on
2010-2017. Score: normalized mean squared error of the 6-hour forecast on 2018
(lower is better), plus the z500 error after 24 hours of rollout. **Caveat:** the
committed runs had no separate validation year, and the learning rates were also
chosen on 2018. New runs can hold a year out with `--val_year`.

| recipe | val MSE | n | z500 RMSE at 24 h (m²/s²) |
|---|---|---|---|
| AdamW baseline (LR 2e-3) | 0.229 | 1 | 346 |
| **Muon baseline** (LR 1.6e-2) | **0.124 ± 0.001** | 2 | 217 |
| Muon + Zoom merge (k=4 for the first 60 %) | 0.159 ± 0.001: **worse** | 2 | 253-258 |
| **Muon + low-resolution warm-up** (4x fewer grid cells for the first 60 %) | **0.112**: about 10 % lower | 1 | 194 |
| Muon baseline at 2x the compute | 0.094 | 1 | 172 |
| Persistence (the forecast is the current state) | | | 569 |

- Muon is again the biggest win.
- **Zoom does not pay off on weather with Muon.** The other merge schedules were also
  no better than the baseline: k=4 for 40 % 0.129, mean merge 0.128, 16→4→1 0.180
  (n = 1 each).
- The low-resolution warm-up (train on a coarser grid first) is
  [progressive resizing](https://docs.mosaicml.com/projects/composer/en/latest/method_cards/progressive_resizing.html),
  established practice and not a Bantam idea. One seed.
- With AdamW, Zoom (0.180) and the low-resolution warm-up (0.158) both beat the AdamW
  baseline (0.229), but every AdamW run loses to the Muon baseline.
- These are not WeatherBench scorecards. At 5.625° this model is a demo, not a
  competitor to published forecasters.

### Against published TinyStories models

Setup: 1,000 TinyStories V1 validation stories that appear in neither the
TinyStories training split nor llama2.c's training data (756,822 bytes), scored in
bits per byte so different tokenizers compare fairly. Each model family is scored in
the text format it was trained on. (The 0.2 comparison did not do this, and it
favoured Bantam by about 25-37 nats per story against the TinyStories models. 0.3
fixes it.) Training FLOPs are 6 x parameters x tokens, from each model's published
configuration.

The Bantam row is one run of the plain Muon baseline recipe (no QK-norm, no Zoom),
retrained with 0.3 (`results/published/save_muon_base_repro.json`). Its 6.4e13 is
6 x parameters x tokens; the 6.8e13 budget above also counts attention.

| model | params | training FLOPs | bits/byte |
|---|---|---|---|
| **Bantam, plain Muon baseline, one CPU run** | 5.8M | 6.4e13 | **1.029** |
| llama2.c stories260K | 0.26M | 1.0e16 | 1.097 |
| TinyStories-1M | 3.75M | ~2.1e17 | 0.822 |
| TinyStories-3M | 8.3M | ~4.7e17 | 0.677 |
| TinyStories-8M | 19.7M | ~8.8e17 to 2.0e18 | 0.581 |
| TinyStories-28M | 52M | ~1.9e18 to 4.3e18 | 0.538 |
| TinyStories-33M | 68.5M | ~3.9e18 | 0.542 |
| llama2.c stories15M | 15.2M | 3.6e18 | 0.577 |
| llama2.c stories42M | 41.7M | 5.4e18 | 0.516 |
| llama2.c stories110M | 110M | 1.7e19 | 0.480 (0.2 score, not re-scored) |

**Bantam is behind every TinyStories model.** It is 0.207 bits/byte behind
TinyStories-1M, which used about 3,350x more compute. It does beat llama2.c
stories260K with at least 157x less compute. That ratio comes from a single Bantam
run (its full compute is an upper bound on what it needed to reach 1.097), and it is
partly a model-size effect: Bantam's model has 22x more parameters, and stories260K
is limited by its size, not by its training.

### Low-memory training and storage

Same 5.8M model and data as above.

| setting | val loss | weights + optimizer state | n |
|---|---|---|---|
| Muon, fp32 weights and fp32 state (full budget) | 2.737 ± 0.012 | 50.4 MB | 3 |
| Muon, 8-bit state + bf16 weights with stochastic rounding (full budget) | 2.728 | 18.5 MB (**2.7x** smaller) | 1 |

- Short runs (100 steps, 1,024 validation windows): Muon 32-bit 3.572 vs 8-bit 3.581;
  AdamW 32-bit 4.482 vs 8-bit 4.479.
- The 2.728 was scored with 0.2's bf16 arithmetic. The measured error of that is
  about ±0.001, and the run is marked for a rerun.
- This is a memory saving, not a speed-up: on this CPU the bf16 + stochastic-rounding
  path ran 1.6x slower in wall-clock at the full budget (2.6x on the short run).

| stored form | size | quality cost |
|---|---|---|
| int8 weight-only layers in RAM (`qweights`) | 3.9x smaller than fp32 | +0.0002 loss |
| int4 weight-only layers in RAM | 7.5x smaller | +0.026 loss |
| WeightPack bf16 file | 2.9x smaller than an fp32 `torch.save` | +0.00002 loss (round-to-nearest from fp32; bit-exact only for weights already in bf16) |
| TokPack token store (`bantam pack-text`) | 2.68x smaller than uint16 ids (5.97 bits/token) | none (exact) |
| GridPack ERA5 store (`bantam pack-era5`) | 5.2x smaller than float32 | error at most 5 % of each channel's 6-hour-change standard deviation |

The weight rows come from `results/llm_lowmem/save_muon_base_weights.json`. The
TokPack and GridPack rows come from the `pack_stats.json` that `pack-text` and
`pack-era5` write next to the data. They were re-checked by re-packing in 0.3, but
those files are not yet committed under `results/`.

### Fine-tuning

- The **race** against Hugging Face Trainer + PEFT: with the same optimizer and a
  comparable number of optimizer steps, Bantam reached the same held-out loss (2.259
  vs 2.260) 1.36x faster with 38 % less peak RAM; Muon on LoRA did not beat AdamW.
  Details [below](#the-fine-tuning-race).
- **Cost of quantizing the frozen base**, measured before any training on
  SmolLM2-135M (88 held-out Dolly examples, one run each, 0.2 code): int8 +0.003
  loss; NF4 linears with int8 embeddings +0.18; NF4 everywhere +0.34; plain int4
  everywhere +0.52. So int8 is close to free at this size, and 4-bit costs real
  quality on a 135M model. The 0.2 code packed the eval examples into shared windows
  without a block-diagonal mask, so each example could see the ones before it; the
  deltas include that effect and have not been re-measured. The QLoRA paper reports
  16-bit-equivalent fine-tuning with NF4 bases on 7B-65B models; that is not
  measurable here.
- Two 0.2 claims were wrong and are withdrawn. The LoRA runs reported "with Muon"
  actually used 8-bit AdamW: with a frozen base, Muon had nothing to train. In 0.3,
  `--optim muon` with LoRA puts the LoRA matrices on Muon (`--muon_lora` auto). And
  packed fine-tuning let examples see each other: 0.2.0 had no block-diagonal mask at
  all, and the mask added afterwards was ignored by models that use eager or flex
  attention (Gemma-2, for example). 0.3 passes the mask in the form each attention
  implementation applies: boolean for SDPA, additive float for eager and flex.

### Memory for a 5B model on one GPU (arithmetic, not measured)

`bantam memory` estimates memory and time from published figures and small-scale
measurements. For a 5.09B-parameter model on one 96 GB RTX PRO 6000 Blackwell
(81.6 GiB usable after headroom and overhead): standard mixed-precision AdamW keeps
16-18 bytes per parameter of model state (about 82-92 GB, i.e. 76-85 GiB) and does
not fit once activations are added. bf16 weights with stochastic rounding plus 8-bit
optimizer state keep about 6 bytes per parameter (about 30 GB, 28 GiB) and fit.
**None of this has been run on a GPU.** It is also not new: ZeRO-Offload
trained models over 13B on one 32 GB V100 in 2021, and torchao ships 8-bit and 4-bit
AdamW with bf16 stochastic rounding. The 0.2 headline ("2.6x faster") mixed GB and
GiB and is withdrawn. The corrected time ratio is about 1.9x, and it comes from
crediting Muon's published compute saving, not from a Bantam measurement. Details:
[`docs/SCOREBOARD_5B.md`](docs/SCOREBOARD_5B.md).

## Install

```bash
pip install bantamtrain                  # core: torch, numpy, zstandard
pip install 'bantamtrain[data]'          # prepare-text, generate: tokenizers, pyarrow, huggingface_hub
pip install 'bantamtrain[finetune]'      # finetune, chat: transformers, safetensors, datasets, sentencepiece
pip install 'bantamtrain[weather]'       # prepare-era5: xarray, zarr, fsspec, aiohttp
pip install 'bantamtrain[test]'          # pytest and what the test suite imports
```

The quotes stop zsh (the default macOS shell) from treating the brackets as a
pattern. Bantam needs Python 3.10+ and PyTorch 2.4+. If you want a specific PyTorch
build (CPU-only, or a given CUDA version), install it first from
[pytorch.org](https://pytorch.org/get-started/locally/).

```bash
bantam --version
bantam --help                # lists the 11 commands
bantam pretrain --help       # every option of one command
```

`bantam <command>` and `python -m bantamtrain <command>` are the same thing. The full
option reference is [`docs/CLI.md`](docs/CLI.md).

From a source checkout (for development):

```bash
cd bantam
pip install -e '.[data,finetune,weather,test]'
python -m pytest tests -q    # 594 passed, 7 skipped (CUDA / slow) on the CPU sandbox
```

## Quickstart: pretrain a small language model

Needs `pip install 'bantamtrain[data]'`. These four commands run the best measured
recipe. (The committed runs used a TinyStories corpus prepared before 0.3; a fresh
preparation has not been compared with it token for token.)

```bash
# 1. Download one shard of TinyStories, train a 4,096-token BPE, write token files.
bantam prepare-text --hf roneneldan/TinyStories --out data/tinystories --vocab 4096

# 2. Remove duplicate documents and compress the tokens (TokPack, about 2.7x smaller).
bantam pack-text --data data/tinystories

# 3. Train: Muon + QK-norm + Zoom (k=2 for the first third of compute, batch x2 meanwhile).
bantam pretrain --data data/tinystories --budget 6.8e13 \
    --optim muon --lr 1.6e-2 --qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2 \
    --name qk_zoom --out runs/tinystories --save runs/tinystories/qk_zoom.pt

# 4. Sample from the trained model.
bantam generate --model runs/tinystories/qk_zoom_pretrained --prompt "Once upon a time"
```

- Step 3 took about 18 minutes of training per seed (1,082-1,111 s) in the 4-core
  sandbox, at 1 thread with other runs sharing the machine. For a quick smoke test,
  use a smaller budget such as `--budget 5e12`.
- `--budget` is the compute budget in model FLOPs. The schedule (`--zoom`, warmup,
  LR decay) is laid out over that budget, so changing the budget changes the run
  length, not the recipe.
- Without `--optim muon`, pretrain uses AdamW at `--lr 1e-3` (the defaults).
- `runs/tinystories/qk_zoom.json` holds the arguments, the validation curve (loss
  against FLOPs, tokens, steps and wall-clock) and the final loss.
- `--save` writes the weights (`qk_zoom.pt`) and a self-contained directory
  `qk_zoom_pretrained/` (config, weights and tokenizer) that `bantam generate` and
  `ZoomGPT.from_pretrained` read.

Any other text dataset works the same way:

```bash
# another Hub dataset; --column names the text field. Repos without parquet files
# (Dolly is JSONL) are read through the Hub's automatic parquet conversion.
bantam prepare-text --hf databricks/databricks-dolly-15k --column response --out data/dolly_text

# reuse an existing tokenizer instead of training a BPE
# (vocabularies over 65,536 tokens are stored as uint32)
bantam prepare-text --hf roneneldan/TinyStories --tokenizer gpt2 --out data/ts_gpt2

# local parquet files
bantam prepare-text --train part0.parquet part1.parquet --val val.parquet --out data/mine
```

Training documents that also appear in the validation set are dropped by default
(`--drop_val_overlap 1`), so exact copies of validation text cannot lower the
validation loss. `pack-text` also removes duplicates and near-duplicates (MinHash)
within the training set and near-copies of validation documents.

## Quickstart: fine-tune a Hugging Face model

Needs `pip install 'bantamtrain[finetune]'`. Takes a Hugging Face causal LM by Hub id
or local path. Checked on CPU with SmolLM2-135M, and in the test suite with tiny
Llama, GPT-2, Gemma-2, Granite, Cohere and Phi models.

```bash
# LoRA rank 16 on every linear layer (the default), with Muon on the LoRA matrices
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --optim muon --out runs/dolly

# Talk to it
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly \
    --prompt "Give me three tips for staying healthy."
```

CPU fine-tuning is slow. Try `--max_examples 200` first to check that everything works.

The same command with a few flags changed:

```bash
# QLoRA: NF4 base (embeddings int8), loaded tensor by tensor, activation checkpointing
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --optim muon --quant nf4 --low_ram 1 --grad_ckpt 1 --out runs/dolly_nf4
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly_nf4 --quant nf4   # interactive

# DoRA instead of LoRA, or LoRA+ (B matrices get 16x the learning rate)
bantam finetune --model HuggingFaceTB/SmolLM2-135M --dataset databricks/databricks-dolly-15k \
    --max_examples 2000 --dora 1 --out runs/dolly_dora
bantam finetune --model HuggingFaceTB/SmolLM2-135M --dataset databricks/databricks-dolly-15k \
    --max_examples 2000 --lora_plus 16 --out runs/dolly_loraplus

# your own data (JSONL); also save a merged model
bantam finetune --model HuggingFaceTB/SmolLM2-135M --jsonl my_data.jsonl \
    --save_merged 1 --out runs/mine

# full fine-tuning: bf16 weights with stochastic rounding, 8-bit optimizer state
bantam finetune --model HuggingFaceTB/SmolLM2-135M --jsonl my_data.jsonl \
    --lora 0 --param_dtype bf16 --grad_ckpt 1 --optim muon --out runs/full
```

What you get by default: LoRA r=16 (alpha 32, dropout 0.05) on every linear layer
except the output head; 8-bit optimizer state; best-fit packing of examples into
512-token windows with block-diagonal masks (packed examples never see each other);
loss on the response tokens only; chunked cross-entropy (for the usual decoder +
linear-head models the full logits are never built). The default optimizer is AdamW
(8-bit) with a WSD schedule; `--optim muon` is Bantam's recipe.

**Data.** `--dataset` takes a Hub dataset, `--jsonl` a local file. Rows can be
`prompt`/`response`, Dolly-style `instruction`/`context`/`response`, chat `messages`,
or ShareGPT `conversations`. The tokenizer's chat template is used when it has one.
`--eval_frac` (default 5 %) is held out, or pass `--eval_jsonl`.

**Outputs** in `--out`:

- `adapter_config.json` + `adapter_model.safetensors`: a standard PEFT adapter;
- `adapter.pt` + `adapter_meta.json`: the same adapter in Bantam's format;
- the tokenizer, saved with the chat template it was trained with;
- `log.json`: arguments, losses, timings, memory;
- with `--save_merged 1`: `merged/`, a plain `save_pretrained()` model
  (`bantam chat --model runs/mine/merged` works too).

Load the adapter in PEFT (needs `pip install peft`; checked to give the same logits as
Bantam):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = PeftModel.from_pretrained(base, "runs/dolly")
tok = AutoTokenizer.from_pretrained("runs/dolly")   # has the chat template used in training
```

`bantam chat --adapter` also accepts a PEFT adapter directory trained elsewhere. In
the interactive chat, `/reset` clears the conversation and `/exit` quits.

Multi-process training uses torchrun (DistributedDataParallel). This was tested only
with 2 CPU processes:

```bash
torchrun --standalone --nproc_per_node 2 -m bantamtrain.finetune \
    --model HuggingFaceTB/SmolLM2-135M --dataset databricks/databricks-dolly-15k \
    --max_examples 2000 --optim muon --out runs/dolly_ddp
```

## Quickstart: a small weather model

Needs `pip install 'bantamtrain[weather]'`.

```bash
# 1. Download ERA5 from WeatherBench2: 5.625 deg, 6-hourly, 2010-2018, 6 channels
#    (z500, t850, t2m, u10, v10, msl); about 650 MB of float32 fields.
bantam prepare-era5 --out data/era5

# 2. Optional: an error-bounded compressed copy (GridPack, about 5x smaller).
bantam pack-era5 --data data/era5

# 3. Train with Muon and the low-resolution warm-up (the best measured recipe),
#    holding out 2017 for validation (train 2010-2016, test 2018). The measured
#    run had no validation year (it trained on 2010-2017), so expect a different score.
bantam weather --data data/era5 --val_year 2017 --budget 5.5e13 \
    --optim muon --lr 1.6e-2 --zoom 4:0.6,1:0.4 --coarse_mode lowres \
    --name lowres --out runs/era5 --save runs/era5/lowres.pt

# 4. Roll the trained model forward 12 steps (3 days) from a start time.
bantam weather --forecast --load runs/era5/lowres.pt --init 2018-03-01T00 --steps 12 \
    --forecast_out runs/era5/forecast.npy
```

- `--coarse_mode lowres` trains on an area-averaged grid with 4x fewer cells for the
  first 60 % of compute. `--coarse_mode merge` (the default) is Zoom's token merge,
  which did worse than the baseline under Muon.
- With `--val_year`, the periodic evaluations use the validation year, and the test
  year is scored only once, at the end. The committed results did not use it.
- The run JSON reports normalized MSE, latitude-weighted RMSE for z500, t850 and t2m
  at 6, 24 and 72 hours, persistence and climatology baselines, and latitude-weighted
  ACC. (The committed runs predate the climatology and ACC fields.)
- The forecast file is a float32 array (steps + 1, channels, lat, lon) in physical
  units, with a `.json` sidecar of valid times and errors where the data covers them.

## Check your setup and plan memory

```bash
bantam doctor            # versions, CUDA/MPS, threads, RAM, disk, which extras import
bantam doctor --json     # the same, for a bug report
```

`bantam memory` estimates memory and time for one GPU. It is arithmetic from published
figures and small-scale measurements, not a measurement:

```bash
bantam memory                                          # 5B model on one 96 GB card
bantam memory --params 1e9 --gpu_gb 24 --tokens 2e10   # your size, card and token count
bantam memory --finetune --params 7e9 --gpu_gb 24      # full / LoRA / QLoRA fine-tuning table
bantam memory --markdown                               # as a Markdown table
```

For example, the fine-tuning table says that a 7B QLoRA run (rank 16) fits a 24 GB
card only with activation checkpointing (about 11 GiB). GPU memory is in GiB, and
`--headroom` (default 0.85) keeps a safety margin.

## Stop and resume

All three trainers write resumable checkpoints: model, optimizer state, data
position and random-number state. On CPU, a resumed run matches an uninterrupted one
bit for bit. The test suite checks this for all three (for finetune also with 2 DDP
processes). The 0.3 end-to-end review also checked pretrain by hand after `kill -9`
and with 2 DDP processes (gloo, CPU).

```bash
# pretrain: checkpoint every 100 steps to <out>/<name>.ckpt
bantam pretrain ... --name qk_zoom --out runs/tinystories --ckpt_every 100
# after an interruption: the same command plus --resume
bantam pretrain ... --name qk_zoom --out runs/tinystories --ckpt_every 100 \
    --resume runs/tinystories/qk_zoom.ckpt

# finetune: checkpoint every 50 steps to <out>/checkpoint; resume from the out dir
bantam finetune ... --out runs/dolly --save_every 50
bantam finetune ... --out runs/dolly --save_every 50 --resume runs/dolly

# weather: like pretrain
bantam weather ... --name lowres --out runs/era5 --ckpt_every 200 --resume runs/era5/lowres.ckpt
```

A resume refuses changed model, data-order or schedule options (and, for pretrain and
finetune, a different number of processes), because they would silently change the
run. `--allow_arg_change` (pretrain) overrides this. In pretrain, a run whose loss
becomes NaN or infinite stops with exit code 3, and its JSON says
`"status": "diverged"`.

## Running on a GPU (should work, untested)

**Nothing in Bantam has run on a GPU yet.** The CUDA code paths exist and are tested
only with CPU stand-ins. What should work:

- `--device auto` (the default) picks CUDA when it is available.
- `--precision auto`: bf16 autocast on compute capability 8.0 and newer (Ampere+), fp16
  with a gradient scaler on older cards. TF32 matmuls are on. CPU runs are always fp32.
- Attention uses PyTorch's `scaled_dot_product_attention`, which picks
  FlashAttention-style kernels when it can.
- `--compile` turns on `torch.compile` (tested on CPU only).
- `bantam pretrain --auto_batch` picks the largest micro-batch that fits and keeps
  batch x accumulation, so the schedule and data order do not change.
- torch's fused AdamW is used when the optimizer state is 32-bit and the weights are
  fp32. 8-bit state and bf16 weights use Bantam's own pure-PyTorch code, which has no
  fused kernels.
- Multi-GPU: `torchrun --standalone --nproc_per_node N -m bantamtrain pretrain ...`
  and `-m bantamtrain.finetune ...` (DDP; tested only with 2 CPU processes over gloo).
  There is no FSDP or ZeRO, so the model, its gradients and the optimizer state must
  fit on every GPU.
- Power: prefer a hardware cap (`sudo nvidia-smi -pl <watts>`). `--duty 0.75` is a
  software fallback that idles between steps.

Not there: FP8 (row 4 of the 5B scoreboard is an estimate only), fused kernels
(Liger, Cut Cross-Entropy, FlashAttention varlen), a bitsandbytes backend, and
`--ce_chunk` for pretraining (fine-tuning has chunked cross-entropy on by default).
Packed fine-tuning uses a dense block-diagonal mask, which rules out FlashAttention's
fastest path, so expect it to be slower than padding-free packing in TRL or Axolotl.
A good first run on a real card: `bantam doctor`, a short `bantam pretrain` with a
small `--budget`, then compare the peak memory with `bantam memory`.

## The fine-tuning race

A head-to-head run of Hugging Face Trainer + PEFT against `bantam finetune`, on the
same model, data (one epoch) and schedule, measuring speed, memory and quality. It is running on
the 4-core CPU sandbox now. The harness is
[`experiments/race_finetune.py`](experiments/race_finetune.py):

```bash
python experiments/race_finetune.py --threads 4    # add --online if SmolLM2 and Dolly are not in the HF cache
```

It runs offline by default and needs about 6 GB of free RAM. Rerunning the same
command resumes; `--force <arm>` redoes one arm.

**Protocol (every arm).**
- Model and data: SmolLM2-135M in fp32 on CPU. Dolly-15k rows shuffled with seed
  1234: 800 for training, 200 held out for judging (rows with no response token
  within 512 tokens are skipped). ChatML chat format; loss on response tokens only.
- Adapters: LoRA r=16, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down_proj. That is
  210 modules and 4,884,480 trainable parameters, identical in every arm.
- Training: LR 2e-4, 3 % warmup then cosine to 10 % (Bantam's CLI default is WSD;
  the race uses cosine in every arm), AdamW betas (0.9, 0.95), no weight decay,
  gradient clipping at 1.0, sequence length 512, micro-batch 2 x accumulation 2,
  1 epoch, seed 1234, 4 threads.
- Bantam arms evaluate on the first 32 judge examples (`--eval_jsonl`) instead of
  holding 5 % of the 800 out of training; these evals are not counted in train time.

**Arms.**

| arm | what it is |
|---|---|
| `base` | no training: the judge's reference |
| `hf_peft` | transformers 5.24 Trainer + peft 0.22, one example per row, dynamic padding, torch AdamW |
| `bantam_adamw` | `bantam finetune --optim adamw --state_bits 32 --pack bfd`: the same torch AdamW as `hf_peft`, so the difference is the training loop (packing, chunked CE) |
| `bantam` | Bantam's recipe: Muon on the LoRA matrices, 8-bit state, packing with block-diagonal masks, chunked CE |
| `bantam_nf4` | the same on an NF4 base (QLoRA, `--quant nf4 --low_ram 1`, int8 embeddings) |

**What is measured.**
- Speed: real (non-padding) trained tokens per second of training wall time.
- Memory: peak RSS of each arm's process tree, sampled every 0.2 s.
- Quality: a separate judge process scores every final model the same way. It
  computes pooled per-token cross-entropy over the response tokens of the 200
  held-out examples, one example at a time, unpacked, in fp32, with bootstrap 95 %
  intervals and paired differences against `hf_peft`. Bantam's float-base adapters
  are judged through their PEFT export loaded by real `peft`; the NF4 arm is judged on
  its NF4 base plus `adapter.pt`, and also on the float base. Each arm also answers
  three fixed prompts greedily.

**Fairness checks.** Token-level parity of the training data across arms; the same
LoRA modules and parameter counts; Bantam's own evaluation and the judge agree to
about 1e-7. Known differences, stated up front:
- Packed micro-batches hold about 1,024 real tokens, so Bantam takes about 3.4x
  fewer, larger optimizer steps at the same learning rate.
- Bantam's "1 epoch" leaves out the last packed windows that do not fill a whole
  optimizer step (about 1.2 % of supervised tokens here). The race counts only the
  tokens each arm actually trained on.
- At this shape, chunked cross-entropy saves nothing (one chunk holds the whole
  micro-batch).
- Bantam turns on flush-to-zero on CPU; the Hugging Face arm runs without it (the
  harness can switch it on with `--hf_flush_denormal 1`).

Follow-up arms (same protocol, overrides in brackets): `bantam_adamw_steps`,
`bantam_muon_steps` and `bantam_nf4_adamw_steps` [micro-batch 1 x accumulation 1,
so one packed window (about 3.4 examples) per optimizer step: 235 steps, close to
`hf_peft`'s 200 steps of 4 examples], `bantam_muon_steps_lr1e-3` and
`bantam_adamw_steps_lr1e-3` [the same, LR 1e-3].

**Results** (full run, 4 threads, nothing else running):

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

## Use it from Python

Everything the commands do is importable. A minimal training loop (tested):

```python
import torch
from bantamtrain.models import ZoomGPT
from bantamtrain.optim import build_optimizer
from bantamtrain.zoom import ZoomSchedule, lr_at

model = ZoomGPT(vocab=4096, d=256, n_layer=6, n_head=4, zoom_ks=[2], qk_norm=True)
opt = build_optimizer(model, "muon", lr=1.6e-2, weight_decay=0.1, state_bits=8)
sched = ZoomSchedule.parse("2:0.33,1:0.67")    # k=2 for the first 33% of the budget

x = torch.randint(0, 4096, (8, 258))           # stand-in for real token windows
for frac in (0.1, 0.5):                         # fraction of the compute budget spent
    k = sched.k_at(frac)
    for g in opt.param_groups:
        g["lr"] = lr_at(frac, 1.6e-2) * g.get("lr_scale", 1.0)
    loss = model(x[:, :256], x[:, k:256 + k], k=k)   # targets: the inputs shifted by k
    opt.zero_grad(); loss.backward(); opt.step()

model.save_pretrained("my_model")               # config.json + weights; zoom gates stripped
model = ZoomGPT.from_pretrained("my_model")
tokens = model.generate(torch.tensor([[0]]), 50)   # KV-cached sampling
```

Other building blocks:

```python
from bantamtrain import qweights, weightpack, lora

weightpack.save("model.zwpk", model.state_dict(), bits=8)   # compressed checkpoint (32/16/8/4)
state = weightpack.load("model.zwpk")
qweights.quantize_model(model, bits=8)   # int8 weight-only layers in RAM ("nf4" and 4 too)
# lora.apply_lora(...), lora.save_peft_adapter(...), lora.load_peft_adapter(...)
# bantamtrain.losses.chunked_cross_entropy(...), bantamtrain.data.tokpack, .gridpack, .dedup
```

## Credit: ideas Bantam implements

Most of Bantam is other people's ideas, re-implemented in plain PyTorch.

- **Zoom** is a learned-gate variant of **patch-level training**
  ([Shao et al., ICLR 2025](https://arxiv.org/abs/2407.12665)). Shao et al. average
  K=4 token patches with no new parameters and predict the next patch; Bantam uses a
  learned per-slot gate, a per-token output, k=2 and a larger batch while merged.
  It is also related to **Hourglass** ([Nawrot et al. 2022](https://arxiv.org/abs/2110.13711)),
  **Funnel-Transformer** ([Dai et al. 2020](https://arxiv.org/abs/2006.03236)),
  **sequence-length warm-up** ([Li et al. 2022](https://arxiv.org/abs/2108.06084)) and
  **token dropping** ([Hou et al. 2022](https://aclanthology.org/2022.acl-long.262)).
  The weather low-resolution warm-up is **progressive resizing**.
- **Muon** ([Keller Jordan et al. 2024](https://kellerjordan.github.io/posts/muon/)),
  with Moonlight's update scaling ([Liu et al. 2025](https://arxiv.org/abs/2502.16982)),
  the **Polar Express** orthogonalizer ([Amsel et al.](https://arxiv.org/abs/2505.16932)),
  **8-bit Muon** ([Gupta et al.](https://arxiv.org/abs/2509.23106)).
- **QK-norm** ([Henry et al. 2020](https://arxiv.org/abs/2010.04245);
  [Dehghani et al. 2023](https://arxiv.org/abs/2302.05442)). QK-norm, ReLU², logit
  softcap and zero-init projections are all used in the
  [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun. The WSD
  (warmup-stable-decay) schedule follows [MiniCPM](https://arxiv.org/abs/2404.06395).
- **8-bit blockwise optimizer state** ([Dettmers et al. 2022](https://arxiv.org/abs/2110.02861),
  bitsandbytes) and **bf16 weights with stochastic rounding**
  ([Zamirai et al. 2020](https://arxiv.org/abs/2010.06192); also in
  [torchao](https://github.com/pytorch/ao/tree/main/torchao/optim)).
- **LoRA** ([Hu et al.](https://arxiv.org/abs/2106.09685)), **QLoRA / NF4**
  ([Dettmers et al.](https://arxiv.org/abs/2305.14314)), **DoRA**
  ([Liu et al.](https://arxiv.org/abs/2402.09353)), **LoRA+**
  ([Hayou et al.](https://arxiv.org/abs/2402.12354)), **rsLoRA**
  ([Kalajdzievski](https://arxiv.org/abs/2312.03732)), and the **PEFT** adapter format.
- Packing SFT examples while keeping them apart (block-causal masks in torchtune,
  padding-free FlashAttention packing in TRL and Axolotl) and chunked or fused
  cross-entropy (torchtune, Liger Kernel, Apple's Cut Cross-Entropy) are standard.
  MinHash deduplication follows the usual MinHash-LSH recipe (as in datatrove /
  FineWeb).
- Data: [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li),
  [llama2.c](https://github.com/karpathy/llama2.c) models (Karpathy) and
  [WeatherBench 2](https://arxiv.org/abs/2308.15560) ERA5 (Rasp et al.).

What is Bantam's own is the particular combination: the Zoom design above, the
TokPack and GridPack file formats (built from standard compression steps), and the
tooling that ties it together. All of it is small-scale and CPU-tested.

## How Bantam compares with other tools

The figures in this table are each project's own published claims, not reproduced
here. Bantam's speed and memory on a GPU are unknown until it runs on one. The full
sourced comparison is in [`docs/COMPARISON.md`](docs/COMPARISON.md).

| if you need | consider | why |
|---|---|---|
| the fastest LoRA / QLoRA on NVIDIA GPUs | Unsloth | Triton kernels; reports 2x speed and over 70 % less VRAM than HF + FlashAttention-2 (H100) |
| multi-GPU (FSDP, DeepSpeed), DPO / GRPO, many model families | Axolotl, TRL + PEFT, LLaMA-Factory | scale-out and post-training; LLaMA-Factory and Axolotl already ship Muon |
| memory-saving fused kernels | Liger Kernel, Apple Cut Cross-Entropy | Liger reports +20 % throughput and -60 % memory (8 x A100) |
| PyTorch-native recipes and low-bit optimizers | torchtune (no longer actively maintained), torchao | pure-PyTorch QLoRA and 8-bit / 4-bit AdamW already exist there |
| a small, readable kit for a CPU or one GPU, Muon on LoRA, PEFT-compatible output, experiments you can rerun on a laptop | Bantam | few dependencies, everything measured on CPU |

## Limitations and what has not been measured

- **No GPU numbers.** Speed and memory against Unsloth, Liger, torchtune, Axolotl or
  plain Hugging Face on a GPU are unknown. The 5B scoreboard is arithmetic.
- **Small scale.** One pretraining size (5.8M), one fine-tuning model (135M), one
  weather model (1.26M). Nothing tells us the gains hold at larger scale. The closest
  prior art (patch-level training) found its own advantage shrinking with model size.
- **Few seeds.** Headline rows have 2-3 seeds; most ablations have 1.
- **Baselines may be under-tuned.** The learning rate was tuned on the baseline and
  reused for Zoom. In Muon runs the AdamW-side parameters get 16x the tuned AdamW
  rate. The AdamW baselines are single runs. Weak baselines can inflate claimed
  speed-ups by up to 2x ([Wen et al. 2025](https://arxiv.org/abs/2509.02046)).
- **FLOP-matched, not time-matched.** The budget counts model FLOPs with the standard
  formula. Muon's own optimizer FLOPs (a few % at this size) are not charged. 0.3 logs
  them (`opt_flops`) and an exact count (`model_flops_exact`) for new runs; the
  committed runs predate these fields. On this CPU, merged steps without a larger
  batch cost more wall-clock per FLOP, because the optimizer step is a fixed cost per
  step; with batch x k they cost about the same. The QK-norm + Zoom runs took about
  as long as the baseline runs made alongside them (1,082-1,111 s vs 1,039-1,067 s,
  1 thread each). Weather wall-clock on this shared machine was noisy: the Zoom runs
  took 8 % (AdamW) and 33-52 % (Muon) more than their baselines, and the Muon
  low-resolution run 24-45 % more, so its 10 % MSE gain is at equal FLOPs, not equal
  time.
- **Evaluation sets.** LLM final losses use the first 2,048 validation windows. The
  periodic curves use the first 256, which read about 0.28 lower, so compare finals
  with finals.
- **Weather.** The committed runs had no validation year, and their learning rates
  were chosen on the test year. The scores are normalized MSE at 5.625°, not
  WeatherBench scorecards.
- **Fine-tuning.**
  - "1 epoch" leaves out the last packed windows that do not fill a whole optimizer
    step (up to batch x accum - 1 windows).
  - With packing, sliding-window attention layers (Gemma-2/3, Mistral, Phi-3,
    Starcoder2) lose their window for examples longer than it. At the default
    `--seq 512` none of these is affected.
  - `--quant` without `--low_ram 1` loads the float model first, so peak RAM is higher
    than a float LoRA run. `--low_ram 1` needs safetensors weights.
  - `bantam chat --quant` always uses int8 embeddings, whatever `--embed_quant` the
    adapter was trained with.
  - Importing `bantamtrain.finetune` turns on flush-to-zero for the whole Python
    process (so does importing the pretrain and weather command modules).
- **Not implemented:** FSDP / ZeRO, RL and preference training (DPO, GRPO), fused GPU
  kernels, FP8.

## Reproduce the experiments

From a source checkout (`cd bantam`). The queue files hold the exact commands of the
committed runs, with this sandbox's data paths: most set `D=` (and `W=` for weather)
near the top, while the sweep files (`llm_sweep*.txt`, `sweep3.txt`,
`weather_sweep*.txt`) write `--data /home/user/data/...` on every line. Point them
at your prepared data.

```bash
python experiments/run_queue.py experiments/queues/llm_main.txt --workers 2 --threads 2 --dry_run
python experiments/run_queue.py experiments/queues/llm_main.txt --workers 2 --threads 2
python experiments/summarize.py --plots   # -> results/SUMMARY.md, results/summary.json, results/plots/
```

`run_queue.py` skips finished runs and re-runs unfinished ones, resuming from a
checkpoint when the run wrote one (`--ckpt_every`). The queues behind the results:

| queue | what it runs |
|---|---|
| `llm_sweep.txt`, `llm_sweep2.txt`, `sweep3.txt` | LR sweeps at 1/3 budget (AdamW and Muon; `sweep3` also has weather) |
| `llm_main.txt` | Muon vs AdamW, baseline vs the original 4→2→1 Zoom (Muon: 3 seeds), the first Zoom ablations, two longer baselines |
| `followup.txt`, `zoom_confirm.txt` | Zoom ablations, and k=2 for 50 % with batch x2 (3 seeds). 7 LLM and 3 weather lines of `followup.txt` were never run |
| `compute_to_target.txt` | the baseline at 1.25x, 1.5x and 2x the budget |
| `upgrades.txt` | QK-norm, WSD, ReLU², softcap, zero-init, Polar Express, Zoom variants |
| `lowmem.txt`, `lowmem2.txt` | 8-bit state and bf16 weights; the saved model for the storage tests |
| `published_cmp.txt` | a long run for the published comparison (stopped at 30 %, not used) |
| `weather_sweep.txt`, `weather_sweep4.txt`, `weather_main.txt` | weather LR sweeps and the main weather matrix |

Six committed runs are not in any queue file: the baseline replicates `up_base_s0`
and `up_base_s1`, the seed-1 runs `up_qknorm_s1` and `up_zoom_2-33_bs2_s1`, and both
QK-norm + Zoom runs. The QK-norm + Zoom command (seed 0 shown; the committed runs
used `--threads 1`):

```bash
bantam pretrain --data data/tinystories --budget 6.8e13 --optim muon --lr 1.6e-2 \
    --qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2 --seed 0 \
    --name up_qknorm_zoom_s0 --out results/llm --threads 1
```

The published comparison (it downloads the pinned validation split and the
published models). This scores your own run from the quickstart; the committed table
scored the plain Muon baseline (`save_muon_base_repro`) and the published models in
[the results section](#against-published-tinystories-models):

```bash
python experiments/compare_published.py --stories 1000 --ctx_cap 512 \
    --ours runs/tinystories/qk_zoom.json:runs/tinystories/qk_zoom.pt \
    --hf roneneldan/TinyStories-1M roneneldan/TinyStories-3M \
    --out runs/compare.json
python experiments/plot_published.py runs/compare.json --png runs/compare.png
```

## Project layout

```
bantam/
  bantamtrain/              the package (what pip installs)
    cli/                    one module per command: pretrain, weather, prepare_text, pack_text,
                            prepare_era5, pack_era5, generate, chat, doctor
    finetune.py             bantam finetune: LoRA / QLoRA / DoRA / full SFT of HF causal LMs
    memory.py               bantam memory: the memory and time estimator
    models/gpt.py           ZoomGPT: RoPE, tied head, QK-norm, ReLU², softcap, KV-cache generate
    models/vit_weather.py   ZoomViT: a ViT on the lat/lon grid, residual 6 h forecaster
    zoom.py                 SlotMerge / SlotUnmerge (gate | mean | linear), ZoomSchedule, LR schedules
    optim.py                Muon (quintic / Polar Express), low-memory AdamW, build_optimizer
    lowmem.py               8-bit blockwise state, bf16 stochastic rounding
    lora.py                 LoRA / DoRA adapters, merging, PEFT save and load
    qweights.py             int8 / int4 / NF4 weight-only layers, low-RAM quantized loading
    weightpack.py           compressed checkpoint files (fp32 / bf16 / int8 / int4)
    losses.py               chunked cross-entropy
    device.py               CPU flush-to-zero, AMP, TF32, compile, batch finder, duty limiter
    data/                   TokPack token store, MinHash dedup, GridPack weather store
  llm/, weather/            old entry points: llm/train.py = bantam pretrain, and so on
  experiments/              run_queue.py, queues/, summarize.py, plots, compare_published.py,
                            race_finetune.py
  tests/                    pytest suite (CPU)
  results/                  the JSON log of every run, SUMMARY.md, plots/
  docs/                     METHOD, GUIDE, CLI, COMPARISON, SCOREBOARD_5B
```

## Documentation

- [`docs/GUIDE.md`](docs/GUIDE.md): a longer walk-through of each workflow.
- [`docs/CLI.md`](docs/CLI.md): every command and option.
- [`RESULTS.md`](RESULTS.md): all measured results, with conditions.
- [`docs/METHOD.md`](docs/METHOD.md): how Zoom works, and the FLOP accounting.
- [`docs/COMPARISON.md`](docs/COMPARISON.md): Bantam against other tools and prior art,
  with sources.
- [`docs/SCOREBOARD_5B.md`](docs/SCOREBOARD_5B.md): the 5B-on-one-GPU estimate.
- [`CHANGELOG.md`](CHANGELOG.md): what changed in each version, including the 0.2
  claims that 0.3 corrects.

## License

MIT. Made by Nathan from Jamaica.
