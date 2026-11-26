# Bantam (`bantamtrain`)

**Train and fine-tune small language models and small weather models on limited
compute, in plain PyTorch.**

Made by Nathan from Jamaica. MIT license. Version 0.3.0.

Bantam is a small, readable toolkit. The core needs only PyTorch, NumPy and
zstandard. One command, `bantam`, covers the whole path:

- **pretrain** a small GPT on any Hugging Face text dataset at a fixed compute budget;
- **fine-tune** any Hugging Face causal language model with LoRA, QLoRA (a 4-bit NF4
  base in pure PyTorch, no bitsandbytes) or DoRA, and export a standard PEFT adapter;
- **forecast weather**: train a small 6-hour ERA5 forecaster;
- **plan**: estimate GPU memory (`bantam memory`) and print an environment report
  (`bantam doctor`).

It is meant for learning how training works, for testing training ideas at small
scale, and for a CPU or a single GPU. It is not a replacement for Unsloth, Axolotl,
TRL + PEFT or LLaMA-Factory on GPUs: Bantam has no fused GPU kernels, no multi-node
training, no RL/DPO, and no GPU benchmarks yet.

**Where the numbers come from.** Every result below was measured in one CPU-only
sandbox (4 cores, 16 GB RAM, PyTorch 2.15 CPU build). **Nothing has been run on a GPU
yet.** `n` is the number of seeds; `±` is the sample standard deviation.

## Install

```bash
pip install bantamtrain                  # core: torch, numpy, zstandard
pip install 'bantamtrain[data]'          # prepare-text, generate (tokenizers, pyarrow, huggingface_hub)
pip install 'bantamtrain[finetune]'      # finetune, chat (transformers, safetensors, datasets, sentencepiece)
pip install 'bantamtrain[weather]'       # prepare-era5 (xarray, zarr, fsspec, aiohttp)
pip install 'bantamtrain[test]'          # pytest and what the test suite imports
```

The quotes keep zsh from treating the brackets as a pattern. Python 3.10+ and
PyTorch 2.4+. `bantam --help` lists the 11 commands; `bantam <command> --help` lists
a command's options. `python -m bantamtrain <command>` works the same way.

## Results at a glance

**LLM pretraining.** A 5.8M-parameter GPT (d=256, 6 layers, context 256) on
TinyStories (4,096-token BPE, about 117M training tokens), every run at the same
budget of 6.8e13 model FLOPs. Validation loss in nats per token (lower is better).
"Compute-to-target" is how much more compute the plain Muon baseline needs to reach
the same loss (read off baseline runs at 1x-2x the budget, which are single seeds).

| recipe | val loss | n | compute-to-target |
|---|---|---|---|
| AdamW baseline (tuned LR) | 3.332 | 1 | |
| Muon baseline | 2.737 ± 0.012 | 3 | 1.00x |
| Zoom: k=2 for the first 50 % of compute, batch x2 | 2.685 ± 0.004 | 3 | 1.12x |
| QK-norm (a known trick) | 2.639 ± 0.003 | 2 | 1.24x |
| QK-norm + Zoom (k=2 for 33 %, batch x2) | 2.589 ± 0.006 | 2 | **1.41x** (about 29 % less compute) |

Muon is the biggest single win at this size (one caveat: the AdamW-side parameters
in Muon runs get 16x the tuned AdamW learning rate). Zoom, which merges pairs of
neighbouring tokens for the first part of training, gives a small gain that repeats
across seeds. It is a learned-gate variant of patch-level training (Shao et al.,
ICLR 2025, which reports bigger savings at 370M-2.7B parameters). QK-norm alone beat
Zoom alone, and the two stack. Several upgrades did not help (WSD schedule, ReLU²,
Polar Express, the original k=4 Zoom schedules).

**Weather.** ERA5 at 5.625°, a 1.26M-parameter ViT, 5.5e13 FLOPs per run, 6-hour
forecast MSE on 2018 (the learning rates were also chosen on 2018; new runs can hold
out a validation year).

| recipe | val MSE | n |
|---|---|---|
| AdamW baseline | 0.229 | 1 |
| Muon baseline | 0.124 ± 0.001 | 2 |
| Muon + Zoom merge (k=4 for 60 %) | 0.159 ± 0.001: **worse** | 2 |
| Muon + low-resolution warm-up (progressive resizing, a known method) | 0.112 | 1 |

**Against published models.** 1,000 TinyStories validation stories that no compared
model trained on, in bits per byte, each model family in its own text format. Bantam
(one CPU run, 6.4e13 FLOPs) scores 1.029. That is **behind every TinyStories model**:
TinyStories-1M scores 0.822 with about 3,350x more compute, and TinyStories-33M 0.542.
It beats llama2.c stories260K (1.097) with at least 157x less compute, but that model
is 22x smaller.

**Low memory.** 8-bit Muon state + bf16 weights with stochastic rounding: loss 2.728
vs 2.737 (n = 1), weights + optimizer state 50.4 MB → 18.5 MB (2.7x smaller). On this
CPU it ran 1.6x slower in wall-clock.

**Quantized bases.** On SmolLM2-135M before any training (88 held-out Dolly
examples, 0.2 code): int8 costs +0.003 loss, NF4 with int8 embeddings +0.18. On a
model this small, 4-bit costs real quality.

**5B on one 96 GB GPU.** About 6 bytes per parameter of model state with bf16
weights and 8-bit state, against 16-18 for standard mixed-precision AdamW. This is
arithmetic from `bantam memory`, **not measured**.

**Fine-tuning race** (Hugging Face Trainer + PEFT against `bantam finetune` on
SmolLM2-135M and 800 Dolly examples, same LoRA modules, schedule and budget; speed,
peak memory, and held-out loss judged the same way for every arm):

<!-- RACE_RESULTS -->

## Quickstart

### Pretrain a small language model

```bash
pip install 'bantamtrain[data]'
bantam prepare-text --hf roneneldan/TinyStories --out data/tinystories --vocab 4096
bantam pack-text --data data/tinystories          # dedup + compressed token store
bantam pretrain --data data/tinystories --budget 6.8e13 \
    --optim muon --lr 1.6e-2 --qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2 \
    --name qk_zoom --out runs/tinystories --save runs/tinystories/qk_zoom.pt
bantam generate --model runs/tinystories/qk_zoom_pretrained --prompt "Once upon a time"
```

Any parquet-backed Hub dataset works (`--column` names the text field), or local
parquet files (`--train ... --val ...`), or an existing tokenizer (`--tokenizer gpt2`).

### Fine-tune a Hugging Face model

```bash
pip install 'bantamtrain[finetune]'

# LoRA r=16 on every linear layer (the default), Muon on the LoRA matrices
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --optim muon --out runs/dolly

# QLoRA (NF4 base) or DoRA: add --quant nf4 --low_ram 1 --grad_ckpt 1, or --dora 1
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly \
    --prompt "Give me three tips for staying healthy."
```

Defaults: 8-bit optimizer state, examples packed into 512-token windows with
block-diagonal masks, loss on response tokens only, chunked cross-entropy. Data can
be a Hub dataset or a JSONL file with `prompt`/`response`,
`instruction`/`context`/`response`, `messages` or `conversations` rows. The output
directory holds a standard PEFT adapter (`adapter_config.json` +
`adapter_model.safetensors`) and the tokenizer:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M"), "runs/dolly")
```

### Train a small weather model

```bash
pip install 'bantamtrain[weather]'
bantam prepare-era5 --out data/era5          # WeatherBench2 ERA5, 5.625 deg, 2010-2018
bantam pack-era5 --data data/era5            # optional error-bounded compressed copy
bantam weather --data data/era5 --val_year 2017 --budget 5.5e13 \
    --optim muon --lr 1.6e-2 --zoom 4:0.6,1:0.4 --coarse_mode lowres \
    --name lowres --out runs/era5 --save runs/era5/lowres.pt
bantam weather --forecast --load runs/era5/lowres.pt --init 2018-03-01T00 --steps 12
```

### Check your setup and plan memory

```bash
bantam doctor                                       # versions, CUDA, threads, RAM, extras
bantam memory --params 1e9 --gpu_gb 24              # memory / time estimate for one GPU
bantam memory --finetune --params 7e9 --gpu_gb 24   # full / LoRA / QLoRA fine-tuning
```

### Stop and resume

`bantam pretrain` and `bantam weather` take `--ckpt_every N` and `--resume PATH`;
`bantam finetune` takes `--save_every N` and `--resume OUT_DIR`. On CPU a resumed run
matches an uninterrupted one bit for bit (tested).

## Use it from Python

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

model.save_pretrained("my_model")               # zoom gates stripped: a plain k=1 model
tokens = ZoomGPT.from_pretrained("my_model").generate(torch.tensor([[0]]), 50)
```

Also importable: `bantamtrain.lora` (LoRA, DoRA, PEFT save/load), `bantamtrain.qweights`
(int8 / int4 / NF4 weight-only layers), `bantamtrain.weightpack` (compressed
checkpoints), `bantamtrain.losses` (chunked cross-entropy), `bantamtrain.data`
(TokPack, MinHash dedup, GridPack).

## On a GPU (should work, untested)

The CUDA paths exist but have only been tested with CPU stand-ins: bf16 autocast on
Ampere and newer (fp16 with a gradient scaler before), TF32, `--compile`,
`bantam pretrain --auto_batch`, and multi-GPU DDP through `torchrun`. There is no
FSDP, no FP8 and no fused kernel; packed fine-tuning uses a dense block-diagonal mask,
so expect it to be slower than padding-free packing with FlashAttention.

## Limitations

- No GPU measurements of speed or memory.
- One small model per task (5.8M pretraining, 135M fine-tuning, 1.26M weather), and
  2-3 seeds at most. Gains at larger scale are not measured.
- Comparisons are matched in model FLOPs, not wall-clock. Muon's own optimizer FLOPs
  (a few % at this size) are not charged to the budget.
- The weather runs had no separate validation year, and the scores are not
  WeatherBench scorecards.
- In fine-tuning, "1 epoch" leaves out the last packed windows that do not fill a
  whole optimizer step, and importing `bantamtrain.finetune` turns on flush-to-zero
  for the whole process.

## Credit

Most of Bantam is other people's ideas, re-implemented in plain PyTorch: Muon (Keller
Jordan et al.), Polar Express (Amsel et al.), QK-norm (Henry et al.; Dehghani et
al.), 8-bit optimizer state (Dettmers et al.), bf16 stochastic rounding (Zamirai et
al.), LoRA, QLoRA / NF4, DoRA, LoRA+ and rsLoRA, the PEFT adapter format, and
progressive resizing. Zoom is a learned-gate variant of patch-level training
([Shao et al., ICLR 2025](https://arxiv.org/abs/2407.12665)), related to Hourglass,
Funnel-Transformer and sequence-length warm-up. Data: TinyStories (Eldan & Li),
llama2.c models (Karpathy), WeatherBench 2 (Rasp et al.).

Made by Nathan from Jamaica. MIT license.
