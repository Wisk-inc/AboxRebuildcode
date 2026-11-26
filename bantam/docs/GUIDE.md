# Bantam guide

Task-oriented recipes for Bantam 0.3.0 (`pip install bantamtrain`, import `bantamtrain`,
command `bantam`). Every flag is described in [CLI.md](CLI.md). The full results, with
every table, are in [../RESULTS.md](../RESULTS.md); how Zoom Training and the FLOP
accounting work is in [METHOD.md](METHOD.md).

**Where the numbers come from.** Every Bantam measurement on this page comes from one
sandbox: a 4-core CPU, 16 GB of RAM, no GPU, PyTorch 2.15 for CPU. The models are small (5.8M
parameters for language, 1.26M for weather). "n seeds" says how many independent runs
stand behind a number, and "1 seed" means one run, so read those as indicative. Nothing
has been run on a GPU, and the 5B-on-one-GPU numbers are arithmetic, not measurements.

Contents:

1. [Install and check](#1-install-and-check)
2. [Pretrain a small language model on any Hugging Face dataset](#2-pretrain-a-small-language-model-on-any-hugging-face-dataset)
3. [Fine-tune any Hugging Face causal LM](#3-fine-tune-any-hugging-face-causal-lm)
4. [Train a weather model on ERA5](#4-train-a-weather-model-on-era5)
5. [Plan for low RAM or a big model with `bantam memory`](#5-plan-for-low-ram-or-a-big-model-with-bantam-memory)
6. [Running on a GPU](#6-running-on-a-gpu)
7. [Reproducing the results](#7-reproducing-the-results)

---

## 1. Install and check

```bash
pip install 'bantamtrain[data,finetune,weather]'   # or only the extras you need
bantam doctor                                      # versions, threads, RAM, GPU, extras
```

The extras are `data` (text preparation, `generate`), `finetune` (`finetune`, `chat`),
`weather` (ERA5 download) and `test`. Quote the brackets in zsh.

From a clone of the source repository (the project lives in `bantam/`):

```bash
pip install -e 'bantam[test,finetune,data,weather]'
cd bantam
python -m pytest tests -q -m "not slow and not network and not gpu"   # what CI runs
```

The full suite on the sandbox: 594 passed, 7 skipped (CUDA-only or slow tests).

---

## 2. Pretrain a small language model on any Hugging Face dataset

### Step 1: tokenize the data

```bash
bantam prepare-text --hf roneneldan/TinyStories --out data/tinystories --vocab 4096
```

This downloads one training shard (`--max_shards 1`, the default) and the validation
split, trains a 4,096-token BPE tokenizer and writes `train.bin`, `val.bin`,
`tokenizer.json` and `meta.json` (which records the document and token counts).
Training documents that are identical to a validation document are dropped, so the
validation loss is honest.

For another dataset, name its text column and, if needed, its config:

```bash
bantam prepare-text --hf <repo_id> --config <config> --column text --max_shards 4 --out data/mycorpus
```

To reuse an existing tokenizer instead of training one, pass `--tokenizer gpt2` (or a
`tokenizer.json` or model directory). Vocabularies larger than 65,536 are stored as
uint32.

The committed TinyStories runs used data prepared with the 0.2 code. Re-preparing with
0.3 gives a slightly different `train.bin` and `tokenizer.json`, because the BPE is now
trained after the validation copies are removed. `--drop_val_overlap 0` reproduces the
0.2 output byte for byte.

### Step 2 (optional): deduplicate and compress

```bash
bantam pack-text --data data/tinystories
```

This removes exact and near-duplicate training documents (MinHash) and writes compressed
`train.tokpack` / `val.tokpack` files. `bantam pretrain` uses them automatically, and the
run log records which source it read (`data_source`). The packed TinyStories training
stream behind the committed runs holds 116.8M tokens (the `data_source` of
`results/published/save_muon_base_repro.json`).

### Step 3: train a baseline

```bash
bantam pretrain --data data/tinystories --budget 6.8e13 --zoom 1 \
    --optim muon --lr 1.6e-2 --name base
```

`--budget` is the training compute in FLOPs. Every run with the same budget gets the
same compute, so you can compare options fairly. 6.8e13 FLOPs is 450 steps of 16
sequences of 256 tokens (1.84M tokens) for the default 5.8M-parameter model. It took
about 13 to 18 minutes per run on the sandbox, with 1 or 2 threads per run and other runs
sharing the machine.

The result goes to `results/llm/base.json`. The number to compare is `final_val_loss`:
cross-entropy in nats per token on the first 2,048 validation windows (lower is better).
The `evals` list is the curve on the first 256 windows; those early windows are easier,
so the curve reads about 0.27 to 0.30 nats lower than the final eval. Compare finals with
finals.

**Choosing the learning rate.** The committed LRs came from a sweep at a third of the
budget (2.3e13 FLOPs, final loss on the first 512 windows, 1 seed per LR). The best were
Muon 1.6e-2 (3.176) and AdamW 1e-3 (3.961). With `--optim muon`, the AdamW part of the
model (embeddings, tied head, norms) runs at the same LR, so the Muon recipe gives those parameters 16x the best
AdamW-only LR. AdamW settings do not transfer to Muon runs, so sweep the LR again when you
change the model, data or batch size:

```bash
for lr in 4e-3 8e-3 1.6e-2 3.2e-2; do
  bantam pretrain --data data/tinystories --budget 2.3e13 --optim muon --lr $lr \
      --evals 4 --final_eval_seqs 512 --name sweep_muon_lr$lr --out results/llm_sweep
done
```

### Step 4: add what helped

Measured on TinyStories with the default model (d=256, 6 layers, 4 heads, 5.8M
parameters, sequence 256, 4k vocabulary), Muon at LR 1.6e-2, FLOP-matched at 6.8e13
model FLOPs. Loss is `final_val_loss` (nats per token, first 2,048 validation windows;
+- is the sample standard deviation over seeds).

| recipe | flags | final val loss | seeds | baseline compute needed to match |
|---|---|---|---|---|
| Muon baseline | `--zoom 1` | 2.737 +- 0.012 | 3 | 1.00x |
| AdamW baseline (LR 1e-3) | `--optim adamw --lr 1e-3` | 3.332 | 1 | below 1x (worse than the Muon baseline) |
| Zoom, k=2 for half the budget, batch x2 | `--zoom 2:0.5,1:0.5 --batch_scale_k 1` | 2.685 +- 0.004 | 3 | 1.12x |
| Zoom, k=2 for a third, batch x2 | `--zoom 2:0.33,1:0.67 --batch_scale 2` | 2.680 +- 0.005 | 2 | 1.14x |
| QK-norm | `--qk_norm 1` | 2.639 +- 0.003 | 2 | 1.24x |
| QK-norm + Zoom (k=2 for a third, batch x2) | `--qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2` | 2.589 +- 0.006 | 2 | 1.41x |

"Baseline compute needed to match" reads each loss off the curve of plain Muon runs
trained at 1x, 1.25x, 1.5x and 2x the budget (interpolated in log-FLOPs;
`experiments/summarize.py` computes it). The runs above 1x are single seeds, and the
three Zoom seeds alone give 1.11x to 1.13x.

How to read this:

- Muon is the biggest single gain at this size. The AdamW baseline is one seed, and the
  two runs differ in more than the optimizer: the LR of the AdamW part (see the LR note
  above), and Muon's uncharged Newton-Schulz arithmetic (see the budget point below).
- QK-norm is a known technique (Henry et al. 2020; it is in the modded-nanogpt speedrun
  stack). It beat Zoom alone here.
- Zoom saved about 11 to 12% of compute (QK-norm plus Zoom: about 29%), and its gain
  (about -0.05 in loss) was the same with and without QK-norm, so the two stack.
- Zoom is a learned-gate variant of patch-level training (Shao et al., ICLR 2025), which
  reports about 0.5x training cost at 370M to 2.7B parameters. It is also related to
  Hourglass, Funnel-Transformer and sequence-length warm-up. Bantam's gain at 5.8M
  parameters is real but smaller than that prior art reports, and it is untested at
  larger scale.
- The budget counts model FLOPs only. Muon's optimizer FLOPs (Newton-Schulz) are logged
  as `opt_flops` but not charged. At this size they are not small: about 20% on top of
  the model FLOPs (1.37e13 against 6.81e13 for a baseline run,
  `results/published/save_muon_base_repro.json`), so the Muon runs did about 1.2x the
  arithmetic of the AdamW run. The Zoom recipes above take fewer optimizer steps than
  the baseline (422 or 432 against 450), so this does not favour them.
- Zoom does not save wall-clock time on this CPU in proportion to FLOPs. At a fixed
  batch, k=2 steps took about 1.2x and k=4 steps about 1.4x the time per FLOP of k=1
  steps (the optimizer step is a fixed cost per step; Muon runs, 2 threads, shared CPU);
  with the batch scaled by k the cost per FLOP is about the same as at k=1.

What did not help (one seed each unless noted, against the 2.737 baseline):

| change | flags | final val loss |
|---|---|---|
| original Zoom schedule, no batch scaling (3 seeds) | `--zoom 4:0.35,2:0.25,1:0.4` | 2.874 |
| mean merge instead of the learned gate | `--gated 0` (with the schedule above) | 2.943 |
| bag-of-tokens loss | `--out_mode bag` (same) | 2.860 |
| other k=4 schedules, no batch scaling | `--zoom 4:0.5,1:0.5` and two others | 2.80 to 3.02 |
| original schedule with batch x k | `--zoom 4:0.35,2:0.25,1:0.4 --batch_scale_k 1` | 2.730 (about equal) |
| LR restart at the stage switch | `--zoom 2:0.5,1:0.5 --lr_restart 1` | 2.760 |
| warmup-stable-decay schedule | `--sched wsd` | 2.773 |
| ReLU squared MLP | `--mlp relu2` | 2.829 |
| Polar Express orthogonalizer | `--ns polar` | 2.755 |
| logit soft cap | `--softcap 15` | 2.726 (about equal) |
| zero-initialized projections | `--zero_init_proj 1` | 2.732 (about equal) |

The WSD, ReLU squared, soft cap and zero-init runs read the raw `train.bin` from before
the data rebuild, which still held copies of validation stories, so treat them as
indicative.

### Step 5: save the model and sample from it

```bash
bantam pretrain --data data/tinystories --budget 6.8e13 --zoom 2:0.33,1:0.67 --batch_scale 2 \
    --qk_norm 1 --optim muon --lr 1.6e-2 --name qk_zoom --save runs/qk_zoom.pt
bantam generate --model runs/qk_zoom_pretrained --prompt "Once upon a time" --num_samples 3
```

`--save runs/qk_zoom.pt` writes the state_dict and a `runs/qk_zoom_pretrained/`
directory with `config.json`, the weights and the data's `tokenizer.json`. The Zoom
merge layers are dropped there, since a trained model only runs at k=1. The same from
Python:

```python
import json
from tokenizers import Tokenizer
from bantamtrain.models import ZoomGPT

d = "runs/qk_zoom_pretrained"
model = ZoomGPT.from_pretrained(d)                  # eval mode
tok = Tokenizer.from_file(f"{d}/tokenizer.json")
eos = json.load(open(f"{d}/config.json"))["eos_id"]
ids = [eos] + tok.encode("Once upon a time").ids    # documents start after an end-of-document token
out = model.generate([ids], 100, temperature=0.8, top_k=40, eos_id=eos)   # KV-cache decoding
print(tok.decode(out[0][len(ids):]))
```

### Step 6: long runs, interruptions and divergence

```bash
bantam pretrain ... --name long --ckpt_every 100                    # writes results/llm/long.ckpt
bantam pretrain ... --name long --resume results/llm/long.ckpt      # same flags as the first run
```

The checkpoint holds the model, optimizer, data position and RNG states. Resume is
bit-exact on CPU (tested). `--resume` refuses a run whose model, data-order or schedule
flags (or number of processes) differ, unless you pass `--allow_arg_change`. A run whose
loss becomes NaN or infinite stops, records `"status": "diverged"`, writes no `--save`
model and exits with status 3.

### Low-memory options

`--state_bits 8` keeps the optimizer state in 8 bits (blockwise, like Dettmers et al.'s
8-bit optimizers). `--param_dtype bf16` stores the weights in bf16 and writes updates
with stochastic rounding, so there is no fp32 master copy. Both are published techniques.

At the full budget, the two together scored 2.728 against the 2.737 baseline (1 run),
and weights plus optimizer state went from 50.4 MB to 18.5 MB (2.7x smaller). That run
was scored with 0.2's bf16 arithmetic, with a measured error of about +-0.001, and is
marked for a rerun. On this CPU the bf16 path ran 2.6x slower in wall-clock (718 s
against 271 s for the same 100 steps, 1 thread): bf16 matmuls are emulated and every
update is stochastically rounded. Use it when memory is the limit, not to save time.

---

## 3. Fine-tune any Hugging Face causal LM

### The data

`--jsonl` or `--dataset` rows can be any of:

```json
{"prompt": "What is the capital of France?", "response": "Paris."}
{"instruction": "Summarize the text.", "context": "The text ...", "response": "A summary."}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
```

`instruction`/`output` and `question`/`completion` also work, and `--prompt_col`,
`--response_col`, `--messages_col` name other columns. Rows that cannot be used are
counted by reason in `log.json`.

The loss covers the response tokens only (`--prompt_loss_weight 1` trains on everything).
The prompt is built with the model's chat template. A base model without one gets ChatML
when its vocabulary has `<|im_end|>` (SmolLM2, Qwen2.5 bases), else a plain
`### Instruction:` / `### Response:` format. The tokenizer, with the template used, is
saved next to the adapter, so `bantam chat` builds prompts exactly as in training.

### LoRA

```bash
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --lora 16 --epochs 1 --out runs/dolly
```

The defaults: LoRA rank 16, alpha 32, dropout 0.05, on every linear layer except the
output head; AdamW with 8-bit state (`--state_bits 8`); LR 2e-4 with 3% warmup and a
warmup-stable-decay schedule; windows of 512 tokens with best-fit-decreasing packing;
5% of the examples held out. `--optim muon` puts the LoRA matrices on Muon instead.
Whether Muon on LoRA beats AdamW has not been measured here yet (see the race in
section 7).

Packing puts several examples in one window. Each window gets a block-diagonal causal
mask with positions restarting per example, so packed examples never see each other.
In 0.2 they did; 0.3 fixed it, also for models that use eager attention, such as
Gemma-2.

### QLoRA: LoRA on a quantized base

```bash
bantam finetune --model HuggingFaceTB/SmolLM2-135M --dataset databricks/databricks-dolly-15k \
    --max_examples 2000 --lora 16 --quant nf4 --low_ram 1 --grad_ckpt 1 --out runs/qlora
```

- `--quant 8` (int8), `4` (linear int4) or `nf4` (the QLoRA 4-bit codebook, Dettmers et
  al. 2023) for the frozen linear layers. Embeddings and the output head are int8 by
  default (`--embed_quant 8`; `same` follows `--quant`, `none` keeps them in float).
- `--low_ram 1` builds the model with its weights quantized one tensor at a time, so the
  float model never sits in RAM. It needs `.safetensors` weights. In one end-to-end check
  during the 0.3 review (SmolLM2-135M, `--quant nf4`, `--seq 256 --batch 2`, 20 steps,
  one run each; not a committed result), peak RSS was 2,339 MB with `--low_ram 1`,
  2,733 MB with the default `--quant` path (it loads the float model first) and 2,668 MB
  for float LoRA.
- `--grad_ckpt 1` (activation checkpointing) is recommended with `--quant`.
- Everything is pure PyTorch. There is no bitsandbytes dependency and no fused GPU
  kernel, so expect it to be slower than GPU tools that have them.

What the frozen base's format costs before any training, on SmolLM2-135M (88 held-out
Dolly examples, response tokens, 256-token windows, one run each; sizes are the base
weights without adapters):

| base weights | flags | held-out loss | change | size |
|---|---|---|---|---|
| fp32 | (none) | 2.702 | | 538 MB |
| int8 linears + int8 embeddings | `--quant 8` | 2.704 | +0.003 | 181 MB |
| NF4 linears + int8 embeddings | `--quant nf4` | 2.878 | +0.18 | 120 MB |
| linear int4 + int8 embeddings | `--quant 4` | 3.035 | +0.33 | 120 MB |
| NF4 everything | `--quant nf4 --embed_quant same` | 3.042 | +0.34 | 87 MB |

This was measured with 0.2 code, whose packed eval windows let examples attend to each
other, so the differences are indicative. The 0.2 sizes also count SmolLM2's tied output
head twice and include int8 block padding; 0.3 stores the head once. In the 0.3 review,
the NF4 model with int8 embeddings took 105 MB and the int8 model 157 MB, adapters
included (0.2: 140 and 200 MB). On a small model, int8 is nearly free and 4-bit costs a
lot. The QLoRA paper reports NF4 close to 16-bit quality for large models (up to
65B); that cannot be checked on this machine.

The 0.2 fine-tunes in `results/finetune/` (40 steps, 8 eval windows of about 17
examples, 1 seed) went from 2.871 to 2.792 on the float base and from 3.389 to 3.178 on
an int4-everything base. The tuned int4 model stayed worse than the untouched float base.
Those runs used 8-bit AdamW: 0.2's `--optim muon` put no parameters on Muon when LoRA
was on. 0.3 fixes that (`--muon_lora` is automatic).

### DoRA, LoRA+, rsLoRA and targets

| flag | method | source |
|---|---|---|
| `--dora 1` | DoRA: magnitude and direction trained separately | Liu et al., ICML 2024 |
| `--lora_plus 16` | LoRA+: the B matrices get a higher LR (here 16x) | Hayou et al., ICML 2024 |
| `--rslora 1` | rank-stabilized scaling (on by default for r >= 64) | Kalajdzievski 2023 |
| `--lora_targets q_proj,v_proj` | adapt only these layers (default: all linear layers) | |

These are other people's methods. The tests check Bantam's LoRA, DoRA and rsLoRA
adapters against PEFT's outputs. Their benefit has not been measured on this machine.

### Evaluation

- `--eval_frac 0.05` holds out 5% of the examples; `--eval_jsonl FILE` uses a separate
  file instead.
- `--eval_every 50` evaluates every 50 steps; the final eval always runs.
- `--eval_only 1` prints the held-out loss and exits, which is a quick way to measure a
  base model (or a quantized base) before training.

The eval loss is per-token cross-entropy over response tokens, in `log.json` as
`evals` and `final_eval_loss`.

### Resume

```bash
bantam finetune ... --out runs/dolly --save_every 100            # writes runs/dolly/checkpoint
bantam finetune ... --out runs/dolly --resume runs/dolly/checkpoint
```

Resume continues the exact run (tested bit-identical, also with LoRA dropout). It
refuses a checkpoint if training options differ.

### Outputs, PEFT export and merging

A LoRA run writes to `--out`: `adapter.pt` and `adapter_meta.json` (Bantam format; about
20 MB for SmolLM2-135M at r=16, since the adapters are fp32), `adapter_config.json` and
`adapter_model.safetensors` (PEFT format, on by default), the tokenizer, and `log.json`.
To use the adapter in the Hugging Face stack:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = PeftModel.from_pretrained(base, "runs/dolly")     # logits match Bantam's to ~1e-5 (tested)
tok = AutoTokenizer.from_pretrained("runs/dolly")          # with the chat template used in training
```

The other way, `bantamtrain.lora.load_peft_adapter(model, adapter_dir)` loads a PEFT
LoRA or DoRA adapter into a float or quantized model, and `bantam chat --adapter` accepts
a PEFT directory.

- `--save_merged 1` also writes `<out>/merged`, a plain `save_pretrained()` model with
  the adapters merged in. With `--quant` the adapters are merged into a freshly loaded
  float base.
- An adapter trained on a quantized base learned against the quantized weights. On the
  float base its outputs differ somewhat.
- Full fine-tuning: `--lora 0 --param_dtype bf16 --state_bits 8 --grad_ckpt 1` (bf16
  weights with stochastic rounding, 8-bit state). The model goes to `<out>/model`.
  `--quant` needs `--lora` > 0.

### Chat with the result

```bash
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly --prompt "Name three colours"
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/qlora --quant nf4    # interactive
```

`/reset` clears the conversation and `/exit` quits. `--quant` always uses int8
embeddings, whatever `--embed_quant` the run used.

### Several processes or GPUs

```bash
torchrun --standalone --nproc_per_node 4 -m bantamtrain.finetune \
    --model HuggingFaceTB/SmolLM2-135M --jsonl my_data.jsonl --out runs/ddp
```

This is DistributedDataParallel. It was tested with 2 CPU processes (gloo), not on GPUs.
The loss is averaged over every supervised token of the step across processes. There is
no FSDP or ZeRO sharding, so the whole model must fit on each device.

### Known limits

- With packing, sliding-window attention layers lose their window. This affects
  Gemma-3 at `--seq` above 512 or 1024, Phi-3, Mistral and Starcoder2 at `--seq` above
  their window, and Gemma-2 above 4096. The default `--seq 512` is unaffected.
- "1 epoch" skips the last packed windows that do not fill a whole optimizer step (up to
  `batch x accum - 1` windows).
- Importing `bantamtrain.finetune` turns on flush-to-zero for the whole process.

---

## 4. Train a weather model on ERA5

```bash
bantam prepare-era5 --out data/era5_64x32 --val_year 2017   # 2010-2018, 6 channels, about 650 MB
bantam pack-era5 --data data/era5_64x32                     # optional: error-bounded GridPack
bantam weather --data data/era5_64x32 --budget 5.5e13 --zoom 1 \
    --optim muon --lr 1.6e-2 --name base --save runs/wx_base.pt
```

The data is WeatherBench2's ERA5 at 5.625 degrees (a 64x32 grid), every 6 hours, with
six channels: z500, t850, t2m, u10, v10 and msl. The model is a small ViT (ZoomViT,
d=128, 6 layers, 1.26M parameters) that predicts the state 6 hours ahead. 5.5e13 FLOPs
is 567 steps of batch 16 for the baseline.

`--val_year 2017` holds 2017 out for model selection and keeps 2018 as the test year,
touched only by the final eval. The committed runs had no validation year: they
evaluated on the 2018 test year, and their LRs were chosen on it too. Use a validation
year for new work.

Metrics in `results/weather/<name>.json`: `final.val_loss` is the one-step MSE in
normalized units; `rmse_<channel>_<h>h` is the latitude-weighted RMSE of a rollout at 6,
24 and 72 h (z500 in m²/s²), next to `persist_*` (repeat the initial state) and, in 0.3,
a train-years climatology and the anomaly correlation (ACC). With the defaults, the
committed runs started their rollouts from 96 initial conditions at 00 UTC between
1 January and 13 October 2018. `--ic_stride 14 --max_ics 0` covers the whole year at
alternating 00 and 12 UTC.

Measured results (ERA5 64x32, ZoomViT 1.26M parameters, 5.5e13 model FLOPs, evaluated on
the 2018 test year; RMSE values are from seed 0):

| recipe | flags | one-step MSE | seeds | z500 RMSE 24 h | z500 RMSE 72 h |
|---|---|---|---|---|---|
| Muon baseline | `--optim muon --lr 1.6e-2 --zoom 1` | 0.124 +- 0.001 | 2 | 217 | 539 |
| Muon + low-resolution warm-up | `--zoom 4:0.6,1:0.4 --coarse_mode lowres` | 0.112 | 1 | 194 | 511 |
| Muon + Zoom merge | `--zoom 4:0.6,1:0.4` | 0.159 | 2 | 253 | 600 |
| Muon + Zoom, k=4 for 40% | `--zoom 4:0.4,1:0.6` | 0.129 | 1 | 219 | 548 |
| Muon + Zoom, mean merge | `--zoom 4:0.6,1:0.4 --gated 0` | 0.128 | 1 | 206 | 527 |
| Muon + Zoom 16-4-1 | `--zoom 16:0.3,4:0.3,1:0.4` | 0.180 | 1 | 275 | 634 |
| AdamW baseline | `--optim adamw --lr 2e-3 --zoom 1` | 0.229 | 1 | 346 | 737 |
| AdamW + Zoom merge | `--optim adamw --lr 2e-3 --zoom 4:0.6,1:0.4` | 0.180 | 1 | 282 | 647 |
| AdamW + low-resolution warm-up | `... --coarse_mode lowres` | 0.158 | 1 | 257 | 601 |
| Muon baseline at 2x compute | `--budget 1.1e14` | 0.094 | 1 | 172 | 461 |
| persistence (no model) | | | | 569 | 907 |

How to read this:

- Muon is again the biggest gain: every AdamW run loses to the Muon baseline.
- Zoom merge made weather worse under Muon. It only helped the weaker AdamW runs.
- Low-resolution warm-up (`--coarse_mode lowres`: train on 2x2-coarsened fields for the
  first 60% of the budget) gave about 10% lower MSE than the Muon baseline, in one seed.
  This is progressive resizing, established practice, not something new.
- Every model beats persistence clearly (z500 at 24 h: 217 against 569).

### Forecasts

```bash
bantam weather --forecast --load runs/wx_base.pt --data data/era5_64x32 \
    --init 2018-03-01T00 --steps 12
```

This writes a `(steps + 1, 6, 32, 64)` float32 array in physical units to
`results/weather/forecasts/`, with a JSON sidecar of valid times and RMSE where the data
covers them.

`bantam weather` runs in a single process; it has no DDP. Resume works as in pretrain
(`--ckpt_every`, `--resume`), is bit-exact on CPU, and refuses any change to the model,
budget, batch, LR, schedule or seed.

---

## 5. Plan for low RAM or a big model with `bantam memory`

`bantam memory` estimates memory and time for one GPU. It is arithmetic from stated
assumptions (every one a flag); nothing in it has been measured on a GPU.

```bash
bantam memory                     # 5B parameters on one 96 GiB card (RTX PRO 6000 Blackwell)
```

At the defaults it prints (memory in GiB, 85% of the card usable):

```
configuration                                                              B/p  static  total fit  maxB   days speedup
1. Standard: AMP bf16, fp32 master + fp32 AdamW                           18.0     85G    90G NO      0    n/a     n/a
2. + bf16 weights (stochastic rounding) + 8-bit AdamW + chunked CE         6.0     28G    78G yes     6    176   1.00x
3. + Muon (8-bit momentum)                                                 5.1     24G    81G yes     7     94   1.87x
4. + FP8 linears                                                           5.1     24G    81G yes     7     78   2.24x
5. + 450 W power cap (75 % power)                                          5.1     24G    81G yes     7     83   2.13x
```

How to read it:

- Row 1, standard mixed precision with an fp32 master copy and fp32 AdamW, does not fit
  with activations. The textbook count for its model state is 16 bytes per parameter
  (about 80 GB at 5B; the tool charges 18). bf16 weights with 8-bit state need about 6
  bytes per parameter (about 30 GB) and fit.
- The 1.87x for Muon comes from the literature, not from a Bantam measurement: the tool
  assumes Muon reaches AdamW's loss with 0.52x the FLOPs (`--muon_flops`), from Liu et
  al. 2025, "Muon is Scalable for LLM Training", which reports about 2x compute
  efficiency in compute-optimal scaling-law runs. The Newton-Schulz cost is then added
  back.
- Row 4 is a literature estimate: Bantam has no FP8 code. Rows 2 to 5 assume chunked
  cross-entropy, which `bantam pretrain` does not expose.
- None of this is new. ZeRO-Offload trained models over 13B on one 32 GB V100 in 2021,
  and torchao ships 8-bit and 4-bit AdamW with bf16 stochastic rounding.
- `--gpu_gb` is read as GiB (a "24 GB" card is 24), and `--headroom 1` drops the 15%
  margin. 0.2 mixed GB and GiB and reported 2.56x; the corrected figure is about 1.9x.

For fine-tuning, ask whether a given model fits a given card:

```bash
bantam memory --finetune --params 7e9 --gpu_gb 24
```

```
fine-tuning a 6.9B base model on one 24 GiB GPU (20.4 GiB usable)
configuration                                                act.ckpt  trainable  static    act  total fit
full fine-tune: fp32 master + AdamW (standard)                    yes    6905.9M  115.8G   6.7G 124.5G NO
full fine-tune: bf16 weights(SR) + 8-bit Muon                     yes    6905.9M   32.8G   4.4G  39.2G NO
LoRA r=16: bf16 base frozen + fp32 adapters/AdamW                 yes      41.3M   13.5G   5.0G  20.5G NO
QLoRA r=16: int4 base (bantamtrain.qweights) + adapters           yes      41.3M    4.1G   5.0G  11.2G yes
```

(The rows without activation checkpointing, which all say NO, are left out here.) Only
QLoRA with activation checkpointing fits a 7B model in 24 GB by this estimate, at the
default micro-batch of 4 and sequence of 2,048. The
matching run is `bantam finetune --quant nf4 --low_ram 1 --grad_ckpt 1`.

### Ways to save memory, by command

| goal | flags |
|---|---|
| quantized frozen base, never holding the float model | `finetune --quant nf4 --low_ram 1` |
| smaller activations | `--grad_ckpt 1` (pretrain, finetune), `--accum N` with a smaller `--batch` |
| never build the full logits | `finetune --ce_chunk 2048` (the default) |
| smaller optimizer state | `--state_bits 8` (pretrain, finetune, weather) |
| no fp32 master weights | `pretrain --param_dtype bf16`; `finetune --param_dtype bf16` |
| GPU micro-batch that fits | `pretrain --auto_batch` |

### Storing weights small

```python
from bantamtrain import qweights, weightpack

weightpack.save("model.zwpk", model.state_dict(), bits=8)   # int8 checkpoint file (also 32, 16, 4)
state = weightpack.load("model.zwpk")                         # tensors in their saved dtype
qweights.quantize_model(model, bits=8)                         # int8 weights in RAM ("nf4", 4 also work)
```

Measured on the 5.8M-parameter TinyStories model with the 0.2 code (final loss on 2,048
windows; one model): int8 weights in RAM are 3.9x smaller than fp32 for +0.0002 loss,
and int4 weights are 7.5x smaller for +0.026. A WeightPack file at `bits=16` is 2.9x
smaller than an fp32 `torch.save` file and costs +0.00002 loss: it rounds fp32 weights
to bf16, so it is lossless only for weights that are already bf16. Use `bits=32` for a
state_dict that is already quantized. `experiments/weights_eval.py` measures all formats
for a saved run.

---

## 6. Running on a GPU

Nothing in Bantam has run on a GPU yet. The CUDA code paths exist and were tested with
CPU stand-ins only. Treat GPU use as a first trial, and please report what happens
(`bantam doctor` output helps).

What is in place:

- `--device cuda` (or `auto`), TF32 matmuls, fused AdamW, and attention through
  `scaled_dot_product_attention`.
- `--precision auto`: bf16 autocast on compute capability 8.0 and newer, fp16 with a
  GradScaler on older cards. The GradScaler state is saved in checkpoints.
- `--compile`: `torch.compile` (`bantam weather` logs compile time apart from training
  time).
- `bantam pretrain --auto_batch`: picks the largest micro-batch that fits and keeps the
  effective batch (`batch x accum`), schedule and data order.
- `--duty 0.75` caps the average load in software. A hardware power cap
  (`sudo nvidia-smi -pl <watts>`) is better when you can set one.
- DDP with torchrun for `pretrain` and `finetune`:

```bash
torchrun --standalone --nproc_per_node 4 -m bantamtrain pretrain \
    --data data/tinystories --budget 2e14 --optim muon --lr 1.6e-2 \
    --device cuda --precision auto --compile --name gpu_base
```

`experiments/gpu_smoke.sh` is the first thing to run on a new card (set `DATA=` to your
prepared data directory). It checks that the bf16, compile and duty-cycle paths run; it
is not a benchmark. Its LR (2e-3) and batch (64) are not the CPU-tuned recipe, and its
Zoom runs use the original 4->2->1 schedule, which lost to the baseline on CPU. The
script's header expects the Zoom run to complete more steps per second; nothing measured
so far supports that. Re-sweep the LR when you change the batch size.

What is untested or missing:

- Everything above on real hardware, including NCCL, compile and fp16.
- Bit-exact resume on GPU: CUDA RNG states are saved, but GPU kernels are not
  deterministic.
- `bantam weather` has no DDP.
- No FSDP or ZeRO sharding, no FP8 path, no fused kernels (FlashAttention varlen for
  packing, Liger, Unsloth-style kernels). Packed fine-tuning uses a dense 4-D mask, which
  loses FlashAttention's speed.
- Speed and memory have not been compared with GPU tools. Unsloth, Liger Kernel and
  torchtune publish GPU numbers; Bantam has none.

---

## 7. Reproducing the results

The committed results are JSON logs under `results/`: one file per run, with its
arguments, the full validation curve and the final metrics. `results/SUMMARY.md` and
`results/summary.json` are generated from them.

### Data

```bash
bantam prepare-text --hf roneneldan/TinyStories --out data/tinystories --vocab 4096
bantam pack-text --data data/tinystories
bantam prepare-era5 --out data/era5_64x32
bantam pack-era5 --data data/era5_64x32
```

The committed runs used data prepared by 0.2 (see section 2 for the difference). The
queue files point at the original machine's paths, so first change them to your data
directories: the `D=` (and, in `followup.txt`, `W=`) line at the top of most files, and
the `--data` path on every line of the sweep files (`llm_sweep.txt`, `llm_sweep2.txt`,
`sweep3.txt`, `weather_sweep.txt`, `weather_sweep4.txt`), which have no `D=` line.

### Queues

`experiments/run_queue.py` runs a queue file, a few runs at a time:

```bash
python experiments/run_queue.py experiments/queues/llm_main.txt --workers 2 --threads 2 --dry_run
python experiments/run_queue.py experiments/queues/llm_main.txt --workers 2 --threads 2
```

It skips runs whose JSON is finished, re-runs unfinished ones (resuming from
`<out>/<name>.ckpt` when there is one), skips diverged runs, and writes logs to
`results/logs/<llm|weather>/<name>.log`. `--dry_run` prints the plan only.

The queues, roughly in the order they were run:

| queue | what it produces |
|---|---|
| `llm_sweep.txt`, `llm_sweep2.txt`, `sweep3.txt` | LLM LR sweeps at 2.3e13 FLOPs (`sweep3.txt` also extends the weather sweep) |
| `weather_sweep.txt`, `weather_sweep4.txt` | weather LR sweeps at 1.8e13 FLOPs |
| `llm_main.txt` | Muon and AdamW baselines, the original Zoom schedule, Zoom ablations, seeds |
| `followup.txt` | Zoom follow-ups, including `fu_2-50_bk_s0` (k=2, batch x k). Only 3 of its 13 runs were run; the other 10 (including both linear-merge runs) have no results, and `run_queue.py` will start them |
| `zoom_confirm.txt` | seeds 1 and 2 of that recipe |
| `compute_to_target.txt` | the Muon baseline at 1.25x, 1.5x and 2x the budget (the compute-to-target curve) |
| `upgrades.txt` | QK-norm, WSD, ReLU squared, soft cap, zero-init, Polar Express, Zoom batch variants, and all five upgrades together (with and without Zoom) |
| `lowmem.txt`, `lowmem2.txt` | 8-bit state and bf16 weights, short and full budget; `save_muon_base` |
| `published_cmp.txt` | a long run (killed at 30%; not used) |
| `weather_main.txt` | the weather matrix: Muon and AdamW, base, Zoom and low-res, ablations, seeds, 2x baseline |

`all_remaining.sh` is an old helper that runs three of them in sequence.

The last six language-model runs are not in a queue file. Their commands, rebuilt from
the arguments in their run logs (each ran with `--threads 1`):

```bash
D=data/tinystories
C="--data $D --budget 6.8e13 --optim muon --lr 1.6e-2 --threads 1"
bantam pretrain $C --zoom 1 --seed 0 --name up_base_s0
bantam pretrain $C --zoom 1 --seed 1 --name up_base_s1
bantam pretrain $C --zoom 1 --qk_norm 1 --seed 1 --name up_qknorm_s1
bantam pretrain $C --zoom 2:0.33,1:0.67 --batch_scale 2 --seed 1 --name up_zoom_2-33_bs2_s1
bantam pretrain $C --zoom 2:0.33,1:0.67 --batch_scale 2 --qk_norm 1 --seed 0 --name up_qknorm_zoom_s0
bantam pretrain $C --zoom 2:0.33,1:0.67 --batch_scale 2 --qk_norm 1 --seed 1 --name up_qknorm_zoom_s1
```

### Tables and plots

```bash
python experiments/summarize.py --plots     # results/SUMMARY.md, results/summary.json, results/plots/
```

`summarize.py` tabulates finished runs only. It lists running, diverged and
needs-a-rerun runs apart, averages seeds (n counts distinct seeds), and computes
compute-to-target from final losses of the matching baseline recipe. Write to another
place with `--results DIR --out FILE`.

### Comparison with published TinyStories models

`experiments/compare_published.py` scores checkpoints in bits per byte on 1,000 of the
4,194 TinyStories validation stories that appear in neither the TinyStories training
split nor llama2.c's training data. Each model family is scored in the boundary format it
was trained with. The validation file is downloaded at a pinned revision and
hash-checked.

```bash
# retrain the Bantam row (the 0.2 weights are gone) and keep the checkpoint
bantam pretrain --data data/tinystories --budget 6.8e13 --zoom 1 --optim muon --lr 1.6e-2 \
    --threads 1 --name save_muon_base_repro --out results/published --save runs/save_muon_base_repro.pt
python experiments/compare_published.py --stories 1000 --ctx_cap 512 \
    --ours results/published/save_muon_base_repro.json:runs/save_muon_base_repro.pt \
    --hf roneneldan/TinyStories-1M roneneldan/TinyStories-8M \
    --l2c <dir>/stories15M.pt:<dir>/tokenizer.model \
    --out results/published/compare_clean1000.json
python experiments/plot_published.py results/published/compare_clean1000.json
```

The llama2.c checkpoints and tokenizers come from Karpathy's llama2.c releases (the
`karpathy/tinyllamas` Hub repository). On that set, Bantam's 5.8M model (one run, 6.4e13
training FLOPs) scored 1.029 bits per byte. It beats llama2.c's stories260K (1.097,
1.0e16 FLOPs) with at least 157x less training compute. That is an upper bound read off
a single point, and stories260K has 0.26M parameters, so it is partly a model-size
effect. Bantam is behind every TinyStories model: TinyStories-1M scores 0.822 with about
3,350x more compute.

### The fine-tuning race

`experiments/race_finetune.py` races Hugging Face Trainer + PEFT against
`bantam finetune` on the same model, data and budget, and measures speed, memory and
quality.

```bash
python experiments/race_finetune.py --threads 4                                     # full race
python experiments/race_finetune.py --max_examples 32 --judge_n 16 --out results/race/smoke   # smoke test
```

The protocol, shared by every arm:

- HuggingFaceTB/SmolLM2-135M in fp32 on CPU. Dolly-15k from the local Hugging Face cache
  (offline by default; `--online` allows downloads), split with Random(1234): 800
  training examples and 200 held-out judge examples (rows with no response token within
  512 tokens are skipped).
- ChatML, loss on response tokens only. LoRA r=16, alpha 32, dropout 0.05 on q, k, v, o,
  gate, up and down projections (210 modules, 4,884,480 trainable parameters).
- LR 2e-4, 3% warmup then cosine to 10%, AdamW betas (0.9, 0.95), no weight decay,
  gradient clip 1.0. Sequence 512, micro-batch 2, gradient accumulation 2, one epoch,
  seed 1234, the same `--threads` everywhere. Each arm trains in its own process.

The arms:

| arm | what it runs |
|---|---|
| `base` | no training; the judge's reference |
| `hf_peft` | transformers Trainer + peft LoraConfig, one example per row, dynamic padding, torch AdamW (fp32 state) |
| `bantam_adamw` | `bantam finetune --optim adamw --state_bits 32 --pack bfd`: the same optimizer as `hf_peft`, so the difference is the training loop and packing |
| `bantam` | `bantam finetune --optim muon --pack bfd`: Muon on the LoRA matrices, 8-bit state, packing with block-diagonal masks, chunked cross-entropy |
| `bantam_nf4` | the `bantam` arm on an NF4 base built with `--quant nf4 --low_ram 1` (int8 embeddings); judged on its NF4 base and on the float base |

What is measured: training-only tokens per second (real, non-padding tokens), peak RSS of
the process tree (sampled every 0.2 s), and the judge. The judge scores every final model
the same way, in its own process: per-token cross-entropy over the response tokens of the
200 held-out examples, one example at a time, fp32, pooled over tokens, with bootstrap
95% confidence intervals and paired differences against `hf_peft`. Bantam arms are judged
through their PEFT export (the NF4 arm through Bantam's quantized loader). It also
records greedy answers to three fixed prompts. Every arm uses the protocol's cosine
schedule, although `bantam finetune` defaults to WSD.

Two differences are part of the design. Packing makes the Bantam arms take about 3.4x
fewer, larger optimizer steps at the same LR (58 against 200 in a full-protocol dry run).
And Bantam's "1 epoch" leaves out the last packed windows that do not fill a step (3 of
235 windows, about 1.2% of the supervised tokens); tokens per second counts only the
tokens trained. A 32-example smoke run checked the harness (token parity, identical
LoRA modules, judge agreement with Bantam's own eval to about 2e-7). Its numbers are not
a comparison.

<!-- RACE_RESULTS -->

### Tests

```bash
cd bantam && python -m pytest tests -q                 # full suite, several minutes on 4 cores
python -m pytest tests -q -m "not slow and not network and not gpu"   # what CI runs
```
