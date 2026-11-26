# Bantam command-line reference

This page lists every `bantam` command in version 0.3.0: what it does, the flags that
matter (grouped, with defaults), and one or two examples you can copy. The full list for
any command is `bantam <command> --help`. Task-oriented recipes are in
[GUIDE.md](GUIDE.md); what changed since 0.2 is in [../CHANGELOG.md](../CHANGELOG.md).

## Running a command

```bash
pip install bantamtrain                  # core: torch, numpy, zstandard
pip install 'bantamtrain[data]'          # prepare-text, generate (tokenizers, pyarrow, huggingface_hub)
pip install 'bantamtrain[finetune]'      # finetune, chat (transformers>=4.56, safetensors, datasets, sentencepiece)
pip install 'bantamtrain[weather]'       # prepare-era5 (xarray, zarr, fsspec, aiohttp)
```

Quote the brackets: zsh (the macOS default shell) treats unquoted `[...]` as a file
pattern and stops.

These three forms are the same command:

```bash
bantam pretrain --help
bantamtrain pretrain --help
python -m bantamtrain pretrain --help
```

`bantam --version` prints the version. `bantam <command> --help` works even when the
command's optional extra is missing. Running such a command without its extra prints
which `pip install 'bantamtrain[...]'` to run and exits with status 1.

Older entry points still work and run the same code:

| old form | same as |
|---|---|
| `python llm/train.py ...` | `bantam pretrain ...` |
| `python llm/prepare_data.py ...` | `bantam prepare-text ...` |
| `python llm/pack_data.py ...` | `bantam pack-text ...` |
| `python weather/train.py ...` | `bantam weather ...` |
| `python weather/prepare_era5.py ...` | `bantam prepare-era5 ...` |
| `python weather/pack_data.py ...` | `bantam pack-era5 ...` |
| `python -m bantamtrain.finetune ...` | `bantam finetune ...` |
| `python -m bantamtrain.memory ...` | `bantam memory ...` |

The `llm/` and `weather/` scripts exist only in the source repository, not in the pip
package. The repository's old package name is gone; see [../CHANGELOG.md](../CHANGELOG.md).

Exit status, for every command: 0 means success, 1 means an error (a message or a
traceback), and 2 means bad arguments (argparse usage error). `bantam pretrain` also uses 3 for a run
that diverged (non-finite loss).

**Hardware note.** Every command was developed and measured on a 4-core CPU with no GPU.
The CUDA code paths (`--device cuda`, `--precision`, `--compile` on GPU, NCCL DDP) exist
but have only been tested with CPU stand-ins. See "Running on a GPU" in
[GUIDE.md](GUIDE.md).

Commands: [pretrain](#bantam-pretrain) ·
[finetune](#bantam-finetune) ·
[weather](#bantam-weather) ·
[prepare-text](#bantam-prepare-text) ·
[pack-text](#bantam-pack-text) ·
[prepare-era5](#bantam-prepare-era5) ·
[pack-era5](#bantam-pack-era5) ·
[memory](#bantam-memory) ·
[generate](#bantam-generate) ·
[chat](#bantam-chat) ·
[doctor](#bantam-doctor)

---

## bantam pretrain

Trains a ZoomGPT language model from scratch at a fixed compute budget (`--budget`, in
training FLOPs). Every run with the same budget gets the same model FLOPs, so runs with
different options can be compared fairly. Muon's own optimizer FLOPs are logged but not
charged unless `--flops_mode total`; at the default model size they add about 20%
(`opt_flops` in `results/published/save_muon_base_repro.json`). It reads token files
written by `bantam prepare-text` (and `bantam pack-text`, if you ran it).

Required: `--data`, `--budget`, `--name`.

**Recipe**

| flag | default | meaning |
|---|---|---|
| `--recipe` | `none` | `best`: set `--optim muon --lr 1.6e-2 --qk_norm 1 --zoom 2:0.33,1:0.67 --batch_scale 2` for every one of these options the command line does not give. `none`: no change |

`--recipe best` is the best combination measured so far, at small scale on CPU only:
TinyStories, the 5.8M-parameter default model, 6.8e13 FLOPs. Its final validation loss
was 2.589 (2 seeds) against 2.737 for the Muon baseline (3 seeds), so the baseline needs
about 1.41x the compute to match. It has not been tested on larger models or on GPUs.
Options you pass explicitly always win, also at their default value (`--zoom 1` keeps
the baseline schedule). The LR is tied to the optimizer it was measured with: with an
explicit `--optim adamw`, `--lr` keeps its default and the run prints a note. Likewise
`--batch_scale`, which overrides `--batch_scale_k`, is not set when you pass
`--batch_scale_k 1`. The run
log records `args.recipe`, the resolved values of every option and
`args.recipe_applied` (the options the recipe set). To resume such a run, repeat the
same command line with `--resume`.

**Data and budget**

| flag | default | meaning |
|---|---|---|
| `--data DIR` | required | directory with `train`/`val` `.bin` or `.tokpack` files and `meta.json` |
| `--data_format` | `auto` | `auto`: use `<split>.tokpack` when present and current, else `<split>.bin`. `bin`, `tokpack`: force one. A pack left over from an earlier prepare run is skipped with a warning in `auto` and is an error with `tokpack` |
| `--budget FLOPS` | required | training FLOPs, for example `6.8e13` |
| `--flops_mode` | `model` | `model`: the budget counts model forward+backward FLOPs. `total`: optimizer FLOPs (Muon Newton-Schulz) count too. Both are logged either way |

**Model** (ZoomGPT: a decoder-only transformer with RoPE and a tied output head)

| flag | default | meaning |
|---|---|---|
| `--d` | 256 | width |
| `--layers` | 6 | number of blocks |
| `--heads` | 4 | attention heads (`d / heads` must be even) |
| `--seq` | 256 | training context length; also the default generation window |
| `--qk_norm` | 0 | 1: normalize queries and keys (QK-norm, a published technique) |
| `--mlp` | `gelu` | `gelu` or `relu2` (ReLU squared) |
| `--softcap` | 0 | logit soft cap (0 = off; 15 is typical) |
| `--zero_init_proj` | 0 | 1: zero-initialize the output projections |

The defaults give 5.8M parameters with a 4,096-token vocabulary.

**Zoom Training**

Zoom Training merges neighbouring tokens into one position for part of the budget, then
trains at full resolution. It is a learned-gate variant of patch-level training (Shao
et al., ICLR 2025); see [METHOD.md](METHOD.md).

| flag | default | meaning |
|---|---|---|
| `--zoom SPEC` | `1` | schedule as `k:fraction,...` over the FLOP budget. `1` is the plain baseline. `2:0.5,1:0.5` merges pairs for the first half. Fractions must sum to 1 and every k must divide `--seq`. A schedule that does not end at k=1 gets a warning. A bare `4` means `4:1` |
| `--batch_scale_k` | 0 | 1: use batch x k sequences while zoomed (same positions per step) |
| `--batch_scale X` | 0 | multiply the batch by X while zoomed (k > 1); overrides `--batch_scale_k` |
| `--merge_mode` | `''` | `gate`, `mean` or `linear`; empty means `--gated` decides |
| `--gated` | 1 | 1: learned gate merge, 0: plain mean |
| `--out_mode` | `slot` | `slot`: per-token outputs; `bag`: bag-of-tokens loss (ablation) |
| `--rope_pos` | `token` | `token` or `patch` positions while zoomed (ablation) |
| `--mix_fine P` | 0 | in coarse stages, run a k=1 step with probability P |
| `--lr_restart` | 0 | 1: restart warmup + cosine at each stage switch (measured worse) |

On TinyStories at 5.8M parameters (CPU, Muon, 2 or 3 seeds), the recipe that helped was
k=2 only, with a larger batch while zoomed (`--zoom 2:0.5,1:0.5 --batch_scale_k 1` or
`--zoom 2:0.33,1:0.67 --batch_scale 2`). Schedules with k=4 and no batch scaling did
worse than the baseline; with `--batch_scale_k 1` the original k=4 schedule only
matched it (1 seed). Numbers and conditions are in [../RESULTS.md](../RESULTS.md).

**Optimizer and schedule**

| flag | default | meaning |
|---|---|---|
| `--optim` | `adamw` | `adamw`, or `muon` (Muon on the hidden matrices, AdamW on embeddings, head and norms) |
| `--lr` | 1e-3 | peak learning rate. With `--optim muon` the AdamW part uses the same LR |
| `--wd` | 0.1 | weight decay |
| `--ns` | `quintic` | Muon orthogonalizer: `quintic` Newton-Schulz or `polar` (Polar Express) |
| `--state_bits` | 32 | 8: blockwise 8-bit optimizer state |
| `--param_dtype` | `fp32` | `bf16`: weights stored in bf16, updates written with stochastic rounding (no fp32 master copy) |
| `--warmup` | 0.05 | warmup fraction of the budget |
| `--sched` | `cosine` | `cosine` or `wsd` (warmup-stable-decay) |
| `--decay_frac` | 0.45 | `wsd` only: fraction of the budget spent decaying |

The measured CPU recipe is `--optim muon --lr 1.6e-2`. That LR applies to both the Muon
and the AdamW parts, so it is 16x the best AdamW-only LR (1e-3). AdamW settings do not
transfer to Muon runs.

**Batch and memory**

| flag | default | meaning |
|---|---|---|
| `--batch` | 16 | sequences per step (per process) |
| `--accum` | 1 | gradient accumulation steps |
| `--grad_ckpt` | 0 | 1: activation checkpointing per block |
| `--ce_chunk N` | 0 | chunked cross-entropy: compute the loss over N positions at a time and recompute their logits in backward, so the full (batch, seq, vocab) logits are never built (for example 2048; 0 = off). Same loss up to fp32 summation order; one extra head matmul in backward (counted in `model_flops_exact`, not in the budget). `--out_mode bag` at k > 1 is not chunked |
| `--auto_batch` | off | GPU only: split `batch x accum` into the largest micro-batch that fits, keeping the effective batch, schedule and data order. On CPU it prints a note and does nothing |

The `bantam memory` scoreboard rows 2-5 assume chunked cross-entropy (`--ce_chunk 2048`).

**Evaluation**

| flag | default | meaning |
|---|---|---|
| `--evals` | 10 | periodic evaluations over the run |
| `--eval_seqs` | 256 | periodic evals score the first this-many val windows |
| `--final_eval_seqs` | 2048 | `final_val_loss` scores the first this-many val windows |

The two subsets differ, and early windows are easier, so a periodic eval reads about 0.27
to 0.30 nats lower than the final eval of the same weights. Compare finals with finals.
The final model is also scored on the periodic subset (`final_val_loss_eval_seqs`). A
bf16-weight model is always scored as an fp32 copy.

**Runtime**

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--precision` | `auto` | GPU autocast: `auto`, `fp32`, `bf16`, `fp16`. `auto` picks bf16 on compute capability 8.0+, else fp16 with a GradScaler. CPU runs are always fp32 |
| `--compile` | off | `torch.compile` the model |
| `--threads` | 4 | CPU threads |
| `--duty` | 1.0 | cap the average device load (0 < duty <= 1) when a hardware power cap is not available |
| `--seed` | 0 | seed for init and data order |

**Output, checkpoints and resume**

| flag | default | meaning |
|---|---|---|
| `--name` | required | run name |
| `--out` | `results/llm` | directory for `<name>.json` (the run log, rewritten at every eval) |
| `--save PATH` | `''` | write the final weights to PATH (a state_dict) and a `<root>_pretrained/` directory (config.json, weights, the data dir's `tokenizer.json`) that `bantam generate` reads |
| `--save_every_eval` | 0 | 1: also save `<root>_f<frac><ext>` at every eval. Needs `--save`: without it the command is a usage error (exit 2) |
| `--ckpt_every N` | 0 | write a resumable checkpoint (model, optimizer, data position, RNG) every N steps |
| `--ckpt PATH` | `<out>/<name>.ckpt` | checkpoint path |
| `--resume PATH` | `''` | continue from a checkpoint. Bit-exact on CPU (tested) |
| `--allow_arg_change` | off | allow `--resume` with different model, data-order or schedule options, or a different number of processes |
| `--stop_step N` | 0 | stop after N steps (for tests) |

The run JSON records the arguments, the data source, the validation curve (loss against
FLOPs, tokens, steps and wall-clock), `model_flops`, `model_flops_exact` (the matmuls the
kernels execute), `opt_flops` and a `status` of `finished` or `diverged`. A diverged run
exits with status 3 and writes no `--save` model. Bad arguments exit with status 2 before
any file is written.

**Examples**

```bash
# baseline, the measured CPU recipe (TinyStories, 5.8M params)
bantam pretrain --data data/tinystories --budget 6.8e13 --zoom 1 \
    --optim muon --lr 1.6e-2 --name base

# the best measured settings (same as the next command without --save)
bantam pretrain --data data/tinystories --budget 6.8e13 --recipe best --name best

# QK-norm plus Zoom (k=2 for the first third, batch x2), keep the model
bantam pretrain --data data/tinystories --budget 6.8e13 --zoom 2:0.33,1:0.67 --batch_scale 2 \
    --qk_norm 1 --optim muon --lr 1.6e-2 --name qk_zoom --save runs/qk_zoom.pt

# two processes with DDP (tested on CPU with 2 gloo ranks; not on GPUs)
torchrun --standalone --nproc_per_node 2 -m bantamtrain pretrain \
    --data data/tinystories --budget 6.8e13 --optim muon --lr 1.6e-2 --name ddp_base
```

---

## bantam finetune

Fine-tunes any Hugging Face causal language model on instruction or chat data: LoRA,
QLoRA (LoRA on a quantized frozen base), DoRA, or full fine-tuning. It needs
`pip install 'bantamtrain[finetune]'`.

Required: `--model`, `--out`, and one of `--dataset` or `--jsonl`. A missing extra, a missing
data source, or a model or dataset that cannot be loaded fails before `--out` is created.

**Model and data**

| flag | default | meaning |
|---|---|---|
| `--model` | required | Hugging Face model id or local directory |
| `--dataset ID` | `''` | Hugging Face dataset id |
| `--config`, `--split` | `''`, `train` | dataset config and split |
| `--jsonl FILE` | `''` | a local JSONL file instead of `--dataset` |
| `--prompt_col`, `--response_col` | `prompt`, `response` | column names. `instruction`/`context`/`response` (Dolly), `instruction`/`output`, `question`/`completion` are also read |
| `--messages_col` | `messages` | chat column. ShareGPT `conversations` is also read |
| `--max_examples N` | 0 (all) | use N examples, chosen by a seeded shuffle |
| `--eval_frac` | 0.05 | fraction held out for evaluation |
| `--eval_jsonl FILE` | `''` | separate held-out JSONL; overrides `--eval_frac` |
| `--seq` | 512 | window length in tokens |
| `--pack` | `bfd` | `bfd`: best-fit-decreasing packing; `wrapped`: concatenate and cut; `none`: one example per row. Packed examples get a block-diagonal mask, so they never see each other; sliding-window layers (Mistral, Phi-3, Starcoder2, Gemma-2/3, Qwen2/3 with `use_sliding_window`) also keep their window |
| `--prompt_loss_weight` | 0 | 0: loss on response tokens only; 1: train on everything. Eval always scores response tokens only |

The tokenizer's chat template is used when it has one. A base model without one gets
ChatML when its vocabulary has `<|im_end|>` (SmolLM2, Qwen2.5). Otherwise a plain format
is used: `### Instruction:` / `### Response:` for prompt/response data, `role: content`
lines for chat data.

**Adapters**

| flag | default | meaning |
|---|---|---|
| `--lora R` | 16 | LoRA rank; 0 = full fine-tuning |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--lora_targets` | `''` = all-linear | comma list of module names, or `all-linear`: every linear layer (also GPT-2 Conv1D) except the output head |
| `--rslora` | -1 | rank-stabilized scaling; -1 = on when r >= 64 |
| `--dora` | 0 | 1: DoRA adapters (magnitude plus direction) |
| `--lora_plus RATIO` | 0 | LoRA+: the B matrices get lr x RATIO (0 = off) |

**Quantized base (QLoRA)**

| flag | default | meaning |
|---|---|---|
| `--quant` | `0` | quantize the frozen base: `8` (int8), `4` (linear int4) or `nf4` (QLoRA codebook). Needs `--lora` > 0 |
| `--embed_quant` | `8` | embeddings and output head: `8` (int8), `same` as `--quant`, or `none` (float) |
| `--low_ram` | 0 | 1 (with `--quant`): load and quantize tensor by tensor, never holding the float model. Needs `.safetensors` weights: with `pytorch_model.bin` only, it warns and loads the float model first (`report.quant.low_ram` is `false` in `log.json`) |
| `--grad_ckpt` | 0 | 1: activation checkpointing. Recommended with `--quant` (a note is printed otherwise) |
| `--ce_chunk` | 2048 | chunked cross-entropy: tokens per chunk, so the full logits are never built |

**Optimizer and schedule**

| flag | default | meaning |
|---|---|---|
| `--optim` | `adamw` | `adamw` or `muon` |
| `--muon_lora` | -1 | put the LoRA matrices on Muon; -1 = on with `--optim muon` and `--lora` > 0 |
| `--state_bits` | 8 | optimizer state bits: 8 (blockwise 8-bit) or 32. The default AdamW is therefore 8-bit AdamW |
| `--param_dtype` | `auto` | `auto`, `fp32`, `bf16`. `auto`: fp32 for LoRA on CPU and on GPUs older than compute capability 8.0, bf16 otherwise; bf16 for `--lora 0` |
| `--lr` | 2e-4 | peak learning rate |
| `--warmup` | 0.03 | warmup fraction |
| `--sched` | `wsd` | `wsd` or `cosine` |
| `--wd` | 0 | weight decay |
| `--ns` | `quintic` | Muon orthogonalizer |
| `--epochs` | 1.0 | passes over the training windows. An epoch trains on every window once; its last optimizer step may hold fewer micro-batches (its loss is the mean over its own supervised tokens) |
| `--max_steps` | 0 | stop after this many optimizer steps (0 = use `--epochs`). Each pass over the windows still ends as an epoch does, possibly with a shorter step |
| `--drop_last` | 0 | 1: the 0.2 schedule, full optimizer steps only, so an epoch skips the windows that do not fill a last step. A checkpoint written before 0.3's full epochs resumes with it |
| `--batch` | 4 | rows per micro-batch |
| `--accum` | 1 | gradient accumulation; the loss is averaged over every supervised token of the step |
| `--examples_per_step N` | 0 | after packing, set `--accum` to `max(1, round(N / (batch x processes x mean examples per window)))`, so an optimizer step sees about N examples (0 = off; it overrides `--accum`). It cannot go below one micro-batch per step, so lower `--batch` for fewer examples per step |

**Evaluation, runtime, output**

| flag | default | meaning |
|---|---|---|
| `--eval_every` | 50 | eval every N steps; 0 = only at step 0 and at the end. The final eval always runs |
| `--eval_only` | 0 | 1: print the held-out loss and exit |
| `--device`, `--precision` | `auto`, `auto` | as in `pretrain` |
| `--threads`, `--duty`, `--seed` | 4, 1.0, 0 | as in `pretrain` |
| `--ftz` | 1 | CPU flush-to-zero for this process (0 = keep denormals). Set in `main()`; importing `bantamtrain.finetune` changes nothing |
| `--out DIR` | required | output directory |
| `--save_peft` | -1 | also write a PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`); -1 = on |
| `--save_merged` | 0 | 1: also `save_pretrained()` the merged model to `<out>/merged`. With `--quant`, the adapters are merged into a freshly loaded float base |
| `--save_every N` | 0 | write a resumable checkpoint to `<out>/checkpoint` every N steps |
| `--resume DIR` | `''` | continue from a checkpoint dir (or its `state.pt`); refused if training options differ |
| `--stop_step N` | 0 | stop after N steps (for tests) |

Outputs in `--out`: `log.json` (arguments, data counts, train and eval curves, memory),
`adapter.pt` and `adapter_meta.json` (LoRA runs; it records `--quant` and `--embed_quant`
for `bantam chat`), the PEFT adapter files, the tokenizer
with the chat template it was trained with, `model/` (full fine-tuning) and `merged/`
(with `--save_merged 1`).

`log.json` also gives `steps_per_epoch`, `windows_used` (windows one epoch trains on, all
processes), `examples_per_window` and `examples_per_step` (mean examples per optimizer
step). Packing puts several examples in a window, so a step sees more examples than
`batch x accum`; when it is more than twice that, a note suggests `--examples_per_step`
or a higher `--lr`.

**Examples**

```bash
# LoRA r=16 on SmolLM2-135M with Dolly; 8-bit AdamW (the default optimizer)
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --lora 16 --epochs 1 --out runs/dolly

# the same with Muon on the LoRA matrices (in the fine-tuning race it did not beat AdamW)
bantam finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --lora 16 --optim muon --epochs 1 --out runs/dolly_muon

# ask for about 4 examples per optimizer step after packing (the chosen --accum is printed)
bantam finetune --model HuggingFaceTB/SmolLM2-135M --jsonl my_data.jsonl \
    --batch 1 --examples_per_step 4 --out runs/mine

# QLoRA-style: NF4 base with int8 embeddings, loaded tensor by tensor, checkpointing
bantam finetune --model HuggingFaceTB/SmolLM2-135M --jsonl my_data.jsonl \
    --lora 16 --quant nf4 --low_ram 1 --grad_ckpt 1 --save_every 100 --out runs/qlora

# multi-process DDP (tested on CPU with 2 gloo ranks; not on GPUs)
torchrun --standalone --nproc_per_node 2 -m bantamtrain.finetune \
    --model HuggingFaceTB/SmolLM2-135M --jsonl my_data.jsonl --out runs/ddp
```

---

## bantam weather

Trains a small ZoomViT (1.26M parameters at the defaults) that forecasts ERA5 weather 6
hours ahead, at a fixed FLOP budget. It reads a directory written by
`bantam prepare-era5`. With `--forecast` it instead rolls out a saved model. Runs in one
process (no DDP).

Required to train: `--data`, `--budget`, `--name`.

**Recipe**

| flag | default | meaning |
|---|---|---|
| `--recipe` | `none` | `best`: set `--optim muon --lr 1.6e-2 --coarse_mode lowres --zoom 4:0.6,1:0.4` for every one of these options the command line does not give. `none`: no change |

`--recipe best` is the best combination measured so far, at small scale on CPU only
(ERA5 64x32, the 1.26M-parameter default ViT, 5.5e13 FLOPs, 2018 test year): one-step
MSE 0.112 (1 seed) against 0.124 for the Muon baseline (2 seeds). It works as in
`pretrain`: explicit options win, the LR is applied only with Muon, and the log records
`args.recipe` and `args.recipe_applied`.

**Data, splits and evaluation**

| flag | default | meaning |
|---|---|---|
| `--data DIR` | none | directory from `prepare-era5`. Reads `fields.gridpack` if present, else `fields.npy`; a pack that does not hold the data of `fields.npy` is skipped with a warning |
| `--val_year Y` | none | train on years before Y, run the periodic evals on [Y, test year) and touch the test year only in the final eval. Data prepared with `prepare-era5 --val_year` uses its validation split by default |
| `--evals` | 10 | periodic evals (0 = first and final only) |
| `--ic_stride` | 12 | rollout initial conditions: every Nth 6 h step of the eval split (12 = 00 UTC every 3 days; 14 alternates 00/12 UTC) |
| `--max_ics` | 96 | cap on rollout initial conditions (0 = no cap) |

Without `--val_year` (and without a validation split in `meta.json`), the periodic and
final evals both use the test year, as all committed runs did. Metrics: one-step MSE in
normalized units (`val_loss`), and latitude-weighted RMSE and ACC at 6, 24 and 72 h for
z500, t850 and t2m, next to persistence and a train-years climatology.

**Budget, model, Zoom**

| flag | default | meaning |
|---|---|---|
| `--budget FLOPS` | none | training FLOPs (required to train) |
| `--flops_mode` | `model` | as in `pretrain` |
| `--d`, `--layers`, `--heads` | 128, 6, 4 | ViT size |
| `--zoom SPEC` | `1` | as in `pretrain`; k is the number of grid tokens merged and must be a square (4 = 2x2) |
| `--coarse_mode` | `merge` | `merge`: Zoom merge of tokens; `lowres`: train on coarsened fields during the coarse stages (progressive resizing, a known technique) |
| `--batch_scale_k` | 0 | 1: batch x k while zoomed (both modes) |
| `--merge_mode`, `--gated`, `--mix_fine`, `--lr_restart` | | as in `pretrain` |

**Optimizer and schedule**

| flag | default | meaning |
|---|---|---|
| `--optim` | `adamw` | `adamw` or `muon` |
| `--lr` | 1e-3 | peak LR (the measured runs used Muon 1.6e-2 and AdamW 2e-3) |
| `--wd` | 0.05 | weight decay |
| `--state_bits`, `--ns` | 32, `quintic` | as in `pretrain` |
| `--batch` | 16 | samples per step |
| `--warmup`, `--sched`, `--decay_frac` | 0.05, `cosine`, 0.45 | as in `pretrain` |

**Runtime, output, resume**

| flag | default | meaning |
|---|---|---|
| `--device`, `--precision`, `--compile`, `--threads`, `--duty`, `--seed` | | as in `pretrain` |
| `--name`, `--out` | none, `results/weather` | run name and log directory |
| `--save PATH` | `''` | write the trained model (weights, config, normalization stats) |
| `--ckpt_every`, `--ckpt`, `--resume`, `--stop_step` | | as in `pretrain`. `--resume` refuses any change to the model, budget, batch, LR, schedule, weight decay or seed |

**Forecast mode**

| flag | default | meaning |
|---|---|---|
| `--forecast` | off | roll out a saved model instead of training |
| `--load PATH` | `''` | model file written by `--save` |
| `--init TIME` | `''` | initial time, for example `2018-03-01T00` |
| `--steps N` | 12 | number of 6 h steps |
| `--forecast_out FILE` | `<out>/forecasts/forecast_<init>.npy` | output array (steps+1, channels, lat, lon) in physical units, plus a `.json` sidecar with valid times and RMSE where the data covers them |

**Examples**

```bash
# Muon baseline, the measured CPU recipe (ERA5 64x32, FLOP-matched at 5.5e13)
bantam weather --data data/era5_64x32 --budget 5.5e13 --zoom 1 \
    --optim muon --lr 1.6e-2 --name base --save runs/wx_base.pt

# low-resolution warm-up for 60% of the budget, then full resolution (= --recipe best)
bantam weather --data data/era5_64x32 --budget 5.5e13 --zoom 4:0.6,1:0.4 --coarse_mode lowres \
    --optim muon --lr 1.6e-2 --name lowres

# 3-day forecast from a saved model
bantam weather --forecast --load runs/wx_base.pt --data data/era5_64x32 \
    --init 2018-03-01T00 --steps 12
```

---

## bantam prepare-text

Turns a text dataset into a tokenizer plus token files: `DIR/tokenizer.json`,
`train.bin`, `val.bin` and `meta.json`. It streams, so memory stays flat however large
the corpus is. Needs `pip install 'bantamtrain[data]'`.

| flag | default | meaning |
|---|---|---|
| `--out DIR` | required | output directory (created only after the inputs check out) |
| `--hf REPO` | `''` | Hugging Face dataset repo, for example `roneneldan/TinyStories` |
| `--config NAME` | none | dataset config (subdirectory) with `--hf` |
| `--max_shards` | 1 | number of training parquet shards to download |
| `--cache DIR` | `~/.cache/bantamtrain_hf` | download cache |
| `--train FILE ...`, `--val FILE` | | local parquet files instead of `--hf` |
| `--val_frac` | 0.01 | held-out fraction when there is no validation split |
| `--column` | `text` | text column |
| `--vocab` | 4096 | BPE vocabulary to train |
| `--tokenizer NAME` | `''` | reuse a tokenizer instead of training one: a `tokenizer.json`, a model directory or a Hub id (for example `gpt2`) |
| `--eos TOKEN` | guessed | end-of-text token of `--tokenizer` |
| `--tok_docs` | 200000 | documents used to train the BPE |
| `--max_train_docs` | none | cap on training documents |
| `--drop_val_overlap` | 1 | 1: drop training documents identical to a validation document. 0 reproduces 0.2's output byte for byte |
| `--batch` | 4096 | documents per tokenizer batch |

Each document is stored without BOS and followed by EOS. Token ids are uint16, or uint32
when the vocabulary is larger than 65,536. Re-preparing a directory deletes the packs of
an earlier `pack-text` run, which no longer match.

```bash
bantam prepare-text --hf roneneldan/TinyStories --out data/tinystories --vocab 4096
bantam prepare-text --hf roneneldan/TinyStories --out data/ts_gpt2 --tokenizer gpt2
```

---

## bantam pack-text

Removes duplicate documents from a prepared corpus and stores it as compressed TokPack
files (`train.tokpack`, `val.tokpack`, `pack_stats.json`). `pretrain` reads the packs
when present (`--data_format auto`). Training documents that duplicate or nearly
duplicate a validation document are dropped. Validation is never deduplicated against
itself.

| flag | default | meaning |
|---|---|---|
| `--data DIR` | required | directory from `prepare-text` |
| `--dedup` | 1 | 1: exact and MinHash near-duplicate removal on the training set |
| `--drop_val_overlap` | 1 | with `--dedup 0`: still drop exact copies of validation documents |
| `--ngram`, `--num_perm`, `--bands`, `--threshold` | 8, 64, 8, 0.8 | MinHash settings (n-gram size, permutations, LSH bands, similarity threshold) |
| `--chunk` | 16384 | tokens per compressed chunk |
| `--level` | 19 | zstd level |

```bash
bantam pack-text --data data/tinystories
```

---

## bantam prepare-era5

Downloads a small ERA5 subset from WeatherBench2 (5.625 degrees, 64x32 grid, 6-hourly;
channels z500, t850, t2m, u10, v10, msl) into `fields.npy` plus `meta.json` with
normalization statistics from the training years only. Needs
`pip install 'bantamtrain[weather]'`.

| flag | default | meaning |
|---|---|---|
| `--out DIR` | required | output directory |
| `--start`, `--end` | `2010-01-01`, `2018-12-31T18` | time range |
| `--test_year` | 2018 | test year |
| `--val_year` | none | hold out this year for validation (for example 2017) |

Every step before the first held-out year is training data. A range that leaves no
training data or too few test steps is an error.

```bash
bantam prepare-era5 --out data/era5_64x32 --val_year 2017
```

---

## bantam pack-era5

Packs `fields.npy` into an error-bounded GridPack (`fields.gridpack`) and reports the
compression and the per-channel error. `bantam weather` reads the pack when it holds the
same data as `fields.npy`.

| flag | default | meaning |
|---|---|---|
| `--data DIR` | required | directory from `prepare-era5` |
| `--err_frac` | 0.05 | maximum error as a fraction of each channel's 6 h change standard deviation |
| `--chunk` | 64 | time steps per compressed chunk |
| `--level` | 10 | zstd level |

```bash
bantam pack-era5 --data data/era5_64x32
```

---

## bantam memory

Estimates memory and training time for one GPU. This is arithmetic from stated
assumptions, not a measurement: nothing in it has been checked on a GPU. Two tables:

- the default **scoreboard**: one model size, each row adding one technique (8-bit
  state, bf16 weights, Muon, ...), with memory, fit, largest micro-batch and days;
- `--finetune`: full, LoRA and QLoRA fine-tuning of a model size, with and without
  activation checkpointing.

**Model and GPU**

| flag | default | meaning |
|---|---|---|
| `--params` | 5e9 | parameter count |
| `--d` | 0 | width (0 = pick from `--params`) |
| `--vocab` | 32000 | vocabulary |
| `--tied` | off | output head tied to the embedding |
| `--seq` | 2048 | sequence length |
| `--gpu_gb` | 96 | GPU memory as sold, read as GiB (a 24 GB card = 24) |
| `--headroom` | 0.85 | usable fraction of GPU memory (1 = the 0.2 framing) |
| `--overhead_gb` | 3 (2 for `--finetune`) | CUDA context and workspaces, GiB |

**Time assumptions** (scoreboard)

| flag | default | meaning |
|---|---|---|
| `--tokens` | 1e11 | training tokens |
| `--peak_tflops` | 503.8 | dense BF16 peak of the GPU |
| `--mfu` | 0.40 | assumed fraction of peak achieved |
| `--muon_flops` | 0.52 | Muon's FLOPs to reach AdamW's loss, from the literature |
| `--global_batch_tokens` | 524288 | tokens per optimizer step (the Newton-Schulz cost is per step) |
| `--ns_steps` | 5 | Newton-Schulz iterations |
| `--fp8_speedup` | 1.20 | assumed FP8 throughput gain |
| `--power_tput` | 0.95 | throughput kept under the power cap |
| `--act_factor` | 1.0 | scale the measured activation constants (0.7 approximates fused kernels) |
| `--ce_chunk` | 2048 | chunked cross-entropy chunk (`--finetune`: 0 = unchunked) |
| `--bits_per_token` | 5.97 | TokPack size for the storage lines (measured on TinyStories with a 4k vocab; a 32k-vocab web corpus needs roughly 7.3-9) |
| `--markdown` | off | print a Markdown table |

**Fine-tune table**

| flag | default | meaning |
|---|---|---|
| `--finetune` | off | print the full / LoRA / QLoRA table instead |
| `--micro` | 4 | micro-batch |
| `--lora_r` | 16 | LoRA rank |
| `--lora_dropout` | 0.05 | LoRA dropout (as `bantam finetune`) |

Part of the scoreboard describes something Bantam cannot run today: there is no FP8
code path (row 4, "FP8 linears", is a literature estimate). Rows 2 to 5 assume chunked
cross-entropy, which `bantam pretrain` turns on with `--ce_chunk 2048` (off by default).

```bash
bantam memory                                        # 5B model on one 96 GiB card
bantam memory --finetune --params 7e9 --gpu_gb 24    # will a 7B fine-tune fit in 24 GB?
```

---

## bantam generate

Samples text from a ZoomGPT saved with `save_pretrained`, for example the
`<root>_pretrained/` directory that `bantam pretrain --save <root>.pt` writes. Uses a KV
cache. Needs `pip install 'bantamtrain[data]'` (for `tokenizers`).

| flag | default | meaning |
|---|---|---|
| `--model DIR` | required | directory with `config.json` (from `save_pretrained`) |
| `--tokenizer FILE` | `DIR/tokenizer.json` | tokenizer file |
| `--prompt TEXT` | required | prompt, encoded as `prepare-text` encodes documents: without the tokenizer's special tokens (no BOS), and special-token text typed in the prompt stays text |
| `--max_new` | 200 | new tokens |
| `--temperature` | 0.8 | <= 0: greedy |
| `--top_k` | 40 | 0: off |
| `--top_p` | 0.95 | 1: off |
| `--repetition_penalty` | 1.0 | > 1 penalizes tokens already in the row |
| `--max_ctx` | 0 | attention window (0 = the model's `block_size`, which `pretrain` sets to its `--seq`) |
| `--num_samples` | 1 | samples to print |
| `--seed` | 0 | sampling seed |
| `--eos_id` | from `config.json` | end-of-document id |
| `--doc_start` | 1 | 1: put the end-of-document token before the prompt, as every training document starts |
| `--device` | `cpu` | device |

```bash
bantam generate --model runs/qk_zoom_pretrained --prompt "Once upon a time" --num_samples 3
```

---

## bantam chat

Talks to a Hugging Face model, optionally with a fine-tuned adapter, and streams the
reply. Without `--prompt` it opens an interactive session: `/reset` clears the
conversation, `/exit` quits. The prompt is built exactly as in training (the saved
tokenizer and chat template when there is one). Needs
`pip install 'bantamtrain[finetune]'`.

| flag | default | meaning |
|---|---|---|
| `--model` | required | base model (Hugging Face id or local dir) |
| `--adapter` | `''` | a `bantam finetune` output dir, an `adapter.pt`, or a PEFT adapter dir |
| `--quant` | as in training | quantize the base: `0`, `8`, `4`, `nf4`. Default: the adapter's training run (`adapter_meta.json` or `log.json`), else `0` |
| `--embed_quant` | as in training | embeddings and output head with `--quant`: `8`, `same`, `none`. Default: the training run's, else `8` |
| `--prompt` | `''` | answer this and exit |
| `--system` | `''` | system message |
| `--max_new` | 256 | new tokens |
| `--temperature` | 0.7 | 0 = greedy |
| `--top_p` | 0.95 | nucleus sampling |
| `--plain_format` | `instruction` | `instruction` or `messages`: prompt format for models without a chat template |
| `--lora_alpha` | none | alpha of an `adapter.pt` that has no `adapter_meta.json` / `log.json` |
| `--rslora` | -1 | override the adapter's rsLoRA setting |
| `--dtype` | `auto` | `auto`, `fp32`, `bf16` |
| `--device`, `--seed` | `auto`, 0 | device and sampling seed |
| `--ftz` | 1 | CPU flush-to-zero (0 = keep denormals) |

```bash
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly --prompt "Name three colours"
bantam chat --model HuggingFaceTB/SmolLM2-135M --adapter runs/dolly        # interactive
```

---

## bantam doctor

Prints an environment report to paste into a bug report: versions of bantamtrain,
Python, torch and the optional packages; CUDA, cuDNN and NCCL; GPU names with free and
total memory; bf16/fp16/TF32 support; the flush-to-zero state; thread counts and CPU
affinity; free RAM and disk; and which extras are importable.

| flag | default | meaning |
|---|---|---|
| `--json` | off | print the report as JSON |

```bash
bantam doctor
bantam doctor --json > env.json
```
