# Changelog

All notable changes to Bantam (PyPI `bantamtrain`, import `bantamtrain`, command
`bantam`). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow semantic versioning; before 1.0, any release may change behaviour,
and every such change is listed under "Breaking and behaviour changes".

Ids in brackets, such as (FT-14), refer to findings of the code audit done for 0.3. A few
ids were issued twice by different parts of the audit; those carry the part after a
slash: `/optim`, `/ft` (fine-tuning), `/gpu`, `/data`, `/pkg` (packaging), `/exp`
(experiments), `/quant`. Fixes from the later review that have no audit id cite their
commit instead.

Every measured number in this file comes from one CPU-only sandbox (4 cores, 16 GB RAM,
PyTorch 2.15 for CPU) with small models. Nothing has been run on a GPU.

## [0.3.0]

### Highlights

- **One command.** The package is `bantamtrain` everywhere, with a single `bantam`
  command (also `python -m bantamtrain`) for pretraining, fine-tuning, weather, data
  preparation and planning. New commands: `bantam generate`, `bantam chat`,
  `bantam doctor`. See [docs/CLI.md](docs/CLI.md) and [docs/GUIDE.md](docs/GUIDE.md).
- **Fine-tuning fixes and interoperability.** 0.2 had bugs that changed results:
  packed examples attended to each other, `--optim muon` with LoRA trained nothing with
  Muon, the chunked loss skipped Gemma/Cohere/Granite logit transforms, and gradient
  accumulation weighted micro-batches wrongly. All are fixed. New: DoRA, LoRA+,
  all-linear targets (including GPT-2 Conv1D), PEFT export and import, a low-RAM
  quantized loader, and exact resume.
- **Weather.** A validation year, ACC and climatology metrics, saved models, resumable
  checkpoints and forecasts.
- **Streaming data pipeline.** Text preparation, deduplication and packing now stream, so memory
  stays flat. Existing Hugging Face tokenizers can be reused, vocabularies above 65,536
  are stored as uint32, and training documents identical to validation documents are
  dropped by default.
- **Corrected numbers.** The published-model comparison now scores every family in its own
  training format, which widened Bantam's gap to the TinyStories models. The 5B memory
  scoreboard is corrected from 2.56x to about 1.9x (GB versus GiB), bf16 runs are scored
  in fp32, and several claims were removed (see "Results corrected").
- **New measured results** (TinyStories, 5.8M parameters, FLOP-matched at 6.8e13 model
  FLOPs, Muon LR 1.6e-2, CPU only; final validation loss in nats per token on the first
  2,048 validation windows, +- the sample standard deviation over seeds):

  | recipe | loss | seeds | baseline compute needed to match |
  |---|---|---|---|
  | Muon baseline | 2.737 +- 0.012 | 3 | 1.00x |
  | Zoom k=2 for half the budget, batch x2 | 2.685 +- 0.004 | 3 | 1.12x |
  | QK-norm | 2.639 +- 0.003 | 2 | 1.24x |
  | QK-norm + Zoom k=2 for a third, batch x2 | 2.589 +- 0.006 | 2 | 1.41x |

  QK-norm is a known technique and beat Zoom alone; the two stack. On weather (ERA5
  64x32, a 1.26M-parameter ViT, 5.5e13 FLOPs, scored on the 2018 test year), Zoom made
  results worse under Muon (one-step MSE 0.159 against 0.124, 2 seeds), while the
  established low-resolution warm-up helped (0.112, 1 seed).
- **Quality checks.** 594 tests pass and 7 are skipped (CUDA-only or slow) on CPU. A
  GitHub Actions workflow runs the suite and a packaging check.

### Breaking and behaviour changes

Grouped by command. Unless a line says otherwise, the experimental protocol is
unchanged: the budget counts model FLOPs, and LR schedules, evaluation subsets, data
splits and data order stay as before. AdamW fp32 pretraining runs were checked to be
bit-identical to the code before the audit.

**Package and command line**

- The repository package `zoomtrain` is now `bantamtrain` (the PyPI releases 0.1 and 0.2
  were already built under that name). The training and data scripts moved into
  `bantamtrain/cli` and run as `bantam <command>`. `llm/*.py` and `weather/*.py` remain
  as thin wrappers in the repository. Old checkpoints that pickled
  `zoomtrain.lowmem.QState` still load.
- Dependency floors: `torch>=2.4` (was 2.2), `transformers>=4.56` plus `safetensors` in
  the `finetune` extra (was 4.40), `setuptools>=77` to build. The `test` extra now holds
  everything the tests import. The version is defined once, in
  `bantamtrain.__version__`.
- `bantam <command> --help` works without the optional extras. A command whose extra is
  missing prints the `pip install 'bantamtrain[...]'` to run and exits 1 instead of a
  traceback. `bantam` returns the command's exit status; an unknown command returns 2.
- An unknown `--precision` value is now an error. `pretrain` checks it with argparse
  choices (`auto`, `fp32`, `bf16`, `fp16`; exit 2); `weather` and `finetune` raise a
  ValueError. Before, any typo silently trained in fp32 on a GPU.
- `--precision auto` on CUDA picks fp16 with a GradScaler below compute capability 8.0.
  Before, it chose emulated bf16 whenever `torch.cuda.is_bf16_supported()` said yes.

**`bantam pretrain` and the ZoomGPT model**

- A bf16-weight model computes its loss in fp32 and is evaluated as an fp32 copy of its
  bf16 weights. Validation losses of `--param_dtype bf16` runs differ from 0.2 logs by
  the rounding noise of 0.2's bf16 arithmetic: up to about 0.01 nats, of either sign
  (`results/llm_lowmem/bf16_eval_bias.json`). fp32 runs are bit-identical.
- The logged train loss is the mean over accumulation micro-batches and DDP ranks (was
  rank 0's last micro-batch). Identical for `--accum 1` on one process.
- Data order: DDP ranks share one permutation and read disjoint slices (they used to read
  independent permutations). Epochs after the first are reshuffled with `[seed, epoch]`
  (they replayed epoch 0), and a wrap never returns a short batch. The first epoch on
  one process is unchanged; every committed TinyStories run used under 0.04 epochs, so
  none reached a second epoch.
- A non-finite loss is checked every step (was every 25). A diverged run exits with
  status 3, its JSON says `"status": "diverged"`, and it runs no final eval and writes no
  `--save` model. Before, it exited 0 with a NaN final loss and a saved model.
- Bad arguments exit with status 2 before any file is written: a validation split too
  small for `--eval_seqs` / `--final_eval_seqs`, a zoom level that does not divide
  `--seq`, a corpus smaller than one step's batch, `--duty <= 0`, model options the
  model rejects (`--d`/`--heads`, `--merge_mode`, `--softcap`), and a missing `--resume`
  file. Zoom levels above 4 now work.
- `--merge_mode` accepts only `''`, `gate`, `mean`, `linear` (exit 2). An unknown mode
  used to select the `mean` ablation silently.
- `--resume` refuses changed model, data-order or schedule options and a changed number
  of processes (exit 2) unless `--allow_arg_change`. The run JSON's `args` now records
  the resuming invocation, with the older args in `resume_history`. Restored parameter
  groups take this run's weight decay, `--ns`, `--state_bits` and LR scales.
- When one step crosses several eval points, the eval schedule catches up in one jump.
- `--save` is written before the final eval, plus a `<root>_pretrained/` directory from
  `save_pretrained` holding the data dir's `tokenizer.json`, with `eos_id` in
  `config.json`. `--save_every_eval` names use the file's own extension (unchanged for
  `*.pt`).
- The model's `block_size` is `--seq` (was 256). This changes only the default
  generation window, the RoPE table size and `config.json`; training is unchanged.
- `--data_format auto|bin` reads through `bantamtrain.data.open_tokens`: a `.tokpack`
  left from an earlier prepare run falls back to the `.bin` with a warning (instead of an
  error), and a `.bin` is read with the dtype in `meta.json` (uint16 when absent). A
  0.2-era `train.bin` gives a warning that it may hold validation copies.
- The run JSON is written atomically as strict JSON (non-finite values become `null`).
  New fields: `status`, `data_source`, `flops_mode`, `opt_flops_per_step`, `eval_subset`,
  `final_val_loss_eval_seqs`, `model_flops`, `model_flops_exact`, `opt_flops`,
  `total_flops` (also per eval), `resume_history`, `saved`, `auto_batch`.
- Checkpoint format: 8-bit optimizer state is stored as plain tensors (loads with
  `torch.load(weights_only=True)`), plus the CUDA RNG and GradScaler states. Pre-0.3
  checkpoints still load, with a warning.
- FLOP accounting: the default budget is unchanged for every recorded run. With
  `--merge_mode linear` (no recorded run used it), the per-slot merge and unmerge maps
  are now charged, so such runs take fewer steps. `flops_per_step(..., exact=True)` gives
  the matmuls the kernels execute and is logged as `model_flops_exact`.
- ZoomSchedule: an invalid schedule (k < 1, a negative or NaN fraction, fractions not
  summing to 1) raises ValueError (was an assert, or silently accepted). `parse("4")`
  means `4:1`. A schedule that does not end at k=1 warns. `stage_at` at or past the end
  returns the last stage with a non-zero share.
- The ZoomGPT constructor raises ValueError for an unknown `mlp` (used to fall back to
  gelu), an odd head dimension or `d % n_head != 0` (used to fail at forward time), and
  a bad `out_mode`/`rope_pos` (were asserts).
- `llm/train.py` exits with the command's status.

**`bantam finetune`**

- `--lora_targets` defaults to all-linear: every `nn.Linear`, quantized linear and
  transformers `Conv1D` except the output head. The 0.2 list of leaf names missed Phi-3
  `qkv_proj`/`gate_up_proj`, OPT/Phi `fc1`/`fc2`/`out_proj` and GPT-J `fc_in`/`out_proj`.
  The old list is kept as `DEFAULT_TARGETS` for explicit use. Names in a comma list are
  stripped of spaces.
- `--muon_lora` defaults to -1 (auto): on with `--optim muon` and `--lora` > 0. The 0.2
  default 0 made `--optim muon` a silent no-op with LoRA. An explicit 0 now warns.
- New `--save_peft`, on by default: a PEFT adapter is written next to `adapter.pt`.
- With `--accum` > 1 or DDP, the objective and the logged loss are the mean token loss
  over all supervised tokens of the whole step (every micro-batch and rank; was the mean
  of per-micro-batch means). Identical for `--accum 1` on one process.
- The eval loss counts response tokens only, also with `--prompt_loss_weight` > 0.
- The final eval always runs at the last step, also with `--eval_every 0`, and the last
  train step is always logged.
- `padding_fraction` is computed from segment ids (a pad token equal to EOS no longer
  counts real EOS tokens), and `train_windows` is the true global count.
- DDP: every rank uses `--seed` for the model init, LoRA dropout and bf16 rounding (was
  seed + rank); the data order still uses seed + rank. Up to world - 1 trailing training
  windows are dropped so every rank gets the same number of steps.
- Plain format (no chat template): the chat prompt ends at `assistant:`, with the space
  moved into the answer (the full text is unchanged). BOS is prepended for tokenizers
  that add it themselves (plain and installed-ChatML paths only). Trailing
  non-assistant turns are dropped, and the prompt/response boundary is the longest common
  token prefix.
- `--pack wrapped` no longer truncates examples to `--seq` before packing.
- An empty eval set is refused (evaluate() returns NaN instead of 0.0).
- `--lora 0` with `--quant` is an argparse error (it crashed or saved an unusable model).
- `--save_merged 1` with `--quant` merges the adapters into a freshly loaded float base
  (it was silently skipped).
- Outputs: the tokenizer is saved next to `adapter.pt` and in `model/` and `merged/`;
  `adapter_meta.json` is written. With installed ChatML, the saved `config.json` and
  `generation_config.json` stop at `<|im_end|>`.
- Models whose attention is not SDPA (eager or flex; Gemma-2 by default) get an additive
  float block-diagonal causal mask for packed or padded windows. They ignored the
  boolean mask before. Gemma-2 losses change and are now correct; SDPA models (Llama,
  SmolLM2, Qwen, GPT-2) are unchanged.
- `--param_dtype auto`: LoRA on a CUDA GPU older than compute capability 8.0 gets fp32
  weights (was emulated bf16). CPU (fp32 for LoRA, bf16 for `--lora 0`) and newer GPUs
  (bf16) are unchanged.
- A quantized output head goes to the chunked loss as a module and is dequantized one
  chunk at a time. Values are the same and peak memory is lower, but the head is
  dequantized about 3x per step instead of once.
- With `--quant` and no `--grad_ckpt 1`, a note recommends checkpointing. `--duty <= 0`
  exits with status 2 before loading.
- `--resume` (new in 0.3) also refuses a changed `--wd`, `--ns` or `--lora_dropout`.
- `--out` is created only once the model and the data are loaded: a missing extra
  (transformers, or `datasets` for `--dataset`), a missing `--dataset`/`--jsonl` (now a
  usage error, exit 2), an unreadable model or unusable data leave no empty directory.
- **Protocol change: an epoch trains on every window once.** `--epochs` used to plan
  `windows // (batch x accum)` full steps per epoch, which skipped up to
  `batch x accum - 1` windows (3 of 235 in the race). Now the last step of an epoch may
  hold fewer micro-batches (its loss is the mean over its own supervised tokens), so
  epoch-based runs take up to one more step per epoch and their LR schedules stretch
  accordingly. `--max_steps` keeps its step count, but a run that reaches the end of a
  pass now also takes that pass's short last step (it used to reshuffle early), so its
  windows from then on differ. `--drop_last 1` restores the old schedule exactly, and a
  checkpoint written before this change resumes with it (with a note).
- Packed or padded windows on models with sliding-window attention give those layers a
  block-diagonal mask cut to their window: one windowed mask for Mistral, Mixtral,
  Phi-3 and Starcoder2 when `sliding_window` is set, and transformers' per-layer-type
  mapping (`{"full_attention": ..., "sliding_attention": ...}`) for Gemma-2/3, Cohere2
  and Qwen2/3 with `use_sliding_window`. Their losses change at `--seq` above the
  window and are now correct; models without a window are unchanged.
- Flush-to-zero is set by `main()` behind the new `--ftz` (default 1, logged as
  `args.ftz`) instead of at import, so importing `bantamtrain.finetune` (as `bantam chat`
  does) no longer changes the process. Runs of the command are unchanged.
- `--low_ram 1` on a checkpoint without `.safetensors` weights falls back to
  `from_pretrained` + `quantize_model` with a warning (it raised FileNotFoundError);
  `report.quant` says `low_ram: false` and why. The `--low_ram` report is now
  `qweights.load_quantized`'s full `quant_report`.
- New `log.json` fields: `steps_per_epoch`, `windows_used`, `examples_per_window`,
  `examples_per_step`; `adapter_meta.json` gains `embed_quant`. A note is printed when
  packing makes an optimizer step see more than twice `batch x accum` examples.
- `--resume` also compares `--drop_last` and `--examples_per_step`.
- Weight decay (only when `--wd` > 0): HF `gate_proj` layers and their adapters are now
  decayed.

**`bantam chat`**

- `--quant` defaults to the adapter's training run (`adapter_meta.json` or `log.json`;
  `0` when unknown), and the new `--embed_quant` (default: the run's, else `8`) replaces
  the fixed int8 embeddings. An adapter trained with `--quant nf4` now chats over an NF4
  base by default (it used the float base); `--quant 0` restores that. New `--ftz`
  (default 1, as before through the finetune import).

**`bantam weather`**

- FLOP accounting: `--merge_mode linear` now charges the per-slot merge maps (fewer steps
  at the same budget), and `--coarse_mode lowres` charges patch embedding and decoding
  for the coarse patches only (more coarse steps). Gate and mean merges and the baseline
  are unchanged.
- The training order is reshuffled every epoch. Runs longer than one epoch differ from
  0.2 after the first wrap.
- `--batch_scale_k 1` with `--coarse_mode lowres` now scales the batch (it was silently
  ignored).
- `--compile`: the first step at each new shape counts as `compile_time`, not
  `train_time`, and the low-resolution stage is compiled too.
- Data source: `fields.gridpack` is used when it holds the data of `fields.npy` (same
  prep id, or matching first, middle and last steps), whatever the file times. A pack
  that does not is skipped for `fields.npy` with a warning (an error when there is no
  `fields.npy`). Data whose shape disagrees with `meta.json` is an error.
- `--data`, `--budget` and `--name` are no longer argparse-required, so `--forecast` can
  run without them. Training still exits with status 2 without them.
- Normalization runs in place and the GridPack is read chunk by chunk. Values are
  identical; peak RAM on the real data is about 1.07 GB.
- With a `meta.json` holding a validation split (`prepare-era5 --val_year`), the test
  split starts after it and the periodic evals use it by default (`split.val_from` in the
  log). Older `meta.json` files behave as before.
- `--resume` (new in 0.3) refuses any change to the model, budget, `--flops_mode`, batch,
  `--batch_scale_k`, LR, warmup, schedule, `--decay_frac`, `--ns`, `--lr_restart`,
  `--mix_fine`, weight decay or seed. The resumed JSON
  records the resuming arguments and `resume_history`. Checkpoints load with
  `weights_only=True`; one holding other pickled objects is refused.
- `--merge_mode` accepts only `''`, `gate`, `mean`, `linear` (exit 2), and `--duty <= 0`
  exits with status 2.
- Forecasts (new in 0.3): the sidecar's RMSE stops at the first gap in the data's times.
- New result fields, all additive: `split`, `flops_mode`, `opt_flops_per_step`,
  `model_flops`, `opt_flops`, `compile_time`, `final.clim_*`, `final.acc_*`,
  `final.acc_persist_*`, `final_val` (with a validation split), `resumed_from`.
- Not changed: the budget counts model FLOPs (`--flops_mode model`), the eval subsets,
  the split when no validation year is given, the LR schedule and the model init. On
  synthetic data, baseline, gate Zoom and Zoom + Muon + `--mix_fine` runs are
  bit-identical to 0.2.

**Data commands and stores (`prepare-text`, `pack-text`, `prepare-era5`, `pack-era5`)**

- `prepare-text` drops training documents identical to a validation document by default
  (`--drop_val_overlap 1`) and trains the BPE on that deduplicated sample. Re-preparing
  TinyStories therefore gives a slightly different `train.bin` and `tokenizer.json`;
  `--drop_val_overlap 0` reproduces 0.2's output byte for byte.
- `prepare-text` tokenizes `--batch` documents at a time (default 4096, was a hard-coded
  10000) with a cap of about 256K characters per batch; the output is unchanged. It
  creates `--out` only after the inputs check out. Vocabularies above 65,536 are stored
  as uint32 (`meta.json` records the dtype).
- `prepare-text` and `prepare-era5` delete stale packs, `pack_stats.json` and
  `meta.json` before writing, and write `meta.json` last with a manifest (a prep id plus
  the size and hash of every file).
- `pack-text --dedup 0` still drops training documents equal to a validation document
  and the tokens after the final EOS (`--drop_val_overlap 0` restores the raw pack).
- Packs written by `pack-text` include a document index and the prep id of their source.
  `pack_stats.json` adds `dtype`, `prep_id`, per-split `docs`, `doc_index_bytes`,
  `raw_bytes` and `ratio_vs_raw`, `dedup.docs_per_source` and `val_overlap`.
- `TokPack.stream()` yields every `n // seq` window of the global grid (it dropped each
  chunk's tail). Window order differs for a given seed; only the pack-text benchmark and
  tests use it, not pretraining.
- TokPack reads return read-only arrays; callers that write must copy.
- TokPack and GridPack check their sibling `meta.json` by default and raise
  `StalePackError` (a ValueError) for a stale pack; `verify=False` opts out.
- GridPack's scale is `2 x err_frac x diff_std` minus a float32 rounding margin, and
  channel means are accumulated in float64. Packs differ from 0.2's by a few bytes, with
  the same compression; the stated error bound now holds exactly.
- `prepare-era5`: the split follows `--start`/`--end`, `--test_year` (default 2018) and
  the new `--val_year`. A range whose years do not fit train < val < test, or that has no
  training data or 13 or fewer test steps, is an error (it used to give NaN statistics
  or an empty test set and exit 0). The default range gives the same split and
  statistics as 0.2.
- `dedup()` of an empty corpus reports `kept_frac` 1.0 (it crashed).

**`bantam memory`**

- GPU memory is in GiB (2^30 bytes), not 1e9 bytes.
- New `--headroom`, default 0.85 (1.0 in effect before). At the defaults, the 5B
  "standard" row no longer fits a 96 GiB card, and speedups are measured against the
  first row that fits. `--headroom 1` gives the old framing.
- Activation constants are measured (48.7 bytes per layer, token and width unit under
  AMP, 40.7 for bf16; was 34). Checkpointed inputs count 4 bytes under AMP. Logits count
  14 bytes per element (was 6). The cross-entropy chunk is 2048 tokens (was 1024).
- Rows 3 to 5 include Muon's Newton-Schulz FLOPs at `--global_batch_tokens 524288`
  (about +3% days), computed with `optim.ns_flops`.
- `--finetune` uses constants measured on HF Llama, adds `--lora_dropout 0.05`, applies
  GiB and the 0.85 headroom, adds a loss term, and charges QLoRA's int8 embeddings and
  dequantized weights. `--act_factor 1.6` is no longer needed.
- Output: the headers say GiB, days and speedup print `n/a` for rows that do not fit,
  and the text output has one extra header line.

**`bantam generate` and model inference**

- `ZoomGPT.generate()`: a temperature <= 0 means greedy decoding; `top_k` is clamped to
  the vocabulary and 0 or None disables it; the window defaults to `block_size`. The old
  signature and argument order still work.
- `ZoomGPT.from_pretrained` returns the model in eval mode.
- `bantam generate` with a `--model` that has no `config.json`, or `--num_samples` < 1, is
  a usage error (exit 2).

**Optimizers (every trainer)**

- `CombinedOptimizer.state_dict()` returns a torch-style dict
  (`state`, `param_groups`, `optimizers`) instead of a list. 0.2 checkpoints still load.
- Weight decay is decided by parameter role. HF `gate_proj` (and any 2-D `gate` Linear)
  is decayed when weight decay > 0; linear-mode merge maps are not; embeddings still are
  (opt out with `build_optimizer(..., decay_embeddings=False)`). No published run is
  affected.
- `--optim muon` on models with transformers `Conv1D` projections (GPT-2) trains them
  with Muon. Before, it silently used AdamW.
- bf16 stochastic rounding (and stochastic 8-bit codes) draw from a counter-based
  generator (`sr_seed`, default 0) instead of the global RNG. New bf16 runs are not
  bit-identical to 0.2 runs, but they are identical across DDP ranks and exact on resume.
- The 8-bit log map changed: code 0 is an exact zero and 254 levels cover the positive
  range. This changes 8-bit AdamW numerics for every tensor whose size is not a multiple
  of 256 or that holds zeros (the OPT-4/optim fix).
- `AdamWLowMem(v_rounding="auto")` rounds the 8-bit second moment stochastically when
  beta2 > 0.95. The CLI betas (0.9, 0.95) keep round-to-nearest, so CLI runs are
  unchanged.
- `--ns polar`: the fifth coefficient tuple gets the paper's safety factor and more than
  5 steps use the 8-tuple schedule (singular values at 5 steps were [0.846, 1.124], now
  [0.840, 1.124]).
- Newton-Schulz uses fused batched matmuls and in-place normalization (changes at fp32
  rounding level only). Muon state gains a per-parameter `step`; parameter groups gain
  `lr_scale`, `sr_seed` and `v_rounding`.
- 8-bit and 32-bit low-memory steps run in chunks of 4M elements, so the extra memory
  per step is per chunk. Rounding noise is drawn per chunk.
- `build_optimizer(name="muon")` warns when no parameter qualifies for Muon.

**Quantization, LoRA and WeightPack (library)**

- `qweights` block size: when a row width is not a multiple of the block, the largest
  divisor of the width in [block/2, block] is used (int8 at width 576 uses 96 instead of
  128 plus 64 padding columns); 4-bit blocks are even. Widths divisible by the block,
  including every ZoomGPT run, are unchanged.
- `qweights` scales: codes are computed against the stored fp16 scale; a tensor whose
  scales leave the fp16 normal range keeps fp32 scales; all-zero blocks get scale 0 (was
  1e-12). Dequantization forms code x scale in fp32 and rounds once to the compute
  dtype. Embedding outputs and `.weight` now follow the module's dtype (always fp32
  before).
- `quantize_model` on a model with tied embeddings: the head shares the embedding's codes
  (`QTiedHead`), `modules_quantized` no longer counts the head (SmolLM2: 211, was 212),
  and `bytes_after` drops by one table. The report gains `tied_head` and `skipped`.
- `quantize_model` keeps `nn.Embedding` with `max_norm` in float (with a warning) and
  keeps the class and forward of `nn.Linear`/`nn.Embedding` subclasses instead of
  replacing them with plain quantized modules.
- `weightpack.load` returns tensors in their saved dtype (was fp32 for every float);
  `load(path, dtype=torch.float32)` restores the old behaviour.
- `weightpack.save` at `bits=8` or `4` stores tensors it does not quantize exactly (was
  bf16), and `bits=16` keeps fp16 tensors exact (they were rounded to bf16). int8 and
  int4 blocks run over the flattened tensor with absmax/127 scales: identical blocks when
  the last dimension is a multiple of 256 (64 for int4), smaller files otherwise. New
  files are format v2; v1 files still load.
- `weightpack.save` keeps a qweights-quantized state_dict's scales (`<module>.s` next to
  integer codes `<module>.q`) exactly at every `bits`; `bits=8`/`4` used to re-quantize
  those with >= 4096 elements. New helper `weightpack.qweights_scales`.
- `lora.load_peft_adapter` leaves the loaded adapters trainable (`freeze_rest` freezes
  the base).

**Runtime (`bantamtrain.device`)**

- `find_batch_size` returns None when the start size does not fit (it returned the start
  size). `DutyCycleLimiter(target <= 0)` raises ValueError. `Runtime.step` returns the
  gradient norm and, with a GradScaler, skips the whole step on overflow.

**Experiment scripts (`experiments/`, repository only)**

- `summarize.py`: compute-to-target uses the final losses of the run's own baseline
  recipe, averaged over seeds, with log-FLOP interpolation (was eval-curve interpolation
  over every `1:1` run). Same-seed replicates are averaged first, so n counts distinct
  seeds (the Muon baseline group goes from n=4 to n=3). Only finished runs are tabulated;
  diverged runs get their own section and runs marked `needs_rerun` are starred. Logs
  written by the 0.3 CLIs group with older logs. It also writes `summary.json`. Weather
  tables gain `valsplit_mse` and ACC columns.
- `plots.py`: `llm_main.png` labels the final loss (2,048 windows), not the last periodic
  eval (256 windows).
- `plot_published.py`: Bantam's curve is the running-minimum frontier, and a match read
  off one point prints as `>= Nx`.
- `compare_published.py`: the default story set is the clean subset
  (`experiments/clean_val_indices.json`; `--all_val` gives the old set); the validation
  file is downloaded at a pinned revision (it was a local path); roneneldan models are
  scored with `<|endoftext|>\n` before and `\n<|endoftext|>` after each story, the prefix
  unscored (was `[eos] story [eos]`, all scored); a results file scored under another
  convention, including `--hf repo#format` overrides, is refused unless `--overwrite`.
- `bench_1b.py --chinchilla_tokens` defaults to 0, meaning 20 x the parameter count (was
  2e10).
- `run_queue.py`: logs go to `results/logs/<llm|weather>/<name>.log`. Unfinished runs are
  re-run, resuming from `<out>/<name>.ckpt` when it exists (weather runs too); diverged
  runs are skipped; a relative `--out` is resolved against the repository root;
  `bantam pretrain` lines count as language-model runs; any failure gives exit status 1.
  New `--dry_run`.
- CI installs `bantam[test,finetune,data,weather]`. The root `.gitignore` ignores
  `build/`, `dist/` and `*.egg-info/`.

### Added

**Commands and models**

- `bantam generate`: sample text from a ZoomGPT saved with `save_pretrained`, with a KV
  cache (greedy tokens match a no-cache reference), top-k, top-p, repetition penalty, an
  end-of-document stop and several samples.
- `bantam chat`: talk to a Hugging Face model plus a Bantam adapter, an `adapter.pt` or a
  PEFT adapter directory; streamed output, an interactive mode with `/reset` and
  `/exit`, and `--quant`. The prompt is rendered exactly as in training.
- `bantam doctor [--json]`: an environment report for bug reports (versions, CUDA, GPUs
  and memory, bf16/TF32 support, flush-to-zero, threads, RAM, disk, extras).
- `ZoomGPT.config()`, `save_pretrained()` and `from_pretrained()`: self-describing
  checkpoints (`config.json` plus `model.safetensors`, or `model.pt` without
  safetensors), with the tokenizer copied alongside and the Zoom-only layers stripped.
- `ZoomGPT.forward(idx)` returns logits, and `generate()` decodes with a KV cache and
  accepts prompts of different lengths.
- Exact FLOP counts: `flops_per_step(..., exact=True)` matches PyTorch's FLOP counter in
  every configuration tested; logged as `model_flops_exact`.
- Optimizer FLOPs: `optim.optimizer_flops()` and `optim.ns_flops()`, logged as
  `opt_flops` by `pretrain` and `weather`; `--flops_mode total` charges them to the
  budget.

**Fine-tuning**

- `--dora 1` (DoRA, Liu et al. 2024), `--lora_plus RATIO` (LoRA+, Hayou et al. 2024),
  `--lora_targets all-linear` (the library's `apply_lora` also takes a regex and
  `exclude`), and LoRA, QLoRA and Muon on transformers `Conv1D` layers (GPT-2 family).
- PEFT export (`--save_peft`, `lora.save_peft_adapter`) and import
  (`lora.load_peft_adapter`, `bantam chat --adapter <peft dir>`). In the tests,
  `PeftModel.from_pretrained` reproduces Bantam's logits to 1e-5, for LoRA and DoRA, on
  Llama and GPT-2 shapes.
- `--low_ram 1`: `qweights.load_quantized` builds the model on the meta device and
  quantizes the safetensors weights tensor by tensor. Its logits and sizes equal
  `from_pretrained` + `quantize_model`. On a 42M-parameter test model (168 MB in fp32),
  peak RSS grew about 40 MB.
- QLoRA memory path: quantized linear layers save only their codes for backward
  (custom autograd, dequantized per row chunk in the compute dtype), and quantized
  embeddings gather rows before dequantizing. In a test case, memory saved for backward
  fell from 35.9 MB to 6.8 MB, the same as plain LoRA.
- `--save_every N`, `--resume DIR`, `--stop_step N`: exact resume (train losses, evals and
  adapters bit-identical to an uninterrupted run, also with LoRA dropout and bf16 full
  fine-tuning).
- `--eval_jsonl FILE` (a separate held-out set), ShareGPT `conversations`, role aliases,
  and counts of skipped rows by reason in `log.json`.
- `losses.chunked_cross_entropy` gains `scale`, `softcap`, `bias`, `ignore_index`,
  `token_weights`, and quantized head modules (dequantized per chunk).
- `--examples_per_step N`: after packing, `--accum` is set so that an optimizer step sees
  about N examples (`max(1, round(N / (batch x processes x mean examples per window)))`);
  the chosen value is printed and logged in `args.accum`. Also `--drop_last` and `--ftz`
  (above), `finetune.window_layout` / `packed_attention`, `steps_per_epoch` and
  `step_windows`, and `qweights.load_quantized(..., fallback=True)`.

**Pretraining, weather and the runtime**

- `pretrain --auto_batch` (GPU only): the largest micro-batch that fits, keeping
  `batch x accum`, the schedule and the data order.
- `pretrain --data_format`, `--flops_mode`, `--allow_arg_change`; DDP that
  shards the data, evaluates on rank 0 only and stops every rank together on a
  non-finite loss.
- `weather --val_year`; latitude-weighted ACC against a train-years climatology plus a
  climatology RMSE baseline; `--save`, `--ckpt_every`, `--ckpt`, `--resume`,
  `--stop_step` (bit-exact on CPU); `--forecast --load --init --steps --forecast_out`;
  `--ic_stride` and `--max_ics` with coverage logged; `--flops_mode`; `compile_time`.
- Resume now exists in all three trainers (pretrain had it; weather and finetune are
  new).
- `prepare-era5 --test_year` and `--val_year`.
- Optimizers: per-group `lr_scale`, `model.muon_splits()` (kept across deepcopy and
  pickling), `CombinedOptimizer` as a real `torch.optim.Optimizer` (LR schedulers,
  GradScaler, hooks), and an unbiased 8-bit second moment (`v_rounding`).

**Data**

- Out-of-core text pipeline: `prepare-text` streams parquet record batches (peak memory
  grew 5 MB for a corpus 6x larger in the test, against 192 MB in 0.2); a streaming
  TokPack writer; `dedup_stream` holds about 14 bytes per document and reproduced the
  TinyStories pack in 28 s at 104 MB RSS (about 5 GB in 0.2).
- `prepare-text --tokenizer NAME_OR_PATH` (reuse a Hugging Face tokenizer) with `--eos`,
  `--config` for multi-config repos, and uint32 token ids for vocabularies above 65,536.
- A document-boundary index in TokPack (`doc_starts`, `n_docs`, `doc(i)`), the prep-id
  manifest with `StalePackError`, `bantamtrain.data.open_tokens`, and thread-safe,
  picklable TokPack and GridPack readers.

**Planning, tooling and packaging**

- `bantam memory`: `--headroom`, `--overhead_gb`, `--tied`, `--ce_chunk`,
  `--bits_per_token`, `--lora_dropout`, `--global_batch_tokens`, `--ns_steps`.
- `experiments/race_finetune.py`: a fine-tuning race of Hugging Face Trainer + PEFT
  against `bantam finetune` on SmolLM2-135M and Dolly (protocol in
  [docs/GUIDE.md](docs/GUIDE.md)). Its results are not in this file. Its replay of
  bantam's data plan follows an arm's `--drop_last` flag (`--drop_last 1` reproduces the
  published runs).
- `compare_published.py` reports the end-of-story cost apart (`nll_end_nats`,
  `bpb_no_end`) and scores a verified local copy of a Hub model (`--hf repo=dir`).
  `run_queue.py --dry_run`.
- GitHub Actions CI: CPU tests on Python 3.10 and 3.13, a build with `twine check
  --strict`, and a core-only install that runs `--help` for seven of the commands
  (pretrain, finetune, weather, pack-text, pack-era5, memory, doctor) and `doctor`.
  Pytest markers `slow`, `network`, `gpu` and `dist`.
- `packaging/release.sh` rebuilt for the new layout.

### Fixed

**Fine-tuning**

- Packed SFT examples attended to each other: a packed window ran as one causal
  sequence, so each example saw the ones before it (logits off by about 20 on SmolLM2).
  Windows now get a block-diagonal causal mask with positions restarting per example
  (commit 60759d3, before the audit). Models on eager or flex attention, such as
  Gemma-2, ignored that mask and trained with no causal mask at all; they now get an
  additive float mask (3474536).
- The chunked loss skipped the model's final logit transform (Gemma-2/3 soft-capping,
  Cohere `logit_scale`, Granite `logits_scaling`, Falcon-H1's multiplier), so training
  and eval used the wrong loss for those models (FT-1, GPU-3, QNT-3).
- `--optim muon` with LoRA put no parameters on Muon, so those runs were 8-bit AdamW
  (FT-14, OPT-X1, CLM-4).
- Gradient accumulation and DDP averaged per-micro-batch means, so `--batch 2 --accum 2`
  was not `--batch 4` (FT-17).
- Plain-format chat examples never supervised the first response token (FT-2). The
  fallback boundary mislabelled the prompt, and a conversation ending in a user turn
  trained on that turn (FT-3). The plain format never added BOS for tokenizers that need
  it (FT-16).
- The default LoRA targets missed fused or differently named projections (Phi-3, Phi-2,
  OPT, GPT-J) (FT-15, LORA-2), and GPT-2's `Conv1D` layers could not be adapted or
  quantized (FT-5, GPU-13).
- DDP: unequal shards gave ranks different step counts, so the job hung on the last step
  and saved nothing (FT-4, GPU-1). Per-rank seeds made bf16 replicas drift apart (FT-7,
  GPU-2, OPT-1/optim). `--eval_only` returned on rank 0 while the other ranks trained
  (FT-8, GPU-11). Every quantized buffer was broadcast on every forward and gradients
  were all-reduced on every micro-batch (OPT-1/ft, OPT-2/gpu, OPT-4/gpu, DDP-1). Under
  DDP, and with `--prompt_loss_weight` > 0, the chunked loss was bypassed and full fp32
  logits were built, with a different loss than evaluation used (FT-11, GPU-6,
  FT-DDP-CE).
- The eval counted prompt tokens with `--prompt_loss_weight` > 0 (FT-9). `--eval_every 0`
  reported the pre-training loss as the final loss (FT-10, OPT-1/gpu). A micro-batch
  with no target gave a NaN loss (OPT-2/ft).
- `--lora 0` with `--quant` crashed, or trained about 10% of the weights and saved a model
  that reloaded re-initialized (FT-6). A quantized base with bf16 parameters and no
  autocast crashed (FT-12).
- `--save_merged 1` was silently skipped with `--quant` (FT-23, CLM-13). A merged model
  with installed ChatML kept `<|endoftext|>` as EOS, so generation never stopped, and the
  adapter-only output lost the chat template (FT-13).
- Empty or unsupported datasets ended in an opaque IndexError after logging an eval of 0
  (FT-21). Truncation silently dropped examples and `--pack wrapped` cut long examples
  (FT-22). `padding_fraction` counted real EOS tokens as padding (FT-24).
- Best-fit-decreasing packing was O(items x bins); it is now O(n log n) with identical
  bins (FT-26).
- The QLoRA loss dequantized the whole output head to fp32 on every micro-batch and kept
  it until backward (OPT-4/ft, MEM-X1); it is now dequantized per chunk. Each loss call
  also dequantized a quantized head just to check for a `.weight` (113 MB per call on
  SmolLM2) (3474536).
- `--lora_targets "q_proj, v_proj"` silently adapted only `q_proj` (8aa2749). `--dataset` with
  `--max_examples` loaded every row into Python first; the same rows are now selected
  without that (2d4e6f6).
- The only data-pipeline test passed without checking anything when offline (OPT-3/ft,
  PKG-6).
- With packing (or padding), sliding-window layers attended to the whole example: one
  block-diagonal mask went to every layer (Mistral, Phi-3, Starcoder2 at `--seq` above
  their window; Gemma-2/3 and Qwen2/3 local layers). Logits of a packed example were
  off by up to about 4 on tiny test models.
- "1 epoch" skipped the windows that did not fill a last optimizer step.
- Importing `bantamtrain.finetune` turned on flush-to-zero for the whole process.
- `--low_ram 1` crashed on checkpoints with only `pytorch_model.bin` weights.
- `bantam chat --quant` always used int8 embeddings, whatever the run's `--embed_quant`.
- A missing `datasets`, data source or model left an empty `--out` directory.

**Quantization, LoRA and WeightPack**

- `quantize_model` replaced `nn.Linear`/`nn.Embedding` subclasses and lost their
  `forward`: Gemma's embedding scale disappeared without an error, and OPT crashed
  (QNT-1).
- Quantized linear layers saved their dequantized weight for backward, so QLoRA used as
  much memory as LoRA or more (QNT-2, GPU-7). Quantized embeddings dequantized the whole
  table in fp32 on every call (QNT-5).
- NF4 copied its codebook to the GPU on every forward and built int64 index tensors
  (GPU-9); quantizing to NF4 briefly needed 128 bytes per weight (QNT-7).
- int4/NF4 packing failed for odd row widths (QNT-4).
- Tied input and output embeddings were quantized into two copies (QNT-6, OPT-3/gpu).
- Scale buffers followed `model.to(dtype)`, and int8's 128 block padded common widths
  (QNT-8).
- LoRA adapters were created on the default device, not the base layer's, so LoRA on a
  model already on a GPU failed (LORA-1, GPU-14).
- ZoomGPT and the Muon parameter routing dequantized a quantized embedding or head in
  full (wave-2 review).
- WeightPack: non-float tensors loaded as int64, so bool/int tensors and quantized
  state_dicts crashed (WP-1); `bits=32` crashed on bf16 and `bits=16` rounded fp16 to
  bf16 (WP-2); small tensors were rounded to bf16 at `bits=8`/`4` (WP-3); files had no
  integrity check or byte-order marker (WP-4); fp16 scales overflowed above 65504 and
  lost precision for small blocks (WP-5); narrow last axes were padded (an int8 file
  larger than bf16) and tied tensors stored twice (WP-6). `bits=8`/`4` re-quantized the
  scale matrices of a qweights-quantized state_dict.
- `experiments/weightpack_eval.py` rebuilt the wrong architecture (EXP-1/quant).

**Optimizers and low-memory state**

- Resuming with bf16 parameters and 32-bit state crashed: the fp32 state was cast to bf16
  (OPT-2/optim, GPU-4, PRE-2).
- 8-bit state was not moved to the parameter's device on load and needed unsafe pickle
  loading, so a GPU resume with `--state_bits 8` failed (OPT-3/optim, GPU-5, PRE-X3).
- bf16 stochastic rounding used the global RNG (OPT-1/optim).
- The 8-bit log map padded blocks with zeros, so the tail block lost about 25x precision
  (OPT-4/optim). The 8-bit second moment was biased upward by 9 to 15% for beta2 >= 0.99
  (OPT-9).
- The 8-bit step built fp32 copies of each whole parameter (29 to 36 bytes per parameter
  of transient memory) (GPU-22).
- Weight decay by name: HF `gate_proj` was excluded (OPT-5), and linear merge maps were
  decayed (OPT-6, WX-12, ZG-8).
- `--optim muon` silently used AdamW for `Conv1D` projections (OPT-7).
- The Polar Express coefficient list was truncated to 5 tuples (OPT-8).
- Resuming with a changed `--state_bits` gave an opaque KeyError, and other optimizer
  flags were silently taken from the checkpoint (OPT-11).
- Muon's split of the fused qkv matrix was lost on `copy.deepcopy` (ZG-12).
- Optimizer tests that could not fail now check the polar factor and the per-block
  update (OPT-10, PKG-7, PKG-X2).

**Pretraining and the ZoomGPT model**

- bf16-weight models computed the loss in bf16 (GPU-17, ZG-3) and were evaluated in bf16
  arithmetic, with rounding errors of up to about 0.01 nats, of either sign (CLM-18,
  PRE-9).
- `generate()` crashed for a vocabulary under 40, batched prompts with EOS, temperature 0
  and `top_k` 0 or None, and its context was fixed at 256 (ZG-4).
- The RoPE cache stored inference tensors (breaking later training) and grew without
  bound (ZG-6); `torch.compile` built two graphs per zoom level (GPU-18).
- Merge-mode typos silently selected the mean ablation (ZG-7); schedules were validated
  only by assert (ZG-9); zoom levels were not checked against `--seq` and the hard-coded
  maximum of 4, so bad ones crashed after spending budget (ZG-5, PRE-15).
- The budget left out the linear merge maps (PRE-13) and the optimizer's FLOPs, which are
  now logged and optionally charged (ZG-1).
- A diverged run was recorded as completed, with a saved model and exit 0 (PRE-3). Under
  DDP, a non-finite loss on one rank hung every rank (PRE-5).
- A small validation split crashed the final eval after the whole budget was spent
  (PRE-4).
- DDP did not shard the data (GPU-12, PRE-6), and non-main ranks also evaluated and wrote
  the run JSON (PRE-7).
- The periodic and final evals used different subsets with no way to compare them; the
  final model is now also scored on the periodic subset (PRE-8).
- `--resume` silently accepted changed arguments and the final JSON kept the old ones
  (PRE-10); it now also refuses a changed number of processes (f5756d5).
- Checkpoints saved only the CPU RNG (GPU-15, PRE-11) and no GradScaler state (PRE-X2),
  and loaded with unsafe unpickling (GPU-21, PRE-12).
- Batches larger than the window count came back short while full FLOPs were charged,
  and every epoch replayed the same order (PRE-14).
- `--save_every_eval` overwrote one file when `--save` had no `.pt` (PRE-17).
- A `train.tokpack` silently switched the training source, unrecorded (PRE-20).
- Bad model options and a missing `--resume` file crashed after creating `--out`
  (f5756d5).

**GPU runtime** (code paths tested on CPU only)

- `--precision auto` never chose fp16 with a GradScaler on pre-Ampere GPUs (GPU-8).
- The GradScaler path crashed on an empty sub-optimizer (GPU-10, OPT-X2), and an overflow
  in one sub-optimizer let the others step with NaN or zeroed gradients (GPU-19, PRE-X1).
- `find_batch_size` returned the start size even when it did not fit, and no command
  used it; it now backs `pretrain --auto_batch` (GPU-20, PRE-18, GAP-4/gpu, CLM-19). The
  prefetcher swallowed errors and `DutyCycleLimiter(0)` was accepted (PRE-18).
- A `--precision` typo silently trained in fp32 (f5756d5).

**Weather**

- There was no validation split: the 2018 test year also chose the LR and the
  ablations; `--val_year` adds one (WX-3, CLM-8).
- A zoom level that is not a perfect square trained at the wrong resolution in
  `lowres` mode and crashed in merge mode (WX-1).
- Muon's Newton-Schulz compute was never counted, while the Zoom arm takes 3.75x more
  optimizer steps (WX-2); it is now logged and optionally charged.
- A stale `fields.gridpack` was used silently in place of new data (WX-5).
- `--batch_scale_k` was ignored in `lowres` mode (WX-6).
- Linear merge FLOPs were not charged (WX-7), and `lowres` steps were over-charged
  (WX-11).
- Rollout initial conditions were fixed and unreported (only 1 January to 13 October,
  00 UTC); `--ic_stride` and `--max_ics` choose them and the log records the coverage
  (WX-10).
- `--evals 0` crashed (WX-13); training never reshuffled between epochs (WX-14); compile
  time was counted as training time and the `lowres` stage ran uncompiled (WX-15).

**Data**

- `prepare-text` left verbatim validation documents in `train.bin` (1,550 of the 21,990
  TinyStories validation stories); only `pack-text` removed them (DATA-1, EXP-9).
- TokPack and GridPack readers were not thread-safe (segfaults, corrupted reads) and
  could not be pickled (DATA-2).
- TokPack returned writable views into its cache (DATA-3), `stream()` dropped windows
  (DATA-4), and slicing crashed where numpy returns an empty array (DATA-10).
- `prepare-era5` hard-coded the 2018 split, giving NaN statistics or an empty test set
  for other ranges (DATA-5, WX-4).
- Validity checks were asserts, so under `python -O` int16 overflow wrapped and NaN wiped
  out channels (DATA-6). GridPack's error bound was broken by float32 rounding (DATA-12),
  `GridPack.read` had no bounds checks (DATA-11), and its docstring described another
  layout (DATA-X2).
- A vocabulary above 65,536 crashed or silently wrapped token ids (DATA-7).
- `--hf` picked the test split as validation, mixed dataset configs and failed on repos
  without `train`-named shards (DATA-8).
- Deduplication crashed on small corpora (DATA-9). A literal `<|eos|>` inside a document
  split it (DATA-13).
- The pipeline held the whole corpus in RAM, about 55 bytes per token in `pack-text`
  (DATA-15).
- Re-preparing a directory left stale packs that the trainers preferred over the new data
  (DATA-X1).
- `prepare-text` created `--out` before checking its inputs (9b7d06a).

**Memory estimator**

- Capacity used 1e9-byte GB for a card sold in GiB (MEM-1). The activation model was too
  small (MEM-2), charged the recomputed layer once instead of per sequence (MEM-3) and
  never fell back to recompute (MEM-4). QLoRA rows ignored the dequantized weights kept
  for backward (MEM-5), and the fine-tune table ignored measured HF activations and LoRA
  dropout (MEM-6).
- The corpus-size line applied a 4k-vocabulary TinyStories rate to every corpus (MEM-9).
  Newton-Schulz FLOPs were not counted (MEM-10). The FLOP convention was not stated
  (MEM-11). The cross-entropy chunk did not match the code (MEM-12). `model_dims`
  silently substituted a different model at small sizes (MEM-15).

**Experiment tooling**

- `summarize.py` crashed on the results tree (CLM-1, SUM-2, EXP-2), and its
  compute-to-target compared final losses with eval curves and pooled the wrong baselines
  (CLM-2, EXP-3). The 1.12x figure was computed by hand; `summarize.py` now produces it
  and states its method (CLM-14). `plots.py` failed on a weather log without
  `coarse_mode` (1779304).
- `run_queue.py` treated in-progress, crashed and killed runs as finished (CLM-10, EXP-4,
  PRE-1), resolved a relative `--out` against the wrong directory and could overwrite
  results (RQ-2), substituted `$NAME` variables by plain string replacement and shared
  log files between trainers (PRE-16).
- `compare_published.py` charged the roneneldan models 19 to 37 nats per story through
  the boundary convention (EXP-1/exp), mixed conventions in one file (EXP-6), scored a
  contaminated story set by default (EXP-7), had a default validation path that no
  command created (CLM-11, EXP-11), matched checkpoint compute by a cwd-relative path
  (EXP-10), loaded checkpoints unsafely (EXP-14) and overstated how evenly llama2.c's
  text was scored (EXP-15).
- `plot_published.py` returned a later crossing when an earlier point met the target
  (EXP-12) and crashed on models missing from `published_compute.json` (EXPV-1).
- `llm_main.png` labelled each curve with the 256-window eval (EXP-8). `bench_1b.py`'s
  model has 0.92B parameters, not 1.0B, and its build time covered the whole script
  (EXP-13).

**Packaging and command line**

- `packaging/release.sh` could not build after the rename (PKG-1). The `finetune` extra
  allowed transformers versions that crash (PKG-2, FT-18); the torch floor was too low
  (PKG-3); the setuptools floor could not build the license metadata (PKG-4).
- `prepare-text --help` and `prepare-era5 --help` crashed without the extras, and
  `finetune` showed a raw traceback (PKG-5). The `test` extra installed only pytest
  (PKG-9). The CLI had no tests (PKG-10). `cli.main(argv)` overwrote `sys.argv` and
  returned None (PKG-12). The end-to-end resume test needed a machine-specific path
  (PKG-8). The version was defined twice (GAP-3/pkg).
- CI skipped the `prepare-era5` tests (no weather extra) (1902d92).

**Documentation**

- The docs were rewritten for 0.3. Commands, imports and paths use `bantamtrain` and
  `bantam` (CLM-3, EXPV-2, FT-19, MEM-7, PKG-11, DATA-14), and `pip install` lines quote
  the extras for zsh (PKG-13). The corrected claims are listed in the next section.

### Results corrected

- **Published-model comparison.** 0.2 scored the roneneldan TinyStories models with
  `[eos] story [eos]`, all scored, which cost them 19 to 37 nats per story (11.5 to 12.7
  of it at the end-of-story token; `results/published/compare_clean1000_legacy_eos.json`).
  Each family is now scored in its own training format (EXP-1/exp, EXP-6, EXP-7).
  On 1,000 of the 4,194 clean validation stories (the 0.2 docs said 4,194; CLM-11,
  EXP-11), in bits per byte: TinyStories-1M 0.873 -> 0.822, 3M 0.719 -> 0.677,
  8M 0.651 -> 0.581, 28M 0.586 -> 0.538, 33M 0.578 -> 0.542. The llama2.c rows are
  bit-identical; stories110M (0.480) was not re-scored. Bantam's 5.8M model, retrained
  with the recorded command because the 0.2 weights are gone, scores 1.029 (1.0288
  against 1.0291 in 0.2, within retrain noise).
  It still beats llama2.c's stories260K (1.097) with at least 157x less compute, a
  single-point upper bound that is partly a model-size effect (0.26M against 5.8M
  parameters) (CLM-6). It is behind every TinyStories model; the gap to TinyStories-1M
  grew from 0.156 to 0.207 bits per byte, and that model used about 3,350x more compute.
- **5B scoreboard.** The 2.56x headline mixed 1e9-byte GB with a 96 GiB card. In GiB with
  a 0.85 headroom, the standard AMP row no longer fits, and the Muon row is about 1.9x
  the first row that fits (1.87x at the `bantam memory` defaults) (MEM-1 to MEM-6,
  MEM-14). It is still arithmetic, not a measurement. The FP8 row is a literature
  estimate with no code behind it, and the chunked cross-entropy the rows assume cannot
  be turned on in `bantam pretrain` (GPU-16, MEM-8, GAP-5/gpu). The memory saving is not
  new: ZeRO-Offload trained over 13B on one 32 GB V100 in 2021, and torchao ships 8-bit
  AdamW with bf16 stochastic rounding.
- **bf16 runs.** bf16-weight runs had been scored with bf16 arithmetic. The two committed
  bf16 rows (2.728 at the full budget, 3.5752 short) have no checkpoints to re-score.
  Their measured error is about +-0.001 and +-0.004, and both are marked `needs_rerun`
  (CLM-18, ZG-3, PRE-9).
- **LoRA runs were 8-bit AdamW.** The 0.2 "Muon with 8-bit state" fine-tunes put no
  parameters on Muon (CLM-4, FT-14, OPT-X1).
- **Fine-tune quantization table.** It used 88 held-out examples, not 100, and its sizes
  included 19.5 MB of LoRA adapters inconsistently. The base-only sizes are fp32 538 MB,
  int8 181 MB, NF4 with int8 embeddings 120 MB, int4 everywhere 87 MB (CLM-16, FT-20).
  These are 0.2 sizes: they count SmolLM2's tied output head a second time and include
  int8 block padding (QNT-6, QNT-8), so 0.3 stores the same bases in less memory.
  The 0.2 headline "QLoRA 3.39 -> 3.18" was an int4-everything base that stayed worse
  than the untouched float base (2.87; 2.79 after LoRA), on 8 eval windows of about 17
  examples, one seed (PKG-X1).
- **Low-memory saving.** On the 5.8M-parameter model with Muon, weights plus optimizer
  state shrink 2.7x (50.4 MB for fp32 weights and 32-bit state, 18.5 MB for bf16 weights
  and 8-bit state), not 3x (CLM-17, OPT-12). The bf16 path also ran 2.6x slower in
  wall-clock on this CPU (718 s against 271 s for 100 steps, 1 thread, 1 run each).
- **WeightPack 16** rounds fp32 weights to bf16 (+0.00002 loss on the 5.8M model). It is
  lossless only for weights already in bf16 (DOC-1, WP-2).
- **Wall-clock.** The claim that Zoom's coarse steps run faster per FLOP on CPU was
  wrong. At a fixed batch, k=2 steps took about 1.2x and k=4 steps about 1.4x the time
  per FLOP of k=1 steps (Muon runs, 2 threads, shared CPU). At equal FLOPs, the
  seed-paired 4->2->1 Zoom runs without batch scaling took 15 to 20% more wall-clock than
  their Muon baselines (6% more with AdamW), and the weather Zoom-merge runs took 8%
  (AdamW) and 33 to 52% (Muon) more. The machine was shared, so these timings are noisy
  (CLM-12, WX-8, ZG-2).
- **Compute-to-target.** The Zoom figure (1.12x, 3 seeds; 1.11x to 1.13x per seed) is now
  generated by `summarize.py` from final losses with log-FLOP interpolation. The
  baseline points above 1x are single seeds (CLM-14, CLM-2).
- **Evaluation subsets.** Periodic evals use the first 256 validation windows and the
  final eval the first 2,048; the curve reads about 0.27 to 0.30 nats lower on the same
  weights. The LR sweeps used 512 windows and the short low-memory runs 1,024. Each table
  now says which (CLM-15, PRE-8).
- **Muon against AdamW.** In Muon runs, the AdamW part (embeddings, head, norms) runs at
  Muon's LR, 16x the best AdamW-only LR, so the comparison changes two things, and the
  AdamW side is one seed. Muon's Newton-Schulz step also adds about 20% arithmetic on
  top of the charged model FLOPs at this size; it is now logged as `opt_flops` but still
  not charged by default (MEM-10, ZG-1). The claim that both "share one learning rate"
  by design of the update scaling is withdrawn: the sweep shows they do not (CLM-5).
- **Weather.** The finished matrix is reported, including Zoom hurting under Muon. Its
  LRs were chosen on the 2018 test year (CLM-7, CLM-8, WX-3), and its rollouts covered 1
  January to 13 October at 00 UTC (WX-10).
- **Upgrade ablations.** WSD (2.773), ReLU squared (2.829) and Polar Express (2.755) did
  worse than the 2.737 baseline, and soft cap and zero-init were about equal (1 seed
  each). Four of those runs read the raw `train.bin`, which held validation copies
  (CLM-9, DATA-1).
- **Batch finder.** The docs said the trainers used `find_batch_size()`; none did. It now
  backs `pretrain --auto_batch` (CLM-19, GPU-20).
- **Stale descriptions** of TokPack v1, of the 4->2->1 schedule as the method (it lost),
  of a "1M-parameter" ViT (1.26M) and of Zoom layers vanishing at k=1 (they stay in
  training checkpoints; `save_pretrained` strips them) were corrected (CLM-20, ZG-11,
  DATA-14). Numbers without a committed source were dropped or sourced (CLM-21).
  `results/SUMMARY.md` is regenerated from the logs (WX-16). The corpus-size line
  applies only to a 4k-vocabulary corpus (MEM-9).

### Known limitations

- **Nothing has run on a GPU.** CUDA paths (precision, compile, fused AdamW, NCCL DDP,
  `--auto_batch`) were tested with CPU stand-ins only; 7 tests are skipped without CUDA
  or as slow. Bit-exact resume is tested on CPU only.
- **Speed and memory have not been compared with GPU tools** (Unsloth, Liger Kernel,
  torchtune, Axolotl, TRL). Bantam has no fused kernels and no bitsandbytes or torchao
  backend (GAP-2/gpu), no FP8 path (GAP-5/gpu), and no FSDP or ZeRO sharding
  (GAP-1/gpu); multi-GPU is DDP only, and `bantam weather` has no DDP.
- `bantam pretrain` has no `--ce_chunk` flag and always builds the full logits.
- Fine-tuning with packing: `flash_attention_*` implementations cannot take the 4-D mask
  (the CLI never selects them), and the dense mask loses FlashAttention's speed. A
  decoder that mixes sliding-window and full layers but takes only one mask gets a note
  and no window in packed windows (none of the tested families).
- Importing the pretrain and weather command modules turns on flush-to-zero for the
  whole process.
- `bantam generate` encodes the prompt with the tokenizer's special tokens, so a reused
  tokenizer that adds BOS (Llama style) gives a prompt that differs from the training
  documents, which have no BOS.
- `torch.save` of a whole quantized model fails for quantized subclasses; state_dicts
  work.
- `--save_every_eval` without `--save` is silently ignored; `--auto_batch` does not probe
  the eval batch; optimizer FLOPs charge parameters with no gradient in the current zoom
  stage (under 0.1%).
- `from_pretrained` prefers `model.safetensors` when a directory also holds a newer
  `model.pt`.
- Data: no DataLoader dataset with worker sharding (GAP-3/data), no multi-shard packs or
  `--data` globs, no BOS-aligned windows (the document index exists but pretraining does
  not use it), and no data mixing or benchmark decontamination (GAP-5/data).
- **Results not yet redone with 0.3:** the committed TinyStories runs used 0.2-prepared
  data; the bf16 low-memory rows need a rerun, and the low-memory set predates the 0.3
  8-bit and rounding changes; the weather sweeps have not been rerun on a validation
  year; there is no linear-merge run; the long published-comparison run was stopped at
  30%; llama2.c stories110M was not re-scored. The fine-tuning race against Hugging Face
  Trainer + PEFT was still running when this file was written.

## [0.2.0] - 2026-10-25

First release with fine-tuning (built from the `zoomtrain` sources).

### Added

- Fine-tuning of any Hugging Face causal LM (`python -m bantamtrain.finetune`): LoRA,
  QLoRA on int8/int4 bases, and full fine-tuning; chunked cross-entropy; DDP with
  torchrun.
- SFT data handling: assistant-token masks, ChatML for base checkpoints, best-fit
  packing, `--prompt_loss_weight`.
- NF4 (the QLoRA codebook) weight quantization and `--eval_only`. QLoRA keeps
  embeddings and the output head at int8 by default (`--embed_quant`).
- The `finetune` extra (transformers, datasets, sentencepiece).
- Results: SmolLM2-135M on Dolly, and a quantization quality table.

Several of these had bugs that 0.3 fixes; see the 0.3.0 entry (packed examples
attending to each other, `--optim muon` with LoRA training nothing with Muon, the
accumulation loss, and others).

## [0.1.0] - 2026-10-25

First PyPI release of `bantamtrain` (built from the `zoomtrain` sources by a rename
script).

### Added

- Zoom Training: token and grid-patch merging (gate, mean or linear), `ZoomSchedule`,
  k-scaled batches, per-stage LR restart and mixed fine steps.
- Models: ZoomGPT (RoPE, tied head, QK-norm, ReLU squared, soft cap, zero-init) and
  ZoomViT (a 6 h ERA5 forecaster).
- Muon (quintic and Polar Express orthogonalizers) with AdamW for the rest, and a WSD
  schedule.
- Low-memory training: 8-bit blockwise optimizer state, bf16 weights with stochastic
  rounding, activation checkpointing, gradient accumulation, resumable checkpoints.
- int8/int4 weight-only layers (`qweights`) and WeightPack checkpoint files.
- Data stores: TokPack v2 (rank remap, LEB128, zstd with a dictionary), MinHash-LSH
  deduplication, GridPack.
- Runtime: flush-to-zero on CPU, AMP, TF32, `torch.compile`, a duty-cycle limiter.
- The single-GPU memory and time estimator behind the 5B scoreboard.
- Experiment tooling: queues, `run_queue.py`, `summarize.py`, a published-model
  comparison in bits per byte, and a CPU benchmark of a 0.92B-parameter model (labelled
  1B at the time).

Credit: Zoom Training is a learned-gate variant of patch-level training (Shao et al.,
ICLR 2025) and is related to Hourglass, Funnel-Transformer and sequence-length warm-up.
Muon (Jordan et al. 2024), Polar Express (Amsel et al.), QK-norm (Henry et al. 2020),
8-bit optimizer state (Dettmers et al.), bf16 stochastic rounding, LoRA/QLoRA, and (in
0.3) DoRA, LoRA+ and rsLoRA are other people's methods that Bantam implements.
