# zoomtrain (to be renamed *Bantam*)

Train small language models and small weather models **faster and cheaper on
very limited compute, without throwing away data**. Everything here was built
and measured in a 4-core CPU sandbox (no GPU), on real HuggingFace and ERA5
data; the same code paths run on a GPU with mixed precision, `torch.compile`
and fused optimizers.

What it contains, and what each part has been shown to do so far:

| part | what it is | measured (5.8M-param LM on TinyStories unless noted) |
|---|---|---|
| **Zoom Training** (`zoom.py`) | the new idea: merge neighbouring tokens / grid patches into one position early in training, anneal back to full resolution | k=2 for the first half with batch ×2: loss **2.685 ± 0.004 vs 2.737 ± 0.011** at equal FLOPs (3 seeds). The original 4→2→1 anneal lost; see [`RESULTS.md`](RESULTS.md) |
| **Muon** (`optim.py`) | Jordan 2024 optimizer, our implementation; Polar Express variant | **2.74 vs 3.33** (AdamW) at equal FLOPs; reaches AdamW's final loss at ~1/3 of the compute |
| **Low-memory training** (`lowmem.py`, `optim.py`) | 8-bit blockwise optimizer state; bf16 weights with stochastic rounding (no fp32 master) | weights+state **3× smaller**, loss 2.728 vs 2.737 at full budget |
| **Weight storage** (`qweights.py`, `weightpack.py`) | int8 / int4 weight-only layers in RAM; int4/int8/bf16-lossless checkpoint files | int8 **3.9× smaller, +0.0002 loss**; int4 7.5×, +0.026; lossless bf16 file 2.9× |
| **Data storage** (`data/`) | TokPack v2 token store, MinHash dedup, GridPack weather store | tokens **2.68× smaller** than uint16 (5.97 bits/token, exact); ERA5 **5.2× smaller** with error ≤ 5 % of a 6 h change |
| **Runtime** (`device.py`) | FTZ, AMP, TF32, compile, batch finder, duty-cycle limiter, resumable checkpoints | the FTZ fix made affected CPU matmuls up to 23× faster; stop/resume is bit-exact (tested) |
| **Scoreboard** (`memory.py`, [`docs/SCOREBOARD_5B.md`](docs/SCOREBOARD_5B.md)) | memory/time estimator for one GPU, sourced assumptions | 5B on one RTX PRO 6000: **234 → 91 days**, 92 → 26 GB (estimate) |
| **Published comparison** (`experiments/compare_published.py`) | scores any checkpoint on stories no model saw, in bits/byte | one short CPU run beats llama2.c `stories260K` with 157× less compute; behind TinyStories-1M+ |

[`RESEARCH.md`](RESEARCH.md) is the fact-checked literature review (~160 sources)
that guided the design; [`docs/METHOD.md`](docs/METHOD.md) explains Zoom
Training and the FLOP accounting.

## Layout

```
zoomtrain/
  zoomtrain/                 the package
    zoom.py                  SlotMerge / SlotUnmerge (gate | mean | linear), ZoomSchedule, LR schedules
    optim.py                 Muon (quintic / Polar Express NS), AdamWLowMem, build_optimizer
    lowmem.py                8-bit blockwise state (QState), stochastic bf16 rounding
    device.py                runtime: FTZ, AMP, TF32, compile, batch finder, duty limiter, prefetch
    memory.py                single-GPU memory/time scoreboard (python -m zoomtrain.memory)
    qweights.py              int8 / int4 weight-only Linear & Embedding
    weightpack.py            compressed checkpoints: fp32 / bf16 lossless / int8 / int4
    models/gpt.py            ZoomGPT: RoPE, tied head, QK-norm, ReLU², softcap, zero-init, act. ckpt
    models/vit_weather.py    ZoomViT: ViT on a lat/lon grid, residual 6 h forecaster
    data/tokpack.py          TokPack v2: rank-remap + LEB128 + zstd(dict), random access
    data/dedup.py            exact + MinHash-LSH near-duplicate removal on token streams
    data/gridpack.py         GridPack: error-bounded int16 + temporal delta + zstd
    lora.py                  LoRA / QLoRA adapters on Linear or int8/int4 QLinear bases
    losses.py                chunked cross-entropy (never materializes tokens x vocab logits)
    finetune.py              python -m zoomtrain.finetune: SFT of any HF causal LM, torchrun DDP
  llm/                       prepare_data.py (HF -> tokens), pack_data.py, train.py
  weather/                   prepare_era5.py (WeatherBench2 -> npy), pack_data.py, train.py
  experiments/               run_queue.py, queues/*.txt, summarize.py, plots.py,
                             weights_eval.py, compare_published.py, plot_published.py, llama2c.py
  tests/                     pytest suite (50 tests, CPU, ~2.5 min incl. a stop/resume run)
  results/                   JSON log of every run, SUMMARY.md, plots/
  docs/                      METHOD.md, SCOREBOARD_5B.md
```

## Install

```bash
pip install torch numpy zstandard tokenizers pyarrow          # core
pip install xarray zarr fsspec aiohttp                        # weather data download
pip install transformers sentencepiece                        # published-model comparison
pip install pytest matplotlib                                 # tests, plots
python -m pytest tests -q
```

## LLM quick start (any HuggingFace text dataset)

```bash
python llm/prepare_data.py --hf roneneldan/TinyStories --out data/tinystories --vocab 4096
python llm/pack_data.py --data data/tinystories            # dedup + TokPack (train.py reads it if present)

# baseline vs Zoom at the same FLOP budget (the measured best recipe)
python llm/train.py --data data/tinystories --budget 6.8e13 --zoom 1 \
    --optim muon --lr 1.6e-2 --name base
python llm/train.py --data data/tinystories --budget 6.8e13 --zoom 2:0.5,1:0.5 --batch_scale 2 \
    --optim muon --lr 1.6e-2 --name zoom
```

Options worth knowing (all measured in `RESULTS.md` or queued):

| flag | effect |
|---|---|
| `--zoom k:frac,...` | zoom schedule over the FLOP budget; `--batch_scale 2` doubles the batch while zoomed |
| `--optim muon --ns polar` | Muon with Polar Express coefficients (`quintic` default) |
| `--sched wsd --decay_frac 0.45` | warmup-stable-decay LR instead of cosine |
| `--mlp relu2 --softcap 15 --zero_init_proj 1 --qk_norm 1` | speedrun architecture options |
| `--state_bits 8 --param_dtype bf16` | 8-bit optimizer state, bf16 weights with stochastic rounding |
| `--grad_ckpt 1 --accum N` | activation checkpointing, gradient accumulation |
| `--ckpt_every N` / `--resume PATH` | resumable checkpoints (model, optimizer, data position, RNG) |
| `--device cuda --precision bf16 --compile --duty 0.75` | GPU path; `--duty` caps average load |
| `--save PATH --save_every_eval 1` | keep a checkpoint at every evaluation (compute curves) |

## Weather quick start (ERA5 5.625°, WeatherBench2)

```bash
python weather/prepare_era5.py --out data/era5_64x32           # 2010-2018, 6 channels, ~650 MB
python weather/pack_data.py --data data/era5_64x32              # -> fields.gridpack (5.2x smaller)
python weather/train.py --data data/era5_64x32 --budget 5.5e13 --zoom 1 --optim muon --lr 1.6e-2 --name base
python weather/train.py --data data/era5_64x32 --budget 5.5e13 --zoom 4:0.6,1:0.4 --optim muon --lr 1.6e-2 --name zoom
python weather/train.py --data data/era5_64x32 --budget 5.5e13 --zoom 4:0.6,1:0.4 \
    --coarse_mode lowres --optim muon --lr 1.6e-2 --name progressive_resizing   # prior-art baseline
```

## Fine-tuning any HuggingFace model (LoRA / QLoRA, full)

```bash
pip install transformers datasets
python -m zoomtrain.finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --lora 16 --quant 4 --optim muon --state_bits 8 --epochs 1 --out runs/dolly
# multi-GPU:  torchrun --nproc_per_node 4 -m zoomtrain.finetune ...
# full fine-tune: --lora 0 --param_dtype bf16 --state_bits 8 --grad_ckpt 1
# size it first: python -m zoomtrain.memory --finetune --params 7e9 --gpu_gb 24
```

Data: a HuggingFace dataset or `--jsonl` with `prompt`/`response` columns,
Dolly-style `instruction`/`context`/`response`, or chat `messages`; the
tokenizer's chat template is used when it has one and the loss covers only
the response tokens. Output: `adapter.pt` (MBs) or, with `--save_merged 1`,
a merged `save_pretrained()` model. Measured here on CPU with SmolLM2-135M
(`results/finetune/`): rank-16 LoRA on an int4 base, held-out loss 3.39 -> 3.18
in 40 steps; on the float base 2.87 -> 2.79. What the frozen base's format
costs *before* training (100 held-out Dolly examples):

| base weights | held-out loss | model RAM |
|---|---|---|
| fp32 | 2.702 | 558 MB |
| int8 linears + int8 embeddings | 2.704 (+0.003) | 200 MB |
| NF4 linears + int8 embeddings (`--quant nf4`, default embed handling) | 2.878 (+0.18) | 140 MB |
| linear int4 linears + int8 embeddings | 3.035 (+0.33) | 140 MB |
| NF4 everything | 3.042 (+0.34) | 106 MB |

So for small models use `--quant 8` (free); 4-bit NF4 is for 7B+ bases where
the literature finds it near-lossless (not measurable on this machine).

## Storing weights and data

```bash
python experiments/weights_eval.py --run results/llm_lowmem/save_muon_base.json \
    --weights results/llm_lowmem/save_muon_base.pt      # bytes + loss for every RAM / file format
python -c "from zoomtrain import qweights; ..."          # quantize_model(model, bits=8|4) in place
python -c "from zoomtrain import weightpack; ..."        # save(path, state_dict, bits=32|16|8|4), load(path)
```

## Comparing against published models

```bash
python experiments/compare_published.py --stories 1000 --clean experiments/clean_val_indices.json \
    --ours results/llm_lowmem/save_muon_base.json:results/llm_lowmem/save_muon_base.pt \
    --hf roneneldan/TinyStories-8M@EleutherAI/gpt-neo-125M \
    --l2c stories15M.pt:tokenizer.model
python experiments/plot_published.py results/published/compare_clean1000.json
```

Scores are bits per byte on 4,194 validation stories that appear in neither
the TinyStories train split nor llama2.c's training shards (the official
validation set leaks into both). `experiments/published_compute.json` holds
each published model's training compute with sources.

## Running on a GPU

`llm/train.py` and `weather/train.py` share `zoomtrain/device.py`:
`--device cuda` (default `auto`) turns on TF32, bf16 autocast (fp16 +
`GradScaler` on pre-Ampere cards), fused AdamW and pinned non-blocking H2D
copies; `--compile` adds `torch.compile`; attention goes through
`scaled_dot_product_attention` (FlashAttention where available).
`find_batch_size()` stays under 85 % of memory. Prefer a hardware power cap
(`sudo nvidia-smi -pl <~75 % of TDP>`); `--duty` is the software fallback.
`experiments/gpu_smoke.sh` is the first thing to run on a real card; nothing
here has been executed on a GPU yet.

## Reproducing the experiments

```bash
python experiments/run_queue.py experiments/queues/llm_main.txt --workers 2 --threads 2
python experiments/run_queue.py experiments/queues/upgrades.txt --workers 2 --threads 1
python experiments/run_queue.py experiments/queues/weather_main.txt --workers 2 --threads 1
python experiments/summarize.py --plots      # -> results/SUMMARY.md, results/plots/
```

Every run writes `results/<group>/<name>.json` with its arguments, the full
validation curve (loss vs FLOPs, tokens, steps, wall-clock) and the final
metrics; `summarize.py` builds the tables in `RESULTS.md` from those files.
