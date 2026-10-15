# zoomtrain

Train small language models and small weather models **faster and cheaper on
very limited compute, without throwing away data**. Everything here was built
and measured in a 4-core CPU sandbox (no GPU), on real HuggingFace and ERA5
data; the same code paths run on a GPU with mixed precision, `torch.compile`
and fused optimizers.

The core idea, **Zoom Training**, is a coarse-to-fine *processing-resolution*
schedule: early in training the transformer body runs on groups of `k`
neighbouring tokens (or `√k×√k` blocks of grid patches) merged into one
position by a learned slot-gated merge, predicts every original element
through a slot-gated unmerge, and `k` is annealed `4 → 2 → 1` over the compute
budget. Every token / grid cell is seen and supervised in every stage; only
the body's internal resolution changes, so the deployable model at `k = 1` is
an ordinary transformer. See [`docs/METHOD.md`](docs/METHOD.md).

Around it: the **Muon** optimizer, a **CPU/GPU runtime** (including a fix that
made CPU training 2× faster), and three **storage** components — a
deduplicating, random-access, compressed token store; and an error-bounded
compressed store for gridded weather fields.

Results are in [`RESULTS.md`](RESULTS.md); the literature review that guided
the design is in [`RESEARCH.md`](RESEARCH.md).

## Layout

```
zoomtrain/
  zoomtrain/                 the package
    zoom.py                  SlotMerge / SlotUnmerge / ZoomSchedule / LR schedule
    optim.py                 Muon (+ AdamW for the rest), CombinedOptimizer
    device.py                runtime: FTZ, AMP, TF32, compile, batch finder, duty limiter
    models/gpt.py            ZoomGPT  (decoder-only LM, RoPE, tied head, QK-norm option)
    models/vit_weather.py    ZoomViT  (ViT on lat/lon grid, residual 6 h forecaster)
    data/tokpack.py          TokPack: compressed random-access token store
    data/dedup.py            exact + MinHash-LSH near-duplicate removal on token streams
    data/gridpack.py         GridPack: error-bounded compressed gridded fields
  llm/                       prepare_data.py (HF -> tokens), pack_data.py, train.py
  weather/                   prepare_era5.py (WeatherBench2 -> npy), pack_data.py, train.py
  experiments/               run_queue.py, queues/*.txt, summarize.py, plots.py
  tests/                     pytest suite (17 tests, CPU, ~5 s)
  results/                   JSON logs of every run + SUMMARY.md + plots
  docs/METHOD.md             the method, FLOP accounting, relation to prior work
```

## Install

```bash
pip install torch numpy zstandard tokenizers pyarrow          # core
pip install xarray zarr fsspec aiohttp                        # weather data download
pip install pytest matplotlib                                 # tests, plots
python -m pytest tests -q
```

## LLM quick start (any HuggingFace text dataset)

```bash
# 1. tokens: export the dataset's parquet shards (shown: roneneldan/TinyStories)
python llm/prepare_data.py --train train-00000-of-00004.parquet --val validation.parquet \
    --out data/tinystories --vocab 4096
# 2. dedup + pack (optional; train.py reads .tokpack if present, else .bin)
python llm/pack_data.py --data data/tinystories
# 3. baseline and Zoom at the same FLOP budget
python llm/train.py --data data/tinystories --budget 6.8e13 --zoom 1 \
    --optim muon --lr 2e-3 --name base
python llm/train.py --data data/tinystories --budget 6.8e13 --zoom 4:0.35,2:0.25,1:0.4 \
    --optim muon --lr 2e-3 --name zoom
```

`--zoom` is `k:fraction,...` over the FLOP budget. Useful flags:
`--optim {adamw,muon}`, `--out_mode {slot,bag}` (bag = Shao et al. style loss),
`--gated 0` (plain mean merge), `--rope_pos {token,patch}`, `--qk_norm 1`,
`--device cuda --precision bf16 --compile`, `--duty 0.7` (cap average load).

## Weather quick start (ERA5 5.625°, WeatherBench2)

```bash
python weather/prepare_era5.py --out data/era5_64x32           # 2010-2018, 6 channels, ~650 MB
python weather/pack_data.py --data data/era5_64x32              # -> fields.gridpack (~5x smaller)
python weather/train.py --data data/era5_64x32 --budget 2e13 --zoom 1 --name base
python weather/train.py --data data/era5_64x32 --budget 2e13 --zoom 4:0.6,1:0.4 --name zoom
python weather/train.py --data data/era5_64x32 --budget 2e13 --zoom 4:0.6,1:0.4 \
    --coarse_mode lowres --name progressive_resizing   # prior-art baseline
```

## Running on a GPU

`llm/train.py` and `weather/train.py` share `zoomtrain/device.py`:

* `--device cuda` (default `auto`) turns on TF32, bf16 autocast (fp16 +
  `GradScaler` on pre-Ampere cards), fused AdamW, pinned non-blocking H2D
  copies; `--compile` adds `torch.compile`. Attention goes through
  `scaled_dot_product_attention`, so FlashAttention is used where available.
* `find_batch_size()` doubles the batch until peak memory would exceed 85 % of
  the card — staying under the OOM edge avoids allocator thrash.
* "Maximize compute without stressing the card": prefer a hardware power cap
  (`sudo nvidia-smi -pl <~70 % of TDP>`; GPUs keep most of their throughput
  there because clocks/voltage scale super-linearly with power). When you
  cannot set one, `--duty 0.7` idles the loop so the *average* load is 70 %.
* Zoom levels change the sequence length seen by the body; with `--compile`
  each `k` is compiled once (three graphs for `4,2,1`).

## Reproducing the experiments

```bash
python experiments/run_queue.py experiments/queues/llm_sweep.txt --workers 1 --threads 4
python experiments/run_queue.py experiments/queues/llm_main.txt  --workers 1 --threads 4
python experiments/run_queue.py experiments/queues/weather.txt   --workers 1 --threads 4
python experiments/summarize.py --plots      # -> results/SUMMARY.md, results/plots/
```

Every run writes `results/<group>/<name>.json` with its arguments, the full
validation curve (loss vs FLOPs, tokens, steps, wall-clock) and the final
metrics; `summarize.py` builds the tables in `RESULTS.md` from those files.
