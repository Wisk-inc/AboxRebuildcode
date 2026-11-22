# Bantam (`bantamtrain`)

**Train small language models and weather models faster and cheaper on limited
compute, without throwing away data.** Made by Nathan, Jamaica.

Bantam is the library behind a set of measured results on real data (TinyStories
from HuggingFace, ERA5 from WeatherBench2), all obtained in a 4-core CPU sandbox;
the same code runs on a GPU with mixed precision and `torch.compile`.

```bash
pip install bantamtrain
```

## What is inside

| module | what it does | measured (5.8M-param LM on TinyStories unless noted) |
|---|---|---|
| `bantamtrain.zoom` | **Zoom Training**: merge neighbouring tokens / grid patches into one position early in training, anneal back to full resolution | k=2 for the first half with batch x2: loss 2.685 +- 0.004 vs 2.737 +- 0.011 at equal FLOPs (3 seeds) |
| `bantamtrain.optim` | **Muon** optimizer (Jordan 2024; quintic or Polar Express orthogonalizer), AdamW for the rest, 8-bit state | 2.74 vs 3.33 (AdamW) at equal FLOPs |
| `bantamtrain.lowmem` | 8-bit blockwise optimizer state; bf16 weights with stochastic rounding (no fp32 master copy) | weights + state 3x smaller, loss 2.728 vs 2.737 |
| `bantamtrain.qweights` | int8 / int4 weight-only `Linear` and `Embedding` kept compressed in RAM | int8: 3.9x smaller, +0.0002 loss; int4: 7.5x smaller, +0.026 |
| `bantamtrain.weightpack` | compressed checkpoints: fp32 / lossless bf16 / int8 / int4 | lossless bf16 file 2.9x smaller than fp32 |
| `bantamtrain.data.tokpack` | TokPack v2: random-access compressed token store (rank remap + LEB128 + zstd dictionary) | 2.68x smaller than uint16 ids (5.97 bits/token), exact |
| `bantamtrain.data.dedup` | exact + MinHash-LSH near-duplicate removal on token streams, vectorised | 4 % exact duplicates found in TinyStories |
| `bantamtrain.data.gridpack` | GridPack: error-bounded int16 + temporal delta + zstd for gridded fields | ERA5 5.2x smaller than float32, error <= 5 % of a 6 h change |
| `bantamtrain.device` | runtime: flush-to-zero on CPU, AMP, TF32, compile, batch finder, duty-cycle limiter | the FTZ fix made affected CPU matmuls up to 23x faster |
| `bantamtrain.memory` | memory / time scoreboard for single-GPU training | `python -m bantamtrain.memory` |
| `bantamtrain.models` | `ZoomGPT` (decoder LM) and `ZoomViT` (weather grid forecaster) with every option above | |
| `bantamtrain.lora` | LoRA / QLoRA adapters on `nn.Linear` or int8 / NF4 / int4 `QLinear` bases; merge, adapter-only state dicts | |
| `bantamtrain.losses` | chunked cross-entropy: never materializes the tokens x vocab logits | |
| `bantamtrain.finetune` | `python -m bantamtrain.finetune`: SFT of any HuggingFace causal LM (chat templates, response-only loss, best-fit packing), Muon + 8-bit state, torchrun DDP | SmolLM2-135M QLoRA on Dolly: held-out loss 3.39 -> 3.18 in 40 CPU steps |

## Quick start

```python
import torch
from bantamtrain.models import ZoomGPT
from bantamtrain.optim import build_optimizer
from bantamtrain.zoom import ZoomSchedule, lr_at

model = ZoomGPT(vocab=4096, d=256, n_layer=6, n_head=4, zoom_ks=[2])
opt = build_optimizer(model, "muon", lr=1.6e-2, weight_decay=0.1, state_bits=8)
sched = ZoomSchedule.parse("2:0.5,1:0.5")      # k=2 for the first half of the budget

x = torch.randint(0, 4096, (16, 258))
for frac in (0.1, 0.6):                         # fraction of the compute budget spent
    k = sched.k_at(frac)
    for g in opt.param_groups:
        g["lr"] = lr_at(frac, 1.6e-2, sched="wsd")
    loss = model(x[:, :256], x[:, k:256 + k], k=k)
    opt.zero_grad(); loss.backward(); opt.step()
```

Store a trained model small:

```python
from bantamtrain import qweights, weightpack
weightpack.save("model.zwpk", model.state_dict(), bits=8)    # int8 checkpoint file
qweights.quantize_model(model, bits=4)                        # int4 weights in RAM, float compute
```

Fine-tune a HuggingFace model (LoRA on an int8 / NF4 base, Muon, 8-bit state):

```bash
pip install bantamtrain[finetune]
python -m bantamtrain.finetune --model HuggingFaceTB/SmolLM2-135M \
    --dataset databricks/databricks-dolly-15k --max_examples 2000 \
    --lora 16 --quant 8 --optim muon --state_bits 8 --epochs 1 --out runs/dolly
torchrun --nproc_per_node 4 -m bantamtrain.finetune ...        # multi-GPU
python -m bantamtrain.memory --finetune --params 7e9 --gpu_gb 24  # will it fit?
```

Measured on SmolLM2-135M before training (100 held-out Dolly examples): int8
base costs +0.003 loss (558 -> 200 MB), NF4 linears with int8 embeddings
+0.18 (140 MB). Use int8 for small bases; NF4 for 7B+ where the literature
finds it near-lossless.

Size a run on one GPU:

```bash
python -m bantamtrain.memory --params 5e9 --gpu_gb 96 --tokens 1e11
```

## Honest notes

- Muon, 8-bit optimizer state, stochastic-rounding bf16, LoRA/QLoRA/NF4 and
  blockwise weight quantization are published techniques re-implemented here;
  Zoom Training and the storage formats are this project's own.
- All quality numbers above were measured on a 5.8M-parameter model; gains at
  larger scale are expected to be smaller for the optimizer and are untested for
  Zoom. Nothing has been run on a GPU yet.
- Source, experiments, tests and the full write-up:
  https://github.com/Wisk-inc/AboxRebuildcode
