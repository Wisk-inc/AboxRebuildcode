"""bantamtrain: train and fine-tune small language and weather models on limited hardware.

Components
  zoom      - Zoom Training (coarse-to-fine resolution annealing), schedules
  optim     - Muon + AdamW
  lowmem    - 8-bit optimizer state, bf16 weights with stochastic rounding
  device    - CPU/GPU runtime (FTZ, AMP, compile, batch finder, duty limiter)
  models    - ZoomGPT (text), ZoomViT (weather grids)
  data      - TokPack token store, dedup, GridPack weather store
  lora, qweights, losses, finetune - LoRA / QLoRA fine-tuning of HF models
  memory    - single-GPU memory / time scoreboard
  cli       - the `bantam` command (`bantam doctor` for an environment report)
"""
# the single source of the version: pyproject.toml reads it (setuptools dynamic attr)
__version__ = "0.3.0"

from .zoom import SlotMerge, SlotUnmerge, ZoomSchedule, lr_at  # noqa: F401,E402
