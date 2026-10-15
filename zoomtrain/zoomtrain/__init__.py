"""zoomtrain: train small LLMs and weather models faster on limited compute.

Components
  zoom      - Zoom Training (coarse-to-fine resolution annealing), schedules
  optim     - Muon + AdamW
  device    - CPU/GPU runtime (FTZ, AMP, compile, batch finder, duty limiter)
  models    - ZoomGPT (text), ZoomViT (weather grids)
  data      - TokPack token store, dedup, GridPack weather store
"""
from .zoom import SlotMerge, SlotUnmerge, ZoomSchedule, lr_at  # noqa: F401

__version__ = "0.1.0"
