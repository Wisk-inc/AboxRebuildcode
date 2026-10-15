#!/bin/bash
# GPU smoke test: run on a machine with CUDA to check the bf16/compile/fused paths.
# Expected: both runs finish, Zoom run completes more steps per wall-clock second.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-/home/user/data/tinystories/tok4k}
python3 llm/train.py --data $DATA --budget 2e14 --zoom 1 --optim muon --lr 2e-3 \
    --device cuda --precision bf16 --compile --batch 64 --evals 4 --name gpu_base --out results/gpu
python3 llm/train.py --data $DATA --budget 2e14 --zoom 4:0.35,2:0.25,1:0.4 --optim muon --lr 2e-3 \
    --device cuda --precision bf16 --compile --batch 64 --evals 4 --name gpu_zoom --out results/gpu
# 70 % duty-cycle cap when a power cap (nvidia-smi -pl) is not available:
python3 llm/train.py --data $DATA --budget 5e13 --zoom 4:0.35,2:0.25,1:0.4 --optim muon --lr 2e-3 \
    --device cuda --precision bf16 --duty 0.7 --batch 64 --evals 2 --name gpu_zoom_duty70 --out results/gpu
