#!/bin/bash
# GPU smoke test: run on a machine with CUDA to check the bf16/compile/fused paths.
# Expected: every run finishes and writes results/gpu/<name>.json. It is not a benchmark:
# at equal FLOPs Zoom showed no wall-clock gain on CPU (RESULTS.md section 8), and nothing
# has been timed on a GPU. --recipe best = the settings measured best on CPU (Muon LR
# 1.6e-2, QK-norm, Zoom 2:0.33,1:0.67 with batch x2); batch 64 is not tuned.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-/home/user/data/tinystories/tok4k}
python3 -m bantamtrain pretrain --data $DATA --budget 2e14 --zoom 1 --optim muon --lr 1.6e-2 \
    --device cuda --precision bf16 --compile --batch 64 --evals 4 --name gpu_base --out results/gpu
python3 -m bantamtrain pretrain --data $DATA --budget 2e14 --recipe best \
    --device cuda --precision bf16 --compile --batch 64 --evals 4 --name gpu_best --out results/gpu
# 70 % duty-cycle cap when a power cap (nvidia-smi -pl) is not available:
python3 -m bantamtrain pretrain --data $DATA --budget 5e13 --recipe best \
    --device cuda --precision bf16 --duty 0.7 --batch 64 --evals 2 --name gpu_best_duty70 --out results/gpu
