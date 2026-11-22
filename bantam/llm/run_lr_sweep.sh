#!/bin/bash
# Baseline learning-rate sweep at 1/3 of the main budget (picks the LR all runs use).
set -e
cd "$(dirname "$0")"
DATA=${DATA:-/home/user/data/tinystories/tok4k}
for LR in 1e-3 2e-3 4e-3; do
  python3 train.py --data $DATA --zoom 1 --budget 2.3e13 --lr $LR --name lrsweep_base_lr$LR \
    --out ../results/llm_lrsweep --evals 4 --final_eval_seqs 512
done
