#!/bin/bash
# Everything still to run, in priority order; run_queue skips runs whose JSON exists.
cd "$(dirname "$0")/../.."
python3 experiments/run_queue.py experiments/queues/followup.txt --workers 2 --threads 2
python3 experiments/run_queue.py experiments/queues/weather_main.txt --workers 2 --threads 2
python3 experiments/run_queue.py experiments/queues/llm_main.txt --workers 2 --threads 2
