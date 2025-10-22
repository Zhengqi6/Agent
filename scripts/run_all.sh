#!/usr/bin/env bash
set -euo pipefail

echo "Running unit tests..."
PYTHONPATH=src python -m unittest discover -s tests

echo "Running capacity sweep on text8..."
PYTHONPATH=src python -m experiments.capacity_sweep \
  --data-path data/text8 \
  --chunk-size 512 \
  --limit 80000 \
  --capacities 4096 8192 \
  --query-interval 10 \
  --baselines sliding reservoir \
  --collector-variants ttl_mark ttl_mark_gen \
  --seeds 1 \
  --output results/text8_capacity_sparse.csv

echo "Running HotpotQA benchmark (train[:200])..."
PYTHONPATH=src python -m experiments.hotpot_benchmark \
  --dataset-split train[:200] \
  --max-samples 200 \
  --hot-capacity 16384 \
  --sliding-window 2048 \
  --output results/hotpot_summary.csv

echo "All experiments completed."
