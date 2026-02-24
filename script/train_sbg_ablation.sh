#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/phuc/miniconda3/envs/kaggle/bin/python}

PROBLEM_TYPE=${PROBLEM_TYPE:-dvrptw}
CUSTOMERS=${CUSTOMERS:-100}
VEHICLES=${VEHICLES:-5}
EPOCHS=${EPOCHS:-100}
ITERS=${ITERS:-300}
BATCH=${BATCH:-128}
TEST_BATCH=${TEST_BATCH:-256}
LR=${LR:-1e-4}
BASE_OUT=${BASE_OUT:-output/sbg_ablation_n${CUSTOMERS}m${VEHICLES}}

cd "$(dirname "$0")/.."

common_args=(
  --problem-type "$PROBLEM_TYPE"
  --customers-count "$CUSTOMERS"
  --vehicles-count "$VEHICLES"
  --epoch-count "$EPOCHS"
  --iter-count "$ITERS"
  --batch-size "$BATCH"
  --test-batch-size "$TEST_BATCH"
  --baseline-type critic
  --learning-rate "$LR"
  --num-workers 4
  --pin-memory
  --amp
)

# 1) Base SBG (no MoE, no latent bottleneck)
PYTHONPATH=. "$PYTHON_BIN" MODEL/train.py "${common_args[@]}" \
  --sbg-enable --sbg-cand-k 16 --sbg-adaptive-k --sbg-k-min 8 --sbg-k-max 32 \
  --adaptive-depth --adaptive-min-layers 1 --adaptive-easy-ratio 0.7 \
  --output-dir "${BASE_OUT}_base"

# 2) Base SBG + MoE guardrail
PYTHONPATH=. "$PYTHON_BIN" MODEL/train.py "${common_args[@]}" \
  --sbg-enable --sbg-cand-k 16 --sbg-adaptive-k --sbg-k-min 8 --sbg-k-max 32 \
  --adaptive-depth --adaptive-min-layers 1 --adaptive-easy-ratio 0.7 \
  --sbg-moe-enable --sbg-moe-strength 0.03 --sbg-moe-uncertainty \
  --sbg-moe-min-strength 0.01 --sbg-moe-entropy-floor 0.35 --sbg-moe-margin-ceil 1.5 \
  --output-dir "${BASE_OUT}_moe"

# 3) Base SBG + MoE + latent bottleneck (for larger instances)
PYTHONPATH=. "$PYTHON_BIN" MODEL/train.py "${common_args[@]}" \
  --sbg-enable --sbg-cand-k 16 --sbg-adaptive-k --sbg-k-min 8 --sbg-k-max 32 \
  --adaptive-depth --adaptive-min-layers 1 --adaptive-easy-ratio 0.7 \
  --sbg-moe-enable --sbg-moe-strength 0.03 --sbg-moe-uncertainty \
  --sbg-moe-min-strength 0.01 --sbg-moe-entropy-floor 0.35 --sbg-moe-margin-ceil 1.5 \
  --latent-bottleneck --latent-tokens 32 --latent-min-nodes 64 \
  --output-dir "${BASE_OUT}_moe_latent"
