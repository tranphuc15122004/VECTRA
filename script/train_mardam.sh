#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/phuc/miniconda3/envs/kaggle/bin/python}

# ── Problem ───────────────────────────────────────────────────────────────────
PROBLEM_TYPE=${PROBLEM_TYPE:-dvrptw}
CUSTOMERS=${CUSTOMERS:-50}
VEHICLES=${VEHICLES:-3}

# ── Training schedule ─────────────────────────────────────────────────────────
EPOCHS=${EPOCHS:-300}
ITERS=${ITERS:-1000}
BATCH=${BATCH:-1024}
TEST_BATCH=${TEST_BATCH:-512}
LR=${LR:-1e-4}

# ── Model architecture (classic MARDAM) ──────────────────────────────────────
MODEL_SIZE=${MODEL_SIZE:-128}
LAYER_COUNT=${LAYER_COUNT:-3}
HEAD_COUNT=${HEAD_COUNT:-8}
FF_SIZE=${FF_SIZE:-512}
TANH_XPLOR=${TANH_XPLOR:-10}

# ── Optimization ──────────────────────────────────────────────────────────────
BASELINE=${BASELINE:-critic}
CRITIC_LR=${CRITIC_LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-2}
NUM_WORKERS=${NUM_WORKERS:-4}

# ── Checkpoint / output ───────────────────────────────────────────────────────
RESUME=${RESUME:-}
MODEL_WEIGHT=${MODEL_WEIGHT:-}

cd "$(dirname "$0")/.."

RESUME_ARG=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARG=(--resume-state "$RESUME")
fi

MODEL_WEIGHT_ARG=()
if [[ -n "$MODEL_WEIGHT" ]]; then
  MODEL_WEIGHT_ARG=(--model-weight "$MODEL_WEIGHT")
fi

PYTHONPATH=. "$PYTHON_BIN" script/train_mardam.py \
  --problem-type      "$PROBLEM_TYPE" \
  --customers-count   "$CUSTOMERS" \
  --vehicles-count    "$VEHICLES" \
  --epoch-count       "$EPOCHS" \
  --iter-count        "$ITERS" \
  --batch-size        "$BATCH" \
  --test-batch-size   "$TEST_BATCH" \
  --learning-rate     "$LR" \
  --weight-decay      "$WEIGHT_DECAY" \
  --max-grad-norm     "$MAX_GRAD_NORM" \
  --model-size        "$MODEL_SIZE" \
  --layer-count       "$LAYER_COUNT" \
  --head-count        "$HEAD_COUNT" \
  --ff-size           "$FF_SIZE" \
  --tanh-xplor        "$TANH_XPLOR" \
  --baseline-type     "$BASELINE" \
  --critic-rate       "$CRITIC_LR" \
  --amp \
  --num-workers       "$NUM_WORKERS" \
  --pin-memory \
  "${RESUME_ARG[@]}" \
  "${MODEL_WEIGHT_ARG[@]}"
