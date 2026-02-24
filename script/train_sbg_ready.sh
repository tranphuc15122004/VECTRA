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
LR=${LR:-5e-5}

# ── Model architecture (matching old training command) ────────────────────────
MODEL_SIZE=${MODEL_SIZE:-128}
LAYER_COUNT=${LAYER_COUNT:-2}
HEAD_COUNT=${HEAD_COUNT:-4}
FF_SIZE=${FF_SIZE:-256}
EDGE_FEAT_SIZE=${EDGE_FEAT_SIZE:-8}
CUST_K=${CUST_K:-20}
MEMORY_SIZE=${MEMORY_SIZE:-128}
LOOKAHEAD_HIDDEN=${LOOKAHEAD_HIDDEN:-128}
DROPOUT=${DROPOUT:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}

# ── Learnable MoE (new) ───────────────────────────────────────────────────────
# --sbg-train-ready enables MoE automatically; tune the load-balance coefficient here.
MOE_LB_COEF=${MOE_LB_COEF:-0.01}

# ── Checkpoint / output ───────────────────────────────────────────────────────
# Set RESUME to a checkpoint path to continue training, e.g.:
#   RESUME=output/DVRPTWn50m3_xxx/chkpt_ep100.pyth ./script/train_sbg_ready.sh
# NOTE: do NOT resume from an n100m5 checkpoint — model dimensions are incompatible.
RESUME=${RESUME:-}

cd "$(dirname "$0")/.."

RESUME_ARG=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARG=(--resume-state "$RESUME")
fi

PYTHONPATH=. "$PYTHON_BIN" MODEL/train.py \
  --problem-type      "$PROBLEM_TYPE" \
  --customers-count   "$CUSTOMERS" \
  --vehicles-count    "$VEHICLES" \
  --epoch-count       "$EPOCHS" \
  --iter-count        "$ITERS" \
  --batch-size        "$BATCH" \
  --test-batch-size   "$TEST_BATCH" \
  --learning-rate     "$LR" \
  --weight-decay      "$WEIGHT_DECAY" \
  --model-size        "$MODEL_SIZE" \
  --layer-count       "$LAYER_COUNT" \
  --head-count        "$HEAD_COUNT" \
  --ff-size           "$FF_SIZE" \
  --edge-feat-size    "$EDGE_FEAT_SIZE" \
  --cust-k            "$CUST_K" \
  --memory-size       "$MEMORY_SIZE" \
  --lookahead-hidden  "$LOOKAHEAD_HIDDEN" \
  --dropout           "$DROPOUT" \
  --baseline-type     critic \
  --amp \
  --num-workers       4 \
  --pin-memory \
  --sbg-train-ready \
  --sbg-moe-load-balance-coef "$MOE_LB_COEF" \
  "${RESUME_ARG[@]}"
