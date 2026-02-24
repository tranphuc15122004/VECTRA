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
BATCH=${BATCH:-1024}           # PPO collects a full rollout per sample; smaller
TEST_BATCH=${TEST_BATCH:-512} # batches are more memory-efficient than REINFORCE
LR=${LR:-1e-4}
CRITIC_LR=${CRITIC_LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}

# ── PPO hyper-parameters ──────────────────────────────────────────────────────
PPO_EPOCHS=${PPO_EPOCHS:-4}          # gradient updates per rollout
PPO_CLIP=${PPO_CLIP:-0.2}             # clipping ratio ε
PPO_VALUE_COEF=${PPO_VALUE_COEF:-0.5} # critic loss coefficient
PPO_ENTROPY_COEF=${PPO_ENTROPY_COEF:-0.01} # entropy bonus (exploration)
PPO_GAMMA=${PPO_GAMMA:-1.0}           # discount factor (1.0 = no discount for routing)
PPO_GAE_LAMBDA=${PPO_GAE_LAMBDA:-0.95} # GAE λ (0.95 balances bias/variance)
PPO_ADV_NORM=${PPO_ADV_NORM:-true}    # normalise advantages per rollout
# PPO_TARGET_KL: set to e.g. 0.01 to enable early stopping; leave empty to disable
PPO_TARGET_KL=${PPO_TARGET_KL:-}

# ── Model architecture ────────────────────────────────────────────────────────
MODEL_SIZE=${MODEL_SIZE:-128}
LAYER_COUNT=${LAYER_COUNT:-2}
HEAD_COUNT=${HEAD_COUNT:-4}
FF_SIZE=${FF_SIZE:-256}
EDGE_FEAT_SIZE=${EDGE_FEAT_SIZE:-8}
CUST_K=${CUST_K:-20}
MEMORY_SIZE=${MEMORY_SIZE:-128}
LOOKAHEAD_HIDDEN=${LOOKAHEAD_HIDDEN:-128}
DROPOUT=${DROPOUT:-0.1}

# ── Learnable MoE ─────────────────────────────────────────────────────────────
# --sbg-train-ready enables SBG + adaptive-k + MoE + latent bottleneck + adaptive depth.
# Tune the load-balance coefficient to control expert diversity.
MOE_LB_COEF=${MOE_LB_COEF:-0.01}

# ── Checkpoint / output ───────────────────────────────────────────────────────
OUTDIR=${OUTDIR:-output/ppo_n${CUSTOMERS}m${VEHICLES}}
# Resume from a PPO checkpoint (model dimensions must match):
#   RESUME=output/ppo_n50m3/chkpt_ep100.pyth ./script/train_ppo.sh
# Warm-start from a REINFORCE checkpoint (same arch, skips critic state):
#   MODEL_WEIGHT=output/sbg_ready_n50m3/chkpt_ep300.pyth ./script/train_ppo.sh
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

PPO_TARGET_KL_ARG=()
if [[ -n "$PPO_TARGET_KL" ]]; then
  PPO_TARGET_KL_ARG=(--ppo-target-kl "$PPO_TARGET_KL")
fi

ADV_NORM_ARG=()
if [[ "$PPO_ADV_NORM" == "true" ]]; then
  ADV_NORM_ARG=(--ppo-adv-norm)
fi

PYTHONPATH=. "$PYTHON_BIN" MODEL/train_PPO.py \
  --problem-type      "$PROBLEM_TYPE" \
  --customers-count   "$CUSTOMERS" \
  --vehicles-count    "$VEHICLES" \
  --epoch-count       "$EPOCHS" \
  --iter-count        "$ITERS" \
  --batch-size        "$BATCH" \
  --test-batch-size   "$TEST_BATCH" \
  --learning-rate     "$LR" \
  --critic-rate       "$CRITIC_LR" \
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
  --ppo-epochs        "$PPO_EPOCHS" \
  --ppo-clip-range    "$PPO_CLIP" \
  --ppo-value-coef    "$PPO_VALUE_COEF" \
  --ppo-entropy-coef  "$PPO_ENTROPY_COEF" \
  --ppo-gamma         "$PPO_GAMMA" \
  --ppo-gae-lambda    "$PPO_GAE_LAMBDA" \
  --amp \
  --num-workers       4 \
  --pin-memory \
  --sbg-train-ready \
  --sbg-moe-load-balance-coef "$MOE_LB_COEF" \
  --output-dir        "$OUTDIR" \
  "${ADV_NORM_ARG[@]}" \
  "${PPO_TARGET_KL_ARG[@]}" \
  "${RESUME_ARG[@]}" \
  "${MODEL_WEIGHT_ARG[@]}"
