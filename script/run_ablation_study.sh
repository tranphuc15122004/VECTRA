#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

# ── Problem ───────────────────────────────────────────────────────────────────
PROBLEM_TYPE=${PROBLEM_TYPE:-dvrptw}
CUSTOMERS=${CUSTOMERS:-50}
VEHICLES=${VEHICLES:-3}

# ── Training schedule ─────────────────────────────────────────────────────────
EPOCHS=${EPOCHS:-500}
ITERS=${ITERS:-1000}
BATCH=${BATCH:-512}
TEST_BATCH=${TEST_BATCH:-10240}
LR=${LR:-1e-4}

# ── Model architecture (matching train_vectra_main.sh) ───────────────────────
MODEL_SIZE=${MODEL_SIZE:-128}
LAYER_COUNT=${LAYER_COUNT:-3}
HEAD_COUNT=${HEAD_COUNT:-4}
FF_SIZE=${FF_SIZE:-256}
EDGE_FEAT_SIZE=${EDGE_FEAT_SIZE:-8}
CUST_K=${CUST_K:-20}
MEMORY_SIZE=${MEMORY_SIZE:-128}
LOOKAHEAD_HIDDEN=${LOOKAHEAD_HIDDEN:-128}
DROPOUT=${DROPOUT:-0.1}

# ── Optimization ──────────────────────────────────────────────────────────────
BASELINE=${BASELINE:-critic}
CRITIC_LR=${CRITIC_LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-2}
NUM_WORKERS=${NUM_WORKERS:-4}

# ── Ablation controls ─────────────────────────────────────────────────────────
# Supported: none coast b0 b1 b3 b5 edgeoff a0 a1 a3 a4 a9
ABLATION_PROFILE=${ABLATION_PROFILE:-coast}
# Optional override for linear mode only: "w_att w_owner w_look"
LINEAR_WEIGHTS=${LINEAR_WEIGHTS:-}

# ── Runtime / reproducibility ─────────────────────────────────────────────────
SEED=${SEED:-42}
AMP=${AMP:-1}

# ── Checkpoint / output ───────────────────────────────────────────────────────
# Resume same profile training:
#   RESUME=output/ablation_b0_seed42/chkpt_ep100.pyth ./script/run_ablation_study.sh
RESUME=${RESUME:-}
# Warm-start from a model checkpoint:
#   MODEL_WEIGHT=data/vectra/chkpt_best.pyth ./script/run_ablation_study.sh
MODEL_WEIGHT=${MODEL_WEIGHT:-}
# If empty, output path is generated from profile/seed.
OUTDIR=${OUTDIR:-}

cd "$(dirname "$0")/.."

if [[ -z "$OUTDIR" ]]; then
	OUTDIR="output/ablation_${ABLATION_PROFILE}_seed${SEED}"
fi

RESUME_ARG=()
if [[ -n "$RESUME" ]]; then
	RESUME_ARG=(--resume-state "$RESUME")
fi

MODEL_WEIGHT_ARG=()
if [[ -n "$MODEL_WEIGHT" ]]; then
	MODEL_WEIGHT_ARG=(--model-weight "$MODEL_WEIGHT")
fi

AMP_ARG=()
if [[ "$AMP" == "1" || "$AMP" == "true" || "$AMP" == "TRUE" ]]; then
	AMP_ARG=(--amp)
fi

LINEAR_ARG=()
if [[ -n "$LINEAR_WEIGHTS" ]]; then
	# shellcheck disable=SC2206
	_lw=($LINEAR_WEIGHTS)
	if [[ ${#_lw[@]} -ne 3 ]]; then
		echo "LINEAR_WEIGHTS must contain exactly 3 values: 'w_att w_owner w_look'"
		exit 1
	fi
	LINEAR_ARG=(--linear-fusion-weights "${_lw[0]}" "${_lw[1]}" "${_lw[2]}")
fi

echo "Profile : $ABLATION_PROFILE"
echo "Seed    : $SEED"
echo "Outdir  : $OUTDIR"

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
	--max-grad-norm     "$MAX_GRAD_NORM" \
	--model-size        "$MODEL_SIZE" \
	--layer-count       "$LAYER_COUNT" \
	--head-count        "$HEAD_COUNT" \
	--ff-size           "$FF_SIZE" \
	--edge-feat-size    "$EDGE_FEAT_SIZE" \
	--cust-k            "$CUST_K" \
	--memory-size       "$MEMORY_SIZE" \
	--lookahead-hidden  "$LOOKAHEAD_HIDDEN" \
	--dropout           "$DROPOUT" \
	--baseline-type     "$BASELINE" \
	--critic-rate       "$CRITIC_LR" \
	--num-workers       "$NUM_WORKERS" \
	--pin-memory \
	--rng-seed          "$SEED" \
	--ablation-profile  "$ABLATION_PROFILE" \
	--output-dir        "$OUTDIR" \
	"${AMP_ARG[@]}" \
	"${LINEAR_ARG[@]}" \
	"${RESUME_ARG[@]}" \
	"${MODEL_WEIGHT_ARG[@]}"
