#!/usr/bin/env bash
# Orchestrate the inference-only H1-H4 experimental pipeline for COAST/VECTRA.
#
# Typical full run:
#   DATASETS_ROOT=/path/to/dvrptw_dynamic_grid \
#   bash script/run_hypothesis_experiments.sh
#
# Useful smoke run:
#   DATASETS_ROOT=/path/to/dvrptw_dynamic_grid \
#   DYNAMIC_MAX_FILES=1 \
#   DIAG_MAX_FILES=1 \
#   OOD_SMOKE_SIZE=8 \
#   INCLUDE_EXTERNAL=0 \
#   bash script/run_hypothesis_experiments.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-output/hypothesis_experiments}"
DATASETS_ROOT="${DATASETS_ROOT:-data/datasets/dvrptw_dynamic_grid}"
RL4DVRPTW_ROOT="${RL4DVRPTW_ROOT:-/home/admin_wsl/projects/RL4DVRPTW}"

RUN_STATIC_CHECKS="${RUN_STATIC_CHECKS:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
RUN_OOD="${RUN_OOD:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
BUILD_HYPOTHESIS="${BUILD_HYPOTHESIS:-1}"

INCLUDE_EXTERNAL="${INCLUDE_EXTERNAL:-1}"
VERIFY_ROUTES="${VERIFY_ROUTES:-1}"
RUN_OR_TOOLS="${RUN_OR_TOOLS:-0}"
RUN_LKH="${RUN_LKH:-0}"

DYNAMIC_OUTPUT="${DYNAMIC_OUTPUT:-$EXPERIMENT_ROOT/dynamic_benchmark_verified}"
DYNAMIC_MODEL_LIST="${DYNAMIC_MODEL_LIST:-vectra,mardam,b0,b1,b3,b5,edgeoff,no_ownership,no_lookahead,am,polynet}"
DYNAMIC_FILE_GLOB="${DYNAMIC_FILE_GLOB:-**/*.csv}"
DYNAMIC_MAX_FILES="${DYNAMIC_MAX_FILES:-0}"

OOD_DATASETS_DIR="${OOD_DATASETS_DIR:-$EXPERIMENT_ROOT/test_sets}"
OOD_OUTPUT="${OOD_OUTPUT:-$EXPERIMENT_ROOT/ood_eval}"
OOD_MODELS="${OOD_MODELS:-vectra,mardam,b0,b1,b3,b5,edgeoff,no_ownership,no_lookahead}"
OOD_BATCH_SIZE="${OOD_BATCH_SIZE:-500}"
OOD_SMOKE_SIZE="${OOD_SMOKE_SIZE:-0}"
OOD_DEVICE_CPU="${OOD_DEVICE_CPU:-0}"
OOD_NO_VERIFY_ROUTES="${OOD_NO_VERIFY_ROUTES:-0}"
OOD_NO_NORMALIZE="${OOD_NO_NORMALIZE:-1}"

DIAG_OUTPUT="${DIAG_OUTPUT:-$EXPERIMENT_ROOT/dynamic_diagnostics_subset}"
DIAG_MODEL_LIST="${DIAG_MODEL_LIST:-vectra,b0,b1,b3,b5,edgeoff,no_ownership,no_lookahead}"
DIAG_FILE_GLOB="${DIAG_FILE_GLOB:-**/instance_0000[0-7].csv}"
DIAG_MAX_FILES="${DIAG_MAX_FILES:-0}"
STEP_DIAG_LIMIT="${STEP_DIAG_LIMIT:-1}"
BEHAVIOR_OUTPUT="${BEHAVIOR_OUTPUT:-$EXPERIMENT_ROOT/behavior_analysis}"

HYPOTHESIS_OUTPUT="${HYPOTHESIS_OUTPUT:-$EXPERIMENT_ROOT/hypothesis_tables}"

log_step() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

require_dynamic_grid() {
  if [[ ! -d "$DATASETS_ROOT" ]]; then
    echo "DATASETS_ROOT not found: $DATASETS_ROOT" >&2
    echo "Set DATASETS_ROOT to a dynamic-grid CSV root with dod_*/tw_*/seed_*/instance_*.csv." >&2
    exit 2
  fi
}

if [[ "$RUN_STATIC_CHECKS" == "1" ]]; then
  log_step "[0/5] Static checks"
  "$PYTHON_BIN" -m py_compile \
    script/build_experimental_report.py \
    script/build_hypothesis_tables.py \
    script/generate_ood_sets.py \
    script/run_ood_experiments.py \
    script/analyze_behavior_diagnostics.py \
    script/infer_all_datasets.py \
    script/infer_all_datasets_mardam.py
  bash -n script/run_dynamic_experiment_matrix.sh
fi

if [[ "$RUN_DYNAMIC" == "1" ]]; then
  require_dynamic_grid
  log_step "[1/5] Dynamic grid: H1/H2/H3/H4 paired ID evidence"
  DATASETS_ROOT="$DATASETS_ROOT" \
  OUTPUT_ROOT="$DYNAMIC_OUTPUT" \
  VERIFY_ROUTES="$VERIFY_ROUTES" \
  INCLUDE_EXTERNAL="$INCLUDE_EXTERNAL" \
  RL4DVRPTW_ROOT="$RL4DVRPTW_ROOT" \
  RUN_OR_TOOLS="$RUN_OR_TOOLS" \
  RUN_LKH="$RUN_LKH" \
  MODEL_LIST="$DYNAMIC_MODEL_LIST" \
  FILE_GLOB="$DYNAMIC_FILE_GLOB" \
  MAX_FILES="$DYNAMIC_MAX_FILES" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash script/run_dynamic_experiment_matrix.sh
fi

if [[ "$RUN_OOD" == "1" ]]; then
  log_step "[2/5] OOD datasets and evaluation: H3/H4 generalization evidence"
  ood_gen_args=(--output-dir "$OOD_DATASETS_DIR")
  if [[ "$OOD_SMOKE_SIZE" != "0" ]]; then
    ood_gen_args+=(--smoke-size "$OOD_SMOKE_SIZE")
  else
    ood_gen_args+=(--batch-size "$OOD_BATCH_SIZE")
  fi
  PYTHONPATH=. "$PYTHON_BIN" script/generate_ood_sets.py "${ood_gen_args[@]}"

  ood_eval_args=(
    --datasets-dir "$OOD_DATASETS_DIR"
    --output-dir "$OOD_OUTPUT"
    --models "$OOD_MODELS"
    --python-executable "$PYTHON_BIN"
  )
  if [[ "$OOD_DEVICE_CPU" == "1" ]]; then
    ood_eval_args+=(--device-cpu)
  fi
  if [[ "$OOD_NO_VERIFY_ROUTES" == "1" ]]; then
    ood_eval_args+=(--no-verify-routes)
  fi
  if [[ "$OOD_NO_NORMALIZE" == "1" ]]; then
    ood_eval_args+=(--no-normalize)
  fi
  PYTHONPATH=. "$PYTHON_BIN" script/run_ood_experiments.py "${ood_eval_args[@]}"
fi

if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
  require_dynamic_grid
  log_step "[3/5] Behavioral diagnostics subset: H1/H2 mechanism evidence"
  DATASETS_ROOT="$DATASETS_ROOT" \
  OUTPUT_ROOT="$DIAG_OUTPUT" \
  VERIFY_ROUTES="$VERIFY_ROUTES" \
  INCLUDE_EXTERNAL=0 \
  MODEL_LIST="$DIAG_MODEL_LIST" \
  FILE_GLOB="$DIAG_FILE_GLOB" \
  MAX_FILES="$DIAG_MAX_FILES" \
  STEP_DIAGNOSTICS=1 \
  STEP_DIAG_LIMIT="$STEP_DIAG_LIMIT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash script/run_dynamic_experiment_matrix.sh

  PYTHONPATH=. "$PYTHON_BIN" script/analyze_behavior_diagnostics.py \
    --input-root "$DIAG_OUTPUT" \
    --input-glob '**/per_case_json/**/*.infer.json' \
    --master-summary "$DIAG_OUTPUT/paper_ready/master_summary.csv" \
    --datasets-root "$DATASETS_ROOT" \
    --output-dir "$BEHAVIOR_OUTPUT"
fi

if [[ "$BUILD_HYPOTHESIS" == "1" ]]; then
  log_step "[4/5] Build H1-H4 hypothesis tables"
  master_summary="${DYNAMIC_MASTER_SUMMARY:-$DYNAMIC_OUTPUT/paper_ready/master_summary.csv}"
  ood_summary="${OOD_SUMMARY:-$OOD_OUTPUT/ood_summary.csv}"
  behavior_summary="${BEHAVIOR_SUMMARY:-$BEHAVIOR_OUTPUT/hypothesis_behavior_summary.csv}"

  if [[ ! -f "$master_summary" ]]; then
    echo "Missing master summary: $master_summary" >&2
    echo "Run dynamic grid first or set DYNAMIC_MASTER_SUMMARY." >&2
    exit 3
  fi

  hypothesis_args=(--master-summary "$master_summary" --output-dir "$HYPOTHESIS_OUTPUT")
  if [[ -f "$ood_summary" ]]; then
    hypothesis_args+=(--ood-summary "$ood_summary")
  else
    echo "[WARN] OOD summary not found, building hypothesis tables without OOD: $ood_summary" >&2
  fi
  if [[ -f "$behavior_summary" ]]; then
    hypothesis_args+=(--behavior-summary "$behavior_summary")
  else
    echo "[WARN] Behavior summary not found, building hypothesis tables without diagnostics: $behavior_summary" >&2
  fi
  PYTHONPATH=. "$PYTHON_BIN" script/build_hypothesis_tables.py "${hypothesis_args[@]}"
fi

log_step "[5/5] Done"
echo "Dynamic report    : $DYNAMIC_OUTPUT/paper_ready"
echo "OOD report        : $OOD_OUTPUT"
echo "Behavior analysis : $BEHAVIOR_OUTPUT"
echo "Hypothesis tables : $HYPOTHESIS_OUTPUT"
