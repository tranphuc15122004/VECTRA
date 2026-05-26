#!/usr/bin/env bash
# Run paper-facing dynamic-grid inference for local COAST/VECTRA checkpoints,
# then rebuild manifest/cell/detail/significance artifacts.
#
# Usage:
#   DATASETS_ROOT=data/datasets/dvrptw_dynamic_grid \
#   OUTPUT_ROOT=output/dynamic_benchmark_raw \
#   bash script/run_dynamic_experiment_matrix.sh
#
# Optional env:
#   VERIFY_ROUTES=0          # append --no-verify-routes
#   STEP_DIAGNOSTICS=1       # include per-step score diagnostics for VECTRA-like models
#   STEP_DIAG_LIMIT=1
#   INCLUDE_EXTERNAL=1       # run AM and PolyNet from RL4DVRPTW_ROOT
#   RL4DVRPTW_ROOT=/home/admin_wsl/projects/RL4DVRPTW
#   RUN_OR_TOOLS=1           # run OR-Tools baseline
#   RUN_LKH=1                # run LKH baseline
#   MODEL_LIST=vectra,b0     # comma-separated local/external models to run
#   FILE_GLOB='**/*.csv'     # forwarded to local batch inference
#   MAX_FILES=0              # optional cap per local model, 0 means all

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS_ROOT="${DATASETS_ROOT:-data/datasets/dvrptw_dynamic_grid}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/dynamic_benchmark_raw}"
VERIFY_ROUTES="${VERIFY_ROUTES:-1}"
STEP_DIAGNOSTICS="${STEP_DIAGNOSTICS:-0}"
STEP_DIAG_LIMIT="${STEP_DIAG_LIMIT:-1}"
INCLUDE_EXTERNAL="${INCLUDE_EXTERNAL:-0}"
RL4DVRPTW_ROOT="${RL4DVRPTW_ROOT:-/home/admin_wsl/projects/RL4DVRPTW}"
RUN_OR_TOOLS="${RUN_OR_TOOLS:-0}"
RUN_LKH="${RUN_LKH:-0}"
ORTOOLS_TIME_LIMIT_MS="${ORTOOLS_TIME_LIMIT_MS:-10000}"
LKH_PARALLEL="${LKH_PARALLEL:-1}"
LKH_INSTANCE_WORKERS="${LKH_INSTANCE_WORKERS:-1}"
MODEL_LIST="${MODEL_LIST:-vectra,mardam,b0,b1,b3,b5,edgeoff,am,polynet,ortools,lkh3}"
FILE_GLOB="${FILE_GLOB:-**/*.csv}"
MAX_FILES="${MAX_FILES:-0}"

DATASETS_ROOT_ABS="$(realpath "$DATASETS_ROOT")"
OUTPUT_ROOT_ABS="$(mkdir -p "$OUTPUT_ROOT" && realpath "$OUTPUT_ROOT")"

COMMON_ARGS=(
  --datasets-root "$DATASETS_ROOT"
  --file-glob "$FILE_GLOB"
  --max-files "$MAX_FILES"
  --vehicles-count 3
  --veh-capa 200
  --veh-speed 1
  --max-print-instances 1
  --verify-rollouts 1
  --greedy
)

if [[ "$VERIFY_ROUTES" == "0" ]]; then
  COMMON_ARGS+=(--no-verify-routes)
fi

VECTRA_EXTRA_ARGS=()
if [[ "$STEP_DIAGNOSTICS" == "1" ]]; then
  VECTRA_EXTRA_ARGS+=(--extra-arg=--save-step-diagnostics)
  VECTRA_EXTRA_ARGS+=(--extra-arg=--step-diagnostics-limit)
  VECTRA_EXTRA_ARGS+=(--extra-arg "$STEP_DIAG_LIMIT")
fi

SUMMARY_SPECS=()
EXTERNAL_SUMMARY_SPECS=()

should_run() {
  local name="$1"
  [[ ",$MODEL_LIST," == *",$name,"* ]]
}

run_vectra_like() {
  local name="$1"
  local config="$2"
  local weight="$3"

  if ! should_run "$name"; then
    return
  fi
  if [[ ! -f "$config" || ! -f "$weight" ]]; then
    echo "[SKIP] $name: missing config or weight"
    return
  fi

  local out_dir="$OUTPUT_ROOT_ABS/$name"
  echo "============================================================"
  echo "[RUN] $name -> $out_dir"
  echo "============================================================"
  PYTHONPATH=. "$PYTHON_BIN" script/infer_all_datasets.py \
    "${COMMON_ARGS[@]}" \
    --config-file "$config" \
    --model-weight "$weight" \
    --output-dir "$out_dir" \
    "${VECTRA_EXTRA_ARGS[@]}"

  SUMMARY_SPECS+=(--summary-csv "$name=$out_dir/summary.csv")
}

run_mardam() {
  local name="mardam"
  local config="data/mardam/args.json"
  local weight="data/mardam/chkpt_best.pyth"

  if ! should_run "$name"; then
    return
  fi
  if [[ ! -f "$config" || ! -f "$weight" ]]; then
    echo "[SKIP] $name: missing config or weight"
    return
  fi

  local out_dir="$OUTPUT_ROOT_ABS/$name"
  echo "============================================================"
  echo "[RUN] $name -> $out_dir"
  echo "============================================================"
  PYTHONPATH=. "$PYTHON_BIN" script/infer_all_datasets_mardam.py \
    "${COMMON_ARGS[@]}" \
    --config-file "$config" \
    --model-weight "$weight" \
    --output-dir "$out_dir"

  SUMMARY_SPECS+=(--summary-csv "$name=$out_dir/summary.csv")
}

run_external_model() {
  local name="$1"
  local model="$2"
  local args_file="$3"
  local weight="$4"

  if ! should_run "$name"; then
    return
  fi
  if [[ "$INCLUDE_EXTERNAL" != "1" ]]; then
    return
  fi
  if [[ ! -d "$RL4DVRPTW_ROOT" ]]; then
    echo "[SKIP] $name: RL4DVRPTW_ROOT not found: $RL4DVRPTW_ROOT"
    return
  fi
  if [[ ! -f "$RL4DVRPTW_ROOT/$args_file" || ! -f "$RL4DVRPTW_ROOT/$weight" ]]; then
    echo "[SKIP] $name: missing args or weight under $RL4DVRPTW_ROOT"
    return
  fi

  local out_dir="$OUTPUT_ROOT_ABS/$name"
  echo "============================================================"
  echo "[RUN] external $name -> $out_dir"
  echo "============================================================"
  (
    cd "$RL4DVRPTW_ROOT"
    PYTHONPATH=. "$PYTHON_BIN" infer_batch.py \
      --model "$model" \
      --model-weight "$weight" \
      --model-args "$args_file" \
      --csv-dir "$DATASETS_ROOT_ABS" \
      --output-dir "$out_dir" \
      --csv-output "$out_dir/aggregated.csv" \
      --problem-type dvrptw \
      --vehicles-count 3 \
      --veh-capa 200 \
      --veh-speed 1 \
      --greedy
  )
  EXTERNAL_SUMMARY_SPECS+=(--external-summary "$name=$out_dir/aggregated.csv")
}

run_ortools() {
  if [[ "$RUN_OR_TOOLS" != "1" ]]; then
    return
  fi
  local name="ortools"
  if ! should_run "$name"; then
    return
  fi
  local out_dir="$OUTPUT_ROOT_ABS/$name"
  echo "============================================================"
  echo "[RUN] $name -> $out_dir"
  echo "============================================================"
  PYTHONPATH=. "$PYTHON_BIN" script/batch_infer_ort.py \
    --datasets-root "$DATASETS_ROOT_ABS" \
    --vehicles-count 3 \
    --veh-capa 200 \
    --veh-speed 1 \
    --time-limit-ms "$ORTOOLS_TIME_LIMIT_MS" \
    --output-dir "$out_dir" \
    --greedy
  SUMMARY_SPECS+=(--summary-csv "$name=$out_dir/summary.csv")
}

run_lkh() {
  if [[ "$RUN_LKH" != "1" ]]; then
    return
  fi
  local name="lkh3"
  if ! should_run "$name"; then
    return
  fi
  local out_dir="$OUTPUT_ROOT_ABS/$name"
  echo "============================================================"
  echo "[RUN] $name -> $out_dir"
  echo "============================================================"
  PYTHONPATH=. "$PYTHON_BIN" script/eval_lkh_all_datasets.py \
    --datasets-root "$DATASETS_ROOT_ABS" \
    --vehicles-count 3 \
    --veh-capa 200 \
    --veh-speed 1 \
    --parallel "$LKH_PARALLEL" \
    --instance-workers "$LKH_INSTANCE_WORKERS" \
    --output-dir "$out_dir" \
    --greedy
  SUMMARY_SPECS+=(--summary-csv "$name=$out_dir/summary.csv")
}

run_vectra_like "vectra" "data/vectra/args.json" "data/vectra/chkpt_best.pyth"
run_mardam
run_vectra_like "b0" "data/_Ablation/B0/args.json" "data/_Ablation/B0/chkpt_best.pyth"
run_vectra_like "b1" "data/_Ablation/B1/args.json" "data/_Ablation/B1/chkpt_best.pyth"
run_vectra_like "b3" "data/_Ablation/B3/args.json" "data/_Ablation/B3/chkpt_best.pyth"
run_vectra_like "b5" "data/_Ablation/B5/args.json" "data/_Ablation/B5/chkpt_best.pyth"
run_vectra_like "edgeoff" "data/_Ablation/Edgeoff/args.json" "data/_Ablation/Edgeoff/chkpt_best.pyth"
run_external_model "am" "am" "data/_AM/args.json" "data/_AM/chkpt_best.pyth"
run_external_model "polynet" "polynet" "data/_PolyNet/args.json" "data/_PolyNet/chkpt_best.pyth"
run_ortools
run_lkh

if [[ ${#SUMMARY_SPECS[@]} -eq 0 && ${#EXTERNAL_SUMMARY_SPECS[@]} -eq 0 ]]; then
  echo "No summaries were generated."
  exit 1
fi

PYTHONPATH=. "$PYTHON_BIN" script/build_experimental_report.py \
  --results-root "$OUTPUT_ROOT_ABS" \
  --output-dir "$OUTPUT_ROOT_ABS/paper_ready" \
  "${SUMMARY_SPECS[@]}" \
  "${EXTERNAL_SUMMARY_SPECS[@]}"

echo "Done. Paper-ready artifacts: $OUTPUT_ROOT_ABS/paper_ready"
