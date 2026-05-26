# COAST/VECTRA Experimental Runbook

This runbook describes how to regenerate the paper-facing experimental artifacts without silently claiming results that were not run.

## 1. Merge Existing Dynamic Results

Use this first. It reuses raw VECTRA and MARDAM summaries already present under `dynamic_benchmark/{vectra,mardam}/dod_*/tw_*/seed_*/summary.csv`.

```bash
PYTHONPATH=. python script/build_experimental_report.py \
  --discover-nested \
  --results-root "Experimental result-20260526T064624Z-3-001/Experimental result/dynamic_benchmark"
```

Paper-facing outputs:

```text
Experimental result-20260526T064624Z-3-001/Experimental result/dynamic_benchmark/paper_ready/
├── manifest.json
├── master_summary.csv
├── cell_summary.csv
├── detail_metrics.csv
├── significance.csv
└── paper_statistics.md
```

Use `significance.csv` for paired VECTRA-vs-MARDAM statistics once it reports `n_pairs=512`.

## 2. Rerun Verified Dynamic Grid

Run local VECTRA, MARDAM, and internal ablations on the same dynamic grid:

```bash
DATASETS_ROOT=/path/to/dvrptw_dynamic_grid \
OUTPUT_ROOT=output/dynamic_benchmark_verified \
VERIFY_ROUTES=1 \
bash script/run_dynamic_experiment_matrix.sh
```

Add AM and PolyNet from the external repo:

```bash
DATASETS_ROOT=/path/to/dvrptw_dynamic_grid \
OUTPUT_ROOT=output/dynamic_benchmark_verified \
VERIFY_ROUTES=1 \
INCLUDE_EXTERNAL=1 \
RL4DVRPTW_ROOT=/home/admin_wsl/projects/RL4DVRPTW \
bash script/run_dynamic_experiment_matrix.sh
```

Optional OR/LKH baselines:

```bash
RUN_OR_TOOLS=1 ORTOOLS_TIME_LIMIT_MS=10000 bash script/run_dynamic_experiment_matrix.sh
RUN_LKH=1 LKH_PARALLEL=1 LKH_INSTANCE_WORKERS=1 bash script/run_dynamic_experiment_matrix.sh
```

Do not mention OR/LKH in the paper unless their rows appear in `paper_ready/master_summary.csv` and `significance.csv`.

## 3. Generate And Evaluate OOD Sets

Smoke test:

```bash
PYTHONPATH=. python script/generate_ood_sets.py \
  --output-dir data/test_sets_smoke \
  --smoke-size 8

PYTHONPATH=. python script/run_ood_experiments.py \
  --datasets-dir data/test_sets_smoke \
  --output-dir output/ood_eval_smoke \
  --models vectra,mardam \
  --device-cpu
```

Full run:

```bash
PYTHONPATH=. python script/generate_ood_sets.py \
  --output-dir data/test_sets \
  --batch-size 500

PYTHONPATH=. python script/run_ood_experiments.py \
  --datasets-dir data/test_sets \
  --output-dir output/ood_eval \
  --models vectra,mardam,b0,b1,b3,b5,edgeoff
```

Outputs:

```text
output/ood_eval/
├── ood_summary.csv
├── ood_report.md
└── ood_manifest.json
```

Embed OOD in the main report:

```bash
PYTHONPATH=. python script/build_experimental_report.py \
  --discover-nested \
  --results-root "Experimental result-20260526T064624Z-3-001/Experimental result/dynamic_benchmark" \
  --ood-summary output/ood_eval/ood_summary.csv
```

## 4. Behavioral Diagnostics

Run inference with step diagnostics for a small, representative subset:

```bash
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --config-file data/vectra/args.json \
  --model-weight data/vectra/chkpt_best.pyth \
  --data-csv /path/to/instance.csv \
  --greedy \
  --save-json output/diagnostics/vectra_instance.infer.json \
  --save-step-diagnostics \
  --step-diagnostics-limit 1
```

Analyze diagnostics:

```bash
PYTHONPATH=. python script/analyze_behavior_diagnostics.py \
  --input-root output/diagnostics \
  --output-dir output/behavior_analysis
```

Outputs:

```text
output/behavior_analysis/
├── behavior_summary.csv
└── behavior_report.md
```

Embed behavior in the main report:

```bash
PYTHONPATH=. python script/build_experimental_report.py \
  --discover-nested \
  --results-root "Experimental result-20260526T064624Z-3-001/Experimental result/dynamic_benchmark" \
  --behavior-summary output/behavior_analysis/behavior_summary.csv
```

## 5. Hypothesis Tables H1-H4

After dynamic, OOD, and behavior artifacts exist, build the paper-facing hypothesis tables:

```bash
PYTHONPATH=. python script/build_hypothesis_tables.py \
  --master-summary output/dynamic_benchmark_verified/paper_ready/master_summary.csv \
  --ood-summary output/ood_eval/ood_summary.csv \
  --behavior-summary output/behavior_analysis/hypothesis_behavior_summary.csv \
  --output-dir output/hypothesis_tables
```

Outputs:

```text
output/hypothesis_tables/
├── hypothesis_summary.csv
└── hypothesis_summary.md
```

The H1/H2 rows are single-checkpoint evidence unless new `no_ownership` or `no_lookahead` checkpoints are trained. Use cautious wording for those claims.

For faster diagnostics on a representative subset:

```bash
MODEL_LIST=vectra,b0,b1,b3,b5,edgeoff \
FILE_GLOB='**/instance_0000[0-7].csv' \
STEP_DIAGNOSTICS=1 \
STEP_DIAG_LIMIT=1 \
VERIFY_ROUTES=1 \
DATASETS_ROOT=/path/to/dvrptw_dynamic_grid \
OUTPUT_ROOT=output/dynamic_diagnostics_subset \
bash script/run_dynamic_experiment_matrix.sh
```

## 6. Paper Table Policy

Use these artifacts in the paper:

- Main dynamic benchmark: `paper_ready/cell_summary.csv`
- Travel/skipped/late details: `paper_ready/detail_metrics.csv`
- Paired tests: `paper_ready/significance.csv`
- OOD: `output/ood_eval/ood_summary.csv`
- Behavior: `output/behavior_analysis/behavior_summary.csv`
- Hypothesis tables: `output/hypothesis_tables/hypothesis_summary.csv`

Avoid these until regenerated:

- `experimental_results_csv/chi_tiết_benchmark.csv`, because its metric cells are currently empty.
- Empty `LKH3` and `Ortools` columns in `experimental_results_csv/Benchmark.csv`.
- Any claim involving mechanisms that are not implemented in the current COAST/VECTRA model.
