# COAST: Coordinated Online Assignment with Structured Trajectory Signals

> Anonymous AAAI submission — code supplement

This repository implements **COAST**, a reinforcement-learning policy for the
Dynamic Vehicle Routing Problem with Time Windows (DVRPTW).  The implementation
class is named `VECTRA` for historical reasons and is defined in
`MODEL/model/vectra.py`.

COAST makes one event-driven dispatch decision at a time.  It combines three
candidate-level signals: edge-aware vehicle--customer compatibility, a
fleet-relative ownership signal derived from per-vehicle memory, and a learned
candidate-conditioned lookahead signal.  All three are optimized end-to-end
with policy-gradient feedback; neither ownership nor lookahead has an auxiliary
supervised objective.

## Repository contents

| Path | Purpose |
|---|---|
| `MODEL/` | COAST/VECTRA training and inference entry points |
| `layers/` | Attention, encoder, feature, fusion, and loss layers |
| `problems/` | DVRPTW data generators and event-driven environments |
| `baselines/` | Critic, rollout, and heuristic baselines |
| `script/` | Training, batch evaluation, data generation, and analysis commands |
| `data/test_sets/` | Fixed in-distribution and OOD DVRPTW test sets |
| `datasets/` | Solomon-style, scale, and dynamic-grid benchmark instances |
| `output/` | Local experiment artefacts, including reference checkpoints in this workspace |
| `paper_results/` | Paper-facing CSV exports, tables, and figure-generation code |

## Environment

The core implementation requires Python, PyTorch, SciPy, Matplotlib, and
TQDM.  Batch reporting additionally uses NumPy, Pandas, and OpenPyXL.
OR-Tools is required only for the OR-Tools baseline; LKH is required only for
the LKH baseline.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a PyTorch build compatible with the available CPU or CUDA runtime.  CPU
execution is supported by adding `--no-cuda` to Python entry points; it is much
slower than GPU inference and training.  The pinned PyTorch version was
validated with CUDA 12.4; replace it with the matching build for another CUDA
runtime or for CPU-only execution.

## Quick verification

Run a small forward/backward smoke test from the repository root:

```bash
PYTHONPATH=. python test/model_test.py
```

The test constructs synthetic VRP and VRPTW instances, executes the VECTRA
policy, and checks that a policy-gradient backward pass completes.

## Reproducing reference COAST inference

The following command evaluates the reference COAST checkpoint on the fixed
500-instance in-distribution test set currently included in this workspace.
The `args.json` file must always be paired with its checkpoint: it records the
architecture and ablation settings used to train that model.

```bash
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --config-file output/Model_DVRPTWn50m3_260311-0727/args.json \
  --model-weight output/Model_DVRPTWn50m3_260311-0727/chkpt_best.pyth \
  --data-file data/test_sets/test_dvrptw_id_n50m3_500.pyth \
  --greedy \
  --max-print-instances 1 \
  --save-json reproduced/coast_id_n50m3.json
```

Add `--no-cuda` when CUDA is unavailable.  The JSON output contains the
selected routes, costs, and route-replay verification results.

## Reproducing the internal OOD comparison

The repository contains fixed ID/OOD instances and checkpoints for COAST,
MARDAM, and the internal ablations.  The command below deliberately excludes
external models that are not redistributed with this repository.

```bash
PYTHONPATH=. python script/run_ood_experiments.py \
  --models vectra,mardam,b0,b1,b3,b5,edgeoff,no_ownership,no_lookahead \
  --datasets-dir data/test_sets \
  --output-dir reproduced/ood_eval \
  --max-print-instances 1
```

The resulting `reproduced/ood_eval/ood_summary.csv` can be compared with the
paper-facing CSV files in `paper_results/csv_exports/`.

## Training

Train the full COAST configuration with the default n50m3 DVRPTW schedule:

```bash
bash script/train_vectra_main.sh
```

To train the paper's internal variants across the configured random seeds:

```bash
bash script/train_all_seeds.sh
```

Both commands generate new timestamped output directories.  Training is
stochastic; the supplied `chkpt_best.pyth` files are the reference artefacts
for result reproduction, while retraining is intended to reproduce the
protocol and performance trend rather than byte-identical weights.

## Benchmarks and reference artefacts

| Evaluation setting | Included input assets |
|---|---|
| In-distribution and OOD DVRPTW | `data/test_sets/` |
| Solomon-style benchmark | `datasets/100/`, `datasets/h200/`, `datasets/h400/` |
| Scale benchmark | `datasets/dvrptw_n{20,50,100,200,400}m*_10240.pyth` |
| Dynamic sensitivity grid | `datasets/dvrptw_dynamic_grid/` |
| Tables and figures | `paper_results/` |

The reference checkpoints in the current workspace are stored under `output/`:

| Model | Checkpoint directory |
|---|---|
| COAST | `output/Model_DVRPTWn50m3_260311-0727/` |
| MARDAM | `output/Mardam_DVRPTWn50m3_260315-1328/` |
| Ablations B0, B1, B3, B5, EdgeOff | `output/ablation/<profile>/seed42/` |
| No-ownership / no-lookahead | `output/ablation/no_ownership/`, `output/ablation/no_lookahead/` |

For each directory, use only the matched `args.json` and `chkpt_best.pyth`
pair.  Other files in `output/` are intermediate training logs or checkpoints
and are not needed for inference.

## External baselines

AM and PolyNet are evaluated through a separate external codebase specified by
`RL4DVRPTW_ROOT`; their implementation and weights are not bundled here.  The
reported summary values are retained in `paper_results/csv_exports/` for
transparency.  LKH is supported through `externals/_lkh.py` when a compatible
LKH executable is available.  OR-Tools runs require the optional `ortools`
package.

## Reproducibility notes

- Run commands from the repository root with `PYTHONPATH=.`.
- Preserve the checkpoint/configuration pairing; loading a checkpoint with CLI
  defaults can silently instantiate a different architecture or ablation.
- Fixed evaluation sets should be used for numerical comparisons.  They can be
  regenerated by `script/generate_ood_sets.py`, but the included files are the
  reference inputs.
- `paper_results/csv_exports/` records the paper-facing aggregates.  Fresh
  runs should be written to a new directory such as `reproduced/`, never over
  the reference artefacts.

## Anonymous release note

This repository is prepared as an anonymous supplementary code submission.
It intentionally contains no author names, affiliations, URLs, or contact
details.  Citation metadata and a software license will be added only in the
camera-ready release, as permitted by the review process.
