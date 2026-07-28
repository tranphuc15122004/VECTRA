# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**COAST** (paper name) / **VECTRA** (code name) — a reinforcement learning policy for **Dynamic Vehicle Routing Problem with Time Windows (DVRPTW)**. The novel contribution is decomposing sequential routing decisions into three parallel learned signals fused by an MLP:

- **Edge-aware compatibility** (`CrossEdgeFusion`): attention score capturing vehicle–customer compatibility under distance, time window, and capacity constraints.
- **Coordination memory / Ownership** (`CoordinationMemory` + `OwnershipHead`): per-vehicle hidden state creates soft assignment bias to reduce fleet overlap.
- **Candidate-conditioned lookahead** (`LookaheadHead`): per-candidate scalar learned end-to-end via policy gradient — **no auxiliary supervised loss**.

Baseline model is `AttentionLearner` (`_learner.py`, paper name: MARDAM) — standard attention without edge features, memory, ownership, or lookahead.

Full architecture math: `README.md`. Step-by-step experimental procedures: `paper/EXPERIMENT_RUNBOOK.md`.

---

## Commands

All commands require `PYTHONPATH=.` from the repo root.

### Training

```bash
# COAST/VECTRA (REINFORCE + critic baseline)
PYTHONPATH=. python MODEL/train.py \
  --problem-type dvrptw --customers-count 50 --vehicles-count 3 \
  --epoch-count 500 --iter-count 1000 --batch-size 512 \
  --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
  --baseline-type critic --amp --output-dir output/vectra_run

# Via shell wrapper (configurable with env vars)
bash script/train_vectra_main.sh

# PPO variant
PYTHONPATH=. python MODEL/train_PPO.py --problem-type dvrptw --ppo-epochs 4

# Ablation (profiles: vectra, b0, b1, b3, b5, edgeoff, no_ownership, no_lookahead)
PYTHONPATH=. python MODEL/train.py --problem-type dvrptw --ablation-profile b3 \
  --output-dir output/ablation_b3

# Baseline AttentionLearner
python script/train_mardam.py  # or bash script/train_mardam.sh
```

### Inference

```bash
# Single model, greedy
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --model-weight data/vectra/chkpt_best.pyth \
  --config-file data/vectra/args.json \
  --greedy --save-json output/infer_vectra.json

# From CSV file (columns: x,y,demand,open,close,servicetime[,time])
PYTHONPATH=. python MODEL/infer.py \
  --data-csv data/datasets/100/h100c101.csv \
  --vehicles-count 3 --veh-capa 200 --veh-speed 1 \
  --model-weight data/vectra/chkpt_best.pyth \
  --config-file data/vectra/args.json --greedy

# With step-by-step diagnostics
PYTHONPATH=. python MODEL/infer.py ... --save-step-diagnostics --step-diagnostics-limit 1

# Batch inference over CSV datasets
PYTHONPATH=. python script/infer_all_datasets.py \
  --datasets-root data/datasets --model-weight data/vectra/chkpt_best.pyth \
  --config-file data/vectra/args.json --vehicles-count 3 --veh-capa 200 \
  --veh-speed 1 --output-dir output/batch_infer --greedy
```

### Evaluation

```bash
# Multi-model evaluation sweep
PYTHONPATH=. python script/eval_unified.py \
  --test-data data/dvrptw_n50m3_test.pyth \
  --models-dir output/ablation --output output/eval_results/in_dist.json \
  --seeds 42,123,456,789,1024

# Full dynamic benchmark matrix (COAST vs MARDAM vs ablations)
DATASETS_ROOT=data/datasets/dvrptw_dynamic_grid \
OUTPUT_ROOT=output/dynamic_benchmark_raw \
bash script/run_dynamic_experiment_matrix.sh

# OOD evaluation
PYTHONPATH=. python script/generate_ood_sets.py --output-dir data/test_sets
PYTHONPATH=. python script/run_ood_experiments.py \
  --datasets-dir data/test_sets --output-dir output/ood_eval \
  --models vectra,mardam,b0,b1,b3,b5,edgeoff

# Hypothesis tables (H1–H4)
bash script/run_hypothesis_experiments.sh
```

### Data Generation

```bash
python -c "
from problems import DVRPTW_Dataset
import torch
ds = DVRPTW_Dataset.generate(batch_size=1000, cust_count=50)
torch.save(ds, 'data/dvrptw_n50m3_test.pyth')
"
```

---

## Architecture

### VECTRA Forward Pass

```
DVRPTW_Environment
  → GraphEncoder (self-attn + RBF distance bias over customers) → cust_repr (N × L_c × D)
  → FleetEncoder (cross-attn: acting vehicle ← cust_repr) → veh_repr (N × 1 × D)
  → EdgeFeatureEncoder (8D per vehicle–customer pair) → edge_emb (N × 1 × L_c × D)
  ↓ Three parallel heads:
  →  CrossEdgeFusion (attention + edge bias) → s_att
  →  LookaheadHead (MLP([veh_repr, cust_j, edge_j])) → s_look
  →  OwnershipHead (softmax over fleet → gather current vehicle → log) → s_owner
  → Z-normalize each signal over valid candidates
  → MLP fusion (3→64→1) or linear weighted sum → compat
  → mask + softmax → greedy argmax / multinomial sample → j*
  → CoordinationMemory update: tanh(W_in[veh_repr, cust_j*, edge_j*] + W_hid·m(t))
```

The environment re-encodes customers when new orders become visible (`dyna.new_customers=True`). Coordination between vehicles comes from `CoordinationMemory`/`OwnershipHead`, not from direct vehicle–vehicle attention.

### Key Module Locations

| Layer/Component | File |
|---|---|
| `VECTRA` main class | `MODEL/model/vectra.py` |
| `GraphEncoder`, `FleetEncoder`, `CrossEdgeFusion`, `CoordinationMemory`, `OwnershipHead`, `LookaheadHead`, `EdgeFeatureEncoder` | `layers/Mymodel_layers.py` |
| Multi-head attention | `layers/_mha.py` |
| `reinforce_loss()` | `layers/_loss.py` |
| `DVRPTW_Environment` | `problems/_env_dtw.py` |
| `DVRPTW_Dataset` | `problems/_data_dtw.py` |
| CLI args + ablation profiles | `utils/_args.py` |
| Checkpoint saving | `utils/_chkpt.py` |
| Model weight / config loading | `utils/_args.py` (`--config-file`), `MODEL/infer.py` |
| `AttentionLearner` (old baseline) | `_learner.py` |

### Ablation Profiles

Defined in `utils/_args.py` → `_apply_ablation_profile()`:

| Profile | Edge | Memory | Ownership | Lookahead | Fusion |
|---|---|---|---|---|---|
| `vectra` | on | on | on | on | MLP |
| `b0` | on | off | off | off | MLP |
| `b1` | on | on | off | off | MLP |
| `b3` | on | off | off | on | MLP |
| `b5` | on | on | on | on | linear |
| `edgeoff` | off | on | on | on | MLP |
| `no_ownership` | on | on | off | on | MLP |
| `no_lookahead` | on | on | on | off | MLP |

**Must use the same profile for both training and inference.**

### Data & Checkpoints

- Model weights: `.pyth` files (PyTorch state dicts)
- Each checkpoint directory contains `args.json` alongside the `.pyth` file — loading without `args.json` silently falls back to CLI defaults and may misconfigure the model
- Pretrained weights: `data/vectra/chkpt_best.pyth`, `data/mardam/`, `data/_Ablation/{b0,b1,b3,b5,edgeoff}/`
- Training saves `chkpt_best.pyth`, periodic `chkpt_ep*.pyth`, `args.json`, and `train_statistics.csv`; only the last 5 periodic checkpoints are kept

---

## Critical Pitfalls

- **CUDA device asserts**: If a CUDA assert fires mid-training, the GPU context is invalidated. `MODEL/train.py` catches this (`_is_cuda_device_assert_error()`), logs, and skips that batch to avoid checkpoint corruption.
- **AMP GradScaler**: Hardcoded to `'cuda'` device string. Use `--no-cuda` for CPU-only runs.
- **PyTorch compatibility**: Code tries `torch.amp` (≥2.0) then falls back to `torch.cuda.amp` (1.x).
- **All-masked action**: If all candidates are masked, `_get_logp()` reopens depot index `0` to avoid NaN; this is intentional fallback behavior.
- **Score fusion input**: The fusion MLP receives Z-normalized scores from `(s_att, s_owner, s_look)`. OwnershipHead uses only `cust_repr` (not `edge_emb`) for its logits.

---

## Conventions

- Functions/variables: `snake_case`; classes: `CamelCase`; internal modules: leading `_` (e.g. `_mha.py`, `_args.py`)
- Star imports via `__init__.py` re-export chains: `from MODEL.model import *`, `from problems import *`
- Tensor shapes documented as `N × L_c × D` (batch × nodes × dim)
- Comments mix English and Vietnamese; section headers use `# ──` style
