# AGENTS.md — COAST / VECTRA for Dynamic VRPTW

> **Paper name:** COAST — **Code name:** VECTRA (`MODEL/model/vectra.py`, class `VECTRA`)  
> Reinforcement learning for Dynamic Vehicle Routing Problem with Time Windows.  
> Full architecture & math: [README.md](README.md) — Quickstart: [paper/IMPLEMENTATION_GUIDE.md](paper/IMPLEMENTATION_GUIDE.md)

## Project Conventions

| Aspect | Rule | Example |
|--------|------|---------|
| **Functions/variables** | `snake_case` | `cust_count`, `reinforce_loss()` |
| **Classes** | `CamelCase` | `VECTRA`, `GraphEncoder`, `DVRPTW_Environment` |
| **Internal modules** | Leading `_` in filename | `_mha.py`, `_args.py`, `_data_dtw.py` |
| **Imports** | Star imports via `__init__.py` re-export chains | `from MODEL.model import *`, `from problems import *` |
| **Comments** | Mix of English & Vietnamese; block headers with `# ──` | `# ── Problem ─────────` |
| **Tensor shapes** | Documented in docstrings as `N × L_c × D` | batch × nodes × dim |
| **Model weights** | Saved as `.pyth` files (PyTorch state dicts) | `chkpt.pyth`, `model_weight.pyth` |

## Key Commands

```bash
# Train
python MODEL/train.py --problem-type dvrptw --customers-count 50 --epoch-count 500
bash script/train_vectra_main.sh

# PPO training
python MODEL/train_PPO.py --model-size 128 --ppo-epochs 4

# Inference
python MODEL/infer.py --model-weight path/to/chkpt.pyth --data-file data.pyth

# Evaluation (multi-model sweep)
python script/eval_unified.py --test-data test.pyth --models-dir output/

# Generate data
python -c "from problems import DVRPTW_Dataset; ds = DVRPTW_Dataset.generate(batch_size=1000, cust_count=50); torch.save(ds, 'test.pyth')"
```

## Architecture at a Glance

```
DATA → DVRPTW_Environment → VECTRA model (forward) → Baseline (trajectory) → reinforce_loss → backward
```

**VECTRA model pipeline:**

1. **Customer Encoding** (`GraphEncoder`): Self-attention over customer nodes with RBF distance bias
2. **Vehicle Encoding** (`FleetEncoder`): Cross-attention from acting vehicle to customer representations, masked by feasibility
3. **Edge Features** (`EdgeFeatureEncoder`): 8D per (vehicle, customer) pair (distance, travel_time, arrival, wait, late, slack, feasible, cap_gap)
4. **Three signal heads** → MLP fusion → action logits:
   - **Attention**: vehicle × customer compatibility
   - **Ownership** (`OwnershipHead`): Memory-based soft assignment to reduce fleet overlap
   - **Lookahead** (`LookaheadHead`): Per-candidate future-impact scalar — *learned end-to-end, no auxiliary loss*

## Critical Pitfalls

- **CUDA device asserts**: If a CUDA assert fires, the context is invalidated. The training loop skips that checkpoint to avoid corruption. See `_is_cuda_device_assert_error()` in `MODEL/train.py`.
- **AMP GradScaler**: Hardcoded to `'cuda'` device string in `train.py`. CPU-only training uses `--no-cuda`.
- **PyTorch version compat**: Tries `torch.amp` (2.0+) with fallback to `torch.cuda.amp` (1.x).
- **Ablation profiles** (`utils/_args.py` → `_apply_ablation_profile()`): 16 profiles (`vectra`, `b0`–`b5`, `edgeoff`, `no_ownership`, `no_lookahead`, `noownership`, `nolookahead`, `a0`–`a4`, `a9`, `none`) toggle feature flags. Must use the same profile for training and inference.
- **Model weight loading** (`utils/_chkpt.py`): Auto-detects `args.json` sibling file to reconstruct config. If missing, falls back to current CLI defaults — may silently misconfigure.
- **No auxiliary loss**: `LookaheadHead` and `OwnershipHead` have no supervised targets. They're trained solely through policy gradient.
- **Checkpoint pruning**: Only last 5 checkpoints kept by default (`CHECKPOINT_PERIOD = 5`).

## Module Map

| Directory | Purpose | Key files |
|-----------|---------|-----------|
| `MODEL/model/` | Neural architecture | `vectra.py` (VECTRA class) |
| `MODEL/` | Training & inference | `train.py`, `train_PPO.py`, `infer.py`, `infer_mardam.py` |
| `layers/` | NN building blocks | `Mymodel_layers.py` (GraphEncoder, FleetEncoder, heads), `_mha.py`, `_loss.py` |
| `problems/` | Environments & data | `_env_dtw.py` (DVRPTW_Environment), `_data_dtw.py` (DVRPTW_Dataset) |
| `baselines/` | Baseline policies | `_base.py` (Baseline wrapper), `_rollout.py`, `_critic.py`, `_near_nb.py` |
| `utils/` | Config & helpers | `_args.py` (CLI + ablation profiles), `_chkpt.py` (checkpoint mgmt) |
| `externals/` | Classical solvers | `_lkh.py` (LKH wrapper), `_ort.py` (OR-Tools wrapper) |
| `script/` | Evaluation & batch | `eval_unified.py`, `infer_all_datasets.py`, `*.sh` wrappers |
| `data/` | Datasets & pretrained weights | `.pyth` files, `vectra/`, `mardam/`, `_Ablation/` |

## Documentation

- **[README.md](README.md)** — Full architecture, math formulas, API reference
- **[paper/IMPLEMENTATION_GUIDE.md](paper/IMPLEMENTATION_GUIDE.md)** — Step-by-step from setup to multi-seed evaluation
- **[paper/EXPERIMENTAL_PROTOCOL.md](paper/EXPERIMENTAL_PROTOCOL.md)** — Research hypotheses (H1–H4) and current experiment status
- **[paper/COAST.pdf](paper/COAST.pdf)** — Research paper
- **[paper/review.md](paper/review.md)** — Reviewer feedback & planned improvements
