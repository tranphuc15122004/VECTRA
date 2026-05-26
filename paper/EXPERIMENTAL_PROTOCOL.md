# COAST Experimental Protocol: Hypothesis-Driven Validation

## *(Updated — May 2026 — Implementation-Ready)*

**Purpose:** This document provides a complete, executable experimental plan to validate the core hypotheses of COAST's structured decomposition approach for DVRPTW, targeting A* conference submission (AAAI/IJCAI/AAMAS).

**Core Hypotheses:**

- **H1:** Per-vehicle coordination memory + latent ownership reduces inter-vehicle conflicts vs. memory-only and no-memory baselines.
- **H2:** Candidate-conditioned lookahead reduces greedy myopia and improves isolated-customer handling.
- **H3:** Edge-aware scoring (8-feature edge encoder + CrossEdgeFusion) improves feasibility under tight time-window and capacity constraints.
- **H4:** Structured multi-signal decomposition with non-linear (MLP) fusion generalizes better than linear fusion and monolithic baselines under distribution shift.

**Inference-only execution path:** For the current paper-strength-without-retraining setting, use existing checkpoints and build H1-H4 tables with:

```bash
PYTHONPATH=. python script/build_hypothesis_tables.py \
  --master-summary output/dynamic_benchmark_verified/paper_ready/master_summary.csv \
  --ood-summary output/ood_eval/ood_summary.csv \
  --behavior-summary output/behavior_analysis/hypothesis_behavior_summary.csv \
  --output-dir output/hypothesis_tables
```

This produces `hypothesis_summary.csv` and `hypothesis_summary.md`. Because this path uses single checkpoints, do not claim multi-seed robustness unless the multi-seed training phase below is completed.

---

### ✅ Current Status Summary

| Item | Status | Detail |
|------|--------|--------|
| **COAST (full model)** | ✅ Trained (1 seed) | `data/vectra/chkpt_best.pyth`, `layer=2, head=4, ff=256, model_size=128` |
| **B0-None** | ✅ Trained (1 seed) | `data/_Ablation/B0/`, same backbone |
| **B1-Memory** | ✅ Trained (1 seed) | `data/_Ablation/B1/`, same backbone |
| **B3-Look** | ✅ Trained (1 seed) | `data/_Ablation/B3/`, same backbone |
| **B5-Linear** | ✅ Trained (1 seed) | `data/_Ablation/B5/`, same backbone |
| **B-EdgeOff** | ✅ Trained (1 seed) | `data/_Ablation/Edgeoff/`, same backbone |
| **AM (Attention Model)** | ✅ Trained (1 seed) | `/home/admin_wsl/projects/RL4DVRPTW/data/_AM/`, `layer=5, head=8, ff=256` |
| **PolyNet** | ✅ Trained (1 seed) | `/home/admin_wsl/projects/RL4DVRPTW/data/_PolyNet/`, `layer=5, head=8, ff=256` |
| **MARDAM** | 🔧 Available | `/home/admin_wsl/projects/RL4DVRPTW/mdam/` (needs training) |
| **Classical baselines** | ❌ Not yet | OR-Tools, Greedy NN need implementation |
| **Multi-seed** | ❌ Not yet | All models currently 1 seed only |
| **OOD evaluation** | ❌ Not yet | No out-of-distribution tests run |
| **Behavioral analysis** | ❌ Not yet | No ownership/lookahead visualizations |

> **Critical note on fairness:** COAST and ALL internal ablations (B0, B1, B3, B5, EdgeOff) share the **identical backbone**: `layer=2, head=4, ff=256, model_size=128`. This is a fair comparison. AM and PolyNet use `layer=5, head=8` — they are **stronger-capacity literature baselines**, which is good: if COAST beats them with fewer parameters, the argument is stronger.

---

## ⚙️ FAIR BACKBONE — Áp dụng cho TẤT CẢ internal models

```python
# Mọi model nội bộ (COAST, B0, B1, B3, B5, EdgeOff) dùng CHUNG:
FAIR_BACKBONE = {
    "model_size": 128,
    "layer_count": 2,        # ← ĐÃ ĐƯỢC XÁC NHẬN: tất cả dùng 2
    "head_count": 4,          # ← ĐÃ ĐƯỢC XÁC NHẬN: tất cả dùng 4
    "ff_size": 256,           # ← ĐÃ ĐƯỢC XÁC NHẬN: tất cả dùng 256
    "edge_feat_size": 8,
    "memory_size": 128,
    "lookahead_hidden": 128,
    "dropout": 0.1,
    "tanh_xplor": 10,
}
```

**Training config chuẩn cho mọi model nội bộ:**

```python
TRAIN_CONFIG = {
    "problem": "dvrptw",
    "n_customers": 50,
    "m_vehicles": 3,
    "veh_capa": 200,
    "veh_speed": 1,
    "horizon": 480,
    "tw_ratio": [0.25, 0.5, 0.75, 1.0],
    "tw_range": [30, 91],
    "deg_of_dyna": [0.1, 0.25, 0.5, 0.75],
    "appear_early_ratio": [0.0, 0.5, 0.75, 1.0],
    "dur_range": [10, 31],
    "dem_range": [5, 41],
    "loc_range": [0, 101],
    "pending_cost": 2,
    "late_cost": 1,
    "epochs": 500,
    "iter_count": 1000,
    "batch_size": 512,
    "lr": 1e-4,
    "baseline_type": "critic",
    "adv_norm": True,
    "entropy_coef": 0.01,
    "max_grad_norm": 2.0,
    "amp": True,
    "test_batch_size": 10240,
}
```

**Ablation profiles (đã có sẵn trong `utils/_args.py`):**

| Profile | `--ablation-profile` | Memory | Ownership | Lookahead | Edge | Fusion |
|---------|---------------------|--------|-----------|-----------|------|--------|
| COAST   | `vectra`            | ✓ | ✓ | ✓ | ✓ | MLP |
| B0      | `b0`                | ✗ | ✗ | ✗ | ✓ | MLP |
| B1      | `b1`                | ✓ | ✗ | ✗ | ✓ | MLP |
| B3      | `b3`                | ✗ | ✗ | ✓ | ✓ | MLP |
| B5      | `b5`                | ✓ | ✓ | ✓ | ✓ | Linear |
| EdgeOff | `edgeoff`           | ✓ | ✓ | ✓ | ✗ | MLP |
| NoOwnership | `no_ownership`  | ✓ | ✗ | ✓ | ✓ | MLP |
| NoLookahead | `no_lookahead`  | ✓ | ✓ | ✗ | ✓ | MLP |

---

## 📊 EXPERIMENTAL PLAN — 6 Phases, 8-10 Tuần

---

## PHASE 1: MULTI-SEED TRAINING (Tuần 1-3)

### 1.1 Mục tiêu

Train tất cả 6 internal models với **5 seeds** mỗi model = 30 runs, cùng backbone, cùng training config.

### 1.2 Model Matrix

| # | Model | Profile | Seeds cần train | Đã có |
|---|-------|---------|-----------------|-------|
| 1 | COAST | `vectra` | 42, 123, 456, 789, 1024 | seed=null (cần train lại với fixed seed) |
| 2 | B0-None | `b0` | 42, 123, 456, 789, 1024 | seed=42 ✓ |
| 3 | B1-Memory | `b1` | 42, 123, 456, 789, 1024 | seed=42 ✓ |
| 4 | B3-Look | `b3` | 42, 123, 456, 789, 1024 | seed=42 ✓ |
| 5 | B5-Linear | `b5` | 42, 123, 456, 789, 1024 | seed=42 (resumed) |
| 6 | EdgeOff | `edgeoff` | 42, 123, 456, 789, 1024 | seed=42 (resumed) |

### 1.3 Lệnh train cho một model

```bash
# COAST (full model) — seed 42
python MODEL/train.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
    --memory-size 128 --lookahead-hidden 128 \
    --ablation-profile vectra \
    --epoch-count 500 --iter-count 1000 --batch-size 512 \
    --learning-rate 0.0001 --baseline-type critic --adv-norm \
    --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
    --rng-seed 42 \
    --output-dir output/ablation/coast/seed42

# B0 — seed 42
python MODEL/train.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
    --ablation-profile b0 \
    --epoch-count 500 --iter-count 1000 --batch-size 512 \
    --learning-rate 0.0001 --baseline-type critic --adv-norm \
    --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
    --rng-seed 42 \
    --output-dir output/ablation/b0/seed42

# B1 — seed 42
python MODEL/train.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
    --ablation-profile b1 \
    --epoch-count 500 --iter-count 1000 --batch-size 512 \
    --learning-rate 0.0001 --baseline-type critic --adv-norm \
    --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
    --rng-seed 42 \
    --output-dir output/ablation/b1/seed42

# B3 — seed 42
python MODEL/train.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
    --ablation-profile b3 \
    --epoch-count 500 --iter-count 1000 --batch-size 512 \
    --learning-rate 0.0001 --baseline-type critic --adv-norm \
    --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
    --rng-seed 42 \
    --output-dir output/ablation/b3/seed42

# B5 — seed 42
python MODEL/train.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
    --ablation-profile b5 \
    --epoch-count 500 --iter-count 2000 --batch-size 256 \
    --learning-rate 0.0001 --baseline-type critic --adv-norm \
    --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
    --rng-seed 42 \
    --output-dir output/ablation/b5/seed42

# EdgeOff — seed 42
python MODEL/train.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
    --ablation-profile edgeoff \
    --epoch-count 500 --iter-count 2000 --batch-size 256 \
    --learning-rate 0.0001 --baseline-type critic --adv-norm \
    --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
    --rng-seed 42 \
    --output-dir output/ablation/edgeoff/seed42
```

> ⚠️ **Lưu ý:** B5 và EdgeOff hiện dùng `iter_count=2000, batch_size=256`. Nếu muốn chuẩn hóa hoàn toàn, đổi về `iter_count=1000, batch_size=512`. Tuy nhiên tổng số samples ≈ bằng nhau (500K) nên có thể giữ nguyên.

### 1.4 Automation script

Tạo file `script/train_all_seeds.sh`:

```bash
#!/bin/bash
MODELS=("vectra" "b0" "b1" "b3" "b5" "edgeoff")
SEEDS=(42 123 456 789 1024)

for profile in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=== Training profile=$profile seed=$seed ==="
        python MODEL/train.py \
            --problem-type dvrptw \
            --customers-count 50 --vehicles-count 3 \
            --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
            --memory-size 128 --lookahead-hidden 128 \
            --ablation-profile "$profile" \
            --epoch-count 500 --iter-count 1000 --batch-size 512 \
            --learning-rate 0.0001 --baseline-type critic --adv-norm \
            --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
            --rng-seed "$seed" \
            --output-dir "output/ablation/${profile}/seed${seed}"
    done
done
```

### 1.5 Deliverables

- [ ] 30 checkpoints trong `output/ablation/{profile}/seed{seed}/chkpt_best.pyth`
- [ ] 30 training logs (`train_statistics.csv`)
- [ ] Bảng GPU-hours consumed
- [ ] Learning curves cho từng model (5 seeds overlaid)

---

## PHASE 2: IN-DISTRIBUTION EVALUATION (Tuần 3-4)

### 2.1 Test Set

```bash
# Sinh test set 1000 instances, fixed seed 9999
python script/gen_val_data.py \
    --problem-type dvrptw \
    --customers-count 50 --vehicles-count 3 \
    --batch-size 1000 \
    --rng-seed 9999 \
    --output data/test_dvrptw_n50m3_1000.pyth
```

### 2.2 Evaluation Script

Tạo file `script/eval_all_models.py`:

```python
"""Evaluate all trained models on the test set."""
import torch, os, json, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from MODEL.model import VECTRA
from problems import DVRPTW_Environment, DVRPTW_Dataset
from utils import set_random_seed
import numpy as np

TEST_DATA_PATH = "data/test_dvrptw_n50m3_1000.pyth"
OUTPUT_DIR = "output/eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROFILES = {
    "COAST": "vectra",
    "B0-None": "b0",
    "B1-Memory": "b1", 
    "B3-Look": "b3",
    "B5-Linear": "b5",
    "EdgeOff": "edgeoff",
}

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    learner = VECTRA(
        cust_feat_size=7,
        veh_state_size=4,
        model_size=args.get("model_size", 128),
        layer_count=args.get("layer_count", 2),
        head_count=args.get("head_count", 4),
        ff_size=args.get("ff_size", 256),
        tanh_xplor=args.get("tanh_xplor", 10),
        greedy=True,
        edge_feat_size=args.get("edge_feat_size", 8),
        memory_size=args.get("memory_size", 128),
        lookahead_hidden=args.get("lookahead_hidden", 128),
        dropout=args.get("dropout", 0.1),
        use_edge_features=args.get("use_edge_features", True),
        use_memory=args.get("use_memory", True),
        use_ownership=args.get("use_ownership", True),
        use_lookahead=args.get("use_lookahead", True),
        fusion_mode=args.get("fusion_mode", "mlp"),
        linear_fusion_weights=args.get("linear_fusion_weights", (1.0, 1.0, 1.0)),
    )
    learner.load_state_dict(checkpoint["model"], strict=False)
    learner.to(device)
    learner.eval()
    return learner

def evaluate(learner, data, device):
    env_params = [2, 1]  # pending_cost=2, late_cost=1
    env = DVRPTW_Environment(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    with torch.no_grad():
        _, _, rewards = learner(env)
    costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    return costs.cpu()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(TEST_DATA_PATH, map_location="cpu", weights_only=False)
    
    results = {}
    for model_name, profile in PROFILES.items():
        seeds = [42, 123, 456, 789, 1024]
        all_costs = []
        for seed in seeds:
            ckpt_path = f"output/ablation/{profile}/seed{seed}/chkpt_best.pyth"
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {ckpt_path} not found")
                continue
            learner = load_model(ckpt_path, device)
            costs = evaluate(learner, data, device)
            all_costs.append(costs)
            print(f"  {model_name} seed={seed}: mean={costs.mean():.4f} std={costs.std():.4f}")
        
        if all_costs:
            stacked = torch.stack(all_costs)
            results[model_name] = {
                "mean": stacked.mean().item(),
                "std": stacked.std().item(),
                "per_seed_mean": stacked.mean(dim=1).tolist(),
                "per_seed_std": stacked.std(dim=1).tolist(),
            }
    
    with open(f"{OUTPUT_DIR}/in_dist_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/in_dist_results.json")

if __name__ == "__main__":
    main()
```

### 2.3 Primary Metrics Table

| Model | Cost (Mean ± Std) | Gap to COAST | Feasibility % | TW Viol. % |
|-------|-------------------|--------------|---------------|------------|
| COAST | — | — | — | — |
| B0-None | — | — | — | — |
| B1-Memory | — | — | — | — |
| B3-Look | — | — | — | — |
| B5-Linear | — | — | — | — |
| EdgeOff | — | — | — | — |

**Statistical tests:** Paired t-test (COAST vs each baseline), Bootstrap 95% CI, Cohen's d.

### 2.4 Deliverables

- [ ] Primary metrics table với mean ± std across 5 seeds
- [ ] Significance annotations (* p<0.05, ** p<0.01, *** p<0.001)
- [ ] Learning curves (cost vs epoch, 5 seeds overlaid per model)
- [ ] Bar chart: cost comparison across all models with error bars

---

## PHASE 3: CLASSICAL & LITERATURE BASELINES (Tuần 4-5)

### 3.1 Classical Baselines

#### 3.1.1 Greedy Nearest Neighbor

Đã có sẵn trong `baselines/_near_nb.py`. Đánh giá:

```bash
python script/eval_baselines_dyn.py \
    --problem-type dvrptw \
    --data-file data/test_dvrptw_n50m3_1000.pyth \
    --baseline near_nb
```

#### 3.1.2 OR-Tools Insertion Heuristic

Tạo file `script/eval_ortools.py`:

```python
"""Evaluate OR-Tools on DVRPTW test set."""
import torch, os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_instance_ortools(nodes, veh_count, veh_capa, veh_speed, time_limit_sec=10):
    """Solve a single DVRPTW instance with OR-Tools.
    
    Since DVRPTW is dynamic, we solve the STATIC version with all customers known.
    This gives an UPPER BOUND on what's achievable in the dynamic setting.
    """
    n = nodes.size(0)  # depot + customers
    depot = 0
    
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, veh_count, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(torch.norm(nodes[from_node, :2] - nodes[to_node, :2]).item() * 1000)
    
    dist_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)
    
    # Demand constraint
    def demand_callback(idx):
        return int(nodes[manager.IndexToNode(idx), 2].item())
    
    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [int(veh_capa)]*veh_count, True, "Capacity")
    
    # Time window constraint
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        dist = torch.norm(nodes[from_node, :2] - nodes[to_node, :2]).item()
        travel_time = int(dist / veh_speed * 1000)
        service_time = int(nodes[from_node, 5].item() * 1000) if from_node > 0 else 0
        return travel_time + service_time
    
    time_cb = routing.RegisterTransitCallback(time_callback)
    horizon = int(nodes[0, 4].item() * 1000)  # depot tw_end as horizon
    routing.AddDimension(time_cb, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    
    for i in range(1, n):
        idx = manager.NodeToIndex(i)
        tw_start = int(nodes[i, 3].item() * 1000)
        tw_end = int(nodes[i, 4].item() * 1000)
        time_dim.CumulVar(idx).SetRange(tw_start, tw_end)
    
    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.time_limit.seconds = time_limit_sec
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return float('inf')
    
    return solution.ObjectiveValue() / 1000.0  # convert back


def main():
    data = torch.load("data/test_dvrptw_n50m3_1000.pyth", map_location="cpu", weights_only=False)
    nodes = data.nodes  # N x L_c x 7
    
    costs = []
    for i in range(len(nodes)):
        cost = solve_instance_ortools(nodes[i], data.veh_count, data.veh_capa, data.veh_speed)
        costs.append(cost)
        print(f"Instance {i}: OR-Tools cost = {cost:.4f}")
    
    costs_t = torch.tensor(costs)
    print(f"\nOR-Tools: mean={costs_t.mean():.4f} std={costs_t.std():.4f}")
    
    with open("output/eval_results/ortools_results.json", "w") as f:
        json.dump({"mean": costs_t.mean().item(), "std": costs_t.std().item(), "costs": costs}, f, indent=2)

if __name__ == "__main__":
    main()
```

> ⚠️ **Quan trọng:** OR-Tools giải STATIC version (biết trước tất cả khách hàng) → đây là **lower bound** (optimistic). DVRPTW là dynamic nên chi phí thực tế sẽ cao hơn. Cần ghi rõ điều này trong paper.

### 3.2 Literature Neural Baselines

#### 3.2.1 AM (Attention Model — Kool et al. 2019)

Checkpoint: `/home/admin_wsl/projects/RL4DVRPTW/data/_AM/chkpt_best.pyth`
Config: `layer=5, head=8, ff=256` (mạnh hơn COAST về capacity)

Đánh giá:

```bash
python script/eval_literature_baselines.py \
    --model-type am \
    --checkpoint /home/admin_wsl/projects/RL4DVRPTW/data/_AM/chkpt_best.pyth \
    --test-data data/test_dvrptw_n50m3_1000.pyth \
    --output output/eval_results/am_results.json
```

#### 3.2.2 PolyNet

Checkpoint: `/home/admin_wsl/projects/RL4DVRPTW/data/_PolyNet/chkpt_best.pyth`
Config: `layer=5, head=8, ff=256`

```bash
python script/eval_literature_baselines.py \
    --model-type polynet \
    --checkpoint /home/admin_wsl/projects/RL4DVRPTW/data/_PolyNet/chkpt_best.pyth \
    --test-data data/test_dvrptw_n50m3_1000.pyth \
    --output output/eval_results/polynet_results.json
```

> ⚠️ **Lưu ý về fairness:** AM và PolyNet dùng `layer=5, head=8` (nhiều tham số hơn COAST). Nếu COAST thắng → argument rất mạnh. Nếu AM/Polynet thắng → COAST ít nhất là competitive với ít tham số hơn.

### 3.3 Comparison Table (All Baselines)

| Model | Type | Params | Cost | Gap | Feas. % | Time/step |
|-------|------|--------|------|-----|---------|-----------|
| OR-Tools (static LB) | Classical | — | — | — | 100% | ~10s |
| Greedy NN | Classical | — | — | — | — | <1ms |
| AM (Kool et al.) | Neural | ~X | — | — | — | — |
| PolyNet | Neural | ~X | — | — | — | — |
| B0-None | Ablation | — | — | — | — | — |
| B1-Memory | Ablation | — | — | — | — | — |
| B3-Look | Ablation | — | — | — | — | — |
| B5-Linear | Ablation | — | — | — | — | — |
| EdgeOff | Ablation | — | — | — | — | — |
| **COAST** | **Ours** | — | — | — | — | — |

### 3.4 Deliverables

- [ ] Full comparison table (neural + classical)
- [ ] OR-Tools anytime performance curve
- [ ] Inference latency comparison
- [ ] Parameter count table

---

## PHASE 4: HYPOTHESIS VALIDATION (Tuần 5-7)

### 4.1 H1 — Coordination Validation

**So sánh:** COAST vs B1-Memory vs B0-None

**Metrics to compute:**

```python
# Trong script/eval_coordination.py

def compute_ownership_entropy(owner_probs, cust_mask=None):
    """H = -sum(p_i * log(p_i)) — lower = more decisive"""
    p = owner_probs.clamp_min(1e-9)
    entropy = -(p * p.log()).sum(dim=1)  # sum over vehicles
    if cust_mask is not None:
        entropy = entropy.masked_fill(cust_mask, 0)
    return entropy.mean()

def compute_service_region_overlap(routes, customer_positions, threshold=0.1):
    """% customers with >= 2 vehicles within threshold distance"""
    overlap_count = 0
    total = 0
    for instance_routes, positions in zip(routes, customer_positions):
        for c in range(1, len(positions)):  # skip depot
            vehicles_nearby = 0
            for route in instance_routes:
                if any(torch.norm(positions[c] - positions[node], p=2) < threshold 
                       for node in route if node > 0 and node < len(positions)):
                    vehicles_nearby += 1
            if vehicles_nearby >= 2:
                overlap_count += 1
            total += 1
    return overlap_count / total if total > 0 else 0

def compute_vehicle_switches_per_cluster(routes, customer_positions, n_clusters=None):
    """K-means cluster customers, count unique vehicles per cluster"""
    from sklearn.cluster import KMeans
    # ... implementation
```

**Deliverables:**

- [ ] Bảng H1 metrics: COAST vs B1 vs B0
- [ ] Ownership heatmap (heatmap theo timestep × customer)
- [ ] Spatial ownership map (vị trí khách, color theo xe có ownership cao nhất)
- [ ] Memory t-SNE plot

### 4.2 H2 — Anticipation Validation

**So sánh:** COAST vs B3-Look vs B0-None

**Metrics to compute:**

```python
# Trong script/eval_anticipation.py

def identify_isolated_customers(customer_positions, percentile=90):
    """Customers whose nearest neighbor distance > p-th percentile"""
    dists = torch.cdist(customer_positions, customer_positions)
    dists.fill_diagonal_(float('inf'))
    nn_dists = dists.min(dim=-1).values
    threshold = torch.quantile(nn_dists, percentile / 100)
    return nn_dists > threshold

def compute_lookahead_intervention_rate(att_scores, final_scores, masks):
    """% steps where argmax(att_score) != argmax(final_score)"""
    att_choice = att_scores.argmax(dim=-1)
    final_choice = final_scores.argmax(dim=-1)
    interventions = (att_choice != final_choice).float()
    # Exclude fully masked steps
    valid = (~masks.all(dim=-1)).float()
    return (interventions * valid).sum() / valid.sum().clamp_min(1)

def compute_lookahead_calibration(lookahead_scores, actual_future_costs):
    """Pearson correlation between predicted and actual"""
    from scipy.stats import pearsonr
    l = lookahead_scores.cpu().numpy().flatten()
    a = actual_future_costs.cpu().numpy().flatten()
    return pearsonr(l, a)
```

**Deliverables:**

- [ ] Bảng H2 metrics: COAST vs B3 vs B0
- [ ] Lookahead vs actual cost scatter plot
- [ ] Intervention case study (2-3 instances với route visualization)
- [ ] Intervention rate over episode timeline

### 4.3 H3 — Edge-Awareness Validation

**So sánh:** COAST vs EdgeOff vs B0-None

**Test regimes:**

1. Loose TW: `tw_range=[60, 120]`
2. Medium TW: `tw_range=[30, 91]` (train distribution)
3. Tight TW: `tw_range=[10, 40]`
4. Capacity stress: `veh_capa=150` (thay vì 200)

**Deliverables:**

- [ ] Bảng H3: TW violation rate × constraint tightness
- [ ] Capacity violation rate
- [ ] Feasibility drop curve

### 4.4 H4 — Generalization Validation

**Train:** n=50, m=3
**Test OOD:**

| OOD Regime | Config | Purpose |
|-----------|--------|---------|
| OOD-Scale | n=100, m=5 | Larger instances |
| OOD-Fleet | n=50, m=6 | More vehicles |
| OOD-Tight | tw_range=[10,40] | Tight time windows |
| OOD-Burst | deg_of_dyna=[0.5,0.75,1.0] | Highly dynamic |
| OOD-Sparse | loc_range=[0,201] | Sparse spatial |

**Deliverables:**

- [ ] OOD degradation table: (C_OOD - C_ID) / C_ID for each model × OOD regime
- [ ] Generalization score heatmap
- [ ] Bar chart: which model degrades least

---

## PHASE 5: BEHAVIORAL ANALYSIS & VISUALIZATION (Tuần 7-8)

### 5.1 Ownership Visualization Pipeline

```python
# script/vis_ownership.py — Pseudocode
for instance in representative_instances:
    # 1. Run COAST, log owner_probs at each step
    owner_probs_over_time = []  # T x L_v x L_c
    trajectories = []  # per-vehicle routes
    
    # 2. Spatial ownership map
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green']  # one per vehicle
    for c in customers:
        dominant_veh = owner_probs[:, c].mean(0).argmax()
        ax.scatter(c.x, c.y, color=colors[dominant_veh])
    for veh_idx, route in enumerate(trajectories):
        route_coords = [customer_positions[n] for n in route]
        ax.plot(*zip(*route_coords), color=colors[veh_idx], alpha=0.5)
    ax.set_title("Spatial Ownership Map — Instance X")
    plt.savefig(f"figures/ownership_map_{instance}.pdf")
    
    # 3. Ownership heatmap over time
    fig, ax = plt.subplots()
    im = ax.imshow(owner_probs_for_veh0.T, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel("Timestep"); ax.set_ylabel("Customer")
    plt.colorbar(im)
    plt.savefig(f"figures/ownership_heatmap_{instance}.pdf")
```

### 5.2 Lookahead Intervention Case Study

```python
# script/vis_lookahead.py
# For 2-3 instances, show:
# - Map with customer positions
# - COAST's route (colored per vehicle)
# - B0's route (greedy, no lookahead)
# - Highlight where COAST's lookahead changed the decision
# - Table: cost comparison
```

### 5.3 Failure Mode Analysis

Chọn 20 worst instances (COAST cost - OR-Tools cost lớn nhất). Phân loại:

1. **Ownership collapse:** ≥80% customers assigned to 1 vehicle
2. **Lookahead miscalibration:** |predicted - actual| > 2σ
3. **Edge misinterpretation:** TW violations despite feasible alternatives
4. **Exploration failure:** All vehicles return to depot early

### 5.4 Deliverables

- [ ] Spatial ownership maps (3-5 instances)
- [ ] Ownership heatmaps (3-5 instances)
- [ ] Memory t-SNE plot
- [ ] Lookahead calibration scatter plot
- [ ] Intervention case studies (2-3 instances)
- [ ] Failure mode taxonomy pie chart

---

## PHASE 6: PAPER WRITING & RELEASE (Tuần 8-10)

### 6.1 Paper Structure (8 pages + references)

```
1. Introduction (1.5p)
   - DVRPTW: coordination + anticipation entanglement
   - COAST: structured decomposition into ownership + lookahead
   - Contributions: (1) decomposition framework, (2) coordination memory,
     (3) candidate-conditioned lookahead, (4) empirical validation

2. Related Work (1p)
   - Neural VRP (AM, POMO, MDAM, PolyNet)
   - Multi-agent coordination (CTDE, implicit communication)
   - Value-based planning in routing

3. Problem Formulation (0.75p)
   - DVRPTW definition
   - sMMDP: state, action, transition, reward
   - Shared sequential policy

4. Method: COAST (2p)
   - Architecture overview (figure)
   - Customer encoding: GraphEncoder + RBF
   - Vehicle encoding: FleetEncoder (cross-attention)
   - Coordination: CoordinationMemory + OwnershipHead
   - Anticipation: EdgeFeatureEncoder + LookaheadHead
   - Decision fusion: CrossEdgeFusion + ScoreFusion (MLP)
   - Training: REINFORCE + critic baseline

5. Experiments (2.5p)
   - Setup: fair backbone, 5 seeds, metrics
   - RQ1: In-distribution comparison (Table 1)
   - RQ2: Does coordination reduce conflicts? (Table 2, Figure 3)
   - RQ3: Does lookahead reduce myopia? (Table 3, Figure 4)
   - RQ4: Does COAST generalize better? (Table 4, Figure 5)
   - RQ5: Ablation & behavioral analysis

6. Conclusion (0.25p)
   - Structured decomposition improves DVRPTW routing
   - Coordination + anticipation are orthogonal benefits
   - Future: B2/B4 baselines, larger scale, real-world data
```

### 6.2 Figure Plan

| Figure | Content | Phase |
|--------|---------|-------|
| Fig 1 | COAST architecture diagram | Writing |
| Fig 2 | Learning curves (5 seeds × 6 models) | Phase 2 |
| Fig 3 | Spatial ownership maps + heatmaps | Phase 5 |
| Fig 4 | Lookahead calibration + intervention case | Phase 5 |
| Fig 5 | OOD generalization bar chart | Phase 4 |
| Fig 6 | Ablation contribution waterfall | Phase 4 |

### 6.3 Table Plan

| Table | Content | Phase |
|-------|---------|-------|
| Table 1 | Primary results: all models on in-dist test | Phase 2 |
| Table 2 | H1 coordination metrics | Phase 4 |
| Table 3 | H2 anticipation metrics | Phase 4 |
| Table 4 | OOD generalization | Phase 4 |
| Table 5 | Ablation matrix | Phase 4 |

### 6.4 Reproducibility Checklist

- [ ] GitHub repo với README rõ ràng
- [ ] `requirements.txt` hoặc `environment.yml`
- [ ] Pretrained checkpoints (COAST + all ablations, 5 seeds)
- [ ] Test data splits
- [ ] Evaluation script (`script/eval_all_models.py`)
- [ ] Verification script (mini-run < 5 phút)
- [ ] Training logs
- [ ] Visualization scripts

---

## 📅 TIMELINE

| Tuần | Phase | Key Deliverable |
|------|-------|-----------------|
| 1 | Phase 1 | Bắt đầu multi-seed training (COAST + B0 + B1) |
| 2 | Phase 1 | Tiếp tục training (B3 + B5 + EdgeOff) |
| 3 | Phase 1→2 | Training done → Bắt đầu evaluation |
| 4 | Phase 2→3 | In-dist results + OR-Tools baseline |
| 5 | Phase 3→4 | Literature baselines + Bắt đầu H1/H2 |
| 6 | Phase 4 | H3/H4 validation + OOD evaluation |
| 7 | Phase 4→5 | Behavioral analysis + visualizations |
| 8 | Phase 5→6 | Final analysis + Bắt đầu viết paper |
| 9 | Phase 6 | Paper draft hoàn chỉnh |
| 10 | Phase 6 | Polish, release code, submit |

---

## 🖥️ RESOURCE ESTIMATION

| Task | GPU-hours (1× RTX 4090) |
|------|-------------------------|
| 30 training runs (6 models × 5 seeds) | ~300h (12.5 days) |
| Evaluation (all models, all regimes) | ~20h |
| OR-Tools (CPU) | ~10h |
| Behavioral analysis | ~10h |
| **TOTAL** | **~340h (14 days)** |

**Với 2 GPUs:** ~7 days. **Với 4 GPUs:** ~3.5 days.

---

## ✅ PRE-SUBMISSION FINAL CHECKLIST

### Results

- [ ] COAST outperforms ALL internal ablations (p<0.05) on in-dist cost
- [ ] COAST competitive or better vs AM, PolyNet (despite fewer params)
- [ ] COAST feasible >95% on in-distribution
- [ ] COAST beats OR-Tools at <1s but OR-Tools catches up at >10s (expected)
- [ ] H1: Ownership-on-memory reduces conflict metrics vs B1 and B0 (p<0.05)
- [ ] H2: Lookahead reduces isolated regret and intervenes on 15-25% decisions
- [ ] H3: Edge features reduce TW violations under tight constraints
- [ ] H4: COAST generalizes better than B5-Linear and B0 on OOD

### Paper Quality

- [ ] Clear hypothesis-driven narrative (not a module laundry list)
- [ ] Every claim backed by a specific table/figure
- [ ] Statistical rigor: CI, p-values, effect sizes
- [ ] Limitations section: B2/B4 not implemented, static OR-Tools bound, single problem size
- [ ] All figures are vector format, readable at print size

### Code

- [ ] All training/evaluation/visualization scripts committed
- [ ] README with quickstart (< 5 commands to reproduce key result)
- [ ] Pretrained checkpoints downloadable
- [ ] License file

---

**END OF UPDATED EXPERIMENTAL PROTOCOL**
