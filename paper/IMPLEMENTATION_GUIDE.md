# COAST Implementation Guide

## Step-by-Step Execution Manual — May 2026

---

## 🚀 QUICKSTART (Day 0)

### 0.1 Environment Setup

```bash
# Activate conda environment
conda activate ai

# Verify dependencies
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from MODEL.model import VECTRA; print('VECTRA import OK')"
python -c "from layers import GraphEncoder, FleetEncoder, CrossEdgeFusion; print('Layers import OK')"

# Check GPU
nvidia-smi
```

### 0.2 Directory Structure

```bash
cd /home/admin_wsl/projects/mardam-master

# Create output directories
mkdir -p output/ablation/{vectra,b0,b1,b3,b5,edgeoff}
mkdir -p output/eval_results
mkdir -p figures
mkdir -p data/test_sets

# Verify existing checkpoints
ls -la data/vectra/chkpt_best.pyth          # COAST full model
ls -la data/_Ablation/B0/chkpt_best.pyth     # B0-None
ls -la data/_Ablation/B1/chkpt_best.pyth     # B1-Memory
ls -la data/_Ablation/B3/chkpt_best.pyth     # B3-Look
ls -la data/_Ablation/B5/chkpt_best.pyth     # B5-Linear
ls -la data/_Ablation/Edgeoff/chkpt_best.pyth # EdgeOff
ls -la /home/admin_wsl/projects/RL4DVRPTW/data/_AM/chkpt_best.pyth      # AM
ls -la /home/admin_wsl/projects/RL4DVRPTW/data/_PolyNet/chkpt_best.pyth  # PolyNet
```

---

## 📍 STEP 1: Generate Test Sets (Day 1, ~30 min)

### 1.1 In-Distribution Test Set (1000 instances)

```bash
cd /home/admin_wsl/projects/mardam-master

python -c "
import torch
from problems import DVRPTW_Dataset

# Generate test set with fixed seed
torch.manual_seed(9999)
data = DVRPTW_Dataset.generate(
    batch_size=1000,
    cust_count=50,
    veh_count=3,
    veh_capa=200,
    veh_speed=1,
    min_cust_count=None,
    cust_loc_range=(0,101),
    cust_dem_range=(5,41),
    horizon=480,
    cust_dur_range=(10,31),
    tw_ratio=[0.25, 0.5, 0.75, 1.0],
    cust_tw_range=[30, 91],
    deg_of_dyna=[0.1, 0.25, 0.5, 0.75],
    appear_early_ratio=[0.0, 0.5, 0.75, 1.0],
)
data.normalize()
torch.save(data, 'data/test_sets/test_dvrptw_n50m3_id_1000.pyth')
print(f'Saved: {data.nodes.size()}, veh_count={data.veh_count}')
print(f'Deg of dyna: min={(data.nodes[:,:,6]>0).sum(dim=1).float().mean().item():.2%}')
"
```

### 1.2 OOD Test Sets

```bash
cd /home/admin_wsl/projects/mardam-master

# OOD-Scale: n=100, m=5
python -c "
import torch
from problems import DVRPTW_Dataset
torch.manual_seed(9998)
data = DVRPTW_Dataset.generate(500, cust_count=100, veh_count=5, veh_capa=200,
    veh_speed=1, cust_loc_range=(0,101), cust_dem_range=(5,41), horizon=480,
    cust_dur_range=(10,31), tw_ratio=[0.25,0.5,0.75,1.0], cust_tw_range=[30,91],
    deg_of_dyna=[0.1,0.25,0.5,0.75], appear_early_ratio=[0.0,0.5,0.75,1.0])
data.normalize()
torch.save(data, 'data/test_sets/test_dvrptw_n100m5_ood_500.pyth')
print('OOD-Scale saved')
"

# OOD-Fleet: n=50, m=6
python -c "
import torch
from problems import DVRPTW_Dataset
torch.manual_seed(9997)
data = DVRPTW_Dataset.generate(500, cust_count=50, veh_count=6, veh_capa=200,
    veh_speed=1, cust_loc_range=(0,101), cust_dem_range=(5,41), horizon=480,
    cust_dur_range=(10,31), tw_ratio=[0.25,0.5,0.75,1.0], cust_tw_range=[30,91],
    deg_of_dyna=[0.1,0.25,0.5,0.75], appear_early_ratio=[0.0,0.5,0.75,1.0])
data.normalize()
torch.save(data, 'data/test_sets/test_dvrptw_n50m6_ood_500.pyth')
print('OOD-Fleet saved')
"

# OOD-Tight: tight time windows
python -c "
import torch
from problems import DVRPTW_Dataset
torch.manual_seed(9996)
data = DVRPTW_Dataset.generate(500, cust_count=50, veh_count=3, veh_capa=200,
    veh_speed=1, cust_loc_range=(0,101), cust_dem_range=(5,41), horizon=480,
    cust_dur_range=(10,31), tw_ratio=[0.8,0.9,1.0], cust_tw_range=[10,40],
    deg_of_dyna=[0.1,0.25,0.5,0.75], appear_early_ratio=[0.0,0.5,0.75,1.0])
data.normalize()
torch.save(data, 'data/test_sets/test_dvrptw_n50m3_tight_ood_500.pyth')
print('OOD-Tight saved')
"

# OOD-Burst: highly dynamic
python -c "
import torch
from problems import DVRPTW_Dataset
torch.manual_seed(9995)
data = DVRPTW_Dataset.generate(500, cust_count=50, veh_count=3, veh_capa=200,
    veh_speed=1, cust_loc_range=(0,101), cust_dem_range=(5,41), horizon=480,
    cust_dur_range=(10,31), tw_ratio=[0.25,0.5,0.75,1.0], cust_tw_range=[30,91],
    deg_of_dyna=[0.5,0.75,1.0], appear_early_ratio=[0.0,0.25,0.5])
data.normalize()
torch.save(data, 'data/test_sets/test_dvrptw_n50m3_burst_ood_500.pyth')
print('OOD-Burst saved')
"

# OOD-Sparse: sparse spatial
python -c "
import torch
from problems import DVRPTW_Dataset
torch.manual_seed(9994)
data = DVRPTW_Dataset.generate(500, cust_count=50, veh_count=3, veh_capa=200,
    veh_speed=1, cust_loc_range=(0,201), cust_dem_range=(5,41), horizon=480,
    cust_dur_range=(10,31), tw_ratio=[0.25,0.5,0.75,1.0], cust_tw_range=[30,91],
    deg_of_dyna=[0.1,0.25,0.5,0.75], appear_early_ratio=[0.0,0.5,0.75,1.0])
data.normalize()
torch.save(data, 'data/test_sets/test_dvrptw_n50m3_sparse_ood_500.pyth')
print('OOD-Sparse saved')
"
```

---

## 📍 STEP 2: Multi-Seed Training (Day 1-7, depends on GPU count)

### 2.1 Create Training Script

Create `script/train_all_seeds.sh`:

```bash
#!/bin/bash
set -e

PROJECT_ROOT="/home/admin_wsl/projects/mardam-master"
cd "$PROJECT_ROOT"

# ============ CONFIGURATION ============
MODELS=("vectra" "b0" "b1" "b3" "b5" "edgeoff")
SEEDS=(42 123 456 789 1024)
GPU_ID=${GPU_ID:-0}  # Which GPU to use (0, 1, 2, ...)

# Training params (SAME FOR ALL)
EPOCHS=500
ITERS=1000
BATCH=512
LR=0.0001

# ============ TRAINING LOOP ============
for profile in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        OUTPUT_DIR="output/ablation/${profile}/seed${seed}"
        
        # Skip if already trained
        if [ -f "${OUTPUT_DIR}/chkpt_best.pyth" ]; then
            echo "[SKIP] ${profile} seed=${seed} — already exists at ${OUTPUT_DIR}"
            continue
        fi
        
        echo "============================================================"
        echo "TRAINING: profile=${profile} seed=${seed}"
        echo "Output: ${OUTPUT_DIR}"
        echo "GPU: ${GPU_ID}"
        echo "============================================================"
        
        CUDA_VISIBLE_DEVICES=${GPU_ID} python MODEL/train.py \
            --problem-type dvrptw \
            --customers-count 50 --vehicles-count 3 \
            --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
            --memory-size 128 --lookahead-hidden 128 \
            --ablation-profile "${profile}" \
            --epoch-count ${EPOCHS} --iter-count ${ITERS} --batch-size ${BATCH} \
            --learning-rate ${LR} --baseline-type critic --adv-norm \
            --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
            --rng-seed "${seed}" \
            --output-dir "${OUTPUT_DIR}" \
            --test-batch-size 10240
        
        echo "[DONE] ${profile} seed=${seed}"
        echo ""
    done
done

echo "============================================================"
echo "ALL TRAINING COMPLETE"
echo "============================================================"
```

### 2.2 Run Training

```bash
# Single GPU (sequential)
chmod +x script/train_all_seeds.sh
bash script/train_all_seeds.sh

# Multi-GPU (run in separate terminals or use GNU parallel)
# Terminal 1: GPU_ID=0 bash script/train_all_seeds.sh
# Terminal 2: GPU_ID=1 bash script/train_all_seeds.sh
```

### 2.3 Monitor Training

```bash
# Check progress
ls output/ablation/*/seed*/chkpt_best.pyth | wc -l

# Quick look at a training curve
python -c "
import pandas as pd
df = pd.read_csv('output/ablation/vectra/seed42/train_statistics.csv')
print(df.tail())
print(f'Final cost: {df.iloc[-1][\"val\"]:.4f}')
"
```

---

## 📍 STEP 3: Evaluate All Models on In-Distribution (Day 7-8)

### 3.1 Create Unified Evaluation Script

Create `script/eval_unified.py`:

```python
#!/usr/bin/env python3
"""
Unified evaluation script for all models on a given test set.
Usage:
    python script/eval_unified.py \
        --test-data data/test_sets/test_dvrptw_n50m3_id_1000.pyth \
        --models-dir output/ablation \
        --output output/eval_results/in_dist.json \
        --seeds 42,123,456,789,1024
"""
import torch, os, sys, json, argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from MODEL.model import VECTRA
from problems import DVRPTW_Environment

def load_vectra_from_checkpoint(ckpt_path, device, use_args_from_ckpt=True):
    """Load a VECTRA model with correct config from checkpoint or defaults."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Try to get args from checkpoint, fall back to defaults
    args = checkpoint.get("args", {})
    
    learner = VECTRA(
        cust_feat_size=7,
        veh_state_size=4,
        model_size=args.get("model_size", 128),
        layer_count=args.get("layer_count", 2),
        head_count=args.get("head_count", 4),
        ff_size=args.get("ff_size", 256),
        tanh_xplor=args.get("tanh_xplor", 10),
        greedy=True,  # GREEDY for evaluation
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

def evaluate_model(learner, data, device, env_params=(2, 1)):
    """Run one evaluation pass, return costs tensor (N,)."""
    env = DVRPTW_Environment(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    with torch.no_grad():
        _, _, rewards = learner(env)
    costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    return costs.cpu()

def compute_stats(costs_list):
    """Compute mean, std, CI from list of cost tensors."""
    stacked = torch.stack(costs_list)  # (n_seeds, N_instances)
    instance_means = stacked.mean(dim=0)  # (N_instances,)
    mean = instance_means.mean().item()
    std = instance_means.std().item()
    per_seed_means = stacked.mean(dim=1).tolist()
    return {"mean": mean, "std": std, "per_seed_means": per_seed_means}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--models-dir", default="output/ablation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    data = torch.load(args.test_data, map_location="cpu", weights_only=False)
    print(f"Test data: {data.nodes.size()}, veh_count={data.veh_count}")
    
    # Models to evaluate
    profiles = {
        "COAST": "vectra",
        "B0-None": "b0",
        "B1-Memory": "b1",
        "B3-Look": "b3",
        "B5-Linear": "b5",
        "EdgeOff": "edgeoff",
    }
    
    all_results = {}
    for model_name, profile in profiles.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name} (profile={profile})")
        
        seed_costs = []
        for seed in seeds:
            ckpt_path = os.path.join(args.models_dir, profile, f"seed{seed}", "chkpt_best.pyth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] seed={seed}: {ckpt_path} not found")
                continue
            
            learner = load_vectra_from_checkpoint(ckpt_path, device)
            costs = evaluate_model(learner, data, device)
            seed_costs.append(costs)
            print(f"  seed={seed}: mean_cost={costs.mean():.4f} ± {costs.std():.4f}")
        
        if seed_costs:
            stats = compute_stats(seed_costs)
            all_results[model_name] = stats
            print(f"  => OVERALL: {stats['mean']:.4f} ± {stats['std']:.4f} (across {len(seed_costs)} seeds)")
        else:
            print(f"  [FAIL] No checkpoints found for {model_name}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
```

### 3.2 Run Evaluation

```bash
cd /home/admin_wsl/projects/mardam-master

# In-distribution evaluation
python script/eval_unified.py \
    --test-data data/test_sets/test_dvrptw_n50m3_id_1000.pyth \
    --models-dir output/ablation \
    --output output/eval_results/in_dist.json

# View results
python -c "
import json
with open('output/eval_results/in_dist.json') as f:
    results = json.load(f)
for model, stats in sorted(results.items(), key=lambda x: x[1]['mean']):
    print(f'{model:15s}: {stats[\"mean\"]:.4f} ± {stats[\"std\"]:.4f}')
"
```

---

## 📍 STEP 4: Classical Baselines (Day 7-8)

### 4.1 Greedy Nearest Neighbor

```bash
cd /home/admin_wsl/projects/mardam-master

python -c "
import torch, sys, os
sys.path.insert(0, '.')
from baselines._near_nb import NearestNeighbourBaseline
from MODEL.model import VECTRA
from problems import DVRPTW_Environment

# Create a minimal learner for the baseline wrapper
learner = VECTRA(7, 4, model_size=128, layer_count=2, head_count=4, ff_size=256, greedy=True)
baseline = NearestNeighbourBaseline(learner)

data = torch.load('data/test_sets/test_dvrptw_n50m3_id_1000.pyth', map_location='cpu', weights_only=False)
env_params = [2, 1]
env = DVRPTW_Environment(data, None, None, *env_params)

with torch.no_grad():
    _, _, rewards = baseline(env)
costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
print(f'Greedy NN: mean={costs.mean():.4f} std={costs.std():.4f}')
"
```

### 4.2 OR-Tools (Static Lower Bound)

See `script/eval_ortools.py` in the protocol document. Requires `ortools` package:

```bash
pip install ortools
python script/eval_ortools.py
```

---

## 📍 STEP 5: Literature Baselines — AM & PolyNet (Day 8-9)

### 5.1 Create Evaluation Script

Create `script/eval_literature.py`:

```python
#!/usr/bin/env python3
"""Evaluate AM and PolyNet checkpoints on DVRPTW test set."""
import torch, os, sys, json, argparse

# Add RL4DVRPTW to path for AM/Polynet imports
RL4DVRPTW = "/home/admin_wsl/projects/RL4DVRPTW"
sys.path.insert(0, RL4DVRPTW)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from problems import DVRPTW_Environment

def evaluate_am(checkpoint_path, data, device):
    """Evaluate AM (Attention Model) from rl4co."""
    from am.model import AM_DVRPTW
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    
    model = AM_DVRPTW(
        cust_feat_size=7,
        veh_state_size=4,
        model_size=args.get("model_size", 128),
        layer_count=args.get("layer_count", 5),
        head_count=args.get("head_count", 8),
        ff_size=args.get("ff_size", 256),
        greedy=True,
    )
    
    if "model" in checkpoint:
        # Try loading state dict - may have prefix differences
        state_dict = checkpoint["model"]
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"  [WARN] State dict load issue: {e}")
            # Try loading policy directly
            if "policy" in state_dict:
                model.policy.load_state_dict(state_dict["policy"], strict=False)
    
    model.to(device)
    model.eval()
    
    env_params = [2, 1]
    env = DVRPTW_Environment(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    # Initialize encoder
    model._encode_customers(env)
    
    with torch.no_grad():
        actions, _, rewards = [], [], []
        while not env.done:
            if env.new_customers:
                model._encode_customers(env)
            cust_idx, logp = model.step(env)
            actions.append((env.cur_veh_idx.clone(), cust_idx.clone()))
            rewards.append(env.step(cust_idx))
    
    costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    return costs.cpu()

def evaluate_polynet(checkpoint_path, data, device):
    """Evaluate PolyNet from rl4co."""
    from polynet.model import PolyNet_DVRPTW
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    
    model = PolyNet_DVRPTW(
        cust_feat_size=7,
        veh_state_size=4,
        model_size=args.get("model_size", 128),
        layer_count=args.get("layer_count", 5),
        head_count=args.get("head_count", 8),
        ff_size=args.get("ff_size", 256),
        greedy=True,
        k=args.get("k", 32),
    )
    
    if "model" in checkpoint:
        try:
            model.load_state_dict(checkpoint["model"], strict=False)
        except Exception as e:
            print(f"  [WARN] State dict load issue: {e}")
    
    model.to(device)
    model.eval()
    
    env_params = [2, 1]
    env = DVRPTW_Environment(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    model._encode_customers(env)
    
    with torch.no_grad():
        actions, _, rewards = [], [], []
        while not env.done:
            if env.new_customers:
                model._encode_customers(env)
            cust_idx, logp = model.step(env)
            actions.append((env.cur_veh_idx.clone(), cust_idx.clone()))
            rewards.append(env.step(cust_idx))
    
    costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    return costs.cpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = torch.load(args.test_data, map_location="cpu", weights_only=False)
    print(f"Test data: {data.nodes.size()}")
    
    results = {}
    
    # AM
    am_ckpt = "/home/admin_wsl/projects/RL4DVRPTW/data/_AM/chkpt_best.pyth"
    if os.path.exists(am_ckpt):
        print("\nEvaluating AM...")
        costs = evaluate_am(am_ckpt, data, device)
        results["AM"] = {"mean": costs.mean().item(), "std": costs.std().item()}
        print(f"  AM: {costs.mean():.4f} ± {costs.std():.4f}")
    
    # PolyNet
    poly_ckpt = "/home/admin_wsl/projects/RL4DVRPTW/data/_PolyNet/chkpt_best.pyth"
    if os.path.exists(poly_ckpt):
        print("\nEvaluating PolyNet...")
        costs = evaluate_polynet(poly_ckpt, data, device)
        results["PolyNet"] = {"mean": costs.mean().item(), "std": costs.std().item()}
        print(f"  PolyNet: {costs.mean():.4f} ± {costs.std():.4f}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
```

### 5.2 Run

```bash
python script/eval_literature.py \
    --test-data data/test_sets/test_dvrptw_n50m3_id_1000.pyth \
    --output output/eval_results/literature_baselines.json
```

---

## 📍 STEP 6: Hypothesis Testing Scripts (Day 9-14)

### 6.1 H1 — Coordination Analysis

Create `script/eval_h1_coordination.py`:

```python
#!/usr/bin/env python3
"""H1: Ownership-on-top-of-memory reduces inter-vehicle conflicts."""
import torch, os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from MODEL.model import VECTRA
from problems import DVRPTW_Environment
import numpy as np
from sklearn.cluster import KMeans

def load_model(ckpt_path, device, greedy=True):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    learner = VECTRA(
        7, 4, model_size=128, layer_count=2, head_count=4, ff_size=256,
        tanh_xplor=10, greedy=greedy, memory_size=128, lookahead_hidden=128,
        dropout=0.1, use_edge_features=args.get("use_edge_features", True),
        use_memory=args.get("use_memory", True),
        use_ownership=args.get("use_ownership", True),
        use_lookahead=args.get("use_lookahead", True),
        fusion_mode=args.get("fusion_mode", "mlp"),
        linear_fusion_weights=args.get("linear_fusion_weights", (1.0, 1.0, 1.0)),
    )
    learner.load_state_dict(checkpoint["model"], strict=False)
    learner.to(device); learner.eval()
    return learner

def compute_ownership_metrics(learner, data, device):
    """Run model and collect ownership-related metrics."""
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    # Storage
    owner_probs_log = []  # list of (N, L_v, L_c) per step
    routes = []  # per instance, per vehicle
    vehicle_positions_over_time = []  # track positions
    
    with torch.no_grad():
        while not env.done:
            if env.new_customers:
                learner._encode_customers(env.nodes, env.cust_mask)
            
            # Get owner probs before decision
            if learner.use_ownership and learner._veh_memory is not None:
                owner_logits = learner.owner_head(learner._veh_memory, learner.cust_repr)
                owner_probs = owner_logits.softmax(dim=1)  # N x L_v x L_c
                owner_probs_log.append(owner_probs.cpu())
            
            cust_idx, _ = learner.step(env)
            env.step(cust_idx)
    
    # Compute metrics
    if owner_probs_log:
        all_probs = torch.stack(owner_probs_log)  # T x N x L_v x L_c
        # Ownership entropy: lower = more decisive
        p = all_probs.clamp_min(1e-9)
        entropy = -(p * p.log()).sum(dim=2).mean(dim=0).mean()  # avg over T, N, customers
        return {"ownership_entropy": entropy.item()}
    return {}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("data/test_sets/test_dvrptw_n50m3_id_1000.pyth", map_location="cpu", weights_only=False)
    
    models_to_compare = {
        "COAST": "output/ablation/vectra/seed42/chkpt_best.pyth",
        "B1-Memory": "output/ablation/b1/seed42/chkpt_best.pyth",
        "B0-None": "output/ablation/b0/seed42/chkpt_best.pyth",
    }
    
    results = {}
    for name, ckpt in models_to_compare.items():
        if not os.path.exists(ckpt):
            print(f"Skip {name}: {ckpt} not found")
            continue
        learner = load_model(ckpt, device)
        metrics = compute_ownership_metrics(learner, data, device)
        results[name] = metrics
        print(f"{name}: {metrics}")
    
    with open("output/eval_results/h1_coordination.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

### 6.2 H2 — Anticipation Analysis

Create `script/eval_h2_anticipation.py`:

```python
#!/usr/bin/env python3
"""H2: Candidate-conditioned lookahead reduces myopia."""
import torch, os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from MODEL.model import VECTRA
from problems import DVRPTW_Environment

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    learner = VECTRA(
        7, 4, model_size=128, layer_count=2, head_count=4, ff_size=256,
        tanh_xplor=10, greedy=True, memory_size=128, lookahead_hidden=128,
        dropout=0.1, use_edge_features=args.get("use_edge_features", True),
        use_memory=args.get("use_memory", True),
        use_ownership=args.get("use_ownership", True),
        use_lookahead=args.get("use_lookahead", True),
        fusion_mode=args.get("fusion_mode", "mlp"),
    )
    learner.load_state_dict(checkpoint["model"], strict=False)
    learner.to(device); learner.eval()
    return learner

def analyze_lookahead(learner, data, device):
    """Track when lookahead overrides attention-only choice."""
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    interventions = 0
    total_decisions = 0
    lookahead_preds = []
    actual_future_costs = []
    
    with torch.no_grad():
        while not env.done:
            if env.new_customers:
                learner._encode_customers(env.nodes, env.cust_mask)
            
            # Get scores before selection
            veh_repr = learner._repr_vehicle(env.vehicles, env.cur_veh_idx, env.mask)
            edge_emb = learner._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)
            
            # Attention-only score
            att_score = learner.cross_fusion(veh_repr, learner.cust_repr, edge_emb)
            
            if learner.use_lookahead:
                lookahead = learner._compute_lookahead(veh_repr, learner.cust_repr, edge_emb)
            
            owner_bias = learner._compute_owner_bias(env.cur_veh_idx)
            final_score = learner._score_customers(veh_repr, learner.cust_repr, edge_emb, owner_bias, lookahead if learner.use_lookahead else None)
            
            # Check if lookahead changes the decision
            mask = env.cur_veh_mask
            att_masked = att_score.clone(); att_masked[mask] = -float('inf')
            final_masked = final_score.clone(); final_masked[mask] = -float('inf')
            
            att_choice = att_masked.argmax(dim=-1)
            final_choice = final_masked.argmax(dim=-1)
            
            if not mask.all():
                total_decisions += (~mask.all(dim=-1)).sum().item()
                interventions += (att_choice != final_choice).sum().item()
            
            if learner.use_lookahead:
                chosen_lookahead = lookahead.gather(2, final_choice.unsqueeze(1).unsqueeze(1))
                lookahead_preds.append(chosen_lookahead.squeeze().cpu())
            
            cust_idx, _ = learner.step(env)
            env.step(cust_idx)
    
    results = {
        "intervention_rate": interventions / max(total_decisions, 1),
        "total_decisions": total_decisions,
        "interventions": interventions,
    }
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("data/test_sets/test_dvrptw_n50m3_id_1000.pyth", map_location="cpu", weights_only=False)
    
    models = {
        "COAST": "output/ablation/vectra/seed42/chkpt_best.pyth",
        "B3-Look": "output/ablation/b3/seed42/chkpt_best.pyth",
        "B0-None": "output/ablation/b0/seed42/chkpt_best.pyth",
    }
    
    results = {}
    for name, ckpt in models.items():
        if not os.path.exists(ckpt):
            continue
        learner = load_model(ckpt, device)
        metrics = analyze_lookahead(learner, data, device)
        results[name] = metrics
        print(f"{name}: intervention_rate={metrics['intervention_rate']:.2%}")
    
    with open("output/eval_results/h2_anticipation.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## 📍 STEP 7: OOD Evaluation (Day 14-16)

```bash
cd /home/admin_wsl/projects/mardam-master

# Evaluate all models on each OOD test set
for ood in n100m5_ood n50m6_ood n50m3_tight_ood n50m3_burst_ood n50m3_sparse_ood; do
    echo "=== OOD: ${ood} ==="
    python script/eval_unified.py \
        --test-data "data/test_sets/test_dvrptw_${ood}_500.pyth" \
        --models-dir output/ablation \
        --output "output/eval_results/ood_${ood}.json"
done

# Also evaluate AM and PolyNet on OOD
for ood in n100m5_ood n50m6_ood n50m3_tight_ood n50m3_burst_ood n50m3_sparse_ood; do
    echo "=== Lit Baselines OOD: ${ood} ==="
    python script/eval_literature.py \
        --test-data "data/test_sets/test_dvrptw_${ood}_500.pyth" \
        --output "output/eval_results/lit_ood_${ood}.json"
done
```

---

## 📍 STEP 8: Generate Final Results Table (Day 16-17)

Create `script/generate_tables.py`:

```python
#!/usr/bin/env python3
"""Aggregate all evaluation results into publication-ready tables."""
import json, os, glob
import numpy as np
from scipy import stats

def load_json(path):
    with open(path) as f:
        return json.load(f)

def format_mean_std(mean, std, precision=2):
    return f"{mean:.{precision}f} ± {std:.{precision}f}"

def paired_ttest(costs_a, costs_b):
    """Paired t-test between two lists of per-seed mean costs."""
    t, p = stats.ttest_rel(costs_a, costs_b)
    return t, p

def significance_marker(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def main():
    results_dir = "output/eval_results"
    
    # Load in-dist results
    in_dist = load_json(f"{results_dir}/in_dist.json")
    lit = load_json(f"{results_dir}/literature_baselines.json")
    
    # Sort by mean cost
    sorted_models = sorted(in_dist.items(), key=lambda x: x[1]["mean"])
    
    print("\n" + "="*80)
    print("TABLE 1: In-Distribution Results (DVRPTW, n=50, m=3)")
    print("="*80)
    print(f"{'Model':<15s} {'Cost':>20s} {'Gap to Best':>12s} {'p-value':>10s}")
    print("-"*60)
    
    best_cost = sorted_models[0][1]["mean"]
    best_name = sorted_models[0][0]
    
    for name, stats in sorted_models:
        gap = (stats["mean"] - best_cost) / best_cost * 100
        marker = ""
        if name != best_name and "per_seed_means" in stats and "per_seed_means" in in_dist[best_name]:
            _, p = paired_ttest(stats["per_seed_means"], in_dist[best_name]["per_seed_means"])
            marker = significance_marker(p)
        print(f"{name:<15s} {format_mean_std(stats['mean'], stats['std']):>20s} {gap:>+11.1f}% {marker:>10s}")
    
    # Add literature baselines
    for name, stats in lit.items():
        gap = (stats["mean"] - best_cost) / best_cost * 100
        print(f"{name:<15s} {format_mean_std(stats['mean'], stats['std']):>20s} {gap:>+11.1f}%")
    
    # OOD summary
    print("\n" + "="*80)
    print("TABLE 2: OOD Generalization (Cost Degradation %)")
    print("="*80)
    
    ood_files = sorted(glob.glob(f"{results_dir}/ood_*.json"))
    ood_names = [os.path.basename(f).replace("ood_","").replace(".json","") for f in ood_files]
    
    header = f"{'Model':<15s}"
    for ood_name in ood_names:
        header += f" {ood_name:>18s}"
    print(header)
    print("-"*len(header))
    
    for name in ["COAST", "B0-None", "B1-Memory", "B3-Look", "B5-Linear", "EdgeOff"]:
        if name not in in_dist:
            continue
        id_cost = in_dist[name]["mean"]
        row = f"{name:<15s}"
        for ood_name, ood_file in zip(ood_names, ood_files):
            ood_data = load_json(ood_file)
            if name in ood_data:
                ood_cost = ood_data[name]["mean"]
                deg = (ood_cost - id_cost) / id_cost * 100
                row += f" {deg:>+17.1f}%"
            else:
                row += f" {'—':>18s}"
        print(row)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
```

---

## 📋 QUICK REFERENCE: All Scripts

| Script | Purpose | Phase |
|--------|---------|-------|
| `script/train_all_seeds.sh` | Multi-seed training (30 runs) | Phase 1 |
| `script/eval_unified.py` | Unified evaluation for COAST + ablations | Phase 2, 7 |
| `script/eval_literature.py` | Evaluate AM + PolyNet | Phase 3 |
| `script/eval_ortools.py` | OR-Tools static lower bound | Phase 3 |
| `script/eval_h1_coordination.py` | H1: Ownership/conflict metrics | Phase 4 |
| `script/eval_h2_anticipation.py` | H2: Lookahead intervention analysis | Phase 4 |
| `script/generate_tables.py` | Aggregate results into tables | Phase 6 |

---

## ⚡ TROUBLESHOOTING

### Checkpoint loading fails

```python
# Always use strict=False
learner.load_state_dict(checkpoint["model"], strict=False)
```

### CUDA out of memory

```bash
# Reduce batch size
--batch-size 256 --iter-count 2000

# Or use gradient accumulation (manual)
```

### NaN in training

```bash
# Reduce learning rate
--learning-rate 5e-5

# Increase grad clip
--max-grad-norm 1.0

# Disable AMP temporarily
# (remove --amp flag)
```

### Model architecture mismatch

```bash
# Always specify all architecture params explicitly:
--model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
--memory-size 128 --lookahead-hidden 128
```

---

**END OF IMPLEMENTATION GUIDE**
