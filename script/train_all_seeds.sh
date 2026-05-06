#!/bin/bash
# =============================================================================
# COAST Multi-Seed Training Script
# Usage:
#   Single GPU:  bash script/train_all_seeds.sh
#   Multi-GPU:   GPU_ID=0 bash script/train_all_seeds.sh
#                 GPU_ID=1 bash script/train_all_seeds.sh  (in another terminal)
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============ CONFIGURATION ============
MODELS=("vectra" "b0" "b1" "b3" "b5" "edgeoff")
SEEDS=(42 123 456 789 1024)
GPU_ID=${GPU_ID:-0}

# Common training params
EPOCHS=500
ITERS=1000
BATCH=512
LR=0.0001
N_CUST=50
N_VEH=3

# ============ TRAINING LOOP ============
TOTAL=$(( ${#MODELS[@]} * ${#SEEDS[@]} ))
CURRENT=0

for profile in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        OUTPUT_DIR="output/ablation/${profile}/seed${seed}"
        
        if [ -f "${OUTPUT_DIR}/chkpt_best.pyth" ]; then
            echo "[${CURRENT}/${TOTAL}] SKIP ${profile} seed=${seed} — already exists"
            continue
        fi
        
        echo ""
        echo "============================================================"
        echo "[${CURRENT}/${TOTAL}] TRAINING: profile=${profile} seed=${seed}"
        echo "Output: ${OUTPUT_DIR}"
        echo "GPU: ${GPU_ID}"
        echo "============================================================"
        
        CUDA_VISIBLE_DEVICES=${GPU_ID} python MODEL/train.py \
            --problem-type dvrptw \
            --customers-count ${N_CUST} --vehicles-count ${N_VEH} \
            --model-size 128 --layer-count 2 --head-count 4 --ff-size 256 \
            --memory-size 128 --lookahead-hidden 128 \
            --ablation-profile "${profile}" \
            --epoch-count ${EPOCHS} --iter-count ${ITERS} --batch-size ${BATCH} \
            --learning-rate ${LR} --baseline-type critic --adv-norm \
            --entropy-coef 0.01 --max-grad-norm 2.0 --amp \
            --rng-seed "${seed}" \
            --output-dir "${OUTPUT_DIR}" \
            --test-batch-size 10240
        
        echo "[${CURRENT}/${TOTAL}] DONE: ${profile} seed=${seed}"
    done
done

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE (${TOTAL} runs)"
echo "============================================================"
