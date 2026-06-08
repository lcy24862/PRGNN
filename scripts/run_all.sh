#!/bin/bash
# =============================================================================
# PRGNN: Run training + testing on all tasks and folds
#
# Usage:
#   bash scripts/run_all.sh                              # all default tasks
#   bash scripts/run_all.sh --tracer 18F-FDG --model PRGNN_ti
#   bash scripts/run_all.sh --task AD_HC                 # single task only
#   bash scripts/run_all.sh --dry-run                    # print commands only
# =============================================================================
set -e

# ---- Defaults ----
MODEL="PRGNN_ti"
TRACER="18F-FDG"
TASKS=("AD_HC" "HC_MCI" "EMCI_LMCI" "HC_ALL_MCI")
DATA_DIR="data_registered"
EPOCHS=100
BATCH_SIZE=8
LR=0.0005
GPU=0
FOLDS=5
DRY_RUN=false

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tracer) TRACER="$2"; shift 2 ;;
        --task) TASKS=("$2"); shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  PRGNN Experiment Runner"
echo "  Model:   ${MODEL}"
echo "  Tracer:  ${TRACER}"
echo "  Tasks:   ${TASKS[*]}"
echo "  Folds:   0-$(($FOLDS - 1))"
echo "  Data:    ${DATA_DIR}"
echo "  Epochs:  ${EPOCHS}"
echo "  GPU:     ${GPU}"
echo "========================================"

RESULTS_DIR="results/${TRACER}"
mkdir -p "${RESULTS_DIR}"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "########## Task: ${TASK} ##########"

    TASK_MODEL_DIR="models/${TASK}"
    mkdir -p "${TASK_MODEL_DIR}"

    # ---- Training ----
    for FOLD in $(seq 0 $((FOLDS - 1))); do
        CMD="python train.py \
            --model ${MODEL} \
            --tracer ${TRACER} \
            --task ${TASK} \
            --data_dir ${DATA_DIR} \
            --fold ${FOLD} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --gpu ${GPU}"

        echo "[Train] Fold ${FOLD}: ${CMD}"

        if [ "${DRY_RUN}" = false ]; then
            eval ${CMD}
        fi
    done

    # ---- Testing ----
    CMD="python test.py \
        --model ${MODEL} \
        --tracer ${TRACER} \
        --task ${TASK} \
        --data_dir ${DATA_DIR} \
        --gpu ${GPU} \
        --which_model best"

    echo ""
    echo "[Test] ${CMD}"

    if [ "${DRY_RUN}" = false ]; then
        eval ${CMD}
    fi

    # ---- Collect results ----
    if [ -f "models/${TASK}/test_results.txt" ]; then
        cp "models/${TASK}/test_results.txt" "${RESULTS_DIR}/results_${TASK}.txt"
        echo "Results saved to ${RESULTS_DIR}/results_${TASK}.txt"
    fi

done

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "  Results: ${RESULTS_DIR}/"
echo "========================================"
