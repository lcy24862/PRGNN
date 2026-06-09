#!/bin/bash
# =============================================================================
# PRGNN: Run training + testing on ALL tracers, all tasks, all folds.
#
# Usage:
#   bash scripts/run_all_tracers.sh                    # all 4 tracers
#   bash scripts/run_all_tracers.sh --tracer 18F-AV45  # single tracer
#   bash scripts/run_all_tracers.sh --dry-run           # print commands only
#   bash scripts/run_all_tracers.sh --epochs 100 --batch_size 8
# =============================================================================
set -e

# ---- Tracer-specific task lists (from data_utils.TRACER_TASKS) ----
# 18F-FDG:     AD_HC, HC_MCI, EMCI_LMCI, HC_ALL_MCI
# 18F-FBB:     HC_MCI
# 18F-AV45:    AD_HC, HC_MCI, EMCI_LMCI, HC_ALL_MCI
# 18F-AV1451:  AD_HC, HC_MCI
# Amyloid:     HC_MCI, HC_ALL_MCI   (merged AV45 + FBB)

declare -A TRACER_TASK_MAP
TRACER_TASK_MAP["18F-FDG"]="AD_HC HC_MCI EMCI_LMCI HC_ALL_MCI"
TRACER_TASK_MAP["18F-FBB"]="HC_MCI"
TRACER_TASK_MAP["18F-AV45"]="AD_HC HC_MCI EMCI_LMCI HC_ALL_MCI"
TRACER_TASK_MAP["18F-AV1451"]="AD_HC HC_MCI"
TRACER_TASK_MAP["Amyloid"]="HC_MCI HC_ALL_MCI"

# ---- Defaults ----
TRACERS=("18F-FDG" "18F-FBB" "18F-AV45" "18F-AV1451" "Amyloid")
SELECTED_TRACER=""
MODEL="PRGNN_ti"
DATA_DIR="data_registered"
EPOCHS=100
BATCH_SIZE=8
LR=0.0005
GPU=0
DRY_RUN=false

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --tracer) SELECTED_TRACER="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Override tracer list if single tracer specified
if [ -n "${SELECTED_TRACER}" ]; then
    TRACERS=("${SELECTED_TRACER}")
fi

echo "=============================================="
echo "  PRGNN Multi-Tracer Experiment Runner"
echo "  Model:   ${MODEL}"
echo "  Tracers: ${TRACERS[*]}"
echo "  Data:    ${DATA_DIR}"
echo "  Epochs:  ${EPOCHS}"
echo "  GPU:     ${GPU}"
echo "=============================================="

TOTAL_START=$(date +%s)

for TRACER in "${TRACERS[@]}"; do
    TASKS_STR="${TRACER_TASK_MAP[${TRACER}]}"
    echo ""
    echo "##################################################"
    echo "  TRACER: ${TRACER}"
    echo "  Tasks:  ${TASKS_STR}"
    echo "##################################################"

    TRACER_START=$(date +%s)

    for TASK in ${TASKS_STR}; do
        TASK_MODEL_DIR="models/${TRACER}/${TASK}"
        mkdir -p "${TASK_MODEL_DIR}"

        # ---- Training ----
        for FOLD in $(seq 0 4); do
            BEST_PTH="${TASK_MODEL_DIR}/${MODEL}_fold${FOLD}_best.pth"

            if [ -f "${BEST_PTH}" ]; then
                echo "[${TRACER}/${TASK}] Fold ${FOLD}: SKIP (best model exists)"
                continue
            fi

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

            echo "[${TRACER}/${TASK}] Fold ${FOLD}: ${CMD}"

            if [ "${DRY_RUN}" = false ]; then
                eval ${CMD}
            fi
        done

        # ---- Testing ----
        RESULT_FILE="${TASK_MODEL_DIR}/test_results.txt"
        if [ -f "${RESULT_FILE}" ]; then
            echo "[${TRACER}/${TASK}] Test: SKIP (results exist)"
        else
            CMD="python test.py \
                --model ${MODEL} \
                --tracer ${TRACER} \
                --task ${TASK} \
                --data_dir ${DATA_DIR} \
                --gpu ${GPU} \
                --which_model best"

            echo "[${TRACER}/${TASK}] Test: ${CMD}"

            if [ "${DRY_RUN}" = false ]; then
                eval ${CMD}
            fi
        fi

        # ---- Copy results ----
        RESULTS_DIR="results/${TRACER}"
        mkdir -p "${RESULTS_DIR}"
        if [ -f "${RESULT_FILE}" ]; then
            cp "${RESULT_FILE}" "${RESULTS_DIR}/results_${TASK}.txt"
            echo "  -> ${RESULTS_DIR}/results_${TASK}.txt"
        fi
    done

    TRACER_ELAPSED=$(($(date +%s) - TRACER_START))
    echo "[${TRACER}] Done in $((TRACER_ELAPSED / 60))m $((TRACER_ELAPSED % 60))s"
done

TOTAL_ELAPSED=$(($(date +%s) - TOTAL_START))
echo ""
echo "=============================================="
echo "  ALL DONE! Total: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "  Results summary: results/"
echo ""
echo "  Run collect_results.py to gather all results:"
echo "    python collect_results.py"
echo "=============================================="
