#!/bin/bash
# =============================================================================
# Setup Amyloid dataset — merge 18F-AV45 + 18F-FBB into a combined "Amyloid"
# tracer. Uses symlinks to avoid copying files.
#
# Usage:
#   bash scripts/setup_amyloid.sh                        # create symlinks
#   bash scripts/setup_amyloid.sh --method copy           # copy files instead
#   bash scripts/setup_amyloid.sh --dry-run               # show what would happen
# =============================================================================
set -e

METHOD="symlink"   # symlink | copy
DRY_RUN=false
DATA_DIR="data"
SOURCES=("18F-AV45" "18F-FBB")
TARGET="Amyloid"
CLASSES=("HC" "AD" "MCI" "EMCI" "LMCI")

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

TARGET_DIR="${DATA_DIR}/${TARGET}"

echo "=============================================="
echo "  Amyloid Dataset Setup"
echo "  Target: ${TARGET_DIR}"
echo "  Sources: ${SOURCES[*]}"
echo "  Method: ${METHOD}"
echo "=============================================="

# Ensure target dir exists
if [ "$DRY_RUN" = false ]; then
    mkdir -p "${TARGET_DIR}"
fi
echo "Created: ${TARGET_DIR}"

# Link/copy class folders from each source tracer
for CLASS in "${CLASSES[@]}"; do
    # Collect all source directories that have this class
    SRC_DIRS=()
    for SRC in "${SOURCES[@]}"; do
        SRC_PATH="${DATA_DIR}/${SRC}/${CLASS}"
        if [ -d "$SRC_PATH" ]; then
            SRC_DIRS+=("$SRC_PATH")
        fi
    done

    if [ ${#SRC_DIRS[@]} -eq 0 ]; then
        echo "  [SKIP] ${CLASS}: no source directory found"
        continue
    fi

    TARGET_CLASS_DIR="${TARGET_DIR}/${CLASS}"

    if [ "$DRY_RUN" = false ]; then
        mkdir -p "${TARGET_CLASS_DIR}"
    fi

    for SRC_PATH in "${SRC_DIRS[@]}"; do
        echo "  ${CLASS}: linking files from ${SRC_PATH} ..."

        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY-RUN] Would process: ${SRC_PATH}/* -> ${TARGET_CLASS_DIR}/"
            continue
        fi

        # Process each file in source
        for SRC_FILE in "${SRC_PATH}"/*.nii "${SRC_PATH}"/*.nii.gz; do
            [ -f "$SRC_FILE" ] || continue
            BASENAME=$(basename "$SRC_FILE")
            DST_FILE="${TARGET_CLASS_DIR}/${BASENAME}"

            if [ -f "$DST_FILE" ] || [ -L "$DST_FILE" ]; then
                continue  # skip if already exists
            fi

            case $METHOD in
                symlink)
                    # Use relative symlink to stay portable
                    REL_PATH=$(realpath --relative-to="${TARGET_CLASS_DIR}" "$SRC_FILE" 2>/dev/null || echo "")
                    if [ -n "$REL_PATH" ]; then
                        ln -s "$REL_PATH" "$DST_FILE"
                    else
                        ln -s "$SRC_FILE" "$DST_FILE"
                    fi
                    ;;
                copy)
                    cp "$SRC_FILE" "$DST_FILE"
                    ;;
            esac
        done
    done

    # Count the result
    COUNT=$(find "${TARGET_CLASS_DIR}" -maxdepth 1 \( -name '*.nii' -o -name '*.nii.gz' \) 2>/dev/null | wc -l)
    echo "    -> ${COUNT} files in ${TARGET_CLASS_DIR}"
done

# ---- Copy fold CSVs from root Amyloid/ if they exist ----
ROOT_AMYLOID="Amyloid"
if [ -d "$ROOT_AMYLOID" ] && [ "$ROOT_AMYLOID" != "$TARGET_DIR" ]; then
    echo ""
    echo "Copying fold CSVs from ${ROOT_AMYLOID}/ to ${TARGET_DIR}/ ..."
    for TASK_DIR in "$ROOT_AMYLOID"/*/; do
        TASK_NAME=$(basename "$TASK_DIR")
        DST_TASK="${TARGET_DIR}/${TASK_NAME}"
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$DST_TASK"
            cp -n "$TASK_DIR"/*.csv "$DST_TASK/" 2>/dev/null || true
        fi
        echo "  ${TASK_NAME}/ -> ${DST_TASK}/"
    done
fi

echo ""
echo "=============================================="
echo "  Amyloid dataset setup complete!"
echo ""
echo "  Now run experiments with:"
echo "    bash scripts/run_all.sh --tracer Amyloid"
echo "    bash scripts/run_all_tracers.sh --tracer Amyloid"
echo "=============================================="
