#!/bin/bash
# =============================================================================
# PRGNN Data Preprocessing
# Register PET images to FDG template space (91x109x91).
#
# Usage:
#   bash scripts/preprocess.sh                      # auto-detect best method
#   bash scripts/preprocess.sh --method ants        # force ANTsPy (SyN/affine)
#   bash scripts/preprocess.sh --method simple      # nibabel-based (no ANTsPy)
#   bash scripts/preprocess.sh --workers 8
# =============================================================================
set -e

# ---- Defaults ----
METHOD="auto"
WORKERS=8
REGISTER_ARGS=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --tracer) REGISTER_ARGS="--tracer $2"; shift 2 ;;
        --dry-run) REGISTER_ARGS="$REGISTER_ARGS --dry-run"; shift ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

echo "========================================"
echo "  PRGNN Data Registration"
echo "  Method:  ${METHOD}"
echo "  Workers: ${WORKERS}"
echo "========================================"

# Check template
if [ ! -f "template/TEMPLATE_FDGPET_100.nii" ]; then
    echo "ERROR: template/TEMPLATE_FDGPET_100.nii not found"
    exit 1
fi

# Check raw data
if [ ! -d "data" ]; then
    echo "ERROR: data/ not found"
    exit 1
fi

TOTAL=$(find data -name '*.nii.gz' \
    -not -path '*/AD_HC/*' -not -path '*/HC_MCI/*' \
    -not -path '*/EMCI_LMCI/*' -not -path '*/HC_ALL_MCI/*' | wc -l)
echo "Images to register: ${TOTAL}"

# ---- Detect method ----
if [ "$METHOD" = "auto" ]; then
    if python -c "import ants" 2>/dev/null; then
        METHOD="ants"
        echo "ANTsPy detected -> using ANTs affine registration"
    else
        METHOD="simple"
        echo "ANTsPy not available -> using nibabel center-crop"
    fi
fi

# ---- Run ----
if [ "$METHOD" = "ants" ]; then
    python register_all.py --run --method Affine --workers "${WORKERS}" ${REGISTER_ARGS}
else
    python register_simple.py --run --workers "${WORKERS}" ${REGISTER_ARGS}
fi

echo ""
echo "========================================"
echo "  Preprocessing complete!"
echo "  Output: data_registered/"
echo "========================================"
