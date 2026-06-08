#!/bin/bash
# =============================================================================
# PRGNN Data Preprocessing
# Registers all PET images to the FDG template space (91×109×91).
# Usage: bash scripts/preprocess.sh [--method Affine|SyN] [--workers 8]
# =============================================================================
set -e

METHOD="${1:-Affine}"
WORKERS="${2:-8}"

echo "========================================"
echo "  PRGNN Data Registration"
echo "  Method:  ${METHOD}"
echo "  Workers: ${WORKERS}"
echo "========================================"

# Check template exists
if [ ! -f "template/TEMPLATE_FDGPET_100.nii" ]; then
    echo "ERROR: Template not found at template/TEMPLATE_FDGPET_100.nii"
    exit 1
fi

# Check raw data exists
if [ ! -d "data" ]; then
    echo "ERROR: data/ directory not found."
    echo "Please place the dataset under data/ before running."
    exit 1
fi

TOTAL=$(find data -name '*.nii.gz' -not -path '*/AD_HC/*' -not -path '*/HC_MCI/*' -not -path '*/EMCI_LMCI/*' -not -path '*/HC_ALL_MCI/*' | wc -l)
echo "Images to register: ${TOTAL}"
echo ""

# Run registration
python register_all.py --run --method "${METHOD}" --workers "${WORKERS}"

echo ""
echo "========================================"
echo "  Preprocessing complete!"
echo "  Output: data_registered/"
echo "========================================"
