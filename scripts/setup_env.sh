#!/bin/bash
# =============================================================================
# PRGNN Environment Setup
# Usage: bash scripts/setup_env.sh
# =============================================================================
set -e

echo "========================================"
echo "  PRGNN Environment Setup"
echo "========================================"

# Detect CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] CUDA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "[WARN] No CUDA GPU found. Will install CPU-only PyTorch."
fi

# ---- Python packages ----
echo ""
echo "Installing Python packages..."

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision

pip install \
    monai \
    timm \
    nibabel \
    scikit-learn \
    pandas \
    numpy \
    openpyxl \
    setproctitle \
    antspyx

echo ""
echo "========================================"
echo "  Verifying installation..."
echo "========================================"

python -c "
import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')
import monai; print(f'MONAI {monai.__version__}')
import nibabel; print(f'nibabel {nibabel.__version__}')
import timm; print(f'timm {timm.__version__}')
import ants; print(f'ANTsPy {ants.__version__}')
import sklearn; print(f'scikit-learn {sklearn.__version__}')
print('All packages OK!')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Register data:    bash scripts/preprocess.sh"
echo "  2. Run experiments:  bash scripts/run_all.sh"
