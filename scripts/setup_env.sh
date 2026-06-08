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
    echo "[WARN] No CUDA GPU found."
fi

# ---- Core packages ----
echo ""
echo "Installing core packages..."

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision

pip install \
    monai \
    timm \
    nibabel \
    scikit-learn \
    scipy \
    pandas \
    numpy \
    openpyxl \
    setproctitle

# ---- Optional: ANTsPy (may fail on some platforms) ----
echo ""
echo "Installing ANTsPy (optional, may fail on some platforms)..."
pip install antspyx 2>/dev/null && echo "[OK] ANTsPy installed" || \
    echo "[WARN] ANTsPy not available — will use nibabel-based registration instead."

echo ""
echo "========================================"
echo "  Verifying installation..."
echo "========================================"

python -c "
import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')
import monai; print(f'MONAI {monai.__version__}')
import nibabel; print(f'nibabel {nibabel.__version__}')
import timm; print(f'timm {timm.__version__}')
import sklearn; print(f'scikit-learn {sklearn.__version__}')
import scipy; print(f'scipy {scipy.__version__}')

# ANTsPy is optional
try:
    import ants; print(f'ANTsPy {ants.__version__}')
except ImportError:
    print('ANTsPy: not installed (will use nibabel fallback)')

print('All core packages OK!')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Register data:    bash scripts/preprocess.sh"
echo "  2. Run experiments:  bash scripts/run_all.sh"
