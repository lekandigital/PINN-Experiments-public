#!/bin/bash
# HGNN-ClothDyn Environment Setup Script for VastAI
# Optimized for NVIDIA L40S (48GB VRAM)

set -e  # Exit on error

echo "=========================================="
echo "HGNN-ClothDyn Environment Setup"
echo "=========================================="
echo "Start time: $(date)"

# Detect CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' | cut -d. -f1,2)
if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="12.1"
    echo "CUDA version not detected, defaulting to $CUDA_VERSION"
else
    echo "Detected CUDA version: $CUDA_VERSION"
fi

# Update pip
echo "[1/5] Updating pip..."
pip install --upgrade pip

# Install PyTorch (should already be installed in pytorch image)
echo "[2/5] Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install PyTorch Geometric dependencies
echo "[3/5] Installing PyTorch Geometric dependencies..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "PyTorch version: $TORCH_VERSION"

# Install based on CUDA version
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu121.html
pip install torch-geometric

# Install other dependencies
echo "[4/5] Installing additional dependencies..."
pip install numpy scipy h5py matplotlib tqdm pyyaml

# Verify installation
echo "[5/5] Verifying installation..."
python -c "
import torch
import torch_geometric
import numpy as np
import h5py
import matplotlib
import scipy

print('='*50)
print('Installation Verification')
print('='*50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print('='*50)
print('All dependencies installed successfully!')
"

echo "=========================================="
echo "Setup Complete: $(date)"
echo "=========================================="
