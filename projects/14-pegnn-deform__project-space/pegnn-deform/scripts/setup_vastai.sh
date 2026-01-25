#!/bin/bash
# PEGNN-Deform: vast.ai Instance Setup Script
# 
# This script sets up a fresh vast.ai instance with all dependencies.
# Designed for: NVIDIA L40S (48GB VRAM)
# Base image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
#
# Usage: bash scripts/setup_vastai.sh

set -e  # Exit on error

echo "=============================================="
echo "PEGNN-Deform: vast.ai Instance Setup"
echo "=============================================="

# Print system info
echo ""
echo "System Information:"
echo "-------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install PyTorch Geometric and dependencies
# Using pre-built wheels for PyTorch 2.1.0 + CUDA 11.8
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install PyG extensions from wheels
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
echo ""
echo "Installing additional dependencies..."
pip install wandb numpy scipy matplotlib tqdm pytest trimesh

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import torch_geometric
print(f'PyTorch Geometric version: {torch_geometric.__version__}')

# Test basic operations
from torch_geometric.nn import MessagePassing
print('MessagePassing import: OK')
"

# Set up workspace
echo ""
echo "Setting up workspace..."
cd /workspace

# If project files exist, run tests
if [ -d "pegnn-deform" ]; then
    echo ""
    echo "Project found. Running setup..."
    cd pegnn-deform
    
    # Generate synthetic data if needed
    if [ ! -d "data/synthetic" ] || [ -z "$(ls -A data/synthetic 2>/dev/null)" ]; then
        echo ""
        echo "Generating synthetic training data..."
        python data/generate_synthetic.py \
            --output-dir data/synthetic \
            --grid-size 20 \
            --num-train 50 \
            --num-val 10 \
            --num-test 10 \
            --steps 20
    fi
    
    echo ""
    echo "Setup complete! Run tests with:"
    echo "  cd /workspace/pegnn-deform"
    echo "  bash scripts/run_tests.sh"
else
    echo ""
    echo "Project directory not found."
    echo "Please copy project files to /workspace/pegnn-deform"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
