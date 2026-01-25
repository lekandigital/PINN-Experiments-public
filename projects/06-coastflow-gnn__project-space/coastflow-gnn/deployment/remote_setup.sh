#!/bin/bash
# ==============================================================================
# Remote Setup Script for CoastFlow-GNN
# ==============================================================================
# Run this script on the Vast.ai (or any cloud GPU) instance after SSH login.
#
# This script:
#   1. Updates pip and installs dependencies
#   2. Installs PyTorch Geometric for the correct CUDA version
#   3. Verifies GPU is available
#   4. Runs basic tests to confirm setup
#
# Usage:
#   cd coastflow-gnn/deployment
#   chmod +x remote_setup.sh
#   ./remote_setup.sh
# ==============================================================================

set -e

echo "=============================================="
echo "CoastFlow-GNN Remote Environment Setup"
echo "=============================================="
echo ""

# Get CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2- || echo "unknown")
echo "📊 System Info:"
echo "   CUDA Version: $CUDA_VERSION"
echo "   Python: $(python --version)"
echo "   GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "   Unable to query GPU"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Determine PyG wheel URL based on CUDA version
if [[ "$CUDA_VERSION" == "12.1"* ]] || [[ "$CUDA_VERSION" == "12.2"* ]]; then
    PYG_WHEELS="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
    PYG_WHEELS="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
else
    PYG_WHEELS="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
    echo "⚠️  Unknown CUDA version, defaulting to CUDA 12.1 wheels"
fi

echo ""
echo "📦 Installing PyTorch Geometric..."
echo "   Using wheels from: $PYG_WHEELS"
pip install torch-geometric torch-scatter torch-sparse -f $PYG_WHEELS

echo ""
echo "📦 Installing other dependencies..."
pip install pyvista h5py matplotlib seaborn pandas numpy scipy pyyaml tqdm pytest ipykernel jupyter

echo ""
echo "📦 Installing development tools..."
pip install pytest-cov ipywidgets

# Verify installation
echo ""
echo "=============================================="
echo "Verifying Installation"
echo "=============================================="

python << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import torch_geometric
print(f"PyTorch Geometric: {torch_geometric.__version__}")

# Quick test
from torch_geometric.nn import GCNConv
conv = GCNConv(16, 32)
x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 30))
out = conv(x, edge_index)
print(f"GCN test: input {x.shape} -> output {out.shape}")
print("✓ All imports successful!")
EOF

echo ""
echo "=============================================="
echo "Testing CoastFlow-GNN"
echo "=============================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Running quick model test..."
python -c "
import torch
import sys
sys.path.insert(0, '.')
from src.models.coastflow_gnn import CoastFlowGNN

model = CoastFlowGNN(in_channels=6, hidden_channels=64, out_channels=4)
print(f'Model parameters: {model.count_parameters():,}')

x = torch.randn(100, 6)
edge_index = torch.randint(0, 100, (2, 300))
batch = torch.zeros(100, dtype=torch.long)

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    edge_index = edge_index.cuda()
    batch = batch.cuda()

out = model(x, edge_index, batch)
print(f'Forward pass: {x.shape} -> {out.shape}')
print('✓ Model test passed!')
"

echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "To start training:"
echo "  cd /workspace/coastflow-gnn"
echo "  python src/training/train_single.py --epochs 50 --batch-size 4"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "To monitor GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "=============================================="
