#!/bin/bash
# Deployment script for vast.ai instance
# Run this after SSH'ing into the instance

set -e

echo "=========================================="
echo "GeoPINN-Manifold Deployment Script"
echo "=========================================="

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi

# Install dependencies (PyTorch should already be in the image)
echo ""
echo "Installing additional dependencies..."
pip install numpy scipy scikit-learn h5py matplotlib pyvista --quiet

# Set random seed
export PYTHONHASHSEED=42

# Create project directory
echo ""
echo "Setting up project..."
mkdir -p /workspace/GeoPINN-Manifold
cd /workspace/GeoPINN-Manifold

# Copy or clone code here...
# For now, we'll create a simple test script

cat > test_gpu.py << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Simple GPU test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.mm(x, y)
    print(f"GPU compute test: OK")
EOF

python test_gpu.py

echo ""
echo "=========================================="
echo "Setup complete! Ready to run tests."
echo "=========================================="
