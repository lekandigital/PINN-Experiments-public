#!/bin/bash
# Environment setup script for Vast.ai instance
# Run this script after SSHing into your Vast.ai instance
#
# Usage: bash setup_environment.sh

set -e
echo "🔧 Setting up ClothGeom-NIF environment..."
echo "==========================================="
echo ""

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git wget vim tmux htop > /dev/null 2>&1
echo "✅ System packages installed"

# Verify GPU
echo ""
echo "🖥️ GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Check CUDA
echo "CUDA Version:"
nvcc --version 2>/dev/null || echo "nvcc not in PATH (using container CUDA)"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch (may already be installed in container)
echo "📦 Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
}

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')"

# Install project dependencies
echo ""
echo "📦 Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo "✅ Dependencies installed from requirements.txt"
else
    # Install individually if no requirements.txt
    pip install numpy>=1.24.0 h5py>=3.8.0 scikit-image>=0.21.0 matplotlib>=3.7.0 trimesh>=3.21.0 tqdm>=4.65.0 pyyaml -q
    echo "✅ Core dependencies installed"
    
    # Try to install pytorch3d (may fail on some systems)
    echo "📦 Installing PyTorch3D (this may take a few minutes)..."
    pip install pytorch3d -q 2>/dev/null || {
        echo "⚠️ PyTorch3D installation failed. Installing from source..."
        pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" -q 2>/dev/null || {
            echo "⚠️ PyTorch3D source installation failed. Metrics will be limited."
        }
    }
fi

# Create output directories
echo ""
echo "📁 Creating project directories..."
mkdir -p outputs checkpoints data/generated logs
echo "✅ Directories created"

# Final verification
echo ""
echo "==========================================="
echo "🔍 Environment Verification:"
echo ""
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import numpy as np
print(f'NumPy: {np.__version__}')

import h5py
print(f'h5py: {h5py.__version__}')

import skimage
print(f'scikit-image: {skimage.__version__}')

try:
    import pytorch3d
    print(f'PyTorch3D: Available')
except ImportError:
    print(f'PyTorch3D: Not available (metrics limited)')

print()
print('✅ All core dependencies verified!')
"

echo ""
echo "==========================================="
echo "✅ Environment setup complete!"
echo ""
echo "🚀 Ready to train. Run:"
echo "   python generate_dataset.py --num_samples 100"
echo "   python train.py --epochs 100"
echo "   python demo.py"
echo ""
echo "💡 Tip: Use tmux to keep training running after disconnect:"
echo "   tmux new -s train"
echo "   python train.py --epochs 100"
echo "   # Press Ctrl+B, then D to detach"
echo "   # Reconnect with: tmux attach -t train"
