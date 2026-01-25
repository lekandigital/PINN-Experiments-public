#!/bin/bash
# =============================================================================
# Remote Setup Script for NIF-Cloth4D-Temporal
# =============================================================================
# Run this script after connecting to a Vast.ai instance
# Sets up the environment and verifies GPU availability
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/nif-cloth4d-temporal/main/scripts/setup_remote.sh | bash
#   # Or after cloning:
#   ./scripts/setup_remote.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  NIF-Cloth4D-Temporal Remote Setup             ${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo -e "${RED}Error: environment.yml not found.${NC}"
    echo "Please run this script from the nif-cloth4d-temporal directory."
    exit 1
fi

# 1. System info
echo -e "${YELLOW}[1/6] System Information${NC}"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo ""

# 2. Check NVIDIA driver and CUDA
echo -e "${YELLOW}[2/6] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
else
    echo -e "${RED}nvidia-smi not found. GPU may not be available.${NC}"
fi
echo ""

# 3. Create/update conda environment
echo -e "${YELLOW}[3/6] Setting up Conda environment...${NC}"
if command -v conda &> /dev/null; then
    # Check if environment exists
    if conda env list | grep -q "nif_cloth4d"; then
        echo "Environment 'nif_cloth4d' already exists. Updating..."
        conda env update -f environment.yml --prune
    else
        echo "Creating environment 'nif_cloth4d'..."
        conda env create -f environment.yml
    fi
    
    # Activate environment
    echo "Activating environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate nif_cloth4d
else
    echo -e "${RED}Conda not found. Installing dependencies with pip...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy scipy h5py pyvista trimesh wandb matplotlib pytest
fi
echo ""

# 4. Verify PyTorch and CUDA
echo -e "${YELLOW}[4/6] Verifying PyTorch installation...${NC}"
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")
else:
    print("WARNING: CUDA not available!")
EOF
echo ""

# 5. Run CI sanity check
echo -e "${YELLOW}[5/6] Running CI sanity check...${NC}"
python tests/ci_sanity_check.py
echo ""

# 6. Run inference speed test
echo -e "${YELLOW}[6/6] Running inference speed test...${NC}"
python scripts/test_inference.py --num_frames 30 --warmup_frames 5
echo ""

# Summary
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Setup Complete!                               ${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Quick commands:"
echo "  # Generate synthetic dataset"
echo "  python -c \"from src.data import generate_synthetic_dataset; generate_synthetic_dataset()\""
echo ""
echo "  # Train model (quick test)"
echo "  python scripts/train.py --config fast_dev --use_synthetic"
echo ""
echo "  # Train model (full, with GRU)"
echo "  python scripts/train.py --epochs 100 --use_gru --use_synthetic"
echo ""
echo "  # Train with W&B logging"
echo "  wandb login"
echo "  python scripts/train.py --epochs 100 --use_gru --use_wandb --use_synthetic"
echo ""
echo "  # Test inference speed"
echo "  python scripts/test_inference.py --hidden_dim 256 --num_layers 5 --use_gru"
echo ""
