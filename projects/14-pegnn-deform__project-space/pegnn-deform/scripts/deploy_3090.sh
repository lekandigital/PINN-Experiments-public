#!/bin/bash
# PEGNN-Deform: RTX 3090 Deployment Script
#
# Deploy and run PEGNN-Deform on home RTX 3090 server
# Target: 192.168.86.152 (24GB VRAM)
#
# Usage: bash scripts/deploy_3090.sh [--skip-sync] [--skip-env]

set -e

# Configuration
REMOTE_USER="o"
REMOTE_HOST="192.168.86.152"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
REMOTE_DIR="~/pegnn-deform"
LOCAL_DIR="$(dirname "$0")/.."

# Parse arguments
SKIP_SYNC=false
SKIP_ENV=false
RUN_TESTS=true
RUN_TRAINING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --train)
            RUN_TRAINING=true
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "PEGNN-Deform: RTX 3090 Deployment"
echo "=============================================="
echo "Remote: $REMOTE"
echo "Local:  $LOCAL_DIR"
echo ""

# Step 1: Test SSH connection
echo "[1/6] Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 "$REMOTE" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "ERROR: Cannot connect to $REMOTE"
    echo "Make sure the remote machine is accessible and SSH is configured"
    exit 1
fi

# Step 2: Check GPU on remote
echo "[2/6] Checking GPU on remote..."
ssh "$REMOTE" "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
echo ""

# Step 3: Create project directory
echo "[3/6] Creating project directory on remote..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

# Step 4: Sync project files
if [ "$SKIP_SYNC" = false ]; then
    echo "[4/6] Syncing project files..."
    rsync -avz --progress \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='checkpoints/*.pth' \
        --exclude='data/synthetic' \
        --exclude='wandb' \
        --exclude='.venv' \
        --exclude='*.egg-info' \
        "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"
    echo "Sync complete."
else
    echo "[4/6] Skipping sync (--skip-sync)"
fi

# Step 5: Setup environment on remote
if [ "$SKIP_ENV" = false ]; then
    echo "[5/6] Setting up Python environment..."
    ssh "$REMOTE" << 'REMOTE_SCRIPT'
cd ~/pegnn-deform

# Check Python version
echo "Python version:"
python3 --version

# Install dependencies with pip
pip3 install --upgrade pip

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip3 install torch_geometric

# Install other dependencies
pip3 install numpy scipy matplotlib tqdm pytest wandb h5py trimesh

# Verify PyTorch + CUDA
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
REMOTE_SCRIPT
else
    echo "[5/6] Skipping environment setup (--skip-env)"
fi

# Step 6: Run tests or training
echo "[6/6] Running on remote..."

if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "Running unit tests..."
    ssh "$REMOTE" << 'REMOTE_SCRIPT'
cd ~/pegnn-deform
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run pytest
python3 -m pytest tests/test_model.py -v --tb=short 2>&1 | head -100

echo ""
echo "Running quick model test..."
python3 -c "
import torch
import sys
sys.path.insert(0, 'src')
from pegdeform_model import PEGNNDeform, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Create model
model = PEGNNDeform(hidden_size=64, num_mp_layers=3).to(device)
print(f'Model parameters: {count_parameters(model):,}')

# Test with larger mesh (RTX 3090 optimized)
N, E = 2500, 15000
pos = torch.randn(N, 3, device=device)
vel = torch.randn(N, 3, device=device) * 0.1
edge_index = torch.randint(0, N, (2, E), device=device)
edge_attr = torch.rand(E, 2, device=device)
edge_attr[:, 0] += 1
edge_attr[:, 1] += 0.1

# Forward pass with AMP
with torch.amp.autocast('cuda', enabled=True):
    new_pos, new_vel, hidden = model(pos, vel, edge_index, edge_attr)

print(f'Input:  pos={pos.shape}, vel={vel.shape}')
print(f'Output: pos={new_pos.shape}, vel={new_vel.shape}')

# Memory usage
print(f'GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
print('All tests passed!')
"
REMOTE_SCRIPT
fi

if [ "$RUN_TRAINING" = true ]; then
    echo ""
    echo "Starting training on RTX 3090..."
    ssh "$REMOTE" << 'REMOTE_SCRIPT'
cd ~/pegnn-deform

# Generate synthetic data if not exists
if [ ! -d "data/synthetic" ]; then
    echo "Generating synthetic data..."
    python3 data/generate_synthetic.py \
        --output-dir data/synthetic \
        --grid-size 25 \
        --num-train 100 \
        --num-val 20 \
        --num-test 20 \
        --steps 50
fi

# Run training with RTX 3090 optimized settings
python3 src/train_pegdeform.py \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --epochs 50 \
    --batch-size 8 \
    --hidden-size 64 \
    --num-mp-layers 3 \
    --learning-rate 1e-4 \
    --seed 42

echo ""
echo "Training complete!"
REMOTE_SCRIPT
fi

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "To connect manually:"
echo "  ssh $REMOTE"
echo "  cd $REMOTE_DIR"
echo ""
echo "To run training:"
echo "  bash scripts/deploy_3090.sh --skip-sync --skip-env --train"
echo ""
echo "To copy results back:"
echo "  scp -r $REMOTE:$REMOTE_DIR/checkpoints ./checkpoints_3090/"
