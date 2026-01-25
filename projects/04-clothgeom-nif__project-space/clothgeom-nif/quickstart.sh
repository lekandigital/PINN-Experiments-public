#!/bin/bash
# ClothGeom-NIF Quickstart Script
# Complete automation: Provision GPU → Train → Download Results
#
# Usage: bash quickstart.sh
#
# Prerequisites:
#   pip install vast
#   vastai set api-key YOUR_API_KEY

set -e  # Exit on error

echo "🚀 ClothGeom-NIF Quickstart"
echo "============================"
echo ""
echo "This script will:"
echo "  1. Provision a GPU instance on Vast.ai"
echo "  2. Upload the project files"
echo "  3. Generate training data"
echo "  4. Train the model (~30 minutes)"
echo "  5. Generate demo meshes"
echo "  6. Download results to local machine"
echo ""
echo "Estimated time: ~45 minutes"
echo "Estimated cost: ~\$1-2 (RTX 3090)"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v vastai &> /dev/null; then
    echo "❌ vastai CLI not found!"
    echo ""
    echo "Install with:"
    echo "   pip install vast"
    echo "   vastai set api-key YOUR_API_KEY"
    echo ""
    echo "Get your API key from: https://vast.ai/console/account/"
    exit 1
fi
echo "  ✓ vastai CLI found"

if ! command -v ssh &> /dev/null; then
    echo "❌ ssh not found"
    exit 1
fi
echo "  ✓ ssh available"

if ! command -v scp &> /dev/null; then
    echo "❌ scp not found"
    exit 1
fi
echo "  ✓ scp available"

# Step 1: Find GPU instance
echo ""
echo "📡 Step 1: Searching for GPU instance..."

# Search for suitable GPUs
SEARCH_RESULT=""
for GPU in "RTX_3090" "RTX_4090" "A6000" "RTX_A5000"; do
    echo "  Searching $GPU..."
    RESULT=$(vastai search offers "gpu_name=$GPU rentable=true verified=true gpu_ram>=24 disk_space>=50" --order dph_total --limit 1 2>/dev/null | grep -E "^[0-9]+" | head -n1 || true)
    if [ -n "$RESULT" ]; then
        SEARCH_RESULT="$RESULT"
        echo "  ✓ Found $GPU instance"
        break
    fi
done

if [ -z "$SEARCH_RESULT" ]; then
    echo "  Trying broader search (any GPU >= 20GB)..."
    SEARCH_RESULT=$(vastai search offers "rentable=true verified=true gpu_ram>=20 disk_space>=50" --order dph_total --limit 1 2>/dev/null | grep -E "^[0-9]+" | head -n1 || true)
fi

if [ -z "$SEARCH_RESULT" ]; then
    echo "❌ No suitable GPU instances found"
    echo ""
    echo "Try later or search manually:"
    echo "   bash search_gpus.sh"
    exit 1
fi

OFFER_ID=$(echo "$SEARCH_RESULT" | awk '{print $1}')
GPU_NAME=$(echo "$SEARCH_RESULT" | awk '{print $3}')
PRICE=$(echo "$SEARCH_RESULT" | awk '{print $6}')

echo ""
echo "  Selected: $GPU_NAME at \$$PRICE/hr"
echo "  Offer ID: $OFFER_ID"
echo ""

read -p "Create this instance? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Step 2: Create instance
echo ""
echo "🔧 Step 2: Creating instance..."
vastai create instance $OFFER_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 50 \
    --ssh

echo "  Waiting for instance to start..."
sleep 30

# Get instance details
INSTANCE_ID=$(vastai show instances | grep -E "^[0-9]+" | head -n1 | awk '{print $1}')
if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to get instance ID"
    exit 1
fi

echo "  Instance ID: $INSTANCE_ID"

# Wait for SSH to be ready
echo "  Waiting for SSH (this may take 1-2 minutes)..."
for i in {1..24}; do
    INSTANCE_INFO=$(vastai show instances | grep "^$INSTANCE_ID" | head -n1)
    STATUS=$(echo "$INSTANCE_INFO" | awk '{print $3}')
    
    if [ "$STATUS" = "running" ]; then
        # Extract SSH details - Vast.ai format varies, try to get host:port
        SSH_INFO=$(vastai ssh-url $INSTANCE_ID 2>/dev/null || echo "")
        
        if [ -n "$SSH_INFO" ]; then
            # Parse ssh://root@host:port format
            SSH_HOST=$(echo "$SSH_INFO" | sed 's/ssh:\/\/root@//' | cut -d':' -f1)
            SSH_PORT=$(echo "$SSH_INFO" | sed 's/ssh:\/\/root@//' | cut -d':' -f2)
            break
        fi
    fi
    
    echo "    Attempt $i/24: Status=$STATUS"
    sleep 10
done

if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "❌ Could not get SSH connection details"
    echo ""
    echo "Check manually:"
    echo "   vastai show instances"
    echo "   vastai ssh-url $INSTANCE_ID"
    echo ""
    echo "Then connect and run:"
    echo "   cd clothgeom-nif && bash setup_environment.sh"
    exit 1
fi

echo "  ✓ Instance ready: $SSH_HOST:$SSH_PORT"

# Step 3: Upload files
echo ""
echo "📦 Step 3: Uploading project files..."

# Create remote directory
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "mkdir -p ~/clothgeom-nif"

# Upload files
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
scp -o StrictHostKeyChecking=no -P $SSH_PORT -r \
    "$SCRIPT_DIR"/*.py \
    "$SCRIPT_DIR"/*.sh \
    "$SCRIPT_DIR"/*.txt \
    "$SCRIPT_DIR"/configs \
    "$SCRIPT_DIR"/models \
    "$SCRIPT_DIR"/data \
    "$SCRIPT_DIR"/inference \
    root@$SSH_HOST:~/clothgeom-nif/ 2>/dev/null || true

echo "  ✓ Files uploaded"

# Step 4: Setup environment
echo ""
echo "🔧 Step 4: Setting up environment..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "cd ~/clothgeom-nif && bash setup_environment.sh"
echo "  ✓ Environment ready"

# Step 5: Generate dataset
echo ""
echo "📊 Step 5: Generating training dataset..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "cd ~/clothgeom-nif && python generate_dataset.py --num_samples 100"
echo "  ✓ Dataset generated"

# Step 6: Train model
echo ""
echo "🎓 Step 6: Training model (this takes ~30 minutes)..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "cd ~/clothgeom-nif && python train.py --epochs 100" &
TRAIN_PID=$!

# Show progress
echo "  Training in progress..."
echo "  (Check logs with: ssh -p $SSH_PORT root@$SSH_HOST 'tail -f ~/clothgeom-nif/outputs/logs/*')"
wait $TRAIN_PID
echo "  ✓ Training complete"

# Step 7: Generate demo meshes
echo ""
echo "🎨 Step 7: Generating demo meshes..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "cd ~/clothgeom-nif && python demo.py --checkpoint outputs/checkpoints/best.pt"
echo "  ✓ Meshes generated"

# Step 8: Download results
echo ""
echo "📥 Step 8: Downloading results..."
mkdir -p local_outputs

scp -o StrictHostKeyChecking=no -P $SSH_PORT -r \
    root@$SSH_HOST:~/clothgeom-nif/outputs/* \
    local_outputs/

echo "  ✓ Results downloaded to local_outputs/"

# Summary
echo ""
echo "============================================"
echo "✅ Training complete!"
echo "============================================"
echo ""
echo "📁 Results saved to: local_outputs/"
echo "   - checkpoints/best.pt  (trained model)"
echo "   - demo/*.obj          (generated meshes)"
echo ""
echo "🖼️  View meshes:"
echo "   - Open local_outputs/demo/*.obj in Blender/MeshLab"
echo "   - Or: https://3dviewer.net/"
echo ""
echo "💰 Remember to destroy the instance to stop billing:"
echo "   vastai destroy instance $INSTANCE_ID"
echo ""
echo "   Or destroy all instances:"
echo "   vastai destroy instance \$(vastai show instances | grep -E '^[0-9]+' | awk '{print \$1}')"
echo ""

# Ask to destroy
read -p "Destroy instance now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    vastai destroy instance $INSTANCE_ID
    echo "✓ Instance destroyed"
fi

echo ""
echo "🎉 Done! Check local_outputs/ for your trained model and meshes."
