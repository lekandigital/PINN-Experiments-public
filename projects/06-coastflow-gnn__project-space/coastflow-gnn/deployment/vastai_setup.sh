#!/bin/bash
# ==============================================================================
# Vast.ai Instance Setup Script for CoastFlow-GNN
# ==============================================================================
# This script helps you find and provision a GPU instance on Vast.ai
# for training CoastFlow-GNN.
#
# Prerequisites:
#   - Vast.ai CLI installed: pip install vastai
#   - Vast.ai API key configured: vastai set api-key <your-key>
#
# Usage:
#   chmod +x vastai_setup.sh
#   ./vastai_setup.sh
# ==============================================================================

set -e

echo "=============================================="
echo "CoastFlow-GNN - Vast.ai Instance Setup"
echo "=============================================="

# Check if vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo "❌ Vast.ai CLI not found. Installing..."
    pip install vastai
    echo "Please configure your API key:"
    echo "  vastai set api-key <your-api-key>"
    exit 1
fi

echo ""
echo "🔍 Searching for L40S instances..."
echo "   (Note: GPU name is 'L40S' not 'LS40')"
echo ""

# Search for L40S instances
vastai search offers \
    "gpu_name=L40S rentable=true verified=true ram>=64" \
    --order dph_total \
    --limit 10

echo ""
echo "=============================================="
echo "Alternative GPU Options (if L40S unavailable):"
echo "=============================================="
echo ""

echo "📊 L40 (without S):"
vastai search offers \
    "gpu_name=L40 rentable=true verified=true ram>=32" \
    --order dph_total \
    --limit 3 2>/dev/null || echo "   No L40 instances found"

echo ""
echo "📊 RTX 4090 (24GB VRAM):"
vastai search offers \
    "gpu_name=RTX_4090 rentable=true verified=true" \
    --order dph_total \
    --limit 3 2>/dev/null || echo "   No RTX 4090 instances found"

echo ""
echo "📊 RTX A6000 (48GB VRAM):"
vastai search offers \
    "gpu_name=RTX_A6000 rentable=true verified=true" \
    --order dph_total \
    --limit 3 2>/dev/null || echo "   No A6000 instances found"

echo ""
echo "=============================================="
echo "Instance Selection"
echo "=============================================="
echo ""
read -p "Enter the ID of the instance to rent (or 'q' to quit): " INSTANCE_ID

if [ "$INSTANCE_ID" = "q" ] || [ -z "$INSTANCE_ID" ]; then
    echo "Exiting..."
    exit 0
fi

echo ""
echo "📦 Creating instance $INSTANCE_ID..."
echo "   Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
echo "   Disk: 50GB"
echo "   Jupyter: Enabled"
echo ""

# Create the instance
vastai create instance $INSTANCE_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 50 \
    --jupyter \
    --direct \
    --env "PYTHONPATH=/workspace/coastflow-gnn" \
    --label "coastflow-gnn-experiment"

echo ""
echo "⏳ Waiting for instance to start..."
sleep 15

echo ""
echo "=============================================="
echo "Your Instances:"
echo "=============================================="
vastai show instances

echo ""
echo "=============================================="
echo "Next Steps:"
echo "=============================================="
echo ""
echo "1. Connect to your instance via SSH:"
echo "   vastai ssh-url <instance_id>"
echo ""
echo "2. Or open Jupyter notebook (URL shown in 'show instances')"
echo ""
echo "3. Once connected, run the remote setup script:"
echo "   git clone <your-repo> coastflow-gnn"
echo "   cd coastflow-gnn/deployment"
echo "   chmod +x remote_setup.sh"
echo "   ./remote_setup.sh"
echo ""
echo "4. Start training:"
echo "   cd /workspace/coastflow-gnn"
echo "   python src/training/train_single.py --epochs 50"
echo ""
echo "=============================================="
