#!/bin/bash
# PEGNN-Deform: vast.ai Deployment Script
#
# Automates the process of:
# 1. Searching for L40S instances
# 2. Creating an instance
# 3. Transferring files
# 4. Running setup and tests
#
# Requirements:
# - vast.ai CLI installed: pip install vastai
# - API key set: export VASTAI_API_KEY="your_key"
#
# Usage: bash scripts/deploy_vastai.sh

set -e

# Configuration
DISK_SIZE=50
IMAGE="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
GPU_NAME="L40S"

echo "=============================================="
echo "PEGNN-Deform: vast.ai Deployment"
echo "=============================================="

# Check for API key
if [ -z "$VASTAI_API_KEY" ]; then
    echo "Error: VASTAI_API_KEY not set"
    echo "Set with: export VASTAI_API_KEY='your_key'"
    exit 1
fi

# Set API key for vast.ai CLI
vastai set api-key "$VASTAI_API_KEY"

echo ""
echo "Step 1: Searching for L40S instances..."
echo "----------------------------------------"

vastai search offers "gpu_name=${GPU_NAME} rentable=true verified=true num_gpus=1" \
    --order dph_total \
    --limit 10

echo ""
echo "Please note the instance ID you want to use."
read -p "Enter instance ID (or 'auto' for cheapest): " INSTANCE_ID

if [ "$INSTANCE_ID" = "auto" ]; then
    # Get the cheapest instance automatically
    INSTANCE_ID=$(vastai search offers "gpu_name=${GPU_NAME} rentable=true verified=true num_gpus=1" \
        --order dph_total \
        --limit 1 \
        --raw | python -c "import json,sys; d=json.load(sys.stdin); print(d[0]['id'])" 2>/dev/null)
    echo "Selected instance ID: $INSTANCE_ID"
fi

echo ""
echo "Step 2: Creating instance..."
echo "----------------------------"

# Create the instance
RESULT=$(vastai create instance "$INSTANCE_ID" \
    --image "$IMAGE" \
    --disk "$DISK_SIZE" \
    --ssh \
    --raw 2>&1)

echo "$RESULT"

# Extract the new instance ID
NEW_INSTANCE_ID=$(echo "$RESULT" | python -c "import json,sys; d=json.load(sys.stdin); print(d['new_contract'])" 2>/dev/null || echo "")

if [ -z "$NEW_INSTANCE_ID" ]; then
    echo "Failed to create instance. Please check the error above."
    exit 1
fi

echo "Created instance: $NEW_INSTANCE_ID"

echo ""
echo "Step 3: Waiting for instance to be ready..."
echo "--------------------------------------------"

# Wait for instance to be ready
MAX_WAIT=300  # 5 minutes
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(vastai show instance "$NEW_INSTANCE_ID" --raw | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status', 'unknown'))" 2>/dev/null || echo "unknown")
    
    if [ "$STATUS" = "running" ]; then
        echo "Instance is running!"
        break
    fi
    
    echo "Status: $STATUS (waiting...)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ "$STATUS" != "running" ]; then
    echo "Instance failed to start within $MAX_WAIT seconds"
    exit 1
fi

echo ""
echo "Step 4: Getting connection details..."
echo "--------------------------------------"

# Get SSH connection details
SSH_INFO=$(vastai show instance "$NEW_INSTANCE_ID" --raw)
SSH_HOST=$(echo "$SSH_INFO" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_host', ''))" 2>/dev/null)
SSH_PORT=$(echo "$SSH_INFO" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_port', ''))" 2>/dev/null)

echo "SSH Host: $SSH_HOST"
echo "SSH Port: $SSH_PORT"

echo ""
echo "Step 5: Transferring project files..."
echo "--------------------------------------"

# Get the project root (one level up from scripts/)
PROJECT_ROOT="$(dirname "$0")/.."
cd "$PROJECT_ROOT"

echo "Project root: $(pwd)"

# Create tarball excluding unnecessary files
tar -czf /tmp/pegnn-deform.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='data/synthetic' \
    --exclude='*.log' \
    .

# Transfer files
scp -P "$SSH_PORT" /tmp/pegnn-deform.tar.gz "root@$SSH_HOST:/workspace/"

echo "Files transferred!"

echo ""
echo "Step 6: Setting up and running tests..."
echo "----------------------------------------"

# SSH and run setup
ssh -p "$SSH_PORT" "root@$SSH_HOST" << 'REMOTE_SCRIPT'
cd /workspace
mkdir -p pegnn-deform
cd pegnn-deform
tar -xzf ../pegnn-deform.tar.gz

# Make scripts executable
chmod +x scripts/*.sh

# Run setup
bash scripts/setup_vastai.sh

# Run tests
bash scripts/run_tests.sh
REMOTE_SCRIPT

echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Instance ID: $NEW_INSTANCE_ID"
echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo ""
echo "To destroy instance when done:"
echo "  vastai destroy instance $NEW_INSTANCE_ID"
