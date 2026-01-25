#!/bin/bash
# =============================================================================
# Vast.ai Instance Provisioning Script for NIF-Cloth4D-Temporal
# =============================================================================
# Searches for NVIDIA L40S instances and creates a development instance
# with PyTorch 2.1 + CUDA 11.8
#
# Prerequisites:
#   - vast.ai CLI installed: pip install vastai
#   - API key configured: vastai set api-key YOUR_KEY
#
# Usage:
#   ./scripts/provision_vastai.sh
#   ./scripts/provision_vastai.sh --auto  # Auto-select cheapest instance
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  NIF-Cloth4D-Temporal - Vast.ai Provisioning   ${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo -e "${RED}Error: vastai CLI not found.${NC}"
    echo "Install with: pip install vastai"
    echo "Then set API key: vastai set api-key YOUR_KEY"
    exit 1
fi

# Configuration
GPU_NAME="L40S"
MIN_DISK_SPACE=100  # GB
MIN_CUDA_VERSION=11.7
DOCKER_IMAGE="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"

echo ""
echo -e "${YELLOW}Searching for NVIDIA ${GPU_NAME} instances...${NC}"
echo "Requirements:"
echo "  - GPU: ${GPU_NAME}"
echo "  - Disk: ${MIN_DISK_SPACE}GB+"
echo "  - CUDA: ${MIN_CUDA_VERSION}+"
echo "  - Verified: Yes"
echo ""

# Search for instances
# Note: The correct GPU name is "L40S" not "LS40"
SEARCH_QUERY="gpu_name=${GPU_NAME} rentable=true verified=true cuda_vers>=${MIN_CUDA_VERSION} disk_space>=${MIN_DISK_SPACE}"

echo -e "${BLUE}Running search...${NC}"
echo "Query: ${SEARCH_QUERY}"
echo ""

vastai search offers "${SEARCH_QUERY}" --order dph_total --limit 10

echo ""
echo -e "${YELLOW}------------------------------------------------${NC}"

# Check for auto mode
if [ "$1" == "--auto" ]; then
    echo -e "${GREEN}Auto mode: Selecting cheapest instance...${NC}"
    
    # Get the cheapest instance ID
    INSTANCE_ID=$(vastai search offers "${SEARCH_QUERY}" --order dph_total --limit 1 --raw | python3 -c "import sys, json; data = json.load(sys.stdin); print(data[0]['id'] if data else '')")
    
    if [ -z "$INSTANCE_ID" ]; then
        echo -e "${RED}No matching instances found!${NC}"
        exit 1
    fi
    
    echo "Selected instance ID: ${INSTANCE_ID}"
else
    echo -e "${GREEN}Enter the ID of the instance you want to rent:${NC}"
    read -r INSTANCE_ID
fi

if [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}No instance ID provided. Exiting.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Creating instance with ID: ${INSTANCE_ID}${NC}"
echo "Docker image: ${DOCKER_IMAGE}"
echo ""

# Create instance
# Options:
#   --disk: Disk space in GB
#   --ssh: Enable SSH access
#   --jupyter: Enable Jupyter access
#   --direct: Direct connection (faster but requires firewall rules)
vastai create instance "${INSTANCE_ID}" \
    --image "${DOCKER_IMAGE}" \
    --disk "${MIN_DISK_SPACE}" \
    --ssh \
    --jupyter \
    --onstart-cmd "echo 'Instance ready for NIF-Cloth4D-Temporal'" \
    --label "nif-cloth4d-temporal"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Instance Created Successfully!                ${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Wait for instance to start (~2-5 minutes)"
echo "  2. Run: vastai show instances"
echo "  3. Get SSH command: vastai ssh-url <instance_id>"
echo "  4. Connect and clone your repo"
echo ""
echo "Or use: ./scripts/connect_vastai.sh"
echo ""
