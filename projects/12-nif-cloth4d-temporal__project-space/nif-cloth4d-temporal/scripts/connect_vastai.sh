#!/bin/bash
# =============================================================================
# Vast.ai Connection Script for NIF-Cloth4D-Temporal
# =============================================================================
# Connects to a running Vast.ai instance and sets up the environment
#
# Usage:
#   ./scripts/connect_vastai.sh           # Interactive mode
#   ./scripts/connect_vastai.sh <id>      # Connect to specific instance
#   ./scripts/connect_vastai.sh --list    # List running instances
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  NIF-Cloth4D-Temporal - Vast.ai Connection     ${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo -e "${RED}Error: vastai CLI not found.${NC}"
    echo "Install with: pip install vastai"
    exit 1
fi

# Handle arguments
if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    echo ""
    echo -e "${YELLOW}Running instances:${NC}"
    vastai show instances
    exit 0
fi

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo ""
    echo "Usage:"
    echo "  ./connect_vastai.sh           # Interactive mode"
    echo "  ./connect_vastai.sh <id>      # Connect to specific instance"
    echo "  ./connect_vastai.sh --list    # List running instances"
    echo "  ./connect_vastai.sh --setup   # Show setup commands after connecting"
    exit 0
fi

# Show running instances
echo ""
echo -e "${YELLOW}Your running instances:${NC}"
vastai show instances

echo ""

# Get instance ID
if [ -n "$1" ] && [ "$1" != "--setup" ]; then
    INSTANCE_ID="$1"
else
    echo -e "${GREEN}Enter instance ID to connect:${NC}"
    read -r INSTANCE_ID
fi

if [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}No instance ID provided. Exiting.${NC}"
    exit 1
fi

# Get SSH URL
echo ""
echo -e "${YELLOW}Getting SSH connection info for instance ${INSTANCE_ID}...${NC}"
SSH_URL=$(vastai ssh-url "${INSTANCE_ID}")

echo ""
echo -e "${GREEN}SSH Command:${NC}"
echo "${SSH_URL}"
echo ""

# Show setup commands
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  After connecting, run these setup commands:   ${NC}"
echo -e "${BLUE}================================================${NC}"
cat << 'EOF'

# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/nif-cloth4d-temporal.git
cd nif-cloth4d-temporal

# 2. Create conda environment
conda env create -f environment.yml
conda activate nif_cloth4d

# 3. Verify GPU is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 4. Run CI sanity check
python tests/ci_sanity_check.py

# 5. Run inference speed test
python scripts/test_inference.py

# 6. Generate synthetic dataset (if needed)
python -c "from src.data import generate_synthetic_dataset; generate_synthetic_dataset()"

# 7. Start training
python scripts/train.py --use_synthetic --epochs 10 --use_gru

# 8. (Optional) Start training with W&B logging
# First: wandb login
python scripts/train.py --use_synthetic --epochs 100 --use_gru --use_wandb

EOF

echo ""
echo -e "${GREEN}Connecting...${NC}"
echo ""

# Execute SSH command
eval "${SSH_URL}"
