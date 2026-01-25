#!/bin/bash
# PINN-Lite-Foil: Sync files between local and remote RTX 3090 machine
# Remote: REDACTED_SERVER
# Local: /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/

REMOTE_HOST="REDACTED_SERVER"
REMOTE_DIR="~/pinn-lite-foil"
LOCAL_DIR="/Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [push|pull|status]"
    echo ""
    echo "Commands:"
    echo "  push    - Push local project to remote machine"
    echo "  pull    - Pull trained models and results from remote"
    echo "  status  - Check remote GPU and project status"
    echo ""
    exit 1
}

push_to_remote() {
    echo -e "${BLUE}Pushing project files to remote...${NC}"
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'venv' \
        "$LOCAL_DIR/" \
        "$REMOTE_HOST:$REMOTE_DIR/"
    echo -e "${GREEN}Push complete!${NC}"
}

pull_from_remote() {
    echo -e "${BLUE}Pulling models and results from remote...${NC}"

    # Create local directories if needed
    mkdir -p "$LOCAL_DIR/models" "$LOCAL_DIR/results" "$LOCAL_DIR/data/processed"

    # Pull models
    rsync -avz --progress \
        "$REMOTE_HOST:$REMOTE_DIR/models/" \
        "$LOCAL_DIR/models/"

    # Pull results
    rsync -avz --progress \
        "$REMOTE_HOST:$REMOTE_DIR/results/" \
        "$LOCAL_DIR/results/"

    # Pull processed data (optional - can be large)
    read -p "Pull processed data files? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_DIR/data/processed/" \
            "$LOCAL_DIR/data/processed/"
    fi

    echo -e "${GREEN}Pull complete!${NC}"
}

check_status() {
    echo -e "${BLUE}Checking remote status...${NC}"
    echo ""
    echo "=== GPU Status ==="
    ssh "$REMOTE_HOST" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv"
    echo ""
    echo "=== Project Files ==="
    ssh "$REMOTE_HOST" "ls -la $REMOTE_DIR/ 2>/dev/null || echo 'Project directory not found'"
    echo ""
    echo "=== Trained Models ==="
    ssh "$REMOTE_HOST" "ls -lh $REMOTE_DIR/models/*/*.h5 2>/dev/null || echo 'No models found'"
    echo ""
    echo "=== ONNX Models ==="
    ssh "$REMOTE_HOST" "ls -lh $REMOTE_DIR/models/onnx/*.onnx 2>/dev/null || echo 'No ONNX models found'"
    echo ""
    echo "=== Results ==="
    ssh "$REMOTE_HOST" "ls -l $REMOTE_DIR/results/*.json 2>/dev/null || echo 'No results found'"
}

case "$1" in
    push)
        push_to_remote
        ;;
    pull)
        pull_from_remote
        ;;
    status)
        check_status
        ;;
    *)
        usage
        ;;
esac
