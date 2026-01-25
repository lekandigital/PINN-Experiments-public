#!/bin/bash
# NIF-Cloth4D Remote Sync Script
#
# Syncs files between local Mac and remote RTX 3090 machine
#
# Usage:
#   ./sync_remote.sh push    # Push local files to remote
#   ./sync_remote.sh pull    # Pull results from remote to local
#   ./sync_remote.sh status  # Check what's on remote

set -e

# Configuration
REMOTE_HOST="REDACTED_SERVER"
REMOTE_DIR="~/projects/nif-cloth4d"
LOCAL_DIR="$(dirname "$0")/nif-cloth4d"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NIF-Cloth4D Remote Sync${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Remote: $REMOTE_HOST:$REMOTE_DIR"
echo "Local:  $LOCAL_DIR"
echo ""

case "$1" in
    push)
        echo -e "${GREEN}Pushing local files to remote...${NC}"

        # Create remote directory
        ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

        # Push source files (exclude generated files)
        rsync -avz --progress \
            --exclude='checkpoints/' \
            --exclude='evaluation/' \
            --exclude='exports/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='.DS_Store' \
            "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

        echo -e "${GREEN}Push complete!${NC}"
        ;;

    pull)
        echo -e "${GREEN}Pulling results from remote...${NC}"

        # Create local directories
        mkdir -p "$LOCAL_DIR/checkpoints"
        mkdir -p "$LOCAL_DIR/evaluation"
        mkdir -p "$LOCAL_DIR/exports"

        # Pull checkpoints
        echo "Pulling checkpoints..."
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_DIR/checkpoints/" \
            "$LOCAL_DIR/checkpoints/" 2>/dev/null || echo "  (no checkpoints found)"

        # Pull evaluation results
        echo "Pulling evaluation..."
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_DIR/evaluation/" \
            "$LOCAL_DIR/evaluation/" 2>/dev/null || echo "  (no evaluation found)"

        # Pull exports
        echo "Pulling exports..."
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_DIR/exports/" \
            "$LOCAL_DIR/exports/" 2>/dev/null || echo "  (no exports found)"

        # Pull any generated output
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_DIR/output/" \
            "$LOCAL_DIR/output/" 2>/dev/null || echo "  (no output dir found)"

        echo -e "${GREEN}Pull complete!${NC}"

        # Show what was pulled
        echo ""
        echo "Files pulled:"
        ls -la "$LOCAL_DIR/checkpoints/"*.pt 2>/dev/null || echo "  - No checkpoint files"
        ls -la "$LOCAL_DIR/evaluation/"*.png 2>/dev/null || echo "  - No evaluation plots"
        ls -la "$LOCAL_DIR/exports/"*.obj 2>/dev/null || echo "  - No mesh exports"
        ;;

    status)
        echo -e "${YELLOW}Checking remote status...${NC}"

        # Check if remote directory exists and list contents
        ssh $REMOTE_HOST "
            echo '=== Remote GPU ===' && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
            echo ''
            echo '=== Project Directory ===' && ls -la $REMOTE_DIR/ 2>/dev/null || echo 'Directory does not exist'
            echo ''
            echo '=== Checkpoints ===' && ls -la $REMOTE_DIR/checkpoints/*.pt 2>/dev/null || echo 'No checkpoints'
            echo ''
            echo '=== Evaluation ===' && ls -la $REMOTE_DIR/evaluation/ 2>/dev/null || echo 'No evaluation'
            echo ''
            echo '=== Exports ===' && ls -la $REMOTE_DIR/exports/ 2>/dev/null || echo 'No exports'
        "
        ;;

    ssh)
        echo -e "${YELLOW}Connecting to remote...${NC}"
        ssh $REMOTE_HOST
        ;;

    run)
        echo -e "${GREEN}Running pipeline on remote...${NC}"
        ssh $REMOTE_HOST "cd $REMOTE_DIR && ./run_pipeline.sh /tmp/cloth_test_data ./output"
        ;;

    *)
        echo "Usage: $0 {push|pull|status|ssh|run}"
        echo ""
        echo "Commands:"
        echo "  push   - Push local source files to remote"
        echo "  pull   - Pull results (checkpoints, evaluation, exports) from remote"
        echo "  status - Check what's on remote machine"
        echo "  ssh    - SSH into remote machine"
        echo "  run    - Run the training pipeline on remote"
        exit 1
        ;;
esac
