#!/bin/bash
# SurfPINN Remote Sync Script
# Usage: ./sync_remote.sh [push|pull|status]

REMOTE_USER="o"
REMOTE_HOST="192.168.86.152"
REMOTE_PATH="~/surfpinn"
LOCAL_PATH="$(dirname "$0")/surfpinn"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

case "$1" in
    push)
        echo -e "${YELLOW}Pushing local code to remote...${NC}"
        rsync -avz --progress \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            --exclude 'venv/' \
            --exclude '.pytest_cache/' \
            --exclude 'data/*.h5' \
            --exclude 'checkpoints/*.pkl' \
            "$LOCAL_PATH/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"
        echo -e "${GREEN}Push complete!${NC}"
        ;;

    pull)
        echo -e "${YELLOW}Pulling results from remote...${NC}"

        # Pull checkpoints
        rsync -avz --progress \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/checkpoints/" \
            "$LOCAL_PATH/checkpoints/" 2>/dev/null || echo "No checkpoints to pull"

        # Pull results/visualizations
        rsync -avz --progress \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/results/" \
            "$LOCAL_PATH/results/" 2>/dev/null || echo "No results to pull"

        # Pull logs and reports
        rsync -avz --progress \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/*.txt" \
            "$LOCAL_PATH/" 2>/dev/null || echo "No logs to pull"

        rsync -avz --progress \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/TEST_REPORT.md" \
            "$LOCAL_PATH/" 2>/dev/null || echo "No TEST_REPORT.md to pull"

        echo -e "${GREEN}Pull complete!${NC}"
        ;;

    status)
        echo -e "${YELLOW}Checking remote status...${NC}"
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "
            echo '=== GPU Status ==='
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
            echo ''
            echo '=== Project Files ==='
            ls -la ${REMOTE_PATH}/src/ 2>/dev/null || echo 'Project not found on remote'
            echo ''
            echo '=== Checkpoints ==='
            ls -lh ${REMOTE_PATH}/checkpoints/*.pkl 2>/dev/null || echo 'No checkpoints'
            echo ''
            echo '=== Python/JAX ==='
            /usr/local/bin/python3.11 -c 'import jax; print(f\"JAX devices: {jax.devices()}\")' 2>/dev/null || echo 'JAX not installed'
        "
        ;;

    ssh)
        echo -e "${YELLOW}Connecting to remote...${NC}"
        ssh "${REMOTE_USER}@${REMOTE_HOST}"
        ;;

    setup)
        echo -e "${YELLOW}Setting up remote environment...${NC}"
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "
            cd ~ && mkdir -p surfpinn
            /usr/local/bin/python3.11 -m venv ~/surfpinn/venv
            source ~/surfpinn/venv/bin/activate
            pip install --upgrade pip
            pip install 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
            pip install dm-haiku optax chex h5py numpy scipy matplotlib pytest
            python -c 'import jax; print(f\"JAX devices: {jax.devices()}\")'
        "
        echo -e "${GREEN}Setup complete!${NC}"
        ;;

    test)
        echo -e "${YELLOW}Running integration test on remote...${NC}"
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "
            cd ~/surfpinn
            source venv/bin/activate 2>/dev/null || true
            /usr/local/bin/python3.11 tests/integration_test.py
        "
        ;;

    train)
        echo -e "${YELLOW}Starting training on remote (background)...${NC}"
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "
            cd ~/surfpinn/src
            source ../venv/bin/activate 2>/dev/null || true
            nohup /usr/local/bin/python3.11 train.py --epochs 100 > ../training_log.txt 2>&1 &
            echo 'Training started in background. Check with: tail -f ~/surfpinn/training_log.txt'
        "
        ;;

    *)
        echo "SurfPINN Remote Sync Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  push    - Push local code to remote machine"
        echo "  pull    - Pull results/checkpoints from remote"
        echo "  status  - Check GPU and project status on remote"
        echo "  ssh     - Connect to remote machine"
        echo "  setup   - Install dependencies on remote"
        echo "  test    - Run integration test on remote"
        echo "  train   - Start training on remote (background)"
        echo ""
        echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
        ;;
esac
