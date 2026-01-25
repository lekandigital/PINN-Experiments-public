#!/bin/bash
# Deploy HGNN-NIF-Cloth to RTX 3090 and run benchmarks
#
# Usage:
#   ./scripts/deploy_rtx3090.sh             # Full deployment
#   ./scripts/deploy_rtx3090.sh --sync-only # Only sync files
#   ./scripts/deploy_rtx3090.sh --bench     # Only run benchmarks
#   ./scripts/deploy_rtx3090.sh --train     # Start production training

set -e

# Configuration
REMOTE_HOST="REDACTED_SERVER"
REMOTE_DIR="/home/o/hgnn-nif-cloth"
LOCAL_DIR="/Users/lekanadeyeri/Dev/PINN-Experiments/projects/09-hgnn-nif-cloth__project-space/hgnn-nif-cloth"
SSH_OPTS="-o StrictHostKeyChecking=no"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HGNN-NIF-Cloth RTX 3090 Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

sync_files() {
    echo -e "\n${YELLOW}[1/4] Syncing files to RTX 3090...${NC}"
    rsync -avz --progress \
        -e "ssh ${SSH_OPTS}" \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'outputs' \
        --exclude 'data/*.h5' \
        --exclude '.pytest_cache' \
        --exclude '*.egg-info' \
        "${LOCAL_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_DIR}/"
    echo -e "${GREEN}Files synced successfully!${NC}"
}

setup_env() {
    echo -e "\n${YELLOW}[2/4] Setting up remote environment...${NC}"
    ssh ${SSH_OPTS} ${REMOTE_HOST} << 'SETUP_EOF'
cd /home/o/hgnn-nif-cloth

# Install dependencies if needed
pip install --quiet numpy scipy h5py matplotlib pytest tensorboard tqdm pyyaml 2>/dev/null || true

# Verify PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
SETUP_EOF
    echo -e "${GREEN}Environment setup complete!${NC}"
}

run_tests() {
    echo -e "\n${YELLOW}[3/4] Running unit tests...${NC}"
    ssh ${SSH_OPTS} ${REMOTE_HOST} << 'TEST_EOF'
cd /home/o/hgnn-nif-cloth
python -m pytest tests/test_forward.py -v --tb=short 2>&1 | head -50
TEST_EOF
    echo -e "${GREEN}Tests complete!${NC}"
}

run_benchmark() {
    echo -e "\n${YELLOW}[4/4] Running RTX 3090 benchmark...${NC}"
    ssh ${SSH_OPTS} ${REMOTE_HOST} << 'BENCH_EOF'
cd /home/o/hgnn-nif-cloth
python scripts/benchmark_rtx3090.py
BENCH_EOF
    echo -e "${GREEN}Benchmark complete!${NC}"
}

generate_data() {
    echo -e "\n${YELLOW}Generating production training data...${NC}"
    ssh ${SSH_OPTS} ${REMOTE_HOST} << 'DATA_EOF'
cd /home/o/hgnn-nif-cloth
python scripts/generate_large_dataset.py \
    --num_samples 10000 \
    --config configs/rtx3090_large.yaml \
    --output data/production_train.h5
DATA_EOF
    echo -e "${GREEN}Data generation complete!${NC}"
}

start_training() {
    echo -e "\n${YELLOW}Starting production training...${NC}"
    ssh ${SSH_OPTS} ${REMOTE_HOST} << 'TRAIN_EOF'
cd /home/o/hgnn-nif-cloth
nohup python scripts/train_production.py \
    --config configs/rtx3090_large.yaml \
    --data data/production_train.h5 \
    --output_dir outputs/production_$(date +%Y%m%d_%H%M%S) \
    > training.log 2>&1 &

echo "Training started in background!"
echo "Monitor with: tail -f /home/o/hgnn-nif-cloth/training.log"
echo "PID: $!"
TRAIN_EOF
    echo -e "${GREEN}Training started!${NC}"
}

sync_results() {
    echo -e "\n${YELLOW}Syncing results back to local...${NC}"
    rsync -avz --progress \
        -e "ssh ${SSH_OPTS}" \
        "${REMOTE_HOST}:${REMOTE_DIR}/outputs/" \
        "${LOCAL_DIR}/outputs/"

    rsync -avz --progress \
        -e "ssh ${SSH_OPTS}" \
        "${REMOTE_HOST}:${REMOTE_DIR}/data/" \
        "${LOCAL_DIR}/data/"

    echo -e "${GREEN}Results synced!${NC}"
}

# Parse arguments
case "$1" in
    --sync-only)
        sync_files
        ;;
    --bench)
        run_benchmark
        ;;
    --train)
        generate_data
        start_training
        ;;
    --sync-results)
        sync_results
        ;;
    --generate-data)
        generate_data
        ;;
    *)
        # Full deployment
        sync_files
        setup_env
        run_tests
        run_benchmark
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Generate data:  ./scripts/deploy_rtx3090.sh --generate-data"
echo "  2. Start training: ./scripts/deploy_rtx3090.sh --train"
echo "  3. Monitor:        ssh ${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/training.log'"
echo "  4. Sync results:   ./scripts/deploy_rtx3090.sh --sync-results"
