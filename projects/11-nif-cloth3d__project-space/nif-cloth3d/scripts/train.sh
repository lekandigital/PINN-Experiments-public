#!/bin/bash
#
# NIF-Cloth3D-Interactive: Training Script
# Wrapper for launching training with different configurations
#

set -e

# Default values
CONFIG="${CONFIG:-configs/rtx4090.yaml}"
OUTPUT="${OUTPUT:-checkpoints}"
RESUME="${RESUME:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config CONFIG] [--output OUTPUT] [--resume CHECKPOINT]"
            echo ""
            echo "Options:"
            echo "  --config    Path to YAML config file (default: configs/rtx4090.yaml)"
            echo "  --output    Output directory for checkpoints (default: checkpoints)"
            echo "  --resume    Path to checkpoint to resume from"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================="
echo "NIF-Cloth3D-Interactive Training"
echo "================================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT"
if [ -n "$RESUME" ]; then
    echo "Resume: $RESUME"
fi
echo "================================================="

# Create output directory
mkdir -p "$OUTPUT"

# Check for CUDA
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Start tensorboard in background
echo ""
echo "Starting TensorBoard on port 6006..."
tensorboard --logdir="$OUTPUT/logs" --port=6006 &
TENSORBOARD_PID=$!

# Trap to kill tensorboard on exit
trap "kill $TENSORBOARD_PID 2>/dev/null || true" EXIT

# Run training
echo ""
echo "Starting training..."
python src/train.py \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    $RESUME

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT"
