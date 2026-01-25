#!/bin/bash
# NIF-Cloth4D: Quick Run Script
# 
# This script runs the complete NIF-Cloth4D pipeline:
# 1. Generate synthetic data
# 2. Train the model
# 3. Evaluate and export
#
# Usage: ./run_pipeline.sh [data_dir] [output_dir]

set -e  # Exit on error

# Configuration
DATA_DIR="${1:-/tmp/cloth_test_data}"
OUTPUT_DIR="${2:-./output}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/evaluation"
EXPORT_DIR="$OUTPUT_DIR/exports"

echo "=============================================="
echo "NIF-Cloth4D Pipeline"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$EVAL_DIR" "$EXPORT_DIR"

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import h5py; print(f'h5py {h5py.__version__}')"

# Check optional dependencies
python3 -c "import mcubes; print('✓ mcubes')" 2>/dev/null || echo "✗ mcubes (mesh extraction disabled)"
python3 -c "from pxr import Usd; print('✓ USD')" 2>/dev/null || echo "✗ USD (USD export disabled)"

echo ""
echo "=============================================="
echo "Step 1: Generate Synthetic Data"
echo "=============================================="
python3 synthetic_data.py \
    --output_dir "$DATA_DIR" \
    --num_frames 10 \
    --grid_size 64 \
    --data_type falling

echo ""
echo "=============================================="
echo "Step 2: Train Model"
echo "=============================================="
python3 train_nif_cloth4d.py \
    --config config_test.yaml \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_DIR"

echo ""
echo "=============================================="
echo "Step 3: Evaluate Model"
echo "=============================================="
python3 test_evaluation.py \
    --checkpoint "$CHECKPOINT_DIR/model_best.pt" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVAL_DIR" \
    --resolution 64 \
    --times "0.0,0.5,1.0"

echo ""
echo "=============================================="
echo "Step 4: Export Meshes"
echo "=============================================="
python3 export_mesh.py \
    --checkpoint "$CHECKPOINT_DIR/model_best.pt" \
    --output "$EXPORT_DIR" \
    --times "0.0,0.5,1.0" \
    --resolution 64 \
    --formats "usd,obj"

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - Checkpoints: $CHECKPOINT_DIR/"
echo "  - Evaluation: $EVAL_DIR/"
echo "  - Exports: $EXPORT_DIR/"
echo ""
echo "Files:"
ls -la "$CHECKPOINT_DIR"/*.pt 2>/dev/null || echo "  (no checkpoints)"
ls -la "$EVAL_DIR"/*.png 2>/dev/null || echo "  (no plots)"
ls -la "$EXPORT_DIR"/*.usda 2>/dev/null || echo "  (no USD files)"
ls -la "$EXPORT_DIR"/*.obj 2>/dev/null || echo "  (no OBJ files)"
