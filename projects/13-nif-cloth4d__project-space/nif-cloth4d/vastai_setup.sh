#!/bin/bash
# NIF-Cloth4D: Vast.ai Setup and Run Script
#
# This script sets up and runs NIF-Cloth4D on a Vast.ai GPU instance.
# Run this after SSH-ing into the instance.
#
# Usage: 
#   curl -sSL <url> | bash
#   OR
#   ./vastai_setup.sh

set -e

echo "=============================================="
echo "NIF-Cloth4D Vast.ai Setup"
echo "=============================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib h5py pyyaml trimesh ipywidgets

# Optional: mesh extraction and USD export
pip install PyMCubes || echo "Warning: PyMCubes installation failed"
pip install usd-core || echo "Warning: USD installation failed"

echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import h5py; print(f'h5py {h5py.__version__}')"
python -c "import mcubes; print('mcubes: OK')" 2>/dev/null || echo "mcubes: NOT AVAILABLE"
python -c "from pxr import Usd; print('USD: OK')" 2>/dev/null || echo "USD: NOT AVAILABLE"

echo ""
echo "=============================================="
echo "Running NIF-Cloth4D Pipeline"
echo "=============================================="
echo ""

# Create working directory
WORK_DIR=/workspace/nif_cloth4d
DATA_DIR=/tmp/cloth_test_data
OUTPUT_DIR=/workspace/output

mkdir -p $WORK_DIR $OUTPUT_DIR

# If code not present, create it inline
if [ ! -f "$WORK_DIR/nif_cloth4d.py" ]; then
    echo "Code files not found. Please copy the nif_cloth4d directory to $WORK_DIR"
    echo "Or clone from your repository."
    exit 1
fi

cd $WORK_DIR

# Step 1: Generate data
echo ""
echo "[1/4] Generating synthetic data..."
python synthetic_data.py \
    --output_dir "$DATA_DIR" \
    --num_frames 10 \
    --grid_size 64 \
    --data_type falling

# Step 2: Train
echo ""
echo "[2/4] Training model..."
python train_nif_cloth4d.py \
    --config config_test.yaml \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR/checkpoints"

# Step 3: Evaluate
echo ""
echo "[3/4] Evaluating model..."
python test_evaluation.py \
    --checkpoint "$OUTPUT_DIR/checkpoints/model_best.pt" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR/evaluation" \
    --times "0.0,0.5,1.0"

# Step 4: Export
echo ""
echo "[4/4] Exporting meshes..."
python export_mesh.py \
    --checkpoint "$OUTPUT_DIR/checkpoints/model_best.pt" \
    --output "$OUTPUT_DIR/exports" \
    --times "0.0,0.5,1.0" \
    --formats "usd,obj"

# Summary
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results:"
ls -la $OUTPUT_DIR/checkpoints/*.pt 2>/dev/null || echo "  No checkpoints"
ls -la $OUTPUT_DIR/evaluation/*.png 2>/dev/null || echo "  No plots"
ls -la $OUTPUT_DIR/exports/*.usda 2>/dev/null || echo "  No USD files"
ls -la $OUTPUT_DIR/exports/*.obj 2>/dev/null || echo "  No OBJ files"

echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "=============================================="
echo "IMPORTANT: Remember to destroy the instance!"
echo "=============================================="
echo "Run: vastai destroy instance \$INSTANCE_ID"
