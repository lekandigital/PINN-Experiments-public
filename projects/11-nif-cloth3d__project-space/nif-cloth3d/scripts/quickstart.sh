#!/bin/bash
#
# NIF-Cloth3D-Interactive: Quick Start Script
# Runs all necessary steps to get started with the project
#

set -e

echo "================================================="
echo "NIF-Cloth3D-Interactive Quick Start"
echo "================================================="

# Check Python version
echo ""
echo "[1/6] Checking Python version..."
python3 --version

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/6] Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "[2/6] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
echo ""
echo "[5/6] Running tests..."
python -m pytest tests/ -v --tb=short || echo "Warning: Some tests may fail without CUDA"

# Generate test data
echo ""
echo "[6/6] Generating test data..."
python tests/generate_test_data.py

echo ""
echo "================================================="
echo "Quick Start Complete!"
echo "================================================="
echo ""
echo "Next steps:"
echo "  1. Train model:    python src/train.py --config configs/rtx4090.yaml"
echo "  2. Run inference:  python src/inference.py --model checkpoints/model_traced.pt"
echo "  3. Open notebook:  jupyter notebook notebooks/evaluation.ipynb"
echo ""
