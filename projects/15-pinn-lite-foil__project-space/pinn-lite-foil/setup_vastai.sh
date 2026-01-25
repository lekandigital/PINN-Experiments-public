#!/bin/bash
# PINN-Lite-Foil Vast.ai Setup Script
# Target: NVIDIA L40S GPU instance

set -e

echo "=========================================="
echo "PINN-Lite-Foil Environment Setup"
echo "=========================================="

# Update system
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libhdf5-dev \
    pkg-config

# Upgrade pip
pip install --upgrade pip

# Install Python requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install ONNX Runtime with GPU support
echo "Installing ONNX Runtime..."
ONNX_VERSION="1.16.3"
wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz
tar -xzf onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-gpu-${ONNX_VERSION}
echo "export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-gpu-${ONNX_VERSION}" >> ~/.bashrc

# Build C++ inference wrapper
echo "Building C++ inference wrapper..."
cd src/deployment/cpp_inference
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)
cd ../../../..

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Generate data:   python src/data_generation/naca_generator.py --n_samples 50"
echo "  2. Train baseline:  python src/training/baseline_pinn.py --epochs 10000"
echo "  3. Run distillation: python src/training/distillation.py --epochs 5000"
echo "  4. Convert to ONNX: python src/deployment/onnx_converter.py"
echo "  5. Benchmark:       python src/benchmarking/latency_benchmark.py"
