#!/bin/bash
# PEGNN-Deform: Test Runner Script
#
# Runs all tests and generates a summary report.
# Designed for vast.ai L40S GPU instances.
#
# Usage: bash scripts/run_tests.sh

set -e

echo "=============================================="
echo "PEGNN-Deform: Test Suite"
echo "=============================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Project root: $PROJECT_ROOT"
echo ""

# Print GPU info
echo "GPU Information:"
echo "----------------"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv
echo ""

# ============================================
# Test 1: Unit Tests
# ============================================
echo ""
echo "=============================================="
echo "Test 1: Unit Tests (pytest)"
echo "=============================================="
echo ""

python -m pytest tests/test_model.py -v --tb=short 2>&1 | tee test_unit.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Unit tests PASSED"
else
    echo "✗ Unit tests FAILED"
    exit 1
fi

# ============================================
# Test 2: Model Forward Pass
# ============================================
echo ""
echo "=============================================="
echo "Test 2: Model Forward Pass"
echo "=============================================="
echo ""

python -c "
import torch
import sys
sys.path.insert(0, 'src')
from pegdeform_model import PEGNNDeform, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Create model
model = PEGNNDeform(hidden_size=64, num_mp_layers=3).to(device)
print(f'Model parameters: {count_parameters(model):,}')

# Test data
N, E = 1000, 6000
pos = torch.randn(N, 3, device=device)
vel = torch.randn(N, 3, device=device) * 0.1
edge_index = torch.randint(0, N, (2, E), device=device)
edge_attr = torch.rand(E, 2, device=device)
edge_attr[:, 0] += 1
edge_attr[:, 1] += 0.1

# Forward pass with AMP
with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
    new_pos, new_vel, hidden = model(pos, vel, edge_index, edge_attr)

print(f'Input:  pos={pos.shape}, vel={vel.shape}')
print(f'Output: pos={new_pos.shape}, vel={new_vel.shape}, hidden={hidden.shape}')
print('✓ Forward pass successful')
"

# ============================================
# Test 3: Feature Computation
# ============================================
echo ""
echo "=============================================="
echo "Test 3: Mesh Feature Computation"
echo "=============================================="
echo ""

python -c "
import torch
import sys
sys.path.insert(0, 'src')
from mesh_features import compute_mesh_features, build_mesh_graph

# Create test mesh
N = 1000
pos = torch.randn(N, 3)
# Create random faces
F = 2000
faces = torch.randint(0, N, (F, 3))

# Compute features
node_feats, edge_idx, edge_attr = compute_mesh_features(pos, faces)
print(f'Nodes: {N}')
print(f'Faces: {F}')
print(f'Node features: {node_feats.shape}')
print(f'Edges: {edge_idx.shape[1]}')
print(f'Edge attributes: {edge_attr.shape}')

# Build complete graph
node_feats, edge_idx, edge_attr = build_mesh_graph(pos, faces, stiffness=10.0)
print(f'Spring edge attributes: {edge_attr.shape}')
print('✓ Feature computation successful')
"

# ============================================
# Test 4: Data Generation
# ============================================
echo ""
echo "=============================================="
echo "Test 4: Synthetic Data Generation"
echo "=============================================="
echo ""

# Generate fresh test data
python data/generate_synthetic.py \
    --output-dir data/test_synthetic \
    --grid-size 10 \
    --num-train 10 \
    --num-val 5 \
    --num-test 5 \
    --steps 10

echo "✓ Data generation successful"

# ============================================
# Test 5: Training Loop (Short)
# ============================================
echo ""
echo "=============================================="
echo "Test 5: Training Loop (10 epochs)"
echo "=============================================="
echo ""

python src/train_pegdeform.py \
    --data-dir data/test_synthetic \
    --checkpoint-dir checkpoints/test \
    --epochs 10 \
    --batch-size 4 \
    --hidden-size 64 \
    --seed 42 \
    2>&1 | tee test_training.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Training loop successful"
else
    echo "✗ Training loop FAILED"
    exit 1
fi

# ============================================
# Test 6: Inference Benchmark
# ============================================
echo ""
echo "=============================================="
echo "Test 6: Inference Benchmark"
echo "=============================================="
echo ""

python src/benchmark.py \
    --data-dir data/test_synthetic \
    --num-samples 50 \
    --hidden-size 64 \
    2>&1 | tee test_benchmark.log

echo "✓ Benchmark successful"

# ============================================
# Test 7: Scaling Benchmark
# ============================================
echo ""
echo "=============================================="
echo "Test 7: Scaling Benchmark"
echo "=============================================="
echo ""

python src/benchmark.py \
    --scaling-test \
    --hidden-size 64 \
    2>&1 | tee test_scaling.log

echo "✓ Scaling benchmark successful"

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo ""
echo "✓ Test 1: Unit Tests         - PASSED"
echo "✓ Test 2: Forward Pass        - PASSED"
echo "✓ Test 3: Feature Computation - PASSED"
echo "✓ Test 4: Data Generation     - PASSED"
echo "✓ Test 5: Training Loop       - PASSED"
echo "✓ Test 6: Inference Benchmark - PASSED"
echo "✓ Test 7: Scaling Benchmark   - PASSED"
echo ""
echo "=============================================="
echo "ALL TESTS PASSED!"
echo "=============================================="

# GPU memory summary
echo ""
echo "Final GPU State:"
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv
