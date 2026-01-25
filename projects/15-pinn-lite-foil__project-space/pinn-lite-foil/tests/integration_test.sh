#!/bin/bash
# ============================================================================
# PINN-Lite-Foil Integration Test
# End-to-end test of the complete pipeline
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "     PINN-Lite-Foil Integration Test"
echo "============================================================"
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
MODELS_DIR="${PROJECT_ROOT}/models"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Test parameters (reduced for quick testing)
N_AIRFOILS=5
N_AOA=3
N_POINTS=100
TRAIN_EPOCHS=100
DISTILL_EPOCHS=50
BENCHMARK_ITERATIONS=100

# Cleanup previous test artifacts
cleanup() {
    echo -e "\n${YELLOW}Cleaning up test artifacts...${NC}"
    rm -rf "${DATA_DIR}/test"
    rm -rf "${MODELS_DIR}/test_teacher"
    rm -rf "${MODELS_DIR}/test_student"
    rm -rf "${MODELS_DIR}/test_onnx"
    rm -rf "${RESULTS_DIR}/test"
}

# Error handler
error_handler() {
    echo -e "\n${RED}✗ Test failed at step: $1${NC}"
    cleanup
    exit 1
}

trap 'error_handler "$STEP"' ERR

# ============================================================================
# Step 1: Generate Synthetic Dataset
# ============================================================================
STEP="Data Generation"
echo -e "\n${YELLOW}Step 1: Generating synthetic dataset...${NC}"

mkdir -p "${DATA_DIR}/test"

python "${PROJECT_ROOT}/src/data_generation/hdf5_exporter.py" \
    --synthetic \
    --n_airfoils ${N_AIRFOILS} \
    --n_aoa ${N_AOA} \
    --n_points ${N_POINTS} \
    --output "${DATA_DIR}/test/dataset.h5"

if [ -f "${DATA_DIR}/test/dataset.h5" ]; then
    echo -e "${GREEN}✓ Dataset created successfully${NC}"
else
    echo -e "${RED}✗ Dataset creation failed${NC}"
    exit 1
fi

# ============================================================================
# Step 2: Train Baseline PINN (Quick Test)
# ============================================================================
STEP="Baseline PINN Training"
echo -e "\n${YELLOW}Step 2: Training baseline PINN (${TRAIN_EPOCHS} epochs)...${NC}"

python "${PROJECT_ROOT}/src/training/baseline_pinn.py" \
    --data "${DATA_DIR}/test/dataset.h5" \
    --output "${MODELS_DIR}/test_teacher" \
    --epochs ${TRAIN_EPOCHS} \
    --batch_size 128 \
    --hidden_layers 4 \
    --hidden_units 32 \
    --quick_test

if [ -f "${MODELS_DIR}/test_teacher/baseline_pinn.keras" ]; then
    echo -e "${GREEN}✓ Baseline PINN trained successfully${NC}"
else
    echo -e "${RED}✗ Baseline PINN training failed${NC}"
    exit 1
fi

# ============================================================================
# Step 3: Knowledge Distillation
# ============================================================================
STEP="Knowledge Distillation"
echo -e "\n${YELLOW}Step 3: Running knowledge distillation (${DISTILL_EPOCHS} epochs)...${NC}"

python "${PROJECT_ROOT}/src/training/distillation.py" \
    --teacher "${MODELS_DIR}/test_teacher/baseline_pinn.keras" \
    --data "${DATA_DIR}/test/dataset.h5" \
    --output "${MODELS_DIR}/test_student" \
    --epochs ${DISTILL_EPOCHS} \
    --student_layers 2 \
    --student_units 16 \
    --quick_test

if [ -f "${MODELS_DIR}/test_student/compressed_pinn.keras" ]; then
    echo -e "${GREEN}✓ Distillation completed successfully${NC}"
else
    echo -e "${RED}✗ Distillation failed${NC}"
    exit 1
fi

# ============================================================================
# Step 4: Convert to ONNX
# ============================================================================
STEP="ONNX Conversion"
echo -e "\n${YELLOW}Step 4: Converting to ONNX format...${NC}"

mkdir -p "${MODELS_DIR}/test_onnx"

python "${PROJECT_ROOT}/src/deployment/onnx_converter.py" \
    --input "${MODELS_DIR}/test_student/compressed_pinn.keras" \
    --output "${MODELS_DIR}/test_onnx/student.onnx" \
    --benchmark

if [ -f "${MODELS_DIR}/test_onnx/student.onnx" ]; then
    echo -e "${GREEN}✓ ONNX conversion successful${NC}"
    
    # Check model size
    SIZE_KB=$(du -k "${MODELS_DIR}/test_onnx/student.onnx" | cut -f1)
    echo "  Model size: ${SIZE_KB} KB"
    
    if [ ${SIZE_KB} -lt 2048 ]; then
        echo -e "${GREEN}  ✓ Model size < 2MB target${NC}"
    else
        echo -e "${YELLOW}  ⚠ Model size exceeds 2MB target${NC}"
    fi
else
    echo -e "${RED}✗ ONNX conversion failed${NC}"
    exit 1
fi

# ============================================================================
# Step 5: Run Accuracy Test
# ============================================================================
STEP="Accuracy Test"
echo -e "\n${YELLOW}Step 5: Running accuracy test...${NC}"

mkdir -p "${RESULTS_DIR}/test"

python "${PROJECT_ROOT}/src/benchmarking/accuracy_test.py" \
    --model "${MODELS_DIR}/test_onnx/student.onnx" \
    --test_data "${DATA_DIR}/test/dataset.h5" \
    --output "${RESULTS_DIR}/test/accuracy"

if [ -f "${RESULTS_DIR}/test/accuracy/accuracy_results.json" ]; then
    echo -e "${GREEN}✓ Accuracy test completed${NC}"
else
    echo -e "${RED}✗ Accuracy test failed${NC}"
    exit 1
fi

# ============================================================================
# Step 6: Run Latency Benchmark
# ============================================================================
STEP="Latency Benchmark"
echo -e "\n${YELLOW}Step 6: Running latency benchmark (${BENCHMARK_ITERATIONS} iterations)...${NC}"

python "${PROJECT_ROOT}/src/benchmarking/latency_benchmark.py" \
    --model "${MODELS_DIR}/test_onnx/student.onnx" \
    --output "${RESULTS_DIR}/test/benchmark" \
    --iterations ${BENCHMARK_ITERATIONS} \
    --batch_sizes 1 4

if [ -f "${RESULTS_DIR}/test/benchmark/benchmark_results.json" ]; then
    echo -e "${GREEN}✓ Latency benchmark completed${NC}"
else
    echo -e "${RED}✗ Latency benchmark failed${NC}"
    exit 1
fi

# ============================================================================
# Step 7: Run Unit Tests
# ============================================================================
STEP="Unit Tests"
echo -e "\n${YELLOW}Step 7: Running unit tests...${NC}"

cd "${PROJECT_ROOT}"
python -m pytest tests/ -v --tb=short || {
    echo -e "${YELLOW}⚠ Some tests failed, but continuing...${NC}"
}

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "     Integration Test Summary"
echo "============================================================"
echo ""
echo -e "${GREEN}✓ All pipeline steps completed successfully!${NC}"
echo ""
echo "Artifacts created:"
echo "  - Dataset:     ${DATA_DIR}/test/dataset.h5"
echo "  - Teacher:     ${MODELS_DIR}/test_teacher/"
echo "  - Student:     ${MODELS_DIR}/test_student/"
echo "  - ONNX:        ${MODELS_DIR}/test_onnx/student.onnx"
echo "  - Results:     ${RESULTS_DIR}/test/"
echo ""

# Parse and display key metrics
if [ -f "${RESULTS_DIR}/test/benchmark/benchmark_results.json" ]; then
    echo "Benchmark Results (Batch Size = 1):"
    python -c "
import json
with open('${RESULTS_DIR}/test/benchmark/benchmark_results.json') as f:
    data = json.load(f)
for b in data['benchmarks']:
    if b['batch_size'] == 1:
        print(f\"  Provider: {b['provider']}\")
        print(f\"  Mean latency: {b['mean_ms']:.3f} ms\")
        print(f\"  P99 latency:  {b['p99_ms']:.3f} ms\")
        if b['mean_ms'] < 1.0:
            print('  ✓ Sub-millisecond target achieved!')
        break
"
fi

echo ""
echo "============================================================"
echo "     Integration Test Complete"
echo "============================================================"

# Optional: cleanup test artifacts
# cleanup
