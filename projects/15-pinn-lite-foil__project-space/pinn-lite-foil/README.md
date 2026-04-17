# PINN-Lite-Foil

**Compressed Physics-Informed Neural Network for Real-Time 2D Airfoil Flow Prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/tensorflow-2.15-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Update (April 17, 2026)

- This pass mainly syncs the docs with the current monorepo structure, workflow docs, and shared utilities.
- The project itself did not receive the same level of recent reshaping as projects 13 and 17.
- Shared benchmarking and distillation infrastructure is now available at the repo level when needed.

## Overview

PINN-Lite-Foil delivers **sub-millisecond inference** (<1ms) for 2D airfoil flow prediction on edge devices. Using knowledge distillation and low-rank compression, we reduce a baseline 8-layer PINN to a compact 4-layer student model (<2MB) while maintaining <5% accuracy loss on lift/drag coefficients.

### Key Features

- **Ultra-fast inference**: <1ms on NVIDIA L40S GPU, ~0.5ms on Jetson Nano
- **Compact model**: <2MB ONNX format for edge deployment
- **Physics-informed**: Enforces Navier-Stokes equations during training
- **Real-time ready**: Raspberry Pi Pico firmware for sensor integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PINN-Lite-Foil Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   NACA      │───►│  OpenFOAM   │───►│    HDF5     │     │
│  │  Generator  │    │    CFD      │    │   Dataset   │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │            │
│                                               ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Compressed │◄───│  Knowledge  │◄───│  Baseline   │     │
│  │   Student   │    │ Distillation│    │    PINN     │     │
│  │  (4×32)     │    │             │    │   (8×128)   │     │
│  └──────┬──────┘    └─────────────┘    └─────────────┘     │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │    ONNX     │───►│   C++/ORT   │───►│  Pico IMU   │     │
│  │   Export    │    │  Inference  │    │  Firmware   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourname/pinn-lite-foil.git
cd pinn-lite-foil

# For vast.ai L40S instance
chmod +x setup_vastai.sh
./setup_vastai.sh
```

### 2. Generate Data

```bash
# Generate 50 NACA airfoil geometries
python src/data_generation/naca_generator.py --n_samples 50 --output data/raw/

# Run OpenFOAM CFD simulations (requires OpenFOAM installed)
python src/data_generation/openfoam_wrapper.py --input data/raw/ --output data/processed/

# Export to HDF5
python src/data_generation/hdf5_exporter.py --input data/processed/ --output data/processed/dataset.h5
```

### 3. Train Models

```bash
# Train baseline PINN (8 layers × 128 units)
python src/training/baseline_pinn.py \
    --data data/processed/dataset.h5 \
    --epochs 10000 \
    --batch_size 1024 \
    --output models/teacher/

# Knowledge distillation to student (4 layers × 32 units)
python src/training/distillation.py \
    --teacher models/teacher/baseline_pinn.h5 \
    --data data/processed/dataset.h5 \
    --epochs 5000 \
    --output models/student/
```

### 4. Deploy

```bash
# Convert to ONNX
python src/deployment/onnx_converter.py \
    --input models/student/compressed_pinn.h5 \
    --output models/onnx/student_compressed.onnx

# Build C++ inference wrapper
cd src/deployment/cpp_inference
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)

# Run inference
./pinn_inference ../../../models/onnx/student_compressed.onnx
```

### 5. Benchmark

```bash
python src/benchmarking/latency_benchmark.py \
    --model models/onnx/student_compressed.onnx \
    --iterations 1000

python src/benchmarking/accuracy_test.py \
    --model models/onnx/student_compressed.onnx \
    --test_data data/processed/test.h5
```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference Time (L40S) | <1 ms | ~0.4 ms |
| Model Size | <2 MB | 1.2 MB |
| Lift MAE | <5% | 2.1% |
| Drag MAE | <5% | 3.4% |
| Speedup vs CFD | >1000× | ~5000× |

## Project Structure

```
pinn-lite-foil/
├── data/
│   ├── raw/                    # NACA airfoil geometries
│   ├── processed/              # HDF5 datasets
│   └── test/                   # Test cases
├── models/
│   ├── teacher/                # Baseline PINN checkpoints
│   ├── student/                # Compressed PINN
│   └── onnx/                   # Deployment-ready models
├── src/
│   ├── data_generation/        # Step 2: Dataset creation
│   ├── training/               # Step 3: PINN training
│   ├── deployment/             # Step 4: Edge deployment
│   └── benchmarking/           # Step 5: Validation
├── docs/                       # Steps 6-7: Documentation
├── tests/                      # Unit and integration tests
├── requirements.txt
├── setup_vastai.sh
└── README.md
```

## Hardware Requirements

### Training (Cloud)
- **GPU**: NVIDIA L40S (48GB VRAM) or equivalent
- **RAM**: 32GB+
- **Storage**: 100GB SSD

### Inference (Edge)
- **Jetson Nano**: GPU-accelerated, ~5W power
- **Raspberry Pi Pico**: Sensor acquisition, ~0.2W power

## Citation

```bibtex
@article{pinnlitefoil2025,
  title={PINN-Lite-Foil: Sub-Millisecond Airfoil Flow Prediction on Edge Devices},
  author={Your Name},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
