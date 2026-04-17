# SurfPINN

Physics-Informed Neural Network for Free-Surface Water Simulations with Dual Eulerian-Lagrangian Architecture.

## Update (April 17, 2026)

- Geometry support was expanded in the current project implementation.
- SurfPINN now sits closer to the repo's coastal-data and shared-geometry infrastructure than earlier README versions implied.
- This doc refresh keeps the project aligned with the current monorepo context.

## Overview

SurfPINN jointly predicts:
- **Eulerian height fields** (grid-based water surface) via 2D CNN
- **Lagrangian particle trajectories** (surface particle paths) via MLP

Both branches share a latent representation and are trained with physics-informed losses including Navier-Stokes residuals, incompressibility constraints, and mean-curvature smoothing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SurfPINN                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ Eulerian Branch │          │Lagrangian Branch│          │
│  │   (2D CNN)      │          │    (MLP)        │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │  Conv2D(32)     │          │   Dense(64)     │          │
│  │  Conv2D(64)     │          │   Dense(64)     │          │
│  │  Conv2D(128)    │          │   Dense(128)    │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│           ▼                            ▼                    │
│        z_eul (128-dim)              z_lag (128-dim)        │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ Height Decoder  │          │Velocity Decoder │          │
│  │  (Nx, Ny, 1)    │          │ (N_particles, 3)│          │
│  └─────────────────┘          └─────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd surfpinn

# Install dependencies (GPU)
pip install -r requirements.txt

# Or for CPU-only testing
pip install -r requirements-cpu.txt
```

### 2. Generate Synthetic Data

```bash
cd src
python data_gen.py
# Creates data/synthetic_dam_break.h5
```

### 3. Run Tests

```bash
# Unit tests
pytest tests/test_model.py -v

# Integration test
python tests/integration_test.py
```

### 4. Train Model

```bash
# Quick test (10 epochs)
python src/train.py --test-mode

# Full training
python src/train.py --epochs 100 --lr 1e-4
```

## vast.ai Deployment

### Search for L40S Instance

```bash
vastai search offers "gpu_name=L40S rentable=true verified=true" \
  --order dph_total --limit 5
```

### Create Instance

```bash
vastai create instance OFFER_ID \
  --image nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 \
  --disk 50
```

### Setup on Instance

```bash
# Install dependencies
pip install "jax[cuda12_pip]" dm-haiku optax h5py matplotlib pytest

# Clone/copy code
cd /workspace
# ... copy surfpinn directory

# Verify GPU
python -c "import jax; print(jax.devices())"

# Run integration test
python tests/integration_test.py

# Train
python src/train.py --epochs 100
```

## Physics Losses

### 1. Data Loss
```
L_data = MSE(height_pred, height_true) + MSE(velocity_pred, velocity_true)
```

### 2. Continuity (Incompressibility)
```
L_cont = mean((∇·u)²)    where ∇·u = ∂u/∂x + ∂v/∂y = 0
```

### 3. Mean Curvature Smoothing
```
κ = ∇·(∇h / √(1 + |∇h|²))
L_curv = mean(κ²)
```

### Total Loss
```
L_total = L_data + 0.1·L_cont + 0.01·L_curv + 0.001·L_latent
```

## Project Structure

```
surfpinn/
├── src/
│   ├── __init__.py
│   ├── model.py        # Dual-branch SurfPINN architecture
│   ├── physics.py      # Physics-informed loss functions
│   ├── data_gen.py     # Synthetic dam-break data generator
│   └── train.py        # Training loop with mixed precision
├── tests/
│   ├── test_model.py       # Unit tests
│   └── integration_test.py # End-to-end validation
├── data/                   # Generated datasets (HDF5)
├── checkpoints/            # Model checkpoints
├── requirements.txt        # GPU dependencies
├── requirements-cpu.txt    # CPU dependencies
└── README.md
```

## Configuration

### Model Config (`SurfPINNConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 128 | Shared latent space dimension |
| `eul_channels` | (32, 64, 128) | Eulerian CNN channels |
| `lag_hidden` | (64, 64, 128) | Lagrangian MLP hidden dims |

### Training Config (`TrainConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Base learning rate |
| `batch_size` | 8 | Batch size |
| `total_epochs` | 100 | Training epochs |
| `gradient_clip` | 1.0 | Gradient clipping norm |
| `use_mixed_precision` | True | Use bfloat16 |

### Physics Config (`PhysicsConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_data` | 1.0 | Data loss weight |
| `weight_continuity` | 0.1 | Incompressibility weight |
| `weight_curvature` | 0.01 | Curvature smoothing weight |

## Metrics

- **PSNR**: Peak Signal-to-Noise Ratio for height field accuracy
- **RMSE**: Root Mean Square Error
- **Relative L2**: Normalized L2 error

## References

Based on research from:
- ELPINN (Thakur & Raissi, 2025)
- DeepLag (Ma et al., NeurIPS 2024)
- SK-PINN (Pan et al., 2024)
- Neural Particle Method (Wessels et al., 2020)

## License

Research code - see LICENSE for details.
