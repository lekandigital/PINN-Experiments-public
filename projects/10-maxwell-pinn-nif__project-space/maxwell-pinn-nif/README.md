# Maxwell-PINN-NIF

**Physics-Informed Neural Network for Full-Vector Maxwell's Equations**

A comprehensive research kit implementing a neural implicit field (NIF) solver for electromagnetic simulations in complex, anisotropic media.

## Update (April 17, 2026)

- This refresh aligns the docs with the current monorepo workflow and shared utility layers.
- The electromagnetics project did not undergo a major private-head layout shift compared with projects 13 and 17.
- Shared benchmarking and export patterns can now be referenced from repo-level tooling where useful.

## Features

- 🧠 **PINN Architecture**: Shared MLP backbone with E/H field output heads
- 📐 **Physics Constraints**: Divergence-free enforcement (∇·εE=0, ∇·μH=0) and curl equation residuals
- 🌊 **PML Boundaries**: Perfectly Matched Layer loss for absorbing boundaries
- ⚡ **Mixed Precision**: FP16 training for GPU memory efficiency
- 🔬 **Validation Suite**: Comprehensive tests against analytical solutions

## Installation

```bash
# Clone or copy the project
cd maxwell-pinn-nif

# Install dependencies
pip install -r requirements.txt

# Optional: Install MEEP for FDTD ground truth
# conda install -c conda-forge pymeep
```

## Quick Start

### Run Validation Tests

```bash
python experiments/quick_test.py
```

### Generate Training Data

```bash
python -c "from src.dataset_generator import generate_training_dataset, save_dataset_hdf5; \
           ds = generate_training_dataset(n_samples=50000, material_type='uniform'); \
           save_dataset_hdf5(ds, 'data/em_data.h5')"
```

### Train the Model

```bash
python src/train.py --epochs 1000 --batch-size 4096 --data data/em_data.h5
```

## Project Structure

```
maxwell-pinn-nif/
├── src/
│   ├── __init__.py           # Package exports
│   ├── pinn_model.py         # MaxwellPINN architecture + physics losses
│   ├── pml_loss.py           # Absorbing boundary conditions
│   ├── dataset_generator.py  # Synthetic data generation
│   └── train.py              # Training loop with mixed-precision
├── experiments/
│   └── quick_test.py         # Validation test suite
├── data/                     # Training datasets (HDF5)
├── checkpoints/              # Model checkpoints
├── notebooks/                # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Architecture

The Maxwell-PINN model enforces physics constraints via automatic differentiation:

```
Input: (x, y, z) coordinates
    ↓
[Optional Fourier Feature Encoding]
    ↓
Shared MLP Backbone (128 hidden × 6 layers, Tanh)
    ↓
┌───────────────────┬───────────────────┐
│   E-field Head    │   H-field Head    │
│  (Ex, Ey, Ez)     │  (Hx, Hy, Hz)     │
└───────────────────┴───────────────────┘
```

### Loss Function

```
L_total = w_pde · L_curl + w_div · L_div + w_data · L_data + w_pml · L_pml

where:
  L_curl = ||∇×E - ωμH||² + ||∇×H - ωεE||²    (Maxwell's curl equations)
  L_div  = ||∇·(εE)||² + ||∇·(μH)||²          (Divergence-free constraint)
  L_data = ||E - E_target||² + ||H - H_target||²  (Supervised loss)
  L_pml  = σ(x) · (|E|² + |H|²)               (PML absorption)
```

## GPU Requirements

- **Minimum**: 8GB VRAM (RTX 3070 / A10)
- **Recommended**: 24GB+ VRAM (RTX 4090 / L40S / A100)

## References

Key papers informing this implementation:

1. Kovacs et al. (2021) - Magnetostatics with PINNs
2. Nohra & Dufour (2024) - PINNs for discontinuous EM media
3. Richter-Powell et al. (2022) - Neural Conservation Laws (divergence-free networks)
4. Shaviner et al. (2025) - PINNs for unsteady Maxwell's equations

## License

MIT License
