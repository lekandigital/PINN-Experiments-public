# WavePINN-NIF-ComplexMedia

Physics-Informed Neural Networks with Neural Implicit Fields for acoustic wave propagation in heterogeneous media.

## Update (April 17, 2026)

- This project's docs were synced to the current monorepo shape and shared coastal-data context.
- The repo now expects real-world ocean and acoustics integrations to come through `data/coastal/` and the shared tooling layers.
- No large private-head layout rewrite landed here recently, so this refresh is intentionally documentation-focused.

## Overview

This project implements a **WavePINN-NIF** architecture that combines:
- **Physics-Informed Neural Networks (PINNs)** for solving the acoustic wave equation
- **Neural Implicit Fields (NIFs)** for representing heterogeneous velocity models

### Key Features

- ✅ JAX/Haiku implementation with JIT compilation
- ✅ Fourier feature encoding to overcome spectral bias
- ✅ Support for 2D and 3D domains
- ✅ Dirichlet, Neumann, and absorbing boundary conditions
- ✅ Mixed precision training support
- ✅ Weights & Biases integration for experiment tracking
- ✅ Checkpoint save/restore

## Governing Equation

The acoustic wave equation in heterogeneous media:

$$\frac{\partial^2 u}{\partial t^2} - c^2(\mathbf{x}) \nabla^2 u = 0$$

where:
- $u(\mathbf{x}, t)$ is the wavefield
- $c(\mathbf{x})$ is the spatially-varying wave speed

## Project Structure

```
WavePINN-NIF-ComplexMedia/
├── config/
│   └── default_config.yaml    # Default configuration
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic wave speed maps + sources
│   ├── model.py               # Neural implicit field architecture
│   ├── physics_loss.py        # PDE residuals + BC/IC terms
│   └── trainer.py             # Training loop with mixed precision
├── tests/
│   └── test_forward_sim.py    # Quick sanity checks
├── notebooks/                 # Jupyter notebooks (to be added)
├── checkpoints/               # Model checkpoints
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.1+ and cuDNN 8+
- Python 3.10+
- Conda (recommended)

### Setup on Vast.ai / Cloud GPU

```bash
# 1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
export PATH="/opt/conda/bin:$PATH"

# 2. Create environment
conda create -n wavepinn python=3.10 -y
conda activate wavepinn

# 3. Install dependencies
pip install -r requirements.txt
```

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd WavePINN-NIF-ComplexMedia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Tests

```bash
# From project root
python tests/test_forward_sim.py
```

Expected output:
```
============================================================
WavePINN-NIF Quick Test
============================================================

[1/6] Initializing model...
✓ Model initialized successfully (0.52s)
  Parameter count: 394,241

[2/6] Generating synthetic data...
✓ Data generated successfully
  Interior points: (1000, 3)

...

============================================================
✅ ALL TESTS PASSED - Model is functional!
============================================================
```

### Training

```python
from src.trainer import WavePINNTrainer
from src.data_generator import SyntheticWaveData
import yaml

# Load configuration
with open('config/default_config.yaml') as f:
    config = yaml.safe_load(f)

# Generate synthetic data
data_gen = SyntheticWaveData(
    domain_size=(1.0, 1.0),
    n_collocation=10000,
    n_boundary=1000,
    n_initial=1000
)
data = data_gen.sample_collocation_points()

# Create trainer and train
trainer = WavePINNTrainer(config, use_wandb=True)
params = trainer.train(data, n_epochs=5000)
```

### Quick Training Test

```python
from src.trainer import quick_train

# Run quick training for testing
trainer, metrics = quick_train(n_epochs=100)
print(f"Final loss: {metrics['loss_total']:.6f}")
```

## Configuration

Edit `config/default_config.yaml` to customize:

```yaml
# Domain
domain:
  spatial_dims: 2
  domain_size: [1.0, 1.0]
  t_max: 1.0

# Model architecture
model:
  wave_net:
    hidden_dims: [256, 256, 256, 256]
    use_fourier: true
    fourier_sigma: 10.0

# Training
training:
  n_epochs: 5000
  batch_size: 1024
  learning_rate: 1.0e-3

# Loss weights
loss_weights:
  lambda_pde: 1.0
  lambda_bc: 10.0
  lambda_ic: 10.0
```

## Architecture

### WavePINN Network

```
Input: (x, y, t) or (x, y, z, t)
    ↓
Fourier Feature Encoder (σ=10.0)
    ↓
MLP [256, 256, 256, 256] with tanh + residual connections
    ↓
Output: u (wavefield scalar)
```

### MediaNIF Network

```
Input: (x, y) or (x, y, z)
    ↓
Fourier Feature Encoder (σ=5.0)
    ↓
MLP [128, 128] with softplus
    ↓
Output: c (wave speed, constrained to [c_min, c_max])
```

## Velocity Models

The data generator supports:

1. **Homogeneous**: Constant wave speed
2. **Layered**: Horizontal layers with random velocities
3. **Random**: Gaussian random field with controllable correlation length

```python
# Generate different velocity models
c_homo = data_gen.generate_homogeneous_medium(c_value=3.0)
c_layer = data_gen.generate_layered_medium(n_layers=5)
c_random = data_gen.generate_random_medium(correlation_length=0.1)
```

## GPU Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 8 GB | 24+ GB (L40S/A100) |
| System RAM | 16 GB | 32+ GB |
| CUDA | 12.1+ | 12.1+ |

## Troubleshooting

### CUDA Errors
```bash
# Check CUDA availability
nvidia-smi
python -c "import jax; print(jax.devices())"
```

### Out of Memory
- Reduce `batch_size` in config
- Reduce `n_collocation` in data generator
- Use gradient checkpointing (not yet implemented)

### NaN Losses
- Reduce `learning_rate`
- Check `fourier_sigma` (try 5.0-20.0)
- Verify input data normalization

### Slow First Run
- Normal! First JAX JIT compilation takes 30-60s
- Subsequent runs are fast (cached)

## Citation

If you use this code, please cite:

```bibtex
@software{wavepinn_nif_2024,
  title = {WavePINN-NIF: Physics-Informed Neural Networks with Neural Implicit Fields for Wave Propagation},
  year = {2024},
  author = {WavePINN Research Team}
}
```

## License

MIT License

## Acknowledgments

- JAX/Haiku teams at Google DeepMind
- PINN literature (Raissi et al., Karniadakis et al.)
- NeRF/NIF community (Mildenhall et al., Tancik et al.)
