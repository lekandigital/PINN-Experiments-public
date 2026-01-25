# WavePINN-NIF-Scalar

A **Physics-Informed Neural Implicit Field (PINN-NIF)** solver for the 2D/3D acoustic scalar wave equation in heterogeneous media. Built with JAX + Haiku for GPU-accelerated autodifferentiation.

## 🎯 Overview

This framework solves the acoustic wave equation:

$$\frac{\partial^2 u}{\partial t^2} = c^2(\mathbf{x}) \nabla^2 u + s(\mathbf{x}, t)$$

where:
- $u(\mathbf{x}, t)$: scalar pressure/displacement field
- $c(\mathbf{x}) = 1/m(\mathbf{x})$: spatially-varying velocity (inverse of slowness $m$)
- $s(\mathbf{x}, t)$: source term (Ricker wavelet)

**Key Features:**
- 🔥 JAX + Haiku for fast autodiff and GPU acceleration
- 📐 Fourier feature encoding to mitigate spectral bias
- ⚡ Mixed-precision (bfloat16) training for L40S/A100/H100 GPUs
- 🔄 Inverse problem support for slowness reconstruction
- 📊 W&B integration for experiment tracking

## 🏗️ Project Structure

```
wavepinn_nif_scalar/
├── configs/
│   ├── environment.yml      # Conda environment
│   ├── Dockerfile           # Container definition
│   └── sweep.yaml           # W&B hyperparameter sweep
├── src/
│   ├── __init__.py
│   ├── data_gen.py          # Synthetic data generation
│   ├── model.py             # WavePINN architecture
│   ├── training.py          # Training loop
│   ├── inverse.py           # Inverse problem solver
│   └── utils.py             # Utilities
├── tests/
│   ├── test_data_gen.py
│   ├── test_model.py
│   └── test_integration.py
├── notebooks/
│   ├── visualization_demo.ipynb
│   └── inverse_demo.ipynb
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🚀 Quick Start

### Installation

**Option 1: pip**
```bash
cd wavepinn_nif_scalar
pip install -e .
```

**Option 2: conda**
```bash
conda env create -f configs/environment.yml
conda activate wavepinn
pip install -e .
```

**Option 3: Docker (GPU)**
```bash
docker build -t wavepinn -f configs/Dockerfile .
docker run --gpus all -it wavepinn
```

### Minimal Example

```python
import jax
import jax.numpy as jnp
import jax.random as jr

from src.data_gen import generate_training_dataset
from src.model import WavePINN, WavePINNConfig
from src.training import Trainer, TrainingConfig

# Generate synthetic data
data = generate_training_dataset(
    seed=42,
    nx=80, nz=80,
    n_interior=5000,
    n_boundary=1000,
    n_initial=500
)

# Create model
config = WavePINNConfig(
    hidden_dims=[128, 128, 64],
    use_fourier_features=True,
    num_fourier_features=64
)
pinn = WavePINN(config, seed=42)

# Train
train_config = TrainingConfig(
    n_epochs=100,
    learning_rate=1e-3,
    batch_size=4096
)
trainer = Trainer(pinn, train_config)
params = trainer.train(data)

# Predict
coords = jnp.array([[0.5, 0.5, 0.2]])  # (x, z, t)
u = pinn.forward(params, coords)
print(f"Wavefield at (0.5, 0.5, t=0.2): {u}")
```

## 📦 Modules

### `src/data_gen.py` - Data Generation

| Function | Description |
|----------|-------------|
| `generate_slowness_map_2d()` | Create random velocity/slowness models with anomalies |
| `ricker_wavelet()` | Generate Ricker (Mexican hat) source wavelet |
| `sample_collocation_points_2d()` | Latin Hypercube sampling for interior, boundary, initial points |
| `generate_training_dataset()` | Complete dataset generation pipeline |
| `export_dataset_hdf5()` | Save dataset to HDF5 for reproducibility |

### `src/model.py` - WavePINN Architecture

| Class/Function | Description |
|----------------|-------------|
| `WavePINNConfig` | Configuration dataclass for model hyperparameters |
| `WavePINN` | Main PINN class with forward pass and loss functions |
| `WavePINN.compute_pde_residual_2d_efficient()` | Compute $\text{res} = u_{tt} - c^2(u_{xx} + u_{zz})$ via autodiff |
| `WavePINN.total_loss()` | Combined PDE + BC + IC loss |
| `predict_on_grid()` | Predict wavefield on 2D spatial grid at fixed time |
| `predict_time_series()` | Predict wavefield time series at fixed location |

### `src/training.py` - Training Infrastructure

| Class | Description |
|-------|-------------|
| `TrainingConfig` | Training hyperparameters (epochs, LR, batch size, etc.) |
| `TrainState` | Training state with params, optimizer state, metrics |
| `Trainer` | Full training loop with mixed-precision, gradient clipping, checkpointing |

### `src/inverse.py` - Inverse Problem

| Class/Function | Description |
|----------------|-------------|
| `InverseConfig` | Inverse problem configuration |
| `InverseProblem` | Joint wavefield + slowness optimization |
| `generate_synthetic_observations()` | Create noisy boundary observations |
| `evaluate_slowness_reconstruction()` | Compute reconstruction metrics |

### `src/utils.py` - Utilities

| Function | Description |
|----------|-------------|
| `init_fourier_basis()` | Initialize random Fourier feature basis |
| `make_fourier_features()` | Apply Fourier feature transform |
| `cast_to_bfloat16()` | Mixed-precision casting |
| `save_checkpoint()` / `load_checkpoint()` | Model checkpointing |
| `MetricsLogger` | Training metrics tracking |

## ⚙️ Configuration

### Model Configuration

```python
config = WavePINNConfig(
    hidden_dims=[256, 256, 128, 64],  # MLP architecture
    activation='tanh',                 # Activation function
    use_fourier_features=True,         # Enable Fourier encoding
    num_fourier_features=128,          # Number of Fourier features
    fourier_scale=10.0,                # Fourier feature scale σ
    lambda_pde=1.0,                    # PDE loss weight
    lambda_bc=10.0,                    # Boundary loss weight
    lambda_ic=10.0,                    # Initial condition weight
)
```

### Training Configuration

```python
train_config = TrainingConfig(
    n_epochs=1000,
    learning_rate=1e-3,
    weight_decay=1e-5,
    batch_size=8192,           # Increase for L40S (48GB)
    use_mixed_precision=True,  # Enable bfloat16
    gradient_clip_norm=1.0,
    lr_schedule='cosine',
    warmup_epochs=50,
    log_every=10,
    checkpoint_every=100,
)
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# Acceptance criteria test
pytest tests/test_integration.py::TestMinimalAcceptance -v
```

### Success Criteria (from spec)

The acceptance test validates:
1. ✅ Data generation produces valid slowness maps
2. ✅ Model forward pass returns finite values
3. ✅ PDE loss is computable via autodiff
4. ✅ Training reduces loss by >50% in 50 epochs
5. ✅ Predictions are finite and bounded (|u| < 10)

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/visualization_demo.ipynb` | Wavefield snapshots, PDE residual heatmaps, training curves |
| `notebooks/inverse_demo.ipynb` | Slowness reconstruction from boundary measurements |

## 🖥️ GPU Performance

### L40S (48GB VRAM)

| Batch Size | Memory | Speed |
|------------|--------|-------|
| 8,192 | ~8 GB | ~150 it/s |
| 32,768 | ~25 GB | ~120 it/s |
| 65,536 | ~40 GB | ~90 it/s |

### Enabling Multi-GPU

```python
import jax

# Check available devices
print(jax.devices())  # Should show multiple GPUs

# Training automatically uses pmap for multi-GPU
trainer = Trainer(pinn, config, multi_gpu=True)
```

## 📈 Weights & Biases Integration

```bash
# Login
wandb login

# Run sweep
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>
```

Track experiments:
```python
import wandb
wandb.init(project="wavepinn-nif")
wandb.config.update(config.__dict__)
# ... training
wandb.log({"loss": loss, "pde_loss": pde_loss})
```

## 🔬 Physics Background

### Acoustic Wave Equation

The 2D scalar wave equation in heterogeneous media:

$$\frac{\partial^2 u}{\partial t^2} = c^2(x, z) \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial z^2} \right) + s(x, z, t)$$

### PINN Loss Function

$$\mathcal{L}_{\text{total}} = \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}}$$

where:
- $\mathcal{L}_{\text{PDE}} = \frac{1}{N_r} \sum_{i=1}^{N_r} |u_{tt} - c^2 \nabla^2 u - s|^2$ (interior residual)
- $\mathcal{L}_{\text{BC}} = \frac{1}{N_b} \sum_{i=1}^{N_b} |u - u_{\text{BC}}|^2$ (boundary)
- $\mathcal{L}_{\text{IC}} = \frac{1}{N_0} \sum_{i=1}^{N_0} (|u_0 - u(t=0)|^2 + |u_t - \partial_t u(t=0)|^2)$ (initial)

### Fourier Feature Encoding

To overcome spectral bias for high-frequency waves:

$$\gamma(\mathbf{x}) = [\cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x})]$$

where $\mathbf{B} \in \mathbb{R}^{m \times d}$ with entries sampled from $\mathcal{N}(0, \sigma^2)$.

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS*.

3. Song, C., Alkhalifah, T., & Waheed, U. bin. (2021). Solving the frequency-domain acoustic VTI wave equation using physics-informed neural networks. *Geophysical Journal International*, 225(2), 846-859.

## 📄 License

MIT License

## 🙋 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
