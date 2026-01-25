# SurfPINN GPU Validation & Optimization - Copilot Agent Instructions

## Context
You have access to a complete SurfPINN implementation at `surfpinn/` that has passed all CPU tests. The goal is to validate and optimize training on an NVIDIA RTX 3090 Ti GPU.

**Remote Machine Details:**
- Host: `REDACTED_SERVER` (SSH from local)
- Password: `REDACTED_PASSWORD` (sudo)
- GPU: NVIDIA GeForce RTX 3090 Ti (24GB VRAM)
- CUDA Version: 13.0
- Driver: 580.82.09
- OS: Ubuntu 22.04.5 LTS
- Python: 3.11 available at `/usr/local/bin/python3.11`

**Current Status:**
- ✅ 8 Python modules implemented (model.py, physics.py, train.py, data_gen.py, tests)
- ✅ 21/21 unit tests passing on CPU
- ✅ Integration test passing (85% loss reduction in 10 steps)
- ❌ Never tested on GPU hardware
- ⚠️ PSNR was -3.88 dB on full training (needs tuning)

---

## Mission
Deploy the SurfPINN codebase to the RTX 3090 Ti, validate GPU training, optimize hyperparameters to achieve meaningful PSNR (>15 dB), and document results.

---

## Phase 1: Remote Environment Setup

### Step 1.1: Transfer Code to Remote Machine
```bash
# From LOCAL machine, sync the project to remote
rsync -avz --progress \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/16-surfpinn__project-space/surfpinn/ \
  REDACTED_SERVER:~/surfpinn/

# Verify transfer
ssh REDACTED_SERVER "ls -la ~/surfpinn/src/"
```

### Step 1.2: Install JAX with CUDA Support
```bash
# SSH into remote machine
ssh REDACTED_SERVER

# Navigate to project
cd ~/surfpinn

# Create virtual environment (recommended)
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate

# Install JAX with CUDA 12+ support
pip install --upgrade pip
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install dm-haiku optax chex h5py numpy scipy matplotlib pytest

# Verify GPU detection
python -c "import jax; print(f'Devices: {jax.devices()}'); print(f'GPU: {jax.devices()[0].device_kind}')"
```

**Expected Output:**
```
Devices: [cuda(id=0)]
GPU: NVIDIA GeForce RTX 3090 Ti
```

### Step 1.3: Validate Integration Test on GPU
```bash
cd ~/surfpinn
python tests/integration_test.py 2>&1

# Expected: Should show "cuda(id=0)" and run significantly faster than CPU
```

---

## Phase 2: GPU Training Validation

### Step 2.1: Run Test Training with GPU Metrics
```bash
cd ~/surfpinn/src

# Short training run with metrics
python train.py --test-mode 2>&1

# Monitor GPU usage in separate terminal
watch -n 1 nvidia-smi
```

**Success Criteria:**
- [ ] JAX detects GPU (`cuda(id=0)`)
- [ ] Training step time < 0.1s (vs ~0.3s on CPU)
- [ ] GPU memory usage < 20GB
- [ ] Loss decreases monotonically
- [ ] No NaN/Inf in outputs

### Step 2.2: Run Full Training (100 Epochs)
```bash
cd ~/surfpinn/src

# Create data directory
mkdir -p data checkpoints

# Generate synthetic dataset
python -c "
import sys; sys.path.insert(0, '.')
from data_gen import generate_dataset
generate_dataset('data/synthetic_dam_break.h5', nx=64, ny=64, nt=32, n_particles=2000)
print('Dataset generated!')
"

# Full training with checkpointing
python train.py --epochs 100 --batch-size 8 --lr 1e-4 2>&1 | tee training_log.txt

# If OOM, reduce batch size:
# python train.py --epochs 100 --batch-size 4 --lr 1e-4
```

**Log Key Metrics:**
- Initial loss
- Final loss after 100 epochs
- Loss reduction percentage
- Average epoch time
- Peak GPU memory (from nvidia-smi)
- Final PSNR and RMSE

---

## Phase 3: Hyperparameter Tuning

The initial implementation has PSNR of -3.88 dB, which indicates the model is not learning the height field accurately. This is likely due to:
1. Curvature loss weight too high (dominating data loss)
2. Learning rate may need adjustment
3. Training duration may be insufficient

### Step 3.1: Analyze Current Loss Weights
Current configuration in `src/physics.py`:
```python
weight_data = 1.0        # Data fitting loss
weight_continuity = 0.1  # Divergence-free constraint
weight_momentum = 0.1    # Shallow water momentum
weight_kinematic = 0.1   # Free surface condition
weight_curvature = 0.01  # Surface smoothness
```

**From training output:** Curvature loss (212.7) >> Data loss (0.21), indicating curvature term is dominating.

### Step 3.2: Experiment with Loss Weight Adjustments

Create a hyperparameter sweep script at `src/hp_sweep.py`:

```python
#!/usr/bin/env python3
"""Hyperparameter sweep for SurfPINN loss weights."""

import subprocess
import json
from datetime import datetime

# Sweep configurations
CONFIGS = [
    {"weight_curvature": 0.01, "weight_data": 1.0},   # Original
    {"weight_curvature": 0.001, "weight_data": 1.0},  # Reduce curvature 10x
    {"weight_curvature": 0.0001, "weight_data": 1.0}, # Reduce curvature 100x
    {"weight_curvature": 0.001, "weight_data": 10.0}, # Boost data loss
    {"weight_curvature": 0.0, "weight_data": 1.0},    # No curvature (ablation)
]

def run_experiment(config, epochs=50):
    """Run training with specific config and return metrics."""
    # Modify physics.py config temporarily
    # ... implementation details ...
    pass

if __name__ == "__main__":
    results = []
    for config in CONFIGS:
        metrics = run_experiment(config)
        results.append({"config": config, "metrics": metrics})

    with open(f"hp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=2)
```

### Step 3.3: Recommended Tuning Experiments

| Experiment | Change | Rationale |
|------------|--------|-----------|
| **A: Lower curvature** | `weight_curvature=0.001` | Curvature loss dominates; reduce to let data loss guide learning |
| **B: No curvature** | `weight_curvature=0.0` | Ablation study to see data-only performance |
| **C: Boost data** | `weight_data=10.0` | Force model to prioritize fitting ground truth |
| **D: Lower LR** | `lr=1e-5` | Slower training may improve convergence |
| **E: Longer training** | `epochs=500` | More iterations for physics to converge |

Run each experiment and record:
- Final loss breakdown (H_MSE, Curv, Grad, etc.)
- PSNR on test data
- Visual inspection of height field predictions

---

## Phase 4: Model Architecture Improvements

If hyperparameter tuning doesn't achieve PSNR > 15 dB, consider these model modifications:

### Step 4.1: Add Residual Connections
In `src/model.py`, the Eulerian encoder uses skip connections but decoder doesn't. Add:

```python
class EulerianDecoder(hk.Module):
    def __call__(self, z):
        # Current: z -> conv -> conv -> conv -> height
        # Improved: z -> conv -> conv + residual -> conv + residual -> height

        h = hk.Conv2D(64, 3, padding="SAME")(z)
        h = jax.nn.gelu(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h))

        # Add residual from input (project if needed)
        if z.shape[-1] != 64:
            residual = hk.Conv2D(64, 1)(z)
        else:
            residual = z
        h = h + residual

        # Continue with more layers...
        return height
```

### Step 4.2: Increase Model Capacity
Current: ~230K parameters. For complex physics, may need more:

```python
# In model.py, increase channel widths:
SurfPINNConfig(
    latent_dim=256,         # Was 128
    eul_channels=(64, 128, 256),  # Was (32, 64, 128)
    lag_hidden=(128, 128, 256),   # Was (64, 64, 128)
)
```

**Note:** This will increase memory usage. Monitor with nvidia-smi.

### Step 4.3: Implement Fourier Feature Encoding
PINNs often benefit from Fourier features for high-frequency learning:

```python
def fourier_features(x, num_frequencies=10, scale=1.0):
    """Encode positions with sinusoidal features."""
    freqs = 2.0 ** jnp.arange(num_frequencies) * scale
    x_freq = x[..., None] * freqs  # [..., D, num_freq]
    return jnp.concatenate([jnp.sin(x_freq), jnp.cos(x_freq)], axis=-1).reshape(*x.shape[:-1], -1)

# Use in LagrangianEncoder:
def __call__(self, positions):
    x = fourier_features(positions, num_frequencies=6)  # 3 * 6 * 2 = 36 features
    x = hk.Linear(self.hidden_dims[0])(x)
    ...
```

---

## Phase 5: Visualization & Results Documentation

### Step 5.1: Generate Visualization Outputs
Create `src/visualize.py`:

```python
#!/usr/bin/env python3
"""Generate visualizations of SurfPINN predictions."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_height_comparison(true_height, pred_height, timestep, save_path):
    """Side-by-side height field comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].imshow(true_height[:, :, timestep, 0], cmap='viridis')
    axes[0].set_title(f'Ground Truth (t={timestep})')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pred_height[:, :, timestep, 0], cmap='viridis')
    axes[1].set_title(f'Prediction (t={timestep})')
    plt.colorbar(im2, ax=axes[1])

    error = np.abs(true_height[:, :, timestep, 0] - pred_height[:, :, timestep, 0])
    im3 = axes[2].imshow(error, cmap='hot')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def animate_particles(positions, save_path):
    """Create animation of Lagrangian particle trajectories."""
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter([], [], c='blue', s=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        scatter.set_offsets(positions[:, frame, :2])
        ax.set_title(f'Particles (t={frame})')
        return scatter,

    anim = animation.FuncAnimation(fig, update, frames=positions.shape[1], blit=True)
    anim.save(save_path, writer='pillow', fps=5)
    plt.close()

if __name__ == "__main__":
    # Load best checkpoint
    ckpt = load_checkpoint('checkpoints/surfpinn_best.pkl')

    # Generate predictions on test data
    # ... load data, run inference ...

    # Create visualizations
    plot_height_comparison(true_h, pred_h, timestep=8, save_path='results/height_comparison.png')
    animate_particles(pred_positions, save_path='results/particles.gif')
```

### Step 5.2: Create Test Report
After training completes, create `TEST_REPORT.md`:

```markdown
# SurfPINN GPU Training Report

## Hardware
- GPU: NVIDIA RTX 3090 Ti (24GB)
- CUDA: 13.0
- Driver: 580.82.09

## Training Configuration
- Epochs: 100
- Batch size: 8
- Learning rate: 1e-4 (cosine decay)
- Loss weights: data=1.0, curv=0.001, cont=0.1

## Results

| Metric | Value |
|--------|-------|
| Initial Loss | X.XXX |
| Final Loss | X.XXX |
| Loss Reduction | XX.X% |
| Final PSNR | XX.X dB |
| Height RMSE | X.XXXX |
| Velocity RMSE | X.XXXX |
| Avg Epoch Time | X.Xs |
| Peak GPU Memory | XX GB |

## Visualizations
![Height Comparison](results/height_comparison.png)
![Particles](results/particles.gif)

## Observations
- [Key findings from training]
- [What worked / didn't work]
- [Recommendations for improvement]
```

---

## Phase 6: Sync Results Back to Local

### Step 6.1: Transfer Results from Remote to Local
```bash
# From LOCAL machine
rsync -avz --progress \
  REDACTED_SERVER:~/surfpinn/checkpoints/ \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/16-surfpinn__project-space/surfpinn/checkpoints/

rsync -avz --progress \
  REDACTED_SERVER:~/surfpinn/results/ \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/16-surfpinn__project-space/surfpinn/results/

rsync -avz --progress \
  REDACTED_SERVER:~/surfpinn/TEST_REPORT.md \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/16-surfpinn__project-space/surfpinn/

rsync -avz --progress \
  REDACTED_SERVER:~/surfpinn/training_log.txt \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/16-surfpinn__project-space/surfpinn/
```

---

## Success Criteria Checklist

### Minimum (Must Achieve)
- [ ] JAX detects RTX 3090 Ti GPU
- [ ] Integration test passes on GPU
- [ ] Full 100-epoch training completes without errors
- [ ] Loss decreases by >50%
- [ ] No NaN/Inf in predictions
- [ ] GPU memory < 20GB
- [ ] Training log saved to `training_log.txt`

### Target (Should Achieve)
- [ ] PSNR > 15 dB on test data
- [ ] Height RMSE < 0.1
- [ ] Visualization outputs generated
- [ ] TEST_REPORT.md completed
- [ ] Results synced back to local machine

### Stretch (Nice to Have)
- [ ] Hyperparameter sweep completed
- [ ] PSNR > 20 dB achieved through tuning
- [ ] Fourier features or residual connections implemented
- [ ] Particle trajectory animation created

---

## Troubleshooting

### JAX doesn't detect GPU
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Force GPU backend
export JAX_PLATFORMS=cuda
python -c "import jax; print(jax.devices())"
```

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch-size 4

# Or enable gradient checkpointing in train.py:
# Add @jax.checkpoint decorator to heavy functions
```

### Training is slow on GPU
```bash
# Ensure XLA is compiling
export XLA_FLAGS="--xla_gpu_autotune_level=2"

# Profile
python -c "
import jax
jax.profiler.start_trace('./jax-trace')
# ... run train step ...
jax.profiler.stop_trace()
"
# Then view with TensorBoard
```

### NaN losses
- Check learning rate (try 1e-5)
- Add gradient clipping in train.py
- Verify physics loss weights aren't too high
- Check for division by zero in curvature calculation

---

## Quick Start Summary

```bash
# 1. Transfer to remote
rsync -avz surfpinn/ REDACTED_SERVER:~/surfpinn/

# 2. SSH and setup
ssh REDACTED_SERVER
cd ~/surfpinn
/usr/local/bin/python3.11 -m venv venv && source venv/bin/activate
pip install "jax[cuda12]" dm-haiku optax h5py numpy scipy matplotlib pytest

# 3. Verify GPU
python -c "import jax; print(jax.devices())"

# 4. Run integration test
python tests/integration_test.py

# 5. Train
cd src && python train.py --epochs 100 2>&1 | tee training_log.txt

# 6. Sync back to local
# (from local machine)
rsync -avz REDACTED_SERVER:~/surfpinn/{checkpoints,results,*.txt,*.md} surfpinn/
```

**GO VALIDATE ON GPU! 🚀**
