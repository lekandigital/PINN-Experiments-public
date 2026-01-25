# PINN-Lite-Foil: Execute Full Pipeline on RTX 3090

## Context & Current Status

You have access to a complete PINN-Lite-Foil project scaffold at `/Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/`. The code exists but has **never been executed**. No trained models, no benchmark results, no ONNX files exist.

**Hardware Available:**
- Remote Machine: `REDACTED_SERVER` (SSH)
- GPU: NVIDIA GeForce RTX 3090 Ti (24GB VRAM)
- CUDA Version: 13.0
- Driver: 580.82.09
- OS: Ubuntu 22.04.5 LTS
- Sudo password: `REDACTED_PASSWORD`

**Mission: Actually execute the pipeline to produce trained models and real benchmark results.**

---

## Phase 1: Remote Environment Setup

### Task 1.1 - SSH and Initial Setup

```bash
# Connect to remote machine
ssh REDACTED_SERVER

# Check GPU is available
nvidia-smi

# Create project directory on remote
mkdir -p ~/pinn-lite-foil
cd ~/pinn-lite-foil

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.15.0
pip install numpy scipy h5py matplotlib
pip install onnx onnxruntime-gpu tf2onnx
pip install pytest

# Verify TensorFlow sees GPU
python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Task 1.2 - Sync Project Files to Remote

From local machine:
```bash
# Sync project to remote
rsync -avz --progress \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/ \
  REDACTED_SERVER:~/pinn-lite-foil/
```

---

## Phase 2: Generate Synthetic Training Data

Since OpenFOAM CFD simulations are complex to set up, we'll generate **synthetic analytical data** for the PINN to learn from. This is valid for demonstrating the pipeline - PINNs can learn from analytical solutions.

### Task 2.1 - Create Synthetic Dataset Generator

Create `/home/o/pinn-lite-foil/src/data_generation/synthetic_data.py`:

```python
#!/usr/bin/env python3
"""
Generate synthetic 2D airfoil flow data using potential flow theory.
This provides ground truth for PINN training without OpenFOAM.

For a thin symmetric airfoil at small angles, potential flow gives:
- Lift coefficient: Cl = 2 * pi * alpha (radians)
- Velocity field from conformal mapping (Joukowski transform)
"""

import numpy as np
import h5py
from pathlib import Path
import argparse

def joukowski_flow(x, y, alpha_deg, c=1.0, U_inf=1.0):
    """
    Compute potential flow around a Joukowski airfoil.

    Args:
        x, y: Coordinates (can be arrays)
        alpha_deg: Angle of attack in degrees
        c: Chord length
        U_inf: Freestream velocity

    Returns:
        u, v, p: Velocity components and pressure coefficient
    """
    alpha = np.deg2rad(alpha_deg)

    # Transform to complex plane
    z = x + 1j * y

    # Joukowski parameters (simplified for thin airfoil)
    a = c / 4  # Circle radius

    # Avoid division by zero at the origin
    z_safe = np.where(np.abs(z) < 1e-10, 1e-10, z)

    # Complex potential derivative (velocity)
    # For flow past circle with circulation
    Gamma = 4 * np.pi * a * U_inf * np.sin(alpha)  # Kutta condition

    # Complex velocity
    dW_dz = U_inf * np.exp(-1j * alpha) - 1j * Gamma / (2 * np.pi * z_safe)

    # Transform to physical plane (simplified)
    u = np.real(dW_dz)
    v = -np.imag(dW_dz)

    # Pressure coefficient from Bernoulli
    V_mag = np.sqrt(u**2 + v**2)
    Cp = 1 - (V_mag / U_inf)**2

    # Normalize pressure
    p = Cp * 0.5 * U_inf**2

    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32)


def generate_naca_points(code='0012', n_points=100):
    """Generate NACA 4-digit airfoil surface points."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0 if int(code[1]) > 0 else 0.5
    t = int(code[2:]) / 100.0

    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2

    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    # Camber
    yc = np.zeros_like(x)
    if m > 0:
        yc = np.where(x < p,
                      m/p**2 * (2*p*x - x**2),
                      m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

    x_upper = x
    y_upper = yc + yt
    x_lower = x
    y_lower = yc - yt

    return x_upper, y_upper, x_lower, y_lower


def generate_training_data(
    n_domain_points=50000,
    n_boundary_points=1000,
    n_airfoils=5,
    aoa_range=(-5, 15),
    output_path='data/processed/training_data.h5'
):
    """
    Generate complete training dataset.

    Creates:
    - Domain collocation points with PDE ground truth
    - Boundary points with BC ground truth
    - Multiple AoA values for parameterized learning
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # NACA codes to use
    naca_codes = ['0012', '2412', '4412', '0015', '2415'][:n_airfoils]
    aoa_values = np.linspace(aoa_range[0], aoa_range[1], 9)  # 9 angles

    all_data = []

    for naca in naca_codes:
        print(f"Generating data for NACA {naca}...")

        for aoa in aoa_values:
            # Domain points (exclude near-airfoil region)
            x_domain = np.random.uniform(-0.5, 2.0, n_domain_points)
            y_domain = np.random.uniform(-1.0, 1.0, n_domain_points)

            # Compute flow field
            u, v, p = joukowski_flow(x_domain, y_domain, aoa)

            # Store with AoA as input feature
            data = {
                'x': x_domain.astype(np.float32),
                'y': y_domain.astype(np.float32),
                'aoa': np.full(n_domain_points, aoa, dtype=np.float32),
                'u': u,
                'v': v,
                'p': p,
                'naca': naca
            }
            all_data.append(data)

    # Combine all data
    print(f"Saving {len(all_data)} cases to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        x_all = np.concatenate([d['x'] for d in all_data])
        y_all = np.concatenate([d['y'] for d in all_data])
        aoa_all = np.concatenate([d['aoa'] for d in all_data])
        u_all = np.concatenate([d['u'] for d in all_data])
        v_all = np.concatenate([d['v'] for d in all_data])
        p_all = np.concatenate([d['p'] for d in all_data])

        f.create_dataset('x', data=x_all, compression='gzip')
        f.create_dataset('y', data=y_all, compression='gzip')
        f.create_dataset('aoa', data=aoa_all, compression='gzip')
        f.create_dataset('u', data=u_all, compression='gzip')
        f.create_dataset('v', data=v_all, compression='gzip')
        f.create_dataset('p', data=p_all, compression='gzip')

        # Metadata
        f.attrs['n_samples'] = len(x_all)
        f.attrs['naca_codes'] = ','.join(naca_codes)
        f.attrs['aoa_range'] = f"{aoa_range[0]},{aoa_range[1]}"

    print(f"Generated {len(x_all):,} training samples")
    return output_path


def generate_test_data(output_path='data/processed/test_data.h5'):
    """Generate held-out test set with different AoA values."""
    return generate_training_data(
        n_domain_points=10000,
        n_airfoils=3,
        aoa_range=(0, 10),
        output_path=output_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/processed/training_data.h5')
    parser.add_argument('--n_points', type=int, default=50000)
    parser.add_argument('--n_airfoils', type=int, default=5)
    args = parser.parse_args()

    # Generate training data
    generate_training_data(
        n_domain_points=args.n_points,
        n_airfoils=args.n_airfoils,
        output_path=args.output
    )

    # Generate test data
    generate_test_data(output_path='data/processed/test_data.h5')

    print("Data generation complete!")
```

### Task 2.2 - Execute Data Generation

```bash
# On remote machine
cd ~/pinn-lite-foil
source venv/bin/activate

python src/data_generation/synthetic_data.py \
  --output data/processed/training_data.h5 \
  --n_points 100000 \
  --n_airfoils 5
```

---

## Phase 3: Train Baseline PINN (Teacher Model)

### Task 3.1 - Update Training Script for RTX 3090

The existing `baseline_pinn.py` should work, but ensure GPU memory is properly utilized:

```bash
# On remote machine
cd ~/pinn-lite-foil
source venv/bin/activate

# Set GPU memory growth to avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Train baseline PINN (teacher model)
python src/training/baseline_pinn.py \
  --data data/processed/training_data.h5 \
  --epochs 10000 \
  --batch_size 4096 \
  --hidden_layers 8 \
  --hidden_units 128 \
  --output models/teacher/ \
  --checkpoint_freq 1000

# Expected output:
# - models/teacher/baseline_pinn.h5
# - models/teacher/training_history.json
# - models/teacher/config.json
```

### Task 3.2 - Monitor Training

```bash
# In separate terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f models/teacher/training.log
```

**Expected Training Time:** ~30-60 minutes for 10k epochs on RTX 3090

---

## Phase 4: Knowledge Distillation (Student Model)

### Task 4.1 - Run Distillation

```bash
# On remote machine
cd ~/pinn-lite-foil
source venv/bin/activate

python src/training/distillation.py \
  --teacher models/teacher/baseline_pinn.h5 \
  --data data/processed/training_data.h5 \
  --epochs 5000 \
  --batch_size 4096 \
  --student_layers 4 \
  --student_units 32 \
  --temperature 2.0 \
  --output models/student/

# Expected output:
# - models/student/compressed_pinn.h5
# - models/student/distillation_history.json
```

### Task 4.2 - Verify Model Size

```bash
# Check model file sizes
ls -lh models/teacher/*.h5
ls -lh models/student/*.h5

# Student model should be < 2MB (target: ~50KB for 4x32 architecture)
```

---

## Phase 5: Convert to ONNX

### Task 5.1 - Export ONNX Model

```bash
cd ~/pinn-lite-foil
source venv/bin/activate

python src/deployment/onnx_converter.py \
  --input models/student/compressed_pinn.h5 \
  --output models/onnx/student_compressed.onnx \
  --opset 13

# Verify ONNX model
python -c "
import onnx
model = onnx.load('models/onnx/student_compressed.onnx')
onnx.checker.check_model(model)
print('ONNX model is valid!')
print(f'Model size: {len(model.SerializeToString()) / 1024:.1f} KB')
"
```

---

## Phase 6: Run Benchmarks

### Task 6.1 - Accuracy Benchmarks

```bash
cd ~/pinn-lite-foil
source venv/bin/activate

python src/benchmarking/accuracy_test.py \
  --model models/onnx/student_compressed.onnx \
  --test_data data/processed/test_data.h5 \
  --output results/accuracy_results.json

# Expected output:
# - results/accuracy_results.json
# - results/pressure_comparison.png
# - results/velocity_comparison.png
# - results/lift_drag_error.png
```

### Task 6.2 - Latency Benchmarks

```bash
python src/benchmarking/latency_benchmark.py \
  --model models/onnx/student_compressed.onnx \
  --iterations 10000 \
  --batch_sizes 1 4 8 16 \
  --output results/latency_results.json

# Expected output:
# - results/latency_results.json
# - results/latency_histogram.png
# - results/throughput_chart.png
```

### Task 6.3 - Generate Summary Report

```bash
python -c "
import json
import os

# Load results
with open('results/accuracy_results.json') as f:
    acc = json.load(f)
with open('results/latency_results.json') as f:
    lat = json.load(f)

print('='*60)
print('PINN-LITE-FOIL BENCHMARK RESULTS')
print('='*60)
print()
print('ACCURACY:')
print(f'  Lift MAE:  {acc.get(\"lift_mae\", \"N/A\"):.4f}')
print(f'  Drag MAE:  {acc.get(\"drag_mae\", \"N/A\"):.4f}')
print(f'  Pressure RMSE: {acc.get(\"pressure_rmse\", \"N/A\"):.4f}')
print()
print('LATENCY (batch_size=1):')
print(f'  Mean:  {lat.get(\"mean_ms\", \"N/A\"):.3f} ms')
print(f'  P99:   {lat.get(\"p99_ms\", \"N/A\"):.3f} ms')
print()
print('MODEL SIZE:')
model_size = os.path.getsize('models/onnx/student_compressed.onnx') / 1024
print(f'  ONNX: {model_size:.1f} KB')
print('='*60)
"
```

---

## Phase 7: Sync Results Back to Local

### Task 7.1 - Sync All Outputs to Local Machine

From local machine:
```bash
# Create results directories locally
mkdir -p /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/{models,results,data/processed}

# Sync trained models
rsync -avz REDACTED_SERVER:~/pinn-lite-foil/models/ \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/models/

# Sync benchmark results
rsync -avz REDACTED_SERVER:~/pinn-lite-foil/results/ \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/results/

# Sync generated data (optional - large files)
rsync -avz REDACTED_SERVER:~/pinn-lite-foil/data/processed/ \
  /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/data/processed/
```

---

## Phase 8: Validation Checklist

After execution, verify these deliverables exist:

### Required Files

| Path | Description | Target |
|------|-------------|--------|
| `models/teacher/baseline_pinn.h5` | Trained 8x128 teacher | ~1.5 MB |
| `models/student/compressed_pinn.h5` | Distilled 4x32 student | < 100 KB |
| `models/onnx/student_compressed.onnx` | Deployment model | < 100 KB |
| `results/accuracy_results.json` | MAE metrics | Lift/Drag < 5% |
| `results/latency_results.json` | Timing data | < 1ms batch=1 |
| `results/*.png` | Visualization plots | N/A |

### Success Criteria

- [ ] Teacher model training loss < 1e-3
- [ ] Student model accuracy within 5% of teacher
- [ ] ONNX model inference < 1ms on GPU
- [ ] Model size < 2MB (target < 100KB for student)
- [ ] Lift MAE < 5%
- [ ] Drag MAE < 5%
- [ ] All benchmark plots generated

---

## Execution Commands Summary

Run these in order on the remote machine:

```bash
# 1. Setup
ssh REDACTED_SERVER
mkdir -p ~/pinn-lite-foil && cd ~/pinn-lite-foil
python3 -m venv venv && source venv/bin/activate
pip install tensorflow[and-cuda] numpy scipy h5py matplotlib onnx onnxruntime-gpu tf2onnx

# 2. Sync code from local (run from local machine)
rsync -avz /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/ REDACTED_SERVER:~/pinn-lite-foil/

# 3. Generate data (on remote)
python src/data_generation/synthetic_data.py --n_points 100000

# 4. Train teacher (on remote)
python src/training/baseline_pinn.py --data data/processed/training_data.h5 --epochs 10000 --output models/teacher/

# 5. Distill student (on remote)
python src/training/distillation.py --teacher models/teacher/baseline_pinn.h5 --epochs 5000 --output models/student/

# 6. Convert to ONNX (on remote)
python src/deployment/onnx_converter.py --input models/student/compressed_pinn.h5 --output models/onnx/student_compressed.onnx

# 7. Benchmark (on remote)
python src/benchmarking/accuracy_test.py --model models/onnx/student_compressed.onnx --output results/
python src/benchmarking/latency_benchmark.py --model models/onnx/student_compressed.onnx --output results/

# 8. Sync back (run from local machine)
rsync -avz REDACTED_SERVER:~/pinn-lite-foil/{models,results}/ /Users/lekanadeyeri/Dev/PINN-Experiments/projects/15-pinn-lite-foil__project-space/pinn-lite-foil/
```

---

## Agent Instructions

**You are Claude Opus 4.5 running in VSCode Copilot agent mode. Execute this pipeline to:**

1. SSH to `REDACTED_SERVER` and set up the Python environment
2. Sync project files from local to remote
3. Create and run the synthetic data generator
4. Train the baseline PINN (teacher) on RTX 3090
5. Perform knowledge distillation to create student model
6. Convert student to ONNX format
7. Run accuracy and latency benchmarks
8. Sync all results back to local machine
9. Generate final validation report

**Important:**
- Use `sshpass` or key-based auth for SSH commands
- Monitor GPU memory with `nvidia-smi`
- Save all outputs to `results/` directory
- Create synthetic data since OpenFOAM is not installed
- Target < 1ms inference time on GPU

**Begin execution now with Phase 1 (SSH and environment setup).**
