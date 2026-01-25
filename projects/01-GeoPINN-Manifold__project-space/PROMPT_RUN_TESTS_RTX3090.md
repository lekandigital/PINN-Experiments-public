# MISSION: Run GeoPINN-Manifold Full Test Suite on Local RTX 3090

You are an expert ML/scientific computing engineer. Your task is to deploy and run the complete GeoPINN-Manifold test suite on a local Ubuntu machine with an NVIDIA RTX 3090 Ti GPU.

## TARGET MACHINE

```
Host: 192.168.86.152
User: o
SSH: ssh REDACTED_SERVER
Sudo password: REDACTED_PASSWORD
GPU: NVIDIA GeForce RTX 3090 Ti (24GB VRAM)
CUDA: 13.0
Driver: 580.82.09
OS: Ubuntu 22.04.5 LTS
```

## PHASE 1: TRANSFER PROJECT TO REMOTE MACHINE

From the local Mac, transfer the GeoPINN-Manifold project:

```bash
# Create tarball of the project
cd /Users/lekanadeyeri/Dev/PINN-Experiments/projects/01-GeoPINN-Manifold__project-space
tar -czf /tmp/geopinn-manifold.tar.gz GeoPINN-Manifold/

# Transfer to remote machine
scp /tmp/geopinn-manifold.tar.gz REDACTED_SERVER:~/

# SSH into remote machine
ssh REDACTED_SERVER
```

## PHASE 2: ENVIRONMENT SETUP ON REMOTE

Once on the remote machine:

```bash
# Extract project
cd ~
tar -xzf geopinn-manifold.tar.gz
cd GeoPINN-Manifold

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (for CUDA 12.x / 13.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy scipy scikit-learn h5py matplotlib pyvista pytest

# Verify GPU is accessible
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090 Ti
```

## PHASE 3: RUN INTEGRATION TESTS

```bash
cd ~/GeoPINN-Manifold

# Run all integration tests with verbose output
python -m pytest tests/test_integration.py -v

# Expected: 10/10 tests should pass
```

**Test breakdown:**
1. `test_tangent_message_passing` - Validates TangentMP layer
2. `test_spectral_conv` - Validates spectral graph convolution
3. `test_dec_operators` - Validates discrete exterior calculus
4. `test_atlas_stitching` - Validates chart atlas construction
5. `test_sphere_data_generation` - Validates SWE data generator
6. `test_shell_elasticity_data` - Validates shell mesh generator
7. `test_torus_data_generation` - Validates torus R-D generator
8. `test_laplace_beltrami_computation` - Validates Δ_S operator
9. `test_adaptive_refinement` - Validates mesh refinement
10. `test_full_training_loop` - Validates end-to-end training

## PHASE 4: RUN SPHERE PDE BENCHMARK

This is the main validation test - training a PINN to solve the Laplace-Beltrami equation on the unit sphere.

```bash
cd ~/GeoPINN-Manifold

# Run with 100 epochs (quick validation)
python tests/test_sphere_pde.py --epochs 100

# Run with 500 epochs (full benchmark, ~5 minutes)
python tests/test_sphere_pde.py --epochs 500

# Run with AMP (mixed precision) for faster training
python tests/test_sphere_pde.py --epochs 500 --amp
```

**Success criteria:**
- Laplace-Beltrami verification: Max error < 1e-5
- Training loss decreases monotonically
- Final L2 error < 0.1 (for 100 epochs)
- Final L2 error < 0.01 (for 500 epochs)
- Checkpoint saved to `tests/test_sphere_model.pth`

## PHASE 5: RUN EXTENDED BENCHMARKS (OPTIONAL)

For comprehensive validation:

```bash
# Generate synthetic datasets
python -c "
from geopinn.data.sphere_swe import generate_sphere_swe_data
from geopinn.data.shell_elasticity import generate_shell_mesh
from geopinn.data.torus_reaction_diffusion import generate_torus_rd_data

# Generate sphere shallow-water data
generate_sphere_swe_data(n_points=5000, output_path='data/sphere_swe.h5')

# Generate shell elasticity mesh
generate_shell_mesh(n_points=2000, output_path='data/shell_mesh.h5')

# Generate torus reaction-diffusion data
generate_torus_rd_data(n_points=3000, n_timesteps=100, output_path='data/torus_rd.h5')

print('All datasets generated successfully!')
"

# Run memory profiling
python -c "
import torch
from geopinn.training.sphere_trainer import SpherePINNTrainer

trainer = SpherePINNTrainer(
    hidden_dim=128,
    n_layers=4,
    n_collocation=5000,
    device='cuda'
)

# Check GPU memory usage
print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')
"
```

## PHASE 6: SAVE RESULTS AND CLEANUP

```bash
# Create results directory
mkdir -p ~/GeoPINN-Manifold/results

# Move checkpoint to results
mv tests/test_sphere_model.pth results/ 2>/dev/null || echo "No checkpoint to move"

# Create results summary
python -c "
import json
import torch
from datetime import datetime

results = {
    'timestamp': datetime.now().isoformat(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    'cuda_version': torch.version.cuda,
    'pytorch_version': torch.__version__,
    'tests_passed': True,
    'notes': 'Full test suite completed on local RTX 3090 Ti'
}

with open('results/test_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
"

# Create tarball of results for backup
tar -czf ~/geopinn_results_$(date +%Y%m%d_%H%M%S).tar.gz results/

# List results
ls -la results/
```

## PHASE 7: TRANSFER RESULTS BACK TO MAC (FROM MAC TERMINAL)

```bash
# From Mac terminal, pull results back
scp -r REDACTED_SERVER:~/GeoPINN-Manifold/results/ \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/01-GeoPINN-Manifold__project-space/GeoPINN-Manifold/

# Also grab the tarball backup
scp REDACTED_SERVER:~/geopinn_results_*.tar.gz \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/01-GeoPINN-Manifold__project-space/
```

## SUCCESS CRITERIA CHECKLIST

- [ ] SSH connection to 192.168.86.152 successful
- [ ] PyTorch detects CUDA and RTX 3090 Ti
- [ ] All 10 integration tests pass
- [ ] Laplace-Beltrami verification passes (max error < 1e-5)
- [ ] Sphere PDE training completes with L2 error < 0.1
- [ ] Checkpoint saved to results/test_sphere_model.pth
- [ ] Results transferred back to Mac

## TROUBLESHOOTING

### CUDA not found
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit if needed
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

### PyTorch CUDA mismatch
```bash
# Uninstall and reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory
```bash
# Reduce batch size / collocation points
python tests/test_sphere_pde.py --epochs 100 --n_collocation 500
```

### Permission denied
```bash
# Use sudo with the password: REDACTED_PASSWORD
sudo <command>
```

## DELIVERABLES

1. **Console output** showing all tests passing
2. **results/test_sphere_model.pth** - trained model checkpoint
3. **results/test_summary.json** - test metadata
4. **GPU utilization screenshot** from `nvidia-smi` during training
5. transfer everything back

---

**START EXECUTION NOW.** Provide step-by-step progress updates with outputs.
