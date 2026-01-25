# SurfPINN Vast.ai Deployment Guide

## Quick Start

### 1. Search for L40S Instance

```bash
# Search for L40S GPUs (48GB VRAM)
vastai search offers "gpu_name=L40S rentable=true verified=true" \
  --order dph_total --limit 10

# Alternative: L40 (without S)
vastai search offers "gpu_name=L40 rentable=true verified=true" \
  --order dph_total --limit 10

# Fallback: Any GPU with 40GB+ VRAM
vastai search offers "gpu_ram>=40 rentable=true verified=true" \
  --order dph_total --limit 10

# Budget option: RTX 4090 (24GB)
vastai search offers "gpu_name=RTX%204090 rentable=true verified=true" \
  --order dph_total --limit 10
```

### 2. Create Instance

```bash
# Replace OFFER_ID with ID from search results
vastai create instance OFFER_ID \
  --image nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 \
  --disk 50 \
  --args "-p 8888:8888 -p 6006:6006"

# Check instance status
vastai show instances
```

### 3. SSH Connection

```bash
# Get SSH URL
vastai ssh-url INSTANCE_ID

# Connect with port forwarding (Jupyter + TensorBoard)
ssh -p PORT root@HOST \
  -L 8888:localhost:8888 \
  -L 6006:localhost:6006
```

## Environment Setup

### Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y \
  python3.10 python3-pip git wget \
  libhdf5-dev build-essential

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install JAX with CUDA 12
pip install --upgrade "jax[cuda12_pip]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install ML dependencies
pip install dm-haiku optax chex

# Install data/viz dependencies
pip install h5py numpy scipy matplotlib pytest tqdm
```

### Verify GPU

```bash
python3 -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
d = jax.devices()[0]
print(f'Device: {d}')
"
```

Expected output:
```
JAX version: 0.4.xx
Devices: [cuda(id=0)]
Device: cuda(id=0)
```

## Project Setup

### Clone/Upload Code

```bash
mkdir -p /workspace/surfpinn
cd /workspace/surfpinn

# Option A: Clone from git
# git clone YOUR_REPO_URL .

# Option B: Use scp from local machine
# scp -P PORT -r ./surfpinn/* root@HOST:/workspace/surfpinn/
```

### Create Directories

```bash
mkdir -p /workspace/surfpinn/{src,data,tests,checkpoints,notebooks}
```

## Running Tests

### Integration Test

```bash
cd /workspace/surfpinn
python tests/integration_test.py
```

Expected output:
```
==============================================================
SurfPINN Integration Test
==============================================================

[1/6] Checking environment...
  JAX version: 0.4.xx
  Devices: [cuda(id=0)]
  ✓ GPU detected

[2/6] Generating synthetic data...
  Eulerian: height (32, 32, 8, 1), velocity (32, 32, 8, 2)
  Lagrangian: positions (200, 8, 3)
  ✓ Data generation successful

[3/6] Initializing model...
  Parameters: xxx,xxx
  ✓ Model initialized

[4/6] Testing forward pass...
  Height output: (1, 32, 32, 1)
  ✓ Forward pass successful

[5/6] Testing training loop (10 iterations)...
  Step 1: loss=x.xxxxx, time=x.xxxs
  Step 10: loss=x.xxxxx, time=x.xxxs
  Loss reduction: xx.x%
  ✓ Training loop successful

[6/6] Validating final predictions...
  Height PSNR: xx.xx dB
  ✓ Final validation passed

==============================================================
STATUS: ✓ PASSED
==============================================================
```

### Unit Tests

```bash
cd /workspace/surfpinn
pytest tests/test_model.py -v
```

## Training

### Quick Test (10 epochs)

```bash
cd /workspace/surfpinn/src
python train.py --test-mode
```

### Full Training

```bash
python train.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --checkpoint-dir ../checkpoints
```

### Monitor GPU Usage

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

### JAX Not Detecting GPU

```bash
# Check CUDA
nvidia-smi
nvcc --version

# Reinstall JAX
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12_pip]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Out of Memory

```python
# In train.py, reduce batch size
config = TrainConfig(batch_size=4)  # or 2

# Enable gradient checkpointing
import jax
jax.config.update("jax_enable_memories", True)
```

### NaN Losses

```python
# Reduce learning rate
config = TrainConfig(learning_rate=1e-5)

# Increase gradient clipping
config = TrainConfig(gradient_clip=0.5)
```

### Slow Training

```bash
# Enable XLA optimizations
export XLA_FLAGS="--xla_gpu_autotune_level=2"
python train.py
```

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Training step time | < 0.5s | After JIT compilation |
| Memory usage | < 40GB | For L40S (48GB) |
| Loss reduction | > 50% | After 100 epochs |
| PSNR | > 25 dB | On synthetic data |

## Cleanup

```bash
# Stop instance when done
vastai stop instance INSTANCE_ID

# Or destroy completely
vastai destroy instance INSTANCE_ID
```

## Cost Estimation

| GPU | $/hr (approx) | 100 epochs time | Total cost |
|-----|---------------|-----------------|------------|
| L40S | $0.50-1.00 | ~1-2 hours | $1-2 |
| RTX 4090 | $0.30-0.50 | ~2-3 hours | $1-1.50 |
| A100 | $1.50-2.50 | ~0.5-1 hour | $1-2.50 |
