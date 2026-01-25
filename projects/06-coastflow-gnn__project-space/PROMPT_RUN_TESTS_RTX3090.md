# CoastFlow-GNN: RTX 3090 Training, Benchmarking & Enhancement

## Project Status: VALIDATED
- All 47 tests pass locally (1 GPU test skipped - no local CUDA)
- Model architecture: Hierarchical GCN with TopKPooling
- Physics losses: Continuity, Momentum, k-epsilon turbulence, Boundary conditions
- Ready for GPU training

---

## Environment

### Local Machine (Mac)
- **Project path**: `/Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/`
- **Monorepo root**: `/Users/lekanadeyeri/Dev/PINN-Experiments/`

### Remote Machine (Ubuntu + RTX 3090)
- **SSH**: `ssh REDACTED_SERVER`
- **Password**: `REDACTED_PASSWORD`
- **GPU**: NVIDIA RTX 3090 Ti (24GB VRAM), CUDA 13.0, Driver 580.82
- **Python**: Use `/usr/bin/python3` (Python 3.9.6)
- **Working PyTorch**: 2.2.2 with PyG 2.6.1, NumPy 1.26.4

---

## Mission

Train the validated CoastFlow-GNN on RTX 3090, collect performance benchmarks, generate visualizations, and copy all results back to local machine.

---

## Phase 1: Deploy to Remote & GPU Training

### 1.1 Copy Project to Remote Machine
```bash
# From LOCAL Mac terminal
scp -r /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn REDACTED_SERVER:~/coastflow-gnn
```

### 1.2 SSH into Remote and Verify GPU
```bash
ssh REDACTED_SERVER
# Password: REDACTED_PASSWORD

# Verify GPU
nvidia-smi

# Verify PyTorch + CUDA
/usr/bin/python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 1.3 Install Any Missing Dependencies
```bash
cd ~/coastflow-gnn

# Install core dependencies (use /usr/bin/python3 to match the working torch install)
/usr/bin/python3 -m pip install torch-geometric --quiet 2>/dev/null || echo "PyG may already be installed"
/usr/bin/python3 -m pip install matplotlib seaborn pyyaml tqdm --quiet
```

### 1.4 Run Tests on GPU Machine
```bash
cd ~/coastflow-gnn
/usr/bin/python3 -m pytest tests/ -v --tb=short
```

### 1.5 Run Full Training on RTX 3090
```bash
cd ~/coastflow-gnn

# Create output directory
mkdir -p outputs_rtx3090

# Full training run
/usr/bin/python3 src/training/train_single.py \
    --epochs 100 \
    --batch-size 8 \
    --hidden 128 \
    --num-samples 500 \
    --num-nodes 500 \
    --lr 5e-4 \
    --output-dir ./outputs_rtx3090

# Monitor GPU in separate terminal:
# watch -n 1 nvidia-smi
```

### 1.6 Run Evaluation
```bash
/usr/bin/python3 src/training/eval_model.py \
    --checkpoint ./outputs_rtx3090/best_model.pth \
    --output-dir ./outputs_rtx3090
```

---

## Phase 2: Performance Benchmarking

### 2.1 Create GPU Benchmark Script
Create file `src/training/benchmark_gpu.py` on the REMOTE machine:

```python
"""
GPU Benchmark Script for CoastFlow-GNN
Measures inference speed, training throughput, and memory usage on RTX 3090.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.coastflow_gnn import CoastFlowGNN
from src.data.dataset import SyntheticCoastalDataset
from torch_geometric.loader import DataLoader


def benchmark_inference(model, loader, device, num_warmup=10, num_runs=100):
    """Benchmark inference speed."""
    model.eval()

    # Get a batch
    batch = next(iter(loader)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batch.x, batch.edge_index, batch.batch)

    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(batch.x, batch.edge_index, batch.batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'samples_per_batch': len(batch.x),
    }


def benchmark_training_throughput(model, loader, device, num_batches=50):
    """Benchmark training throughput."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = out.mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    total_samples = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        total_samples += batch.num_graphs

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        'samples_per_second': total_samples / elapsed,
        'batches_per_second': num_batches / elapsed,
        'total_samples': total_samples,
        'elapsed_seconds': elapsed,
    }


def benchmark_memory(model, loader, device):
    """Benchmark GPU memory usage."""
    if device.type != 'cuda':
        return {'peak_memory_mb': 0, 'allocated_mb': 0}

    torch.cuda.reset_peak_memory_stats()

    model.train()
    batch = next(iter(loader)).to(device)

    # Forward + backward
    out = model(batch.x, batch.edge_index, batch.batch)
    loss = out.mean()
    loss.backward()

    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    return {
        'peak_memory_mb': peak_memory,
        'allocated_mb': allocated,
    }


def run_benchmarks(output_dir: str = './benchmarks'):
    """Run all benchmarks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
        'benchmarks': {}
    }

    # Test different configurations
    configs = [
        {'hidden': 64, 'batch_size': 4, 'num_nodes': 200},
        {'hidden': 128, 'batch_size': 8, 'num_nodes': 500},
        {'hidden': 256, 'batch_size': 4, 'num_nodes': 1000},
    ]

    for config in configs:
        config_name = f"h{config['hidden']}_b{config['batch_size']}_n{config['num_nodes']}"
        print(f"\nBenchmarking config: {config_name}")

        # Create model
        model = CoastFlowGNN(
            in_channels=6,
            hidden_channels=config['hidden'],
            out_channels=4,
        ).to(device)

        # Create dataset
        dataset = SyntheticCoastalDataset(
            num_samples=100,
            num_nodes=config['num_nodes'],
            seed=42,
        )
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # Run benchmarks
        inference_results = benchmark_inference(model, loader, device)
        throughput_results = benchmark_training_throughput(model, loader, device)
        memory_results = benchmark_memory(model, loader, device)

        results['benchmarks'][config_name] = {
            'config': config,
            'inference': inference_results,
            'throughput': throughput_results,
            'memory': memory_results,
            'model_params': model.count_parameters(),
        }

        print(f"  Inference: {inference_results['mean_ms']:.2f} ms/batch")
        print(f"  Throughput: {throughput_results['samples_per_second']:.1f} samples/sec")
        print(f"  Peak Memory: {memory_results['peak_memory_mb']:.1f} MB")

    # Save results
    results_path = output_dir / 'gpu_benchmark.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./benchmarks')
    args = parser.parse_args()

    run_benchmarks(args.output_dir)
```

### 2.2 Run Benchmarks
```bash
cd ~/coastflow-gnn
/usr/bin/python3 src/training/benchmark_gpu.py --output-dir ./benchmarks
```

---

## Phase 3: Copy Results Back to Local Machine

### 3.1 From LOCAL Mac Terminal
```bash
# Create local results directory
mkdir -p /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/outputs_rtx3090
mkdir -p /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/benchmarks

# Copy training outputs
scp -r REDACTED_SERVER:~/coastflow-gnn/outputs_rtx3090/* \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/outputs_rtx3090/

# Copy benchmarks
scp -r REDACTED_SERVER:~/coastflow-gnn/benchmarks/* \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/benchmarks/

# Copy any new/modified source files
scp REDACTED_SERVER:~/coastflow-gnn/src/training/benchmark_gpu.py \
    /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/src/training/
```

---

## Phase 4: Generate Visualizations (Local)

### 4.1 Plot Training Curves
```bash
cd /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn

python3 -c "
import json
import matplotlib.pyplot as plt

with open('outputs_rtx3090/training_history.json') as f:
    history = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0,0].plot(history['train_loss'], label='Train')
axes[0,0].plot(history['val_loss'], label='Val')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training & Validation Loss')
axes[0,0].legend()
axes[0,0].set_yscale('log')

# Learning rate
axes[0,1].plot(history['lr'])
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Learning Rate')
axes[0,1].set_title('Learning Rate Schedule')

# Physics residuals
axes[1,0].plot(history['physics_residuals']['continuity'], label='Continuity')
axes[1,0].plot(history['physics_residuals']['momentum'], label='Momentum')
axes[1,0].plot(history['physics_residuals']['turbulence'], label='Turbulence')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Residual')
axes[1,0].set_title('Physics Residuals')
axes[1,0].legend()
axes[1,0].set_yscale('log')

# Epoch time
axes[1,1].plot(history['epoch_time'])
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Time (s)')
axes[1,1].set_title('Epoch Duration')

plt.tight_layout()
plt.savefig('outputs_rtx3090/training_curves.png', dpi=150)
print('Saved: outputs_rtx3090/training_curves.png')
"
```

---

## Phase 5: Model Enhancements (Optional)

### 5.1 Add Graph Attention Variant
Create `src/models/coastflow_gat.py` with GATConv instead of GCNConv:
- 4 attention heads per layer
- Concatenate heads in hidden layers, average in output
- Same TopKPooling structure

### 5.2 Adaptive Loss Weighting
Implement curriculum learning for physics loss weights:
- Start: high data weight, low physics weights
- Gradually increase physics weights over training
- Use gradient magnitude balancing (GradNorm)

### 5.3 Real DEM Data Integration
Enhance `src/data/dem_fetcher.py` to:
- Fetch real NOAA BlueTopo DEMs
- Parse GeoTIFF format
- Generate mesh from coastline geometry

---

## Expected Deliverables

After completing all phases, you should have:

### On Local Machine
```
/Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/
├── outputs_rtx3090/
│   ├── best_model.pth              # Trained model checkpoint
│   ├── latest_checkpoint.pth       # Latest checkpoint
│   ├── training_history.json       # Full training log
│   ├── training_curves.png         # Loss/LR/residual plots
│   └── evaluation_results.json     # Test set metrics
├── benchmarks/
│   └── gpu_benchmark.json          # Performance metrics
└── src/training/
    └── benchmark_gpu.py            # New benchmark script
```

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Training completes | 100 epochs | No OOM, no NaN |
| GPU utilization | > 70% | Check with nvidia-smi |
| Final val loss | < 0.5 | Synthetic data target |
| Inference speed | < 100ms/batch | RTX 3090 should exceed |
| Peak memory | < 20GB | Out of 24GB available |
| Physics residuals | Decreasing | All three components |

---

## Quick Command Reference

```bash
# === LOCAL MACHINE ===
# Copy to remote
scp -r /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn REDACTED_SERVER:~/

# Copy results back
scp -r REDACTED_SERVER:~/coastflow-gnn/outputs_rtx3090 /Users/lekanadeyeri/Dev/PINN-Experiments/projects/06-coastflow-gnn__project-space/coastflow-gnn/

# === REMOTE MACHINE ===
# SSH in
ssh REDACTED_SERVER

# Check GPU
nvidia-smi

# Run tests
cd ~/coastflow-gnn && /usr/bin/python3 -m pytest tests/ -v

# Train
/usr/bin/python3 src/training/train_single.py --epochs 100 --hidden 128 --batch-size 8 --output-dir ./outputs_rtx3090

# Benchmark
/usr/bin/python3 src/training/benchmark_gpu.py --output-dir ./benchmarks
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` to 4 or 2
- Reduce `--hidden` to 64
- Reduce `--num-nodes` to 200

### Import Errors
- Always use `/usr/bin/python3` (not python3 or python)
- PyTorch is installed in `/Users/o/Library/Python/3.9/`

### NumPy Warnings
- Ignore the "Failed to initialize NumPy" warning - it still works
- Already using numpy<2 (1.26.4)

### SSH Connection Issues
- Verify: `ping 192.168.86.152`
- Check local network connection
- Try: `ssh -v REDACTED_SERVER` for debug output
