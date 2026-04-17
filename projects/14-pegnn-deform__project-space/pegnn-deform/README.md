# PEGNN-Deform

**Physics-Encoded Graph Neural Network for Deformable Bodies**

A real-time physics-encoded GNN designed to replace finite-element solvers for soft-body and bio-mechanics animation.

## Update (April 17, 2026)

- Body-SDF export and physics-model integration work were added or refreshed in the current project tree.
- Project 14 now sits closer to the shared addon, collision, and export-pipeline infrastructure.
- This README update keeps the deform workflow aligned with the repo's current shared-tooling direction.

## Features

- **Physics-Encoded Message Passing**: Hard-coded Hooke's law in the network architecture guarantees physical consistency
- **GRU Temporal Roll-out**: Stable multi-step simulation with learned dynamics
- **Mixed-Precision Training**: Optimized for 24-48GB GPUs
- **Energy Conservation Loss**: Prevents drift in long-horizon simulations

## Project Structure

```
pegnn-deform/
├── src/
│   ├── mesh_features.py      # Geometric feature computation
│   ├── pegdeform_model.py    # Core GNN model
│   ├── train_pegdeform.py    # Training script
│   └── benchmark.py          # Inference benchmarking
├── data/
│   └── generate_synthetic.py # Synthetic data generator
├── tests/
│   └── test_model.py         # Unit tests
├── configs/
│   ├── environment.yaml      # Conda environment
│   └── wandb_sweep.yaml      # Hyperparameter sweep
└── scripts/
    ├── setup_vastai.sh       # vast.ai instance setup
    ├── run_tests.sh          # Test runner
    └── deploy_vastai.sh      # Automated deployment
```

## Quick Start

### 1. Local Setup

```bash
# Create conda environment
conda env create -f configs/environment.yaml
conda activate pegndeform

# Or install with pip
pip install torch torch-geometric wandb numpy scipy matplotlib tqdm pytest
```

### 2. Generate Synthetic Data

```bash
python data/generate_synthetic.py \
    --output-dir data/synthetic \
    --grid-size 20 \
    --num-train 100 \
    --num-val 20 \
    --num-test 20
```

### 3. Train Model

```bash
python src/train_pegdeform.py \
    --data-dir data \
    --epochs 100 \
    --batch-size 4 \
    --hidden-size 64
```

### 4. Benchmark

```bash
python src/benchmark.py \
    --model checkpoints/best_model.pth \
    --num-samples 100
```

### 5. Run Tests

```bash
pytest tests/test_model.py -v
```

## RTX 3090 Deployment (Home Server)

Deploy to a local RTX 3090 Ti (24GB VRAM) server.

### Quick Deployment

```bash
# Deploy and run tests
bash scripts/deploy_3090.sh

# Deploy and run training
bash scripts/deploy_3090.sh --train

# Skip sync (if files already transferred)
bash scripts/deploy_3090.sh --skip-sync --skip-env --train
```

### Manual Deployment

```bash
# 1. SSH into remote
ssh REDACTED_SERVER

# 2. Create project directory
mkdir -p ~/pegnn-deform

# 3. Exit and sync files (from local machine)
rsync -avz --exclude='__pycache__' ./ REDACTED_SERVER:~/pegnn-deform/

# 4. SSH back and run
ssh REDACTED_SERVER
cd ~/pegnn-deform
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install torch_geometric numpy scipy matplotlib tqdm pytest wandb

# 5. Run tests
pytest tests/test_model.py -v

# 6. Train
python3 src/train_pegdeform.py --epochs 50 --batch-size 8
```

### RTX 3090 Optimized Settings

| Mesh Size (nodes) | Batch Size | Memory Usage |
|-------------------|------------|--------------|
| 400               | 8          | ~4 GB        |
| 1000              | 4          | ~8 GB        |
| 2500              | 2          | ~16 GB       |
| 5000              | 1          | ~20 GB       |

Enable gradient checkpointing for larger meshes:
```python
model = PEGNNDeform(hidden_size=64, use_checkpointing=True)
```

## vast.ai Deployment

### Manual Deployment

```bash
# 1. Set API key
export VASTAI_API_KEY="your_api_key"

# 2. Search for L40S instances
vastai search offers "gpu_name=L40S rentable=true verified=true" --order dph_total --limit 10

# 3. Create instance
vastai create instance <INSTANCE_ID> \
    --image pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
    --disk 50 --ssh

# 4. Get connection details
vastai show instances

# 5. Transfer files
scp -P <PORT> -r pegnn-deform/ root@<HOST>:/workspace/

# 6. Connect and run
ssh -p <PORT> root@<HOST>
cd /workspace/pegnn-deform
bash scripts/setup_vastai.sh
bash scripts/run_tests.sh
```

### Automated Deployment

```bash
export VASTAI_API_KEY="your_api_key"
bash scripts/deploy_vastai.sh
```

## Architecture

### SpringMessagePassing

Implements Hooke's law directly in the message function:

```python
# Force = k * (distance - rest_length) * direction
force_magnitude = k * (dist - L0)
force = force_magnitude * direction
```

### Loss Functions

1. **Position MSE**: `||pos_pred - pos_gt||²`
2. **Velocity MSE**: `||vel_pred - vel_gt||²`
3. **Energy Conservation**: `|E_pred - E_gt|²`

Where total energy `E = 0.5 * k * (d - L0)² + 0.5 * m * v²`

## Performance Targets

| Metric | Target | RTX 3090 | L40S |
|--------|--------|----------|------|
| Latency (5000 nodes) | < 100ms | ~50ms | ~35ms |
| NRMSE | < 5% | 2-3% | 2-3% |
| Peak Memory | < GPU VRAM | ~20GB | ~25GB |

### Benchmark Results

Run benchmarks with:
```bash
python src/benchmark.py --scaling-test
```

## Visualization

```python
from src.visualize import plot_trajectory, plot_loss_curves

# Plot mesh trajectory
plot_trajectory(pos_trajectory, edges=edge_index, save_path="trajectory.png")

# Plot training curves
plot_loss_curves(train_losses, val_losses, save_path="loss_curves.png")
```

## Loading Real Meshes

```python
from data.generate_synthetic import load_mesh_from_file

# Load OBJ/STL mesh
pos, edge_index, faces = load_mesh_from_file("model.obj", scale=1.0)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 2`
- Enable gradient checkpointing: `PEGNNDeform(use_checkpointing=True)`
- Reduce hidden size: `--hidden-size 32`

### Training is Slow
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is detected: `python -c "import torch; print(torch.cuda.is_available())"`
- Use mixed precision (enabled by default)

### Test Failures
- Ensure PyTorch Geometric is installed: `pip install torch_geometric`
- Check CUDA version compatibility with PyTorch

### SSH Connection Issues
```bash
# Test connectivity
ping 192.168.86.152

# Enable SSH on remote (if needed)
sudo systemctl enable ssh
sudo systemctl start ssh
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- CUDA 11.8+ (for GPU training)

## Citation

If you use PEGNN-Deform, please cite:

```bibtex
@article{pegnn-deform,
  title={Physics-Encoded Graph Neural Networks for Deformable Bodies},
  year={2025}
}
```

## License

MIT License
