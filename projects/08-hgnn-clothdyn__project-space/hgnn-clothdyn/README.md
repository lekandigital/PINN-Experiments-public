# HGNN-ClothDyn: Hierarchical Graph Neural Network for Cloth Dynamics

A PyTorch implementation of hierarchical graph neural networks for learning cloth simulation dynamics with physics-informed learned edge stiffness.

## Features

- **Multi-Resolution Graph Pyramid**: Graclus clustering for coarse-to-fine message passing
- **Learned Edge Stiffness**: Physics-informed spring force computation (Hookean constraints)
- **Hierarchical Message Passing**: Fine → Coarse → Fine architecture for local + global interactions
- **Scheduled Sampling**: Teacher forcing with gradual transition to autoregressive prediction
- **Mixed Precision Training**: AMP for efficient GPU utilization

## Architecture

```
Input (pos, vel) → MLP → Fine Conv → Pool → Coarse Conv → Upsample → Fine Conv → Output (Δvel)
```

The `EdgeForceConv` layer computes spring forces:
```
F_spring = -k_learned * (||x_i - x_j|| - L_rest) * (x_i - x_j) / ||x_i - x_j||
```

## Installation

### On VastAI Instance

```bash
# SSH into your instance
ssh -p <PORT> root@<HOST>

# Clone or upload the code
cd /workspace
# Upload hgnn_clothdyn folder

# Run setup
bash setup.sh
```

### Local Development

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data

```bash
python synthetic_data.py --output data/synthetic_cloth.h5 --frames 100 --mesh-size 20
```

### 2. Run Integration Test

```bash
python test_integration.py --quick-test --epochs 5 --batch-size 4
```

### 3. Train Model

```bash
python train.py \
    --data data/synthetic_cloth.h5 \
    --epochs 50 \
    --hidden-dim 128 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints/
```

### 4. Run Benchmarks

```bash
python benchmark.py \
    --checkpoint checkpoints/checkpoint_epoch50.pt \
    --test-data data/synthetic_cloth.h5 \
    --output-dir results/
```

## Project Structure

```
hgnn_clothdyn/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── setup.sh               # VastAI environment setup
├── mesh_to_graph.py       # Mesh → Graph conversion + pyramid
├── model.py               # HGNNClothDyn model architecture
├── synthetic_data.py      # Synthetic cloth data generation
├── train.py               # Training loop with scheduled sampling
├── benchmark.py           # Performance benchmarking
├── test_integration.py    # Integration tests
└── config.yaml            # Default configuration
```

## Configuration

Key hyperparameters (adjustable via command line or config.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Hidden layer dimension |
| `num_levels` | 2 | Graph pyramid levels |
| `num_message_passes` | 3 | Message passing iterations per level |
| `learning_rate` | 1e-3 | Adam optimizer LR |
| `scheduled_sampling_start` | 5 | Epoch to begin scheduled sampling |
| `sampling_increment` | 0.05 | Per-epoch probability increase |

## Hardware Requirements

- **Recommended**: NVIDIA L40S/A100 (48GB VRAM)
- **Minimum**: RTX 3090/A6000 (24GB VRAM) - reduce `hidden_dim` to 64

## Benchmark Metrics

Expected performance on L40S with 20x20 mesh:

| Metric | Target | Description |
|--------|--------|-------------|
| Position RMSE | < 0.1 | 50-frame rollout stability |
| FPS | > 50 | Inference speed |
| Edge Length Error | < 0.05 | Constraint preservation |
| GPU Memory | < 40GB | Peak usage |

## License

MIT License

## References

- MeshGraphNets (Pfaff et al., 2020)
- Neural Cloth Simulation (Grigorev et al., 2022)
- Learning Mesh-Based Simulation with Graph Networks (DeepMind)
