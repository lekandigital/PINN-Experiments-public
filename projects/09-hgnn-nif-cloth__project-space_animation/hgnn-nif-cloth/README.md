# HGNN-NIF-Cloth

**Hierarchical Graph Neural Network + Neural Implicit Field for Cloth Simulation**

A hybrid framework that fuses hierarchical graph neural networks (HGNNs) for internal lattice dynamics with a latent-conditioned neural implicit surface decoder (SIREN) for physically accurate cloth simulation.

## Update (April 17, 2026)

- This project-space is now the fuller demo-oriented variant, with rollout metrics, outputs, demo-site assets, and Taichi media.
- Training, physics-data generation, and rollout/metrics scripts were added around the hybrid model workflow.
- Use this variant when you need the end-to-end animation and demo path rather than the lighter base tree.

## 🎯 Overview

This project implements a novel hybrid approach to cloth simulation:

1. **Adaptive HGNN**: Processes multi-resolution cloth graphs with energy-based resolution control
2. **Cross-Level Attention**: Enables information flow between fine and coarse graph representations
3. **SIREN Decoder**: Reconstructs continuous implicit surfaces from learned latent codes
4. **Physics Losses**: Combines edge spring constraints with SDF reconstruction for physical accuracy

## 📦 Project Structure

```
hgnn-nif-cloth/
├── src/
│   ├── models/
│   │   ├── graph_conv.py      # Vectorized graph convolution layer
│   │   ├── attention.py       # Cross-level attention mechanism
│   │   ├── hgnn.py           # Adaptive Hierarchical GNN
│   │   ├── siren.py          # SIREN decoder for implicit fields
│   │   └── hybrid_model.py   # Combined HGNN-NIF model
│   ├── data/
│   │   ├── synthetic_data.py # Generate test data without Blender
│   │   └── dataset.py        # PyTorch Dataset for HDF5 data
│   ├── training/
│   │   ├── losses.py         # Physics-based loss functions
│   │   └── trainer.py        # Training loop with curriculum learning
│   └── utils/
│       └── metrics.py        # Chamfer distance, utilities
├── tests/
│   ├── test_model.py         # Unit tests for model components
│   └── test_forward.py       # Integration tests
├── scripts/
│   ├── train_local.py        # Local training script
│   ├── deploy_vastai.py      # Vast.ai deployment automation
│   └── test_remote.py        # Remote testing script
├── docker/
│   ├── Dockerfile           # Container configuration
│   └── requirements.txt     # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Local Setup

```bash
# Clone or navigate to project
cd hgnn-nif-cloth

# Install dependencies
pip install torch torchvision numpy scipy h5py matplotlib pytest tensorboard tqdm

# Run tests
python -m pytest tests/test_forward.py -v

# Quick training test (3 epochs)
python scripts/train_local.py --test_mode
```

### Generate Synthetic Data

```bash
python -m src.data.synthetic_data --num_samples 100 --output data/train_data.h5
```

### Full Training

```bash
python scripts/train_local.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --use_curriculum \
    --output_dir outputs/experiment1
```

## 🐳 Docker

```bash
# Build image
docker build -t hgnn-nif-cloth -f docker/Dockerfile .

# Run with GPU
docker run --gpus all -v $(pwd)/outputs:/workspace/hgnn-nif-cloth/outputs hgnn-nif-cloth

# Interactive mode
docker run --gpus all -it hgnn-nif-cloth bash
```

## ☁️ Vast.ai Deployment

Deploy to a cloud GPU instance (L40S with 48GB VRAM):

```bash
# Install Vast.ai CLI
pip install vastai

# Deploy (interactive)
python scripts/deploy_vastai.py

# Test remote instance
python scripts/test_remote.py --instance_id <ID> --benchmark
```

## 🏗️ Architecture

### Model Components

1. **GraphConv**: Vectorized message-passing using scatter operations
2. **CrossLevelAttention**: Multi-head attention between graph resolutions
3. **AdaptiveHGNN**: 
   - Processes fine (400 nodes) and coarse (100 nodes) graphs
   - Energy-based gating for adaptive resolution
   - Produces latent code for SIREN conditioning
4. **SIRENDecoder**: 
   - Sinusoidal activation for high-frequency detail
   - Latent-conditioned for shape family representation
   - Outputs SDF values at arbitrary query points

### Loss Functions

```
L_total = λ_spring * L_spring + λ_sdf * L_SDF + λ_eik * L_eikonal
```

- **Edge Spring Loss**: Enforces elastic constraints `(||x_i - x_j|| - l_ij)²`
- **SDF Reconstruction**: `(φ_pred - φ_gt)²` with surface-aware weighting
- **Eikonal Regularization**: `(||∇φ|| - 1)²` for valid SDFs

### Training Strategy

Curriculum learning in 3 stages:
1. **Stage 0** (0-33%): Focus on SDF reconstruction
2. **Stage 1** (33-66%): Gradually add physics constraints  
3. **Stage 2** (66-100%): Full training with all losses

## 📊 Benchmarks

Target performance on NVIDIA L40S (48GB):

| Metric | Target | Notes |
|--------|--------|-------|
| Inference FPS | >50 | Batch size 1, 1000 query points |
| Training batch | 8-16 | Mixed precision enabled |
| Memory usage | <40GB | Leaves headroom for larger batches |
| Chamfer distance | <0.01 | On synthetic test data |

## 🧪 Testing

```bash
# Unit tests
python -m pytest tests/test_model.py -v

# Integration tests  
python -m pytest tests/test_forward.py -v

# All tests
python -m pytest tests/ -v
```

## 📈 TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

## 🔧 Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 64 | Latent code dimension |
| `hidden_dim` | 64 | HGNN hidden dimension |
| `siren_hidden` | 128 | SIREN hidden dimension |
| `siren_layers` | 3 | Number of SIREN layers |
| `num_heads` | 4 | Attention heads |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `batch_size` | 8 | Training batch size |
| `epochs` | 100 | Number of epochs |
| `lambda_spring` | 0.1 | Spring loss weight |
| `lambda_sdf` | 1.0 | SDF loss weight |

## 📚 References

- [SIREN](https://www.vincentsitzmann.com/siren/) - Implicit Neural Representations with Periodic Activation Functions
- [MeshGraphNets](https://arxiv.org/abs/2010.03409) - Learning Mesh-Based Simulation with Graph Networks
- [Graph U-Nets](https://arxiv.org/abs/1905.05178) - Hierarchical Graph Pooling

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 🐛 Troubleshooting

**CUDA OOM errors**: Reduce batch size or enable gradient checkpointing
```bash
python scripts/train_local.py --batch_size 4
```

**Import errors**: Ensure PYTHONPATH includes project root
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Slow training**: Enable mixed precision
```bash
python scripts/train_local.py --use_amp
```
