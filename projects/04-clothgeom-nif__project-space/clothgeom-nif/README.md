# ClothGeom-NIF

**Neural Implicit Field Decoder for High-Detail Cloth Geometry**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ClothGeom-NIF maps latent cloth states (node positions + edge strains) to watertight, high-detail cloth surfaces using neural implicit fields with SIREN activations.

## 🚀 Quick Start

### Option 1: Fully Automated (Recommended)

```bash
# One command to provision GPU, train, and get results
bash quickstart.sh
# Results will be in local_outputs/ after ~45 minutes
```

### Option 2: Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python generate_dataset.py --num_samples 100

# 3. Train model
python train.py --epochs 100

# 4. Extract meshes
python demo.py
```

### Option 3: Manual Vast.ai Workflow

```bash
# 1. Search for GPUs
bash search_gpus.sh

# 2. Create instance (replace ID)
vastai create instance <ID> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 50 --ssh

# 3. SSH and setup
ssh root@<host> -p <port>
git clone <your-repo> && cd clothgeom-nif
bash setup_environment.sh

# 4. Train
python generate_dataset.py --num_samples 100
python train.py --epochs 100
python demo.py

# 5. Clean up
vastai destroy instance <ID>
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 8GB+ GPU VRAM (24GB recommended for full training)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch 2.1+
- NumPy
- h5py
- scikit-image
- trimesh
- matplotlib

Optional:
- PyTorch3D (for advanced metrics)
- tinycudann (for hash encoding acceleration)

## 🏗️ Architecture

ClothGeom-NIF uses a SIREN-based neural network to predict signed distance fields (SDF):

```
Input: 3D coords [x,y,z] + Latent code [128D]
        ↓
   SIREN Layer (ω₀=30) × 8 layers
   256 hidden units per layer
   Skip connection at layer 4
        ↓
   ┌────────┴────────┐
   ↓                 ↓
SDF Head         Variance Head
   ↓                 ↓
Signed Distance  Uncertainty (LOD)
```

### Key Features

- **SIREN Activations**: Sine activations with ω₀=30 for high-frequency detail
- **Latent Modulation**: Cloth state encoded as 128D latent vector
- **Dual Outputs**: SDF for geometry + Variance for LOD blending
- **Progressive Resolution**: Extract meshes at 64³, 128³, or 256³
- **Mixed Precision**: FP16 training for 40% memory reduction

## 📊 Data Generation

Generate synthetic cloth training data:

```bash
# Quick test (10 samples, 32³ SDF)
python generate_dataset.py --num_samples 10 --resolution 32

# Full training (100 samples, 64³ SDF)
python generate_dataset.py --num_samples 100 --resolution 64 --validate
```

The generator creates:
- 32×32 cloth meshes with random deformations
- Sine waves, gravity drape, and wind effects
- Ground-truth SDF volumes
- Latent codes from node positions + edge strains

## 🎓 Training

Train the NIF decoder:

```bash
# Default training (100 epochs)
python train.py --data data/generated/cloth_dataset.h5 --epochs 100

# Quick test (10 epochs)
python train.py --data data/generated/cloth_dataset.h5 --epochs 10 --quick

# Resume from checkpoint
python train.py --resume outputs/checkpoints/latest.pt
```

### Training Configuration

Edit `configs/train_config.yaml`:

```yaml
model:
  latent_dim: 128
  hidden_dim: 256
  num_layers: 8
  omega_0: 30.0

training:
  batch_size: 8
  num_points: 8192
  learning_rate: 1e-4
  epochs: 100

loss:
  lambda_sdf: 1.0
  lambda_eikonal: 0.1
  lambda_variance: 0.01
```

### Expected Training Time

| GPU | 100 Samples | 100 Epochs |
|-----|-------------|------------|
| RTX 3090 | ~30 min | Loss < 0.01 |
| RTX 4090 | ~20 min | Loss < 0.01 |
| A100 | ~15 min | Loss < 0.01 |

## 🔧 Mesh Extraction

Extract watertight meshes from trained model:

```bash
# Extract 5 random meshes
python extract_meshes.py --checkpoint outputs/checkpoints/best.pt --num 5

# Higher resolution
python extract_meshes.py --checkpoint outputs/checkpoints/best.pt --resolution 256

# Progressive multi-resolution
python extract_meshes.py --checkpoint outputs/checkpoints/best.pt --progressive
```

Output formats: OBJ, PLY, STL

## 🎬 Demo

Run end-to-end demonstration:

```bash
# With trained model
python demo.py --checkpoint outputs/checkpoints/best.pt

# Test with random weights
python demo.py --test

# Latent interpolation
python demo.py --checkpoint outputs/checkpoints/best.pt --interpolate --num 10

# With visualization
python demo.py --checkpoint outputs/checkpoints/best.pt --visualize
```

## 🧪 Testing

Run unit tests:

```bash
# All tests
python -m pytest tests/ -v

# Model tests only
python tests/test_model.py

# Training tests only
python tests/test_training.py
```

## 📁 Project Structure

```
clothgeom-nif/
├── README.md                    # This file
├── ARCHITECTURE.md              # Technical details
├── requirements.txt             # Python dependencies
│
├── quickstart.sh               # Full automation script
├── setup_vastai.sh             # Vast.ai provisioning
├── search_gpus.sh              # GPU search helper
├── setup_environment.sh        # Remote setup script
│
├── generate_dataset.py         # Data generation CLI
├── train.py                    # Training script
├── extract_meshes.py           # Mesh extraction CLI
├── demo.py                     # Demo script
│
├── configs/
│   └── train_config.yaml       # Training config
│
├── models/
│   ├── __init__.py
│   ├── siren.py               # SIREN layers
│   └── nif_decoder.py         # Main decoder
│
├── data/
│   ├── __init__.py
│   ├── synthetic_cloth_generator.py
│   └── cloth_dataset.py
│
├── inference/
│   ├── __init__.py
│   └── mesh_extractor.py
│
├── tests/
│   ├── test_model.py
│   └── test_training.py
│
├── outputs/                    # Generated outputs
│   ├── checkpoints/           # Model checkpoints
│   ├── meshes/                # Extracted meshes
│   └── logs/                  # TensorBoard logs
│
└── data/
    └── generated/             # Training data
```

## 🔬 Technical Details

### SDF Loss Function

The training uses a combined loss:

```
L = λ_sdf × L_sdf + λ_eik × L_eikonal + λ_var × L_variance
```

Where:
- **L_sdf**: MSE between predicted and ground-truth SDF
- **L_eikonal**: Gradient magnitude constraint (||∇SDF|| = 1)
- **L_variance**: Regularization on uncertainty estimates

### SIREN Initialization

Critical for learning high-frequency details:

- **First layer**: Weights ~ U(-1/fan_in, 1/fan_in)
- **Hidden layers**: Weights ~ U(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀)
- **ω₀ = 30**: Enables learning of wrinkle-scale details

### Mesh Extraction

Uses marching cubes on the SDF grid:
1. Evaluate SDF at regular grid points
2. Find zero-crossing isosurface (SDF = 0)
3. Extract triangulated mesh
4. Optional: Laplacian smoothing

## 📈 Expected Results

| Metric | Target | Description |
|--------|--------|-------------|
| Training Loss | < 0.01 | Combined SDF + Eikonal loss |
| Mesh Vertices | 10k-100k | At 128³ resolution |
| Mesh Quality | Watertight | No holes or artifacts |
| Extraction Time | < 5s | Per mesh at 128³ |

## 🐛 Troubleshooting

### No suitable GPU found on Vast.ai

```bash
# Try broader search
vastai search offers "rentable=true verified=true gpu_ram>=16" --order dph_total --limit 20
```

### Training loss not decreasing

1. Check SDF ground truth: `python generate_dataset.py --validate`
2. Reduce learning rate: `--lr 1e-5`
3. Increase Eikonal weight in config

### Empty meshes extracted

1. SDF may not have zero crossing
2. Check SDF range: should have both positive and negative values
3. Try different latent codes or lower resolution

### GPU out of memory

1. Reduce batch size: `--batch_size 4`
2. Reduce num_points: `--num_points 4096`
3. Enable gradient checkpointing (in config)

## 📄 License

MIT License - see LICENSE file.

## 📚 References

- SIREN: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- DeepSDF: [Learning Continuous Signed Distance Functions](https://arxiv.org/abs/1901.05103)
- Instant-NGP: [Neural Graphics Primitives with Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `python -m pytest tests/ -v`
4. Submit a pull request
