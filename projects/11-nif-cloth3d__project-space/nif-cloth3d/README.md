# NIF-Cloth3D-Interactive

A neural implicit field model for real-time cloth simulation with physics-informed losses and interactive Blender integration.

## Update (April 17, 2026)

- The model path was refreshed and shared collision hooks were added around the interactive cloth workflow.
- Project 11 now fits more cleanly into the shared addon and cloth-tooling story in the repo.
- This README update is mainly about keeping docs aligned with the current shared infrastructure.

## 🎯 Project Overview

NIF-Cloth3D-Interactive is a SIREN-based neural network that predicts cloth geometry under arbitrary forces. The system enables artists to interactively control cloth simulation parameters (wind, pins, materials) and see real-time results in Blender.

### Key Features

- **SIREN Architecture**: Sine-activated MLP for high-frequency cloth deformation modeling
- **Physics-Informed Losses**: Stretch, bend, and momentum conservation constraints
- **Real-Time Inference**: FP16 optimization and voxel-hash caching for <10ms latency
- **Curriculum Learning**: Force magnitude ramping for stable training
- **Blender Integration**: Interactive plugin with GUI controls
- **Cloud-Ready**: Docker deployment for vast.ai / cloud GPU training

## 📁 Project Structure

```
nif-cloth3d/
├── src/
│   ├── model.py          # SIREN MLP architecture
│   ├── losses.py         # Physics-informed loss functions
│   ├── train.py          # Training loop with curriculum learning
│   └── inference.py      # Optimized real-time inference
├── configs/
│   ├── rtx4090.yaml      # RTX 4090 (24GB) configuration
│   ├── l40s.yaml         # NVIDIA L40S (48GB) configuration
│   └── low_vram.yaml     # Low-VRAM (8GB) configuration
├── tests/
│   ├── test_model.py     # Model unit tests
│   ├── test_inference.py # Inference benchmarks
│   └── generate_test_data.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── blender_addon/
│   ├── nif_cloth_addon.py        # Blender UI plugin
│   └── blender_cloth_pipeline.py # Data generation script
├── notebooks/
│   └── evaluation.ipynb  # Evaluation notebook
├── checkpoints/          # Saved models
├── data/                 # Training data
└── logs/                 # Training logs
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and enter project
cd nif-cloth3d

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Test Data

```bash
python tests/generate_test_data.py
```

### 3. Train Model

```bash
# RTX 4090
python src/train.py --config configs/rtx4090.yaml --output checkpoints

# Low VRAM (8GB)
python src/train.py --config configs/low_vram.yaml --output checkpoints

# NVIDIA L40S (Cloud)
python src/train.py --config configs/l40s.yaml --output checkpoints
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Benchmark Inference

```bash
python src/inference.py --model checkpoints/model_traced.pt --vertices 10000
```

## 🐳 Docker Deployment

### Build and Run Locally

```bash
cd docker
docker-compose build
docker-compose up train
```

### vast.ai Deployment

```bash
# Search for L40S instances
vastai search offers "gpu_name=L40S rentable=true verified=true" --order dph_total

# Create instance
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  --disk 100

# SSH and run training
ssh root@<INSTANCE_IP> -p <PORT>
cd /workspace && git clone <YOUR_REPO>
cd nif-cloth3d && python src/train.py --config configs/l40s.yaml
```

## 🎨 Blender Integration

### Install Add-on

1. Install PyTorch in Blender's Python:
   ```bash
   /path/to/blender/python/bin/pip install torch
   ```

2. In Blender: Edit → Preferences → Add-ons → Install
   Select `blender_addon/nif_cloth_addon.py`

3. Enable "NIF-Cloth3D Interactive"

### Usage

1. Create or select a mesh named "Cloth"
2. Open 3D View → Sidebar → Cloth3D tab
3. Load trained model (.pt file)
4. Adjust Wind Speed, Material, Time sliders
5. Click "Apply Neural Step" to update mesh

### Generate Training Data from Blender

```bash
blender --background --python blender_addon/blender_cloth_pipeline.py -- \
  --output data/sim_001 \
  --frames 100 \
  --resolution 30 \
  --wind-strength 25.0
```

## 📊 Model Architecture

```
Input: 8D tensor [x, y, z, t, fx, fy, fz, material_id]
       ↓
   SineLayer (ω₀=30)  ← First layer, higher frequency
       ↓
   SineLayer (ω₀=1)   ×N hidden layers
       ↓
   Linear (→3D)
       ↓
Output: 3D displacement [dx, dy, dz]
```

### Physics Losses

| Loss | Formula | Purpose |
|------|---------|---------|
| Stretch | $L = \sum_{(i,j)} (\|u_i - u_j\| - d_{ij})^2$ | Preserve edge lengths |
| Bend | $L = \sum_i \|\Delta u_i\|^2$ | Smooth curvature |
| Momentum | $L = \|\rho \ddot{u} - f_{net}\|^2$ | Dynamic consistency |

## ⚡ Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference latency (10k verts) | <10ms | ✓ |
| Training (1000 epochs, L40S) | ~2 hours | ✓ |
| Final loss | <0.01 | ✓ |

## 📚 References

- [SIREN: Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- [DiffCloth: Differentiable Cloth Simulation](https://people.csail.mit.edu/liyifei/publication/diffcloth/)
- [Neural Cloth Simulation](https://arxiv.org/abs/2202.00658)

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

---

**NIF-Cloth3D-Interactive** - Neural Implicit Fields for Interactive Cloth Simulation
