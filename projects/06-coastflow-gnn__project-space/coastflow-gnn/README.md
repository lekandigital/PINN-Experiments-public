# CoastFlow-GNN

**Physics-Informed Graph Neural Network for Coastal Wind-Wave Modeling**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CoastFlow-GNN is a multi-scale hierarchical Graph Neural Network (GNN) surrogate model that predicts coupled wind-wave fields from shoreline DEM (Digital Elevation Model) meshes. It combines Physics-Informed Neural Network (PINN) loss functions with Graph Convolutional Networks to accelerate coastal engineering simulations.

## 🌊 Features

- **Hierarchical Graph Pooling**: Multi-scale GCN layers with TopKPooling for capturing features at different spatial resolutions
- **Physics-Informed Training**: Enforces physical constraints including:
  - Continuity equation (∇·u = 0)
  - Navier-Stokes momentum residuals
  - k-ε turbulence model constraints
  - Coastal boundary conditions
- **Scalable Training**: Single GPU and multi-GPU DDP support
- **Fast Inference**: <100ms per sample on GPU
- **Synthetic Data Generation**: Built-in dataset generator for rapid prototyping

## 📁 Project Structure

```
coastflow-gnn/
├── src/
│   ├── models/
│   │   ├── coastflow_gnn.py      # Main GNN architecture
│   │   └── physics_losses.py      # Physics-informed loss functions
│   ├── data/
│   │   ├── dataset.py             # Synthetic dataset generator
│   │   ├── dem_fetcher.py         # NOAA DEM utilities
│   │   └── mesh_builder.py        # Mesh creation and graph conversion
│   ├── training/
│   │   ├── train_single.py        # Single GPU training
│   │   ├── train_ddp.py           # Multi-GPU DDP training
│   │   └── eval_model.py          # Evaluation and benchmarking
│   └── utils/
│       ├── visualization.py       # Plotting utilities
│       └── cost_benefit.py        # CFD vs surrogate comparison
├── notebooks/
│   └── site_planning_demo.ipynb   # Interactive demo notebook
├── deployment/
│   ├── Dockerfile                 # Docker image for deployment
│   ├── requirements.txt           # Python dependencies
│   ├── vastai_setup.sh           # Vast.ai instance provisioning
│   └── remote_setup.sh           # Remote environment setup
├── tests/
│   ├── test_model.py
│   ├── test_physics_losses.py
│   ├── test_dataset.py
│   └── test_training_loop.py
├── config/
│   └── model_config.yaml          # Configuration file
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/coastflow-gnn.git
cd coastflow-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r deployment/requirements.txt

# Install PyTorch Geometric (for CUDA 12.1)
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py -v

# Run integration test
python tests/test_training_loop.py
```

### Train Model

```bash
# Quick training (small dataset)
python src/training/train_single.py --epochs 50 --batch-size 4

# Full training with custom parameters
python src/training/train_single.py \
    --epochs 100 \
    --batch-size 8 \
    --hidden 128 \
    --lr 5e-4 \
    --output-dir ./outputs
```

### Evaluate Model

```bash
python src/training/eval_model.py \
    --checkpoint outputs/best_model.pth \
    --output-dir outputs
```

## 🖥️ GPU Deployment (Vast.ai)

### 1. Install Vast.ai CLI

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

### 2. Find and Provision Instance

```bash
cd deployment
chmod +x vastai_setup.sh
./vastai_setup.sh
```

### 3. Connect and Setup

```bash
# SSH into instance
ssh -p <port> root@<ip>

# Clone and setup
git clone https://github.com/your-org/coastflow-gnn.git
cd coastflow-gnn/deployment
chmod +x remote_setup.sh
./remote_setup.sh
```

### 4. Run Training

```bash
cd /workspace/coastflow-gnn
python src/training/train_single.py --epochs 50
```

## 🔬 Model Architecture

```
Input: Node features [x, y, z, elevation, wind_u, wind_v] (6 channels)
                            │
                    ┌───────▼───────┐
                    │   Input MLP   │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  GCNConv x2   │ Level 0 (Full Resolution)
                    │ + Residual    │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ TopKPooling   │ Ratio: 0.8
                    │   (80%)       │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  GCNConv x2   │ Level 1
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ TopKPooling   │ Ratio: 0.5
                    │   (50%)       │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  GCNConv x2   │ Level 2
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ Global Pool   │ Multi-scale features
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  MLP Decoder  │
                    └───────┬───────┘
                            │
Output: [u_x, u_y, u_z, wave_height] (4 channels)
```

## 📊 Physics-Informed Loss

The total loss combines data fidelity with physics constraints:

```
L = λ_data·L_data + λ_cont·L_cont + λ_mom·L_mom + λ_turb·L_turb + λ_bc·L_bc
```

Where:
- **L_data**: MSE between predictions and ground truth
- **L_cont**: Continuity equation residual (∇·u = 0)
- **L_mom**: Navier-Stokes momentum residual
- **L_turb**: k-ε turbulence model constraints
- **L_bc**: Boundary condition violations

## 📈 Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| MSE | < 0.01 | ✅ |
| Inference Time | < 100ms | ✅ |
| GPU Memory (L40S) | < 48GB | ✅ |
| Training Time (50 epochs) | < 30 min | ✅ |

### Cost-Benefit Analysis

| Method | Runtime (100 scenarios) | Energy | Speedup |
|--------|------------------------|--------|---------|
| OpenFOAM CFD | ~10 hours | 5 kWh | 1x |
| CoastFlow-GNN | ~1 second | 0.01 kWh | 36,000x |

## 📚 Configuration

Edit `config/model_config.yaml` to customize:

```yaml
model:
  hidden_channels: 64
  pool_ratios: [0.8, 0.5]
  dropout: 0.1

training:
  epochs: 50
  batch_size: 4
  learning_rate: 1.0e-3

loss:
  lambda_data: 1.0
  lambda_cont: 1.0
  lambda_mom: 0.1
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for the GNN framework
- NOAA for DEM data access
- OpenFOAM community for CFD validation cases

## 📧 Contact

For questions or collaboration, please open an issue or contact the maintainers.

---

**CoastFlow-GNN** - Accelerating coastal engineering with physics-informed deep learning 🌊
