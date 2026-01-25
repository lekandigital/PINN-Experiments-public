# NIF-Cloth4D: Neural Implicit Field for Cloth Dynamics

A PyTorch implementation of a neural implicit signed distance field (SDF) network that predicts cloth geometry at any timestamp, replacing traditional FEM/mass-spring solvers.

## Overview

NIF-Cloth4D learns a continuous function `f(x, y, z, t) → SDF` that represents cloth geometry over time. Given a 4D spacetime coordinate, the network predicts the signed distance to the cloth surface.

### Key Features

- **SIREN Architecture**: Uses sinusoidal activations for high-frequency detail capture
- **4D Spacetime Input**: Predicts cloth at any continuous timestamp
- **Physics-Informed Losses**: Optional stretch, bend, and energy regularization
- **USD/OBJ Export**: Export predictions to standard 3D formats
- **Memory Efficient**: Chunked sampling for low-VRAM training

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: mesh extraction and USD export
pip install PyMCubes usd-core
```

### 2. Generate Test Data

```bash
python synthetic_data.py --output_dir /tmp/cloth_test_data --num_frames 10 --data_type falling
```

### 3. Train Model

```bash
python train_nif_cloth4d.py --config config_test.yaml --data_dir /tmp/cloth_test_data --output_dir ./checkpoints
```

### 4. Evaluate

```bash
python test_evaluation.py --checkpoint ./checkpoints/model_best.pt --data_dir /tmp/cloth_test_data
```

### 5. Export Meshes

```bash
python export_mesh.py --checkpoint ./checkpoints/model_best.pt --times 0.0,0.5,1.0 --output ./exports
```

### One-Command Pipeline

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh /tmp/cloth_test_data ./output
```

## Project Structure

```
nif_cloth4d/
├── __init__.py              # Package initialization
├── nif_cloth4d.py           # Core SIREN network and losses
├── synthetic_data.py        # Synthetic SDF data generator
├── train_nif_cloth4d.py     # Training script
├── test_evaluation.py       # Evaluation and metrics
├── export_mesh.py           # Mesh export (USD/OBJ)
├── config_test.yaml         # Training configuration
├── requirements.txt         # Dependencies
├── run_pipeline.sh          # Full pipeline script
├── TEST_REPORT_TEMPLATE.md  # Test report template
└── README.md                # This file
```

## Architecture

### SIREN Network

```
Input: (x, y, z, t) → Linear(4, 256) → sin(30·x) → 
       [Linear(256, 256) → sin(x)] × 5 → 
       Linear(256, 1) → SDF value
```

- **First layer**: High frequency scaling (w0=30) for coordinate encoding
- **Hidden layers**: Standard SIREN with w0=1
- **Output**: Linear layer for unbounded SDF output

### Loss Functions

| Loss | Description | Default Weight |
|------|-------------|----------------|
| SDF | MSE reconstruction | 1.0 |
| Eikonal | Gradient magnitude = 1 | 0.01 |
| Stretch | Edge length preservation | 0.1 (optional) |
| Bend | Curvature preservation | 0.1 (optional) |

## Configuration

Edit `config_test.yaml` for training parameters:

```yaml
model:
  hidden_dim: 128        # Network width
  hidden_layers: 4       # Network depth
  w0_initial: 30.0       # SIREN frequency

training:
  epochs: 50             # Training epochs
  batch_size: 8192       # Points per batch
  learning_rate: 1e-4    # Adam learning rate
```

## GPU Requirements

| Config | VRAM | Batch Size | Hidden Dim |
|--------|------|------------|------------|
| Test | 4 GB | 8,192 | 128 |
| Standard | 12 GB | 32,768 | 256 |
| Full | 24+ GB | 65,536 | 256 |

## Vast.ai Deployment

```bash
# Search for L40S GPU
vastai search offers "gpu_name=L40S rentable=true verified=true" --order dph_total --limit 10

# Create instance
vastai create instance <ID> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel --disk 100

# SSH and run
vastai ssh instance <INSTANCE_ID>
cd /workspace && git clone <repo> && cd nif_cloth4d
pip install -r requirements.txt
./run_pipeline.sh

# IMPORTANT: Destroy when done!
vastai destroy instance <INSTANCE_ID>
```

## Evaluation Metrics

- **Chamfer Distance**: Average bidirectional point-to-point distance
- **Hausdorff Distance**: Maximum deviation (worst case)
- **Energy Error**: Physical plausibility measure

Target for test: Chamfer < 0.05, Loss < 0.01

## Export Formats

- **USD (.usda)**: Universal Scene Description for DCC tools
- **OBJ (.obj)**: Wavefront format for wide compatibility

## References

This implementation is based on:

1. **SIREN**: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions" (NeurIPS 2020)
2. **Neural Implicit Flow**: Pan et al., "A Mesh-Agnostic Dimensionality Reduction Paradigm" (JMLR 2023)
3. **NeuralClothSim**: Kairanda et al., "Neural Deformation Fields Meet Thin Shell Theory" (NeurIPS 2024)

## License

Research use only. See LICENSE for details.

---

*NIF-Cloth4D: Neural Implicit Field for 4D Cloth Dynamics*
