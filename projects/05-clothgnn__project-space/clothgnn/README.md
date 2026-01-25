# ClothGNN

Recurrent Graph Neural Network for Cloth and Hair Dynamics Prediction.

Based on MeshGraphNetRP architecture with temporal consistency via GRU cells.

## Architecture

```
Input Features (N, 16) → ClothEncoder (GNN) → GRU Cell → ClothDecoder (MLP) → Displacements (N, 3)
                              ↑                  ↑
                        edge_index          hidden state
```

### Components

- **ClothEncoder**: MessagePassing GNN with mean aggregation and edge features (relative position + distance)
- **GRUCell**: Temporal dynamics with hidden_dim=64
- **ClothDecoder**: 3-layer MLP outputting 3D vertex displacements
- **OccupancyGrid**: Voxelized SDF collision detection (32³ grid)

### Loss Functions

| Loss | Weight | Description |
|------|--------|-------------|
| Position | λ=1.0 | L2 on predicted vs ground-truth displacements |
| Edge Length | λ=0.1 | Preserve rest edge lengths |
| Shear | λ=0.05 | Penalize triangle angle changes |
| Seam | λ=0.02 | Smooth seam deformation |

## Test Results (NVIDIA L40S)

```
Final training loss: 0.029732
Inference speed (1K vertices): 1901.1 FPS
GPU Memory: 0.02 GB allocated

Performance by mesh size:
  100 vertices:  1,907 FPS
  500 vertices:  1,876 FPS
1,000 vertices:  1,876 FPS
2,000 vertices:  1,860 FPS
5,000 vertices:  1,832 FPS
```

## Project Structure

```
clothgnn/
├── models/
│   ├── __init__.py
│   └── clothgnn.py          # Main model (Encoder, GRU, Decoder)
├── utils/
│   ├── __init__.py
│   ├── collision.py         # OccupancyGrid, SDF functions
│   └── losses.py            # Physics-informed losses
├── tests/
│   ├── __init__.py
│   ├── generate_test_data.py # Synthetic cloth data generator
│   └── test_model.py        # Comprehensive test suite
├── results/
│   ├── test_results.txt     # Performance benchmarks
│   └── gpu_info.txt         # nvidia-smi output
└── data/                    # Generated HDF5 datasets
```

## Usage

### Run Tests
```bash
cd clothgnn
python tests/test_model.py
```

### Generate Data
```python
from tests.generate_test_data import generate_all_test_data
generate_all_test_data(output_dir="./data")
```

### Use Model
```python
import torch
from torch_geometric.data import Data
from models.clothgnn import ClothGNNModel

# Initialize
model = ClothGNNModel(node_feat_dim=16, hidden_dim=64).cuda()
h = model.init_hidden(num_nodes=1000, device="cuda")

# Forward pass
data = Data(x=node_features, edge_index=edge_index, pos=positions)
pred_displacement, h_next = model(data, h)
```

## Dependencies

- PyTorch >= 2.1.0
- PyTorch Geometric
- h5py
- numpy

## Tested On

- NVIDIA L40S (48GB VRAM)
- PyTorch 2.1.0 + CUDA 12.1
- VastAI cloud instance
