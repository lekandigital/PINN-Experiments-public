# Geodesic Trajectory PINNs

**Physics-Informed Neural Networks for Trajectory Prediction on Potential Landscapes**

A domain-agnostic framework for predicting optimal trajectories as geodesics on learnable potential fields. Originally developed for microbe chemotaxis (Cell-Path PINNs), now generalized to support robotics, migration, finance, game AI, and more.

## 🌟 Key Features

- **Domain-Agnostic Core**: Abstract geodesic trajectory framework works across application domains
- **Physics-Informed**: Enforces constant speed, boundary conditions, gradient following, and curvature constraints
- **Real-Time Ready**: ONNX export for <1ms inference (game AI) to <10ms (robotics)
- **Extensible**: Easy to add new domains with custom potential fields and loss functions

## Supported Domains

| Domain | Use Case | Potential Field | Performance Target |
|--------|----------|-----------------|-------------------|
| **Biology** | Microbe chemotaxis | Nutrient concentration | Research |
| **Robotics** | Path planning | Terrain + obstacles | <10ms inference |
| **Migration** | Animal movement | Environmental factors | Analysis |
| **Finance** | Portfolio transitions | Profit/risk landscape | Batch |
| **Game AI** | NPC pathfinding | Game world costs | <1ms inference |

## Quick Start

### Installation

```bash
cd cell-path-pinns
pip install -e .
```

### Basic Usage (Original Biology API)

```python
from cell_path_pinns import CellPathModel, generate_synthetic_trajectory

# Generate synthetic data
t, x, y = generate_synthetic_trajectory(n_steps=200)

# Train model
model = CellPathModel(device='cuda')  # Use 'cpu' if no GPU
model.fit(t, x, y, epochs=200)

# Predict trajectories
import numpy as np
t_pred = np.linspace(0, 20, 100)
xy_pred = model.predict_paths(t_pred)
```

### New Domain-Agnostic API

```python
import torch
from src.core import LearnedPotentialField, GeodesicLoss, TrajectoryPINN
from src.domains.robotics import TerrainField, RobotPathLoss

# Define environment
obstacles = torch.tensor([[0.3, 0.3], [0.7, 0.5]])
goal = torch.tensor([1.0, 0.0])

# Create potential field and loss
terrain = TerrainField(input_dim=2, obstacles=obstacles, goal=goal)
loss_fn = RobotPathLoss(terrain_field=terrain)

# Create trajectory network
trajectory = TrajectoryPINN(output_dim=2, hidden_dims=[64, 64], context_dim=4)

# Train with physics constraints...
```

## Core Concepts

### Physics-Informed Constraints

The framework enforces four key physics constraints:

1. **Constant Speed Loss**: Enforce |v|² = c² for efficient motion
2. **Boundary Loss**: Match start/end positions exactly
3. **Gradient Following Loss**: Velocity aligns with -∇φ (steepest descent)
4. **Curvature Penalty**: Minimize acceleration for smooth paths

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrajectoryPINN                           │
│  t, context → [MLP] → position x(t)                         │
│              → velocity v(t) = dx/dt (autograd)             │
│              → acceleration a(t) = dv/dt (autograd)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PotentialFieldBase                         │
│  x → [Domain-Specific Network] → scalar φ(x)                │
│                                → gradient ∇φ(x)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GeodesicLoss                             │
│  Combines: speed + boundary + gradient + curvature losses   │
└─────────────────────────────────────────────────────────────┘
```

**Domain Adapters:**
- `NutrientField` → Biology (chemotaxis)
- `TerrainField` → Robotics (obstacles + goal)
- `EnvironmentalField` → Migration (multi-factor)
- `ProfitLandscape` → Finance (risk-adjusted return)
- `GameWorldField` → Game AI (costs + navigation)

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (optional for small models)
- **CPU**: Works well for ~10K parameter models
- **Tested on**: NVIDIA L40S (48GB VRAM), Apple M1/M2

## Performance Benchmarks

### Training Times (L40S GPU)

| Task | Time |
|------|------|
| Training (200 epochs, 200 points) | ~30-60 seconds |
| Inference (100 points) | <100ms |
| Full demo notebook | ~2 minutes |

### Real-Time Inference (CPU)

| Domain | Model Size | Inference Time |
|--------|------------|----------------|
| Game AI | ~8.5K params | <1ms |
| Robotics | ~35K params | <10ms |
| Biology | ~35K params | <10ms |

## Project Structure

```
cell-path-pinns/
├── cell_path_pinns/          # Original biology API (backward compatible)
│   ├── __init__.py
│   ├── models.py             # PathNet, PotentialNet
│   ├── losses.py             # Physics losses
│   ├── data_utils.py         # Synthetic data
│   └── api.py                # CellPathModel wrapper
│
├── src/                      # New domain-agnostic framework
│   ├── core/                 # Abstract base classes
│   │   ├── potential_field.py    # PotentialFieldBase, LearnedPotentialField
│   │   ├── geodesic_loss.py      # ConstantSpeedLoss, BoundaryLoss, etc.
│   │   └── trajectory_pinn.py    # TrajectoryPINN, BoundaryConditionedTrajectory
│   │
│   ├── domains/              # Domain-specific adapters
│   │   ├── biology.py        # NutrientField, ChemotaxisLoss
│   │   ├── robotics.py       # TerrainField, RobotPathLoss
│   │   ├── migration.py      # EnvironmentalField, MigrationLoss
│   │   ├── finance.py        # ProfitLandscape, FinanceLoss
│   │   └── game_ai.py        # GameWorldField, GameAILoss
│   │
│   └── export/               # ONNX export utilities
│       └── __init__.py
│
├── demos/                    # Domain demonstrations
│   ├── robot_planning.py     # Robot path planning demo
│   └── game_ai.py            # Game AI pathfinding demo
│
├── tests/                    # Comprehensive test suite
│   ├── test_core.py          # Core abstraction tests
│   ├── test_domains.py       # Domain adapter tests
│   └── test_trajectory_module.py  # Shared utilities tests
│
├── notebooks/
│   └── quick_demo.ipynb      # Interactive demo
│
└── README.md
```

## Adding a New Domain

Creating a new domain adapter is straightforward:

### Step 1: Create a Potential Field

```python
from src.core.potential_field import PotentialFieldBase
import torch.nn as nn

class MyDomainField(PotentialFieldBase):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.network(x)
```

### Step 2: Create a Domain Loss

```python
from src.core.geodesic_loss import GeodesicLoss

class MyDomainLoss(GeodesicLoss):
    def __init__(self, potential_field, **kwargs):
        super().__init__(
            potential_field=potential_field,
            speed_weight=1.0,
            boundary_weight=10.0,
            gradient_weight=1.0,
            curvature_weight=0.1,
        )
    
    # Optionally add domain-specific loss terms
    def domain_specific_loss(self, positions, velocity):
        # Your custom physics here
        return custom_loss
```

### Step 3: Create an Adapter

```python
class MyDomainAdapter:
    def get_default_config(self):
        return {
            'hidden_dims': [64, 64],
            'target_speed': 1.0,
        }
    
    def create_model_and_loss(self, **kwargs):
        field = MyDomainField(input_dim=2, **kwargs)
        loss_fn = MyDomainLoss(potential_field=field)
        return field, loss_fn
```

## ONNX Export for Real-Time Inference

```python
from src.export import export_trajectory_model, benchmark_onnx_inference

# Export for game AI (optimized for speed)
export_trajectory_model(
    trajectory_model=model,
    output_dir="exports/game_ai",
    target_platform="browser",
    model_name="npc_pathfinder",
)

# Benchmark
results = benchmark_onnx_inference("exports/game_ai/npc_pathfinder.onnx")
print(f"Mean inference: {results['mean_ms']:.3f}ms")
```

## For Biologist Collaborators

### Minimal Python Setup

1. Install Python 3.8+ (Anaconda recommended)
2. Install PyTorch: https://pytorch.org/get-started/locally/
3. Install this package: `pip install -e .`

### Using Your Own Data

Prepare your data as three arrays:
- `t`: Time points (seconds or frames)
- `x`: X positions (pixels or micrometers)
- `y`: Y positions (pixels or micrometers)

```python
import pandas as pd
from cell_path_pinns import CellPathModel

# Load your centroid data
df = pd.read_csv('my_tracking_data.csv')
t = df['frame'].values
x = df['x'].values
y = df['y'].values

# Train and predict
model = CellPathModel()
model.fit(t, x, y, epochs=500)
predictions = model.predict_paths(t)
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{geodesic_trajectory_pinns,
  title={Geodesic Trajectory PINNs: Physics-Informed Neural Networks for Trajectory Prediction},
  year={2025},
  url={https://github.com/your-repo/geodesic-trajectory-pinns},
  note={Originally developed as Cell-Path PINNs for microbe chemotaxis}
}
```

## Related Work

This framework builds on concepts from:
- Physics-Informed Neural Networks (Raissi et al., 2019)
- Neural Implicit Representations (Mildenhall et al., 2020)
- Geodesic distance computation on manifolds

## License

MIT License
