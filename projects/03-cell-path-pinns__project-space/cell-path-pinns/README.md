# Cell-Path PINNs

**Physics-Informed Neural Networks for Microbe Trajectory Prediction**

A deep learning framework that models microbe movement as geodesics on an evolving nutrient manifold. Uses physics-informed constraints to learn biologically meaningful trajectories.

## Quick Start

### Installation

```bash
# Clone or download the project
cd cell-path-pinns

# Install dependencies (PyTorch assumed pre-installed)
pip install -e .
```

### Basic Usage

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

## Core Concepts

### Physics-Informed Constraints

The model enforces three key physics constraints:

1. **Data Loss**: Fit observed trajectory positions
2. **Geodesic Loss**: Enforce constant-speed motion (microbes move efficiently)
3. **Chemotactic Loss**: Velocity follows nutrient gradient

### Architecture

- **PathNet**: Maps time → (x, y) position
- **PotentialNet**: Maps (x, y, t) → nutrient concentration U

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **Tested on**: NVIDIA L40S (48GB VRAM)
- **CPU**: Works but significantly slower

## Expected Run Times (L40S GPU)

| Task | Time |
|------|------|
| Training (200 epochs, 200 points) | ~30-60 seconds |
| Inference (100 points) | <100ms |
| Full demo notebook | ~2 minutes |

## Project Structure

```
cell-path-pinns/
├── cell_path_pinns/
│   ├── __init__.py      # Package exports
│   ├── models.py        # PathNet, PotentialNet architectures
│   ├── losses.py        # Physics-informed loss functions
│   ├── data_utils.py    # Synthetic data generation
│   └── api.py           # CellPathModel API wrapper
├── tests/
│   └── test_model.py    # Unit tests
├── notebooks/
│   └── quick_demo.ipynb # Interactive demo
├── setup.py
├── requirements.txt
└── README.md
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
@software{cell_path_pinns,
  title={Cell-Path PINNs: Physics-Informed Neural Networks for Microbe Trajectory Prediction},
  year={2025},
  url={https://github.com/your-repo/cell-path-pinns}
}
```

## License

MIT License
