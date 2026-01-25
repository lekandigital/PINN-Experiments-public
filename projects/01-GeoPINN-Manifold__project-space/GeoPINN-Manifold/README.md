# GeoPINN-Manifold

**Physics-Informed Neural Networks for PDEs on Curved Surfaces, Graphs, and Riemannian Manifolds**

GeoPINN-Manifold extends the standard PINN framework to handle partial differential equations defined on non-Euclidean domains. This includes:

- **Spheres**: Climate models, shallow-water equations
- **Thin Shells**: Structural mechanics on curved surfaces
- **Tori**: Reaction-diffusion patterns
- **General Meshes**: PDEs on arbitrary triangle meshes

## Key Features

- **Laplace-Beltrami Operator**: Automatic differentiation for surface Laplacians
- **Discrete Exterior Calculus (DEC)**: Build gradient, curl, and Laplacian operators on meshes
- **Tangent Message Passing**: Geometry-aware graph neural network layers
- **Spectral Convolution**: Frequency-domain operations using Laplacian eigenbasis
- **Chart Atlas**: Handle manifolds without global coordinates via patch-based approach

## Installation

### Quick Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Conda Environment

```bash
conda env create -f environment.yml
conda activate geopinn_manifold
pip install -e .
```

### Docker

```bash
docker build -t geopinn-manifold .
docker run --gpus all geopinn-manifold
```

## Quick Start

### Sphere PDE Test

Solve the Laplace-Beltrami equation on the unit sphere:

```python
from geopinn.training import SpherePINNTrainer
from geopinn.training.sphere_trainer import SphereMLP

# Create model
model = SphereMLP(in_dim=3, out_dim=1, hidden_dim=64)

# Define source term: f(x,y,z) = -2xy
def source_fn(points):
    return (-2 * points[:, 0] * points[:, 1]).unsqueeze(-1)

# Train
trainer = SpherePINNTrainer(model, source_fn, device='cuda')
history = trainer.train(n_points=1000, n_epochs=100)
```

### Run Tests

```bash
# Integration tests
python tests/test_integration.py

# Sphere PDE test
python tests/test_sphere_pde.py

# With mixed precision
python tests/test_sphere_pde.py --amp --epochs 200
```

## Project Structure

```
GeoPINN-Manifold/
├── geopinn/
│   ├── layers/
│   │   ├── tangent_message_passing.py  # Geometry-aware message passing
│   │   ├── spectral_conv.py            # Spectral graph convolution
│   │   ├── dec_operators.py            # DEC (gradient, curl, Laplacian)
│   │   └── chart_atlas.py              # Multi-chart approach
│   ├── data/
│   │   ├── sphere_swe.py               # Shallow-water equations on sphere
│   │   ├── shell_elasticity.py         # Thin-shell mechanics
│   │   └── torus_reaction_diffusion.py # Gray-Scott on torus
│   └── training/
│       ├── sphere_trainer.py           # PINN trainer for sphere PDEs
│       ├── mesh_trainer.py             # PINN trainer for mesh domains
│       └── adaptive_refinement.py      # Adaptive collocation points
├── tests/
│   ├── test_integration.py             # Module import tests
│   └── test_sphere_pde.py              # Full training test
├── environment.yml                      # Conda environment
├── requirements.txt                     # Pip requirements
├── Dockerfile                           # Docker build
└── README.md                            # This file
```

## Mathematical Background

### Laplace-Beltrami on Unit Sphere

For the unit sphere, the Laplace-Beltrami operator can be computed from the Euclidean Laplacian:

$$\Delta_S u = \Delta u - (\mathbf{n} \cdot \nabla)(\mathbf{n} \cdot \nabla u)$$

where $\mathbf{n} = \mathbf{x}$ for the unit sphere (normal equals position).

**Test solution**: $u(x,y,z) = xy$ with $\Delta_S(xy) = -2xy$

### Discrete Exterior Calculus

On triangle meshes, we build discrete differential operators:

- **B0** (gradient): Maps vertex values to edge differences
- **B1** (curl): Maps edge values to face circulations
- **Hodge stars**: Diagonal matrices encoding metric information
- **Laplacian**: $\Delta_0 = B_0^T \cdot H_1^{-1} \cdot B_0 \cdot H_0$

### Tangent Message Passing

For graph neural networks on manifolds:

1. Project neighbor positions to local tangent plane
2. Encode 2D tangent coordinates
3. Concatenate with features and apply MLP
4. Aggregate messages respecting local geometry

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8+ GB VRAM (tested on L40S, A100, RTX 3090)
- **CUDA**: 11.8+ recommended
- **RAM**: 16+ GB system memory

## Citation

If you use this code, please cite:

```bibtex
@software{geopinn_manifold,
  title={GeoPINN-Manifold: Physics-Informed Neural Networks on Curved Surfaces},
  year={2024},
  url={https://github.com/geopinn/geopinn-manifold}
}
```

## License

MIT License
