# Physics-Encoded Graph Convolution Library

A shared library of physics-informed graph neural network layers for the PINN-Experiments repository.

## Update (April 17, 2026)

- `shared/` now sits alongside broader common stacks such as collision, diffgeo, export, pipeline, and trajectory helpers.
- Cross-project reuse expanded around cloth, coastal, and deform workflows instead of isolated per-project utilities.
- Treat this library as part of the shared infrastructure story rather than a one-off side package.

## Design Principle

In standard GNN message passing, the network learns the **entire** message function from data. In physics-encoded convolution, we **hard-code** known physical laws as the primary message function, and the network learns **only corrections/residuals**.

```
Standard GNN:     message = MLP(x_i, x_j, edge_attr)           ← learns EVERYTHING
Physics-Encoded:  message = PHYSICS(x_i, x_j) + MLP(...)       ← learns CORRECTIONS
```

This dramatically improves:
- **Data efficiency**: The network doesn't waste capacity learning F=ma
- **Physical plausibility**: Base predictions always satisfy known laws
- **Generalization**: Analytical terms extrapolate correctly
- **Training stability**: Smoother loss landscape with smaller corrections

## Installation

The library is part of the PINN-Experiments repository. Ensure you have PyTorch and optionally PyTorch Geometric installed:

```bash
pip install torch torch_geometric torch_scatter
```

Then import from the shared library:

```python
import sys
sys.path.insert(0, '/path/to/PINN-Experiments')

from shared.physics_conv import (
    ClothForceConv,
    ElasticForceConv,
    LiteClothConv,
    ShallowWaterConv,
)
```

## Available Layers

### ClothForceConv

Physics-encoded convolution for cloth simulation using Hooke's law.

```python
from shared.physics_conv import ClothForceConv, ClothConvConfig

config = ClothConvConfig(
    stretch_stiffness=1000.0,
    compute_damping=True,
    damping_coefficient=0.1,
)
conv = ClothForceConv(config)

# Forward pass
forces = conv(
    x=positions,           # [num_nodes, 3]
    edge_index=edges,      # [2, num_edges]
    edge_attr=rest_lengths, # [num_edges, 1]
    vel=velocities,        # [num_nodes, 3] optional
)
```

**Physics encoded**: 
- Stretch forces: `F = k * (|x_j - x_i| - L0) / L0 * direction`
- Damping: `F_damp = -d * (v_rel · edge_dir) * edge_dir`
- Optional bending forces (via `ClothForceConvWithBending`)

**What the network learns**:
- Nonlinear material response at large strains
- Anisotropic effects (warp vs weft)
- Self-collision response

### ElasticForceConv

Physics-encoded convolution for 3D elastic deformation.

```python
from shared.physics_conv import ElasticForceConv, ElasticConvConfig

config = ElasticConvConfig(
    youngs_modulus=1e6,
    use_velocity_gru=True,  # Temporal state tracking
)
conv = ElasticForceConv(config)

forces, hidden = conv(
    x=positions,
    edge_index=edges,
    edge_attr=material_props,  # [rest_length, stiffness]
    vel=velocities,
    hidden=gru_hidden,
    return_hidden=True,
)
```

**Physics encoded**: 3D Hooke's law `F = k * (d - L0) * direction`

**Features**:
- Per-edge material properties
- Optional VelocityGRU for temporal dynamics
- Integration with time steppers

### LiteClothConv

Lightweight version for models with tight parameter budgets (< 80K params).

```python
from shared.physics_conv import LiteClothConv, PurePhysicsClothConv

# With minimal correction (~3K params)
lite = LiteClothConv()

# Pure physics, no learnable params
pure = PurePhysicsClothConv(stiffness=1000.0)
```

**Simplifications**:
- No bending forces (stretch only)
- Single-layer correction MLP
- No damping computation
- Fixed global stiffness

### ShallowWaterConv

Physics-encoded convolution for shallow water equations.

```python
from shared.physics_conv import ShallowWaterConv, ShallowWaterConfig

config = ShallowWaterConfig(
    gravity=9.81,
    drag_coefficient=0.003,
    enforce_mass_conservation=True,
)
conv = ShallowWaterConv(config)

# Node state: [η, u, v, h_bathy]
dstate_dt = conv(
    x=state,           # [num_nodes, 4]
    edge_index=edges,
    edge_attr=edge_props,  # [length, normal_x, normal_y, type]
)
```

**Physics encoded**:
- Pressure gradient: `F = -g * ∇η`
- Bottom friction: `F = -C_d * |u| * u / h`
- Mass/momentum fluxes
- Mass conservation projection

## Time Integrators

The library includes several time integration schemes:

```python
from shared.physics_conv import ExplicitEuler, SemiImplicitEuler, VelocityVerlet

# Create integrator
integrator = VelocityVerlet(damping=0.99)

# Step forward in time
new_pos, new_vel = integrator.step(pos, vel, acc, dt=0.01)
```

| Integrator | Order | Energy | Best For |
|------------|-------|--------|----------|
| `ExplicitEuler` | 1 | Grows | Quick prototyping |
| `SemiImplicitEuler` | 1 | Bounded | General physics |
| `VelocityVerlet` | 2 | Conserved | Cloth, springs |

## Diagnostics

All physics-encoded layers provide diagnostic utilities:

```python
conv = ClothForceConv()
conv.train()

# Forward pass
forces = conv(x, edge_index, edge_attr)

# Check physics fraction (should be > 0.7 at convergence)
pf = conv.physics_fraction()
print(f"Physics fraction: {pf:.2f}")

# Check correction magnitude (should start small, grow slowly)
cm = conv.correction_magnitude()
print(f"Correction magnitude: {cm:.4f}")
```

### Interpreting Diagnostics

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| `physics_fraction` | > 0.7 | < 0.5 means network overriding physics |
| `correction_magnitude` | Starts ~0, grows slowly | Explosion indicates unstable training |

## Creating New Subclasses

To add a new physics domain:

1. **Inherit from `PhysicsEncodedConv`**:

```python
from shared.physics_conv.base import PhysicsEncodedConv, PhysicsConvConfig

class MyPhysicsConv(PhysicsEncodedConv):
    def __init__(self, config=None):
        super().__init__(config or PhysicsConvConfig())
        # Build correction MLP
        self._build_correction_mlp(input_dim=10)
    
    def physics_name(self) -> str:
        return "My Custom Physics"
    
    def compute_edge_physics(self, x_i, x_j, edge_attr=None, **kwargs):
        """
        MUST NOT have learnable parameters!
        Encode your known physics here.
        """
        # Example: gravitational force
        diff = x_j - x_i
        dist = diff.norm(dim=-1, keepdim=True)
        direction = diff / (dist + 1e-8)
        
        G = 6.67e-11
        m_i, m_j = x_i[:, 3:4], x_j[:, 3:4]  # Mass in feature
        force = G * m_i * m_j / (dist ** 2) * direction
        
        return force
```

2. **Test your implementation**:

```python
def test_my_physics():
    conv = MyPhysicsConv()
    
    # Verify no params in physics
    x = torch.randn(100, 4)
    physics_msg = conv.compute_edge_physics(x[:50], x[50:])
    
    # Verify physics fraction is high
    conv.train()
    _ = conv(x, edge_index, edge_attr)
    assert conv.physics_fraction() > 0.9
```

3. **Add to `__init__.py`**:

```python
from .my_conv import MyPhysicsConv
__all__ = [..., "MyPhysicsConv"]
```

## Configuration

All layers accept a configuration dataclass:

```python
from dataclasses import dataclass
from shared.physics_conv.base import PhysicsConvConfig

@dataclass
class PhysicsConvConfig:
    correction_hidden_dim: int = 64      # MLP hidden dimension
    correction_layers: int = 2            # Number of MLP layers
    correction_activation: str = "silu"   # Activation function
    combine_mode: str = "additive"        # "additive" or "multiplicative"
    correction_scale_init: float = 0.01   # Initial output scale
    aggregation: str = "sum"              # "sum", "mean", "attention"
    track_diagnostics: bool = True
```

## Running Tests

```bash
cd /path/to/PINN-Experiments
pytest shared/tests/ -v
```

Or run individual test files:

```bash
pytest shared/tests/test_cloth_conv.py -v
pytest shared/tests/test_integrators.py -v
```

## Projects Using This Library

| Project | Layer | Physics |
|---------|-------|---------|
| 05-ClothGNN | `LiteClothConv` | Spring forces |
| 06-CoastFlow-GNN | `ShallowWaterConv` | Shallow water eqs |
| 08-HGNN-ClothDyn | `ClothForceConv` | Hooke's law + damping |
| 09-HGNN-NIF-Cloth | `ClothForceConv` | Spring forces |
| 14-PEGNN-Deform | `ElasticForceConv` | 3D elasticity |

## References

- Physics-Informed Neural Networks: [Raissi et al., 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Graph Neural Networks for Physics: [Sanchez-Gonzalez et al., 2020](https://arxiv.org/abs/2002.09405)
- Learning to Simulate: [Pfaff et al., 2021](https://arxiv.org/abs/2010.03409)
