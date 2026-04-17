# Shared Collision Module

Unified SDF-based collision system for all cloth simulation projects in PINN-Experiments.

## Update (April 17, 2026)

- The shared stack now includes body interfaces, SDF fields, detection, response, loss, and mesh utilities with dedicated tests.
- Cloth projects are expected to integrate here instead of carrying separate collision logic in each project tree.
- This README update aligns the module with the repo's newer shared-training and export flows.

## Installation

The module has minimal dependencies:
- `torch >= 2.0.0`
- `numpy >= 1.24.0`  
- `scipy >= 1.10.0`
- `scikit-image >= 0.21.0` (for marching cubes)

## Quick Start

```python
from shared.collision import (
    DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss, SDFField
)

# 1. Setup body with SDF
body = DeformableBody(body_vertices, body_faces, sdf_resolution=128)

# 2. Create detector and response
detector = CollisionDetector(proximity_threshold=0.005)
response = CollisionResponse(stiffness=1000.0, friction=0.3, damping=0.1)
loss_fn = CollisionLoss(weights={'penetration': 10.0, 'proximity': 1.0})

# 3. Per-frame collision handling
body.update_from_deformation(new_body_vertices)  # or update_from_skeleton()
collisions = detector.detect(cloth_vertices, body.get_sdf())
corrected_vertices = response.resolve_positions(cloth_vertices, collisions)

# 4. For training
training_loss = loss_fn(cloth_vertices, body.get_sdf())
training_loss.backward()
```

## Components

### SDFField
Signed distance field from triangle mesh with differentiable queries.

```python
sdf = SDFField.from_mesh(vertices, faces, resolution=128)
distances = sdf.query(points)  # (N,) signed distances
gradients = sdf.gradient(points)  # (N, 3) gradient vectors
```

### CollisionDetector  
Detects penetrating and proximal vertices.

```python
detector = CollisionDetector(proximity_threshold=0.005)
result = detector.detect(cloth_vertices, body_sdf)
# result.penetrating_mask, result.penetration_depths, result.surface_normals
```

### CollisionResponse
Position-based and force-based collision resolution with friction.

```python
response = CollisionResponse(stiffness=1000.0, friction=0.3)

# Position-based (for PBD-style models)
corrected = response.resolve_positions(vertices, result)

# Force-based (for physics simulations)
forces = response.resolve_forces(vertices, result)
```

### CollisionLoss
Differentiable losses for training: penetration, proximity, contact, eikonal.

```python
loss_fn = CollisionLoss(weights={
    'penetration': 10.0,
    'proximity': 1.0,
    'contact': 0.5,
    'eikonal': 0.1
})
loss = loss_fn(cloth_vertices, body_sdf)
```

### DeformableBody
Wraps body mesh with auto-updating SDF for PEGNN integration.

```python
body = DeformableBody(rest_vertices, faces, sdf_resolution=128)
body.update_from_deformation(pegnn_output)  # Updates SDF
# or
body.update_from_skeleton(joint_transforms, skinning_weights)  # LBS fallback
```

## Performance

| Operation | Time (128³, 10K vertices) |
|-----------|---------------------------|
| SDF from mesh | ~30-50ms |
| Query 10K points | <1ms |
| Full collision pipeline | <5ms |

## Project Integration

Each cloth project has a `collision_integration.py` file showing how to wire in collision:

- **Project 09 (HGNN-NIF-Cloth)**: Position-based + training loss
- **Project 14 (PEGNN-Deform)**: Body SDF export wrapper  
- **Project 05 (ClothGNN)**: Force-based via `resolve_forces()`
- **Project 08 (HGNN-ClothDyn)**: Collision forces + EdgeForceConv
- **Project 11 (NIF-Cloth3D)**: Blender addon panel
- **Project 13 (NIF-Cloth4D)**: Lightweight 64³ SDF

## Running Tests

```bash
cd /Users/lekan/Dev/PINN-Experiments
pytest shared/collision/tests/ -v
```

## API Reference

### Configuration

```python
from shared.collision import CollisionConfig, SDFConfig

config = CollisionConfig(
    proximity_threshold=0.005,
    stiffness=1000.0,
    friction=0.3,
    damping=0.1,
    loss_weights={'penetration': 10.0, 'proximity': 1.0},
    sdf=SDFConfig(resolution=128)
)
```

### Loss Functions

Individual loss functions are also available:

```python
from shared.collision import penetration_loss, proximity_loss, contact_loss, eikonal_loss

# Direct usage
pen = penetration_loss(vertices, sdf)
prox = proximity_loss(vertices, sdf, margin=0.01)
cont = contact_loss(vertices, sdf, contact_mask)
eik = eikonal_loss(gradients)
```

### Mesh Utilities

```python
from shared.collision import marching_cubes_mesh, laplacian_smooth

# Extract mesh from SDF
vertices, faces = marching_cubes_mesh(sdf_grid, bbox_min, bbox_max)

# Smooth mesh
smoothed = laplacian_smooth(vertices, faces, iterations=3)
```
