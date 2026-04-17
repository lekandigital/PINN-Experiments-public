"""
Unified SDF-Based Collision System for Cloth Simulation.

This module provides a shared collision detection and response system
for all cloth simulation projects in PINN-Experiments.

Usage:
    from shared.collision import DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss, SDFField

    # Setup (once at initialization)
    body = DeformableBody(body_vertices, body_faces, sdf_resolution=128)
    detector = CollisionDetector(proximity_threshold=0.005)
    response = CollisionResponse(stiffness=1000.0, friction=0.3, damping=0.1)
    loss_fn = CollisionLoss(weights={'penetration': 10.0, 'proximity': 1.0})

    # Per-frame usage (in simulation loop or training step)
    body.update_from_deformation(pegnn_output_vertices)
    collisions = detector.detect(cloth_vertices, body.get_sdf())
    corrected_vertices = response.resolve_positions(cloth_vertices, collisions)
    training_loss = loss_fn(cloth_vertices, body.get_sdf())
"""

from .config import CollisionConfig, SDFConfig
from .sdf_field import SDFField
from .detection import CollisionDetector, CollisionResult
from .response import CollisionResponse
from .losses import CollisionLoss, penetration_loss, proximity_loss, contact_loss, eikonal_loss
from .body_interface import DeformableBody
from .mesh_utils import marching_cubes_mesh, laplacian_smooth, compute_mesh_sdf_grid

__all__ = [
    # Config
    'CollisionConfig',
    'SDFConfig',
    # Core classes
    'SDFField',
    'CollisionDetector',
    'CollisionResult',
    'CollisionResponse',
    'CollisionLoss',
    'DeformableBody',
    # Loss functions
    'penetration_loss',
    'proximity_loss',
    'contact_loss',
    'eikonal_loss',
    # Mesh utilities
    'marching_cubes_mesh',
    'laplacian_smooth',
    'compute_mesh_sdf_grid',
]

__version__ = '0.1.0'
