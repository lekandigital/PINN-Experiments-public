# Utility functions for HGNN-NIF-Cloth
"""
Utility functions for metrics and visualization.

Components:
- chamfer_distance: Compute Chamfer distance between point sets
- generate_grid_points: Create 3D query points from SDF grid
"""

from .metrics import chamfer_distance, generate_grid_points

__all__ = [
    "chamfer_distance",
    "generate_grid_points",
]
