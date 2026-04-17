"""
Chart Atlas Module for Manifold Geometry
========================================

Handles manifolds without global coordinates by partitioning into overlapping charts.
Each chart provides local 2D coordinates via tangent plane projection.

Migrated from Project 01 (GeoPINN-Manifold) for use across multiple projects.

Key Components:
- Atlas: NumPy-based atlas using k-means clustering
- TorchAtlas: PyTorch GPU-compatible atlas for training
- ChartMLP: MLP operating in local chart coordinates
- AtlasPINN: PINN using chart atlas with smooth blending

Usage:
    >>> from diffgeo.charts import Atlas, ChartMLP, AtlasPINN
    >>> 
    >>> # Build atlas from point cloud
    >>> atlas = Atlas(points, normals, num_charts=6)
    >>> 
    >>> # Project points to local coordinates
    >>> local_coords = atlas.project_to_chart(points, chart_idx=0)
    >>> 
    >>> # Get blending weights for multi-chart evaluation
    >>> weights = atlas.compute_chart_weights(points)

For Earth-surface domains:
    >>> from diffgeo.charts import Atlas
    >>> atlas = Atlas.from_earth_patch(lat_min=36, lat_max=40, lon_min=-77, lon_max=-73)

For garment panels:
    >>> atlas = Atlas.from_garment_panel(flat_vertices, draped_vertices, faces)
"""

from .atlas import Atlas, TorchAtlas
from .chart_mlp import ChartMLP, AtlasPINN
from .projections import (
    tangent_plane_projection,
    stereographic_projection,
    gnomonic_projection,
    exponential_map_projection,
)

__all__ = [
    # Core classes
    "Atlas",
    "TorchAtlas",
    "ChartMLP", 
    "AtlasPINN",
    # Projection functions
    "tangent_plane_projection",
    "stereographic_projection",
    "gnomonic_projection",
    "exponential_map_projection",
]
