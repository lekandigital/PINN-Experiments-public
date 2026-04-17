"""
Mesh subpackage - Triangle mesh data structures and utilities.

Provides:
- TriangleMesh: Core mesh class with DEC operator precomputation
- Mesh generation utilities (icosphere, torus, flat grid)
- Mesh I/O (OBJ, PLY formats)
- Boundary detection and handling
"""

from .trimesh import TriangleMesh
from .generation import icosphere, torus_mesh, flat_grid, cubed_sphere
from .dual import compute_dual_areas, compute_dual_edge_lengths
from .boundary import find_boundary_edges, find_boundary_vertices

__all__ = [
    "TriangleMesh",
    "icosphere",
    "torus_mesh", 
    "flat_grid",
    "cubed_sphere",
    "compute_dual_areas",
    "compute_dual_edge_lengths",
    "find_boundary_edges",
    "find_boundary_vertices",
]
