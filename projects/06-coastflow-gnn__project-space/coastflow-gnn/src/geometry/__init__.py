"""
Geometry Subpackage for CoastFlow-GNN

Provides discrete differential geometry operators using the shared diffgeo module.
Includes both graph-based operators (for general point clouds) and mesh-based
operators (for triangulated domains).

The mesh-based operators use cotangent-weighted Laplace-Beltrami which provides
better accuracy for:
- Spherical Earth geometry
- Variable mesh resolution near coastlines
- Accurate diffusion modeling
"""

from .diffgeo_operators import (
    MeshDifferentialOperators,
    create_mesh_from_graph,
    spherical_laplacian,
    SphereGeometry,
)

__all__ = [
    "MeshDifferentialOperators",
    "create_mesh_from_graph",
    "spherical_laplacian",
    "SphereGeometry",
]
