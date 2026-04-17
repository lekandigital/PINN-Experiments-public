"""
Mean Curvature
==============

Compute discrete mean curvature using the Laplace-Beltrami operator.

For a surface in R³, the Laplace-Beltrami of the position vector gives:
    Δx = 2Hn
    
where H is the mean curvature and n is the unit normal.

Mean curvature H = (κ₁ + κ₂)/2, where κ₁, κ₂ are principal curvatures.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def mean_curvature(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute mean curvature at each vertex.
    
    Uses Δx = 2Hn, where:
        Δ is the Laplace-Beltrami operator
        x is the position vector
        H is mean curvature
        n is the unit normal
    
    The sign convention: H > 0 for convex (sphere-like) surfaces.
    
    Args:
        mesh: TriangleMesh with precomputed Laplacian
        
    Returns:
        H: (V,) mean curvature at each vertex
    """
    # Get mean curvature vector
    H_vec = mean_curvature_vector(mesh)
    
    # Take magnitude for unsigned mean curvature
    H_mag = np.linalg.norm(H_vec, axis=1)
    
    # Determine sign by checking alignment with vertex normal
    # H > 0 if curvature vector points in same direction as normal (convex)
    dot = np.sum(H_vec * mesh.vertex_normals, axis=1)
    sign = np.sign(dot)
    sign[sign == 0] = 1  # Default to positive
    
    return sign * H_mag


def mean_curvature_vector(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute mean curvature vector at each vertex.
    
    The mean curvature vector is Δx / 2, which equals H·n.
    
    This is useful for:
    - Mean curvature flow: dx/dt = -H·n
    - Willmore energy: E = ∫ H² dA
    - Cloth simulation: bending energy
    
    Args:
        mesh: TriangleMesh with precomputed Laplacian
        
    Returns:
        H_vec: (V, 3) mean curvature vectors
    """
    if mesh.laplacian is None:
        raise ValueError("Mesh Laplacian not computed. Use TriangleMesh.from_vertices_faces()")
    
    # Apply Laplacian to each coordinate
    Lx = mesh.laplacian @ mesh.vertices
    
    # Normalize by dual area (strong form Laplacian)
    dual_areas = mesh.dual_areas[:, np.newaxis]
    Lx_strong = Lx / np.maximum(dual_areas, 1e-12)
    
    # Mean curvature vector is Δx / 2
    H_vec = Lx_strong / 2
    
    return H_vec


def mean_curvature_normal(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute mean curvature normal Hn at each vertex.
    
    This is the raw output of (1/2) Δx without separating H and n.
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        Hn: (V, 3) mean curvature normal vectors
    """
    return mean_curvature_vector(mesh)


def integrated_mean_curvature(mesh: 'TriangleMesh') -> float:
    """
    Compute integrated mean curvature ∫ H dA.
    
    For a closed surface, this relates to the total edge length
    of the convex hull (Steiner formula).
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        Total integrated mean curvature
    """
    H = mean_curvature(mesh)
    return float(np.sum(H * mesh.dual_areas))


def willmore_energy(mesh: 'TriangleMesh') -> float:
    """
    Compute Willmore energy W = ∫ H² dA.
    
    The Willmore energy measures surface bending. A sphere minimizes
    Willmore energy among all surfaces of the same topology.
    
    For a sphere of radius R: W = 4π
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        Willmore energy
    """
    H = mean_curvature(mesh)
    return float(np.sum(H**2 * mesh.dual_areas))
