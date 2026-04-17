"""
Gaussian Curvature
==================

Compute discrete Gaussian curvature using the angle defect formula.

For a vertex v in a triangulated surface:
    K(v) = (2π - Σ θ_i) / A_v*
    
where θ_i are the angles at v in each adjacent face, and A_v* is the dual area.

The Gauss-Bonnet theorem states that the total Gaussian curvature equals 2π χ,
where χ is the Euler characteristic of the surface.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def gaussian_curvature(
    mesh: 'TriangleMesh',
    use_dual_area: bool = True
) -> np.ndarray:
    """
    Compute Gaussian curvature at each vertex using angle defect.
    
    K(v) = (2π - Σ θ_i) / A_v*  for interior vertices
    K(v) = (π - Σ θ_i) / A_v*   for boundary vertices
    
    Args:
        mesh: TriangleMesh with precomputed geometry
        use_dual_area: If True, normalize by dual area (gives per-area curvature).
                       If False, return angle defect only.
        
    Returns:
        K: (V,) Gaussian curvature per vertex
    """
    V = mesh.vertices.shape[0]
    
    # Compute angle sum at each vertex
    angle_sum = np.zeros(V, dtype=np.float64)
    
    for f_idx, face in enumerate(mesh.faces):
        i, j, k = face
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        
        # Angle at vertex i
        e_ij = vj - vi
        e_ik = vk - vi
        cos_i = np.dot(e_ij, e_ik) / (np.linalg.norm(e_ij) * np.linalg.norm(e_ik) + 1e-12)
        angle_i = np.arccos(np.clip(cos_i, -1, 1))
        
        # Angle at vertex j
        e_ji = vi - vj
        e_jk = vk - vj
        cos_j = np.dot(e_ji, e_jk) / (np.linalg.norm(e_ji) * np.linalg.norm(e_jk) + 1e-12)
        angle_j = np.arccos(np.clip(cos_j, -1, 1))
        
        # Angle at vertex k
        e_ki = vi - vk
        e_kj = vj - vk
        cos_k = np.dot(e_ki, e_kj) / (np.linalg.norm(e_ki) * np.linalg.norm(e_kj) + 1e-12)
        angle_k = np.arccos(np.clip(cos_k, -1, 1))
        
        angle_sum[i] += angle_i
        angle_sum[j] += angle_j
        angle_sum[k] += angle_k
    
    # Angle defect
    # Interior vertices: 2π - angle_sum
    # Boundary vertices: π - angle_sum
    defect = np.full(V, 2 * np.pi, dtype=np.float64)
    
    if len(mesh.boundary_vertices) > 0:
        defect[mesh.boundary_vertices] = np.pi
    
    defect = defect - angle_sum
    
    # Normalize by dual area
    if use_dual_area:
        K = defect / np.maximum(mesh.dual_areas, 1e-12)
    else:
        K = defect
    
    return K


def total_gaussian_curvature(mesh: 'TriangleMesh') -> float:
    """
    Compute total Gaussian curvature ∫ K dA.
    
    By the Gauss-Bonnet theorem, this equals 2π χ where χ is the Euler
    characteristic. For a closed surface of genus g:
        χ = 2 - 2g
        ∫ K dA = 4π(1 - g)
    
    For a sphere (g=0): ∫ K dA = 4π
    For a torus (g=1):  ∫ K dA = 0
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        Total Gaussian curvature (should be 2π χ)
    """
    K = gaussian_curvature(mesh, use_dual_area=False)
    return float(np.sum(K))


def verify_gauss_bonnet(
    mesh: 'TriangleMesh',
    expected_euler_characteristic: int = None,
    tol: float = 0.1
) -> dict:
    """
    Verify Gauss-Bonnet theorem: ∫ K dA = 2π χ
    
    Args:
        mesh: TriangleMesh
        expected_euler_characteristic: If provided, compare against this
        tol: Tolerance for verification
        
    Returns:
        dict with:
            'total_curvature': ∫ K dA
            'euler_characteristic_computed': χ from curvature
            'euler_characteristic_topological': V - E + F
            'passes': bool, True if theorem holds
    """
    total_K = total_gaussian_curvature(mesh)
    chi_from_curvature = total_K / (2 * np.pi)
    
    # Topological Euler characteristic
    V = mesh.num_vertices
    E = mesh.num_edges
    F = mesh.num_faces
    chi_topological = V - E + F
    
    passes = abs(chi_from_curvature - chi_topological) < tol
    
    return {
        'total_curvature': total_K,
        'euler_characteristic_computed': chi_from_curvature,
        'euler_characteristic_topological': chi_topological,
        'passes': passes,
    }
