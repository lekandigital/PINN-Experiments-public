"""
Discrete Curl Operator
======================

Computes the discrete curl (exterior derivative d1) that maps
edge-based 1-forms to face-based 2-forms.

Mathematical definition:
    (d1 ω)[f] = Σ_{e ∈ ∂f} ±ω[e]

The sign depends on whether the edge orientation matches the face winding.
"""

import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def discrete_curl(
    mesh: 'TriangleMesh',
    omega: np.ndarray
) -> np.ndarray:
    """
    Apply discrete curl (d1) to an edge 1-form.
    
    Computes the circulation of the 1-form around each face.
    For a conservative (curl-free) field, this should be zero.
    
    Args:
        mesh: TriangleMesh with precomputed d1 operator
        omega: (E,) or (E, C) 1-form values on edges
        
    Returns:
        curl_omega: (F,) or (F, C) curl values on faces
    """
    if mesh.d1 is None:
        raise ValueError("Mesh d1 operator not computed. Use TriangleMesh.from_vertices_faces()")
    
    omega = np.asarray(omega)
    return mesh.d1 @ omega


def build_curl_matrix(
    faces: np.ndarray,
    edges: np.ndarray,
    edge_to_idx: Dict[Tuple[int, int], int]
) -> sparse.csr_matrix:
    """
    Build the discrete curl (d1) matrix.
    
    The curl matrix has shape (F, E) where:
        d1[f, e] = +1  if edge e appears with positive orientation in face f
        d1[f, e] = -1  if edge e appears with negative orientation in face f
        d1[f, e] = 0   if edge e is not in face f
    
    Args:
        faces: (F, 3) face vertex indices
        edges: (E, 2) edge vertex indices (each row [i, j] with i < j)
        edge_to_idx: Dict mapping (i, j) tuple to edge index
        
    Returns:
        d1: (F, E) sparse matrix
    """
    F = faces.shape[0]
    E = edges.shape[0]
    
    rows, cols, data = [], [], []
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        
        # Three edges of face, in winding order
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            e_idx = edge_to_idx[edge]
            
            # Sign: +1 if (a, b) matches canonical order (a < b)
            # This means edge goes from a to b in the face winding
            sign = 1.0 if a < b else -1.0
            
            rows.append(f_idx)
            cols.append(e_idx)
            data.append(sign)
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(F, E))


def verify_exactness(mesh: 'TriangleMesh') -> float:
    """
    Verify that d1 ∘ d0 = 0 (curl of gradient is zero).
    
    This is a fundamental property of the DEC complex.
    Returns the Frobenius norm of d1 @ d0, which should be ~0.
    
    Args:
        mesh: TriangleMesh with precomputed operators
        
    Returns:
        error: ||d1 @ d0||_F (should be ~0)
    """
    if mesh.d0 is None or mesh.d1 is None:
        raise ValueError("Mesh operators not computed.")
    
    d1_d0 = mesh.d1 @ mesh.d0
    return sparse.linalg.norm(d1_d0, 'fro')
