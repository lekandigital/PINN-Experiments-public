"""
Discrete Gradient Operator
==========================

Computes the discrete gradient (exterior derivative d0) that maps
vertex scalar fields (0-forms) to edge-based 1-forms.

Mathematical definition:
    (d0 f)[e] = f[j] - f[i]  for edge e = (i, j) with i < j

This is the discrete analog of the gradient projected onto edge directions.
"""

import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def discrete_gradient(
    mesh: 'TriangleMesh',
    f: np.ndarray
) -> np.ndarray:
    """
    Apply discrete gradient to a vertex scalar field.
    
    Computes the discrete exterior derivative d0, mapping vertex values
    to edge differences: (∇f)[edge] = f[j] - f[i].
    
    Args:
        mesh: TriangleMesh with precomputed d0 operator
        f: (V,) or (V, C) scalar or vector field on vertices
        
    Returns:
        df: (E,) or (E, C) gradient values on edges
    """
    if mesh.d0 is None:
        raise ValueError("Mesh d0 operator not computed. Use TriangleMesh.from_vertices_faces()")
    
    f = np.asarray(f)
    
    if f.ndim == 1:
        return mesh.d0 @ f
    else:
        # Multi-channel: apply to each channel
        return mesh.d0 @ f


def build_gradient_matrix(
    edges: np.ndarray,
    num_vertices: int
) -> sparse.csr_matrix:
    """
    Build the discrete gradient (d0) matrix from edge list.
    
    The gradient matrix has shape (E, V) where:
        d0[e, i] = -1  (source vertex of edge e)
        d0[e, j] = +1  (target vertex of edge e)
    
    For edge e = (i, j) with i < j, (d0 @ f)[e] = f[j] - f[i].
    
    Args:
        edges: (E, 2) edge vertex indices (each row [i, j] with i < j)
        num_vertices: Total number of vertices V
        
    Returns:
        d0: (E, V) sparse matrix
    """
    E = edges.shape[0]
    
    rows = np.repeat(np.arange(E), 2)  # [0, 0, 1, 1, 2, 2, ...]
    cols = edges.ravel()  # [i0, j0, i1, j1, i2, j2, ...]
    data = np.tile([-1.0, 1.0], E)  # [-1, 1, -1, 1, ...]
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(E, num_vertices))


def gradient_per_face(
    mesh: 'TriangleMesh',
    f: np.ndarray
) -> np.ndarray:
    """
    Compute gradient vectors per face (in 3D ambient space).
    
    For each face, computes the gradient of the piecewise-linear
    interpolant defined by vertex values.
    
    Formula for face k with vertices (i, j, l), area A, and normal N:
        ∇f = (1/2A) * Σ f[v] * (N × edge_opposite_to_v)
    
    Args:
        mesh: TriangleMesh with precomputed geometry
        f: (V,) scalar field on vertices
        
    Returns:
        grad_f: (F, 3) gradient vectors per face (in 3D)
    """
    f = np.asarray(f)
    F = mesh.faces.shape[0]
    grad_f = np.zeros((F, 3), dtype=np.float64)
    
    for face_idx in range(F):
        i, j, k = mesh.faces[face_idx]
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        
        # Edge vectors (opposite to each vertex)
        e_i = vk - vj  # opposite to vertex i
        e_j = vi - vk  # opposite to vertex j
        e_k = vj - vi  # opposite to vertex k
        
        # Face normal (unnormalized, magnitude = 2*area)
        N = np.cross(vj - vi, vk - vi)
        area = 0.5 * np.linalg.norm(N)
        
        if area < 1e-12:
            continue
        
        N_unit = N / (2 * area)
        
        # Gradient formula: (1/2A) * Σ f[v] * (N × e_opposite)
        # Note: N × e rotates e by 90° in the tangent plane
        grad_f[face_idx] = (
            f[i] * np.cross(N_unit, e_i) +
            f[j] * np.cross(N_unit, e_j) +
            f[k] * np.cross(N_unit, e_k)
        ) / (2 * area)
    
    return grad_f
