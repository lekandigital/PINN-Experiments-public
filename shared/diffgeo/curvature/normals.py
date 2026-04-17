"""
Normal Computation Utilities
============================

Compute vertex and face normals with various weighting schemes.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def face_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute face normals via cross product of edges.
    
    For consistent orientation, assumes counter-clockwise vertex ordering.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices
        normalize: If True, return unit normals
        
    Returns:
        normals: (F, 3) face normal vectors
    """
    F = faces.shape[0]
    normals = np.zeros((F, 3), dtype=np.float64)
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        
        e1 = vj - vi
        e2 = vk - vi
        n = np.cross(e1, e2)
        
        if normalize:
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
            else:
                n = np.array([0., 0., 1.])
        
        normals[f_idx] = n
    
    return normals


def vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    weighting: str = 'area'
) -> np.ndarray:
    """
    Compute vertex normals by averaging adjacent face normals.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices
        weighting: 'uniform' - equal weights
                   'area' - weight by face area
                   'angle' - weight by angle at vertex
        
    Returns:
        normals: (V, 3) unit vertex normals
    """
    V = vertices.shape[0]
    F = faces.shape[0]
    normals = np.zeros((V, 3), dtype=np.float64)
    
    # Compute face normals (unnormalized for area weighting)
    f_normals = face_normals(vertices, faces, normalize=False)
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        n = f_normals[f_idx]
        
        if weighting == 'uniform':
            normals[i] += n / (np.linalg.norm(n) + 1e-12)
            normals[j] += n / (np.linalg.norm(n) + 1e-12)
            normals[k] += n / (np.linalg.norm(n) + 1e-12)
            
        elif weighting == 'area':
            # n is already proportional to face area (cross product magnitude)
            normals[i] += n
            normals[j] += n
            normals[k] += n
            
        elif weighting == 'angle':
            # Compute angles at each vertex
            e_ij = vj - vi
            e_ik = vk - vi
            e_ji = vi - vj
            e_jk = vk - vj
            e_ki = vi - vk
            e_kj = vj - vk
            
            def angle_between(e1, e2):
                cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
                return np.arccos(np.clip(cos_a, -1, 1))
            
            angle_i = angle_between(e_ij, e_ik)
            angle_j = angle_between(e_ji, e_jk)
            angle_k = angle_between(e_ki, e_kj)
            
            n_unit = n / (np.linalg.norm(n) + 1e-12)
            normals[i] += angle_i * n_unit
            normals[j] += angle_j * n_unit
            normals[k] += angle_k * n_unit
        
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = normals / norms
    
    return normals


def area_weighted_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute vertex normals with area weighting.
    
    Alias for vertex_normals(..., weighting='area').
    """
    return vertex_normals(vertices, faces, weighting='area')


def angle_weighted_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute vertex normals with angle weighting.
    
    This is generally more accurate for visualization and
    produces smoother normals at vertices with unequal face sizes.
    """
    return vertex_normals(vertices, faces, weighting='angle')
