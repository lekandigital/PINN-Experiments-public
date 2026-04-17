"""
Discrete Divergence Operator
============================

Computes the discrete divergence (codifferential δ0) that maps
edge-based 1-forms back to vertex scalar fields.

The divergence is the adjoint of the gradient with respect to the
L2 inner products weighted by the Hodge stars:
    ⟨d0 f, ω⟩₁ = ⟨f, δ0 ω⟩₀

Mathematical formula:
    δ0 = ★0⁻¹ d0ᵀ ★1
"""

import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def discrete_divergence(
    mesh: 'TriangleMesh',
    omega: np.ndarray,
    use_mass_normalization: bool = True
) -> np.ndarray:
    """
    Apply discrete divergence to an edge 1-form.
    
    Computes the discrete codifferential δ0 = ★0⁻¹ d0ᵀ ★1, mapping
    edge values to vertex values.
    
    Args:
        mesh: TriangleMesh with precomputed operators
        omega: (E,) or (E, C) 1-form values on edges
        use_mass_normalization: If True, divide by vertex areas (strong form).
                                If False, return integrated values (weak form).
        
    Returns:
        div_omega: (V,) or (V, C) divergence at vertices
    """
    if mesh.d0 is None or mesh.star1 is None:
        raise ValueError("Mesh operators not computed. Use TriangleMesh.from_vertices_faces()")
    
    omega = np.asarray(omega)
    
    # Compute d0ᵀ @ (★1 @ omega)
    # ★1 is diagonal with cotangent weights
    star1_omega = mesh.star1 @ omega
    div_weak = mesh.d0.T @ star1_omega
    
    if use_mass_normalization:
        # Strong form: divide by dual areas (★0)
        dual_areas = mesh.dual_areas
        div_strong = div_weak / np.maximum(dual_areas, 1e-12)
        return div_strong
    else:
        return div_weak


def build_divergence_matrix(
    d0: sparse.csr_matrix,
    star0_inv: np.ndarray,
    star1: sparse.dia_matrix
) -> sparse.csr_matrix:
    """
    Build the discrete divergence matrix.
    
    The divergence matrix is: div = diag(★0⁻¹) @ d0ᵀ @ ★1
    
    Args:
        d0: (E, V) gradient matrix
        star0_inv: (V,) inverse of vertex dual areas
        star1: (E, E) diagonal Hodge star on 1-forms
        
    Returns:
        div: (V, E) sparse divergence matrix
    """
    # d0.T @ star1 gives (V, E) matrix
    div_weak = d0.T @ star1
    
    # Multiply by diagonal star0_inv
    div = sparse.diags(star0_inv) @ div_weak
    
    return div


def divergence_from_face_vectors(
    mesh: 'TriangleMesh',
    X: np.ndarray,
    use_mass_normalization: bool = True
) -> np.ndarray:
    """
    Compute divergence of a vector field defined per face.
    
    Given a vector field X defined at face centroids, computes its
    divergence at vertices using the integral formula:
    
        (div X)_i = (1/A_i*) ∫_{∂Ω_i*} X · n ds
    
    Approximated by summing flux through dual cell edges.
    
    Args:
        mesh: TriangleMesh
        X: (F, 3) vector field defined per face
        use_mass_normalization: If True, normalize by dual area
        
    Returns:
        div_X: (V,) divergence at vertices
    """
    X = np.asarray(X)
    V = mesh.vertices.shape[0]
    div_X = np.zeros(V, dtype=np.float64)
    
    # For each face, distribute divergence contribution to vertices
    for f_idx, face in enumerate(mesh.faces):
        i, j, k = face
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        X_f = X[f_idx]
        
        # Edge vectors
        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk
        
        # Outward normal to each edge in the tangent plane
        # For edge e, the dual edge normal points toward the opposite vertex
        N_face = np.cross(e_ij, -e_ki)
        N_norm = np.linalg.norm(N_face)
        if N_norm < 1e-12:
            continue
        N_face = N_face / N_norm
        
        # Rotate each edge by 90° in tangent plane to get outward normals
        n_ij = np.cross(N_face, e_ij)  # points toward k
        n_jk = np.cross(N_face, e_jk)  # points toward i
        n_ki = np.cross(N_face, e_ki)  # points toward j
        
        # Flux through half-edges (each edge contributes to 2 vertices)
        # This is simplified - proper DEC would use circumcenter decomposition
        flux_ij = np.dot(X_f, n_ij) / 6  # Divided among 2 vertices × 3 edges
        flux_jk = np.dot(X_f, n_jk) / 6
        flux_ki = np.dot(X_f, n_ki) / 6
        
        div_X[i] += flux_jk  # n_jk points toward i
        div_X[j] += flux_ki  # n_ki points toward j
        div_X[k] += flux_ij  # n_ij points toward k
    
    if use_mass_normalization:
        div_X = div_X / np.maximum(mesh.dual_areas, 1e-12)
    
    return div_X
