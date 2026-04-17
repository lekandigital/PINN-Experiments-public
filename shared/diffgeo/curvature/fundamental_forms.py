"""
Fundamental Forms
=================

Compute the first and second fundamental forms for triangle meshes.

First Fundamental Form (I): Measures intrinsic geometry (distances, angles on surface)
    I = E du² + 2F du dv + G dv²
    where (E, F, G) are metric tensor components

Second Fundamental Form (II): Measures extrinsic geometry (how surface bends in space)
    II = L du² + 2M du dv + N dv²
    where (L, M, N) are shape operator components
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def first_fundamental_form(
    mesh: 'TriangleMesh'
) -> np.ndarray:
    """
    Compute first fundamental form (metric tensor) per face.
    
    For each face, the metric tensor maps the 2D parameter space
    to the 3D surface. Returns the 2×2 matrix [[E, F], [F, G]]
    for each face.
    
    The determinant sqrt(EG - F²) gives the area element.
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        I: (F, 2, 2) first fundamental form per face
    """
    F = mesh.faces.shape[0]
    metric = np.zeros((F, 2, 2), dtype=np.float64)
    
    for f_idx, face in enumerate(mesh.faces):
        i, j, k = face
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        
        # Edge vectors define parameterization
        e1 = vj - vi  # ∂x/∂u direction
        e2 = vk - vi  # ∂x/∂v direction
        
        # Metric components: g_ab = ∂x/∂a · ∂x/∂b
        E = np.dot(e1, e1)
        F_coef = np.dot(e1, e2)
        G = np.dot(e2, e2)
        
        metric[f_idx] = [[E, F_coef], [F_coef, G]]
    
    return metric


def second_fundamental_form(
    mesh: 'TriangleMesh'
) -> np.ndarray:
    """
    Compute second fundamental form per face.
    
    The second fundamental form measures how the surface normal
    changes, encoding the extrinsic curvature.
    
    II_ab = -∂n/∂a · ∂x/∂b = n · ∂²x/∂a∂b
    
    For a triangle mesh, we approximate using face normal and
    vertex positions.
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        II: (F, 2, 2) second fundamental form per face
    """
    F = mesh.faces.shape[0]
    shape = np.zeros((F, 2, 2), dtype=np.float64)
    
    for f_idx, face in enumerate(mesh.faces):
        i, j, k = face
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        n = mesh.face_normals[f_idx]
        
        # For a planar triangle, the second fundamental form is zero
        # We need to look at the variation of normals across the face
        
        # Approximate using vertex normals
        ni, nj, nk = mesh.vertex_normals[i], mesh.vertex_normals[j], mesh.vertex_normals[k]
        
        # Edge vectors
        e1 = vj - vi
        e2 = vk - vi
        
        # Normal differences
        dn1 = nj - ni  # ∂n/∂u approximation
        dn2 = nk - ni  # ∂n/∂v approximation
        
        # Second fundamental form: II_ab = -dn_a · e_b
        L = -np.dot(dn1, e1)
        M = -0.5 * (np.dot(dn1, e2) + np.dot(dn2, e1))
        N = -np.dot(dn2, e2)
        
        shape[f_idx] = [[L, M], [M, N]]
    
    return shape


def shape_operator(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute shape operator (Weingarten map) per face.
    
    The shape operator S = I⁻¹ II maps tangent vectors to tangent vectors
    and encodes the directional curvature.
    
    Principal curvatures are eigenvalues of S.
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        S: (F, 2, 2) shape operator per face
    """
    I = first_fundamental_form(mesh)
    II = second_fundamental_form(mesh)
    
    F = mesh.faces.shape[0]
    S = np.zeros((F, 2, 2), dtype=np.float64)
    
    for f_idx in range(F):
        # S = I^{-1} @ II
        I_inv = np.linalg.inv(I[f_idx] + 1e-10 * np.eye(2))
        S[f_idx] = I_inv @ II[f_idx]
    
    return S


def face_gaussian_curvature(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute Gaussian curvature per face using fundamental forms.
    
    K = det(II) / det(I) = (LN - M²) / (EG - F²)
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        K: (F,) Gaussian curvature per face
    """
    I = first_fundamental_form(mesh)
    II = second_fundamental_form(mesh)
    
    F = mesh.faces.shape[0]
    K = np.zeros(F, dtype=np.float64)
    
    for f_idx in range(F):
        det_I = np.linalg.det(I[f_idx])
        det_II = np.linalg.det(II[f_idx])
        
        if det_I > 1e-12:
            K[f_idx] = det_II / det_I
    
    return K


def face_mean_curvature(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute mean curvature per face using fundamental forms.
    
    H = (EN - 2FM + GL) / (2(EG - F²))
    
    Equivalently: H = trace(I⁻¹ II) / 2
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        H: (F,) mean curvature per face
    """
    S = shape_operator(mesh)
    
    # H = trace(S) / 2
    H = 0.5 * (S[:, 0, 0] + S[:, 1, 1])
    
    return H


def deformation_gradient(
    mesh_rest: 'TriangleMesh',
    mesh_deformed: 'TriangleMesh'
) -> np.ndarray:
    """
    Compute deformation gradient F for each face.
    
    The deformation gradient maps reference (rest) tangent vectors
    to deformed tangent vectors: F = ∂x_def/∂x_rest
    
    Used in cloth simulation for strain computation.
    
    Args:
        mesh_rest: Rest state mesh
        mesh_deformed: Deformed mesh (same topology)
        
    Returns:
        F: (num_faces, 3, 2) deformation gradient per face
    """
    if mesh_rest.faces.shape != mesh_deformed.faces.shape:
        raise ValueError("Meshes must have same topology")
    
    num_F = mesh_rest.faces.shape[0]
    def_grad = np.zeros((num_F, 3, 2), dtype=np.float64)
    
    for f_idx, face in enumerate(mesh_rest.faces):
        i, j, k = face
        
        # Rest edges
        vi_rest = mesh_rest.vertices[i]
        e1_rest = mesh_rest.vertices[j] - vi_rest
        e2_rest = mesh_rest.vertices[k] - vi_rest
        
        # Deformed edges
        vi_def = mesh_deformed.vertices[i]
        e1_def = mesh_deformed.vertices[j] - vi_def
        e2_def = mesh_deformed.vertices[k] - vi_def
        
        # 2D rest coordinates (flatten to 2D)
        # Using the parametric coordinates from fundamental form
        E = np.dot(e1_rest, e1_rest)
        F_coef = np.dot(e1_rest, e2_rest)
        G = np.dot(e2_rest, e2_rest)
        
        # Rest metric inverse
        det = E * G - F_coef * F_coef
        if det < 1e-12:
            continue
        
        I_inv = np.array([[G, -F_coef], [-F_coef, E]]) / det
        
        # Deformation gradient: F = [e1_def | e2_def] @ I_inv
        # This maps 2D rest coordinates to 3D deformed positions
        def_grad[f_idx] = np.column_stack([e1_def, e2_def]) @ I_inv
    
    return def_grad


def cauchy_green_strain(
    mesh_rest: 'TriangleMesh',
    mesh_deformed: 'TriangleMesh'
) -> np.ndarray:
    """
    Compute right Cauchy-Green strain tensor C = Fᵀ F for each face.
    
    The eigenvalues of C are the squared principal stretches (λ₁², λ₂²).
    
    Args:
        mesh_rest: Rest state mesh
        mesh_deformed: Deformed mesh
        
    Returns:
        C: (F, 2, 2) Cauchy-Green strain tensor per face
    """
    F = deformation_gradient(mesh_rest, mesh_deformed)
    
    num_F = F.shape[0]
    C = np.zeros((num_F, 2, 2), dtype=np.float64)
    
    for f_idx in range(num_F):
        C[f_idx] = F[f_idx].T @ F[f_idx]
    
    return C
