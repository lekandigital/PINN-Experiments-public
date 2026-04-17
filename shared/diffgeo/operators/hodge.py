"""
Hodge Star Operators
====================

Discrete Hodge star operators for converting between primal and dual forms.

In DEC, the Hodge star maps k-forms to (n-k)-forms where n is the surface
dimension (n=2 for triangle meshes). On surfaces:
    ★0: 0-forms (vertex) → 2-forms (vertex dual)  [multiply by dual area]
    ★1: 1-forms (edge) → 1-forms (edge dual)      [multiply by dual/primal ratio]
    ★2: 2-forms (face) → 0-forms (face dual)      [multiply by 1/face area]
"""

import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def hodge_star_0(
    mesh: 'TriangleMesh',
    f: np.ndarray
) -> np.ndarray:
    """
    Apply Hodge star ★0 to a 0-form (vertex values).
    
    Multiplies by dual cell areas: (★0 f)_i = A_i* f_i
    
    Args:
        mesh: TriangleMesh with precomputed dual areas
        f: (V,) 0-form on vertices
        
    Returns:
        star_f: (V,) Hodge-star transformed
    """
    f = np.asarray(f)
    return mesh.dual_areas * f


def hodge_star_1(
    mesh: 'TriangleMesh',
    omega: np.ndarray
) -> np.ndarray:
    """
    Apply Hodge star ★1 to a 1-form (edge values).
    
    For cotangent DEC, uses cotangent weights.
    
    Args:
        mesh: TriangleMesh with precomputed cotangent weights
        omega: (E,) 1-form on edges
        
    Returns:
        star_omega: (E,) Hodge-star transformed
    """
    omega = np.asarray(omega)
    return mesh.cotangent_weights * omega


def hodge_star_2(
    mesh: 'TriangleMesh',
    sigma: np.ndarray
) -> np.ndarray:
    """
    Apply Hodge star ★2 to a 2-form (face values).
    
    Multiplies by inverse face areas: (★2 σ)_f = σ_f / A_f
    
    Args:
        mesh: TriangleMesh with precomputed face areas
        sigma: (F,) 2-form on faces
        
    Returns:
        star_sigma: (F,) Hodge-star transformed
    """
    sigma = np.asarray(sigma)
    return sigma / np.maximum(mesh.face_areas, 1e-12)


def hodge_star_0_inverse(
    mesh: 'TriangleMesh',
    f: np.ndarray
) -> np.ndarray:
    """
    Apply inverse Hodge star ★0⁻¹.
    
    Divides by dual cell areas.
    """
    f = np.asarray(f)
    return f / np.maximum(mesh.dual_areas, 1e-12)


def hodge_star_1_inverse(
    mesh: 'TriangleMesh',
    omega: np.ndarray
) -> np.ndarray:
    """
    Apply inverse Hodge star ★1⁻¹.
    
    Divides by cotangent weights.
    """
    omega = np.asarray(omega)
    return omega / np.maximum(mesh.cotangent_weights, 1e-12)


def hodge_star_2_inverse(
    mesh: 'TriangleMesh',
    sigma: np.ndarray
) -> np.ndarray:
    """
    Apply inverse Hodge star ★2⁻¹.
    
    Multiplies by face areas.
    """
    sigma = np.asarray(sigma)
    return sigma * mesh.face_areas


def build_hodge_matrices(
    dual_areas: np.ndarray,
    cotangent_weights: np.ndarray,
    face_areas: np.ndarray
) -> tuple:
    """
    Build sparse diagonal Hodge star matrices.
    
    Args:
        dual_areas: (V,) dual cell areas
        cotangent_weights: (E,) cotangent weights
        face_areas: (F,) face areas
        
    Returns:
        star0: (V, V) diagonal Hodge star on 0-forms
        star1: (E, E) diagonal Hodge star on 1-forms
        star2: (F, F) diagonal Hodge star on 2-forms
    """
    star0 = sparse.diags(dual_areas, format='dia')
    star1 = sparse.diags(cotangent_weights, format='dia')
    star2 = sparse.diags(1.0 / np.maximum(face_areas, 1e-12), format='dia')
    
    return star0, star1, star2
