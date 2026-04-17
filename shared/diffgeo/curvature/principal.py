"""
Principal Curvatures
====================

Compute principal curvatures κ₁, κ₂ and principal directions at each vertex.

Principal curvatures are the eigenvalues of the shape operator (Weingarten map).
Mean curvature H = (κ₁ + κ₂)/2
Gaussian curvature K = κ₁ · κ₂
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def principal_curvatures(
    mesh: 'TriangleMesh',
    method: str = 'cubic_fit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal curvatures at each vertex.
    
    Args:
        mesh: TriangleMesh
        method: 'cubic_fit' (fit quadric to neighborhood) or
                'from_HK' (derive from mean and Gaussian curvature)
        
    Returns:
        k1: (V,) maximum principal curvature
        k2: (V,) minimum principal curvature
    """
    if method == 'from_HK':
        return _principal_from_HK(mesh)
    elif method == 'cubic_fit':
        return _principal_from_quadric_fit(mesh)
    else:
        raise ValueError(f"Unknown method: {method}")


def _principal_from_HK(mesh: 'TriangleMesh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal curvatures from mean and Gaussian curvature.
    
    κ₁ = H + sqrt(H² - K)
    κ₂ = H - sqrt(H² - K)
    """
    from .gaussian import gaussian_curvature
    from .mean import mean_curvature
    
    H = mean_curvature(mesh)
    K = gaussian_curvature(mesh)
    
    # Discriminant (can be negative for saddle points due to discretization)
    discriminant = H**2 - K
    discriminant = np.maximum(discriminant, 0)  # Clamp
    
    sqrt_disc = np.sqrt(discriminant)
    
    k1 = H + sqrt_disc  # Maximum curvature
    k2 = H - sqrt_disc  # Minimum curvature
    
    return k1, k2


def _principal_from_quadric_fit(mesh: 'TriangleMesh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal curvatures by fitting a quadric to the local neighborhood.
    
    For each vertex:
    1. Collect 1-ring neighborhood
    2. Express neighbors in local tangent coordinates (u, v)
    3. Fit quadric h = a*u² + b*u*v + c*v²
    4. Eigenvalues of Hessian matrix [[2a, b], [b, 2c]] give principal curvatures
    """
    V = mesh.vertices.shape[0]
    k1 = np.zeros(V, dtype=np.float64)
    k2 = np.zeros(V, dtype=np.float64)
    
    for v_idx in range(V):
        # Get 1-ring neighborhood
        neighbor_faces = mesh.vertex_faces[v_idx]
        neighbor_set = set()
        for f_idx in neighbor_faces:
            for vn in mesh.faces[f_idx]:
                if vn != v_idx:
                    neighbor_set.add(vn)
        
        if len(neighbor_set) < 3:
            # Not enough neighbors for fitting
            k1[v_idx] = k2[v_idx] = 0
            continue
        
        neighbors = list(neighbor_set)
        
        # Local coordinate system at vertex
        p_center = mesh.vertices[v_idx]
        n = mesh.vertex_normals[v_idx]
        
        # Tangent basis
        t1, t2 = _tangent_basis(n)
        
        # Express neighbors in local coordinates
        u_coords = []
        v_coords = []
        h_coords = []
        
        for nb in neighbors:
            diff = mesh.vertices[nb] - p_center
            u = np.dot(diff, t1)
            v = np.dot(diff, t2)
            h = np.dot(diff, n)  # Height above tangent plane
            u_coords.append(u)
            v_coords.append(v)
            h_coords.append(h)
        
        u_coords = np.array(u_coords)
        v_coords = np.array(v_coords)
        h_coords = np.array(h_coords)
        
        # Fit quadric: h = a*u² + b*u*v + c*v²
        # Least squares: [u², uv, v²] @ [a, b, c]ᵀ = h
        A = np.column_stack([u_coords**2, u_coords*v_coords, v_coords**2])
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, h_coords, rcond=None)
            a, b, c = coeffs
        except:
            k1[v_idx] = k2[v_idx] = 0
            continue
        
        # Shape operator (Hessian of quadric)
        # S = [[2a, b], [b, 2c]]
        S = np.array([[2*a, b], [b, 2*c]])
        
        # Eigenvalues are principal curvatures
        eigvals = np.linalg.eigvalsh(S)
        k2[v_idx] = eigvals[0]  # Minimum
        k1[v_idx] = eigvals[1]  # Maximum
    
    return k1, k2


def _tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthonormal tangent basis from normal."""
    # Choose reference not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt
    t1 = ref - np.dot(ref, normal) * normal
    t1 = t1 / (np.linalg.norm(t1) + 1e-10)
    t2 = np.cross(normal, t1)
    
    return t1, t2


def shape_operator_eigenvalues(mesh: 'TriangleMesh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Alias for principal_curvatures using quadric fit.
    
    The shape operator (Weingarten map) S has eigenvalues equal to
    the principal curvatures.
    """
    return principal_curvatures(mesh, method='cubic_fit')


def principal_directions(
    mesh: 'TriangleMesh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal directions at each vertex.
    
    Principal directions are the eigenvectors of the shape operator,
    expressed in 3D ambient space.
    
    Args:
        mesh: TriangleMesh
        
    Returns:
        d1: (V, 3) direction of maximum curvature
        d2: (V, 3) direction of minimum curvature
    """
    V = mesh.vertices.shape[0]
    d1 = np.zeros((V, 3), dtype=np.float64)
    d2 = np.zeros((V, 3), dtype=np.float64)
    
    for v_idx in range(V):
        n = mesh.vertex_normals[v_idx]
        t1, t2 = _tangent_basis(n)
        
        # Get neighbors and fit quadric (same as principal_curvatures)
        neighbor_faces = mesh.vertex_faces[v_idx]
        neighbor_set = set()
        for f_idx in neighbor_faces:
            for vn in mesh.faces[f_idx]:
                if vn != v_idx:
                    neighbor_set.add(vn)
        
        if len(neighbor_set) < 3:
            d1[v_idx] = t1
            d2[v_idx] = t2
            continue
        
        neighbors = list(neighbor_set)
        p_center = mesh.vertices[v_idx]
        
        u_coords = []
        v_coords = []
        h_coords = []
        
        for nb in neighbors:
            diff = mesh.vertices[nb] - p_center
            u_coords.append(np.dot(diff, t1))
            v_coords.append(np.dot(diff, t2))
            h_coords.append(np.dot(diff, n))
        
        u_coords = np.array(u_coords)
        v_coords = np.array(v_coords)
        h_coords = np.array(h_coords)
        
        A = np.column_stack([u_coords**2, u_coords*v_coords, v_coords**2])
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, h_coords, rcond=None)
            a, b, c = coeffs
        except:
            d1[v_idx] = t1
            d2[v_idx] = t2
            continue
        
        S = np.array([[2*a, b], [b, 2*c]])
        eigvals, eigvecs = np.linalg.eigh(S)
        
        # Convert 2D eigenvectors to 3D
        e1_2d = eigvecs[:, 1]  # Maximum curvature direction
        e2_2d = eigvecs[:, 0]  # Minimum curvature direction
        
        d1[v_idx] = e1_2d[0] * t1 + e1_2d[1] * t2
        d2[v_idx] = e2_2d[0] * t1 + e2_2d[1] * t2
    
    return d1, d2
