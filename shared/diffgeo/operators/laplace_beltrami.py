"""
Laplace-Beltrami Operator
=========================

The cotangent-weighted discrete Laplace-Beltrami operator for triangle meshes.

This is the central operator for diffusion, smoothing, and solving PDEs
on surfaces. Uses the cotangent formula for accurate geometry encoding.

Mathematical definition (strong form):
    (Δf)_i = (1/A_i*) Σ_j w_ij (f_j - f_i)

where:
    w_ij = (cot α_ij + cot β_ij) / 2  (cotangent weights)
    A_i* = dual cell area at vertex i
    α_ij, β_ij = angles opposite to edge (i,j) in adjacent triangles

References:
- Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
- Crane et al., "Digital Geometry Processing with Discrete Exterior Calculus"
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import TYPE_CHECKING, Tuple, Optional
import warnings

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


def laplace_beltrami(
    mesh: 'TriangleMesh',
    f: np.ndarray
) -> np.ndarray:
    """
    Apply strong-form Laplace-Beltrami operator to a scalar field.
    
    Computes Δf = M⁻¹ L f, where:
        L = cotangent weight matrix (weak Laplacian)
        M = diagonal mass matrix (dual areas)
    
    This returns pointwise Laplacian values at each vertex.
    
    Args:
        mesh: TriangleMesh with precomputed Laplacian
        f: (V,) or (V, C) scalar or vector field on vertices
        
    Returns:
        Lf: (V,) or (V, C) Laplace-Beltrami of f
    """
    if mesh.laplacian is None:
        raise ValueError("Mesh Laplacian not computed. Use TriangleMesh.from_vertices_faces()")
    
    f = np.asarray(f)
    
    # Weak Laplacian application
    Lf_weak = mesh.laplacian @ f
    
    # Normalize by mass (dual areas) for strong form
    dual_areas = mesh.dual_areas
    Lf_strong = Lf_weak / np.maximum(dual_areas, 1e-12)[:, None] if f.ndim > 1 else \
                Lf_weak / np.maximum(dual_areas, 1e-12)
    
    return Lf_strong


def laplace_beltrami_weak(
    mesh: 'TriangleMesh',
    f: np.ndarray
) -> np.ndarray:
    """
    Apply weak-form Laplace-Beltrami operator.
    
    Returns L @ f without mass matrix normalization.
    Useful for loss functions where you integrate against test functions.
    
    Args:
        mesh: TriangleMesh with precomputed Laplacian
        f: (V,) or (V, C) scalar field on vertices
        
    Returns:
        Lf: (V,) or (V, C) weak Laplacian of f
    """
    if mesh.laplacian is None:
        raise ValueError("Mesh Laplacian not computed.")
    
    f = np.asarray(f)
    return mesh.laplacian @ f


def build_cotangent_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    return_mass: bool = False
) -> sparse.csr_matrix:
    """
    Build the cotangent Laplacian matrix from scratch.
    
    The weak Laplacian L has entries:
        L[i,j] = -w_ij = -(cot α_ij + cot β_ij)/2  for neighbors i,j
        L[i,i] = Σ_j w_ij
    
    This is symmetric negative semi-definite.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices
        return_mass: If True, also return mass matrix
        
    Returns:
        L: (V, V) cotangent Laplacian matrix
        M: (V, V) diagonal mass matrix (if return_mass=True)
    """
    V = vertices.shape[0]
    F = faces.shape[0]
    
    # Build Laplacian using triplet format
    rows, cols, data = [], [], []
    
    # Also compute dual areas for mass matrix
    dual_areas = np.zeros(V, dtype=np.float64)
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        
        # Edge vectors
        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk
        
        # Face area (for mass matrix)
        cross = np.cross(e_ij, -e_ki)
        area = 0.5 * np.linalg.norm(cross)
        
        dual_areas[i] += area / 3
        dual_areas[j] += area / 3
        dual_areas[k] += area / 3
        
        # Cotangent weights
        # cot(angle at i) for edge (j,k) opposite to i
        # cot(θ) = (a · b) / |a × b|
        def cotan(e1, e2):
            cross = np.cross(e1, e2)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                return 0.0
            return np.dot(e1, e2) / cross_norm
        
        # Angles at each vertex
        cot_i = cotan(e_ij, -e_ki)  # angle at i, opposite to edge (j,k)
        cot_j = cotan(e_jk, -e_ij)  # angle at j, opposite to edge (i,k)
        cot_k = cotan(-e_jk, e_ki)  # angle at k, opposite to edge (i,j)
        
        # Add contributions to Laplacian
        # Edge (j, k): weight = cot_i / 2
        w_jk = max(cot_i / 2, 1e-8)  # Clamp negative weights
        rows.extend([j, k, j, k])
        cols.extend([k, j, j, k])
        data.extend([-w_jk, -w_jk, w_jk, w_jk])
        
        # Edge (i, k): weight = cot_j / 2
        w_ik = max(cot_j / 2, 1e-8)
        rows.extend([i, k, i, k])
        cols.extend([k, i, i, k])
        data.extend([-w_ik, -w_ik, w_ik, w_ik])
        
        # Edge (i, j): weight = cot_k / 2
        w_ij = max(cot_k / 2, 1e-8)
        rows.extend([i, j, i, j])
        cols.extend([j, i, i, j])
        data.extend([-w_ij, -w_ij, w_ij, w_ij])
    
    L = sparse.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
    
    # Symmetrize (numerical cleanup)
    L = 0.5 * (L + L.T)
    
    if return_mass:
        M = sparse.diags(dual_areas, format='dia')
        return L, M
    
    return L


def compute_cotangent_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray
) -> np.ndarray:
    """
    Compute cotangent weights for each edge.
    
    For edge (i,j):
        w_ij = (cot α + cot β) / 2
    where α, β are angles opposite to the edge in adjacent triangles.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        edges: (E, 2) edge indices
        
    Returns:
        weights: (E,) cotangent weight per edge
    """
    E = edges.shape[0]
    weights = np.zeros(E, dtype=np.float64)
    
    # Build edge-to-faces mapping
    edge_to_faces = {tuple(e): [] for e in edges}
    for f_idx, face in enumerate(faces):
        i, j, k = face
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            if edge in edge_to_faces:
                edge_to_faces[edge].append(f_idx)
    
    edge_to_idx = {tuple(e): idx for idx, e in enumerate(edges)}
    
    for e_idx, edge in enumerate(edges):
        edge_tuple = tuple(edge)
        adjacent_faces = edge_to_faces.get(edge_tuple, [])
        
        total_cotan = 0.0
        for f_idx in adjacent_faces:
            face = faces[f_idx]
            
            # Find vertex opposite to this edge
            opposite = None
            for v in face:
                if v not in edge:
                    opposite = v
                    break
            
            if opposite is None:
                continue
            
            # Compute cotangent of angle at opposite vertex
            vi, vj = edge
            pi, pj, pk = vertices[vi], vertices[vj], vertices[opposite]
            
            # Vectors from opposite vertex
            e1 = pi - pk
            e2 = pj - pk
            
            cross_norm = np.linalg.norm(np.cross(e1, e2))
            if cross_norm > 1e-12:
                cotan = np.dot(e1, e2) / cross_norm
                total_cotan += cotan
        
        weights[e_idx] = total_cotan / 2
    
    # Clamp negative weights
    n_negative = np.sum(weights < 0)
    if n_negative > 0:
        warnings.warn(
            f"Clamped {n_negative}/{E} negative cotangent weights (obtuse triangles)."
        )
    weights = np.maximum(weights, 1e-8)
    
    return weights


def laplacian_eigenpairs(
    mesh: 'TriangleMesh',
    k: int = 50,
    use_mass: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k smallest eigenpairs of the Laplace-Beltrami operator.
    
    Solves the generalized eigenvalue problem:
        L φ = λ M φ  (if use_mass=True)
        L φ = λ φ    (if use_mass=False)
    
    where L is the cotangent Laplacian and M is the mass matrix.
    
    Args:
        mesh: TriangleMesh with precomputed operators
        k: Number of eigenpairs to compute
        use_mass: If True, solve generalized problem (correct for non-uniform mesh)
        
    Returns:
        eigenvalues: (k,) eigenvalues (sorted, smallest first)
        eigenvectors: (V, k) corresponding eigenvectors
    """
    if mesh.laplacian is None:
        raise ValueError("Mesh Laplacian not computed.")
    
    L = mesh.laplacian
    
    if use_mass:
        M = mesh.mass_matrix
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigsh(L, k=k, M=M, which='SM')
    else:
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def mean_curvature_from_laplacian(
    mesh: 'TriangleMesh',
) -> np.ndarray:
    """
    Compute mean curvature using Δx = 2Hn.
    
    For a surface in 3D, the Laplace-Beltrami of the position vector
    gives twice the mean curvature times the normal:
        Δx = 2 H n
    
    Returns:
        H: (V,) mean curvature at each vertex (signed)
    """
    # Apply Laplacian to position vectors
    Lx = laplace_beltrami(mesh, mesh.vertices)
    
    # Mean curvature vector
    H_vec = Lx / 2
    
    # Project onto normals to get signed scalar curvature
    H = np.sum(H_vec * mesh.vertex_normals, axis=1)
    
    return H
