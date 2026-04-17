"""
Laplacian Eigenpair Computation
===============================

Compute eigenpairs of the Laplace-Beltrami operator for spectral methods.
Uses scipy.sparse.linalg.eigsh for efficient computation on sparse matrices.

The eigenvectors form a basis for functions on the manifold (manifold harmonics),
analogous to Fourier basis on flat domains.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


@dataclass
class SpectralBasis:
    """
    Spectral basis for a manifold mesh.
    
    Stores Laplacian eigenpairs and provides methods for spectral analysis.
    
    Attributes:
        eigenvalues: (k,) sorted eigenvalues (smallest first)
        eigenvectors: (V, k) corresponding eigenvectors
        mesh_vertices: Number of vertices in the mesh
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mesh_vertices: int
    
    @property
    def k(self) -> int:
        """Number of eigenpairs."""
        return len(self.eigenvalues)
    
    def project(self, f: np.ndarray) -> np.ndarray:
        """
        Project a vertex function to spectral coefficients.
        
        Args:
            f: (V,) or (V, C) vertex function(s)
            
        Returns:
            coeffs: (k,) or (k, C) spectral coefficients
        """
        return self.eigenvectors.T @ f
    
    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct vertex function from spectral coefficients.
        
        Args:
            coeffs: (k,) or (k, C) spectral coefficients
            
        Returns:
            f: (V,) or (V, C) reconstructed function
        """
        return self.eigenvectors @ coeffs
    
    def filter(self, f: np.ndarray, filter_fn: callable) -> np.ndarray:
        """
        Apply spectral filter to vertex function.
        
        Args:
            f: (V,) or (V, C) input function
            filter_fn: Function λ -> h(λ) defining the filter
            
        Returns:
            filtered: (V,) or (V, C) filtered function
        """
        coeffs = self.project(f)
        h = filter_fn(self.eigenvalues)
        
        if coeffs.ndim == 1:
            filtered_coeffs = h * coeffs
        else:
            filtered_coeffs = h[:, None] * coeffs
        
        return self.reconstruct(filtered_coeffs)
    
    def heat_diffusion(self, f: np.ndarray, t: float) -> np.ndarray:
        """Apply heat diffusion for time t: h(λ) = exp(-tλ)."""
        return self.filter(f, lambda lam: np.exp(-t * lam))
    
    def smoothness_energy(self, f: np.ndarray) -> float:
        """
        Compute smoothness energy (Dirichlet energy) of function.
        
        E(f) = Σ λ_i |c_i|² where c_i are spectral coefficients.
        """
        coeffs = self.project(f)
        return np.sum(self.eigenvalues * coeffs**2)


def compute_laplacian_eigenpairs(
    mesh: 'TriangleMesh',
    k: int = 50,
    which: str = 'SM',
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k smallest eigenpairs of the Laplace-Beltrami operator.
    
    Uses the generalized eigenvalue problem: L @ v = λ @ M @ v
    where L is the cotangent Laplacian and M is the mass matrix.
    
    Args:
        mesh: TriangleMesh with precomputed Laplacian
        k: Number of eigenpairs to compute
        which: Which eigenvalues ('SM' = smallest magnitude, 'LM' = largest)
        sigma: Shift for shift-invert mode (better numerical stability)
        
    Returns:
        eigenvalues: (k,) sorted eigenvalues
        eigenvectors: (V, k) corresponding eigenvectors (M-orthonormal)
        
    Notes:
        - First eigenvalue is always ~0 (constant eigenfunction)
        - For unit sphere: eigenvalues are l(l+1) for l = 0, 1, 2, ...
          with multiplicities 1, 3, 5, 7, ...
    """
    L = mesh.laplacian
    M = mesh.mass_matrix
    
    # Ensure L is symmetric (handle numerical issues)
    L = 0.5 * (L + L.T)
    
    # Compute eigenpairs
    if sigma is not None:
        eigenvalues, eigenvectors = eigsh(L, k=k, M=M, which=which, sigma=sigma)
    else:
        eigenvalues, eigenvectors = eigsh(L, k=k, M=M, which=which)
    
    # Sort by eigenvalue (smallest first)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure consistent sign (first non-zero entry positive)
    for i in range(k):
        if eigenvectors[:, i].sum() < 0:
            eigenvectors[:, i] *= -1
    
    return eigenvalues, eigenvectors


def compute_eigenpairs_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenpairs directly from vertex/face arrays.
    
    Builds the cotangent Laplacian internally.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        k: Number of eigenpairs
        
    Returns:
        eigenvalues: (k,) eigenvalues
        eigenvectors: (V, k) eigenvectors
    """
    # Build cotangent Laplacian
    N = vertices.shape[0]
    F = faces.shape[0]
    
    rows, cols, data = [], [], []
    dual_areas = np.zeros(N)
    
    for f_idx in range(F):
        i, j, l = faces[f_idx]
        vi, vj, vl = vertices[i], vertices[j], vertices[l]
        
        # Edge vectors
        e_ij = vj - vi
        e_jl = vl - vj
        e_li = vi - vl
        
        # Face area (for dual areas)
        face_area = 0.5 * np.linalg.norm(np.cross(e_ij, -e_li))
        dual_areas[i] += face_area / 3
        dual_areas[j] += face_area / 3
        dual_areas[l] += face_area / 3
        
        # Cotangent weights
        def cotan(e1, e2):
            cross = np.cross(e1, e2)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-10:
                return 0.0
            return np.dot(e1, e2) / cross_norm
        
        cot_i = cotan(e_ij, -e_li)
        cot_j = cotan(e_jl, -e_ij)
        cot_l = cotan(e_li, -e_jl)
        
        # Add to Laplacian
        for (a, b, w) in [(j, l, cot_i), (i, l, cot_j), (i, j, cot_l)]:
            w = 0.5 * w
            rows.extend([a, b, a, b])
            cols.extend([b, a, a, b])
            data.extend([-w, -w, w, w])
    
    L = csr_matrix((data, (rows, cols)), shape=(N, N))
    L = 0.5 * (L + L.T)
    
    M = csr_matrix(np.diag(dual_areas))
    
    eigenvalues, eigenvectors = eigsh(L, k=k, M=M, which='SM')
    
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def get_spectral_basis(
    mesh: 'TriangleMesh',
    k: int = 50,
) -> SpectralBasis:
    """
    Get SpectralBasis object for a mesh.
    
    Convenience function that returns a SpectralBasis with useful methods.
    """
    eigenvalues, eigenvectors = compute_laplacian_eigenpairs(mesh, k)
    return SpectralBasis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        mesh_vertices=mesh.vertices.shape[0],
    )
