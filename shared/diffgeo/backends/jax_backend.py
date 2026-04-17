"""
JAX Backend
===========

JAX backend for differentiable DEC operations.
Used by Project 16 (SurfPINN).

Converts SciPy sparse matrices to JAX BCOO format and provides
jax.grad-compatible sparse matrix-vector products.
"""

import numpy as np
from scipy import sparse
from typing import Any, Optional

from .base import DiffGeoBackend

# Lazy imports
jax = None
jnp = None
jsparse = None


def _ensure_jax():
    """Lazy import JAX."""
    global jax, jnp, jsparse
    if jax is None:
        try:
            import jax as _jax
            import jax.numpy as _jnp
            from jax.experimental import sparse as _jsparse
            
            jax = _jax
            jnp = _jnp
            jsparse = _jsparse
        except ImportError:
            raise ImportError(
                "JAX backend requires jax. Install with: pip install jax jaxlib"
            )


class JAXBackend(DiffGeoBackend):
    """
    JAX backend for differentiable DEC operations.
    
    Converts SciPy sparse matrices to JAX BCOO format for efficient
    sparse operations with automatic differentiation.
    
    Note: JAX's sparse support is less mature than PyTorch's. For small
    meshes (<10K vertices), consider using dense matrices as a fallback.
    
    Args:
        use_dense_fallback: If True, convert small sparse matrices to dense
        dense_threshold: Use dense for matrices with fewer than this many elements
    """
    
    def __init__(
        self,
        use_dense_fallback: bool = True,
        dense_threshold: int = 100_000
    ):
        _ensure_jax()
        
        self.use_dense_fallback = use_dense_fallback
        self.dense_threshold = dense_threshold
    
    @property
    def name(self) -> str:
        return 'jax'
    
    def sparse_to_tensor(self, sparse_matrix: sparse.csr_matrix) -> Any:
        """
        Convert SciPy sparse matrix to JAX format.
        
        Returns JAX BCOO sparse array if matrix is large enough,
        otherwise returns dense JAX array.
        """
        n_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
        
        if self.use_dense_fallback and n_elements < self.dense_threshold:
            # Use dense for small matrices
            return jnp.array(sparse_matrix.toarray())
        
        # Convert to BCOO format
        coo = sparse_matrix.tocoo()
        indices = np.column_stack([coo.row, coo.col])
        
        return jsparse.BCOO(
            (jnp.array(coo.data), jnp.array(indices)),
            shape=coo.shape
        )
    
    def tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert JAX array to NumPy."""
        if hasattr(tensor, 'todense'):
            # BCOO sparse
            tensor = tensor.todense()
        return np.asarray(tensor)
    
    def numpy_to_tensor(self, array: np.ndarray) -> Any:
        """Convert NumPy array to JAX array."""
        return jnp.array(array)
    
    def spmv(self, sparse_tensor: Any, dense_vector: Any) -> Any:
        """
        Sparse (or dense) matrix-vector product.
        
        Handles both BCOO sparse and dense JAX arrays.
        Differentiable via jax.grad.
        """
        if hasattr(sparse_tensor, 'shape') and not hasattr(sparse_tensor, 'todense'):
            # Dense array
            return sparse_tensor @ dense_vector
        else:
            # BCOO sparse
            return sparse_tensor @ dense_vector
    
    def apply_laplacian(
        self,
        mesh,
        field_tensor: Any,
        use_cached: bool = True,
        strong_form: bool = True
    ) -> Any:
        """
        Apply Laplace-Beltrami operator to a field tensor.
        
        Optimized for JAX with caching.
        
        Args:
            mesh: TriangleMesh with precomputed Laplacian
            field_tensor: (V,) or (V, C) field
            use_cached: Reuse cached tensor if available
            strong_form: If True, normalize by mass matrix
            
        Returns:
            Laplacian of field
        """
        cache_key = '_jax_laplacian'
        
        if not hasattr(mesh, cache_key) or not use_cached:
            setattr(mesh, cache_key, self.sparse_to_tensor(mesh.laplacian))
        
        L = getattr(mesh, cache_key)
        Lf = self.spmv(L, field_tensor)
        
        if strong_form:
            dual_areas = self.numpy_to_tensor(mesh.dual_areas)
            if field_tensor.ndim > 1:
                dual_areas = dual_areas[:, None]
            Lf = Lf / (dual_areas + 1e-12)
        
        return Lf
    
    def apply_gradient(
        self,
        mesh,
        field_tensor: Any,
        use_cached: bool = True
    ) -> Any:
        """Apply discrete gradient."""
        cache_key = '_jax_d0'
        
        if not hasattr(mesh, cache_key) or not use_cached:
            setattr(mesh, cache_key, self.sparse_to_tensor(mesh.d0))
        
        d0 = getattr(mesh, cache_key)
        return self.spmv(d0, field_tensor)
    
    def apply_divergence(
        self,
        mesh,
        edge_field_tensor: Any,
        use_cached: bool = True,
        strong_form: bool = True
    ) -> Any:
        """Apply discrete divergence."""
        cache_key = '_jax_d0T_star1'
        
        if not hasattr(mesh, cache_key) or not use_cached:
            d0T_star1 = mesh.d0.T @ mesh.star1
            setattr(mesh, cache_key, self.sparse_to_tensor(d0T_star1))
        
        d0T_star1 = getattr(mesh, cache_key)
        div_weak = self.spmv(d0T_star1, edge_field_tensor)
        
        if strong_form:
            dual_areas = self.numpy_to_tensor(mesh.dual_areas)
            div_weak = div_weak / (dual_areas + 1e-12)
        
        return div_weak


# Utility functions for use in JAX-based training loops

def make_laplacian_fn(mesh, backend: JAXBackend = None):
    """
    Create a JIT-compiled Laplacian function.
    
    Usage:
        laplacian_fn = make_laplacian_fn(mesh)
        Lf = laplacian_fn(field)
        grad_fn = jax.grad(lambda f: jnp.sum(laplacian_fn(f) ** 2))
    """
    _ensure_jax()
    
    if backend is None:
        backend = JAXBackend()
    
    # Cache the sparse matrix
    L = backend.sparse_to_tensor(mesh.laplacian)
    dual_areas = backend.numpy_to_tensor(mesh.dual_areas)
    
    @jax.jit
    def laplacian_fn(field):
        Lf = L @ field if not hasattr(L, 'todense') else L @ field
        return Lf / (dual_areas + 1e-12)
    
    return laplacian_fn
