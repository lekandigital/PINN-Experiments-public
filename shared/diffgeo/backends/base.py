"""
Base Backend Interface
======================

Abstract interface for framework-specific backends.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh


class DiffGeoBackend(ABC):
    """
    Abstract interface for differential geometry backends.
    
    Backends convert SciPy sparse matrices to framework-specific sparse
    tensors and provide differentiable sparse matrix-vector products.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'numpy', 'torch', 'jax')."""
        pass
    
    @abstractmethod
    def sparse_to_tensor(self, sparse_matrix: sparse.csr_matrix) -> Any:
        """
        Convert SciPy sparse matrix to backend's sparse tensor format.
        
        Args:
            sparse_matrix: SciPy sparse matrix (CSR or COO)
            
        Returns:
            Backend-specific sparse tensor
        """
        pass
    
    @abstractmethod
    def tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert backend tensor to NumPy array.
        
        Args:
            tensor: Backend-specific tensor
            
        Returns:
            NumPy array
        """
        pass
    
    @abstractmethod
    def numpy_to_tensor(self, array: np.ndarray) -> Any:
        """
        Convert NumPy array to backend tensor.
        
        Args:
            array: NumPy array
            
        Returns:
            Backend-specific tensor
        """
        pass
    
    @abstractmethod
    def spmv(self, sparse_tensor: Any, dense_vector: Any) -> Any:
        """
        Sparse matrix-vector product.
        
        This operation must be differentiable for use in training loops.
        
        Args:
            sparse_tensor: Sparse matrix in backend format
            dense_vector: Dense vector in backend format
            
        Returns:
            Result of sparse_tensor @ dense_vector
        """
        pass
    
    def apply_laplacian(
        self, 
        mesh: 'TriangleMesh', 
        field_tensor: Any,
        use_cached: bool = True
    ) -> Any:
        """
        Apply Laplace-Beltrami operator to a field.
        
        This is the primary operation used in physics losses.
        
        Args:
            mesh: TriangleMesh with precomputed Laplacian
            field_tensor: (V,) or (V, C) field as backend tensor
            use_cached: If True, use cached sparse tensor if available
            
        Returns:
            Laplacian of field
        """
        # Default implementation using spmv
        if not hasattr(mesh, '_laplacian_tensor') or not use_cached:
            mesh._laplacian_tensor = self.sparse_to_tensor(mesh.laplacian)
        
        Lf = self.spmv(mesh._laplacian_tensor, field_tensor)
        
        # Normalize by mass matrix for strong form
        dual_areas = self.numpy_to_tensor(mesh.dual_areas)
        return Lf / (dual_areas + 1e-12)
    
    def apply_gradient(
        self,
        mesh: 'TriangleMesh',
        field_tensor: Any,
        use_cached: bool = True
    ) -> Any:
        """
        Apply discrete gradient to a scalar field.
        
        Args:
            mesh: TriangleMesh with precomputed d0
            field_tensor: (V,) scalar field as backend tensor
            
        Returns:
            (E,) edge 1-form
        """
        if not hasattr(mesh, '_d0_tensor') or not use_cached:
            mesh._d0_tensor = self.sparse_to_tensor(mesh.d0)
        
        return self.spmv(mesh._d0_tensor, field_tensor)
    
    def apply_divergence(
        self,
        mesh: 'TriangleMesh',
        edge_field_tensor: Any,
        use_cached: bool = True
    ) -> Any:
        """
        Apply discrete divergence to an edge 1-form.
        
        Args:
            mesh: TriangleMesh with precomputed operators
            edge_field_tensor: (E,) edge 1-form as backend tensor
            
        Returns:
            (V,) vertex scalar field
        """
        if not hasattr(mesh, '_d0T_star1_tensor') or not use_cached:
            # Build d0^T @ star1
            d0T_star1 = mesh.d0.T @ mesh.star1
            mesh._d0T_star1_tensor = self.sparse_to_tensor(d0T_star1)
        
        div_weak = self.spmv(mesh._d0T_star1_tensor, edge_field_tensor)
        
        # Normalize by dual areas for strong form
        dual_areas = self.numpy_to_tensor(mesh.dual_areas)
        return div_weak / (dual_areas + 1e-12)


def get_backend(name: str = 'numpy', **kwargs) -> DiffGeoBackend:
    """
    Get a backend by name.
    
    Args:
        name: 'numpy', 'torch', or 'jax'
        **kwargs: Backend-specific arguments (e.g., device='cuda' for torch)
        
    Returns:
        DiffGeoBackend instance
    """
    name = name.lower()
    
    if name == 'numpy':
        from .numpy_backend import NumPyBackend
        return NumPyBackend()
    elif name in ('torch', 'pytorch'):
        from .torch_backend import TorchBackend
        return TorchBackend(**kwargs)
    elif name == 'jax':
        from .jax_backend import JAXBackend
        return JAXBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}. Choose from: numpy, torch, jax")
