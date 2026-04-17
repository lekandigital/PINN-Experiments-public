"""
NumPy Backend
=============

Default backend using NumPy/SciPy for all operations.
Not differentiable but useful for validation and non-training code.
"""

import numpy as np
from scipy import sparse
from typing import Any

from .base import DiffGeoBackend


class NumPyBackend(DiffGeoBackend):
    """
    NumPy/SciPy backend for differential geometry operations.
    
    This is the default backend used for operator construction and
    non-differentiable operations.
    """
    
    @property
    def name(self) -> str:
        return 'numpy'
    
    def sparse_to_tensor(self, sparse_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """Returns the sparse matrix as-is (already in SciPy format)."""
        if not sparse.issparse(sparse_matrix):
            return sparse.csr_matrix(sparse_matrix)
        return sparse_matrix.tocsr()
    
    def tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert to NumPy array."""
        if sparse.issparse(tensor):
            return tensor.toarray()
        return np.asarray(tensor)
    
    def numpy_to_tensor(self, array: np.ndarray) -> np.ndarray:
        """Return array as-is."""
        return np.asarray(array)
    
    def spmv(self, sparse_matrix: sparse.csr_matrix, dense_vector: np.ndarray) -> np.ndarray:
        """Sparse matrix-vector product using SciPy."""
        return sparse_matrix @ dense_vector
    
    def spgemm(
        self, 
        sparse_a: sparse.csr_matrix, 
        sparse_b: sparse.csr_matrix
    ) -> sparse.csr_matrix:
        """Sparse matrix-matrix product."""
        return sparse_a @ sparse_b
