"""
PyTorch Backend
===============

PyTorch backend for differentiable DEC operations.
Used by Projects 01, 06, and 09.

Converts SciPy sparse matrices to torch.sparse tensors and provides
autograd-compatible sparse matrix-vector products.
"""

import numpy as np
from scipy import sparse
from typing import Any, Optional

from .base import DiffGeoBackend

# Import torch only when backend is instantiated
torch = None


def _ensure_torch():
    """Lazy import torch."""
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            raise ImportError(
                "PyTorch backend requires torch. Install with: pip install torch"
            )


class TorchBackend(DiffGeoBackend):
    """
    PyTorch backend for differentiable DEC operations.
    
    Converts SciPy sparse matrices to torch.sparse_coo_tensor format
    and uses torch.sparse.mm for matrix-vector products.
    
    Gradients flow through all sparse operations via PyTorch autograd.
    
    Args:
        device: PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: PyTorch dtype (default: torch.float32)
    """
    
    def __init__(self, device: str = 'cpu', dtype: Optional[Any] = None):
        _ensure_torch()
        
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.float32
    
    @property
    def name(self) -> str:
        return f'torch:{self.device}'
    
    def sparse_to_tensor(self, sparse_matrix: sparse.csr_matrix) -> torch.Tensor:
        """
        Convert SciPy sparse matrix to PyTorch sparse COO tensor.
        
        The resulting tensor is coalesced (indices sorted, duplicates summed)
        and placed on the specified device.
        """
        # Convert to COO format for easier conversion
        coo = sparse_matrix.tocoo()
        
        # Build index tensor
        indices = torch.stack([
            torch.from_numpy(coo.row.astype(np.int64)),
            torch.from_numpy(coo.col.astype(np.int64))
        ], dim=0)
        
        # Build values tensor
        values = torch.from_numpy(coo.data.astype(np.float32))
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, 
            values, 
            size=coo.shape,
            dtype=self.dtype,
            device=self.device
        )
        
        # Coalesce for efficient operations
        return sparse_tensor.coalesce()
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array."""
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        return tensor.detach().cpu().numpy()
    
    def numpy_to_tensor(
        self, 
        array: np.ndarray,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor."""
        tensor = torch.from_numpy(array.astype(np.float32))
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        if requires_grad:
            tensor = tensor.requires_grad_(True)
        return tensor
    
    def spmv(
        self, 
        sparse_tensor: torch.Tensor, 
        dense_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparse matrix-vector product.
        
        Handles both 1D vectors (V,) and 2D matrices (V, C).
        Gradients flow through both inputs.
        """
        # torch.sparse.mm expects 2D inputs
        if dense_vector.dim() == 1:
            # (V,) -> (V, 1) -> matmul -> (V, 1) -> (V,)
            result = torch.sparse.mm(sparse_tensor, dense_vector.unsqueeze(-1))
            return result.squeeze(-1)
        else:
            # (V, C) input
            return torch.sparse.mm(sparse_tensor, dense_vector)
    
    def apply_laplacian(
        self,
        mesh,
        field_tensor: torch.Tensor,
        use_cached: bool = True,
        strong_form: bool = True
    ) -> torch.Tensor:
        """
        Apply Laplace-Beltrami operator to a field tensor.
        
        Optimized for PyTorch with optional strong-form normalization.
        
        Args:
            mesh: TriangleMesh with precomputed Laplacian
            field_tensor: (V,) or (V, C) field
            use_cached: Reuse cached sparse tensor if available
            strong_form: If True, normalize by mass matrix
            
        Returns:
            Laplacian of field
        """
        cache_key = f'_torch_laplacian_{self.device}'
        
        if not hasattr(mesh, cache_key) or not use_cached:
            setattr(mesh, cache_key, self.sparse_to_tensor(mesh.laplacian))
        
        L = getattr(mesh, cache_key)
        Lf = self.spmv(L, field_tensor)
        
        if strong_form:
            dual_areas = self.numpy_to_tensor(mesh.dual_areas)
            if field_tensor.dim() > 1:
                dual_areas = dual_areas.unsqueeze(-1)
            Lf = Lf / (dual_areas + 1e-12)
        
        return Lf
    
    def compute_mean_curvature_energy(
        self,
        mesh,
        vertices_tensor: torch.Tensor,
        rest_mcv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute intrinsic bending energy based on mean curvature.
        
        E_bend = Σ_v A_v* ||H(v) - H_rest(v)||²
        
        where H is the mean curvature vector Δx/2.
        
        This is the key integration point for Project 09 (cloth).
        
        Args:
            mesh: TriangleMesh (rest state topology)
            vertices_tensor: (V, 3) current vertex positions
            rest_mcv_tensor: (V, 3) rest-state mean curvature vectors
                            If None, uses flat (zero curvature) rest state
                            
        Returns:
            Bending energy scalar
        """
        # Compute mean curvature vector: H = Δx / 2
        Lx = self.apply_laplacian(mesh, vertices_tensor, strong_form=True)
        mcv = Lx / 2  # (V, 3)
        
        if rest_mcv_tensor is None:
            # Flat rest state: H_rest = 0
            diff = mcv
        else:
            diff = mcv - rest_mcv_tensor
        
        # Energy per vertex: ||H - H_rest||²
        energy_per_vertex = torch.sum(diff ** 2, dim=-1)  # (V,)
        
        # Weight by dual areas
        dual_areas = self.numpy_to_tensor(mesh.dual_areas)
        
        return torch.dot(energy_per_vertex, dual_areas)
    
    def compute_membrane_energy(
        self,
        mesh_rest,
        vertices_tensor: torch.Tensor,
        mu: float = 1.0,
        lam: float = 1.0
    ) -> torch.Tensor:
        """
        Compute membrane (stretch) energy from metric tensor deviation.
        
        Uses a simple Saint Venant-Kirchhoff energy:
        E = Σ_f A_f * (μ ||E||² + λ/2 tr(E)²)
        
        where E = (C - I)/2 is the Green strain tensor.
        
        Args:
            mesh_rest: Rest-state TriangleMesh
            vertices_tensor: (V, 3) current vertex positions
            mu, lam: Lamé parameters
            
        Returns:
            Membrane energy scalar
        """
        # This is a simplified version - full implementation would use
        # deformation gradient computed per-face
        
        # For each edge, compute stretch
        d0 = self.sparse_to_tensor(mesh_rest.d0)
        
        # Current edge vectors
        edge_vecs = self.spmv(d0, vertices_tensor)  # (E, 3)
        edge_lengths = torch.norm(edge_vecs, dim=-1)  # (E,)
        
        # Rest edge lengths
        rest_lengths = self.numpy_to_tensor(mesh_rest.edge_lengths)
        
        # Stretch = (current - rest) / rest
        stretch = (edge_lengths - rest_lengths) / (rest_lengths + 1e-8)
        
        # Simple spring energy (approximation of membrane energy)
        energy = mu * torch.sum(stretch ** 2)
        
        return energy
