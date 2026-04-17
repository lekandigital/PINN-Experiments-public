"""
DiffGeo Compatibility Layer for GeoPINN
=======================================

Provides backward-compatible wrappers that use the shared diffgeo module
while maintaining the original GeoPINN API.

This layer allows gradual migration from the original dec_operators.py
to the shared diffgeo module without breaking existing code.

Usage (recommended):
    # Instead of:
    from geopinn.layers.dec_operators import build_dec_operators
    
    # Use:
    from geopinn.layers.diffgeo_compat import build_dec_operators
    
    # API remains identical, but uses improved cotangent weights
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import sparse

# Import shared diffgeo module
try:
    import sys
    from pathlib import Path
    shared_path = Path(__file__).parents[5] / "shared"
    if str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))
    
    from diffgeo.mesh import TriangleMesh
    from diffgeo.backends.torch_backend import TorchBackend
    DIFFGEO_AVAILABLE = True
except ImportError:
    DIFFGEO_AVAILABLE = False
    TriangleMesh = None
    TorchBackend = None

# Import original for fallback
from .dec_operators import build_dec_operators as build_dec_operators_legacy
from .dec_operators import DECLaplacian as DECLaplacianLegacy


def build_dec_operators(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    use_diffgeo: bool = True,
    return_mesh: bool = False
) -> Dict[str, Any]:
    """
    Build DEC operators from mesh data.
    
    This is a drop-in replacement for the original build_dec_operators
    that uses the shared diffgeo module for improved accuracy.
    
    Args:
        vertices: Vertex positions [V, 3]
        faces: Triangle indices [F, 3]
        use_diffgeo: If True, use shared diffgeo module. If False, use legacy.
        return_mesh: If True, also return the TriangleMesh object
        
    Returns:
        Dictionary with DEC operators:
        - 'd0': Edge incidence matrix [E, V]
        - 'd1': Face incidence matrix [F, E]
        - 'star0': Hodge star on 0-forms (dual areas) [V, V]
        - 'star1': Hodge star on 1-forms [E, E]
        - 'star2': Hodge star on 2-forms (face areas) [F, F]
        - 'laplacian': Cotangent Laplacian [V, V]
        - 'edge_lengths': Edge lengths [E]
        - 'face_areas': Face areas [F]
        - 'mesh': TriangleMesh object (if return_mesh=True)
    """
    if not use_diffgeo or not DIFFGEO_AVAILABLE:
        # Fall back to legacy implementation
        return build_dec_operators_legacy(vertices, faces)
    
    # Convert to numpy
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    # Build mesh with diffgeo
    mesh = TriangleMesh.from_vertices_faces(verts_np, faces_np)
    
    # Convert operators to torch sparse
    device = vertices.device
    dtype = vertices.dtype
    
    def scipy_to_torch_sparse(sp_matrix):
        """Convert SciPy sparse to PyTorch sparse tensor."""
        coo = sp_matrix.tocoo()
        indices = torch.stack([
            torch.from_numpy(coo.row.astype(np.int64)),
            torch.from_numpy(coo.col.astype(np.int64))
        ], dim=0)
        values = torch.from_numpy(coo.data.astype(np.float32))
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, coo.shape,
            dtype=dtype, device=device
        )
        return sparse_tensor.coalesce()
    
    result = {
        'd0': scipy_to_torch_sparse(mesh.d0),
        'd1': scipy_to_torch_sparse(mesh.d1),
        'star0': scipy_to_torch_sparse(mesh.star0),
        'star1': scipy_to_torch_sparse(mesh.star1),
        'star2': scipy_to_torch_sparse(mesh.star2),
        'laplacian': scipy_to_torch_sparse(mesh.laplacian),
        'edge_lengths': torch.from_numpy(mesh.edge_lengths).to(device, dtype),
        'face_areas': torch.from_numpy(mesh.face_areas).to(device, dtype),
        'dual_areas': torch.from_numpy(mesh.dual_areas).to(device, dtype),
    }
    
    if return_mesh:
        result['mesh'] = mesh
    
    return result


class DECLaplacian(nn.Module):
    """
    Discrete Laplace-Beltrami operator using DEC.
    
    Drop-in replacement for the original DECLaplacian that uses
    the shared diffgeo module for improved cotangent weights.
    
    The key improvement is using proper cotangent weights:
        w_ij = (cot α_ij + cot β_ij) / 2
    
    instead of simplified:
        w_ij = edge_length / 2
    
    Args:
        vertices: Initial vertex positions [V, 3]
        faces: Triangle indices [F, 3]
        use_diffgeo: If True, use shared module. If False, use legacy.
    """
    
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        use_diffgeo: bool = True
    ):
        super().__init__()
        
        self.use_diffgeo = use_diffgeo and DIFFGEO_AVAILABLE
        
        if self.use_diffgeo:
            # Build mesh and cache operators
            verts_np = vertices.detach().cpu().numpy()
            faces_np = faces.detach().cpu().numpy()
            
            self.mesh = TriangleMesh.from_vertices_faces(verts_np, faces_np)
            self.backend = TorchBackend(device=str(vertices.device))
            
            # Pre-convert to torch
            self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
            self.register_buffer(
                'dual_areas',
                torch.from_numpy(self.mesh.dual_areas).float()
            )
        else:
            # Use legacy implementation
            self._legacy = DECLaplacianLegacy(vertices, faces)
    
    def forward(
        self,
        f: torch.Tensor,
        strong_form: bool = True
    ) -> torch.Tensor:
        """
        Apply Laplace-Beltrami operator.
        
        Args:
            f: Input field [V] or [V, C]
            strong_form: If True, return point-wise Laplacian (normalized by area)
            
        Returns:
            Laplacian of f, same shape as input
        """
        if not self.use_diffgeo:
            return self._legacy(f, strong_form)
        
        # Apply sparse Laplacian
        if f.dim() == 1:
            Lf = torch.sparse.mm(self._laplacian, f.unsqueeze(-1)).squeeze(-1)
        else:
            Lf = torch.sparse.mm(self._laplacian, f)
        
        if strong_form:
            # Normalize by dual areas
            dual_areas = self.dual_areas.to(f.device)
            if f.dim() > 1:
                dual_areas = dual_areas.unsqueeze(-1)
            Lf = Lf / (dual_areas + 1e-12)
        
        return Lf
    
    def update_vertices(self, vertices: torch.Tensor):
        """
        Update geometry for deforming mesh.
        
        Recomputes Laplacian with new vertex positions while
        keeping same topology.
        """
        if not self.use_diffgeo:
            self._legacy.update_vertices(vertices)
            return
        
        verts_np = vertices.detach().cpu().numpy()
        self.mesh = self.mesh.update_vertices(verts_np)
        
        self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
        self.dual_areas = torch.from_numpy(self.mesh.dual_areas).float().to(vertices.device)


class CotangentLaplacian(DECLaplacian):
    """
    Alias for DECLaplacian with explicit cotangent naming.
    
    Makes it clear that this uses cotangent weights rather than
    simplified edge-length weights.
    """
    pass


# Additional utilities

def build_spectral_basis(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    k: int = 32,
    use_diffgeo: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Laplacian eigenfunctions for spectral methods.
    
    Uses the shared diffgeo module's eigensolver for improved accuracy.
    
    Args:
        vertices: Vertex positions [V, 3]
        faces: Triangle indices [F, 3]
        k: Number of eigenfunctions to compute
        use_diffgeo: If True, use shared module
        
    Returns:
        eigenvalues: [k] ascending eigenvalues (0, λ_1, λ_2, ...)
        eigenvectors: [V, k] corresponding eigenfunctions
    """
    if not use_diffgeo or not DIFFGEO_AVAILABLE:
        # Fall back to legacy spectral conv
        from .spectral_conv import compute_laplacian_eigenbasis
        return compute_laplacian_eigenbasis(vertices, faces, k)
    
    from diffgeo.operators.laplace_beltrami import laplacian_eigenpairs
    
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    mesh = TriangleMesh.from_vertices_faces(verts_np, faces_np)
    eigenvalues, eigenvectors = laplacian_eigenpairs(mesh, k)
    
    device = vertices.device
    dtype = vertices.dtype
    
    return (
        torch.from_numpy(eigenvalues).to(device, dtype),
        torch.from_numpy(eigenvectors).to(device, dtype)
    )
