"""
Intrinsic Bending Energy for Cloth Simulation
==============================================

Proper discrete differential geometry-based bending energy using the
shared diffgeo module's cotangent Laplacian.

The bending energy is based on mean curvature:

    E_bend = ∫_S (H - H₀)² dA

where H is the mean curvature and H₀ is the rest-state mean curvature.
In discrete form:

    E_bend = Σ_v A*_v ||H(v) - H₀(v)||²

where A*_v is the dual area (Voronoi or barycentric) around vertex v.

Mean curvature vector is computed as H = (Δx) / 2, where Δ is the
cotangent-weighted Laplace-Beltrami operator.

This replaces the placeholder `bending_loss()` in losses.py which used
edge direction variance as a crude approximation.

Reference:
- M. Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

# Conditional import for the shared diffgeo module
try:
    import sys
    from pathlib import Path
    # Add shared module to path if not installed
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


class IntrinsicBendingEnergy(nn.Module):
    """
    Intrinsic bending energy based on discrete mean curvature.
    
    This module caches the mesh topology and DEC operators to avoid
    recomputation. Only the geometry-dependent operations are performed
    each forward pass.
    
    Args:
        vertices: Rest-state vertex positions (V, 3) as numpy array
        faces: Triangle indices (F, 3) as numpy array
        rest_curvature: If True, compute rest-state mean curvature as target
                       If False, use zero (flat) target
        device: PyTorch device
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        rest_curvature: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if not DIFFGEO_AVAILABLE:
            raise ImportError(
                "Intrinsic bending energy requires the diffgeo module. "
                "Make sure shared/diffgeo is in your Python path."
            )
        
        # Create mesh from rest state
        self.mesh = TriangleMesh.from_vertices_faces(vertices, faces)
        self.backend = TorchBackend(device=device)
        
        # Cache rest-state mean curvature vector
        if rest_curvature:
            rest_mcv = self._compute_mean_curvature_vector_np(self.mesh)
            self.register_buffer(
                'rest_mcv',
                torch.from_numpy(rest_mcv).float().to(device)
            )
        else:
            self.register_buffer(
                'rest_mcv',
                torch.zeros(vertices.shape[0], 3, device=device)
            )
        
        # Cache dual areas for weighting
        self.register_buffer(
            'dual_areas',
            torch.from_numpy(self.mesh.dual_areas).float().to(device)
        )
        
        # Pre-convert Laplacian to torch sparse
        self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
        
    def _compute_mean_curvature_vector_np(self, mesh: 'TriangleMesh') -> np.ndarray:
        """Compute mean curvature vector H = Δx/2 using NumPy."""
        Lx = mesh.laplacian @ mesh.vertices
        # Strong form: divide by dual areas
        mcv = Lx / (mesh.dual_areas[:, None] + 1e-12) / 2
        return mcv
    
    def forward(
        self,
        vertices: torch.Tensor,
        return_curvature: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bending energy for current vertex positions.
        
        Args:
            vertices: Current vertex positions (V, 3) or (B, V, 3)
            return_curvature: If True, also return mean curvature vector
            
        Returns:
            energy: Bending energy scalar
            mcv: Mean curvature vector (optional)
        """
        batched = vertices.dim() == 3
        if batched:
            # Process each sample in batch
            energies = []
            mcvs = []
            for b in range(vertices.shape[0]):
                e, m = self._compute_single(vertices[b], return_curvature=True)
                energies.append(e)
                mcvs.append(m)
            energy = torch.stack(energies).mean()
            if return_curvature:
                return energy, torch.stack(mcvs)
            return energy
        
        return self._compute_single(vertices, return_curvature)
    
    def _compute_single(
        self,
        vertices: torch.Tensor,
        return_curvature: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute bending energy for single mesh."""
        # Compute Laplacian * vertices
        Lx = torch.sparse.mm(self._laplacian, vertices)  # (V, 3)
        
        # Strong form: normalize by dual areas
        dual_areas = self.dual_areas.unsqueeze(-1)  # (V, 1)
        Lx_normalized = Lx / (dual_areas + 1e-12)
        
        # Mean curvature vector: H = Δx / 2
        mcv = Lx_normalized / 2  # (V, 3)
        
        # Deviation from rest curvature
        diff = mcv - self.rest_mcv  # (V, 3)
        
        # Energy per vertex: ||H - H₀||²
        energy_per_vertex = torch.sum(diff ** 2, dim=-1)  # (V,)
        
        # Total energy: weighted by dual areas
        energy = torch.dot(energy_per_vertex, self.dual_areas)
        
        if return_curvature:
            return energy, mcv
        return energy
    
    def update_vertices(self, vertices: np.ndarray):
        """
        Update rest-state vertices if mesh topology unchanged.
        
        Useful for changing rest shape without recreating the module.
        """
        self.mesh = self.mesh.update_vertices(vertices)
        rest_mcv = self._compute_mean_curvature_vector_np(self.mesh)
        self.rest_mcv = torch.from_numpy(rest_mcv).float().to(self.rest_mcv.device)


def intrinsic_bending_loss(
    vertices: torch.Tensor,
    rest_mesh: 'TriangleMesh',
    rest_mcv: Optional[torch.Tensor] = None,
    backend: Optional['TorchBackend'] = None
) -> torch.Tensor:
    """
    Functional interface for intrinsic bending loss.
    
    For use when you don't want to maintain a Module instance.
    
    Args:
        vertices: Current vertex positions (V, 3)
        rest_mesh: TriangleMesh with precomputed operators
        rest_mcv: Rest-state mean curvature vector. If None, uses zero.
        backend: TorchBackend instance. Created if None.
        
    Returns:
        Bending energy scalar
    """
    if not DIFFGEO_AVAILABLE:
        raise ImportError("intrinsic_bending_loss requires the diffgeo module")
    
    if backend is None:
        device = str(vertices.device)
        backend = TorchBackend(device=device)
    
    energy = backend.compute_mean_curvature_energy(
        rest_mesh,
        vertices,
        rest_mcv_tensor=rest_mcv
    )
    
    return energy


class MembraneEnergy(nn.Module):
    """
    Membrane (stretch) energy based on metric tensor deviation.
    
    Simpler version using edge stretch:
    
        E_membrane = Σ_e k_e (||e|| - ||e₀||)² / ||e₀||
        
    where e is an edge vector and e₀ is its rest length.
    
    This is equivalent to a linearized Saint Venant-Kirchhoff model.
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        stiffness: float = 1.0,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if not DIFFGEO_AVAILABLE:
            raise ImportError("MembraneEnergy requires the diffgeo module")
        
        self.mesh = TriangleMesh.from_vertices_faces(vertices, faces)
        self.backend = TorchBackend(device=device)
        self.stiffness = stiffness
        
        # Cache rest edge lengths
        self.register_buffer(
            'rest_lengths',
            torch.from_numpy(self.mesh.edge_lengths).float().to(device)
        )
        
        # Cache edge extraction operator (d0)
        self._d0 = self.backend.sparse_to_tensor(self.mesh.d0)
        
    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute membrane energy.
        
        Args:
            vertices: Current vertex positions (V, 3)
            
        Returns:
            Membrane energy scalar
        """
        # Compute current edge vectors: e = v_target - v_source
        edge_vecs = torch.sparse.mm(self._d0, vertices)  # (E, 3)
        curr_lengths = torch.norm(edge_vecs, dim=-1)  # (E,)
        
        # Relative stretch
        stretch = (curr_lengths - self.rest_lengths) / (self.rest_lengths + 1e-8)
        
        # Quadratic energy
        energy = self.stiffness * torch.sum(stretch ** 2)
        
        return energy


def membrane_loss(
    vertices: torch.Tensor,
    rest_mesh: 'TriangleMesh',
    stiffness: float = 1.0,
    backend: Optional['TorchBackend'] = None
) -> torch.Tensor:
    """
    Functional interface for membrane loss.
    
    Args:
        vertices: Current vertex positions (V, 3)
        rest_mesh: TriangleMesh with precomputed operators
        stiffness: Spring stiffness
        backend: TorchBackend instance
        
    Returns:
        Membrane energy scalar
    """
    if not DIFFGEO_AVAILABLE:
        raise ImportError("membrane_loss requires the diffgeo module")
    
    if backend is None:
        device = str(vertices.device)
        backend = TorchBackend(device=device)
    
    return backend.compute_membrane_energy(rest_mesh, vertices, mu=stiffness)
