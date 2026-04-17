"""
DiffGeo-Based Differential Operators for CoastFlow-GNN
======================================================

Provides accurate cotangent-weighted Laplacian and other DEC operators
for coastal flow simulation, replacing simple graph-based approximations.

Key improvements over GraphDifferentialOperators:
- Cotangent weights: Accurate for non-uniform meshes, handles spherical geometry
- Dual area normalization: Proper point-wise scaling
- Spherical corrections: Can handle Earth-surface simulations

Configuration:
- Set domain.geometry: 'flat' for planar domains (default)
- Set domain.geometry: 'spherical' for Earth-surface simulations
"""

import torch
from typing import Optional, Tuple, Union
import numpy as np

# Conditional import for shared diffgeo module
try:
    import sys
    from pathlib import Path
    shared_path = Path(__file__).parents[5] / "shared"
    if str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))
    
    from diffgeo.mesh import TriangleMesh
    from diffgeo.mesh.generation import icosphere, cubed_sphere
    from diffgeo.backends.torch_backend import TorchBackend
    DIFFGEO_AVAILABLE = True
except ImportError:
    DIFFGEO_AVAILABLE = False
    TriangleMesh = None
    TorchBackend = None


def create_mesh_from_graph(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    face_index: Optional[torch.Tensor] = None
) -> 'TriangleMesh':
    """
    Create a TriangleMesh from graph data.
    
    If face_index is not provided, attempts Delaunay triangulation (2D/3D).
    
    Args:
        pos: Node positions [N, 2] or [N, 3]
        edge_index: Graph connectivity [2, E] (used for validation)
        face_index: Optional triangle indices [F, 3]
        
    Returns:
        TriangleMesh with precomputed DEC operators
    """
    if not DIFFGEO_AVAILABLE:
        raise ImportError("create_mesh_from_graph requires diffgeo module")
    
    vertices = pos.detach().cpu().numpy()
    
    # Handle 2D case
    if vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((len(vertices), 1))])
    
    if face_index is not None:
        faces = face_index.detach().cpu().numpy()
    else:
        # Use Delaunay triangulation
        from scipy.spatial import Delaunay
        
        # Project to 2D for triangulation if nearly flat
        if np.std(vertices[:, 2]) < 1e-6:
            tri = Delaunay(vertices[:, :2])
        else:
            # For 3D, this is more complex - use convex hull or alphashape
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices)
            # Note: This only gives surface triangulation for convex hull
            # For general meshes, use face_index directly
            raise NotImplementedError(
                "3D mesh triangulation not implemented. "
                "Please provide face_index explicitly."
            )
        faces = tri.simplices
    
    return TriangleMesh.from_vertices_faces(vertices, faces)


class MeshDifferentialOperators:
    """
    Differential operators on mesh using cotangent weights.
    
    Drop-in replacement for GraphDifferentialOperators that uses
    the shared diffgeo module for accurate geometry.
    
    Args:
        pos: Initial node positions [N, 3]
        face_index: Triangle faces [F, 3]
        device: PyTorch device
        geometry: 'flat' or 'spherical' (for Earth-surface)
    """
    
    def __init__(
        self,
        pos: torch.Tensor,
        face_index: torch.Tensor,
        device: str = 'cpu',
        geometry: str = 'flat'
    ):
        if not DIFFGEO_AVAILABLE:
            raise ImportError("MeshDifferentialOperators requires diffgeo module")
        
        self.device = device
        self.geometry = geometry
        
        vertices = pos.detach().cpu().numpy()
        faces = face_index.detach().cpu().numpy()
        
        # Handle 2D case
        if vertices.shape[1] == 2:
            vertices = np.hstack([vertices, np.zeros((len(vertices), 1))])
        
        self.mesh = TriangleMesh.from_vertices_faces(vertices, faces)
        self.backend = TorchBackend(device=device)
        
        # Pre-convert sparse matrices
        self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
        self._d0 = self.backend.sparse_to_tensor(self.mesh.d0)
        self._dual_areas = self.backend.numpy_to_tensor(self.mesh.dual_areas)
    
    def compute_laplacian(
        self,
        f: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        strong_form: bool = True
    ) -> torch.Tensor:
        """
        Compute cotangent-weighted Laplacian.
        
        Uses DEC cotangent Laplacian instead of simple weighted differences.
        
        Args:
            f: Scalar or vector field [N, C]
            pos: Ignored (for API compatibility with GraphDifferentialOperators)
            edge_index: Ignored
            strong_form: If True, normalize by dual areas
            
        Returns:
            lap_f: Laplacian [N, C]
        """
        # Ensure f is on correct device
        f = f.to(self.device)
        
        # Apply sparse Laplacian
        Lf = torch.sparse.mm(self._laplacian, f)  # [N, C]
        
        if strong_form:
            # Normalize by dual areas for point-wise Laplacian
            dual_areas = self._dual_areas
            if f.dim() > 1:
                dual_areas = dual_areas.unsqueeze(-1)
            Lf = Lf / (dual_areas + 1e-12)
        
        return Lf
    
    def compute_gradient(
        self,
        f: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient using DEC operators.
        
        Returns gradient as edge 1-form (per-edge values).
        
        Args:
            f: Scalar field [N] or [N, 1]
            pos, edge_index: Ignored (for API compatibility)
            
        Returns:
            grad_f: Edge gradient [E]
        """
        if f.dim() > 1:
            f = f.squeeze(-1)
        
        f = f.to(self.device)
        
        # d0 is the edge incidence matrix: grad_e f = f_dst - f_src
        grad_f = torch.sparse.mm(self._d0, f.unsqueeze(-1)).squeeze(-1)  # [E]
        
        return grad_f
    
    def compute_divergence(
        self,
        u: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute divergence using DEC operators.
        
        For vector field on vertices, first interpolates to edges,
        then applies divergence.
        
        Args:
            u: Vector field [N, 3]
            pos, edge_index: Ignored (for API compatibility)
            
        Returns:
            div_u: Divergence [N, 1]
        """
        # This is a simplified version - proper implementation would use
        # edge-based vector fields with Hodge star
        
        # For now, compute component-wise Laplacian as approximation
        # div(u) ≈ trace of Jacobian
        u = u.to(self.device)
        
        # Sum of d(u_i)/d(x_i) approximated via Laplacian eigenfunction
        # This is a placeholder - proper DEC divergence uses dual cells
        div_approx = torch.zeros(u.size(0), 1, device=self.device)
        
        for i in range(min(3, u.size(1))):
            Lu_i = self.compute_laplacian(u[:, i:i+1], strong_form=True)
            div_approx = div_approx + Lu_i
        
        return div_approx
    
    def update_positions(self, pos: torch.Tensor):
        """
        Update vertex positions (for deforming meshes).
        
        Recomputes geometry-dependent quantities while preserving topology.
        """
        vertices = pos.detach().cpu().numpy()
        if vertices.shape[1] == 2:
            vertices = np.hstack([vertices, np.zeros((len(vertices), 1))])
        
        self.mesh = self.mesh.update_vertices(vertices)
        
        # Re-convert operators
        self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
        self._dual_areas = self.backend.numpy_to_tensor(self.mesh.dual_areas)


class SphereGeometry:
    """
    Spherical Earth geometry support.
    
    Provides:
    - Lat/lon to Cartesian conversion
    - Great circle distances
    - Spherical Laplacian via stereographic projection or direct DEC
    
    Reference radius: 6371 km (mean Earth radius)
    """
    
    def __init__(self, radius: float = 6371.0):
        self.radius = radius
    
    def latlon_to_xyz(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert latitude/longitude to 3D Cartesian coordinates.
        
        Args:
            lat: Latitude in radians [N]
            lon: Longitude in radians [N]
            
        Returns:
            xyz: Cartesian coordinates [N, 3]
        """
        x = self.radius * torch.cos(lat) * torch.cos(lon)
        y = self.radius * torch.cos(lat) * torch.sin(lon)
        z = self.radius * torch.sin(lat)
        
        return torch.stack([x, y, z], dim=-1)
    
    def xyz_to_latlon(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert 3D Cartesian to latitude/longitude.
        
        Args:
            xyz: Cartesian coordinates [N, 3]
            
        Returns:
            lat, lon: Latitude and longitude in radians
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        lat = torch.asin(z / self.radius)
        lon = torch.atan2(y, x)
        
        return lat, lon
    
    def great_circle_distance(
        self,
        lat1: torch.Tensor, lon1: torch.Tensor,
        lat2: torch.Tensor, lon2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute great circle distance between points.
        
        Uses Haversine formula.
        """
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        return self.radius * c


def spherical_laplacian(
    f: torch.Tensor,
    mesh_ops: MeshDifferentialOperators
) -> torch.Tensor:
    """
    Compute Laplace-Beltrami on the sphere.
    
    The cotangent Laplacian automatically handles spherical geometry
    when the mesh vertices are on the sphere.
    
    For improved accuracy on coarse meshes, consider using:
    - Cubed sphere mesh (avoids pole singularities)
    - Icosphere (quasi-uniform triangulation)
    
    Args:
        f: Scalar field on sphere [N] or [N, C]
        mesh_ops: MeshDifferentialOperators on spherical mesh
        
    Returns:
        lap_f: Spherical Laplacian [N] or [N, C]
    """
    return mesh_ops.compute_laplacian(f, strong_form=True)
