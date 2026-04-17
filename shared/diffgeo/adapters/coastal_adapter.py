"""
Coastal Manifold Adapter
========================

Bridge differential geometry infrastructure to coastal/ocean dynamics.
Targets Projects 06 (CoastFlow-GNN) and 16 (SurfPINN).

Domain context:
- Earth surface is curved (sphere or local projection)
- Shallow water equations on curved bathymetry
- Coriolis effects from Earth rotation
- Tidal forcing and boundary conditions
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class CoastalDomainConfig:
    """Configuration for a coastal domain."""
    
    # Coordinate system
    projection: str = 'utm'  # 'utm', 'mercator', 'equirectangular', 'gnomonic'
    utm_zone: Optional[int] = None
    utm_hemisphere: str = 'N'
    
    # Physical parameters
    gravity: float = 9.81  # m/s²
    earth_radius: float = 6371000.0  # m
    earth_rotation: float = 7.2921e-5  # rad/s
    reference_density: float = 1025.0  # kg/m³ (seawater)
    
    # Domain bounds (in projected coordinates)
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    
    # Characteristic scales for non-dimensionalization
    length_scale: float = 1000.0  # m
    velocity_scale: float = 1.0  # m/s
    depth_scale: float = 10.0  # m


class CoastalManifoldAdapter:
    """
    Adapter connecting diffgeo infrastructure to coastal dynamics.
    
    Provides:
    1. Coordinate transformation (lat/lon ↔ projected ↔ mesh local)
    2. Coriolis parameter computation
    3. Bathymetry handling
    4. Spectral/chart setup for ocean domains
    
    Example:
        >>> adapter = CoastalManifoldAdapter.from_bathymetry(
        ...     bathymetry_file='coastal_dem.tif',
        ...     bounds=(lon_min, lon_max, lat_min, lat_max),
        ... )
        >>> mesh = adapter.create_mesh(resolution=500)  # 500m resolution
        >>> atlas = adapter.create_chart_atlas(n_charts=16)
    """
    
    def __init__(
        self,
        config: CoastalDomainConfig,
        mesh: Optional['TriangleMesh'] = None,
        bathymetry: Optional[np.ndarray] = None,  # Depth values at mesh vertices
    ):
        self.config = config
        self.mesh = mesh
        self.bathymetry = bathymetry
        
        # Lazy-initialized components
        self._atlas = None
        self._spectral_basis = None
        self._coriolis = None
    
    @classmethod
    def from_bathymetry(
        cls,
        bathymetry_grid: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        config: Optional[CoastalDomainConfig] = None,
        simplify_ratio: float = 0.5,
    ) -> 'CoastalManifoldAdapter':
        """
        Create adapter from bathymetry data on regular grid.
        
        Args:
            bathymetry_grid: (ny, nx) depth values (positive down)
            x_coords: (nx,) x coordinates
            y_coords: (ny,) y coordinates
            config: Domain configuration
            simplify_ratio: Mesh simplification factor
        """
        from ..mesh.generation import generate_from_heightfield
        
        if config is None:
            config = CoastalDomainConfig(
                x_min=x_coords.min(),
                x_max=x_coords.max(),
                y_min=y_coords.min(),
                y_max=y_coords.max(),
            )
        
        # Generate mesh from bathymetry
        # Treat bathymetry as a 2.5D heightfield (x, y, -depth)
        mesh = generate_from_heightfield(
            x_coords, y_coords, -bathymetry_grid,
            simplify_ratio=simplify_ratio,
        )
        
        # Interpolate bathymetry to mesh vertices
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (y_coords, x_coords),
            bathymetry_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        bathymetry_at_verts = interp(mesh.vertices[:, :2])
        
        return cls(config=config, mesh=mesh, bathymetry=bathymetry_at_verts)
    
    @classmethod
    def from_coastline(
        cls,
        coastline_points: np.ndarray,
        domain_extent: float = 10000.0,  # meters
        resolution: float = 500.0,
        config: Optional[CoastalDomainConfig] = None,
    ) -> 'CoastalManifoldAdapter':
        """
        Create adapter from coastline polygon.
        
        Generates a mesh that conforms to the coastline boundary
        with adaptive refinement near the coast.
        """
        from ..mesh.generation import generate_constrained_mesh
        
        if config is None:
            centroid = coastline_points.mean(axis=0)
            config = CoastalDomainConfig(
                x_min=centroid[0] - domain_extent,
                x_max=centroid[0] + domain_extent,
                y_min=centroid[1] - domain_extent,
                y_max=centroid[1] + domain_extent,
            )
        
        # Generate mesh with coastline as constraint
        mesh = generate_constrained_mesh(
            boundary_segments=[coastline_points],
            min_area=resolution ** 2 / 2,
        )
        
        return cls(config=config, mesh=mesh, bathymetry=None)
    
    def project_latlon_to_local(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """
        Project lat/lon coordinates to local mesh coordinates.
        
        Returns:
            coords: (N, 2) projected coordinates
        """
        from ..projection import latlon_to_utm, gnomonic_projection
        
        if self.config.projection == 'utm':
            x, y = latlon_to_utm(
                lat, lon,
                zone=self.config.utm_zone,
                northern=self.config.utm_hemisphere == 'N',
            )
        elif self.config.projection == 'gnomonic':
            # Center of domain as tangent point
            lat0 = (self.config.y_min + self.config.y_max) / 2
            lon0 = (self.config.x_min + self.config.x_max) / 2
            x, y = gnomonic_projection(lat, lon, lat0, lon0)
            x *= self.config.earth_radius
            y *= self.config.earth_radius
        else:
            raise ValueError(f"Unknown projection: {self.config.projection}")
        
        return np.stack([x, y], axis=-1)
    
    def compute_coriolis_parameter(
        self,
        latitude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Coriolis parameter f = 2Ω sin(φ).
        
        If latitude not provided, uses mesh vertex y-coordinates
        assuming they represent latitude in the projection.
        
        Returns:
            f: (N,) Coriolis parameter at each point
        """
        if latitude is None:
            if self.mesh is None:
                raise ValueError("No mesh or latitude provided")
            # Approximate: use y-coordinate as proxy for latitude
            # This works for small domains with UTM projection
            latitude = self.mesh.vertices[:, 1] / self.config.earth_radius
            latitude = np.degrees(latitude)
        
        from ..projection import compute_coriolis_parameter
        return compute_coriolis_parameter(latitude, self.config.earth_rotation)
    
    def create_chart_atlas(
        self,
        n_charts: int = 16,
        overlap: float = 0.3,
    ):
        """
        Create chart atlas for the coastal domain.
        
        Charts are placed adaptively, with more charts near complex bathymetry.
        
        Args:
            n_charts: Number of charts
            overlap: Overlap factor between charts
            
        Returns:
            Atlas object from charts subpackage
        """
        if self.mesh is None:
            raise ValueError("Mesh required to create chart atlas")
        
        from ..charts import Atlas
        
        self._atlas = Atlas.from_mesh(
            vertices=self.mesh.vertices,
            faces=self.mesh.faces,
            n_charts=n_charts,
            overlap=overlap,
        )
        return self._atlas
    
    def create_spectral_basis(
        self,
        n_eigenpairs: int = 50,
    ):
        """
        Create spectral basis from mesh Laplacian.
        
        Useful for:
        - Spectral filtering of solutions
        - Low-rank representation
        - Scale-aware operations
        
        Args:
            n_eigenpairs: Number of Laplacian eigenpairs
            
        Returns:
            SpectralBasis object from spectral subpackage
        """
        if self.mesh is None:
            raise ValueError("Mesh required to create spectral basis")
        
        from ..spectral import SpectralBasis, compute_laplacian_eigenpairs
        
        eigenvalues, eigenvectors = compute_laplacian_eigenpairs(
            self.mesh,
            k=n_eigenpairs,
        )
        
        self._spectral_basis = SpectralBasis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mass_matrix=self.mesh.mass_matrix,
        )
        return self._spectral_basis
    
    def get_boundary_vertices(
        self,
        boundary_type: str = 'all',
    ) -> np.ndarray:
        """
        Get indices of boundary vertices.
        
        Args:
            boundary_type: 'all', 'coastline', 'open_ocean', 'inflow', 'outflow'
            
        Returns:
            indices: (N_boundary,) vertex indices
        """
        if self.mesh is None:
            raise ValueError("Mesh required")
        
        # Find boundary edges (edges with only one adjacent face)
        edge_counts = {}
        for face in self.mesh.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        boundary_edges = [e for e, c in edge_counts.items() if c == 1]
        boundary_verts = list(set(v for e in boundary_edges for v in e))
        
        if boundary_type == 'all':
            return np.array(boundary_verts)
        
        # Classify by bathymetry or position
        boundary_verts = np.array(boundary_verts)
        positions = self.mesh.vertices[boundary_verts]
        
        if boundary_type == 'coastline' and self.bathymetry is not None:
            # Shallow water threshold
            mask = self.bathymetry[boundary_verts] < 5.0
            return boundary_verts[mask]
        
        return boundary_verts


if HAS_TORCH:
    class ShallowWaterManifoldLoss(nn.Module):
        """
        Physics-informed loss for shallow water equations on manifold.
        
        Equations:
        ∂η/∂t + ∇·(h*u) = 0  (continuity)
        ∂u/∂t + (u·∇)u + f×u + g∇η = -τ_b/(ρh) + τ_s/(ρh)  (momentum)
        
        where:
        - η: sea surface elevation
        - h: total water depth (H + η, H = bathymetry)
        - u: depth-averaged velocity (u, v)
        - f: Coriolis parameter
        - g: gravity
        - τ_b: bottom friction
        - τ_s: wind stress
        
        The manifold structure enters through:
        - Metric tensor for gradient/divergence computation
        - Coriolis varies with latitude
        - Curvature corrections for large domains
        """
        
        def __init__(
            self,
            adapter: CoastalManifoldAdapter,
            gravity: float = 9.81,
            bottom_drag: float = 0.0025,
            use_coriolis: bool = True,
            use_curvature_correction: bool = False,
        ):
            super().__init__()
            self.adapter = adapter
            self.gravity = gravity
            self.bottom_drag = bottom_drag
            self.use_coriolis = use_coriolis
            self.use_curvature_correction = use_curvature_correction
            
            # Precompute Coriolis if mesh available
            if use_coriolis and adapter.mesh is not None:
                coriolis = adapter.compute_coriolis_parameter()
                self.register_buffer('coriolis', torch.from_numpy(coriolis).float())
            else:
                self.coriolis = None
            
            # Precompute bathymetry tensor
            if adapter.bathymetry is not None:
                self.register_buffer(
                    'bathymetry',
                    torch.from_numpy(adapter.bathymetry).float()
                )
            else:
                self.bathymetry = None
        
        def forward(
            self,
            eta: torch.Tensor,  # (batch, N) or (N,) surface elevation
            u: torch.Tensor,    # (batch, N, 2) or (N, 2) velocity
            d_eta_dt: torch.Tensor,  # Time derivative of eta
            d_u_dt: torch.Tensor,    # Time derivative of u
            grad_eta: torch.Tensor,  # Spatial gradient of eta (N, 2)
            div_hu: torch.Tensor,    # Divergence of h*u (N,)
            advection: Optional[torch.Tensor] = None,  # (u·∇)u term
        ) -> dict:
            """
            Compute shallow water equation residuals.
            
            Returns:
                losses: Dict with 'continuity', 'momentum_u', 'momentum_v', 'total'
            """
            # Total depth: H (bathymetry) + η (elevation)
            if self.bathymetry is not None:
                H = self.bathymetry.unsqueeze(0) if eta.dim() > 1 else self.bathymetry
                h = H + eta
            else:
                h = eta  # Assume eta is total depth if no bathymetry
            
            # Continuity equation: ∂η/∂t + ∇·(h*u) = 0
            continuity_residual = d_eta_dt + div_hu
            
            # Momentum equations
            # ∂u/∂t + (u·∇)u + f×u + g∇η = friction terms
            
            # Pressure gradient: g∇η
            pressure_grad = self.gravity * grad_eta  # (N, 2) or (batch, N, 2)
            
            # Momentum residual (without advection and Coriolis for now)
            momentum_residual = d_u_dt + pressure_grad
            
            # Add Coriolis: f×u = (-f*v, f*u) for 2D
            if self.use_coriolis and self.coriolis is not None:
                f = self.coriolis
                if u.dim() == 3:  # (batch, N, 2)
                    f = f.unsqueeze(0).unsqueeze(-1)
                elif u.dim() == 2:  # (N, 2)
                    f = f.unsqueeze(-1)
                
                coriolis_term = torch.stack([
                    -f[..., 0] * u[..., 1],  # -f*v
                    f[..., 0] * u[..., 0],   # f*u
                ], dim=-1)
                momentum_residual = momentum_residual + coriolis_term
            
            # Add advection if provided
            if advection is not None:
                momentum_residual = momentum_residual + advection
            
            # Bottom friction: -C_d |u| u / h
            if self.bottom_drag > 0:
                u_mag = torch.norm(u, dim=-1, keepdim=True) + 1e-6
                h_expanded = h.unsqueeze(-1) if h.dim() < u.dim() else h
                friction = self.bottom_drag * u_mag * u / h_expanded
                momentum_residual = momentum_residual + friction
            
            # Compute losses
            continuity_loss = torch.mean(continuity_residual ** 2)
            momentum_u_loss = torch.mean(momentum_residual[..., 0] ** 2)
            momentum_v_loss = torch.mean(momentum_residual[..., 1] ** 2)
            
            total_loss = continuity_loss + momentum_u_loss + momentum_v_loss
            
            return {
                'continuity': continuity_loss,
                'momentum_u': momentum_u_loss,
                'momentum_v': momentum_v_loss,
                'total': total_loss,
                'residuals': {
                    'continuity': continuity_residual,
                    'momentum': momentum_residual,
                }
            }
        
        def compute_energy(
            self,
            eta: torch.Tensor,
            u: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute total (kinetic + potential) energy.
            
            E = (1/2) ∫ (h|u|² + g η²) dA
            """
            if self.bathymetry is not None:
                H = self.bathymetry
                h = H + eta
            else:
                h = eta
            
            kinetic = 0.5 * h * torch.sum(u ** 2, dim=-1)
            potential = 0.5 * self.gravity * eta ** 2
            
            return torch.mean(kinetic + potential)
    
    
    def create_coastal_pipeline(
        adapter: CoastalManifoldAdapter,
        model_type: str = 'atlas_pinn',
        n_charts: int = 16,
        hidden_dims: List[int] = [64, 64, 64],
        use_spectral: bool = False,
        n_eigenpairs: int = 50,
    ) -> Tuple[nn.Module, ShallowWaterManifoldLoss]:
        """
        Create a complete coastal simulation pipeline.
        
        This builds:
        1. Neural network model (AtlasPINN or SpectralConv)
        2. Physics-informed loss function
        3. Preprocessing utilities
        
        Args:
            adapter: CoastalManifoldAdapter with mesh and config
            model_type: 'atlas_pinn', 'spectral', or 'tangent_gnn'
            n_charts: Number of charts for atlas models
            hidden_dims: Hidden layer dimensions
            use_spectral: Whether to use spectral convolutions
            n_eigenpairs: Number of spectral basis functions
            
        Returns:
            model: Neural network for predicting (η, u, v)
            loss_fn: ShallowWaterManifoldLoss
        """
        if adapter.mesh is None:
            raise ValueError("Adapter must have a mesh")
        
        # Create model based on type
        if model_type == 'atlas_pinn':
            from ..charts import AtlasPINN
            
            atlas = adapter.create_chart_atlas(n_charts=n_charts)
            
            model = AtlasPINN(
                atlas=atlas,
                input_dim=2,  # (x, y) local coords
                hidden_dims=hidden_dims,
                output_dim=3,  # (η, u, v)
            )
        
        elif model_type == 'spectral':
            from ..spectral import SpectralConvStack
            
            basis = adapter.create_spectral_basis(n_eigenpairs=n_eigenpairs)
            
            # Create spectral convolution stack
            model = SpectralConvStack(
                in_channels=2,  # Input features
                hidden_channels=hidden_dims[0],
                out_channels=3,  # (η, u, v)
                n_layers=len(hidden_dims),
                eigenvectors=torch.from_numpy(basis.eigenvectors).float(),
                eigenvalues=torch.from_numpy(basis.eigenvalues).float(),
            )
        
        elif model_type == 'tangent_gnn':
            from ..tangent import TangentMessagePassingStack
            
            # Build edge index from mesh
            edge_index = torch.from_numpy(adapter.mesh.edges.T).long()
            
            model = TangentMessagePassingStack(
                in_channels=2,
                hidden_channels=hidden_dims[0],
                out_channels=3,
                n_layers=len(hidden_dims),
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create loss function
        loss_fn = ShallowWaterManifoldLoss(
            adapter=adapter,
            gravity=adapter.config.gravity,
            use_coriolis=True,
        )
        
        return model, loss_fn

else:
    # No-torch placeholders
    class ShallowWaterManifoldLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for ShallowWaterManifoldLoss")
    
    def create_coastal_pipeline(*args, **kwargs):
        raise ImportError("PyTorch required for create_coastal_pipeline")
