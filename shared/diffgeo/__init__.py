"""
Differential Geometry Module for PINN-Experiments
==================================================

Shared discrete differential geometry operators extracted from Project 01 (GeoPINN-Manifold)
for use across multiple physics-informed neural network projects.

Core Capabilities:
- Discrete Exterior Calculus (DEC) operators on triangle meshes
- Cotangent Laplace-Beltrami operator
- Curvature computations (Gaussian, mean, principal)
- Chart atlas for multi-chart manifold handling
- Tangent-space operations and parallel transport
- Spectral graph convolutions (Laplacian eigenbasis, Chebyshev)
- Geodesic distance computation (heat method, fast marching)
- Map projections for geographic coordinates

Domain Adapters:
- Coastal: Shallow water equations, Coriolis, bathymetry (Projects 06, 16)
- Cloth: Garment physics, strain/bending energy, materials (Projects 05, 09, 11)

Supported Backends:
- NumPy/SciPy (default, for operator construction)
- PyTorch (for differentiable operations in Projects 01, 06, 09)
- JAX (for differentiable operations in Project 16)

Quick Start:
    >>> from diffgeo.mesh import TriangleMesh
    >>> from diffgeo.operators import laplace_beltrami
    >>>
    >>> # Create mesh from vertices and faces
    >>> mesh = TriangleMesh.from_vertices_faces(vertices, faces)
    >>>
    >>> # Apply Laplace-Beltrami to a scalar field
    >>> Lf = laplace_beltrami(mesh, f)

For coastal simulations:
    >>> from diffgeo.adapters import CoastalManifoldAdapter, ShallowWaterManifoldLoss
    >>> adapter = CoastalManifoldAdapter.from_bathymetry(grid, x, y)
    >>> model, loss_fn = create_coastal_pipeline(adapter)

For cloth simulations:
    >>> from diffgeo.adapters import ClothManifoldAdapter, ClothManifoldLoss
    >>> adapter = ClothManifoldAdapter.from_obj('garment.obj')
    >>> model, loss_fn = create_cloth_pipeline(adapter)

For GPU-accelerated operations:
    >>> from diffgeo.backends import TorchBackend
    >>> backend = TorchBackend(device='cuda')
    >>> Lf_torch = backend.apply_laplacian(mesh, f_tensor)
"""

__version__ = "0.2.0"

# Lazy imports to avoid requiring all backends
def __getattr__(name):
    # Mesh
    if name == "TriangleMesh":
        from .mesh.trimesh import TriangleMesh
        return TriangleMesh
    
    # Operators
    elif name == "laplace_beltrami":
        from .operators.laplace_beltrami import laplace_beltrami
        return laplace_beltrami
    elif name == "discrete_gradient":
        from .operators.gradient import discrete_gradient
        return discrete_gradient
    elif name == "discrete_divergence":
        from .operators.divergence import discrete_divergence
        return discrete_divergence
    
    # Curvature
    elif name == "gaussian_curvature":
        from .curvature.gaussian import gaussian_curvature
        return gaussian_curvature
    elif name == "mean_curvature":
        from .curvature.mean import mean_curvature
        return mean_curvature
    elif name == "mean_curvature_vector":
        from .curvature.mean import mean_curvature_vector
        return mean_curvature_vector
    elif name == "principal_curvatures":
        from .curvature.principal import principal_curvatures
        return principal_curvatures
    
    # Backends
    elif name == "get_backend":
        from .backends.base import get_backend
        return get_backend
    elif name == "TorchBackend":
        from .backends.torch_backend import TorchBackend
        return TorchBackend
    elif name == "JAXBackend":
        from .backends.jax_backend import JAXBackend
        return JAXBackend
    
    # Charts (NEW)
    elif name == "Atlas":
        from .charts import Atlas
        return Atlas
    elif name == "TorchAtlas":
        from .charts import TorchAtlas
        return TorchAtlas
    elif name == "ChartMLP":
        from .charts import ChartMLP
        return ChartMLP
    elif name == "AtlasPINN":
        from .charts import AtlasPINN
        return AtlasPINN
    
    # Spectral (NEW)
    elif name == "SpectralBasis":
        from .spectral import SpectralBasis
        return SpectralBasis
    elif name == "compute_laplacian_eigenpairs":
        from .spectral import compute_laplacian_eigenpairs
        return compute_laplacian_eigenpairs
    elif name == "SpectralGraphConv":
        from .spectral import SpectralGraphConv
        return SpectralGraphConv
    elif name == "ChebConv":
        from .spectral import ChebConv
        return ChebConv
    
    # Tangent (NEW)
    elif name == "compute_tangent_basis":
        from .tangent import compute_tangent_basis
        return compute_tangent_basis
    elif name == "parallel_transport":
        from .tangent import parallel_transport
        return parallel_transport
    elif name == "TangentMessagePassing":
        from .tangent import TangentMessagePassing
        return TangentMessagePassing
    
    # Geodesic (NEW)
    elif name == "geodesic_distance_heat":
        from .geodesic import geodesic_distance_heat
        return geodesic_distance_heat
    elif name == "geodesic_distance_dijkstra":
        from .geodesic import geodesic_distance_dijkstra
        return geodesic_distance_dijkstra
    elif name == "geodesic_distance_fast_marching":
        from .geodesic import geodesic_distance_fast_marching
        return geodesic_distance_fast_marching
    
    # Projection (NEW)
    elif name == "latlon_to_xyz":
        from .projection import latlon_to_xyz
        return latlon_to_xyz
    elif name == "gnomonic_projection":
        from .projection import gnomonic_projection
        return gnomonic_projection
    elif name == "compute_coriolis_parameter":
        from .projection import compute_coriolis_parameter
        return compute_coriolis_parameter
    
    # Adapters (NEW)
    elif name == "CoastalManifoldAdapter":
        from .adapters import CoastalManifoldAdapter
        return CoastalManifoldAdapter
    elif name == "ShallowWaterManifoldLoss":
        from .adapters import ShallowWaterManifoldLoss
        return ShallowWaterManifoldLoss
    elif name == "create_coastal_pipeline":
        from .adapters import create_coastal_pipeline
        return create_coastal_pipeline
    elif name == "ClothManifoldAdapter":
        from .adapters import ClothManifoldAdapter
        return ClothManifoldAdapter
    elif name == "ClothManifoldLoss":
        from .adapters import ClothManifoldLoss
        return ClothManifoldLoss
    elif name == "create_cloth_pipeline":
        from .adapters import create_cloth_pipeline
        return create_cloth_pipeline
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Mesh
    "TriangleMesh",
    
    # Operators
    "laplace_beltrami",
    "discrete_gradient", 
    "discrete_divergence",
    
    # Curvature
    "gaussian_curvature",
    "mean_curvature",
    "mean_curvature_vector",
    "principal_curvatures",
    
    # Backends
    "get_backend",
    "TorchBackend",
    "JAXBackend",
    
    # Charts (NEW in v0.2.0)
    "Atlas",
    "TorchAtlas",
    "ChartMLP",
    "AtlasPINN",
    
    # Spectral (NEW in v0.2.0)
    "SpectralBasis",
    "compute_laplacian_eigenpairs",
    "SpectralGraphConv",
    "ChebConv",
    
    # Tangent (NEW in v0.2.0)
    "compute_tangent_basis",
    "parallel_transport",
    "TangentMessagePassing",
    
    # Geodesic (NEW in v0.2.0)
    "geodesic_distance_heat",
    "geodesic_distance_dijkstra",
    "geodesic_distance_fast_marching",
    
    # Projection (NEW in v0.2.0)
    "latlon_to_xyz",
    "gnomonic_projection",
    "compute_coriolis_parameter",
    
    # Adapters (NEW in v0.2.0)
    "CoastalManifoldAdapter",
    "ShallowWaterManifoldLoss",
    "create_coastal_pipeline",
    "ClothManifoldAdapter",
    "ClothManifoldLoss",
    "create_cloth_pipeline",
]
