"""
Implicit Fields Library for Neural Implicit Representations.

A consolidated library providing SIREN networks, Fourier feature encoding,
physics-informed losses, and mesh extraction utilities for neural implicit fields.

This library supports:
- SIREN (Sinusoidal Representation Networks) with proper initialization
- Fourier feature positional encoding
- Eikonal and collision losses for SDF training
- Progressive marching cubes mesh extraction
- NeRF-style positional encoding

Example:
    >>> from implicit_fields import SirenNetwork, eikonal_loss, gradient, extract_mesh
    >>> 
    >>> # Create a SIREN network for SDF
    >>> model = SirenNetwork(
    ...     in_features=3,
    ...     hidden_features=256,
    ...     hidden_layers=5,
    ...     out_features=1
    ... )
    >>> 
    >>> # Training loop
    >>> coords = torch.randn(1000, 3, requires_grad=True)
    >>> sdf = model(coords)
    >>> grads = gradient(sdf, coords)
    >>> loss = eikonal_loss(grads)
    >>> 
    >>> # Extract mesh after training
    >>> vertices, faces = extract_mesh(
    ...     lambda x: model(x),
    ...     bounds=(torch.tensor([-1,-1,-1]), torch.tensor([1,1,1]))
    ... )
"""

__version__ = "1.0.0"
__author__ = "PINN-Experiments Team"

# Core SIREN networks
from .siren import (
    SineActivation,
    SirenLayer,
    SirenNetwork,
    LatentConditionedSiren,
    ModulatedSirenLayer,
)

# Fourier feature encoding
from .fourier_features import (
    FourierFeatureEncoding,
    FourierFeatureMLP,
    MultiScaleFourierFeatures,
)

# Physics-informed losses
from .losses import (
    gradient,
    eikonal_loss,
    sdf_collision_loss,
    sdf_boundary_loss,
    divergence_free_loss,
    laplacian_loss,
    CombinedSDFLoss,
)

# Mesh extraction
from .mesh_extraction import (
    extract_mesh,
    extract_mesh_mcubes,
    laplacian_smooth,
    taubin_smooth,
    export_mesh_obj,
    export_mesh_ply,
    compute_mesh_quality,
    MeshExtractionConfig,
    MeshExtractor,
)

# Positional encoding
from .encoding import (
    positional_encoding,
    PositionalEncoding,
    IntegratedPositionalEncoding,
    compute_encoding_dim,
)

__all__ = [
    # Version
    "__version__",
    # SIREN
    "SineActivation",
    "SirenLayer",
    "SirenNetwork",
    "LatentConditionedSiren",
    "ModulatedSirenLayer",
    # Fourier features
    "FourierFeatureEncoding",
    "FourierFeatureMLP",
    "MultiScaleFourierFeatures",
    # Losses
    "gradient",
    "eikonal_loss",
    "sdf_collision_loss",
    "sdf_boundary_loss",
    "divergence_free_loss",
    "laplacian_loss",
    "CombinedSDFLoss",
    # Mesh extraction
    "extract_mesh",
    "extract_mesh_mcubes",
    "laplacian_smooth",
    "taubin_smooth",
    "export_mesh_obj",
    "export_mesh_ply",
    "compute_mesh_quality",
    "MeshExtractionConfig",
    "MeshExtractor",
    # Encoding
    "positional_encoding",
    "PositionalEncoding",
    "IntegratedPositionalEncoding",
    "compute_encoding_dim",
]
