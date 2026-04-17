"""
Domain-Specific Adapters for Differential Geometry Library
==========================================================

Adapters bridge the generic diffgeo infrastructure to specific
application domains:

- Coastal: Ocean dynamics, shallow water equations, bathymetry
- Cloth: Garment simulation, fabric physics, deformation

Each adapter provides:
1. Domain-specific coordinate handling
2. Physics-informed loss functions
3. Pre-built pipelines for common tasks
"""

from .coastal_adapter import (
    CoastalManifoldAdapter,
    ShallowWaterManifoldLoss,
    create_coastal_pipeline,
)
from .cloth_adapter import (
    ClothManifoldAdapter,
    ClothManifoldLoss,
    create_cloth_pipeline,
)

__all__ = [
    # Coastal
    'CoastalManifoldAdapter',
    'ShallowWaterManifoldLoss',
    'create_coastal_pipeline',
    # Cloth
    'ClothManifoldAdapter',
    'ClothManifoldLoss',
    'create_cloth_pipeline',
]
