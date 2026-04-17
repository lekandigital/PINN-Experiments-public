"""
Physics-Encoded Graph Convolution Layers.

This module provides graph convolution layers that encode known physical laws
directly into the message-passing computation, with the network learning only
corrections to the analytical physics.

Design Principle:
    Total message = analytical_physics_term + learned_correction_term

Classes:
    PhysicsEncodedConv: Abstract base class for all physics-encoded convolutions
    ClothForceConv: Hooke's law for cloth simulation (stretch + bending)
    ElasticForceConv: 3D elasticity for deformable bodies
    LiteClothConv: Lightweight cloth physics for small models
    ShallowWaterConv: Shallow water equations for coastal flow

Integrators:
    ExplicitEuler: Simple forward Euler integration
    SemiImplicitEuler: Symplectic Euler (better energy conservation)
    VelocityVerlet: Second-order symplectic integrator
"""

from .base import PhysicsEncodedConv, PhysicsConvConfig
from .cloth_conv import ClothForceConv, ClothConvConfig
from .deform_conv import ElasticForceConv
from .lightweight_conv import LiteClothConv, PurePhysicsClothConv
from .shallow_water_conv import ShallowWaterConv
from .integrators import ExplicitEuler, SemiImplicitEuler, VelocityVerlet

__all__ = [
    # Base
    "PhysicsEncodedConv",
    "PhysicsConvConfig",
    # Domain-specific
    "ClothForceConv",
    "ClothConvConfig",
    "ElasticForceConv",
    "LiteClothConv",
    "PurePhysicsClothConv",
    "ShallowWaterConv",
    # Integrators
    "ExplicitEuler",
    "SemiImplicitEuler",
    "VelocityVerlet",
]
