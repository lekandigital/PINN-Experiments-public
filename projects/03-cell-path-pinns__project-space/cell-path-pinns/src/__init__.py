"""
Geodesic Trajectory Library

A domain-agnostic framework for physics-informed trajectory prediction using
geodesic principles. Originally developed for microbe chemotaxis (Cell-Path-PINNs),
now generalized for multiple domains:

- Robotics: Path planning on terrain/traversability maps
- Migration: Animal movement on environmental gradients  
- Finance: Agent trajectories in risk/profit state space
- Game AI: NPC pathfinding on cost landscapes
- Biology: Microbe chemotaxis (original domain)

Core Concepts:
- PotentialField: Learnable scalar field agents move on
- GeodesicLoss: Physics losses enforcing geodesic properties
- TrajectoryPINN: Network predicting continuous trajectories x(t)
"""

from .core import (
    PotentialFieldBase,
    GeodesicLoss,
    TrajectoryPINN,
    ConstantSpeedLoss,
    BoundaryLoss,
)

__all__ = [
    "PotentialFieldBase",
    "GeodesicLoss", 
    "TrajectoryPINN",
    "ConstantSpeedLoss",
    "BoundaryLoss",
]
