"""
Core domain-agnostic components for geodesic trajectory prediction.

This module provides the foundational abstractions that can be specialized
for any domain involving trajectory prediction on potential fields.
"""

from .potential_field import PotentialFieldBase, LearnedPotentialField
from .geodesic_loss import GeodesicLoss, ConstantSpeedLoss, BoundaryLoss, GradientFollowingLoss
from .trajectory_pinn import TrajectoryPINN

__all__ = [
    # Potential fields
    "PotentialFieldBase",
    "LearnedPotentialField",
    # Losses
    "GeodesicLoss",
    "ConstantSpeedLoss", 
    "BoundaryLoss",
    "GradientFollowingLoss",
    # Networks
    "TrajectoryPINN",
]
