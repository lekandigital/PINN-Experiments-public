"""
Physics-aware loss functions for cloth simulation.

Exports:
    - PhysicsLossStack: Combined physics losses (stretch, bend, momentum, collision)
    - ScheduledSamplingLoss: Loss wrapper with scheduled sampling support
"""

from .physics_losses import (
    PhysicsLossStack,
    compute_stretch_loss,
    compute_bend_loss,
    compute_momentum_loss,
    compute_collision_loss,
    compute_eikonal_loss,
)
from .scheduled_sampling import ScheduledSamplingLoss

__all__ = [
    "PhysicsLossStack",
    "compute_stretch_loss",
    "compute_bend_loss",
    "compute_momentum_loss",
    "compute_collision_loss",
    "compute_eikonal_loss",
    "ScheduledSamplingLoss",
]
