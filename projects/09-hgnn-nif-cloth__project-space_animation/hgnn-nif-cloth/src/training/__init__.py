# Training utilities for HGNN-NIF-Cloth
"""
Training pipeline components.

Components:
- PhysicsLoss: Multi-task physics loss (spring + SDF)
- TemporalPhysicsLoss: Temporal losses for animation training
- Trainer: Training loop with curriculum learning
"""

from .losses import (
    edge_spring_loss,
    sdf_reconstruction_loss,
    PhysicsLoss,
    CurriculumWeightScheduler,
    temporal_consistency_loss,
    physics_velocity_loss,
    momentum_conservation_loss,
    inertia_regularization,
    TemporalPhysicsLoss,
)
from .trainer import Trainer

__all__ = [
    "edge_spring_loss",
    "sdf_reconstruction_loss",
    "PhysicsLoss",
    "CurriculumWeightScheduler",
    "temporal_consistency_loss",
    "physics_velocity_loss",
    "momentum_conservation_loss",
    "inertia_regularization",
    "TemporalPhysicsLoss",
    "Trainer",
]
