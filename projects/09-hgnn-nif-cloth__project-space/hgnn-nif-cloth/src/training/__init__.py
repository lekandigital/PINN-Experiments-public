# Training utilities for HGNN-NIF-Cloth
"""
Training pipeline components.

Components:
- PhysicsLoss: Multi-task physics loss (spring + SDF)
- Trainer: Training loop with curriculum learning
"""

from .losses import edge_spring_loss, sdf_reconstruction_loss, PhysicsLoss
from .trainer import Trainer

__all__ = [
    "edge_spring_loss",
    "sdf_reconstruction_loss", 
    "PhysicsLoss",
    "Trainer",
]
