"""
NIF-Cloth3D-Interactive: Neural Implicit Field for Real-Time Cloth Simulation

A SIREN-based neural network for predicting cloth deformations in real-time.
"""

__version__ = "0.1.0"
__author__ = "NIF-Cloth3D Team"

from .model import SineMLP, SineLayer, ConditionedSineMLP, create_model
from .losses import StretchLoss, BendLoss, MomentumLoss, PhysicsLoss

__all__ = [
    "SineMLP",
    "SineLayer",
    "ConditionedSineMLP",
    "create_model",
    "StretchLoss",
    "BendLoss",
    "MomentumLoss",
    "PhysicsLoss",
]
