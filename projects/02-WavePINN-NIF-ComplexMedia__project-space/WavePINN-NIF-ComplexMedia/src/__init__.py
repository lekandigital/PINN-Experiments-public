"""
WavePINN-NIF-ComplexMedia: Physics-Informed Neural Networks with Neural Implicit Fields
for acoustic wave propagation in heterogeneous media.
"""

__version__ = "0.1.0"
__author__ = "WavePINN Research Team"

from .model import create_model, WavePINN, MediaNIF, FourierFeatureEncoder
from .data_generator import SyntheticWaveData
from .physics_loss import WavePDELoss
from .trainer import WavePINNTrainer

__all__ = [
    "create_model",
    "WavePINN",
    "MediaNIF", 
    "FourierFeatureEncoder",
    "SyntheticWaveData",
    "WavePDELoss",
    "WavePINNTrainer",
]
