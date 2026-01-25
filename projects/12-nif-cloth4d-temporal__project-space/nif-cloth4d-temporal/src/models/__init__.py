"""
Neural network models for NIF-Cloth4D-Temporal.

Exports:
    - SIRENLayer: SIREN layer with sinusoidal activation
    - FourierFeatureMLP: Main neural implicit field Φθ(x,y,z,t)
    - TemporalGRU: Optional GRU for temporal conditioning
"""

from .siren import SIRENLayer, SIRENNetwork
from .fourier_mlp import FourierFeatureMLP
from .temporal_gru import TemporalGRU

__all__ = [
    "SIRENLayer",
    "SIRENNetwork", 
    "FourierFeatureMLP",
    "TemporalGRU",
]
