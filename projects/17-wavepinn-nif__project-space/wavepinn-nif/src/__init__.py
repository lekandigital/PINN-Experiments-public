"""
WavePINN-NIF-Scalar: Physics-Informed Neural Implicit Field Solver
for the 2D/3D Acoustic Wave Equation in Heterogeneous Media.

This package provides:
- Synthetic data generation for slowness maps and wave sources
- Neural implicit field models using JAX/Haiku
- Physics-informed training with PDE, BC, and IC losses
- Inverse problem solving for slowness tomography
- Visualization utilities for wavefields and residuals
"""

from . import data_gen
from . import model
from . import training
from . import inverse
from . import utils

__version__ = "0.1.0"
__author__ = "WavePINN Team"
