"""
NIF-Cloth4D-Temporal: Neural Implicit Field for Continuous-Time Cloth Simulation

A PyTorch implementation of neural implicit fields Φθ(x,y,z,t) for physics-based
cloth simulation using Fourier features, SIREN activations, and scheduled sampling.

Main Components:
    - models: FourierFeatureMLP with SIREN layers and optional GRU conditioning
    - losses: Physics-aware loss stack (stretch, bend, momentum, collision)
    - training: Scheduled sampling trainer with AMP support
    - data: PyFlex simulation and HDF5 dataset utilities
"""

__version__ = "0.1.0"
__author__ = "NIF-Cloth4D-Temporal Team"
