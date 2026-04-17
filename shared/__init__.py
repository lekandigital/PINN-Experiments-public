"""
Shared physics-informed neural network components for PINN-Experiments.

This library provides reusable physics-encoded graph convolution layers
that hard-code known physical laws into the network architecture, letting
the network learn only corrections and residuals.

Also includes a unified collision detection and response system for
cloth simulation projects.
"""

from . import physics_conv
from . import collision

__version__ = "0.1.0"
__all__ = ["physics_conv", "collision"]
