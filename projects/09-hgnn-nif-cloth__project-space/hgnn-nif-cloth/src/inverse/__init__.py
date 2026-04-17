"""
Inverse Design Module for HGNN-NIF-Cloth.

Provides optimization-based inverse design and pose matching capabilities,
ported from Project 04 (ClothGeom-NIF) with enhancements for temporal models.
"""

from .pose_matching import (
    InverseDesignConfig,
    InverseDesignOptimizer,
    PoseMatchingOptimizer,
    interpolate_latents,
)

__all__ = [
    "InverseDesignConfig",
    "InverseDesignOptimizer",
    "PoseMatchingOptimizer",
    "interpolate_latents",
]
