"""
Stage wrappers for the animation pipeline.

Each stage wraps a PINN-Experiments project model:
- MotionStage: Project 07 (Geom-INR-Motion) - continuous-time skeleton
- BodyStage: Project 14 (PEGNN-Deform) - LBS + soft tissue
- ClothStage: Project 09 (HGNN-NIF-Cloth) - cloth simulation
"""

from .motion_stage import MotionStage, MotionOutput, JointRemapper
from .body_stage import BodyStage, BodyOutput
from .cloth_stage import ClothStage, ClothOutput, ClothGraphBuilder

__all__ = [
    "MotionStage", "MotionOutput", "JointRemapper",
    "BodyStage", "BodyOutput",
    "ClothStage", "ClothOutput", "ClothGraphBuilder",
]
