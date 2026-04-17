"""
Core module for the Neural Simulation addon.

Contains utilities for mesh conversion, frame handling, and result application.
"""

from .mesh_bridge import (
    blender_mesh_to_numpy,
    numpy_displacements_to_blender,
    numpy_positions_to_blender,
    sdf_grid_to_blender_mesh,
    apply_joint_values_to_armature,
)
from .frame_handler import (
    register_frame_handler,
    unregister_frame_handler,
    on_frame_change,
)
from .request_builder import build_request
from .result_applier import apply_result
from .performance import PerformanceTracker

__all__ = [
    # Mesh bridge
    "blender_mesh_to_numpy",
    "numpy_displacements_to_blender",
    "numpy_positions_to_blender",
    "sdf_grid_to_blender_mesh",
    "apply_joint_values_to_armature",
    # Frame handler
    "register_frame_handler",
    "unregister_frame_handler",
    "on_frame_change",
    # Request/result
    "build_request",
    "apply_result",
    # Performance
    "PerformanceTracker",
]
