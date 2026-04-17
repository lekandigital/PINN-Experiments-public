"""
UI Module for Neural Simulation addon.

Contains all Blender panels, operators, and property definitions.
"""

from .properties import NeuralSimProperties, NeuralSimJointMapping
from .panels import (
    NEURALSIM_PT_BackendPanel,
    NEURALSIM_PT_TargetPanel,
    NEURALSIM_PT_ForcesPanel,
    NEURALSIM_PT_MaterialPanel,
    NEURALSIM_PT_ResolutionPanel,
    NEURALSIM_PT_PlaybackPanel,
)
from .operators import (
    NEURALSIM_OT_LoadBackend,
    NEURALSIM_OT_Play,
    NEURALSIM_OT_Pause,
    NEURALSIM_OT_Reset,
    NEURALSIM_OT_StepForward,
    NEURALSIM_OT_StepBack,
    NEURALSIM_OT_BakeToKeyframes,
    NEURALSIM_OT_RefreshBackends,
)

# All classes to register with Blender
classes = [
    # Properties (must be first)
    NeuralSimJointMapping,
    NeuralSimProperties,
    # Operators
    NEURALSIM_OT_LoadBackend,
    NEURALSIM_OT_Play,
    NEURALSIM_OT_Pause,
    NEURALSIM_OT_Reset,
    NEURALSIM_OT_StepForward,
    NEURALSIM_OT_StepBack,
    NEURALSIM_OT_BakeToKeyframes,
    NEURALSIM_OT_RefreshBackends,
    # Panels
    NEURALSIM_PT_BackendPanel,
    NEURALSIM_PT_TargetPanel,
    NEURALSIM_PT_ForcesPanel,
    NEURALSIM_PT_MaterialPanel,
    NEURALSIM_PT_ResolutionPanel,
    NEURALSIM_PT_PlaybackPanel,
]

__all__ = [
    "NeuralSimProperties",
    "NeuralSimJointMapping",
    "classes",
]
