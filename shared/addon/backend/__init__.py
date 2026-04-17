"""
Neural Simulation Backend System

This module provides the abstract interface and management system for
pluggable neural simulation backends in the Blender addon.
"""

from .interface import (
    ModelBackend,
    ModelCategory,
    OutputFormat,
    InputRequirement,
    BackendCapabilities,
    SimulationState,
    PredictionRequest,
    PredictionResult,
)
from .manager import BackendManager, get_manager

__all__ = [
    # Interface
    "ModelBackend",
    "ModelCategory",
    "OutputFormat",
    "InputRequirement",
    "BackendCapabilities",
    "SimulationState",
    "PredictionRequest",
    "PredictionResult",
    # Manager
    "BackendManager",
    "get_manager",
]
