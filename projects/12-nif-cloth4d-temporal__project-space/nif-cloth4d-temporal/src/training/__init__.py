"""
Training utilities for NIF-Cloth4D-Temporal.

Exports:
    - ScheduledSamplingTrainer: Main trainer with AMP and scheduled sampling
    - TrainingConfig: Hyperparameter configuration dataclass
"""

from .trainer import ScheduledSamplingTrainer
from .config import TrainingConfig

__all__ = [
    "ScheduledSamplingTrainer",
    "TrainingConfig",
]
