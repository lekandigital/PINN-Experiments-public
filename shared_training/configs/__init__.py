"""
Configuration Classes for Rollout Training

Provides dataclass-based configurations with domain-specific defaults.
"""

from .base_config import (
    BaseRolloutConfig,
    ScheduleConfig,
    CurriculumConfig,
    LossConfig,
    OptimizerConfig,
)

from .cloth_config import (
    ClothSimulationConfig,
    ClothPhysicsConfig,
    Project05Config,
    Project08Config,
    Project09Config,
    Project12Config,
    CLOTH_CURRICULUM_STAGES,
)

from .motion_config import (
    MotionPredictionConfig,
    MotionPhysicsConfig,
    Project07Config,
    MOTION_CURRICULUM_STAGES,
)

from .coastal_config import (
    CoastalFlowConfig,
    CoastalPhysicsConfig,
    COASTAL_CURRICULUM_STAGES,
)


__all__ = [
    # Base
    'BaseRolloutConfig',
    'ScheduleConfig',
    'CurriculumConfig',
    'LossConfig',
    'OptimizerConfig',
    # Cloth
    'ClothSimulationConfig',
    'ClothPhysicsConfig',
    'Project05Config',
    'Project08Config',
    'Project09Config', 
    'Project12Config',
    'CLOTH_CURRICULUM_STAGES',
    # Motion
    'MotionPredictionConfig',
    'MotionPhysicsConfig',
    'Project07Config',
    'MOTION_CURRICULUM_STAGES',
    # Coastal
    'CoastalFlowConfig',
    'CoastalPhysicsConfig',
    'COASTAL_CURRICULUM_STAGES',
]
