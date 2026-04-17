"""
Coastal Flow Configuration

Specialized configs for coastal simulation projects:
- Project 06: CoastFlow-GNN (future)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .base_config import (
    BaseRolloutConfig,
    ScheduleConfig,
    CurriculumConfig,
    LossConfig,
    OptimizerConfig,
)


# Coastal simulations need very long rollouts (hours of storm surge)
COASTAL_CURRICULUM_STAGES = [
    (0.0, 6),       # Start: 6 timesteps (1 hour @ 10min)
    (0.15, 12),     # 15%: 2 hours
    (0.30, 24),     # 30%: 4 hours
    (0.50, 48),     # 50%: 8 hours
    (0.70, 96),     # 70%: 16 hours
    (0.85, 144),    # 85%: 24 hours
    (0.95, 288),    # 95%: 48 hours
]


@dataclass
class CoastalPhysicsConfig:
    """Physics config for coastal simulations."""
    # Shallow water equations
    gravity: float = 9.81
    manning_n: float = 0.025  # Manning roughness
    coriolis_f: float = 1e-4  # Coriolis parameter
    
    # Conservation laws
    mass_conservation_weight: float = 1.0
    momentum_conservation_weight: float = 0.5
    
    # Boundary conditions
    tidal_weight: float = 1.0
    wind_stress_weight: float = 0.1
    
    # Time
    dt_hours: float = 10.0 / 60.0  # 10 minute timesteps


@dataclass
class CoastalFlowConfig(BaseRolloutConfig):
    """Configuration for coastal flow simulation projects."""
    
    schedule: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(
        schedule_type='exponential',
        eps_start=1.0,
        eps_min=0.05,  # Very low for stable long rollouts
        decay_rate=0.98,  # Slower decay
    ))
    
    curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(
        enabled=True,
        min_rollout=6,
        max_rollout=288,
        growth_rate=1.15,  # Slower growth for stability
        stages=COASTAL_CURRICULUM_STAGES,
    ))
    
    loss: LossConfig = field(default_factory=lambda: LossConfig(
        reconstruction_weight=1.0,
        physics_weights={
            'mass_conservation': 1.0,
            'momentum': 0.5,
            'boundary': 0.1,
        },
        free_running_multiplier=3.0,  # High penalty for instability
        temporal_discount=0.995,  # Slow discount for long sequences
    ))
    
    physics: CoastalPhysicsConfig = field(default_factory=CoastalPhysicsConfig)
    
    # Coastal-specific
    batch_size: int = 2  # Large spatial domains
    sequence_length: int = 288  # 48 hours
    
    # Multi-scale GNN
    hidden_dim: int = 128
    num_scales: int = 4
