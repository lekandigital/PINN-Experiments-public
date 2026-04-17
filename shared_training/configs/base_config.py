"""
Base Configuration Classes for Rollout Training

Provides dataclass-based configuration with sensible defaults
for long-horizon rollout training across different project types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ScheduleConfig:
    """Configuration for epsilon schedule."""
    schedule_type: str = 'exponential'
    eps_start: float = 1.0
    eps_min: float = 0.1
    decay_rate: float = 0.95
    warmup_epochs: int = 0


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    enabled: bool = True
    min_rollout: int = 4
    max_rollout: int = 256
    growth_rate: float = 1.2
    stages: Optional[List[Tuple[float, int]]] = None  # (epoch_frac, length)


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    reconstruction_weight: float = 1.0
    physics_weights: Dict[str, float] = field(default_factory=dict)
    free_running_multiplier: float = 2.0
    temporal_discount: float = 0.99


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    gradient_clip: float = 1.0
    use_amp: bool = True


@dataclass
class BaseRolloutConfig:
    """Base configuration for rollout training.
    
    Combines all sub-configurations into a unified config object.
    Inherit from this to create project-specific configs.
    """
    # Sub-configs
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Training params
    batch_size: int = 8
    num_epochs: int = 200
    eval_every: int = 10
    checkpoint_every: int = 20
    
    # Data params
    sequence_length: int = 128
    dt: float = 1.0 / 30.0  # 30 FPS default
    
    # Model params (project-specific)
    hidden_dim: int = 256
    num_layers: int = 4
    
    # Hardware
    device: str = 'cuda'
    num_workers: int = 4
    
    def to_dict(self) -> Dict:
        """Convert config to flat dictionary for logging."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                for sub_key, sub_value in value.__dict__.items():
                    result[f'{key}.{sub_key}'] = sub_value
            else:
                result[key] = value
        return result
