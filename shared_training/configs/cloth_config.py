"""
Cloth Simulation Configuration

Specialized configs for cloth simulation projects:
- Project 05: ClothGNN
- Project 08: HGNN-ClothDyn
- Project 09: HGNN-NIF-Cloth
- Project 12: NIF-Cloth4D-Temporal
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


# Predefined curriculum stages for cloth
CLOTH_CURRICULUM_STAGES = [
    (0.0, 4),     # Start: 4 frames
    (0.15, 8),    # 15%: 8 frames
    (0.30, 16),   # 30%: 16 frames
    (0.50, 32),   # 50%: 32 frames
    (0.70, 64),   # 70%: 64 frames
    (0.85, 128),  # 85%: 128 frames
    (0.95, 256),  # 95%: 256 frames (full)
]


@dataclass
class ClothPhysicsConfig:
    """Physics-specific config for cloth simulation."""
    # Material properties
    stiffness: float = 1000.0
    damping: float = 0.99
    mass: float = 1.0
    
    # Collision
    collision_margin: float = 0.01
    collision_stiffness: float = 5000.0
    
    # Physics loss weights
    edge_strain_weight: float = 0.1
    velocity_smooth_weight: float = 0.01
    collision_weight: float = 1.0
    
    # Time integration
    dt: float = 1.0 / 30.0
    substeps: int = 1


@dataclass
class ClothSimulationConfig(BaseRolloutConfig):
    """Configuration for cloth simulation projects."""
    
    # Override defaults for cloth
    schedule: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(
        schedule_type='exponential',
        eps_start=1.0,
        eps_min=0.2,
        decay_rate=0.95,
    ))
    
    curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(
        enabled=True,
        min_rollout=4,
        max_rollout=256,
        growth_rate=1.2,
        stages=CLOTH_CURRICULUM_STAGES,
    ))
    
    loss: LossConfig = field(default_factory=lambda: LossConfig(
        reconstruction_weight=1.0,
        physics_weights={
            'edge_strain': 0.1,
            'velocity_smooth': 0.01,
            'collision': 1.0,
        },
        free_running_multiplier=2.0,
    ))
    
    # Cloth-specific
    physics: ClothPhysicsConfig = field(default_factory=ClothPhysicsConfig)
    
    # Training overrides
    batch_size: int = 4  # Cloth sequences are memory-heavy
    sequence_length: int = 256
    dt: float = 1.0 / 30.0
    
    # Model size
    hidden_dim: int = 128
    num_layers: int = 3


@dataclass  
class Project05Config(ClothSimulationConfig):
    """Config for Project 05: ClothGNN."""
    # GNN-specific
    gnn_layers: int = 4
    message_passing_steps: int = 3
    use_world_edges: bool = True
    

@dataclass
class Project08Config(ClothSimulationConfig):
    """Config for Project 08: HGNN-ClothDyn."""
    # Hierarchical GNN-specific
    num_hierarchy_levels: int = 3
    pooling_ratio: float = 0.5
    use_physics_encoded_edges: bool = True


@dataclass
class Project09Config(ClothSimulationConfig):
    """Config for Project 09: HGNN-NIF-Cloth."""
    # INR-specific
    siren_hidden_dim: int = 128
    siren_layers: int = 3
    omega_0: float = 30.0
    frame_history: int = 4
    
    # Override schedule for NIF
    schedule: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(
        schedule_type='cosine',
        eps_start=1.0,
        eps_min=0.15,
    ))


@dataclass
class Project12Config(ClothSimulationConfig):
    """Config for Project 12: NIF-Cloth4D-Temporal."""
    # Temporal GRU + INR specific
    gru_hidden_dim: int = 256
    gru_layers: int = 2
    fourier_features: int = 64
    time_encoding_dim: int = 32
    
    # Uses longer sequences
    sequence_length: int = 512
