"""
Shared Training Library for Long-Horizon Rollout Stabilization

This library provides model-agnostic scheduled sampling and curriculum learning
utilities for training autoregressive models that need stable long-horizon rollouts.

Key components:
- ScheduledSampler: Epsilon-greedy teacher forcing with configurable decay schedules
- CurriculumRolloutScheduler: Progressive rollout length extension
- RolloutModelAdapter: Abstract interface for different model architectures
- RolloutMetrics: Stability horizon, drift rate, and energy conservation metrics

Designed for physics-informed ML projects including:
- Cloth simulation (GNN, NIF, hybrid architectures)
- Motion prediction (SIREN/INR with continuous time)
- Coastal flow simulation (multi-physics GNN)
- Wave propagation (Fourier feature networks)

Usage:
    from shared_training import (
        ScheduledSampler,
        CurriculumRolloutScheduler,
        LinearDecaySchedule,
        ExponentialDecaySchedule,
        RolloutMetrics,
    )
    from shared_training.adapters import GRUAdapter, INRAdapter
    from shared_training.configs import ClothConfig, MotionConfig
"""

from .schedules import (
    EpsilonSchedule,
    LinearDecaySchedule,
    ExponentialDecaySchedule,
    CosineAnnealSchedule,
    StepDecaySchedule,
    ConstantSchedule,
    InverseSigmoidSchedule,
    LinearRampSchedule,
)

from .scheduled_sampling import (
    ScheduledSampler,
    ScheduledSamplingTrainer,
    TemporalLossAccumulator,
)

from .curriculum_rollout import (
    CurriculumRolloutScheduler,
    CurriculumStage,
    CombinedCurriculum,
)

from .metrics import (
    RolloutMetrics,
    compute_rollout_mse_curve,
    compute_drift_rate,
    compute_stability_horizon,
    compute_energy_conservation,
    compute_spectral_divergence,
)

__version__ = "0.1.0"
__all__ = [
    # Schedules
    "EpsilonSchedule",
    "LinearDecaySchedule",
    "ExponentialDecaySchedule",
    "CosineAnnealSchedule",
    "StepDecaySchedule",
    "ConstantSchedule",
    "InverseSigmoidSchedule",
    "LinearRampSchedule",
    # Sampling
    "ScheduledSampler",
    "ScheduledSamplingTrainer",
    "TemporalLossAccumulator",
    # Curriculum
    "CurriculumRolloutScheduler",
    "CurriculumStage",
    "CombinedCurriculum",
    # Metrics
    "RolloutMetrics",
    "compute_rollout_mse_curve",
    "compute_drift_rate",
    "compute_stability_horizon",
    "compute_energy_conservation",
    "compute_spectral_divergence",
]
