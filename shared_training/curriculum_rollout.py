"""
Curriculum-Based Rollout Length Extension

This module implements progressive rollout length scheduling to help models
learn stable long-horizon predictions. The key insight: a model that can
barely predict 1 step ahead will produce garbage on 256-step rollouts.
Instead, we gradually extend the rollout length as training progresses.

Stages (default for cloth simulation):
    Stage 1 (epochs 0-25%):   Train on 4-frame rollouts
    Stage 2 (epochs 25-50%):  Train on 16-frame rollouts
    Stage 3 (epochs 50-75%):  Train on 64-frame rollouts
    Stage 4 (epochs 75-100%): Train on 256-frame rollouts

This is analogous to curriculum learning in education — learn addition
before calculus.

Also incorporates Project 09's loss weight curriculum, which adjusts
physics loss weights over training (easy → hard).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import math


@dataclass
class CurriculumStage:
    """A single stage in the curriculum.
    
    Attributes:
        epoch_fraction: When this stage starts (0.0 = beginning, 1.0 = end)
        rollout_length: Number of frames for rollout in this stage
        loss_weights: Optional dict of loss weight overrides for this stage
    """
    epoch_fraction: float
    rollout_length: int
    loss_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if not 0.0 <= self.epoch_fraction <= 1.0:
            raise ValueError(f"epoch_fraction must be in [0, 1], got {self.epoch_fraction}")
        if self.rollout_length < 1:
            raise ValueError(f"rollout_length must be >= 1, got {self.rollout_length}")


class CurriculumRolloutScheduler:
    """Gradually increases rollout length during training.
    
    Motivation: A model that can barely predict 1 step ahead will produce
    garbage on 256-step rollouts, and training on that garbage teaches nothing
    useful. Instead, we start with short rollouts and gradually extend.
    
    Supports two transition modes:
    - 'step': Discrete jumps at stage boundaries (default)
    - 'smooth': Linear interpolation between stages
    
    Example (cloth simulation):
        >>> scheduler = CurriculumRolloutScheduler(
        ...     stages=[
        ...         CurriculumStage(0.0, 4),
        ...         CurriculumStage(0.25, 16),
        ...         CurriculumStage(0.5, 64),
        ...         CurriculumStage(0.75, 256),
        ...     ]
        ... )
        >>> scheduler.get_rollout_length(epoch=0, total_epochs=100)
        4
        >>> scheduler.get_rollout_length(epoch=50, total_epochs=100)
        64
    """
    
    # Predefined stage configurations
    CLOTH_STAGES = [
        CurriculumStage(0.0, 4),
        CurriculumStage(0.25, 16),
        CurriculumStage(0.5, 64),
        CurriculumStage(0.75, 256),
    ]
    
    COASTAL_STAGES = [
        CurriculumStage(0.0, 4),
        CurriculumStage(0.2, 16),
        CurriculumStage(0.4, 64),
        CurriculumStage(0.6, 256),
        CurriculumStage(0.8, 1024),
    ]
    
    MOTION_STAGES = [
        CurriculumStage(0.0, 8),
        CurriculumStage(0.3, 32),
        CurriculumStage(0.6, 128),
        CurriculumStage(0.85, 512),
    ]
    
    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
        transition: str = 'step',
        min_length: int = 4,
        max_length: int = 256,
        warmup_epochs: int = 0,
        growth_rate: Optional[float] = None,
    ):
        """
        Args:
            stages: List of CurriculumStage defining the curriculum.
                   If None, uses exponential growth from min_length to max_length.
            transition: 'step' for discrete jumps, 'smooth' for interpolation
            min_length: Minimum rollout length (used if stages is None)
            max_length: Maximum rollout length (used if stages is None)
            warmup_epochs: Stay at initial length for this many epochs
            growth_rate: If set (and stages is None), use exponential growth:
                        length = min_length * (growth_rate ^ effective_epoch)
        """
        self.transition = transition
        self.warmup_epochs = warmup_epochs
        self.min_length = min_length
        self.max_length = max_length
        self.growth_rate = growth_rate
        
        if stages is not None:
            self.stages = sorted(stages, key=lambda s: s.epoch_fraction)
        elif growth_rate is not None:
            # Exponential growth mode (from Project 12's CurriculumScheduler)
            self.stages = None
        else:
            # Default: use cloth stages
            self.stages = self.CLOTH_STAGES.copy()
        
        self._current_length = min_length
        self._current_weights: Dict[str, float] = {}
    
    @classmethod
    def for_cloth(cls, max_length: int = 256, **kwargs) -> 'CurriculumRolloutScheduler':
        """Create scheduler with cloth simulation defaults."""
        return cls(stages=cls.CLOTH_STAGES, max_length=max_length, **kwargs)
    
    @classmethod
    def for_coastal(cls, max_length: int = 1024, **kwargs) -> 'CurriculumRolloutScheduler':
        """Create scheduler with coastal simulation defaults."""
        return cls(stages=cls.COASTAL_STAGES, max_length=max_length, **kwargs)
    
    @classmethod
    def for_motion(cls, max_length: int = 512, **kwargs) -> 'CurriculumRolloutScheduler':
        """Create scheduler with motion prediction defaults."""
        return cls(stages=cls.MOTION_STAGES, max_length=max_length, **kwargs)
    
    @classmethod
    def exponential(
        cls,
        min_length: int = 4,
        max_length: int = 256,
        growth_rate: float = 1.2,
        warmup_epochs: int = 5,
    ) -> 'CurriculumRolloutScheduler':
        """Create scheduler with exponential growth (Project 12 style).
        
        length = min_length * (growth_rate ^ (epoch - warmup_epochs))
        clamped to [min_length, max_length]
        """
        return cls(
            stages=None,
            growth_rate=growth_rate,
            min_length=min_length,
            max_length=max_length,
            warmup_epochs=warmup_epochs,
        )
    
    def get_rollout_length(self, epoch: int, total_epochs: int) -> int:
        """Get the rollout length for the given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Number of frames for rollout
        """
        if epoch < self.warmup_epochs:
            self._current_length = self.min_length
            return self._current_length
        
        effective_epoch = epoch - self.warmup_epochs
        effective_total = max(1, total_epochs - self.warmup_epochs)
        
        # Exponential growth mode
        if self.stages is None and self.growth_rate is not None:
            length = int(self.min_length * (self.growth_rate ** effective_epoch))
            self._current_length = min(self.max_length, max(self.min_length, length))
            return self._current_length
        
        # Stage-based mode
        progress = effective_epoch / effective_total
        
        if self.transition == 'step':
            # Find the applicable stage
            length = self.stages[0].rollout_length
            for stage in self.stages:
                if progress >= stage.epoch_fraction:
                    length = stage.rollout_length
                else:
                    break
            self._current_length = min(self.max_length, length)
        
        elif self.transition == 'smooth':
            # Interpolate between stages
            prev_stage = self.stages[0]
            next_stage = self.stages[-1]
            
            for i, stage in enumerate(self.stages):
                if progress < stage.epoch_fraction:
                    next_stage = stage
                    break
                prev_stage = stage
            
            if prev_stage == next_stage:
                length = prev_stage.rollout_length
            else:
                # Linear interpolation
                stage_progress = (progress - prev_stage.epoch_fraction) / (
                    next_stage.epoch_fraction - prev_stage.epoch_fraction + 1e-8
                )
                length = prev_stage.rollout_length + stage_progress * (
                    next_stage.rollout_length - prev_stage.rollout_length
                )
                length = int(round(length))
            
            self._current_length = min(self.max_length, max(self.min_length, length))
        
        else:
            raise ValueError(f"Unknown transition type: {self.transition}")
        
        return self._current_length
    
    def get_loss_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Get loss weight overrides for the given epoch.
        
        Returns empty dict if no weight curriculum is defined.
        """
        if self.stages is None:
            return {}
        
        progress = epoch / max(1, total_epochs)
        
        weights = {}
        for stage in self.stages:
            if progress >= stage.epoch_fraction and stage.loss_weights:
                weights.update(stage.loss_weights)
        
        self._current_weights = weights
        return weights
    
    def current_rollout_length(self) -> int:
        """Get the current rollout length (from last call to get_rollout_length)."""
        return self._current_length
    
    def suggest_gradient_checkpointing(self, threshold: int = 64) -> bool:
        """Suggest whether to enable gradient checkpointing based on rollout length.
        
        Args:
            threshold: Enable checkpointing if rollout length exceeds this
            
        Returns:
            True if gradient checkpointing is recommended
        """
        return self._current_length > threshold
    
    def get_stage_info(self, epoch: int, total_epochs: int) -> Dict[str, any]:
        """Get detailed information about current curriculum stage."""
        progress = epoch / max(1, total_epochs)
        
        if self.stages is None:
            return {
                'mode': 'exponential',
                'progress': progress,
                'rollout_length': self._current_length,
            }
        
        current_stage_idx = 0
        for i, stage in enumerate(self.stages):
            if progress >= stage.epoch_fraction:
                current_stage_idx = i
        
        return {
            'mode': 'staged',
            'transition': self.transition,
            'progress': progress,
            'stage_index': current_stage_idx,
            'stage_count': len(self.stages),
            'rollout_length': self._current_length,
            'at_stage_fraction': self.stages[current_stage_idx].epoch_fraction,
        }


class LossWeightCurriculum:
    """Curriculum for loss weight adjustment over training.
    
    Extracted from Project 09's CurriculumWeightScheduler. Adjusts physics
    loss weights from easy → hard during training:
    
    Stage 0 (early): Focus on primary loss (SDF/position), minimal physics
    Stage 1 (mid): Ramp up physics losses
    Stage 2 (late): Full physics losses
    
    This helps the model first learn basic prediction before being
    constrained by complex physics losses.
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        stage_fractions: Tuple[float, float, float] = (0.33, 0.33, 0.34),
        initial_physics_scale: float = 0.1,
    ):
        """
        Args:
            base_weights: Dict of {loss_name: final_weight}
            stage_fractions: (stage0_frac, stage1_frac, stage2_frac) summing to 1.0
            initial_physics_scale: Scale for physics losses in stage 0
        """
        self.base_weights = base_weights
        self.stage_fractions = stage_fractions
        self.initial_physics_scale = initial_physics_scale
        
        # Precompute stage boundaries
        self.stage1_start = stage_fractions[0]
        self.stage2_start = stage_fractions[0] + stage_fractions[1]
        
        # Identify physics vs primary losses
        self.physics_losses = {
            'spring', 'stretch', 'bend', 'bending', 'collision', 
            'momentum', 'velocity', 'energy', 'physics'
        }
    
    def get_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Get loss weights for the given epoch."""
        progress = epoch / max(1, total_epochs)
        weights = self.base_weights.copy()
        
        for name, base_weight in self.base_weights.items():
            is_physics = any(p in name.lower() for p in self.physics_losses)
            
            if not is_physics:
                # Primary losses stay at full weight
                continue
            
            if progress < self.stage1_start:
                # Stage 0: Minimal physics
                weights[name] = base_weight * self.initial_physics_scale
            
            elif progress < self.stage2_start:
                # Stage 1: Ramp up physics
                stage_progress = (progress - self.stage1_start) / (
                    self.stage2_start - self.stage1_start
                )
                scale = self.initial_physics_scale + (
                    1.0 - self.initial_physics_scale
                ) * stage_progress
                weights[name] = base_weight * scale
            
            else:
                # Stage 2: Full physics
                weights[name] = base_weight
        
        return weights


class CombinedCurriculum:
    """Combines rollout length and loss weight curricula.
    
    This is the recommended way to use curriculum learning, as the two
    techniques address complementary aspects:
    - Rollout curriculum: teaches stability over increasing horizons
    - Loss weight curriculum: teaches physics compliance gradually
    
    The combination is more powerful than either alone.
    
    Example joint schedule for cloth simulation:
    
    | Epoch % | Rollout Length | Epsilon | Physics Weight |
    |---------|----------------|---------|----------------|
    | 0%      | 4              | 1.0     | 10%            |
    | 25%     | 16             | 0.7     | 40%            |
    | 50%     | 64             | 0.4     | 70%            |
    | 75%     | 256            | 0.2     | 100%           |
    """
    
    def __init__(
        self,
        rollout_scheduler: CurriculumRolloutScheduler,
        loss_weight_curriculum: Optional[LossWeightCurriculum] = None,
    ):
        """
        Args:
            rollout_scheduler: Scheduler for rollout length
            loss_weight_curriculum: Optional curriculum for loss weights
        """
        self.rollout_scheduler = rollout_scheduler
        self.loss_weight_curriculum = loss_weight_curriculum
    
    def update(self, epoch: int, total_epochs: int) -> Dict[str, any]:
        """Update curriculum and return current settings.
        
        Returns:
            Dict with 'rollout_length', 'loss_weights', and diagnostic info
        """
        rollout_length = self.rollout_scheduler.get_rollout_length(epoch, total_epochs)
        
        if self.loss_weight_curriculum:
            loss_weights = self.loss_weight_curriculum.get_weights(epoch, total_epochs)
        else:
            loss_weights = self.rollout_scheduler.get_loss_weights(epoch, total_epochs)
        
        return {
            'rollout_length': rollout_length,
            'loss_weights': loss_weights,
            'use_gradient_checkpointing': self.rollout_scheduler.suggest_gradient_checkpointing(),
            'stage_info': self.rollout_scheduler.get_stage_info(epoch, total_epochs),
        }
    
    def get_rollout_length(self, epoch: int, total_epochs: int) -> int:
        """Convenience method to get rollout length."""
        return self.rollout_scheduler.get_rollout_length(epoch, total_epochs)
    
    def get_loss_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Convenience method to get loss weights."""
        if self.loss_weight_curriculum:
            return self.loss_weight_curriculum.get_weights(epoch, total_epochs)
        return self.rollout_scheduler.get_loss_weights(epoch, total_epochs)


def create_curriculum(
    domain: str = 'cloth',
    max_rollout: int = 256,
    loss_weights: Optional[Dict[str, float]] = None,
    transition: str = 'step',
) -> CombinedCurriculum:
    """Factory function to create a combined curriculum for a domain.
    
    Args:
        domain: One of 'cloth', 'coastal', 'motion'
        max_rollout: Maximum rollout length
        loss_weights: Base loss weights for loss weight curriculum
        transition: 'step' or 'smooth' for rollout transitions
        
    Returns:
        CombinedCurriculum instance
    """
    domain = domain.lower()
    
    if domain == 'cloth':
        rollout_scheduler = CurriculumRolloutScheduler.for_cloth(
            max_length=max_rollout, transition=transition
        )
    elif domain == 'coastal':
        rollout_scheduler = CurriculumRolloutScheduler.for_coastal(
            max_length=max_rollout, transition=transition
        )
    elif domain == 'motion':
        rollout_scheduler = CurriculumRolloutScheduler.for_motion(
            max_length=max_rollout, transition=transition
        )
    else:
        raise ValueError(f"Unknown domain: {domain}. Choose from: cloth, coastal, motion")
    
    loss_curriculum = None
    if loss_weights:
        loss_curriculum = LossWeightCurriculum(base_weights=loss_weights)
    
    return CombinedCurriculum(rollout_scheduler, loss_curriculum)
