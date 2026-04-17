"""
Epsilon Decay Schedules for Scheduled Sampling

Controls the teacher forcing ratio over training epochs.
- epsilon = 1.0 means 100% teacher forcing (use ground truth as input)
- epsilon = 0.0 means 0% teacher forcing (always use model predictions)

Project 12 uses decay from 1.0 → 0.2, meaning even at the end of training,
20% of steps still use ground truth. This prevents catastrophic forgetting
and provides ongoing curriculum signal.

Extracted from Project 12 (NIF-Cloth4D-Temporal) and extended with additional
schedule types. Also incorporates Project 08's "linear ramp" pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import math


class EpsilonSchedule(ABC):
    """Abstract base class for epsilon decay schedules.
    
    All schedules must implement get_epsilon(epoch, total_epochs) -> float.
    The returned value should be in [0, 1] representing the probability of
    using teacher forcing (ground truth input) at each step.
    """
    
    @abstractmethod
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        """Get epsilon value for the given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
            
        Returns:
            Epsilon value in [0, 1], where 1.0 = full teacher forcing
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class LinearDecaySchedule(EpsilonSchedule):
    """Linear decay from start to end over a fraction of training.
    
    This is the simplest schedule: epsilon decreases linearly from `start`
    to `end` over `decay_fraction` of total epochs, then stays at `end`.
    
    Default configuration (cloth simulation):
        start=1.0, end=0.2, decay_fraction=0.8
        
    This means:
        - Epochs 0-80% of training: epsilon linearly decreases 1.0 → 0.2
        - Epochs 80-100%: epsilon stays at 0.2
        
    Args:
        start: Initial epsilon value (default 1.0 = full teacher forcing)
        end: Final epsilon value (default 0.2 = 80% model predictions)
        decay_fraction: Fraction of total epochs over which to decay (default 0.8)
        warmup_epochs: Number of initial epochs to stay at `start` (default 0)
    """
    start: float = 1.0
    end: float = 0.2
    decay_fraction: float = 0.8
    warmup_epochs: int = 0
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        if epoch < self.warmup_epochs:
            return self.start
        
        effective_epoch = epoch - self.warmup_epochs
        effective_total = total_epochs - self.warmup_epochs
        decay_epochs = int(effective_total * self.decay_fraction)
        
        if decay_epochs <= 0:
            return self.end
        
        if effective_epoch >= decay_epochs:
            return self.end
        
        # Linear interpolation
        progress = effective_epoch / decay_epochs
        return self.start + (self.end - self.start) * progress
    
    def __repr__(self) -> str:
        return (f"LinearDecaySchedule(start={self.start}, end={self.end}, "
                f"decay_fraction={self.decay_fraction}, warmup_epochs={self.warmup_epochs})")


@dataclass
class ExponentialDecaySchedule(EpsilonSchedule):
    """Exponential decay: epsilon = start * (decay_rate ^ epoch).
    
    This is the schedule used by Project 12 (NIF-Cloth4D-Temporal).
    Faster initial decrease, slower at the end — good for quickly
    transitioning away from teacher forcing while maintaining stability.
    
    Default configuration (from Project 12):
        start=1.0, min_epsilon=0.2, decay_rate=0.95
        
    This means:
        - Epoch 0: epsilon = 1.0
        - Epoch 10: epsilon ≈ 0.60
        - Epoch 20: epsilon ≈ 0.36
        - Epoch 30: epsilon ≈ 0.21
        - Epoch 31+: epsilon = 0.2 (clamped to min)
        
    Args:
        start: Initial epsilon value (default 1.0)
        min_epsilon: Minimum epsilon value (default 0.2)
        decay_rate: Multiplicative decay factor per epoch (default 0.95)
        warmup_epochs: Number of initial epochs to stay at `start` (default 0)
    """
    start: float = 1.0
    min_epsilon: float = 0.2
    decay_rate: float = 0.95
    warmup_epochs: int = 0
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        if epoch < self.warmup_epochs:
            return self.start
        
        effective_epoch = epoch - self.warmup_epochs
        epsilon = self.start * (self.decay_rate ** effective_epoch)
        return max(self.min_epsilon, epsilon)
    
    def __repr__(self) -> str:
        return (f"ExponentialDecaySchedule(start={self.start}, min_epsilon={self.min_epsilon}, "
                f"decay_rate={self.decay_rate}, warmup_epochs={self.warmup_epochs})")


@dataclass
class CosineAnnealSchedule(EpsilonSchedule):
    """Cosine annealing from start to end.
    
    Smooth cosine curve that starts slow, accelerates in the middle,
    and slows down again at the end. Widely used in learning rate
    schedules and works well for epsilon decay too.
    
    Formula: epsilon = end + 0.5 * (start - end) * (1 + cos(π * progress))
    
    Args:
        start: Initial epsilon value (default 1.0)
        end: Final epsilon value (default 0.2)
        warmup_epochs: Number of initial epochs to stay at `start` (default 0)
    """
    start: float = 1.0
    end: float = 0.2
    warmup_epochs: int = 0
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        if epoch < self.warmup_epochs:
            return self.start
        
        effective_epoch = epoch - self.warmup_epochs
        effective_total = max(1, total_epochs - self.warmup_epochs)
        
        progress = min(1.0, effective_epoch / effective_total)
        # Cosine annealing formula
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * progress))
    
    def __repr__(self) -> str:
        return (f"CosineAnnealSchedule(start={self.start}, end={self.end}, "
                f"warmup_epochs={self.warmup_epochs})")


@dataclass
class StepDecaySchedule(EpsilonSchedule):
    """Step-wise decay at specified epoch milestones.
    
    Discrete drops at specified epoch fractions. Useful when you want
    clear "phases" of training with different teacher forcing ratios.
    
    Example (4-stage training):
        milestones=[(0.0, 1.0), (0.25, 0.6), (0.5, 0.3), (0.75, 0.2)]
        
    This means:
        - Epochs 0-25%: epsilon = 1.0 (full teacher forcing)
        - Epochs 25-50%: epsilon = 0.6
        - Epochs 50-75%: epsilon = 0.3
        - Epochs 75-100%: epsilon = 0.2
        
    Args:
        milestones: List of (epoch_fraction, epsilon_value) tuples.
                   Must be sorted by epoch_fraction in ascending order.
    """
    milestones: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.milestones is None:
            # Default: 4-stage decay
            self.milestones = [
                (0.0, 1.0),
                (0.25, 0.6),
                (0.5, 0.3),
                (0.75, 0.2),
            ]
        # Ensure milestones are sorted
        self.milestones = sorted(self.milestones, key=lambda x: x[0])
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 0:
            return self.milestones[0][1] if self.milestones else 1.0
        
        progress = epoch / total_epochs
        
        # Find the applicable milestone
        epsilon = self.milestones[0][1]
        for fraction, value in self.milestones:
            if progress >= fraction:
                epsilon = value
            else:
                break
        
        return epsilon
    
    def __repr__(self) -> str:
        return f"StepDecaySchedule(milestones={self.milestones})"


@dataclass
class ConstantSchedule(EpsilonSchedule):
    """Constant epsilon throughout training.
    
    Useful for ablation studies and debugging. Set epsilon=1.0 for
    pure teacher forcing (baseline), or epsilon=0.0 for pure
    autoregressive (to measure raw model stability).
    
    Args:
        epsilon: Fixed epsilon value (default 1.0)
    """
    epsilon: float = 1.0
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        return self.epsilon
    
    def __repr__(self) -> str:
        return f"ConstantSchedule(epsilon={self.epsilon})"


@dataclass
class InverseSigmoidSchedule(EpsilonSchedule):
    """Inverse sigmoid decay for smooth S-curve transition.
    
    This schedule provides a smooth S-curve transition that's slow
    at the start, fast in the middle, and slow at the end. Based on
    the inverse sigmoid (logistic) function.
    
    Project 12 mentions this as an option alongside exponential.
    
    Args:
        start: Initial epsilon value (default 1.0)
        end: Final epsilon value (default 0.2)
        k: Steepness of the sigmoid curve (default 10.0)
        warmup_epochs: Number of initial epochs to stay at `start` (default 0)
    """
    start: float = 1.0
    end: float = 0.2
    k: float = 10.0
    warmup_epochs: int = 0
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        if epoch < self.warmup_epochs:
            return self.start
        
        effective_epoch = epoch - self.warmup_epochs
        effective_total = max(1, total_epochs - self.warmup_epochs)
        
        # Map progress to [-k/2, k/2] for sigmoid input
        progress = effective_epoch / effective_total
        x = self.k * (progress - 0.5)
        
        # Inverse sigmoid: 1 / (1 + exp(x))
        sigmoid = 1.0 / (1.0 + math.exp(x))
        
        # Scale to [end, start]
        return self.end + (self.start - self.end) * sigmoid
    
    def __repr__(self) -> str:
        return (f"InverseSigmoidSchedule(start={self.start}, end={self.end}, "
                f"k={self.k}, warmup_epochs={self.warmup_epochs})")


@dataclass
class LinearRampSchedule(EpsilonSchedule):
    """Linear ramp UP from low to high teacher forcing probability.
    
    This is the OPPOSITE of the standard decay schedules. Used by
    Project 08 (HGNN-ClothDyn) where training starts with mostly
    model predictions and gradually increases teacher forcing.
    
    Rationale: Start by letting the model learn to handle its own
    predictions (hard), then gradually add more ground truth signal
    to refine accuracy.
    
    Project 08 configuration:
        start=0.1, end=1.0, start_epoch=5, increment_per_epoch=0.05
        
    Args:
        start: Initial epsilon value (default 0.1 = mostly model predictions)
        end: Final epsilon value (default 1.0 = full teacher forcing)
        start_epoch: Epoch to begin ramping (default 5)
        increment_per_epoch: How much to increase epsilon each epoch (default 0.05)
    """
    start: float = 0.1
    end: float = 1.0
    start_epoch: int = 5
    increment_per_epoch: float = 0.05
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        if epoch < self.start_epoch:
            return 0.0  # No teacher forcing before start_epoch
        
        effective_epoch = epoch - self.start_epoch
        epsilon = self.start + self.increment_per_epoch * effective_epoch
        return min(self.end, epsilon)
    
    def __repr__(self) -> str:
        return (f"LinearRampSchedule(start={self.start}, end={self.end}, "
                f"start_epoch={self.start_epoch}, increment_per_epoch={self.increment_per_epoch})")


def create_schedule(
    schedule_type: str,
    start: float = 1.0,
    end: float = 0.2,
    **kwargs
) -> EpsilonSchedule:
    """Factory function to create epsilon schedules by name.
    
    Args:
        schedule_type: One of 'linear', 'exponential', 'cosine', 'step',
                      'constant', 'inverse_sigmoid', 'linear_ramp'
        start: Initial epsilon value
        end: Final epsilon value (or min_epsilon for exponential)
        **kwargs: Additional arguments for specific schedule types
        
    Returns:
        An EpsilonSchedule instance
        
    Example:
        >>> schedule = create_schedule('exponential', start=1.0, end=0.2, decay_rate=0.95)
        >>> schedule.get_epsilon(epoch=10, total_epochs=100)
        0.5987...
    """
    schedule_type = schedule_type.lower()
    
    if schedule_type == 'linear':
        return LinearDecaySchedule(
            start=start,
            end=end,
            decay_fraction=kwargs.get('decay_fraction', 0.8),
            warmup_epochs=kwargs.get('warmup_epochs', 0),
        )
    elif schedule_type == 'exponential':
        return ExponentialDecaySchedule(
            start=start,
            min_epsilon=end,
            decay_rate=kwargs.get('decay_rate', 0.95),
            warmup_epochs=kwargs.get('warmup_epochs', 0),
        )
    elif schedule_type == 'cosine':
        return CosineAnnealSchedule(
            start=start,
            end=end,
            warmup_epochs=kwargs.get('warmup_epochs', 0),
        )
    elif schedule_type == 'step':
        return StepDecaySchedule(
            milestones=kwargs.get('milestones', None),
        )
    elif schedule_type == 'constant':
        return ConstantSchedule(
            epsilon=kwargs.get('epsilon', start),
        )
    elif schedule_type == 'inverse_sigmoid':
        return InverseSigmoidSchedule(
            start=start,
            end=end,
            k=kwargs.get('k', 10.0),
            warmup_epochs=kwargs.get('warmup_epochs', 0),
        )
    elif schedule_type == 'linear_ramp':
        return LinearRampSchedule(
            start=kwargs.get('ramp_start', 0.1),
            end=kwargs.get('ramp_end', 1.0),
            start_epoch=kwargs.get('start_epoch', 5),
            increment_per_epoch=kwargs.get('increment_per_epoch', 0.05),
        )
    else:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Choose from: linear, exponential, cosine, step, constant, "
            f"inverse_sigmoid, linear_ramp"
        )
