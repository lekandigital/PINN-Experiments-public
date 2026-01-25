"""
Scheduled Sampling loss wrapper for stable temporal training.

Scheduled sampling gradually transitions from teacher forcing (using ground
truth as input) to autoregressive rollout (using model predictions), helping
the model learn to be robust to its own prediction errors.

Reference:
    Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent
    Neural Networks", NeurIPS 2015.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class ScheduledSamplingLoss(nn.Module):
    """
    Scheduled sampling wrapper for temporal sequence training.
    
    During training, with probability ε, uses ground truth from the previous
    timestep as input (teacher forcing). With probability (1-ε), uses the
    model's own prediction (autoregressive).
    
    Schedule: ε(epoch) = max(ε_min, ε_start * decay^epoch)
    
    This helps prevent error accumulation during long rollouts by training
    the model to recover from its own mistakes.
    
    Args:
        eps_start: Initial teacher forcing probability (default: 1.0)
        eps_min: Minimum teacher forcing probability (default: 0.2)
        eps_decay: Decay factor per epoch (default: 0.95)
        schedule_type: 'exponential', 'linear', or 'inverse_sigmoid'
        warmup_epochs: Number of epochs before starting decay
    """
    
    def __init__(
        self,
        eps_start: float = 1.0,
        eps_min: float = 0.2,
        eps_decay: float = 0.95,
        schedule_type: str = 'exponential',
        warmup_epochs: int = 0,
    ):
        super().__init__()
        
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        
        # Current epsilon (updated each epoch)
        self.current_eps = eps_start
        self.current_epoch = 0
    
    def update_epsilon(self, epoch: int) -> float:
        """
        Update sampling probability based on current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Updated epsilon value
        """
        self.current_epoch = epoch
        
        # Warmup period: use full teacher forcing
        if epoch < self.warmup_epochs:
            self.current_eps = self.eps_start
            return self.current_eps
        
        # Effective epoch after warmup
        eff_epoch = epoch - self.warmup_epochs
        
        if self.schedule_type == 'exponential':
            self.current_eps = max(
                self.eps_min,
                self.eps_start * (self.eps_decay ** eff_epoch)
            )
        elif self.schedule_type == 'linear':
            # Linear decay over 100 epochs
            decay_epochs = 100
            self.current_eps = max(
                self.eps_min,
                self.eps_start - (self.eps_start - self.eps_min) * eff_epoch / decay_epochs
            )
        elif self.schedule_type == 'inverse_sigmoid':
            # Slower initial decay, then faster
            k = 0.1  # Steepness
            self.current_eps = max(
                self.eps_min,
                self.eps_start / (1 + k * eff_epoch)
            )
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return self.current_eps
    
    def should_use_teacher_forcing(self) -> bool:
        """
        Randomly decide whether to use teacher forcing for current step.
        
        Returns:
            True if should use ground truth, False for model prediction
        """
        return random.random() < self.current_eps
    
    def get_scheduled_input(
        self,
        ground_truth: torch.Tensor,
        model_prediction: torch.Tensor,
        per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Select input based on scheduled sampling.
        
        Args:
            ground_truth: Ground truth values to use with teacher forcing
            model_prediction: Model's own predictions to use for autoregressive
            per_sample: If True, make decision per sample in batch
            
        Returns:
            Selected input tensor
        """
        if per_sample:
            # Different decision for each sample in batch
            batch_size = ground_truth.size(0)
            mask = torch.rand(batch_size, device=ground_truth.device) < self.current_eps
            mask = mask.view(-1, *([1] * (ground_truth.dim() - 1)))
            return torch.where(mask, ground_truth, model_prediction)
        else:
            # Same decision for entire batch
            if self.should_use_teacher_forcing():
                return ground_truth
            else:
                return model_prediction


class TemporalLossAccumulator:
    """
    Accumulates losses over temporal sequences with optional weighting.
    
    Features:
    - Per-timestep loss tracking
    - Temporal discount (later frames weighted less during early training)
    - Gradient checkpointing support for long sequences
    """
    
    def __init__(
        self,
        temporal_discount: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            temporal_discount: Discount factor for later frames (1.0 = no discount)
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        self.temporal_discount = temporal_discount
        self.gradient_checkpointing = gradient_checkpointing
        
        self.losses: List[torch.Tensor] = []
        self.timestep_losses: Dict[int, float] = {}
    
    def add_timestep_loss(
        self,
        loss: torch.Tensor,
        timestep: int,
    ) -> None:
        """
        Add loss for a specific timestep.
        
        Args:
            loss: Loss tensor for this timestep
            timestep: Timestep index
        """
        # Apply temporal discount
        weight = self.temporal_discount ** timestep
        weighted_loss = loss * weight
        
        self.losses.append(weighted_loss)
        self.timestep_losses[timestep] = loss.item()
    
    def get_total_loss(self) -> torch.Tensor:
        """
        Get total loss summed over all timesteps.
        
        Returns:
            Total weighted loss
        """
        if not self.losses:
            return torch.tensor(0.0)
        
        return torch.stack(self.losses).sum()
    
    def get_mean_loss(self) -> torch.Tensor:
        """
        Get mean loss averaged over timesteps.
        
        Returns:
            Mean weighted loss
        """
        if not self.losses:
            return torch.tensor(0.0)
        
        return torch.stack(self.losses).mean()
    
    def get_timestep_losses(self) -> Dict[int, float]:
        """Get dictionary of per-timestep losses."""
        return self.timestep_losses.copy()
    
    def reset(self) -> None:
        """Clear accumulated losses."""
        self.losses = []
        self.timestep_losses = {}


class CurriculumScheduler:
    """
    Curriculum learning scheduler for sequence length.
    
    Gradually increases the rollout length during training:
    - Start with short sequences (easier to learn)
    - Progressively extend to full sequence length
    
    This helps stabilize training of long temporal sequences.
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 120,
        warmup_epochs: int = 5,
        growth_rate: float = 1.2,
    ):
        """
        Args:
            min_length: Starting sequence length
            max_length: Maximum sequence length (full rollout)
            warmup_epochs: Epochs before starting curriculum
            growth_rate: Multiplicative increase per epoch
        """
        self.min_length = min_length
        self.max_length = max_length
        self.warmup_epochs = warmup_epochs
        self.growth_rate = growth_rate
        
        self.current_length = min_length
    
    def update_length(self, epoch: int) -> int:
        """
        Update sequence length based on current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current sequence length to use
        """
        if epoch < self.warmup_epochs:
            self.current_length = self.min_length
        else:
            eff_epoch = epoch - self.warmup_epochs
            self.current_length = min(
                self.max_length,
                int(self.min_length * (self.growth_rate ** eff_epoch))
            )
        
        return self.current_length
    
    def get_current_length(self) -> int:
        """Get current sequence length."""
        return self.current_length
