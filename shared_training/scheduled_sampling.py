"""
Scheduled Sampling Trainer for Long-Horizon Rollout Stabilization

This module implements the core scheduled sampling logic extracted from
Project 12 (NIF-Cloth4D-Temporal) and generalized for any autoregressive model.

Key concepts:
- Scheduled sampling gradually transitions from teacher forcing to free-running
- At each step, we flip a Bernoulli(epsilon) coin to decide input source
- epsilon starts high (teacher forcing) and decays to low (model predictions)
- This trains the model to handle its own imperfect predictions

The trainer is model-agnostic: it operates through a RolloutModelAdapter interface
that each project implements to bridge between the trainer and their model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler

from .schedules import EpsilonSchedule, ExponentialDecaySchedule


@dataclass
class SamplingDecision:
    """Record of a single sampling decision for logging/debugging."""
    step: int
    epsilon: float
    used_teacher_forcing: bool
    random_value: float


class TemporalLossAccumulator:
    """Accumulates losses over a rollout sequence for proper averaging.
    
    Supports separate tracking of teacher-forced vs free-running frame losses,
    which helps diagnose how well the model handles its own predictions.
    
    Extracted from Project 12's training loop.
    """
    
    def __init__(
        self,
        free_running_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            free_running_weight: Extra weight for free-running (non-TF) losses.
                                 Default 1.0 = equal weight. Use 2.0 to weight
                                 free-running losses twice as much.
            reduction: 'mean' or 'sum' for final loss aggregation.
        """
        self.free_running_weight = free_running_weight
        self.reduction = reduction
        self.reset()
    
    def reset(self):
        """Reset accumulator for a new sequence."""
        self.losses: List[Tensor] = []
        self.weights: List[float] = []
        self.teacher_forced_losses: List[float] = []
        self.free_running_losses: List[float] = []
        self.total_steps = 0
    
    def add_timestep_loss(
        self,
        loss: Tensor,
        step: int,
        is_teacher_forced: bool = True,
        loss_dict: Optional[Dict[str, float]] = None,
    ):
        """Add loss for a single timestep.
        
        Args:
            loss: Scalar loss tensor for this step
            step: Timestep index (for logging)
            is_teacher_forced: Whether this step used teacher forcing
            loss_dict: Optional dict of individual loss components
        """
        weight = 1.0 if is_teacher_forced else self.free_running_weight
        self.losses.append(loss)
        self.weights.append(weight)
        self.total_steps += 1
        
        # Track separately for diagnostics
        loss_val = loss.detach().item()
        if is_teacher_forced:
            self.teacher_forced_losses.append(loss_val)
        else:
            self.free_running_losses.append(loss_val)
    
    def get_mean_loss(self) -> Tensor:
        """Get weighted mean loss over all timesteps."""
        if not self.losses:
            return torch.tensor(0.0)
        
        weighted_losses = [l * w for l, w in zip(self.losses, self.weights)]
        total = sum(weighted_losses)
        
        if self.reduction == 'mean':
            return total / sum(self.weights)
        else:
            return total
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic statistics for logging."""
        diagnostics = {
            'total_steps': self.total_steps,
            'teacher_forced_steps': len(self.teacher_forced_losses),
            'free_running_steps': len(self.free_running_losses),
        }
        
        if self.teacher_forced_losses:
            diagnostics['mean_tf_loss'] = sum(self.teacher_forced_losses) / len(self.teacher_forced_losses)
        if self.free_running_losses:
            diagnostics['mean_fr_loss'] = sum(self.free_running_losses) / len(self.free_running_losses)
        
        # Ratio indicates how much harder free-running is
        if self.teacher_forced_losses and self.free_running_losses:
            diagnostics['fr_tf_loss_ratio'] = (
                diagnostics['mean_fr_loss'] / (diagnostics['mean_tf_loss'] + 1e-8)
            )
        
        return diagnostics


class ScheduledSampler:
    """Core scheduled sampling logic.
    
    This class handles the epsilon-greedy decision of whether to use
    teacher forcing (ground truth) or free-running (model prediction)
    as input to the next timestep.
    
    Extracted from Project 12's ScheduledSampler class and enhanced
    with per-sample decision support and RNG reproducibility.
    
    Usage:
        sampler = ScheduledSampler(
            schedule=ExponentialDecaySchedule(start=1.0, min_epsilon=0.2),
            per_sample=True,
        )
        
        for epoch in range(total_epochs):
            sampler.update_epsilon(epoch, total_epochs)
            for batch in dataloader:
                for t in range(rollout_length):
                    # Get next input
                    next_input = sampler.get_scheduled_input(
                        ground_truth=gt_frames[t+1],
                        model_prediction=pred_frames[t],
                    )
    """
    
    def __init__(
        self,
        schedule: Optional[EpsilonSchedule] = None,
        per_sample: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            schedule: Epsilon schedule to use. Default is exponential decay.
            per_sample: If True, sample independently for each item in batch.
                       If False, use same decision for entire batch.
            seed: Random seed for reproducibility. If None, uses default RNG.
        """
        self.schedule = schedule or ExponentialDecaySchedule()
        self.per_sample = per_sample
        self.current_epsilon = 1.0
        
        # For reproducibility
        self.seed = seed
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
        
        # Logging
        self.decisions: List[SamplingDecision] = []
        self.log_decisions = False
    
    def update_epsilon(self, epoch: int, total_epochs: int) -> float:
        """Update epsilon for the current epoch.
        
        Call this once at the start of each epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
            
        Returns:
            The new epsilon value
        """
        self.current_epsilon = self.schedule.get_epsilon(epoch, total_epochs)
        return self.current_epsilon
    
    def should_use_teacher_forcing(
        self,
        epoch: int = None,
        batch_idx: int = None,
        step_idx: int = None,
    ) -> bool:
        """Decide whether to use teacher forcing for this step.
        
        Args:
            epoch, batch_idx, step_idx: Optional identifiers for deterministic
                                        seeding (for reproducibility)
                                        
        Returns:
            True if should use teacher forcing (ground truth),
            False if should use model prediction
        """
        # Deterministic seeding if all identifiers provided
        if self.seed is not None and all(x is not None for x in [epoch, batch_idx, step_idx]):
            seed = hash((self.seed, epoch, batch_idx, step_idx)) % (2**32)
            self.rng.seed(seed)
        
        random_value = self.rng.random()
        use_tf = random_value < self.current_epsilon
        
        if self.log_decisions and step_idx is not None:
            self.decisions.append(SamplingDecision(
                step=step_idx,
                epsilon=self.current_epsilon,
                used_teacher_forcing=use_tf,
                random_value=random_value,
            ))
        
        return use_tf
    
    def get_scheduled_input(
        self,
        ground_truth: Tensor,
        model_prediction: Tensor,
        epoch: int = None,
        batch_idx: int = None,
        step_idx: int = None,
    ) -> Tuple[Tensor, bool]:
        """Select input for next timestep based on scheduled sampling.
        
        This is the main API for scheduled sampling. It either returns
        the ground truth (teacher forcing) or the model's own prediction
        (free running) based on the current epsilon and a random draw.
        
        Args:
            ground_truth: Ground truth input for next step, shape (B, ...)
            model_prediction: Model's prediction, shape (B, ...)
            epoch, batch_idx, step_idx: For deterministic RNG
            
        Returns:
            Tuple of (selected_input, is_teacher_forced)
        """
        if self.per_sample:
            # Per-sample decision: independent coin flip for each batch item
            batch_size = ground_truth.size(0)
            device = ground_truth.device
            
            # Generate per-sample random values
            if self.seed is not None and all(x is not None for x in [epoch, batch_idx, step_idx]):
                # Deterministic per-sample
                mask_list = []
                for b in range(batch_size):
                    seed = hash((self.seed, epoch, batch_idx, step_idx, b)) % (2**32)
                    self.rng.seed(seed)
                    mask_list.append(self.rng.random() < self.current_epsilon)
                mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            else:
                mask = torch.rand(batch_size, device=device) < self.current_epsilon
            
            # Expand mask to match input dimensions
            mask = mask.view(-1, *([1] * (ground_truth.dim() - 1)))
            
            # Select based on mask
            selected = torch.where(mask, ground_truth, model_prediction)
            
            # For logging: report majority decision
            is_tf = mask.float().mean().item() > 0.5
            return selected, is_tf
        else:
            # Batch-level decision: same for entire batch
            use_tf = self.should_use_teacher_forcing(epoch, batch_idx, step_idx)
            if use_tf:
                return ground_truth, True
            else:
                return model_prediction, False
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about sampling decisions."""
        if not self.decisions:
            return {'current_epsilon': self.current_epsilon}
        
        tf_count = sum(1 for d in self.decisions if d.used_teacher_forcing)
        total = len(self.decisions)
        
        return {
            'current_epsilon': self.current_epsilon,
            'actual_tf_rate': tf_count / total if total > 0 else 0,
            'total_decisions': total,
            'teacher_forced': tf_count,
            'free_running': total - tf_count,
        }
    
    def reset_logging(self):
        """Clear logged decisions."""
        self.decisions = []


class ScheduledSamplingTrainer:
    """High-level trainer that orchestrates scheduled sampling training.
    
    This trainer manages the full training loop including:
    - Epsilon schedule updates
    - Per-timestep scheduled sampling decisions
    - Loss accumulation with free-running weighting
    - AMP (automatic mixed precision) support
    - Gradient clipping and accumulation
    - Logging and diagnostics
    
    The trainer is model-agnostic: it operates through a RolloutModelAdapter
    interface that each project implements.
    
    Usage:
        from shared_training.adapters import GRUAdapter
        
        adapter = GRUAdapter(model, ...)
        trainer = ScheduledSamplingTrainer(
            adapter=adapter,
            epsilon_schedule=ExponentialDecaySchedule(),
            curriculum=CurriculumRolloutScheduler(...),
        )
        
        for epoch in range(total_epochs):
            for batch in dataloader:
                loss, diagnostics = trainer.train_step(
                    batch, epoch, total_epochs, optimizer
                )
    """
    
    def __init__(
        self,
        adapter: 'RolloutModelAdapter',
        epsilon_schedule: Optional[EpsilonSchedule] = None,
        curriculum: Optional['CurriculumRolloutScheduler'] = None,
        free_running_loss_weight: float = 1.0,
        detach_predictions: bool = False,
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        truncated_bptt_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            adapter: Model adapter implementing RolloutModelAdapter interface
            epsilon_schedule: Schedule for teacher forcing probability
            curriculum: Optional curriculum for rollout length
            free_running_loss_weight: Extra weight for free-running losses
            detach_predictions: If True, detach predictions before feeding back.
                               Default False allows gradient flow through rollout.
            use_amp: Use automatic mixed precision
            grad_clip_norm: Max gradient norm for clipping
            truncated_bptt_steps: If set, truncate BPTT at this many steps
            seed: Random seed for reproducibility
        """
        self.adapter = adapter
        self.curriculum = curriculum
        self.free_running_loss_weight = free_running_loss_weight
        self.detach_predictions = detach_predictions
        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self.truncated_bptt_steps = truncated_bptt_steps
        
        # Scheduled sampler
        self.sampler = ScheduledSampler(
            schedule=epsilon_schedule or ExponentialDecaySchedule(),
            per_sample=True,
            seed=seed,
        )
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Metrics tracking
        self.epoch_metrics: Dict[str, List[float]] = {
            'loss': [],
            'epsilon': [],
            'rollout_length': [],
            'tf_rate': [],
            'fr_tf_ratio': [],
        }
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
        epoch: int,
        total_epochs: int,
        optimizer: torch.optim.Optimizer,
        batch_idx: int = 0,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Execute one training step with scheduled sampling.
        
        Args:
            batch: Batch dict containing sequence data
            epoch: Current epoch
            total_epochs: Total number of epochs
            optimizer: PyTorch optimizer
            batch_idx: Batch index within epoch (for logging)
            
        Returns:
            Tuple of (loss tensor, diagnostics dict)
        """
        # Update epsilon for this epoch
        self.sampler.update_epsilon(epoch, total_epochs)
        
        # Get rollout length from curriculum or batch
        if self.curriculum is not None:
            rollout_length = self.curriculum.get_rollout_length(epoch, total_epochs)
        else:
            rollout_length = self.adapter.get_sequence_length(batch)
        
        # Initialize hidden state and loss accumulator
        hidden = self.adapter.init_hidden(batch)
        loss_accum = TemporalLossAccumulator(
            free_running_weight=self.free_running_loss_weight
        )
        
        # Get initial state from batch
        current_state = self.adapter.get_initial_state(batch)
        
        # Truncated BPTT tracking
        tbptt_step = 0
        
        # Mixed precision context
        amp_context = autocast(enabled=self.use_amp)
        
        with amp_context:
            for t in range(rollout_length):
                # Get ground truth for this timestep
                gt_target = self.adapter.get_target(batch, t)
                
                # Forward pass
                prediction, hidden = self.adapter.step(
                    current_state, hidden, t, batch
                )
                
                # Compute loss
                is_tf = (t == 0)  # First step is always teacher-forced
                if t > 0:
                    _, is_tf = self._last_sampling_decision
                
                loss = self.adapter.compute_loss(
                    prediction, gt_target, is_teacher_forced=is_tf
                )
                loss_accum.add_timestep_loss(loss, t, is_teacher_forced=is_tf)
                
                # Prepare next input via scheduled sampling
                if t < rollout_length - 1:
                    gt_next_input = self.adapter.get_teacher_input(batch, t + 1)
                    pred_next_input = self.adapter.prediction_to_input(prediction)
                    
                    if self.detach_predictions:
                        pred_next_input = pred_next_input.detach()
                    
                    current_state, is_tf = self.sampler.get_scheduled_input(
                        gt_next_input, pred_next_input,
                        epoch=epoch, batch_idx=batch_idx, step_idx=t,
                    )
                    self._last_sampling_decision = (current_state, is_tf)
                
                # Truncated BPTT: detach hidden state periodically
                if self.truncated_bptt_steps is not None:
                    tbptt_step += 1
                    if tbptt_step >= self.truncated_bptt_steps:
                        hidden = self.adapter.detach_hidden(hidden)
                        tbptt_step = 0
        
        # Get total loss
        total_loss = loss_accum.get_mean_loss()
        
        # Backward pass with AMP handling
        optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(), self.grad_clip_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(), self.grad_clip_norm
            )
            optimizer.step()
        
        # Collect diagnostics
        diagnostics = {
            'loss': total_loss.detach().item(),
            'epsilon': self.sampler.current_epsilon,
            'rollout_length': rollout_length,
            **loss_accum.get_diagnostics(),
        }
        
        return total_loss, diagnostics
    
    def evaluate_rollout(
        self,
        batch: Dict[str, Tensor],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate free-running rollout stability (no teacher forcing).
        
        Args:
            batch: Batch containing sequence data
            max_steps: Maximum rollout steps (default: full sequence)
            
        Returns:
            Dict with stability metrics
        """
        self.adapter.model.eval()
        
        with torch.no_grad():
            hidden = self.adapter.init_hidden(batch)
            current_state = self.adapter.get_initial_state(batch)
            
            predictions = []
            targets = []
            
            seq_len = max_steps or self.adapter.get_sequence_length(batch)
            
            for t in range(seq_len):
                gt_target = self.adapter.get_target(batch, t)
                prediction, hidden = self.adapter.step(
                    current_state, hidden, t, batch
                )
                
                predictions.append(prediction.detach())
                targets.append(gt_target.detach())
                
                # Always use model prediction for next step (pure autoregressive)
                if t < seq_len - 1:
                    current_state = self.adapter.prediction_to_input(prediction)
        
        self.adapter.model.train()
        
        # Compute stability metrics
        predictions = torch.stack(predictions, dim=1)  # (B, T, ...)
        targets = torch.stack(targets, dim=1)
        
        # Per-step MSE
        mse_per_step = ((predictions - targets) ** 2).mean(dim=tuple(range(2, predictions.dim())))
        mse_per_step = mse_per_step.mean(dim=0)  # Average over batch
        
        # Stability horizon: steps until MSE > 2x initial
        initial_mse = mse_per_step[0].item()
        threshold = 2.0 * initial_mse
        stability_horizon = seq_len
        for t, mse in enumerate(mse_per_step):
            if mse.item() > threshold:
                stability_horizon = t
                break
        
        # Drift rate: linear fit to log-MSE
        log_mse = torch.log(mse_per_step + 1e-8)
        steps = torch.arange(len(log_mse), dtype=torch.float32)
        drift_rate = ((steps * log_mse).mean() - steps.mean() * log_mse.mean()) / (
            (steps ** 2).mean() - steps.mean() ** 2 + 1e-8
        )
        
        return {
            'mse_per_step': mse_per_step.cpu().numpy(),
            'stability_horizon': stability_horizon,
            'drift_rate': drift_rate.item(),
            'initial_mse': initial_mse,
            'final_mse': mse_per_step[-1].item(),
        }
    
    def get_epoch_summary(self) -> Dict[str, float]:
        """Get summary statistics for the epoch."""
        summary = {}
        for key, values in self.epoch_metrics.items():
            if values:
                summary[f'{key}_mean'] = sum(values) / len(values)
        return summary
    
    def reset_epoch_metrics(self):
        """Reset epoch metrics tracking."""
        for key in self.epoch_metrics:
            self.epoch_metrics[key] = []


# Import adapter base class for type hints
# Actual implementation is in adapters/base_adapter.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .adapters.base_adapter import RolloutModelAdapter
    from .curriculum_rollout import CurriculumRolloutScheduler
