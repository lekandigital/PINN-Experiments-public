"""
Knowledge Distillation Training Engine

This module provides the core DistillationTrainer class that:
1. Loads and freezes a teacher model
2. Creates and trains a student model
3. Uses configurable loss functions (soft target, physics, feature matching)
4. Supports curriculum/progressive distillation
5. Handles checkpointing and logging

Extracted and generalized from Project 15 (PINN-Lite-Foil).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from .config import DistillConfig, TrainingConfig
from .losses import DistillationLoss, FeatureMatchingLoss, create_physics_loss
from .registry import ModelRegistration, registry
from .sampling import SamplingStrategy, create_sampler
from .utils import (
    MetricsTracker,
    count_parameters,
    freeze_model,
    get_device,
    model_size_mb,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


@dataclass
class DistillationResult:
    """Results from a distillation run."""
    
    student_path: str  # Path to best student checkpoint
    final_loss: float
    best_loss: float
    epochs_trained: int
    training_time_seconds: float
    
    # Metrics
    teacher_params: int
    student_params: int
    compression_ratio: float
    
    # Per-epoch history
    loss_history: list[dict[str, float]]
    eval_history: list[dict[str, float]]


class DistillationTrainer:
    """
    Model-agnostic knowledge distillation trainer.
    
    Takes a frozen teacher model and trains a student to mimic it,
    optionally with physics constraints and feature matching.
    
    Usage:
        trainer = DistillationTrainer(config)
        result = trainer.train()
        
        # Or step-by-step
        trainer.setup()
        for epoch in range(trainer.config.training.num_epochs):
            trainer.train_epoch(epoch)
            if epoch % eval_every == 0:
                trainer.evaluate(epoch)
        trainer.finalize()
    """
    
    def __init__(
        self,
        config: DistillConfig,
        teacher: nn.Module | None = None,
        student: nn.Module | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            config: Full distillation configuration
            teacher: Optional pre-loaded teacher model
            student: Optional pre-created student model
            device: Device to train on ("auto", "cuda", "cpu")
        """
        self.config = config
        self.device = get_device(device or config.teacher.device if config.teacher else "auto")
        
        # Models (loaded in setup())
        self._teacher = teacher
        self._student = student
        self._teacher_reg: ModelRegistration | None = None
        self._student_reg: ModelRegistration | None = None
        
        # Training components (initialized in setup())
        self.loss_fn: DistillationLoss | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: Any = None
        self.sampler: SamplingStrategy | None = None
        
        # Feature matching (optional)
        self.feature_matching: FeatureMatchingLoss | None = None
        
        # Tracking
        self.metrics = MetricsTracker()
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Output paths
        self.output_dir = Path(config.output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        # Training state
        self._is_setup = False
        self._start_time: float | None = None
    
    @property
    def teacher(self) -> nn.Module:
        """Get the teacher model."""
        if self._teacher is None:
            raise RuntimeError("Teacher not loaded. Call setup() first.")
        return self._teacher
    
    @property
    def student(self) -> nn.Module:
        """Get the student model."""
        if self._student is None:
            raise RuntimeError("Student not created. Call setup() first.")
        return self._student
    
    def setup(self) -> None:
        """
        Set up all training components.
        
        - Loads teacher model and freezes it
        - Creates student model
        - Initializes loss function, optimizer, scheduler
        - Sets up sampling strategy
        """
        if self._is_setup:
            logger.warning("Trainer already set up, skipping")
            return
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        set_seed(self.config.training.seed)
        
        # Setup logging
        setup_logging(
            log_file=str(self.logs_dir / "training.log"),
            name="distillation"
        )
        
        # Load teacher
        self._setup_teacher()
        
        # Create student
        self._setup_student()
        
        # Setup loss function
        self._setup_loss()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup sampler
        self._setup_sampler()
        
        self._is_setup = True
        
        # Log setup info
        teacher_params = count_parameters(self.teacher, trainable_only=False)
        student_params = count_parameters(self.student)
        
        logger.info(f"Distillation setup complete:")
        logger.info(f"  Teacher: {self.config.teacher.name} ({teacher_params:,} params, {model_size_mb(self.teacher):.2f} MB)")
        logger.info(f"  Student: {self.config.student.name} ({student_params:,} params, {model_size_mb(self.student):.2f} MB)")
        logger.info(f"  Compression ratio: {teacher_params / student_params:.1f}x")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output: {self.output_dir}")
    
    def _setup_teacher(self) -> None:
        """Load and freeze the teacher model."""
        if self._teacher is not None:
            # Teacher provided externally
            self._teacher = self._teacher.to(self.device)
            freeze_model(self._teacher)
            return
        
        if self.config.teacher is None:
            raise ValueError("Teacher configuration not provided")
        
        # Get teacher from registry
        self._teacher_reg = registry.get(self.config.teacher.name)
        
        # Load checkpoint
        self._teacher = registry.load_model(
            name=self.config.teacher.name,
            checkpoint_path=self.config.teacher.checkpoint,
            device=self.device,
            strict=False,  # Allow partial loading for flexibility
        )
        
        # Freeze teacher
        freeze_model(self._teacher)
        
        logger.info(f"Loaded teacher from {self.config.teacher.checkpoint}")
    
    def _setup_student(self) -> None:
        """Create the student model."""
        if self._student is not None:
            # Student provided externally
            self._student = self._student.to(self.device)
            return
        
        if self.config.student is None:
            raise ValueError("Student configuration not provided")
        
        # Get student from registry
        self._student_reg = registry.get(self.config.student.name)
        
        # Merge default config with overrides
        student_config = self._student_reg.default_config.copy()
        student_config.update(self.config.student.config)
        
        # Create student
        self._student = self._student_reg.model_class(**student_config)
        self._student = self._student.to(self.device)
        
        # Optionally load pretrained weights
        if self.config.student.pretrained_checkpoint:
            checkpoint = torch.load(
                self.config.student.pretrained_checkpoint,
                map_location=self.device
            )
            if 'model_state_dict' in checkpoint:
                self._student.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self._student.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded pretrained student weights from {self.config.student.pretrained_checkpoint}")
        
        logger.info(f"Created student: {self.config.student.name}")
    
    def _setup_loss(self) -> None:
        """Set up the distillation loss function."""
        loss_config = self.config.loss
        
        # Create physics loss if specified
        physics_loss = None
        if 'physics' in loss_config.weights:
            # Try to get physics loss from model registration
            if self._student_reg and self._student_reg.physics_loss_class:
                physics_loss = self._student_reg.physics_loss_class(**loss_config.physics_config)
            elif self._teacher_reg and self._teacher_reg.physics_loss_class:
                physics_loss = self._teacher_reg.physics_loss_class(**loss_config.physics_config)
            else:
                # Try to infer from config
                physics_type = loss_config.physics_config.get('type', 'none')
                physics_loss = create_physics_loss(physics_type, **loss_config.physics_config)
        
        # Create feature matching loss if specified
        if 'feature_match' in loss_config.weights and loss_config.feature_matching:
            self.feature_matching = FeatureMatchingLoss(
                layer_mapping=loss_config.feature_matching
            )
            # Register hooks
            teacher_layers = list(loss_config.feature_matching.keys())
            student_layers = list(loss_config.feature_matching.values())
            self.feature_matching.register_teacher_hooks(self.teacher, teacher_layers)
            self.feature_matching.register_student_hooks(self.student, student_layers)
        
        # Create combined loss
        self.loss_fn = DistillationLoss(
            weights=loss_config.weights,
            temperature=loss_config.temperature,
            physics_loss=physics_loss,
            feature_matching=self.feature_matching,
        )
        
        logger.info(f"Loss weights: {loss_config.weights}")
    
    def _setup_optimizer(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        train_config = self.config.training
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        
        # Scheduler
        if train_config.scheduler == "cosine_annealing":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.num_epochs,
                **train_config.scheduler_params
            )
        elif train_config.scheduler == "cosine_annealing_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=train_config.scheduler_params.get('T_0', 50),
                T_mult=train_config.scheduler_params.get('T_mult', 2),
            )
        elif train_config.scheduler == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=train_config.scheduler_params.get('patience', 10),
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: Adam (lr={train_config.learning_rate})")
        if self.scheduler:
            logger.info(f"Scheduler: {train_config.scheduler}")
    
    def _setup_sampler(self) -> None:
        """Set up input sampling strategy."""
        sampling_config = self.config.sampling
        
        # Convert dataclass to dict for create_sampler
        config_dict = {
            'spatial_range': sampling_config.spatial_range,
            'temporal_range': sampling_config.temporal_range,
            'num_query_points_per_sample': sampling_config.num_query_points_per_sample,
            'source_dataset': sampling_config.source_dataset,
            'augmentation': sampling_config.augmentation,
            'complexity_metric': sampling_config.complexity_metric,
            'oversample_ratio': sampling_config.oversample_ratio,
            'parameter_range': getattr(sampling_config, 'parameter_range', None),
        }
        
        self.sampler = create_sampler(
            strategy=sampling_config.strategy,
            config=config_dict,
            device=self.device,
        )
        
        # If using teacher-guided sampling, set the teacher
        if hasattr(self.sampler, 'set_teacher'):
            self.sampler.set_teacher(self.teacher)
        
        logger.info(f"Sampler: {sampling_config.strategy}")
    
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average loss values for this epoch
        """
        self.student.train()
        self.metrics.reset()
        
        train_config = self.config.training
        sampling_config = self.config.sampling
        
        # Get curriculum parameters for this epoch
        curriculum_params = self._get_curriculum_params(epoch)
        
        # Determine number of samples for this epoch
        num_samples = sampling_config.num_samples_per_epoch
        
        # Training loop
        batch_count = 0
        for batch in self.sampler.sample_epoch(train_config.batch_size, num_samples):
            # Apply curriculum adjustments if any
            batch = self._apply_curriculum(batch, curriculum_params)
            
            # Ensure inputs require grad for physics losses
            inputs = batch.get('input', batch.get('coords'))
            if inputs is not None:
                inputs = inputs.clone().requires_grad_(True)
                batch['input'] = inputs
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_output = self.teacher(inputs)
            
            # Student forward
            student_output = self.student(inputs)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                teacher_output=teacher_output,
                student_output=student_output,
                inputs=batch,
                ground_truth=batch.get('output'),
                student_model=self.student,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional but helps stability)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            self.metrics.update({
                'loss': loss.item(),
                **{k: v.item() for k, v in loss_dict.items()}
            })
            
            batch_count += 1
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.metrics.average('loss'))
            else:
                self.scheduler.step()
        
        # Clear feature matching cache
        if self.feature_matching:
            self.feature_matching.clear_features()
        
        return self.metrics.summary()
    
    def _get_curriculum_params(self, epoch: int) -> dict[str, Any]:
        """Get curriculum parameters for the current epoch."""
        curriculum = self.config.training.curriculum
        if not curriculum:
            return {}
        
        for stage in curriculum:
            if stage.epochs[0] <= epoch < stage.epochs[1]:
                return stage.params
        
        return {}
    
    def _apply_curriculum(
        self, 
        batch: dict[str, torch.Tensor], 
        params: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Apply curriculum adjustments to a batch."""
        if not params:
            return batch
        
        # Example: filter by forcing range for coastal model
        if 'forcing_range' in params:
            # This is domain-specific and should be handled by custom samplers
            pass
        
        return batch
    
    def evaluate(self, epoch: int) -> dict[str, float]:
        """
        Evaluate student model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.student.eval()
        
        eval_losses = []
        eval_errors = []
        
        with torch.no_grad():
            # Generate evaluation samples
            num_eval_samples = 1000
            for batch in self.sampler.sample_epoch(
                self.config.training.batch_size, 
                num_eval_samples
            ):
                inputs = batch.get('input', batch.get('coords'))
                
                teacher_output = self.teacher(inputs)
                student_output = self.student(inputs)
                
                # Compute loss
                loss, _ = self.loss_fn(
                    teacher_output=teacher_output,
                    student_output=student_output,
                    inputs=batch,
                )
                eval_losses.append(loss.item())
                
                # Compute relative error
                error = torch.mean(torch.abs(student_output - teacher_output))
                rel_error = error / (torch.mean(torch.abs(teacher_output)) + 1e-8)
                eval_errors.append(rel_error.item())
        
        metrics = {
            'eval_loss': sum(eval_losses) / len(eval_losses),
            'relative_error': sum(eval_errors) / len(eval_errors),
        }
        
        # Check if this is the best model
        if metrics['eval_loss'] < self.best_loss:
            self.best_loss = metrics['eval_loss']
            self.best_epoch = epoch
            
            # Save best model
            save_checkpoint(
                model=self.student,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                step=epoch,
                loss=self.best_loss,
                path=self.checkpoints_dir / "best_student.pt",
                metrics=metrics,
            )
            
            logger.info(f"New best model at epoch {epoch} (loss: {self.best_loss:.6f})")
        
        self.student.train()
        return metrics
    
    def train(self) -> DistillationResult:
        """
        Run the full distillation training loop.
        
        Returns:
            DistillationResult with training outcomes
        """
        if not self._is_setup:
            self.setup()
        
        self._start_time = time.time()
        train_config = self.config.training
        
        loss_history: list[dict[str, float]] = []
        eval_history: list[dict[str, float]] = []
        
        logger.info(f"Starting distillation training for {train_config.num_epochs} epochs")
        
        # Early stopping tracking
        epochs_without_improvement = 0
        
        for epoch in range(train_config.num_epochs):
            # Train epoch
            epoch_metrics = self.train_epoch(epoch)
            loss_history.append(epoch_metrics)
            
            # Log progress
            if epoch % 10 == 0 or epoch == train_config.num_epochs - 1:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch:4d}/{train_config.num_epochs} | "
                    f"Loss: {epoch_metrics['loss']:.6f} | "
                    f"LR: {lr:.2e}"
                )
            
            # Evaluate periodically
            if epoch % train_config.eval_every == 0 or epoch == train_config.num_epochs - 1:
                eval_metrics = self.evaluate(epoch)
                eval_history.append({'epoch': epoch, **eval_metrics})
                
                logger.info(
                    f"  Eval | Loss: {eval_metrics['eval_loss']:.6f} | "
                    f"Rel Error: {eval_metrics['relative_error']:.4f}"
                )
                
                # Early stopping check
                if eval_metrics['eval_loss'] < self.best_loss * 1.01:  # Within 1% of best
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += train_config.eval_every
                
                if train_config.patience and epochs_without_improvement >= train_config.patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improvement for {train_config.patience} epochs)")
                    break
            
            # Save checkpoint periodically
            if epoch % train_config.save_every == 0 and epoch > 0:
                save_checkpoint(
                    model=self.student,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    step=epoch,
                    loss=epoch_metrics['loss'],
                    path=self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt",
                )
        
        # Final evaluation
        final_metrics = self.evaluate(train_config.num_epochs - 1)
        
        # Save final model
        save_checkpoint(
            model=self.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=train_config.num_epochs,
            step=train_config.num_epochs,
            loss=final_metrics['eval_loss'],
            path=self.checkpoints_dir / "final_student.pt",
            metrics=final_metrics,
        )
        
        training_time = time.time() - self._start_time
        
        # Compile results
        teacher_params = count_parameters(self.teacher, trainable_only=False)
        student_params = count_parameters(self.student)
        
        result = DistillationResult(
            student_path=str(self.checkpoints_dir / "best_student.pt"),
            final_loss=final_metrics['eval_loss'],
            best_loss=self.best_loss,
            epochs_trained=epoch + 1,
            training_time_seconds=training_time,
            teacher_params=teacher_params,
            student_params=student_params,
            compression_ratio=teacher_params / student_params,
            loss_history=loss_history,
            eval_history=eval_history,
        )
        
        # Save results
        save_json(
            {
                'student_path': result.student_path,
                'final_loss': result.final_loss,
                'best_loss': result.best_loss,
                'epochs_trained': result.epochs_trained,
                'training_time_seconds': result.training_time_seconds,
                'teacher_params': result.teacher_params,
                'student_params': result.student_params,
                'compression_ratio': result.compression_ratio,
            },
            self.output_dir / "distillation_result.json"
        )
        
        logger.info(f"Distillation complete in {training_time:.1f}s")
        logger.info(f"Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
        logger.info(f"Compression ratio: {result.compression_ratio:.1f}x")
        
        return result
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.feature_matching:
            self.feature_matching.remove_hooks()


def distill(
    config: DistillConfig | str,
    teacher: nn.Module | None = None,
    student: nn.Module | None = None,
    device: str | None = None,
) -> DistillationResult:
    """
    Convenience function to run distillation.
    
    Args:
        config: DistillConfig or path to YAML config file
        teacher: Optional pre-loaded teacher model
        student: Optional pre-created student model
        device: Device to train on
        
    Returns:
        DistillationResult
    """
    if isinstance(config, str):
        from .config import load_config
        config = load_config(config)
    
    trainer = DistillationTrainer(
        config=config,
        teacher=teacher,
        student=student,
        device=device,
    )
    
    try:
        return trainer.train()
    finally:
        trainer.cleanup()
