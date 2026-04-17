"""
Base Adapter Interface for Rollout Models

This module defines the abstract interface that all model adapters must
implement to work with the ScheduledSamplingTrainer.

The adapter pattern decouples the training logic from model-specific details:
- The trainer doesn't know about cloth, motion, waves, etc.
- Each project implements an adapter for their specific model
- The trainer operates through the uniform adapter interface

Key design decisions:
- Adapters own a reference to the model
- State is passed as a generic dict to accommodate different model types
- Hidden state handling is explicit (not buried in the model)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class AdapterState:
    """Container for rollout state between timesteps.
    
    This is a flexible container that can hold whatever state the
    adapter needs to propagate between timesteps.
    """
    hidden: Any  # GRU hidden state, or None for non-recurrent models
    position: Optional[Tensor] = None  # Current position (for physics models)
    velocity: Optional[Tensor] = None  # Current velocity (for physics models)
    time: Optional[float] = None  # Current time (for INR models)
    extra: Optional[Dict[str, Any]] = None  # Model-specific extra state


class RolloutModelAdapter(ABC):
    """Abstract base class for model adapters.
    
    Each project implements this interface to bridge between the
    ScheduledSamplingTrainer and their specific model forward pass.
    
    The adapter handles:
    - Translating batch format to model inputs
    - Managing recurrent hidden state
    - Converting predictions back to input format
    - Computing loss with model-specific components
    
    Example implementation for a GRU-based cloth model:
    
        class MyClothAdapter(RolloutModelAdapter):
            def __init__(self, model, loss_fn):
                self.model = model
                self.loss_fn = loss_fn
            
            def init_hidden(self, batch):
                batch_size = batch['positions'].size(0)
                device = batch['positions'].device
                return self.model.gru.init_hidden(batch_size, device)
            
            def step(self, state, hidden, t, batch):
                pred, new_hidden = self.model(state, hidden)
                return pred, new_hidden
            
            # ... etc
    """
    
    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """Return the underlying PyTorch model."""
        pass
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    @abstractmethod
    def init_hidden(self, batch: Dict[str, Tensor]) -> Any:
        """Initialize the recurrent/hidden state for a new sequence.
        
        For GRU models: return zero hidden state tensor
        For INR models: return initial time value or None
        For non-recurrent models: return None
        
        Args:
            batch: Batch dict containing sequence data, used to determine
                  batch size and device
                  
        Returns:
            Initial hidden state (type depends on model architecture)
        """
        pass
    
    @abstractmethod
    def step(
        self,
        state: Union[Tensor, Dict[str, Tensor]],
        hidden: Any,
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Any]:
        """Advance one timestep.
        
        This is the core forward pass. Takes current state + hidden,
        returns prediction + new hidden.
        
        Args:
            state: Current input state (position, features, etc.)
            hidden: Current hidden state (from init_hidden or previous step)
            t: Current timestep index
            batch: Full batch dict (for accessing static data like edges)
            
        Returns:
            Tuple of (prediction, new_hidden)
            - prediction: Model output for this timestep
            - new_hidden: Updated hidden state for next timestep
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        is_teacher_forced: bool = True,
        **kwargs,
    ) -> Tensor:
        """Compute loss for a single timestep prediction.
        
        The is_teacher_forced flag allows applying different loss weighting
        or additional regularization for free-running frames.
        
        For physics-informed models, this should include physics losses
        (spring, collision, etc.) in addition to data-matching loss.
        
        Args:
            prediction: Model prediction for this timestep
            target: Ground truth target for this timestep
            is_teacher_forced: Whether this step used teacher forcing
            **kwargs: Additional arguments (prev_prediction, dt, etc.)
            
        Returns:
            Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def get_initial_state(self, batch: Dict[str, Tensor]) -> Union[Tensor, Dict[str, Tensor]]:
        """Get the initial input state from batch.
        
        This is the first input fed to step() at t=0.
        
        Args:
            batch: Batch dict containing sequence data
            
        Returns:
            Initial state (format depends on model)
        """
        pass
    
    @abstractmethod
    def get_target(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        """Get ground truth target for timestep t.
        
        Args:
            batch: Batch dict containing sequence data
            t: Timestep index
            
        Returns:
            Target tensor for this timestep
        """
        pass
    
    @abstractmethod
    def get_teacher_input(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        """Get ground truth input for timestep t (for teacher forcing).
        
        This is what would be fed to step() if using teacher forcing.
        Often the same as get_target(batch, t-1) but not always.
        
        Args:
            batch: Batch dict containing sequence data
            t: Timestep index
            
        Returns:
            Ground truth input for this timestep
        """
        pass
    
    @abstractmethod
    def prediction_to_input(self, prediction: Tensor) -> Tensor:
        """Convert model prediction to input format for next step.
        
        For many models this is identity (prediction IS the next input).
        For others, may need transformation (e.g., add displacement to position).
        
        Args:
            prediction: Model prediction from step()
            
        Returns:
            Input tensor for next timestep
        """
        pass
    
    def get_sequence_length(self, batch: Dict[str, Tensor]) -> int:
        """Get the sequence length from batch.
        
        Default implementation assumes batch has 'positions' with shape (B, T, ...).
        Override if your batch format is different.
        
        Args:
            batch: Batch dict containing sequence data
            
        Returns:
            Number of timesteps in sequence
        """
        # Try common key names
        for key in ['positions', 'frames', 'sequence', 'sdf_samples', 'trajectory']:
            if key in batch:
                return batch[key].size(1)
        
        # Fall back to first tensor with 3+ dims
        for key, val in batch.items():
            if isinstance(val, Tensor) and val.dim() >= 3:
                return val.size(1)
        
        raise ValueError("Could not determine sequence length from batch")
    
    def detach_hidden(self, hidden: Any) -> Any:
        """Detach hidden state from computation graph.
        
        Used for truncated BPTT. Default handles common cases.
        Override for complex hidden state structures.
        
        Args:
            hidden: Hidden state to detach
            
        Returns:
            Detached hidden state
        """
        if hidden is None:
            return None
        if isinstance(hidden, Tensor):
            return hidden.detach()
        if isinstance(hidden, tuple):
            return tuple(h.detach() if isinstance(h, Tensor) else h for h in hidden)
        if isinstance(hidden, dict):
            return {k: v.detach() if isinstance(v, Tensor) else v for k, v in hidden.items()}
        return hidden
    
    def on_epoch_start(self, epoch: int, total_epochs: int):
        """Hook called at the start of each epoch.
        
        Override to do per-epoch setup (e.g., update loss weights).
        """
        pass
    
    def on_epoch_end(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """Hook called at the end of each epoch.
        
        Override to do per-epoch cleanup or logging.
        """
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get adapter-specific diagnostics for logging.
        
        Override to expose model-specific metrics.
        """
        return {}


class SimpleAdapter(RolloutModelAdapter):
    """Simple adapter for models with standard GRU-like interface.
    
    This adapter works out of the box for models that:
    - Have a forward(state, hidden) -> (prediction, new_hidden) signature
    - Take position tensors as input
    - Produce position tensors as output
    
    For more complex models, subclass RolloutModelAdapter directly.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        position_key: str = 'positions',
        hidden_init_fn: Optional[callable] = None,
    ):
        """
        Args:
            model: PyTorch model with forward(state, hidden) signature
            loss_fn: Loss function loss_fn(pred, target) -> scalar
            position_key: Key in batch dict for position sequence
            hidden_init_fn: Optional function to init hidden state.
                           If None, looks for model.init_hidden()
        """
        self._model = model
        self.loss_fn = loss_fn
        self.position_key = position_key
        self.hidden_init_fn = hidden_init_fn
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    def init_hidden(self, batch: Dict[str, Tensor]) -> Any:
        positions = batch[self.position_key]
        batch_size = positions.size(0)
        device = positions.device
        
        if self.hidden_init_fn is not None:
            return self.hidden_init_fn(batch_size, device)
        elif hasattr(self._model, 'init_hidden'):
            return self._model.init_hidden(batch_size, device)
        else:
            return None
    
    def step(
        self,
        state: Tensor,
        hidden: Any,
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Any]:
        if hidden is None:
            prediction = self._model(state)
            return prediction, None
        else:
            prediction, new_hidden = self._model(state, hidden)
            return prediction, new_hidden
    
    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        is_teacher_forced: bool = True,
        **kwargs,
    ) -> Tensor:
        return self.loss_fn(prediction, target)
    
    def get_initial_state(self, batch: Dict[str, Tensor]) -> Tensor:
        return batch[self.position_key][:, 0]
    
    def get_target(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        return batch[self.position_key][:, t]
    
    def get_teacher_input(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        return batch[self.position_key][:, t]
    
    def prediction_to_input(self, prediction: Tensor) -> Tensor:
        return prediction
