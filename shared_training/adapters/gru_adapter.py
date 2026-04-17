"""
GRU Adapter for models with explicit GRU/RNN hidden state.

This adapter is designed for:
- Project 05 (ClothGNN): Encoder-GRU-Decoder with GRUCell
- Project 08 (HGNN-ClothDyn): VelocityGRU for temporal dynamics
- Project 09 (HGNN-NIF-Cloth animation): TemporalHGNN_NIF with velocity encoder
- Project 12 (NIF-Cloth4D-Temporal): TemporalGRU conditioning

Common pattern: model has a GRU that produces hidden state which conditions
the next prediction. The adapter manages the hidden state lifecycle and
ensures it's properly passed between timesteps.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base_adapter import RolloutModelAdapter


class GRUAdapter(RolloutModelAdapter):
    """Adapter for models with GRU-based temporal dynamics.
    
    Handles the common pattern where:
    1. Current state (position/features) is encoded
    2. GRU updates hidden state based on encoded features
    3. Decoder produces prediction conditioned on hidden state
    
    Supports both:
    - Models with explicit GRU module (forward returns (pred, hidden))
    - Models where GRU is internal and hidden is managed separately
    
    Example usage for Project 05 (ClothGNN):
    
        model = ClothGNNModel(...)
        adapter = GRUAdapter(
            model=model,
            loss_fn=compute_cloth_loss,
            position_key='positions',
            # ClothGNN outputs displacement, not position
            output_is_displacement=True,
        )
        
    Example usage for Project 09 (HGNN-NIF-Cloth animation):
    
        model = TemporalHGNN_NIF(base_model, ...)
        adapter = GRUAdapter(
            model=model,
            loss_fn=physics_loss,
            position_key='frame_history',
            # Model takes frame history as input
            use_frame_history=True,
            history_length=3,
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        position_key: str = 'positions',
        edge_key: str = 'edge_index',
        velocity_key: Optional[str] = 'velocities',
        hidden_dim: Optional[int] = None,
        output_is_displacement: bool = False,
        use_frame_history: bool = False,
        history_length: int = 3,
        dt: float = 1.0 / 30.0,
        physics_losses: Optional[Dict[str, Callable]] = None,
        physics_loss_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            model: PyTorch model with GRU-based forward pass
            loss_fn: Primary loss function (pred, target) -> scalar
            position_key: Key in batch for position sequence
            edge_key: Key in batch for edge connectivity
            velocity_key: Key in batch for velocity sequence (optional)
            hidden_dim: GRU hidden dimension (auto-detected if None)
            output_is_displacement: If True, model outputs displacement not position
            use_frame_history: If True, model takes frame history as input
            history_length: Number of history frames (if use_frame_history)
            dt: Timestep for physics calculations
            physics_losses: Dict of {name: loss_fn} for physics losses
            physics_loss_weights: Dict of {name: weight} for physics losses
        """
        self._model = model
        self.loss_fn = loss_fn
        self.position_key = position_key
        self.edge_key = edge_key
        self.velocity_key = velocity_key
        self.output_is_displacement = output_is_displacement
        self.use_frame_history = use_frame_history
        self.history_length = history_length
        self.dt = dt
        self.physics_losses = physics_losses or {}
        self.physics_loss_weights = physics_loss_weights or {}
        
        # Auto-detect hidden dimension
        if hidden_dim is None:
            hidden_dim = self._detect_hidden_dim()
        self.hidden_dim = hidden_dim
        
        # Track previous prediction for velocity/physics calculations
        self._prev_prediction: Optional[Tensor] = None
        self._frame_history: Optional[Tensor] = None
    
    def _detect_hidden_dim(self) -> int:
        """Try to detect GRU hidden dimension from model."""
        # Look for common GRU attribute names
        for name in ['gru', 'rnn', 'velocity_encoder', 'temporal_gru']:
            if hasattr(self._model, name):
                module = getattr(self._model, name)
                if hasattr(module, 'hidden_size'):
                    return module.hidden_size
                elif hasattr(module, 'hidden_dim'):
                    return module.hidden_dim
        
        # Default fallback
        return 64
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    def init_hidden(self, batch: Dict[str, Tensor]) -> Tensor:
        """Initialize GRU hidden state.
        
        Returns tensor of shape (num_layers * num_directions, batch_size, hidden_dim)
        or (batch_size, hidden_dim) for GRUCell.
        """
        positions = batch[self.position_key]
        
        # Handle frame history format (B, T, N, 3)
        if positions.dim() == 4:
            batch_size = positions.size(0)
        else:
            batch_size = positions.size(0)
        
        device = positions.device
        dtype = positions.dtype
        
        # Reset tracking state
        self._prev_prediction = None
        self._frame_history = None
        
        # Use model's init method if available
        if hasattr(self._model, 'init_hidden'):
            return self._model.init_hidden(batch_size, device, dtype)
        elif hasattr(self._model, 'gru') and hasattr(self._model.gru, 'init_hidden'):
            return self._model.gru.init_hidden(batch_size, device, dtype)
        else:
            # Default: single layer GRUCell format
            return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
    
    def step(
        self,
        state: Union[Tensor, Dict[str, Tensor]],
        hidden: Tensor,
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Forward one timestep through the model.
        
        Handles multiple model interface patterns:
        1. forward(state, hidden) -> (prediction, new_hidden)
        2. forward(frame_history, edges, ...) -> dict with 'positions'
        3. forward(data, hidden) -> (prediction, new_hidden) for PyG Data
        """
        # Get static batch data
        edge_index = batch.get(self.edge_key)
        
        # Build model input based on interface
        if self.use_frame_history:
            # Project 09 animation variant: takes frame history
            if self._frame_history is None:
                # Initialize from batch
                positions = batch[self.position_key]
                if positions.dim() == 4:
                    # Already (B, T, N, 3)
                    self._frame_history = positions[:, :self.history_length]
                else:
                    # (B, N, 3) single frame - replicate
                    self._frame_history = state.unsqueeze(1).repeat(1, self.history_length, 1, 1)
            
            # Forward with frame history
            output = self._model(
                self._frame_history,
                edge_index,
                batch.get('coarse_edge_index'),
            )
            
            # Extract prediction
            if isinstance(output, dict):
                prediction = output.get('positions', output.get('prediction'))
            else:
                prediction = output
            
            # Update frame history
            self._frame_history = torch.cat([
                self._frame_history[:, 1:],
                prediction.unsqueeze(1)
            ], dim=1)
            
            # No explicit hidden state for this pattern
            new_hidden = hidden
            
        elif hasattr(self._model, 'forward') and 'hidden' in str(self._model.forward.__code__.co_varnames):
            # Standard pattern: forward(state, hidden) -> (pred, hidden)
            prediction, new_hidden = self._model(state, hidden)
        else:
            # Try calling with just state
            output = self._model(state)
            if isinstance(output, tuple):
                prediction, new_hidden = output
            else:
                prediction = output
                new_hidden = hidden
        
        # Handle displacement output
        if self.output_is_displacement:
            if isinstance(state, dict):
                current_pos = state.get('position', state.get('pos'))
            else:
                current_pos = state
            prediction = current_pos + prediction
        
        # Track for physics loss
        self._prev_prediction = prediction.detach()
        
        return prediction, new_hidden
    
    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        is_teacher_forced: bool = True,
        **kwargs,
    ) -> Tensor:
        """Compute loss including physics components.
        
        For free-running steps, physics losses are especially important
        as they provide ground-truth-independent training signal.
        """
        # Primary data loss
        loss = self.loss_fn(prediction, target)
        
        # Add physics losses
        for name, loss_fn in self.physics_losses.items():
            weight = self.physics_loss_weights.get(name, 1.0)
            
            # Pass relevant kwargs to physics loss
            physics_kwargs = {
                'prediction': prediction,
                'target': target,
                'prev_prediction': kwargs.get('prev_prediction', self._prev_prediction),
                'edge_index': kwargs.get('edge_index'),
                'dt': self.dt,
            }
            
            # Filter to only kwargs the loss function accepts
            try:
                physics_loss = loss_fn(**physics_kwargs)
            except TypeError:
                # Try simpler signature
                physics_loss = loss_fn(prediction, target)
            
            loss = loss + weight * physics_loss
        
        return loss
    
    def get_initial_state(self, batch: Dict[str, Tensor]) -> Tensor:
        """Get initial position state."""
        positions = batch[self.position_key]
        
        if positions.dim() == 4:
            # (B, T, N, 3) -> first frame
            return positions[:, 0]
        else:
            # (B, N, 3) already single frame
            return positions[:, 0] if positions.dim() == 3 else positions
    
    def get_target(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        """Get target position for timestep t."""
        positions = batch[self.position_key]
        
        if positions.dim() == 4:
            return positions[:, t]
        else:
            return positions[:, t]
    
    def get_teacher_input(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        """Get ground truth input for teacher forcing."""
        return self.get_target(batch, t)
    
    def prediction_to_input(self, prediction: Tensor) -> Tensor:
        """Convert prediction to input for next step."""
        # For GRU models, prediction IS the next input
        return prediction
    
    def get_sequence_length(self, batch: Dict[str, Tensor]) -> int:
        """Get sequence length from batch."""
        positions = batch[self.position_key]
        
        if positions.dim() == 4:
            # (B, T, N, 3)
            return positions.size(1)
        else:
            # (B, T, 3) or similar
            return positions.size(1)


class Project05Adapter(GRUAdapter):
    """Specialized adapter for Project 05 (ClothGNN).
    
    ClothGNN has:
    - Encoder-GRU-Decoder architecture
    - GRUCell (not full GRU)
    - Outputs displacement, not position
    - Forward signature: forward(data, hidden) -> (displacement, new_hidden)
    
    The model expects PyG Data objects, so we need to build those.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        rest_lengths: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            output_is_displacement=True,
            **kwargs,
        )
        self.rest_lengths = rest_lengths
        self._current_velocity: Optional[Tensor] = None
    
    def init_hidden(self, batch: Dict[str, Tensor]) -> Tensor:
        """Initialize GRUCell hidden state (N, hidden_dim)."""
        positions = batch[self.position_key]
        
        # ClothGNN uses per-node hidden state
        if positions.dim() == 3:
            # (B, N, 3) - use N
            num_nodes = positions.size(1)
        else:
            num_nodes = positions.size(0)
        
        device = positions.device
        dtype = positions.dtype
        
        self._prev_prediction = None
        self._current_velocity = None
        
        if hasattr(self._model, 'init_hidden'):
            return self._model.init_hidden(num_nodes, device)
        else:
            return torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype)
    
    def step(
        self,
        state: Tensor,
        hidden: Tensor,
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Forward through ClothGNN.
        
        Builds PyG Data object and calls model.
        """
        from torch_geometric.data import Data
        
        edge_index = batch[self.edge_key]
        
        # Compute velocity
        if self._prev_prediction is not None:
            velocity = (state - self._prev_prediction) / self.dt
        else:
            velocity = torch.zeros_like(state)
        
        # Build node features: [position, velocity, ...]
        node_features = self._build_node_features(state, velocity)
        
        # Create PyG Data
        data = Data(
            x=node_features,
            edge_index=edge_index,
            pos=state,
        )
        
        # Forward pass
        displacement, new_hidden = self._model(data, hidden)
        
        # Add displacement to get prediction
        prediction = state + displacement
        
        self._prev_prediction = prediction.detach()
        self._current_velocity = velocity
        
        return prediction, new_hidden
    
    def _build_node_features(self, position: Tensor, velocity: Tensor) -> Tensor:
        """Build node feature vector."""
        # Default: concatenate position and velocity
        return torch.cat([position, velocity], dim=-1)


class Project09Adapter(GRUAdapter):
    """Specialized adapter for Project 09 (HGNN-NIF-Cloth animation variant).
    
    TemporalHGNN_NIF has:
    - GRU velocity encoder processing frame history
    - Base HGNN-NIF model for spatial encoding
    - Outputs dict with 'positions', 'velocity', 'latent'
    
    Forward signature: forward(frame_history, fine_edges, coarse_edges, ...) -> dict
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        sdf_loss_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            use_frame_history=True,
            **kwargs,
        )
        self.sdf_loss_fn = sdf_loss_fn
    
    def step(
        self,
        state: Tensor,
        hidden: Tensor,
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Forward through TemporalHGNN_NIF."""
        # Update frame history
        if self._frame_history is None:
            positions = batch[self.position_key]
            self._frame_history = positions[:, :self.history_length].clone()
        
        # Replace latest frame with current state
        self._frame_history = torch.cat([
            self._frame_history[:, 1:],
            state.unsqueeze(1)
        ], dim=1)
        
        # Get graph data
        fine_edges = batch.get('fine_edge_index', batch.get(self.edge_key))
        coarse_edges = batch.get('coarse_edge_index')
        
        # Forward pass
        output = self._model(
            self._frame_history,
            fine_edges,
            coarse_edges,
        )
        
        # Extract position prediction
        prediction = output['positions']
        
        self._prev_prediction = prediction.detach()
        
        # No explicit hidden state
        return prediction, hidden
    
    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        is_teacher_forced: bool = True,
        **kwargs,
    ) -> Tensor:
        """Compute loss including SDF loss if available."""
        loss = super().compute_loss(prediction, target, is_teacher_forced, **kwargs)
        
        # Add SDF loss if provided
        if self.sdf_loss_fn is not None and 'query_points' in kwargs:
            sdf_loss = self.sdf_loss_fn(
                prediction,
                kwargs['query_points'],
                kwargs.get('gt_sdf'),
            )
            loss = loss + sdf_loss
        
        return loss


class Project12Adapter(GRUAdapter):
    """Specialized adapter for Project 12 (NIF-Cloth4D-Temporal).
    
    Uses TemporalGRU to condition a Fourier Feature MLP implicit field.
    
    Forward signature: forward(xyz, time, hidden) -> (sdf, new_hidden)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        collision_loss_fn: Optional[Callable] = None,
        **kwargs,
    ):
        # Remove keys not applicable to SDF models
        kwargs.pop('output_is_displacement', None)
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            position_key='sdf_samples',  # (B, T, num_points, 4) where last dim is [x,y,z,sdf]
            **kwargs,
        )
        self.collision_loss_fn = collision_loss_fn
    
    def get_initial_state(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Get initial SDF query state."""
        sdf_samples = batch[self.position_key]  # (B, T, num_points, 4)
        time = batch.get('time', torch.zeros(sdf_samples.size(0), 1))
        
        return {
            'xyz': sdf_samples[:, 0, :, :3],  # (B, num_points, 3)
            'time': time[:, 0:1],
        }
    
    def get_target(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        """Get target SDF values for timestep t."""
        sdf_samples = batch[self.position_key]
        return sdf_samples[:, t, :, 3:4]  # (B, num_points, 1)
    
    def get_teacher_input(self, batch: Dict[str, Tensor], t: int) -> Dict[str, Tensor]:
        """Get ground truth input for timestep t."""
        sdf_samples = batch[self.position_key]
        time = batch.get('time')
        
        return {
            'xyz': sdf_samples[:, t, :, :3],
            'time': time[:, t:t+1] if time is not None else None,
        }
    
    def step(
        self,
        state: Dict[str, Tensor],
        hidden: Tensor,
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Forward through NIF-Cloth4D-Temporal."""
        xyz = state['xyz']
        time = state.get('time')
        
        # Forward pass
        pred_sdf, new_hidden = self._model(xyz, time, hidden)
        
        return pred_sdf, new_hidden
    
    def prediction_to_input(self, prediction: Tensor) -> Dict[str, Tensor]:
        """For SDF models, we don't feed SDF back as input.
        
        Instead, the time advances and we query at new points.
        This adapter returns None to signal special handling needed.
        """
        # This is handled specially - the xyz points come from batch
        return prediction
