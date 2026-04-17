"""
Physics-Informed Neural Network for Trajectory Prediction.

TrajectoryPINN maps time t → position x(t), optionally conditioned on
context like start/end points, agent properties, or environment parameters.

Architecture is designed for:
- Smooth trajectories (Tanh activation for continuous derivatives)
- Small model size (~13K params for real-time inference)
- Easy ONNX export for deployment
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class TrajectoryPINN(nn.Module):
    """
    Physics-Informed Neural Network for trajectory prediction.
    
    Maps time t (and optional context) to spatial position x(t).
    
    The network predicts continuous trajectories that can be evaluated
    at any time point, enabling physics losses that require derivatives
    (velocity, acceleration) computed via autograd.
    
    Architecture: (1 + context_dim) → hidden → hidden → ... → spatial_dim
    
    Example:
        >>> model = TrajectoryPINN(spatial_dim=2, context_dim=4)
        >>> t = torch.linspace(0, 1, 100).unsqueeze(-1)  # [100, 1]
        >>> context = torch.tensor([[0, 0, 1, 1]])  # start=(0,0), end=(1,1)
        >>> path = model(t, context.expand(100, -1))  # [100, 2]
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        context_dim: int = 0,
        activation: str = "tanh",
        use_skip_connections: bool = False,
    ):
        """
        Initialize TrajectoryPINN.
        
        Args:
            spatial_dim: Output dimension (2 for 2D, 3 for 3D)
            hidden_dim: Width of hidden layers
            num_layers: Number of hidden layers
            context_dim: Dimension of conditioning context (0 for unconditional)
            activation: Activation function ('tanh', 'silu', 'gelu')
            use_skip_connections: Add residual connections
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.use_skip_connections = use_skip_connections
        
        # Select activation (Tanh preferred for smooth derivatives)
        activations = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
        }
        self.act_cls = activations.get(activation.lower(), nn.Tanh)
        
        # Input: time (1) + context
        input_dim = 1 + context_dim
        
        # Build network
        if use_skip_connections:
            self._build_residual_network(input_dim)
        else:
            self._build_sequential_network(input_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _build_sequential_network(self, input_dim: int):
        """Build standard sequential MLP."""
        layers = [nn.Linear(input_dim, self.hidden_dim), self.act_cls()]
        
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.act_cls()
            ])
        
        layers.append(nn.Linear(self.hidden_dim, self.spatial_dim))
        self.net = nn.Sequential(*layers)
    
    def _build_residual_network(self, input_dim: int):
        """Build network with skip connections."""
        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.input_act = self.act_cls()
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.hidden_acts.append(self.act_cls())
        
        self.output_layer = nn.Linear(self.hidden_dim, self.spatial_dim)
        self.net = None  # Not used in residual mode
    
    def _init_weights(self):
        """Initialize weights using Xavier (good for Tanh)."""
        modules = self.net.modules() if self.net is not None else self.modules()
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict position at time t.
        
        Args:
            t: Time tensor [batch, 1] or [batch]
            context: Optional context [batch, context_dim]
            
        Returns:
            Position tensor [batch, spatial_dim]
        """
        # Ensure t is 2D
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Concatenate input
        if context is not None and self.context_dim > 0:
            x = torch.cat([t, context], dim=-1)
        else:
            x = t
        
        # Forward pass
        if self.use_skip_connections:
            h = self.input_act(self.input_layer(x))
            for layer, act in zip(self.hidden_layers, self.hidden_acts):
                h = h + act(layer(h))  # Residual connection
            return self.output_layer(h)
        else:
            return self.net(x)
    
    def trajectory(
        self,
        t_samples: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get full trajectory at multiple time points.
        
        Args:
            t_samples: Time points [n_times] or [n_times, 1]
            context: Context for trajectory [context_dim]
            
        Returns:
            Trajectory tensor [n_times, spatial_dim]
        """
        if t_samples.dim() == 1:
            t_samples = t_samples.unsqueeze(-1)
        
        n_times = t_samples.shape[0]
        
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            context = context.expand(n_times, -1)
        
        return self(t_samples, context)
    
    def velocity(
        self,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity dx/dt at time t using autograd.
        
        Args:
            t: Time tensor [batch, 1]
            context: Optional context [batch, context_dim]
            
        Returns:
            Velocity tensor [batch, spatial_dim]
        """
        t = t.requires_grad_(True)
        x = self(t, context)
        
        # Compute dx/dt for each spatial dimension
        velocity = torch.zeros_like(x)
        for i in range(self.spatial_dim):
            grad = torch.autograd.grad(
                x[:, i].sum(), t,
                create_graph=True,
                retain_graph=True
            )[0]
            velocity[:, i] = grad.squeeze(-1)
        
        return velocity
    
    def acceleration(
        self,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute acceleration d²x/dt² at time t using autograd.
        
        Args:
            t: Time tensor [batch, 1]
            context: Optional context [batch, context_dim]
            
        Returns:
            Acceleration tensor [batch, spatial_dim]
        """
        t = t.requires_grad_(True)
        v = self.velocity(t, context)
        
        # Compute dv/dt for each spatial dimension
        acceleration = torch.zeros_like(v)
        for i in range(self.spatial_dim):
            grad = torch.autograd.grad(
                v[:, i].sum(), t,
                create_graph=True,
                retain_graph=True
            )[0]
            acceleration[:, i] = grad.squeeze(-1)
        
        return acceleration
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            'spatial_dim': self.spatial_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'context_dim': self.context_dim,
            'use_skip_connections': self.use_skip_connections,
        }


class BoundaryConditionedTrajectory(TrajectoryPINN):
    """
    Trajectory network with hard-coded boundary conditions.
    
    Guarantees x(0) = start and x(1) = end exactly (not approximately
    via loss), using the formulation:
    
        x(t) = (1-t) * start + t * end + t*(1-t) * NN(t, context)
    
    This eliminates the need for boundary loss terms and ensures
    exact endpoint matching.
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        context_dim: int = 0,
        activation: str = "tanh",
    ):
        """
        Initialize boundary-conditioned trajectory network.
        
        Note: context_dim should NOT include start/end points, as those
        are handled separately by the boundary formulation.
        """
        # We need start+end in context, but they're handled specially
        super().__init__(
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            context_dim=context_dim,  # Additional context beyond start/end
            activation=activation,
        )
    
    def forward(
        self,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        start: Optional[torch.Tensor] = None,
        end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict position with guaranteed boundary conditions.
        
        Args:
            t: Time tensor [batch, 1] in range [0, 1]
            context: Optional additional context [batch, context_dim]
            start: Start position [batch, spatial_dim] or [spatial_dim]
            end: End position [batch, spatial_dim] or [spatial_dim]
            
        Returns:
            Position tensor [batch, spatial_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        batch_size = t.shape[0]
        
        # Handle start/end dimensions
        if start is not None and start.dim() == 1:
            start = start.unsqueeze(0).expand(batch_size, -1)
        if end is not None and end.dim() == 1:
            end = end.unsqueeze(0).expand(batch_size, -1)
        
        # Get NN output (deviation from linear interpolation)
        nn_out = super().forward(t, context)  # [batch, spatial_dim]
        
        if start is not None and end is not None:
            # x(t) = (1-t)*start + t*end + t*(1-t)*NN(t)
            # This guarantees x(0) = start, x(1) = end
            linear_interp = (1 - t) * start + t * end
            deviation = t * (1 - t) * nn_out
            return linear_interp + deviation
        else:
            return nn_out


class MultiTrajectoryPINN(nn.Module):
    """
    Network for predicting multiple trajectories simultaneously.
    
    Useful for:
    - Batch inference of many trajectories
    - Multi-agent path planning
    - Ensemble predictions
    """
    
    def __init__(
        self,
        n_trajectories: int,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        shared_encoder: bool = True,
    ):
        """
        Initialize multi-trajectory network.
        
        Args:
            n_trajectories: Number of trajectories to predict
            spatial_dim: Spatial dimension
            hidden_dim: Hidden layer width
            num_layers: Number of layers
            shared_encoder: Share encoder across trajectories
        """
        super().__init__()
        
        self.n_trajectories = n_trajectories
        self.spatial_dim = spatial_dim
        
        if shared_encoder:
            # Shared encoder, separate output heads
            self.encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                *[layer for _ in range(num_layers - 1) 
                  for layer in [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]]
            )
            self.heads = nn.ModuleList([
                nn.Linear(hidden_dim, spatial_dim) for _ in range(n_trajectories)
            ])
        else:
            # Separate networks for each trajectory
            self.trajectories = nn.ModuleList([
                TrajectoryPINN(spatial_dim, hidden_dim, num_layers)
                for _ in range(n_trajectories)
            ])
            self.encoder = None
            self.heads = None
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Predict all trajectories at time t.
        
        Args:
            t: Time tensor [batch, 1]
            
        Returns:
            Positions [batch, n_trajectories, spatial_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        if self.encoder is not None:
            # Shared encoder
            h = self.encoder(t)
            outputs = torch.stack([head(h) for head in self.heads], dim=1)
        else:
            # Separate networks
            outputs = torch.stack([net(t) for net in self.trajectories], dim=1)
        
        return outputs
