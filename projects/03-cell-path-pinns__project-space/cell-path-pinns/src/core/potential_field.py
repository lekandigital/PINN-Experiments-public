"""
Abstract Potential Field for Geodesic Trajectory Prediction.

A potential field φ(x) defines a scalar value at each point in space.
Agents move along geodesics that balance:
- Minimizing path length (efficiency)
- Following -∇φ (goal-seeking)

The gradient of the potential defines the "force" pulling agents toward
low-potential regions (or high-value regions depending on sign convention).

Domain Examples:
- Robotics: Terrain elevation + obstacle costs
- Biology: Negative log nutrient concentration
- Finance: Negative risk-adjusted profit
- Game AI: Cost landscape combining goals and obstacles
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class PotentialFieldBase(ABC, nn.Module):
    """
    Abstract base class for learnable potential fields.
    
    A potential field maps spatial positions to scalar values:
        φ: R^n → R
    
    The gradient ∇φ defines the direction of steepest ascent.
    Agents typically move in the direction of -∇φ (gradient descent)
    to reach low-potential regions (goals).
    
    Subclasses must implement:
        - forward(x): Compute potential at positions
        - gradient(x): Compute ∇φ at positions
    
    Optionally override:
        - metric_tensor(x): For Riemannian geometry with non-Euclidean metrics
    """
    
    def __init__(self, spatial_dim: int = 2, time_dependent: bool = False):
        """
        Initialize potential field.
        
        Args:
            spatial_dim: Dimensionality of the space (2 for 2D, 3 for 3D)
            time_dependent: Whether potential varies with time
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.time_dependent = time_dependent
    
    @abstractmethod
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute potential at positions x.
        
        Args:
            x: Position tensor of shape [batch, spatial_dim]
            t: Optional time tensor of shape [batch, 1] for time-dependent fields
            
        Returns:
            Potential tensor of shape [batch, 1]
        """
        pass
    
    def gradient(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient ∇φ at positions x using autograd.
        
        Args:
            x: Position tensor of shape [batch, spatial_dim]
            t: Optional time tensor of shape [batch, 1]
            
        Returns:
            Gradient tensor of shape [batch, spatial_dim]
        """
        x = x.requires_grad_(True)
        phi = self.forward(x, t)
        
        grad = torch.autograd.grad(
            phi.sum(), x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad
    
    def metric_tensor(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Riemannian metric tensor at positions x.
        
        Default implementation: conformal metric g_ij = exp(φ) * δ_ij
        
        This means higher potential = harder to traverse, which naturally
        makes geodesics avoid high-potential (costly) regions.
        
        For more complex geometries (anisotropic costs, etc.), override this method.
        
        Args:
            x: Position tensor of shape [batch, spatial_dim]
            t: Optional time tensor of shape [batch, 1]
            
        Returns:
            Metric tensor of shape [batch, spatial_dim, spatial_dim]
        """
        batch_size = x.shape[0]
        phi = self.forward(x, t)  # [batch, 1]
        
        # Conformal scaling: higher potential = harder to traverse
        scale = torch.exp(phi)  # [batch, 1]
        
        # Identity matrix scaled by potential
        identity = torch.eye(self.spatial_dim, device=x.device, dtype=x.dtype)
        metric = scale.unsqueeze(-1) * identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        return metric
    
    def laplacian(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Laplacian ∇²φ at positions x.
        
        Useful for analyzing potential field properties (e.g., finding
        local minima where ∇²φ > 0).
        
        Args:
            x: Position tensor of shape [batch, spatial_dim]
            t: Optional time tensor of shape [batch, 1]
            
        Returns:
            Laplacian tensor of shape [batch, 1]
        """
        x = x.requires_grad_(True)
        phi = self.forward(x, t)
        
        # First derivatives
        grad = torch.autograd.grad(
            phi.sum(), x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Sum of second derivatives (trace of Hessian)
        laplacian = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for i in range(self.spatial_dim):
            grad_i = grad[:, i:i+1]
            grad_ii = torch.autograd.grad(
                grad_i.sum(), x,
                create_graph=True,
                retain_graph=True
            )[0][:, i:i+1]
            laplacian = laplacian + grad_ii
        
        return laplacian


class LearnedPotentialField(PotentialFieldBase):
    """
    Fully learnable potential field using an MLP.
    
    This is the most flexible option - the potential is learned entirely
    from trajectory data without any prior structure.
    
    Architecture: spatial_dim (+ 1 if time_dependent) → hidden → hidden → 1
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        time_dependent: bool = False,
    ):
        """
        Initialize learned potential field.
        
        Args:
            spatial_dim: Input spatial dimension
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
            activation: Activation function ('relu', 'tanh', 'silu')
            time_dependent: Include time as input
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=time_dependent)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Select activation
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        act_cls = activations.get(activation.lower(), nn.ReLU)
        
        # Input dimension
        input_dim = spatial_dim + (1 if time_dependent else 0)
        
        # Build network
        layers = [nn.Linear(input_dim, hidden_dim), act_cls()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using appropriate scheme for activation."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute potential at positions.
        
        Args:
            x: Position tensor [batch, spatial_dim]
            t: Optional time tensor [batch, 1]
            
        Returns:
            Potential tensor [batch, 1]
        """
        if self.time_dependent:
            if t is None:
                raise ValueError("Time tensor required for time-dependent field")
            inputs = torch.cat([x, t], dim=-1)
        else:
            inputs = x
        
        return self.net(inputs)


class AnalyticPotentialField(PotentialFieldBase):
    """
    Potential field defined by an analytic function.
    
    Useful for:
    - Testing with known ground truth
    - Combining learned and fixed components
    - Encoding prior domain knowledge
    
    Example (quadratic bowl):
        >>> field = AnalyticPotentialField(
        ...     func=lambda x, t: (x**2).sum(dim=-1, keepdim=True),
        ...     spatial_dim=2
        ... )
    """
    
    def __init__(
        self,
        func: callable,
        spatial_dim: int = 2,
        time_dependent: bool = False,
    ):
        """
        Initialize analytic potential field.
        
        Args:
            func: Callable (x, t) -> potential where x is [batch, dim] and t is [batch, 1]
            spatial_dim: Spatial dimension
            time_dependent: Whether function uses time argument
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=time_dependent)
        self.func = func
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate analytic potential function."""
        return self.func(x, t)


class ComposedPotentialField(PotentialFieldBase):
    """
    Combine multiple potential fields with learned weights.
    
    Useful for multi-factor potentials:
        φ_total = w1*φ_obstacles + w2*φ_goal + w3*φ_terrain
    
    Example:
        >>> combined = ComposedPotentialField([
        ...     obstacle_field,
        ...     goal_field,
        ...     terrain_field,
        ... ], learnable_weights=True)
    """
    
    def __init__(
        self,
        fields: list,
        weights: Optional[list] = None,
        learnable_weights: bool = True,
    ):
        """
        Initialize composed potential field.
        
        Args:
            fields: List of PotentialFieldBase instances
            weights: Initial weights (default: uniform)
            learnable_weights: Whether weights are learnable parameters
        """
        # Infer spatial_dim and time_dependent from first field
        first_field = fields[0]
        super().__init__(
            spatial_dim=first_field.spatial_dim,
            time_dependent=any(f.time_dependent for f in fields)
        )
        
        self.fields = nn.ModuleList(fields)
        
        # Initialize weights
        n_fields = len(fields)
        if weights is None:
            weights = [1.0 / n_fields] * n_fields
        
        if learnable_weights:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        self.learnable_weights = learnable_weights
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute weighted sum of component potentials."""
        # Normalize weights if learnable (softmax)
        if self.learnable_weights:
            w = torch.softmax(self.weights, dim=0)
        else:
            w = self.weights
        
        # Sum weighted potentials
        total = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for field, weight in zip(self.fields, w):
            total = total + weight * field(x, t)
        
        return total
