"""
PhysicsEncodedConv — Base class for physics-encoded graph convolution layers.

Design Principle:
    In standard GNN message passing, the network learns the ENTIRE message function
    from data. In physics-encoded convolution, we HARD-CODE the known physics
    (conservation laws, constitutive relations, governing equations) as the primary
    message function, and the network learns ONLY the corrections/residuals.

    Total message = analytical_physics_term + learned_correction_term

    This dramatically improves:
    - Data efficiency (the network doesn't waste capacity learning F=ma)
    - Physical plausibility (the base prediction always satisfies known laws)
    - Generalization (analytical terms extrapolate correctly; learned terms handle
      the regime-specific corrections the physics doesn't capture)
    - Training stability (the loss landscape is much smoother when the network
      only needs to learn small corrections)

Architecture Pattern:
    1. compute_edge_physics()  — ANALYTICAL, no learnable parameters
       Computes forces/fluxes/accelerations from KNOWN physical laws.
       This is the part you'd write down on a whiteboard with equations.

    2. compute_edge_correction() — LEARNED, has parameters
       An MLP (or other network) that takes edge features and predicts
       a correction/modulation to the analytical term.

    3. aggregate() — Scatter messages back to nodes
       Standard GNN aggregation (sum, mean) with optional attention.

    4. update() — Apply aggregated messages to update node states
       May include time integration (Euler, Verlet, RK4).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Literal

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.nn import MessagePassing
    from torch_scatter import scatter_add, scatter_mean
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = nn.Module  # Fallback


@dataclass
class PhysicsConvConfig:
    """Configuration for PhysicsEncodedConv layers.
    
    Attributes:
        correction_hidden_dim: Hidden dimension for correction MLP
        correction_layers: Number of layers in correction MLP
        correction_activation: Activation function ("silu", "relu", "tanh")
        combine_mode: How to combine physics and correction ("additive" or "multiplicative")
        correction_scale_init: Initial scaling of correction output (start near zero)
        aggregation: Message aggregation method ("sum", "mean", "attention")
        track_diagnostics: Whether to track physics_fraction and correction_magnitude
        physics_dtype: Dtype for physics computations (float64 for precision)
        edge_dim: Dimension of edge attributes (for correction MLP input)
        node_dim: Dimension of node features
        output_dim: Dimension of output (typically 3 for forces/velocities)
    """
    correction_hidden_dim: int = 64
    correction_layers: int = 2
    correction_activation: str = "silu"
    combine_mode: Literal["additive", "multiplicative"] = "additive"
    correction_scale_init: float = 0.01
    aggregation: Literal["sum", "mean", "attention"] = "sum"
    track_diagnostics: bool = True
    physics_dtype: torch.dtype = torch.float64
    edge_dim: int = 1
    node_dim: int = 3
    output_dim: int = 3


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
    }
    return activations.get(name.lower(), nn.SiLU())


class CorrectionMLP(nn.Module):
    """Small MLP for learning corrections to analytical physics.
    
    Initialized with small weights so output starts near zero,
    meaning the model initially outputs pure analytical physics.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "silu",
        scale_init: float = 0.01,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.scale = scale_init
        
        # Initialize final layer with small weights
        self._init_near_zero()
    
    def _init_near_zero(self):
        """Initialize final layer to output near-zero values."""
        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.bias)
        nn.init.normal_(final_layer.weight, mean=0.0, std=self.scale)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PhysicsEncodedConvBase(nn.Module):
    """Base class for physics-encoded convolution (non-PyG version).
    
    Use this when PyTorch Geometric is not available or when you need
    custom message-passing logic.
    """
    
    def __init__(self, config: Optional[PhysicsConvConfig] = None):
        super().__init__()
        self.config = config or PhysicsConvConfig()
        
        # Diagnostics tracking
        self._physics_norm = 0.0
        self._correction_norm = 0.0
        self._total_norm = 0.0
        self._num_edges = 0
        
        # Build correction MLP (can be None for pure physics)
        self.correction_mlp: Optional[CorrectionMLP] = None
    
    def _build_correction_mlp(self, input_dim: int):
        """Build the correction MLP with proper dimensions."""
        self.correction_mlp = CorrectionMLP(
            input_dim=input_dim,
            output_dim=self.config.output_dim,
            hidden_dim=self.config.correction_hidden_dim,
            num_layers=self.config.correction_layers,
            activation=self.config.correction_activation,
            scale_init=self.config.correction_scale_init,
        )
    
    @abstractmethod
    def compute_edge_physics(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute analytical physics-based message for each edge.
        
        This method MUST NOT contain any learnable parameters.
        It encodes known physical laws (Hooke's law, conservation laws, etc.)
        
        Args:
            x_i: Source node features [num_edges, node_dim]
            x_j: Target node features [num_edges, node_dim]
            edge_attr: Edge attributes [num_edges, edge_dim]
            **kwargs: Additional domain-specific inputs
        
        Returns:
            physics_message: Analytical physics term [num_edges, output_dim]
        """
        raise NotImplementedError
    
    @abstractmethod
    def physics_name(self) -> str:
        """Return human-readable name of the encoded physics."""
        raise NotImplementedError
    
    def compute_edge_correction(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        physics_message: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute learned correction to the analytical physics.
        
        Default implementation uses a small MLP. Override for custom logic.
        
        Args:
            x_i: Source node features [num_edges, node_dim]
            x_j: Target node features [num_edges, node_dim]
            edge_attr: Edge attributes [num_edges, edge_dim]
            physics_message: The analytical physics term [num_edges, output_dim]
            **kwargs: Additional inputs (e.g., GRU hidden states)
        
        Returns:
            correction: Learned correction term [num_edges, output_dim]
        """
        if self.correction_mlp is None:
            # No correction MLP - return zeros
            return torch.zeros_like(physics_message)
        
        # Build input features for correction MLP
        features = [x_i, x_j]
        if edge_attr is not None:
            features.append(edge_attr)
        if physics_message is not None:
            features.append(physics_message)
        
        mlp_input = torch.cat(features, dim=-1)
        return self.correction_mlp(mlp_input)
    
    def combine_physics_and_correction(
        self,
        physics_msg: Tensor,
        correction_msg: Tensor
    ) -> Tensor:
        """Combine analytical physics with learned correction.
        
        Args:
            physics_msg: Analytical physics term [num_edges, output_dim]
            correction_msg: Learned correction [num_edges, output_dim]
        
        Returns:
            combined: Final message [num_edges, output_dim]
        """
        if self.config.combine_mode == "additive":
            return physics_msg + correction_msg
        elif self.config.combine_mode == "multiplicative":
            # Multiplicative: physics * (1 + correction)
            # This modulates the physics rather than adding to it
            return physics_msg * (1.0 + correction_msg)
        else:
            raise ValueError(f"Unknown combine_mode: {self.config.combine_mode}")
    
    def aggregate(
        self,
        messages: Tensor,
        index: Tensor,
        dim_size: int
    ) -> Tensor:
        """Aggregate messages to nodes.
        
        Args:
            messages: Edge messages [num_edges, output_dim]
            index: Target node indices [num_edges]
            dim_size: Number of nodes
        
        Returns:
            aggregated: Per-node aggregated messages [num_nodes, output_dim]
        """
        aggr = self.config.aggregation
        if aggr in ("sum", "add"):
            if HAS_PYG:
                return scatter_add(messages, index, dim=0, dim_size=dim_size)
            else:
                out = torch.zeros(dim_size, messages.size(-1), 
                                  device=messages.device, dtype=messages.dtype)
                return out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
        
        elif self.config.aggregation == "mean":
            if HAS_PYG:
                return scatter_mean(messages, index, dim=0, dim_size=dim_size)
            else:
                out = torch.zeros(dim_size, messages.size(-1),
                                  device=messages.device, dtype=messages.dtype)
                out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
                count = torch.zeros(dim_size, device=messages.device)
                count.scatter_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
                return out / count.clamp(min=1).unsqueeze(-1)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")
    
    def update(
        self,
        aggr_out: Tensor,
        x: Tensor,
        **kwargs
    ) -> Tensor:
        """Update node states with aggregated messages.
        
        Default: identity (return aggregated messages).
        Override for time integration or other update logic.
        
        Args:
            aggr_out: Aggregated messages [num_nodes, output_dim]
            x: Original node features [num_nodes, node_dim]
            **kwargs: Additional inputs (dt, mass, etc.)
        
        Returns:
            updated: Updated node states
        """
        return aggr_out
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass with physics-encoded message passing.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            **kwargs: Additional domain-specific inputs
        
        Returns:
            output: Updated node features or forces [num_nodes, output_dim]
        """
        src, tgt = edge_index[0], edge_index[1]
        x_i, x_j = x[src], x[tgt]
        
        # Compute analytical physics (no learnable params)
        physics_msg = self.compute_edge_physics(
            x_i, x_j, edge_attr, **kwargs
        )
        
        # Compute learned correction
        correction_msg = self.compute_edge_correction(
            x_i, x_j, edge_attr, physics_msg, **kwargs
        )
        
        # Track diagnostics
        if self.config.track_diagnostics and self.training:
            self._update_diagnostics(physics_msg, correction_msg)
        
        # Combine physics and correction
        messages = self.combine_physics_and_correction(physics_msg, correction_msg)
        
        # Aggregate to nodes
        num_nodes = x.size(0)
        aggr_out = self.aggregate(messages, tgt, num_nodes)
        
        # Update nodes
        return self.update(aggr_out, x, **kwargs)
    
    def _update_diagnostics(self, physics_msg: Tensor, correction_msg: Tensor):
        """Update running diagnostics for physics fraction tracking."""
        with torch.no_grad():
            self._physics_norm = physics_msg.norm().item()
            self._correction_norm = correction_msg.norm().item()
            self._total_norm = (physics_msg + correction_msg).norm().item()
            self._num_edges = physics_msg.size(0)
    
    def physics_fraction(self) -> float:
        """Return fraction of output coming from hard-coded physics.
        
        Should be > 0.7 at convergence. If < 0.5, the network is
        overriding physics rather than correcting it.
        
        Returns:
            fraction: ||physics|| / ||total|| averaged over edges
        """
        if self._total_norm < 1e-8:
            return 1.0
        return self._physics_norm / (self._total_norm + 1e-8)
    
    def correction_magnitude(self) -> float:
        """Return L2 norm of correction term.
        
        Should start near zero and grow slowly during training.
        If it explodes, training is unstable.
        
        Returns:
            magnitude: ||correction|| averaged over edges
        """
        if self._num_edges < 1:
            return 0.0
        return self._correction_norm / (self._num_edges ** 0.5)
    
    def physics_violation(self) -> Dict[str, float]:
        """Return domain-specific physics violation metrics.
        
        Override in subclasses to compute conservation errors, etc.
        
        Returns:
            violations: Dict of named violation metrics
        """
        return {}


if HAS_PYG:
    class PhysicsEncodedConv(MessagePassing, PhysicsEncodedConvBase):
        """Physics-encoded convolution using PyTorch Geometric's MessagePassing.
        
        This is the preferred implementation when PyG is available.
        Inherits message-passing machinery from PyG with physics encoding.
        """
        
        def __init__(
            self,
            config: Optional[PhysicsConvConfig] = None,
            aggr: str = "add",
            **kwargs
        ):
            # Map our aggregation config to PyG's aggr parameter
            config = config or PhysicsConvConfig()
            pyg_aggr = "add" if config.aggregation == "sum" else config.aggregation
            
            MessagePassing.__init__(self, aggr=pyg_aggr, **kwargs)
            PhysicsEncodedConvBase.__init__(self, config)
        
        def message(
            self,
            x_i: Tensor,
            x_j: Tensor,
            edge_attr: Optional[Tensor] = None,
            **kwargs
        ) -> Tensor:
            """Compute edge messages with physics + correction."""
            # Compute analytical physics
            physics_msg = self.compute_edge_physics(x_i, x_j, edge_attr, **kwargs)
            
            # Compute learned correction
            correction_msg = self.compute_edge_correction(
                x_i, x_j, edge_attr, physics_msg, **kwargs
            )
            
            # Track diagnostics
            if self.config.track_diagnostics and self.training:
                self._update_diagnostics(physics_msg, correction_msg)
            
            # Combine
            return self.combine_physics_and_correction(physics_msg, correction_msg)
        
        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            **kwargs
        ) -> Tensor:
            """Forward pass using PyG's propagate."""
            # Use PyG's propagate which calls message() and aggregate()
            aggr_out = self.propagate(
                edge_index, 
                x=x, 
                edge_attr=edge_attr,
                **kwargs
            )
            return self.update(aggr_out, x, **kwargs)
        
        @abstractmethod
        def compute_edge_physics(
            self,
            x_i: Tensor,
            x_j: Tensor,
            edge_attr: Optional[Tensor] = None,
            **kwargs
        ) -> Tensor:
            raise NotImplementedError
        
        @abstractmethod
        def physics_name(self) -> str:
            raise NotImplementedError

else:
    # Fallback when PyG is not available
    PhysicsEncodedConv = PhysicsEncodedConvBase


def verify_no_learnable_params(method):
    """Decorator to verify a method introduces no learnable parameters.
    
    Use this on compute_edge_physics to enforce the constraint.
    """
    def wrapper(self, *args, **kwargs):
        params_before = set(id(p) for p in self.parameters())
        result = method(self, *args, **kwargs)
        params_after = set(id(p) for p in self.parameters())
        
        if params_after != params_before:
            raise RuntimeError(
                f"{method.__name__} introduced learnable parameters! "
                "compute_edge_physics must be purely analytical."
            )
        return result
    return wrapper
