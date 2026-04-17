"""
LiteClothConv — Lightweight physics-encoded convolution for small models.

A stripped-down version of ClothForceConv designed for models with
tight parameter budgets (< 80K total parameters).

Simplifications:
    1. No bending forces (stretch only)
    2. Scalar stiffness (no per-edge variation)
    3. Minimal correction MLP (single layer, 32 units) or none
    4. No damping computation (let GRU handle temporal smoothing)

Physics Encoded:
    F = k * (|x_j - x_i| - L0) / L0 * direction
    
    This is pure Hooke's law with no corrections - the model's other
    components (GRU, decoder) handle residual dynamics.

Target: Add < 5K parameters while providing strong physics inductive bias.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import PhysicsConvConfig
from .cloth_conv import ClothForceConv, ClothConvConfig


@dataclass
class LiteClothConvConfig(ClothConvConfig):
    """Configuration for LiteClothConv.
    
    Defaults are set for minimal parameter count.
    """
    # Override defaults for lightweight operation
    correction_hidden_dim: int = 32      # Smaller hidden dim
    correction_layers: int = 1           # Single layer
    stretch_stiffness: float = 1000.0    # Global stiffness
    compute_bending: bool = False        # No bending
    compute_damping: bool = False        # No damping
    stiffness_mode: str = "fixed"        # Fixed stiffness (no MLP)
    
    # Option to disable correction MLP entirely
    use_correction: bool = True          # Set False for pure physics


class LiteClothConv(ClothForceConv):
    """Lightweight cloth force convolution for small models.
    
    Inherits from ClothForceConv but uses minimal correction network
    and disables bending/damping computations.
    
    Parameter count (approximate):
        - With correction (default): ~3K parameters
        - Without correction: 0 parameters (pure physics)
    
    Expected inputs:
        - x: Node positions [num_nodes, 3]
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Rest lengths [num_edges, 1]
    
    Output:
        - forces: Per-node spring forces [num_nodes, 3]
    """
    
    def __init__(
        self,
        config: Optional[LiteClothConvConfig] = None,
        **kwargs
    ):
        # Use lightweight defaults
        config = config or LiteClothConvConfig()
        
        # Initialize parent (which builds correction MLP)
        super().__init__(config=config, **kwargs)
        
        self.lite_config = config
        
        # Replace correction MLP with minimal version or None
        if not config.use_correction:
            self.correction_mlp = None
        elif config.correction_layers == 1:
            # Single-layer correction: just a linear projection
            # Input: x_i (3) + x_j (3) + rest_length (1) + physics (3) = 10
            self.correction_mlp = nn.Sequential(
                nn.Linear(10, config.correction_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.correction_hidden_dim, 3),
            )
            # Initialize near zero
            nn.init.zeros_(self.correction_mlp[-1].bias)
            nn.init.normal_(
                self.correction_mlp[-1].weight, 
                mean=0.0, 
                std=config.correction_scale_init
            )
    
    def physics_name(self) -> str:
        return "Hooke's Law (Lightweight)"
    
    def compute_edge_physics(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute simple spring forces.
        
        Pure Hooke's law without damping or bending.
        
        Args:
            x_i: Source node positions [num_edges, 3]
            x_j: Target node positions [num_edges, 3]
            edge_attr: Rest lengths [num_edges, 1]
        
        Returns:
            force: Spring force vectors [num_edges, 3]
        """
        cfg = self.lite_config
        eps = cfg.eps
        
        # Edge vector and length
        edge_vec = x_j - x_i
        current_length = torch.norm(edge_vec, dim=-1, keepdim=True)
        direction = edge_vec / (current_length + eps)
        
        # Rest length
        if edge_attr is not None and edge_attr.size(-1) >= 1:
            rest_length = edge_attr[:, 0:1]
        else:
            rest_length = current_length.detach()
        
        # Strain (normalized displacement)
        strain = (current_length - rest_length) / (rest_length + eps)
        strain = torch.clamp(strain, -cfg.strain_clamp, cfg.strain_clamp)
        
        # Hooke's law with fixed stiffness
        force = cfg.stretch_stiffness * strain * direction
        
        return force
    
    def compute_edge_correction(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        physics_message: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute minimal correction.
        
        Uses single-layer MLP or returns zeros if disabled.
        """
        if self.correction_mlp is None:
            return torch.zeros_like(physics_message)
        
        # Minimal feature set
        if edge_attr is not None:
            rest_len = edge_attr[:, 0:1]
        else:
            rest_len = torch.zeros(x_i.size(0), 1, device=x_i.device)
        
        mlp_input = torch.cat([x_i, x_j, rest_len, physics_message], dim=-1)
        
        return self.correction_mlp(mlp_input)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass computing lightweight cloth forces.
        
        Simplified version without velocity/damping handling.
        
        Args:
            x: Node positions [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Rest lengths [num_edges, 1]
        
        Returns:
            forces: Per-node force vectors [num_nodes, 3]
        """
        src, tgt = edge_index[0], edge_index[1]
        x_i, x_j = x[src], x[tgt]
        
        # Compute physics
        physics_msg = self.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Compute correction
        correction_msg = self.compute_edge_correction(
            x_i, x_j, edge_attr, physics_msg
        )
        
        # Track diagnostics
        if self.config.track_diagnostics and self.training:
            self._update_diagnostics(physics_msg, correction_msg)
        
        # Combine
        messages = self.combine_physics_and_correction(physics_msg, correction_msg)
        
        # Aggregate
        num_nodes = x.size(0)
        forces = self.aggregate(messages, tgt, num_nodes)
        
        return forces
    
    def count_parameters(self) -> int:
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PurePhysicsClothConv(LiteClothConv):
    """Pure physics cloth convolution with NO learnable parameters.
    
    This is the minimal version - just Hooke's law encoded in the
    message passing, with no learned corrections at all.
    
    Use when you want the strongest physics inductive bias and
    have other model components handling corrections.
    """
    
    def __init__(
        self,
        stiffness: float = 1000.0,
        strain_clamp: float = 2.0,
        **kwargs
    ):
        config = LiteClothConvConfig(
            use_correction=False,
            stretch_stiffness=stiffness,
            strain_clamp=strain_clamp,
        )
        super().__init__(config=config, **kwargs)
        
        # Remove any learnable parameters
        self.correction_mlp = None
    
    def physics_name(self) -> str:
        return "Hooke's Law (Pure Physics, No Learning)"
    
    def compute_edge_correction(self, *args, **kwargs) -> Tensor:
        """No correction - returns zeros."""
        physics_msg = kwargs.get('physics_message', args[3] if len(args) > 3 else None)
        if physics_msg is None:
            # Shouldn't happen, but handle gracefully
            x_i = args[0] if args else kwargs.get('x_i')
            return torch.zeros(x_i.size(0), 3, device=x_i.device, dtype=x_i.dtype)
        return torch.zeros_like(physics_msg)
