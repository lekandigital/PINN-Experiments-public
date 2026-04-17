"""
ElasticForceConv — Physics-encoded convolution for 3D elastic deformation.

Encodes Hooke's law for elastic solids, supporting material properties
like Young's modulus and Poisson's ratio per edge or globally.

Physics Encoded:
    For each edge (i, j) representing a spring in the mesh:
    F = k * (|x_j - x_i| - L0) * direction
    
    Where:
    - k is derived from Young's modulus E and mesh geometry
    - L0 is the rest length (initial edge length)
    
    For full 3D elasticity, the stiffness k relates to:
    - Young's modulus E (material stiffness)
    - Poisson's ratio ν (volume preservation)
    - Edge cross-sectional area A

What the Network Learns:
    - Nonlinear material response at large deformations
    - Material heterogeneity beyond simple per-edge properties
    - Plastic deformation (permanent shape change)
    - Contact/collision response
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import PhysicsEncodedConv, PhysicsConvConfig, CorrectionMLP


@dataclass
class ElasticConvConfig(PhysicsConvConfig):
    """Configuration for ElasticForceConv.
    
    Extends PhysicsConvConfig with elastic deformation parameters.
    """
    # Material properties (used if not provided per-edge)
    youngs_modulus: float = 1e6      # Young's modulus E (Pa)
    poissons_ratio: float = 0.3      # Poisson's ratio ν
    density: float = 1000.0          # Material density (kg/m³)
    
    # Stiffness computation
    stiffness_mode: str = "per_edge"  # "fixed", "per_edge", "from_material"
    
    # Physics precision
    strain_clamp: float = 5.0        # Clamp strain for stability
    eps: float = 1e-8
    
    # Temporal integration
    use_velocity_gru: bool = False   # Use GRU for velocity refinement
    gru_hidden_dim: int = 64


class VelocityGRU(nn.Module):
    """GRU-based temporal processor for velocity updates.
    
    Maintains per-node hidden state to capture temporal dynamics
    and refine velocity predictions from the physics-based forces.
    """
    
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        output_size: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        velocity_update: Tensor,
        hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Refine velocity update through GRU.
        
        Args:
            velocity_update: Raw velocity update from physics [num_nodes, 3]
            hidden: GRU hidden state [num_nodes, hidden_size] or None
        
        Returns:
            refined_velocity: Refined velocity update [num_nodes, 3]
            new_hidden: Updated hidden state [num_nodes, hidden_size]
        """
        x = self.input_proj(velocity_update)
        
        if hidden is None:
            hidden = torch.zeros(
                velocity_update.size(0), 
                self.hidden_size,
                device=velocity_update.device,
                dtype=velocity_update.dtype
            )
        
        new_hidden = self.gru(x, hidden)
        refined_velocity = self.output_proj(new_hidden)
        
        return refined_velocity, new_hidden
    
    def init_hidden(self, num_nodes: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Initialize hidden state for a batch."""
        return torch.zeros(num_nodes, self.hidden_size, device=device, dtype=dtype)


class ElasticForceConv(PhysicsEncodedConv):
    """Physics-encoded convolution layer for elastic solid deformation.
    
    Implements Hooke's law for 3D elasticity with optional VelocityGRU
    for temporal state tracking (from Project 14's PEGNN-Deform).
    
    Expected inputs:
        - x: Node positions [num_nodes, 3]
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Should contain [rest_length, stiffness] [num_edges, 2+]
        - vel: Node velocities [num_nodes, 3]
        - mass: Node masses [num_nodes, 1] (optional)
        - hidden: GRU hidden state [num_nodes, hidden_dim] (optional)
    
    Output:
        - forces: Per-node force vectors [num_nodes, 3]
        - (optional) hidden: Updated GRU hidden state
    """
    
    def __init__(
        self,
        config: Optional[ElasticConvConfig] = None,
        **kwargs
    ):
        config = config or ElasticConvConfig()
        super().__init__(config=config, **kwargs)
        self.elastic_config = config
        
        # Build correction MLP
        # Input: x_i (3) + x_j (3) + edge_attr (2+) + physics_msg (3)
        correction_input_dim = 3 + 3 + max(2, config.edge_dim) + 3
        self._build_correction_mlp(correction_input_dim)
        
        # Optional VelocityGRU for temporal processing
        if config.use_velocity_gru:
            self.velocity_gru = VelocityGRU(
                input_size=3,
                hidden_size=config.gru_hidden_dim,
                output_size=3
            )
        else:
            self.velocity_gru = None
    
    def physics_name(self) -> str:
        return "Hooke's Law (3D Elasticity)"
    
    def compute_edge_physics(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute elastic spring forces using Hooke's law.
        
        F = k * (|x_j - x_i| - L0) * direction
        
        Args:
            x_i: Source node positions [num_edges, 3]
            x_j: Target node positions [num_edges, 3]
            edge_attr: Edge attributes [num_edges, 2+]
                      Column 0: rest_length (L0)
                      Column 1: stiffness (k)
        
        Returns:
            force: Spring force vectors [num_edges, 3]
        """
        cfg = self.elastic_config
        eps = cfg.eps
        
        # Compute displacement vector
        diff = x_j - x_i  # [num_edges, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [num_edges, 1]
        direction = diff / (dist + eps)
        
        # Get rest length and stiffness from edge_attr
        if edge_attr is not None and edge_attr.size(-1) >= 2:
            rest_length = edge_attr[:, 0:1]  # L0
            stiffness = edge_attr[:, 1:2]    # k
        elif edge_attr is not None and edge_attr.size(-1) >= 1:
            rest_length = edge_attr[:, 0:1]
            stiffness = torch.full_like(rest_length, cfg.youngs_modulus)
        else:
            # Fallback: no stretching force if no edge attributes
            rest_length = dist.detach()
            stiffness = torch.full_like(dist, cfg.youngs_modulus)
        
        # === HOOKE'S LAW ===
        # F = k * (d - L0) * direction
        # Positive when stretched (d > L0), negative when compressed
        displacement = dist - rest_length
        
        # Clamp for numerical stability
        max_displacement = cfg.strain_clamp * rest_length
        displacement = torch.clamp(
            displacement, 
            -max_displacement, 
            max_displacement
        )
        
        force = stiffness * displacement * direction
        
        return force
    
    def compute_edge_correction(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        physics_message: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute learned correction to elastic forces.
        
        Can optionally use GRU hidden states for temporal context.
        
        Args:
            x_i: Source node positions [num_edges, 3]
            x_j: Target node positions [num_edges, 3]
            edge_attr: Edge attributes [num_edges, edge_dim]
            physics_message: Analytical physics term [num_edges, 3]
            hidden: GRU hidden states [num_nodes, hidden_dim] (optional)
            edge_index: Edge connectivity [2, num_edges] (for gathering hidden)
        
        Returns:
            correction: Learned correction [num_edges, 3]
        """
        if self.correction_mlp is None:
            return torch.zeros_like(physics_message)
        
        # Build input features
        features = [x_i, x_j]
        
        if edge_attr is not None:
            features.append(edge_attr)
        else:
            features.append(torch.zeros(x_i.size(0), 2, device=x_i.device))
        
        if physics_message is not None:
            features.append(physics_message)
        
        # Optionally include GRU hidden states
        if hidden is not None and edge_index is not None:
            src, tgt = edge_index[0], edge_index[1]
            hidden_i = hidden[src]
            hidden_j = hidden[tgt]
            features.extend([hidden_i, hidden_j])
        
        mlp_input = torch.cat(features, dim=-1)
        
        # Adjust MLP input size if needed (for hidden state case)
        if mlp_input.size(-1) != self.correction_mlp.net[0].in_features:
            # Truncate or pad to match expected input
            expected_dim = self.correction_mlp.net[0].in_features
            if mlp_input.size(-1) > expected_dim:
                mlp_input = mlp_input[..., :expected_dim]
            else:
                pad = torch.zeros(
                    mlp_input.size(0), 
                    expected_dim - mlp_input.size(-1),
                    device=mlp_input.device,
                    dtype=mlp_input.dtype
                )
                mlp_input = torch.cat([mlp_input, pad], dim=-1)
        
        return self.correction_mlp(mlp_input)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        vel: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
        dt: float = 0.01,
        return_hidden: bool = False,
        **kwargs
    ) -> Tensor:
        """Forward pass computing elastic forces.
        
        Optionally performs time integration and GRU velocity refinement.
        
        Args:
            x: Node positions [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            vel: Node velocities [num_nodes, 3]
            mass: Node masses [num_nodes, 1]
            hidden: GRU hidden state [num_nodes, hidden_dim]
            dt: Time step for integration
            return_hidden: Whether to return updated hidden state
        
        Returns:
            forces: Per-node force vectors [num_nodes, 3]
            (optional) hidden: Updated GRU hidden state
        """
        src, tgt = edge_index[0], edge_index[1]
        x_i, x_j = x[src], x[tgt]
        
        # Compute analytical physics
        physics_msg = self.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Compute learned correction (with optional hidden state)
        correction_msg = self.compute_edge_correction(
            x_i, x_j, edge_attr, physics_msg, 
            hidden=hidden, edge_index=edge_index
        )
        
        # Track diagnostics
        if self.config.track_diagnostics and self.training:
            self._update_diagnostics(physics_msg, correction_msg)
        
        # Combine physics and correction
        messages = self.combine_physics_and_correction(physics_msg, correction_msg)
        
        # Aggregate to nodes
        num_nodes = x.size(0)
        forces = self.aggregate(messages, tgt, num_nodes)
        
        # Optional: Refine through VelocityGRU
        new_hidden = None
        if self.velocity_gru is not None and vel is not None:
            # Compute acceleration from forces
            if mass is not None:
                acc = forces / (mass + 1e-8)
            else:
                acc = forces
            
            # Compute raw velocity update
            vel_update = vel + dt * acc
            
            # Refine through GRU
            refined_vel, new_hidden = self.velocity_gru(vel_update, hidden)
            
            # Return refined velocity as "forces" (actually velocity update)
            if return_hidden:
                return refined_vel, new_hidden
            return refined_vel
        
        if return_hidden:
            return forces, new_hidden
        return forces
    
    def physics_violation(self) -> Dict[str, float]:
        """Compute physics violation metrics for elastic deformation.
        
        Returns:
            violations: Dictionary with conservation errors
        """
        return {
            "momentum_error": 0.0,  # Would need full simulation state
            "energy_rate": 0.0,     # Would need velocity information
        }


class ElasticForceConvWithIntegration(ElasticForceConv):
    """ElasticForceConv with built-in time integration.
    
    Returns updated positions and velocities rather than forces.
    """
    
    def __init__(
        self,
        config: Optional[ElasticConvConfig] = None,
        integrator: str = "explicit_euler",
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        
        from .integrators import IntegratorFactory
        self.integrator = IntegratorFactory.create(integrator)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        vel: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
        dt: float = 0.01,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass with integration.
        
        Args:
            x: Node positions [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            vel: Node velocities [num_nodes, 3]
            mass: Node masses [num_nodes, 1]
            hidden: GRU hidden state [num_nodes, hidden_dim]
            dt: Time step
        
        Returns:
            new_pos: Updated positions [num_nodes, 3]
            new_vel: Updated velocities [num_nodes, 3]
            new_hidden: Updated GRU hidden state (if using GRU)
        """
        # Initialize velocity if not provided
        if vel is None:
            vel = torch.zeros_like(x)
        
        # Get forces from parent class
        result = super().forward(
            x, edge_index, edge_attr, vel, mass, hidden, dt,
            return_hidden=True
        )
        
        if isinstance(result, tuple):
            forces, new_hidden = result
        else:
            forces, new_hidden = result, None
        
        # Compute acceleration
        if mass is not None:
            acc = forces / (mass + 1e-8)
        else:
            acc = forces
        
        # Integrate
        new_pos, new_vel = self.integrator.step(x, vel, acc, dt)
        
        return new_pos, new_vel, new_hidden
