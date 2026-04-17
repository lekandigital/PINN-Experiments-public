"""
ClothForceConv — Physics-encoded convolution for cloth simulation.

Encodes Hooke's law (spring forces) directly into the message-passing layer.
The network learns only corrections for nonlinear behavior, anisotropy, etc.

Physics Encoded:
    1. Stretch forces: F = k * (|x_j - x_i| - L0) * direction
       Where L0 is the rest length and k is stiffness
    
    2. Bending forces (optional): Penalize dihedral angle deviation
       between adjacent triangles sharing an edge
    
    3. Damping forces (optional): F_damp = -d * (v_j - v_i) · edge_dir * edge_dir
       Damps relative motion along the edge

What the Network Learns:
    - Nonlinear material response (real cloth isn't perfectly Hookean)
    - Anisotropic effects (warp vs weft in woven fabric)
    - Self-collision response
    - Other coupling effects
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import PhysicsEncodedConv, PhysicsConvConfig, CorrectionMLP


@dataclass
class ClothConvConfig(PhysicsConvConfig):
    """Configuration for ClothForceConv.
    
    Extends PhysicsConvConfig with cloth-specific parameters.
    """
    # Stiffness parameters
    stretch_stiffness: float = 1000.0  # Spring constant for stretch
    shear_stiffness: float = 100.0     # Spring constant for shear edges
    bend_stiffness: float = 0.1        # Bending stiffness
    
    # Damping
    damping_coefficient: float = 0.1   # Velocity damping
    
    # Feature flags
    compute_bending: bool = False      # Whether to compute bending forces
    compute_damping: bool = True       # Whether to compute damping forces
    
    # Physics precision
    strain_clamp: float = 2.0          # Clamp strain to [-clamp, clamp] for stability
    eps: float = 1e-8                  # Numerical stability epsilon
    
    # Stiffness source: "fixed", "per_edge", or "learned"
    stiffness_mode: str = "fixed"


class ClothForceConv(PhysicsEncodedConv):
    """Physics-encoded convolution layer for cloth simulation.
    
    Implements Hooke's law as the primary message function with optional
    bending and damping forces. The correction MLP learns residuals.
    
    Expected inputs:
        - x: Node positions [num_nodes, 3]
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Should contain rest_length at minimum [num_edges, 1+]
                    Optional: stiffness [num_edges, 2] if stiffness_mode="per_edge"
        - vel: Node velocities [num_nodes, 3] (optional, for damping)
    
    Output:
        - forces: Per-node force vectors [num_nodes, 3]
    """
    
    def __init__(
        self,
        config: Optional[ClothConvConfig] = None,
        **kwargs
    ):
        config = config or ClothConvConfig()
        super().__init__(config=config, **kwargs)
        self.cloth_config = config
        
        # Build correction MLP
        # Input: x_i (3) + x_j (3) + edge_attr (1+) + physics_msg (3)
        correction_input_dim = 3 + 3 + max(1, config.edge_dim) + 3
        self._build_correction_mlp(correction_input_dim)
        
        # Optional learnable stiffness
        if config.stiffness_mode == "learned":
            self.stiffness_mlp = nn.Sequential(
                nn.Linear(1, config.correction_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.correction_hidden_dim, 1),
                nn.Softplus(),  # Ensure positive stiffness
            )
        else:
            self.stiffness_mlp = None
    
    def physics_name(self) -> str:
        return "Hooke's Law (Cloth Springs)"
    
    def compute_edge_physics(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        vel_i: Optional[Tensor] = None,
        vel_j: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute spring forces using Hooke's law.
        
        F = k * strain * direction
        strain = (current_length - rest_length) / rest_length
        
        Args:
            x_i: Source node positions [num_edges, 3]
            x_j: Target node positions [num_edges, 3]
            edge_attr: Edge attributes with rest_length [num_edges, 1+]
            vel_i: Source node velocities [num_edges, 3] (optional)
            vel_j: Target node velocities [num_edges, 3] (optional)
        
        Returns:
            force: Spring force vectors [num_edges, 3]
        """
        cfg = self.cloth_config
        eps = cfg.eps
        
        # Compute edge vector and length
        edge_vec = x_j - x_i  # [num_edges, 3]
        current_length = torch.norm(edge_vec, dim=-1, keepdim=True)  # [num_edges, 1]
        direction = edge_vec / (current_length + eps)  # Unit vector
        
        # Get rest length from edge_attr
        if edge_attr is not None and edge_attr.size(-1) >= 1:
            rest_length = edge_attr[:, 0:1]  # [num_edges, 1]
        else:
            # If no rest length provided, use current length (no force)
            rest_length = current_length.detach()
        
        # Compute strain (relative extension)
        strain = (current_length - rest_length) / (rest_length + eps)
        
        # Clamp strain for numerical stability
        strain = torch.clamp(strain, -cfg.strain_clamp, cfg.strain_clamp)
        
        # Get stiffness
        stiffness = self._get_stiffness(strain, edge_attr)
        
        # === HOOKE'S LAW ===
        # Force on node i from spring to node j
        # Positive strain (stretched) -> force pulls i toward j
        # Negative strain (compressed) -> force pushes i away from j
        stretch_force = stiffness * strain * direction
        
        total_force = stretch_force
        
        # === DAMPING FORCE (optional) ===
        if cfg.compute_damping and vel_i is not None and vel_j is not None:
            damping_force = self._compute_damping(
                vel_i, vel_j, direction, cfg.damping_coefficient
            )
            total_force = total_force + damping_force
        
        return total_force
    
    def _get_stiffness(
        self,
        strain: Tensor,
        edge_attr: Optional[Tensor]
    ) -> Tensor:
        """Get stiffness coefficient based on configuration.
        
        Args:
            strain: Current strain values [num_edges, 1]
            edge_attr: Edge attributes [num_edges, edge_dim]
        
        Returns:
            stiffness: Stiffness values [num_edges, 1]
        """
        cfg = self.cloth_config
        
        if cfg.stiffness_mode == "fixed":
            # Use fixed global stiffness
            return torch.full_like(strain, cfg.stretch_stiffness)
        
        elif cfg.stiffness_mode == "per_edge":
            # Get stiffness from edge_attr[:, 1]
            if edge_attr is not None and edge_attr.size(-1) >= 2:
                return edge_attr[:, 1:2]
            else:
                return torch.full_like(strain, cfg.stretch_stiffness)
        
        elif cfg.stiffness_mode == "learned":
            # Predict stiffness from strain using MLP
            # Note: This adds learnable params, but they're for stiffness prediction,
            # not for the force law itself (Hooke's law is still hard-coded)
            return self.stiffness_mlp(strain.abs())
        
        else:
            return torch.full_like(strain, cfg.stretch_stiffness)
    
    def _compute_damping(
        self,
        vel_i: Tensor,
        vel_j: Tensor,
        direction: Tensor,
        damping_coeff: float
    ) -> Tensor:
        """Compute damping force along edge direction.
        
        Damps relative velocity projected onto edge direction:
        F_damp = -d * (v_rel · dir) * dir
        
        Args:
            vel_i: Source node velocities [num_edges, 3]
            vel_j: Target node velocities [num_edges, 3]
            direction: Edge unit vectors [num_edges, 3]
            damping_coeff: Damping coefficient
        
        Returns:
            damping_force: Damping force vectors [num_edges, 3]
        """
        # Relative velocity
        rel_vel = vel_j - vel_i  # [num_edges, 3]
        
        # Project onto edge direction
        rel_vel_along_edge = (rel_vel * direction).sum(dim=-1, keepdim=True)
        
        # Damping force opposes relative motion
        damping_force = -damping_coeff * rel_vel_along_edge * direction
        
        return damping_force
    
    def compute_edge_correction(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        physics_message: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute learned correction to spring forces.
        
        The correction MLP takes:
        - Node positions (captures geometry)
        - Edge attributes (rest length, material properties)
        - Physics message (allows correction to depend on force magnitude)
        
        And outputs a correction vector added to (or modulating) the physics force.
        """
        if self.correction_mlp is None:
            return torch.zeros_like(physics_message)
        
        # Build input features
        features = [x_i, x_j]
        
        if edge_attr is not None:
            features.append(edge_attr)
        else:
            # Pad with zeros if no edge_attr
            features.append(torch.zeros(x_i.size(0), 1, device=x_i.device))
        
        if physics_message is not None:
            features.append(physics_message)
        
        mlp_input = torch.cat(features, dim=-1)
        return self.correction_mlp(mlp_input)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        vel: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass computing cloth forces.
        
        Args:
            x: Node positions [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
                      Must contain rest_length at index 0
            vel: Node velocities [num_nodes, 3] (optional, for damping)
            **kwargs: Additional arguments
        
        Returns:
            forces: Per-node force vectors [num_nodes, 3]
        """
        src, tgt = edge_index[0], edge_index[1]
        x_i, x_j = x[src], x[tgt]
        
        # Get velocities if provided
        vel_i = vel[src] if vel is not None else None
        vel_j = vel[tgt] if vel is not None else None
        
        # Compute analytical physics
        physics_msg = self.compute_edge_physics(
            x_i, x_j, edge_attr, vel_i, vel_j, **kwargs
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
        forces = self.aggregate(messages, tgt, num_nodes)
        
        return forces
    
    def physics_violation(self) -> Dict[str, float]:
        """Compute physics violation metrics for cloth.
        
        Returns:
            violations: Dictionary with:
                - energy_rate: Rate of energy change (should be ~0 or negative)
                - momentum_error: Total momentum (should be conserved)
        """
        # These would need to be computed during forward pass with full state
        # For now, return empty dict - subclasses can implement
        return {}


class ClothForceConvWithBending(ClothForceConv):
    """ClothForceConv extended with bending forces.
    
    Bending forces penalize deviation of dihedral angles between
    adjacent triangles from their rest configuration.
    
    Requires face connectivity information in addition to edge connectivity.
    """
    
    def __init__(
        self,
        config: Optional[ClothConvConfig] = None,
        **kwargs
    ):
        if config is None:
            config = ClothConvConfig(compute_bending=True)
        super().__init__(config=config, **kwargs)
    
    def compute_bending_forces(
        self,
        x: Tensor,
        face_pairs: Tensor,
        rest_angles: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute bending forces from dihedral angles.
        
        Args:
            x: Node positions [num_nodes, 3]
            face_pairs: Adjacent face pairs sharing an edge [num_pairs, 4]
                       Each row: [v0, v1, v2, v3] where v0-v1 is shared edge,
                       v2 is third vertex of first face, v3 of second face
            rest_angles: Rest dihedral angles [num_pairs] (optional)
        
        Returns:
            bending_forces: Per-node bending forces [num_nodes, 3]
        """
        if face_pairs.size(0) == 0:
            return torch.zeros_like(x)
        
        cfg = self.cloth_config
        eps = cfg.eps
        
        # Get vertices for each face pair
        v0 = x[face_pairs[:, 0]]  # Shared edge vertex 1
        v1 = x[face_pairs[:, 1]]  # Shared edge vertex 2
        v2 = x[face_pairs[:, 2]]  # Third vertex of face 1
        v3 = x[face_pairs[:, 3]]  # Third vertex of face 2
        
        # Compute face normals
        e01 = v1 - v0  # Shared edge
        e02 = v2 - v0  # Edge to third vertex of face 1
        e03 = v3 - v0  # Edge to third vertex of face 2
        
        n1 = torch.cross(e01, e02, dim=-1)  # Normal of face 1
        n2 = torch.cross(e03, e01, dim=-1)  # Normal of face 2 (note order for consistent orientation)
        
        n1_norm = torch.norm(n1, dim=-1, keepdim=True)
        n2_norm = torch.norm(n2, dim=-1, keepdim=True)
        
        n1 = n1 / (n1_norm + eps)
        n2 = n2 / (n2_norm + eps)
        
        # Compute dihedral angle
        cos_angle = (n1 * n2).sum(dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        
        # Use rest angles if provided, otherwise assume flat (angle = pi, cos = -1)
        if rest_angles is not None:
            rest_cos = torch.cos(rest_angles).unsqueeze(-1)
        else:
            rest_cos = torch.ones_like(cos_angle) * (-1.0)  # Flat rest state
        
        # Bending energy derivative (simplified - actual derivative is more complex)
        # F_bend ∝ -k_bend * (cos_theta - cos_theta_rest) * gradient
        angle_deviation = cos_angle - rest_cos
        
        # Distribute forces to the 4 vertices
        # This is a simplified version - full derivation involves gradients of dihedral angle
        bending_forces = torch.zeros_like(x)
        
        # Force magnitude
        force_mag = -cfg.bend_stiffness * angle_deviation
        
        # Apply forces perpendicular to faces (along normals)
        # v2 and v3 get pushed/pulled to restore rest angle
        bending_forces.scatter_add_(
            0, 
            face_pairs[:, 2:3].expand(-1, 3), 
            force_mag * n1
        )
        bending_forces.scatter_add_(
            0,
            face_pairs[:, 3:4].expand(-1, 3),
            force_mag * n2
        )
        
        return bending_forces
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        vel: Optional[Tensor] = None,
        face_pairs: Optional[Tensor] = None,
        rest_angles: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass with stretch and bending forces.
        
        Args:
            x: Node positions [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            vel: Node velocities [num_nodes, 3]
            face_pairs: Adjacent face pairs [num_pairs, 4] for bending
            rest_angles: Rest dihedral angles [num_pairs]
        
        Returns:
            forces: Total forces (stretch + bending) [num_nodes, 3]
        """
        # Get stretch forces from parent class
        stretch_forces = super().forward(x, edge_index, edge_attr, vel, **kwargs)
        
        # Add bending forces if face_pairs provided
        if face_pairs is not None and self.cloth_config.compute_bending:
            bending_forces = self.compute_bending_forces(x, face_pairs, rest_angles)
            return stretch_forces + bending_forces
        
        return stretch_forces
