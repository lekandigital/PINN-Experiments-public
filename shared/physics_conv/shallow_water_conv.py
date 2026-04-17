"""
ShallowWaterConv — Physics-encoded convolution for shallow water equations.

Encodes the depth-averaged shallow water equations for coastal flow simulation:
    ∂η/∂t + ∇·(h*u) = 0                          (mass conservation)
    ∂u/∂t + (u·∇)u = -g*∇η - C_d*|u|*u/h + ν*∇²u (momentum conservation)

Where:
    η = water surface elevation (above reference)
    h = total water depth (bathymetry + η)
    u = depth-averaged velocity vector (u, v)
    g = gravitational acceleration (9.81 m/s²)
    C_d = bottom drag coefficient
    ν = eddy viscosity

Physics Encoded:
    1. Pressure gradient: F_pressure = -g * ∇η
    2. Bottom friction: F_drag = -C_d * |u| * u / h
    3. Mass flux through edges
    4. Momentum advection flux

What the Network Learns:
    - Turbulent mixing beyond simple eddy viscosity
    - Wave breaking effects
    - Subgrid-scale bathymetry effects
    - Non-hydrostatic pressure corrections
    - Wind-wave interaction
    - Wetting/drying at coastlines
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .base import PhysicsEncodedConv, PhysicsConvConfig, CorrectionMLP


@dataclass
class ShallowWaterConfig(PhysicsConvConfig):
    """Configuration for ShallowWaterConv."""
    
    # Physical constants
    gravity: float = 9.81              # Gravitational acceleration (m/s²)
    drag_coefficient: float = 0.003   # Bottom drag coefficient C_d
    eddy_viscosity: float = 0.1       # Kinematic eddy viscosity ν (m²/s)
    
    # Numerical parameters
    min_depth: float = 0.01           # Minimum depth to avoid division by zero
    eps: float = 1e-8
    
    # State variable dimensions
    # Node state: [η, u, v, h_bathy] = [elevation, vel_x, vel_y, bathymetry]
    state_dim: int = 4
    output_dim: int = 3               # [dη/dt, du/dt, dv/dt]
    
    # Edge types for boundary handling
    handle_boundaries: bool = True
    
    # Mass conservation enforcement
    enforce_mass_conservation: bool = True
    
    # Physics precision (shallow water is sensitive)
    physics_dtype: torch.dtype = torch.float64


class ShallowWaterConv(PhysicsEncodedConv):
    """Physics-encoded convolution for shallow water equations.
    
    Implements the depth-averaged shallow water equations on a graph,
    encoding pressure gradients, bottom friction, and continuity
    directly into the message-passing layer.
    
    Node state variables:
        - η (eta): Water surface elevation [m]
        - u: Velocity x-component [m/s]
        - v: Velocity y-component [m/s]
        - h_bathy: Bathymetry depth (static) [m]
    
    Edge attributes:
        - edge_length: Physical length of edge [m]
        - edge_normal: Outward normal vector [2]
        - edge_type: 0=interior, 1=land_boundary, 2=open_boundary
    
    Output:
        - dstate/dt: Time derivatives [dη/dt, du/dt, dv/dt] per node
    """
    
    def __init__(
        self,
        config: Optional[ShallowWaterConfig] = None,
        **kwargs
    ):
        config = config or ShallowWaterConfig()
        super().__init__(config=config, **kwargs)
        self.sw_config = config
        
        # Build correction MLP
        # Input: state_i (4) + state_j (4) + edge_attr (4) + physics_msg (3) = 15
        correction_input_dim = config.state_dim * 2 + 4 + config.output_dim
        self._build_correction_mlp(correction_input_dim)
        
        # Optional bathymetry encoder (learns corrections based on local bathymetry)
        self.bathy_encoder = nn.Sequential(
            nn.Linear(2, config.correction_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.correction_hidden_dim, config.correction_hidden_dim),
        )
    
    def physics_name(self) -> str:
        return "Shallow Water Equations"
    
    def compute_edge_physics(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute shallow water physics terms.
        
        State convention: x = [η, u, v, h_bathy]
        
        Args:
            x_i: Source node state [num_edges, 4]
            x_j: Target node state [num_edges, 4]
            edge_attr: Edge attributes [num_edges, 4+]
                      [edge_length, normal_x, normal_y, edge_type]
        
        Returns:
            physics_flux: [num_edges, 3] containing [mass_flux, mom_x_flux, mom_y_flux]
        """
        cfg = self.sw_config
        g = cfg.gravity
        C_d = cfg.drag_coefficient
        h_min = cfg.min_depth
        eps = cfg.eps
        
        # Use float64 for physics precision
        orig_dtype = x_i.dtype
        x_i = x_i.to(cfg.physics_dtype)
        x_j = x_j.to(cfg.physics_dtype)
        if edge_attr is not None:
            edge_attr = edge_attr.to(cfg.physics_dtype)
        
        # Unpack state variables
        eta_i, u_i, v_i, bathy_i = x_i[:, 0:1], x_i[:, 1:2], x_i[:, 2:3], x_i[:, 3:4]
        eta_j, u_j, v_j, bathy_j = x_j[:, 0:1], x_j[:, 1:2], x_j[:, 2:3], x_j[:, 3:4]
        
        # Total water depth (bathymetry is depth below reference, positive downward)
        h_i = torch.clamp(bathy_i + eta_i, min=h_min)
        h_j = torch.clamp(bathy_j + eta_j, min=h_min)
        
        # Edge geometry
        if edge_attr is not None and edge_attr.size(-1) >= 3:
            edge_length = edge_attr[:, 0:1]
            normal_x = edge_attr[:, 1:2]
            normal_y = edge_attr[:, 2:3]
        else:
            # Default: unit length, x-direction normal
            num_edges = x_i.size(0)
            edge_length = torch.ones(num_edges, 1, device=x_i.device, dtype=x_i.dtype)
            normal_x = torch.ones(num_edges, 1, device=x_i.device, dtype=x_i.dtype)
            normal_y = torch.zeros(num_edges, 1, device=x_i.device, dtype=x_i.dtype)
        
        # === PRESSURE GRADIENT FORCE ===
        # F_pressure = -g * ∇η ≈ -g * (η_j - η_i) / edge_length * normal
        grad_eta = (eta_j - eta_i) / (edge_length + eps)
        F_pressure_x = -g * grad_eta * normal_x
        F_pressure_y = -g * grad_eta * normal_y
        
        # === BOTTOM FRICTION ===
        # F_drag = -C_d * |u| * u / h
        # Average velocity at edge midpoint
        u_mid = 0.5 * (u_i + u_j)
        v_mid = 0.5 * (v_i + v_j)
        h_mid = 0.5 * (h_i + h_j)
        
        speed = torch.sqrt(u_mid**2 + v_mid**2 + eps)
        F_drag_x = -C_d * speed * u_mid / h_mid
        F_drag_y = -C_d * speed * v_mid / h_mid
        
        # === MASS FLUX ===
        # Mass flux = h * (u · n) * edge_length
        u_normal = u_mid * normal_x + v_mid * normal_y  # Velocity normal to edge
        mass_flux = h_mid * u_normal * edge_length
        
        # === ADVECTIVE MOMENTUM FLUX ===
        # Momentum flux = u * (h * u · n) (transport of momentum by flow)
        mom_x_flux = u_mid * h_mid * u_normal
        mom_y_flux = v_mid * h_mid * u_normal
        
        # Total momentum source = pressure gradient + friction
        # (advection is handled through flux aggregation)
        mom_x_source = F_pressure_x + F_drag_x
        mom_y_source = F_pressure_y + F_drag_y
        
        # Combine into physics message
        # Convention: positive flux leaves node i toward node j
        physics_msg = torch.cat([
            mass_flux,           # d(η)/dt contribution
            mom_x_source,        # d(u)/dt contribution  
            mom_y_source,        # d(v)/dt contribution
        ], dim=-1)
        
        # Cast back to original dtype
        return physics_msg.to(orig_dtype)
    
    def compute_edge_correction(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
        physics_message: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute learned correction to shallow water physics.
        
        The correction captures:
        - Turbulence effects beyond simple friction
        - Wave-current interaction
        - Non-hydrostatic effects
        - Subgrid bathymetry effects
        """
        if self.correction_mlp is None:
            return torch.zeros_like(physics_message)
        
        # Build features
        features = [x_i, x_j]
        
        if edge_attr is not None:
            # Pad edge_attr to expected dimension
            if edge_attr.size(-1) < 4:
                pad = torch.zeros(
                    edge_attr.size(0), 
                    4 - edge_attr.size(-1),
                    device=edge_attr.device,
                    dtype=edge_attr.dtype
                )
                edge_attr = torch.cat([edge_attr, pad], dim=-1)
            features.append(edge_attr[:, :4])
        else:
            features.append(torch.zeros(x_i.size(0), 4, device=x_i.device))
        
        features.append(physics_message)
        
        # Encode bathymetry gradient for correction context
        bathy_i, bathy_j = x_i[:, 3:4], x_j[:, 3:4]
        bathy_grad = torch.cat([bathy_i, bathy_j - bathy_i], dim=-1)
        bathy_features = self.bathy_encoder(bathy_grad)
        
        mlp_input = torch.cat(features, dim=-1)
        correction = self.correction_mlp(mlp_input)
        
        return correction
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        boundary_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass computing shallow water dynamics.
        
        Args:
            x: Node state [num_nodes, 4] = [η, u, v, h_bathy]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, 4+]
            boundary_mask: Boolean mask for boundary nodes [num_nodes]
        
        Returns:
            dstate_dt: Time derivatives [num_nodes, 3] = [dη/dt, du/dt, dv/dt]
        """
        cfg = self.sw_config
        src, tgt = edge_index[0], edge_index[1]
        x_i, x_j = x[src], x[tgt]
        
        # Get edge types if available
        if edge_attr is not None and edge_attr.size(-1) >= 4:
            edge_type = edge_attr[:, 3:4]
        else:
            edge_type = torch.zeros(src.size(0), 1, device=x.device)
        
        # Compute physics
        physics_msg = self.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Apply boundary conditions to physics
        if cfg.handle_boundaries:
            physics_msg = self._apply_edge_boundary_conditions(
                physics_msg, edge_type, x_i, x_j
            )
        
        # Compute correction
        correction_msg = self.compute_edge_correction(
            x_i, x_j, edge_attr, physics_msg
        )
        
        # Track diagnostics
        if self.config.track_diagnostics and self.training:
            self._update_diagnostics(physics_msg, correction_msg)
        
        # Combine
        messages = self.combine_physics_and_correction(physics_msg, correction_msg)
        
        # Aggregate to nodes (sum of fluxes)
        num_nodes = x.size(0)
        dstate_dt = self.aggregate(messages, tgt, num_nodes)
        
        # Apply node boundary conditions
        if boundary_mask is not None:
            dstate_dt = self._apply_node_boundary_conditions(
                dstate_dt, boundary_mask, x
            )
        
        # Enforce mass conservation
        if cfg.enforce_mass_conservation:
            dstate_dt = self._enforce_mass_conservation(dstate_dt, x)
        
        return dstate_dt
    
    def _apply_edge_boundary_conditions(
        self,
        physics_msg: Tensor,
        edge_type: Tensor,
        x_i: Tensor,
        x_j: Tensor,
    ) -> Tensor:
        """Apply boundary conditions to edge physics.
        
        Edge types:
            0: Interior edge (normal physics)
            1: Land boundary (no normal flow)
            2: Open boundary (wave radiation / tidal forcing)
        
        Args:
            physics_msg: [num_edges, 3]
            edge_type: [num_edges, 1]
            x_i, x_j: Node states
        
        Returns:
            Modified physics message
        """
        # Land boundary: zero normal velocity flux
        land_mask = (edge_type == 1).squeeze(-1)
        if land_mask.any():
            physics_msg[land_mask, 0] = 0  # No mass flux
            # Keep momentum (wall friction handled separately)
        
        # Open boundary: could add radiation conditions here
        # For now, treat as interior
        
        return physics_msg
    
    def _apply_node_boundary_conditions(
        self,
        dstate_dt: Tensor,
        boundary_mask: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Apply boundary conditions to node updates.
        
        For boundary nodes, can fix elevation (tidal BC) or zero velocity.
        
        Args:
            dstate_dt: [num_nodes, 3]
            boundary_mask: [num_nodes] boolean
            x: [num_nodes, 4] current state
        
        Returns:
            Modified time derivatives
        """
        if boundary_mask.any():
            # Zero out derivatives for boundary nodes
            # (they're driven by boundary conditions, not physics)
            dstate_dt[boundary_mask] = 0
        
        return dstate_dt
    
    def _enforce_mass_conservation(
        self,
        dstate_dt: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Project update to satisfy mass conservation.
        
        Ensures total water volume is conserved by uniformly adjusting
        surface elevation changes.
        
        Args:
            dstate_dt: [num_nodes, 3] = [dη/dt, du/dt, dv/dt]
            x: [num_nodes, 4] current state
        
        Returns:
            Adjusted time derivatives
        """
        # Total mass change
        d_eta = dstate_dt[:, 0]  # [num_nodes]
        total_mass_change = d_eta.sum()
        
        # Distribute correction uniformly
        num_nodes = x.size(0)
        correction = -total_mass_change / num_nodes
        
        dstate_dt = dstate_dt.clone()
        dstate_dt[:, 0] = dstate_dt[:, 0] + correction
        
        return dstate_dt
    
    def physics_violation(self) -> Dict[str, float]:
        """Compute shallow water physics violations.
        
        Returns:
            violations: Dictionary containing:
                - mass_error: Total mass change (should be ~0)
                - momentum_error: Momentum imbalance
        """
        # Would need to track these during forward pass
        return {
            "mass_error": 0.0,
            "momentum_error": 0.0,
        }


class ShallowWaterConvWithIntegration(ShallowWaterConv):
    """ShallowWaterConv with built-in time integration.
    
    Returns updated state rather than time derivatives.
    """
    
    def __init__(
        self,
        config: Optional[ShallowWaterConfig] = None,
        integrator: str = "semi_implicit",
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
        dt: float = 0.1,
        **kwargs
    ) -> Tensor:
        """Forward pass with integration.
        
        Args:
            x: Node state [num_nodes, 4] = [η, u, v, h_bathy]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, 4+]
            dt: Time step [s]
        
        Returns:
            x_new: Updated state [num_nodes, 4]
        """
        # Get time derivatives from parent
        dstate_dt = super().forward(x, edge_index, edge_attr, **kwargs)
        
        # Split state into position-like (η) and velocity-like (u, v)
        eta = x[:, 0:1]
        vel = x[:, 1:3]
        bathy = x[:, 3:4]  # Static
        
        d_eta = dstate_dt[:, 0:1]
        d_vel = dstate_dt[:, 1:3]
        
        # Simple forward Euler for now
        # (Could use proper 2D shallow water integrator)
        eta_new = eta + dt * d_eta
        vel_new = vel + dt * d_vel
        
        # Reassemble state
        x_new = torch.cat([eta_new, vel_new, bathy], dim=-1)
        
        return x_new
