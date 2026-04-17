"""
Geodesic Loss Functions for Trajectory Prediction.

Physics-informed losses that enforce geodesic properties on predicted trajectories:

1. Constant Speed: |v(t)|² = c²  (geodesic parameterization)
2. Boundary Conditions: x(0) = start, x(1) = end
3. Gradient Following: v ≈ -∇φ (agents follow potential gradient)
4. Curvature Penalty: Limits path curvature for kinematic constraints
5. Geodesic Equation: d²x/dt² + Γⁱⱼₖ(dx^j/dt)(dx^k/dt) = 0

These losses are domain-agnostic and can be combined as needed for
different applications (robotics, biology, finance, etc.).
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

from .potential_field import PotentialFieldBase


class ConstantSpeedLoss(nn.Module):
    """
    Enforces constant-speed motion along trajectory.
    
    L_speed = mean((|v(t)|² - c²)²)
    
    This is a fundamental property of geodesics when parameterized by arc length.
    Constant speed means the agent moves efficiently without speeding up or
    slowing down unnecessarily.
    """
    
    def __init__(self, target_speed: float = 1.0):
        """
        Initialize constant speed loss.
        
        Args:
            target_speed: Target speed c (default 1.0 for normalized trajectories)
        """
        super().__init__()
        self.target_speed = target_speed
        self.target_speed_squared = target_speed ** 2
    
    def forward(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute constant speed loss.
        
        Args:
            positions: Trajectory positions [batch, time, dim] or [time, dim]
            times: Time points [batch, time] or [time]
            velocities: Pre-computed velocities (optional, computed if not provided)
            
        Returns:
            Scalar loss tensor
        """
        # Handle 2D vs 3D input
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            times = times.unsqueeze(0)
        
        if velocities is None:
            # Compute velocities via finite differences
            dt = times[:, 1:] - times[:, :-1]  # [batch, time-1]
            dx = positions[:, 1:] - positions[:, :-1]  # [batch, time-1, dim]
            velocities = dx / dt.unsqueeze(-1)  # [batch, time-1, dim]
        
        # Speed squared
        speed_squared = (velocities ** 2).sum(dim=-1)  # [batch, time-1]
        
        # Loss: deviation from constant speed
        loss = ((speed_squared - self.target_speed_squared) ** 2).mean()
        
        return loss


class BoundaryLoss(nn.Module):
    """
    Enforces start and end point constraints.
    
    L_boundary = |x(0) - x_start|² + |x(1) - x_end|²
    
    Essential for path planning: trajectory must connect specified endpoints.
    """
    
    def __init__(self, weight_start: float = 1.0, weight_end: float = 1.0):
        """
        Initialize boundary loss.
        
        Args:
            weight_start: Weight for start point constraint
            weight_end: Weight for end point constraint
        """
        super().__init__()
        self.weight_start = weight_start
        self.weight_end = weight_end
    
    def forward(
        self,
        positions: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            positions: Trajectory [batch, time, dim] or [time, dim]
            start: Start position [batch, dim] or [dim]
            end: End position [batch, dim] or [dim]
            
        Returns:
            Scalar loss tensor
        """
        # Handle dimensions
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if start.dim() == 1:
            start = start.unsqueeze(0)
        if end.dim() == 1:
            end = end.unsqueeze(0)
        
        # Start and end constraints
        loss_start = ((positions[:, 0] - start) ** 2).sum(dim=-1).mean()
        loss_end = ((positions[:, -1] - end) ** 2).sum(dim=-1).mean()
        
        return self.weight_start * loss_start + self.weight_end * loss_end


class GradientFollowingLoss(nn.Module):
    """
    Enforces that velocity follows potential gradient.
    
    L_grad = mean(|v(t) - α∇φ(x(t))|²)
    
    Where α = -1 for gradient descent (moving to low potential)
    or α = +1 for gradient ascent (moving to high potential).
    
    This is the generalization of chemotactic loss - agents move in the
    direction indicated by the potential field.
    """
    
    def __init__(
        self,
        potential_field: PotentialFieldBase,
        direction: str = "descent",  # "descent" or "ascent"
        alignment_weight: float = 1.0,
    ):
        """
        Initialize gradient following loss.
        
        Args:
            potential_field: The potential field to follow
            direction: "descent" (toward low φ) or "ascent" (toward high φ)
            alignment_weight: Loss weight
        """
        super().__init__()
        self.potential_field = potential_field
        self.alpha = -1.0 if direction == "descent" else 1.0
        self.alignment_weight = alignment_weight
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gradient following loss.
        
        Args:
            positions: Trajectory positions [batch, time, dim] or [time, dim]
            velocities: Velocities [batch, time-1, dim] or [time-1, dim]
            times: Optional time points for time-dependent potentials
            
        Returns:
            Scalar loss tensor
        """
        # Handle dimensions
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if velocities.dim() == 2:
            velocities = velocities.unsqueeze(0)
        
        batch_size, n_points, dim = positions.shape
        
        # Use positions at velocity evaluation points (midpoints or first n-1)
        pos_flat = positions[:, :-1].reshape(-1, dim)  # [batch*(time-1), dim]
        
        # Handle time
        if times is not None and self.potential_field.time_dependent:
            if times.dim() == 1:
                times = times.unsqueeze(0)
            t_flat = times[:, :-1].reshape(-1, 1)
        else:
            t_flat = None
        
        # Compute gradient at positions
        grad = self.potential_field.gradient(pos_flat, t_flat)  # [batch*(time-1), dim]
        grad = grad.reshape(batch_size, -1, dim)  # [batch, time-1, dim]
        
        # Target velocity is α * gradient (descent or ascent)
        target_velocity = self.alpha * grad
        
        # MSE between actual and target velocity
        loss = ((velocities - target_velocity) ** 2).sum(dim=-1).mean()
        
        return self.alignment_weight * loss


class CurvaturePenalty(nn.Module):
    """
    Penalizes high curvature in trajectories.
    
    L_curvature = mean(max(0, κ(t) - κ_max)²)
    
    Useful for:
    - Robot path planning (turning radius constraints)
    - Smooth trajectory generation
    - Kinematic feasibility
    """
    
    def __init__(self, max_curvature: float = 1.0, penalty_weight: float = 1.0):
        """
        Initialize curvature penalty.
        
        Args:
            max_curvature: Maximum allowed curvature κ_max
            penalty_weight: Loss weight
        """
        super().__init__()
        self.max_curvature = max_curvature
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute curvature penalty.
        
        Curvature κ = |v × a| / |v|³ (for 2D: |v_x*a_y - v_y*a_x| / |v|³)
        
        Args:
            positions: Trajectory [batch, time, dim] or [time, dim]
            velocities: Optional pre-computed velocities
            accelerations: Optional pre-computed accelerations
            
        Returns:
            Scalar loss tensor
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        # Compute derivatives if not provided
        if velocities is None:
            velocities = positions[:, 1:] - positions[:, :-1]
        if accelerations is None:
            accelerations = velocities[:, 1:] - velocities[:, :-1]
            velocities = velocities[:, :-1]  # Align with accelerations
        
        # Speed
        speed = torch.norm(velocities, dim=-1, keepdim=True) + 1e-8  # [batch, time, 1]
        
        # Curvature computation
        dim = positions.shape[-1]
        if dim == 2:
            # 2D: κ = |v_x*a_y - v_y*a_x| / |v|³
            cross = velocities[..., 0] * accelerations[..., 1] - velocities[..., 1] * accelerations[..., 0]
            curvature = torch.abs(cross) / (speed.squeeze(-1) ** 3)
        else:
            # 3D: κ = |v × a| / |v|³
            cross = torch.cross(velocities, accelerations, dim=-1)
            curvature = torch.norm(cross, dim=-1) / (speed.squeeze(-1) ** 3)
        
        # Penalize curvature exceeding maximum
        excess_curvature = torch.relu(curvature - self.max_curvature)
        loss = (excess_curvature ** 2).mean()
        
        return self.penalty_weight * loss


class GeodesicLoss(nn.Module):
    """
    Combined physics-informed loss for geodesic trajectory prediction.
    
    Combines multiple loss components:
    - Constant speed (geodesic parameterization)
    - Boundary conditions (start/end points)
    - Gradient following (potential-directed motion)
    - Curvature penalty (optional, for kinematic constraints)
    - Data fitting (optional, for supervised learning)
    
    Total loss:
        L = λ_speed * L_speed + λ_boundary * L_boundary + 
            λ_gradient * L_gradient + λ_curvature * L_curvature + λ_data * L_data
    """
    
    def __init__(
        self,
        potential_field: Optional[PotentialFieldBase] = None,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_gradient: float = 1.0,
        lambda_curvature: float = 0.0,
        lambda_data: float = 1.0,
        target_speed: float = 1.0,
        max_curvature: float = 1.0,
        gradient_direction: str = "descent",
    ):
        """
        Initialize geodesic loss.
        
        Args:
            potential_field: Potential field for gradient-following loss
            lambda_speed: Weight for constant speed loss
            lambda_boundary: Weight for boundary conditions
            lambda_gradient: Weight for gradient following
            lambda_curvature: Weight for curvature penalty (0 to disable)
            lambda_data: Weight for data fitting loss
            target_speed: Target speed for constant speed loss
            max_curvature: Maximum curvature for penalty
            gradient_direction: "descent" or "ascent"
        """
        super().__init__()
        
        # Store weights
        self.lambda_speed = lambda_speed
        self.lambda_boundary = lambda_boundary
        self.lambda_gradient = lambda_gradient
        self.lambda_curvature = lambda_curvature
        self.lambda_data = lambda_data
        
        # Loss components
        self.speed_loss = ConstantSpeedLoss(target_speed)
        self.boundary_loss = BoundaryLoss()
        
        if potential_field is not None:
            self.gradient_loss = GradientFollowingLoss(
                potential_field, 
                direction=gradient_direction
            )
        else:
            self.gradient_loss = None
        
        if lambda_curvature > 0:
            self.curvature_loss = CurvaturePenalty(max_curvature)
        else:
            self.curvature_loss = None
    
    def forward(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        positions_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined geodesic loss.
        
        Args:
            positions: Predicted trajectory [batch, time, dim] or [time, dim]
            times: Time points [batch, time] or [time]
            start: Start position [batch, dim] or [dim]
            end: End position [batch, dim] or [dim]
            positions_true: Optional ground truth for data loss
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual components
        """
        losses = {}
        
        # Handle dimensions
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if times.dim() == 1:
            times = times.unsqueeze(0)
        
        # Compute velocities once
        dt = times[:, 1:] - times[:, :-1]
        dx = positions[:, 1:] - positions[:, :-1]
        velocities = dx / (dt.unsqueeze(-1) + 1e-8)
        
        # Constant speed loss
        if self.lambda_speed > 0:
            losses['speed'] = self.speed_loss(positions, times, velocities)
        
        # Boundary loss
        if self.lambda_boundary > 0:
            losses['boundary'] = self.boundary_loss(positions, start, end)
        
        # Gradient following loss
        if self.lambda_gradient > 0 and self.gradient_loss is not None:
            losses['gradient'] = self.gradient_loss(positions, velocities, times)
        
        # Curvature penalty
        if self.lambda_curvature > 0 and self.curvature_loss is not None:
            losses['curvature'] = self.curvature_loss(positions, velocities)
        
        # Data fitting loss
        if self.lambda_data > 0 and positions_true is not None:
            if positions_true.dim() == 2:
                positions_true = positions_true.unsqueeze(0)
            losses['data'] = ((positions - positions_true) ** 2).mean()
        
        # Weighted sum
        total = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        weights = {
            'speed': self.lambda_speed,
            'boundary': self.lambda_boundary,
            'gradient': self.lambda_gradient,
            'curvature': self.lambda_curvature,
            'data': self.lambda_data,
        }
        
        for name, loss in losses.items():
            total = total + weights[name] * loss
        
        losses['total'] = total
        
        return total, losses
