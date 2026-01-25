"""
Physics-Informed Loss Functions for Cell-Path PINNs

Implements three key loss components:
1. Data Loss: MSE between predicted and observed positions
2. Geodesic Loss: Enforces constant-speed motion (|v|² = c²)
3. Chemotactic Loss: Velocity follows nutrient gradient (v ≈ ∇U)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .models import PathNet, PotentialNet


def compute_data_loss(
    path_net: PathNet,
    t: torch.Tensor,
    x_true: torch.Tensor,
    y_true: torch.Tensor
) -> torch.Tensor:
    """
    Compute data fitting loss (MSE).
    
    L_data = mean(|r_pred(t) - r_true(t)|²)
    
    Args:
        path_net: Trajectory network
        t: Time tensor (batch_size, 1)
        x_true: True x positions (batch_size, 1)
        y_true: True y positions (batch_size, 1)
        
    Returns:
        Scalar loss tensor
    """
    # Predict positions
    xy_pred = path_net(t)
    x_pred = xy_pred[:, 0:1]
    y_pred = xy_pred[:, 1:2]
    
    # MSE loss
    loss = torch.mean((x_pred - x_true)**2 + (y_pred - y_true)**2)
    return loss


def compute_geodesic_loss(
    path_net: PathNet,
    t: torch.Tensor,
    target_speed: float = 1.0
) -> torch.Tensor:
    """
    Compute geodesic (constant-speed) loss.
    
    L_geo = mean((|v(t)|² - c²)²)
    
    Enforces that the microbe moves at constant speed along its trajectory,
    which is a key property of geodesics.
    
    Args:
        path_net: Trajectory network
        t: Time tensor (batch_size, 1) with requires_grad=True
        target_speed: Target speed c (default: 1.0 for normalized trajectories)
        
    Returns:
        Scalar loss tensor
    """
    # Ensure t requires grad for autograd
    t = t.requires_grad_(True)
    
    # Forward pass
    xy_pred = path_net(t)
    x_pred = xy_pred[:, 0:1]
    y_pred = xy_pred[:, 1:2]
    
    # Compute velocity components using autograd
    v_x = torch.autograd.grad(
        x_pred, t,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    v_y = torch.autograd.grad(
        y_pred, t,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Speed squared
    speed_squared = v_x**2 + v_y**2
    
    # Loss: deviation from constant speed
    target_speed_squared = target_speed ** 2
    loss = torch.mean((speed_squared - target_speed_squared)**2)
    
    return loss


def compute_chemotactic_loss(
    path_net: PathNet,
    potential_net: PotentialNet,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Compute chemotactic loss (velocity follows potential gradient).
    
    L_chem = mean(|v(t) - ∇U(r(t), t)|²)
    
    Enforces that the microbe's velocity aligns with the gradient of the
    nutrient field, modeling chemotactic behavior.
    
    Args:
        path_net: Trajectory network
        potential_net: Potential field network
        t: Time tensor (batch_size, 1)
        
    Returns:
        Scalar loss tensor
    """
    batch_size = t.shape[0]
    
    # Create input tensors that require grad for potential gradient computation
    # These are the spatial coordinates where we evaluate U
    x_input = torch.zeros(batch_size, 1, device=t.device, requires_grad=True)
    y_input = torch.zeros(batch_size, 1, device=t.device, requires_grad=True)
    t_input = t.detach().clone().requires_grad_(True)
    
    # Ensure t requires grad for velocity computation
    t_for_vel = t.requires_grad_(True)
    
    # Get predicted positions from path network
    xy_pred = path_net(t_for_vel)
    x_pred = xy_pred[:, 0:1]
    y_pred = xy_pred[:, 1:2]
    
    # Compute velocity from path network (dr/dt)
    v_x = torch.autograd.grad(
        x_pred, t_for_vel,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    v_y = torch.autograd.grad(
        y_pred, t_for_vel,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Now compute potential at the predicted positions
    # We use x_pred and y_pred values but need fresh tensors for gradient computation
    x_for_U = x_pred.detach().requires_grad_(True)
    y_for_U = y_pred.detach().requires_grad_(True)
    t_for_U = t.detach().requires_grad_(True)
    
    # Compute potential - gradients will flow through potential_net
    U_pred = potential_net(x_for_U, y_for_U, t_for_U)
    
    # Compute gradient of U with respect to spatial coordinates
    U_x = torch.autograd.grad(
        U_pred, x_for_U,
        grad_outputs=torch.ones_like(U_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    U_y = torch.autograd.grad(
        U_pred, y_for_U,
        grad_outputs=torch.ones_like(U_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Chemotactic loss: velocity should follow gradient
    # The loss depends on U_x and U_y, which depend on potential_net parameters
    loss = torch.mean((v_x - U_x)**2 + (v_y - U_y)**2)
    
    return loss


def compute_losses(
    path_net: PathNet,
    potential_net: PotentialNet,
    t: torch.Tensor,
    x_true: torch.Tensor,
    y_true: torch.Tensor,
    lambda_data: float = 1.0,
    lambda_geo: float = 0.1,
    lambda_chem: float = 1.0,
    target_speed: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all physics-informed loss components.
    
    L_total = λ_data * L_data + λ_geo * L_geo + λ_chem * L_chem
    
    Args:
        path_net: Trajectory network
        potential_net: Potential field network
        t: Time tensor (batch_size, 1)
        x_true: True x positions (batch_size, 1)
        y_true: True y positions (batch_size, 1)
        lambda_data: Weight for data loss (default: 1.0)
        lambda_geo: Weight for geodesic loss (default: 0.1)
        lambda_chem: Weight for chemotactic loss (default: 1.0)
        target_speed: Target speed for geodesic constraint (default: 1.0)
        
    Returns:
        Tuple of (total_loss, loss_data, loss_geo, loss_chem)
    """
    # Compute individual losses
    loss_data = compute_data_loss(path_net, t, x_true, y_true)
    loss_geo = compute_geodesic_loss(path_net, t, target_speed)
    loss_chem = compute_chemotactic_loss(path_net, potential_net, t)
    
    # Weighted sum
    total_loss = (
        lambda_data * loss_data +
        lambda_geo * loss_geo +
        lambda_chem * loss_chem
    )
    
    return total_loss, loss_data, loss_geo, loss_chem


class CellPathLoss(nn.Module):
    """
    Module wrapper for physics-informed loss computation.
    
    Useful for integrating with PyTorch training pipelines.
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_geo: float = 0.1,
        lambda_chem: float = 1.0,
        target_speed: float = 1.0
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_geo = lambda_geo
        self.lambda_chem = lambda_chem
        self.target_speed = target_speed
    
    def forward(
        self,
        path_net: PathNet,
        potential_net: PotentialNet,
        t: torch.Tensor,
        x_true: torch.Tensor,
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses."""
        return compute_losses(
            path_net, potential_net, t, x_true, y_true,
            self.lambda_data, self.lambda_geo, self.lambda_chem,
            self.target_speed
        )
