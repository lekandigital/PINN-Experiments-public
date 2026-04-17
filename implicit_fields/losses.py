"""
Physics-informed loss functions for implicit neural representations.

This module provides loss functions commonly used with neural implicit fields:
- Eikonal loss for signed distance fields
- Collision loss for SDF interpenetration
- Gradient utilities for autograd computations
- Divergence-free loss for incompressible fields

Example:
    >>> from implicit_fields import eikonal_loss, gradient
    >>> coords = torch.randn(1000, 3, requires_grad=True)
    >>> sdf = model(coords)
    >>> grads = gradient(sdf, coords)
    >>> loss = eikonal_loss(grads)
"""

from typing import Optional

import torch
import torch.nn.functional as F


def gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
    allow_unused: bool = False,
) -> torch.Tensor:
    """Compute gradients of outputs with respect to inputs via autograd.
    
    This utility handles batched gradient computation correctly and is used
    by other loss functions (eikonal, divergence) that need spatial gradients.
    
    Args:
        outputs: Output tensor from the network, shape (batch_size, *).
                 For SDF networks, typically (batch_size, 1).
        inputs: Input tensor that requires_grad=True, shape (batch_size, in_dim).
                For SDF, typically (batch_size, 3) for xyz coordinates.
        create_graph: If True, gradients can be differentiated further.
                     Required for second-order optimization. Default: True.
        retain_graph: If True, the computation graph is retained. Default: True.
        allow_unused: If True, don't error if inputs aren't used. Default: False.
    
    Returns:
        Gradient tensor of shape (batch_size, in_dim).
    
    Example:
        >>> coords = torch.randn(100, 3, requires_grad=True)
        >>> sdf = model(coords)  # (100, 1)
        >>> grads = gradient(sdf, coords)  # (100, 3)
        >>> print(grads.shape)
        torch.Size([100, 3])
    
    Note:
        The input tensor MUST have requires_grad=True for this to work.
        If computing eikonal loss, create_graph must be True.
    """
    # Sum outputs to get scalar for grad computation
    # This computes sum of gradients, which equals gradient when summed over batch
    grad_outputs = torch.ones_like(outputs)
    
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )[0]
    
    return grads


def eikonal_loss(
    gradients: torch.Tensor,
    reduction: str = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """Eikonal loss enforcing unit gradient norm for signed distance fields.
    
    For a valid SDF, the gradient magnitude should be 1 everywhere:
        |∇f(x)| = 1
    
    This loss penalizes deviations from unit norm:
        L_eikonal = (|∇f| - 1)²
    
    Args:
        gradients: Spatial gradients of the SDF, shape (batch_size, spatial_dim).
                   Typically (batch_size, 3) for 3D SDFs.
        reduction: How to reduce the loss over the batch.
                   'mean' (default), 'sum', or 'none'.
        eps: Small constant for numerical stability in norm computation.
             Prevents NaN gradients when gradient is near zero. Default: 1e-8.
    
    Returns:
        Eikonal loss. Scalar if reduction='mean' or 'sum', else (batch_size,).
    
    Example:
        >>> coords = torch.randn(1000, 3, requires_grad=True)
        >>> sdf = model(coords)
        >>> grads = gradient(sdf, coords)
        >>> loss = eikonal_loss(grads)
        >>> loss.backward()
    
    Note:
        The gradients must be computed with create_graph=True to allow
        backpropagation through the eikonal loss.
    """
    # Compute gradient magnitude with numerical stability
    # Using torch.linalg.norm is cleaner, but we add eps for safety
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=-1) + eps)
    
    # Eikonal loss: deviation from unit norm
    loss = (grad_norm - 1.0) ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")


def sdf_collision_loss(
    sdf_a: torch.Tensor,
    sdf_b: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Collision loss penalizing interpenetration between two SDF fields.
    
    Collision occurs where both SDFs are negative (inside both surfaces).
    The loss is the product of penetration depths:
        L_collision = max(0, -sdf_a) * max(0, -sdf_b)
    
    This is useful for cloth-body collision in cloth simulation, or
    object-object collision in multi-object scenes.
    
    Args:
        sdf_a: SDF values for first object, shape (batch_size, 1) or (batch_size,).
        sdf_b: SDF values for second object, shape (batch_size, 1) or (batch_size,).
        reduction: How to reduce over batch. 'mean' (default), 'sum', or 'none'.
    
    Returns:
        Collision loss. Zero when objects don't interpenetrate.
    
    Example:
        >>> # Cloth and body SDFs at same query points
        >>> cloth_sdf = cloth_model(coords)  # (N, 1)
        >>> body_sdf = body_model(coords)    # (N, 1)
        >>> collision = sdf_collision_loss(cloth_sdf, body_sdf)
    """
    # Ensure tensors are the same shape
    sdf_a = sdf_a.squeeze(-1) if sdf_a.dim() > 1 else sdf_a
    sdf_b = sdf_b.squeeze(-1) if sdf_b.dim() > 1 else sdf_b
    
    # Penetration depth: positive only when inside (SDF < 0)
    penetration_a = F.relu(-sdf_a)
    penetration_b = F.relu(-sdf_b)
    
    # Loss is product of penetration depths
    loss = penetration_a * penetration_b
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def divergence_free_loss(
    field: torch.Tensor,
    coords: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Divergence-free loss enforcing ∇·F = 0 for incompressible fields.
    
    For incompressible fluid flow or magnetic fields, the divergence should
    be zero everywhere:
        ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z = 0
    
    Args:
        field: Vector field values, shape (batch_size, field_dim).
               For 3D flow: (batch_size, 3) representing (vx, vy, vz).
        coords: Spatial coordinates with requires_grad=True,
                shape (batch_size, spatial_dim).
        reduction: How to reduce. 'mean' (default), 'sum', or 'none'.
    
    Returns:
        Divergence loss (squared divergence).
    
    Example:
        >>> coords = torch.randn(1000, 3, requires_grad=True)
        >>> velocity = flow_model(coords)  # (1000, 3)
        >>> div_loss = divergence_free_loss(velocity, coords)
    """
    field_dim = field.shape[-1]
    coord_dim = coords.shape[-1]
    
    if field_dim != coord_dim:
        raise ValueError(
            f"Field dimension ({field_dim}) must match coordinate dimension ({coord_dim}) "
            "for divergence computation."
        )
    
    # Compute divergence: sum of diagonal elements of Jacobian
    divergence = torch.zeros(field.shape[0], device=field.device, dtype=field.dtype)
    
    for i in range(field_dim):
        # Gradient of i-th field component with respect to all coordinates
        grad_i = gradient(field[:, i:i+1], coords, create_graph=True)
        # Add diagonal element (∂Fi/∂xi)
        divergence = divergence + grad_i[:, i]
    
    # Loss is squared divergence
    loss = divergence ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def sdf_boundary_loss(
    sdf_values: torch.Tensor,
    target_values: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Loss for SDF values at known surface points.
    
    At points known to be on the surface, SDF should be zero.
    At points with known signed distance, SDF should match.
    
    Args:
        sdf_values: Predicted SDF values, shape (batch_size, 1) or (batch_size,).
        target_values: Ground truth SDF values. For surface points, this is 0.
        reduction: Reduction method.
    
    Returns:
        MSE loss between predicted and target SDF values.
    """
    sdf_values = sdf_values.squeeze(-1) if sdf_values.dim() > 1 else sdf_values
    target_values = target_values.squeeze(-1) if target_values.dim() > 1 else target_values
    
    loss = (sdf_values - target_values) ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def laplacian_loss(
    sdf_values: torch.Tensor,
    coords: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Laplacian regularization loss.
    
    Penalizes the Laplacian (sum of second derivatives) of the SDF,
    encouraging smooth surfaces. The Laplacian of an SDF relates to
    mean curvature.
    
    Args:
        sdf_values: Predicted SDF, shape (batch_size, 1).
        coords: Input coordinates with requires_grad=True.
        reduction: Reduction method.
    
    Returns:
        Mean squared Laplacian.
    """
    # First derivatives
    grads = gradient(sdf_values, coords, create_graph=True)
    
    # Second derivatives (diagonal of Hessian)
    laplacian = torch.zeros(sdf_values.shape[0], device=sdf_values.device)
    
    for i in range(coords.shape[-1]):
        grad_i = grads[:, i:i+1]
        grad2_i = gradient(grad_i, coords, create_graph=True)
        laplacian = laplacian + grad2_i[:, i]
    
    loss = laplacian ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class CombinedSDFLoss(torch.nn.Module):
    """Combined loss for SDF training with configurable weights.
    
    Combines:
    - Data loss (SDF values at known points)
    - Eikonal loss (unit gradient constraint)
    - Optional: Laplacian smoothness
    - Optional: Collision loss
    
    Args:
        w_data: Weight for data fitting loss. Default: 1.0.
        w_eikonal: Weight for eikonal regularization. Default: 0.1.
        w_laplacian: Weight for Laplacian smoothness. Default: 0.0.
        w_collision: Weight for collision loss. Default: 0.0.
    
    Example:
        >>> loss_fn = CombinedSDFLoss(w_data=1.0, w_eikonal=0.1)
        >>> coords = torch.randn(1000, 3, requires_grad=True)
        >>> sdf_pred = model(coords)
        >>> sdf_target = torch.zeros(1000)  # Surface points
        >>> total_loss, loss_dict = loss_fn(sdf_pred, sdf_target, coords)
    """
    
    def __init__(
        self,
        w_data: float = 1.0,
        w_eikonal: float = 0.1,
        w_laplacian: float = 0.0,
        w_collision: float = 0.0,
    ):
        super().__init__()
        self.w_data = w_data
        self.w_eikonal = w_eikonal
        self.w_laplacian = w_laplacian
        self.w_collision = w_collision
    
    def forward(
        self,
        sdf_pred: torch.Tensor,
        sdf_target: torch.Tensor,
        coords: torch.Tensor,
        sdf_obstacle: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Compute combined loss.
        
        Args:
            sdf_pred: Predicted SDF values.
            sdf_target: Ground truth SDF values.
            coords: Query coordinates (requires_grad=True for eikonal).
            sdf_obstacle: Optional SDF of obstacle for collision loss.
        
        Returns:
            Tuple of (total_loss, dict of individual losses).
        """
        losses = {}
        total = 0.0
        
        # Data loss
        if self.w_data > 0:
            losses['data'] = sdf_boundary_loss(sdf_pred, sdf_target)
            total = total + self.w_data * losses['data']
        
        # Eikonal loss
        if self.w_eikonal > 0:
            grads = gradient(sdf_pred, coords)
            losses['eikonal'] = eikonal_loss(grads)
            total = total + self.w_eikonal * losses['eikonal']
        
        # Laplacian loss
        if self.w_laplacian > 0:
            losses['laplacian'] = laplacian_loss(sdf_pred, coords)
            total = total + self.w_laplacian * losses['laplacian']
        
        # Collision loss
        if self.w_collision > 0 and sdf_obstacle is not None:
            losses['collision'] = sdf_collision_loss(sdf_pred, sdf_obstacle)
            total = total + self.w_collision * losses['collision']
        
        losses['total'] = total
        return total, losses
