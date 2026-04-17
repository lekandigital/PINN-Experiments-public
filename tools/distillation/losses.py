"""
Loss functions for knowledge distillation.

This module provides:
1. Soft target loss (temperature-scaled MSE/KL divergence)
2. Feature matching loss (intermediate layer alignment)
3. Physics loss interface (domain-specific PDE residuals)
4. Combined distillation loss with configurable weights
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Soft Target Loss (Core Distillation Signal)
# =============================================================================

class SoftTargetLoss(nn.Module):
    """
    Temperature-scaled loss for matching teacher outputs.
    
    For regression tasks (typical in physics models), uses MSE.
    The temperature scaling softens the targets, making them more
    informative for the student.
    
    L = MSE(student/T, teacher/T) * T²
    
    Args:
        temperature: Softening temperature (higher = softer)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, temperature: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self, 
        student_output: torch.Tensor, 
        teacher_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temperature-scaled MSE loss.
        
        Args:
            student_output: Student model predictions
            teacher_output: Teacher model predictions (detached, no grad)
            
        Returns:
            Scaled loss value
        """
        # Scale by temperature
        student_scaled = student_output / self.temperature
        teacher_scaled = teacher_output / self.temperature
        
        # MSE loss
        loss = F.mse_loss(student_scaled, teacher_scaled, reduction=self.reduction)
        
        # Scale back by T² (standard Hinton distillation for regression)
        return loss * (self.temperature ** 2)


class HardTargetLoss(nn.Module):
    """
    Direct loss against ground truth labels (if available).
    
    Combines with soft target loss:
    L_total = α * L_soft + (1-α) * L_hard
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        student_output: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE against ground truth."""
        return F.mse_loss(student_output, ground_truth, reduction=self.reduction)


# =============================================================================
# Feature Matching Loss
# =============================================================================

class FeatureMatchingLoss(nn.Module):
    """
    Loss for matching intermediate layer activations.
    
    Forces the student to learn similar internal representations
    to the teacher, which can improve distillation quality.
    
    Usage:
        loss_fn = FeatureMatchingLoss()
        
        # Register hooks during forward pass
        teacher_features = loss_fn.register_hooks(teacher, ["encoder.layer3"])
        student_features = loss_fn.register_hooks(student, ["encoder.layer1"])
        
        # After forward pass
        loss = loss_fn(student_features, teacher_features)
    """
    
    def __init__(
        self, 
        layer_mapping: dict[str, str] | None = None,
        reduction: str = "mean",
        normalize: bool = True
    ):
        """
        Args:
            layer_mapping: {teacher_layer_name: student_layer_name}
            reduction: 'mean', 'sum', or 'none'
            normalize: Whether to L2-normalize features before comparison
        """
        super().__init__()
        self.layer_mapping = layer_mapping or {}
        self.reduction = reduction
        self.normalize = normalize
        
        # Storage for captured activations
        self._teacher_features: dict[str, torch.Tensor] = {}
        self._student_features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
    
    def _make_hook(self, storage: dict, name: str) -> Callable:
        """Create a forward hook that captures activations."""
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            storage[name] = output.detach()
        return hook
    
    def register_teacher_hooks(self, model: nn.Module, layer_names: list[str]) -> None:
        """Register hooks on teacher model layers."""
        for name in layer_names:
            module = dict(model.named_modules()).get(name)
            if module is not None:
                hook = module.register_forward_hook(
                    self._make_hook(self._teacher_features, name)
                )
                self._hooks.append(hook)
            else:
                logger.warning(f"Teacher layer '{name}' not found")
    
    def register_student_hooks(self, model: nn.Module, layer_names: list[str]) -> None:
        """Register hooks on student model layers."""
        for name in layer_names:
            module = dict(model.named_modules()).get(name)
            if module is not None:
                hook = module.register_forward_hook(
                    self._make_hook(self._student_features, name)
                )
                self._hooks.append(hook)
            else:
                logger.warning(f"Student layer '{name}' not found")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def clear_features(self) -> None:
        """Clear captured features."""
        self._teacher_features.clear()
        self._student_features.clear()
    
    def forward(
        self,
        teacher_layer: str | None = None,
        student_layer: str | None = None,
    ) -> torch.Tensor:
        """
        Compute feature matching loss using captured activations.
        
        If layer names not provided, uses the layer_mapping.
        """
        total_loss = torch.tensor(0.0)
        
        if teacher_layer and student_layer:
            pairs = [(teacher_layer, student_layer)]
        else:
            pairs = list(self.layer_mapping.items())
        
        for t_layer, s_layer in pairs:
            if t_layer not in self._teacher_features:
                logger.warning(f"Teacher features for '{t_layer}' not captured")
                continue
            if s_layer not in self._student_features:
                logger.warning(f"Student features for '{s_layer}' not captured")
                continue
            
            t_feat = self._teacher_features[t_layer]
            s_feat = self._student_features[s_layer]
            
            # Handle shape mismatch by adaptive pooling or projection
            if t_feat.shape != s_feat.shape:
                # Simple case: just flatten and compare norms
                t_feat = t_feat.flatten(1)
                s_feat = s_feat.flatten(1)
                
                if t_feat.shape[1] != s_feat.shape[1]:
                    # Project to smaller dimension
                    min_dim = min(t_feat.shape[1], s_feat.shape[1])
                    t_feat = t_feat[:, :min_dim]
                    s_feat = s_feat[:, :min_dim]
            
            if self.normalize:
                t_feat = F.normalize(t_feat, dim=-1)
                s_feat = F.normalize(s_feat, dim=-1)
            
            loss = F.mse_loss(s_feat, t_feat, reduction=self.reduction)
            total_loss = total_loss + loss
        
        return total_loss


# =============================================================================
# Physics Loss Interface
# =============================================================================

class PhysicsLoss(ABC, nn.Module):
    """
    Abstract base class for domain-specific physics losses.
    
    Each domain (airfoil, cloth, coastal) provides its own implementation
    that computes PDE residuals or other physics-informed constraints.
    
    Example implementations:
    - NavierStokesLoss: Continuity + momentum equations
    - EikonalLoss: |∇SDF| = 1 for signed distance fields
    - ShallowWaterLoss: Continuity + momentum for coastal flows
    - ClothPhysicsLoss: Spring energy + collision constraints
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(
        self,
        model_output: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            model_output: Output from the student model
            inputs: Input tensors (may need for computing gradients)
            model: The student model (may need for computing gradients via autograd)
            
        Returns:
            Physics loss value
        """
        pass


class EikonalLoss(PhysicsLoss):
    """
    Eikonal equation loss for signed distance fields: |∇SDF| = 1
    
    Used for SIREN-based SDF models (Projects 09, 13).
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        model_output: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute eikonal loss.
        
        Args:
            model_output: SDF values [B, 1]
            inputs: Must contain 'coords' [B, 3 or 4] with requires_grad=True
            model: Not used, gradients computed from model_output
        """
        coords = inputs.get('coords') or inputs.get('input')
        if coords is None:
            raise ValueError("EikonalLoss requires 'coords' or 'input' in inputs dict")
        
        # Ensure coords require grad for gradient computation
        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)
        
        # Compute gradient of SDF w.r.t. spatial coordinates
        grad_outputs = torch.ones_like(model_output)
        gradients = torch.autograd.grad(
            outputs=model_output,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Only use spatial dimensions (x, y, z), not time
        spatial_grad = gradients[:, :3] if gradients.shape[1] > 3 else gradients
        
        # |∇SDF| should equal 1
        grad_norm = torch.norm(spatial_grad, dim=-1)
        eikonal_loss = F.mse_loss(grad_norm, torch.ones_like(grad_norm), reduction=self.reduction)
        
        return eikonal_loss


class NavierStokesLoss(PhysicsLoss):
    """
    Navier-Stokes equations loss for incompressible flow.
    
    Continuity: ∇·v = 0
    Momentum: ∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v
    
    Used for airfoil flows (Project 15).
    """
    
    def __init__(
        self,
        nu: float = 0.001,  # Kinematic viscosity
        rho: float = 1.0,   # Density
        reduction: str = "mean",
    ):
        super().__init__()
        self.nu = nu
        self.rho = rho
        self.reduction = reduction
    
    def forward(
        self,
        model_output: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute Navier-Stokes residuals.
        
        Args:
            model_output: [u, v, p] velocities and pressure [B, 3]
            inputs: Must contain 'coords' [B, 2 or 3] (x, y, [aoa])
        """
        coords = inputs.get('coords') or inputs.get('input')
        if coords is None:
            raise ValueError("NavierStokesLoss requires 'coords' in inputs dict")
        
        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)
        
        # Parse outputs
        u = model_output[:, 0:1]  # x-velocity
        v = model_output[:, 1:2]  # y-velocity
        p = model_output[:, 2:3]  # pressure
        
        # Compute first derivatives
        def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(
                y, x, torch.ones_like(y),
                create_graph=True, retain_graph=True
            )[0]
        
        # Velocity gradients
        grads_u = grad(u, coords)
        grads_v = grad(v, coords)
        grads_p = grad(p, coords)
        
        u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
        v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
        p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]
        
        # Second derivatives for viscous term
        grads_u_x = grad(u_x, coords)
        grads_u_y = grad(u_y, coords)
        grads_v_x = grad(v_x, coords)
        grads_v_y = grad(v_y, coords)
        
        u_xx = grads_u_x[:, 0:1]
        u_yy = grads_u_y[:, 1:2]
        v_xx = grads_v_x[:, 0:1]
        v_yy = grads_v_y[:, 1:2]
        
        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y
        
        # X-momentum: u∂u/∂x + v∂u/∂y + (1/ρ)∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²) = 0
        momentum_x = (
            u * u_x + v * u_y + 
            (1.0 / self.rho) * p_x - 
            self.nu * (u_xx + u_yy)
        )
        
        # Y-momentum: u∂v/∂x + v∂v/∂y + (1/ρ)∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²) = 0
        momentum_y = (
            u * v_x + v * v_y +
            (1.0 / self.rho) * p_y -
            self.nu * (v_xx + v_yy)
        )
        
        # Combine losses
        loss_continuity = F.mse_loss(continuity, torch.zeros_like(continuity), reduction=self.reduction)
        loss_momentum_x = F.mse_loss(momentum_x, torch.zeros_like(momentum_x), reduction=self.reduction)
        loss_momentum_y = F.mse_loss(momentum_y, torch.zeros_like(momentum_y), reduction=self.reduction)
        
        return loss_continuity + loss_momentum_x + loss_momentum_y


class EdgePreservationLoss(PhysicsLoss):
    """
    Cloth edge preservation loss: penalize stretching/compression.
    
    Used for cloth simulation models (Projects 05, 09, 13).
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        model_output: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute edge length preservation loss.
        
        Args:
            model_output: Per-vertex displacements [B, N, 3]
            inputs: Must contain 'positions' [B, N, 3] and 'edges' [B, E, 2]
        """
        positions = inputs.get('positions')
        edges = inputs.get('edges')
        
        if positions is None or edges is None:
            # Return zero if not applicable
            return torch.tensor(0.0, device=model_output.device)
        
        # New positions after applying displacement
        new_positions = positions + model_output
        
        # Compute original and new edge lengths
        src_idx = edges[:, :, 0].long()
        dst_idx = edges[:, :, 1].long()
        
        # Gather vertices for each edge
        batch_size = positions.shape[0]
        batch_idx = torch.arange(batch_size, device=positions.device).view(-1, 1).expand_as(src_idx)
        
        orig_src = positions[batch_idx, src_idx]
        orig_dst = positions[batch_idx, dst_idx]
        new_src = new_positions[batch_idx, src_idx]
        new_dst = new_positions[batch_idx, dst_idx]
        
        orig_lengths = torch.norm(orig_dst - orig_src, dim=-1)
        new_lengths = torch.norm(new_dst - new_src, dim=-1)
        
        # Penalize length changes
        length_change = (new_lengths - orig_lengths) / (orig_lengths + 1e-8)
        loss = F.mse_loss(length_change, torch.zeros_like(length_change), reduction=self.reduction)
        
        return loss


# =============================================================================
# Combined Distillation Loss
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation with configurable components.
    
    Supports:
    - Soft target loss (teacher output matching)
    - Hard target loss (ground truth matching, optional)
    - Feature matching loss (intermediate layer matching, optional)
    - Physics loss (domain-specific PDE constraints, optional)
    - Custom losses (any additional terms)
    
    Usage:
        loss_fn = DistillationLoss(
            weights={
                'soft_target': 1.0,
                'physics': 0.1,
                'feature_match': 0.3,
            },
            temperature=2.0,
            physics_loss=EikonalLoss(),
        )
        
        total_loss, loss_dict = loss_fn(
            teacher_output=teacher_out,
            student_output=student_out,
            inputs=inputs,
            student_model=student,
        )
    """
    
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        temperature: float = 2.0,
        physics_loss: PhysicsLoss | None = None,
        feature_matching: FeatureMatchingLoss | None = None,
    ):
        super().__init__()
        
        self.weights = weights or {'soft_target': 1.0}
        self.soft_target_loss = SoftTargetLoss(temperature=temperature)
        self.hard_target_loss = HardTargetLoss()
        self.physics_loss = physics_loss
        self.feature_matching = feature_matching
    
    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        inputs: dict[str, torch.Tensor] | None = None,
        ground_truth: torch.Tensor | None = None,
        student_model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined distillation loss.
        
        Args:
            teacher_output: Teacher predictions (detached)
            student_output: Student predictions
            inputs: Input tensors (needed for physics loss)
            ground_truth: Optional ground truth labels
            student_model: Student model (needed for physics loss gradients)
            
        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Individual loss values for logging
        """
        losses: dict[str, torch.Tensor] = {}
        
        # Soft target loss (core distillation signal)
        if 'soft_target' in self.weights:
            losses['soft_target'] = self.soft_target_loss(student_output, teacher_output)
        
        # Hard target loss (optional ground truth)
        if 'hard_target' in self.weights and ground_truth is not None:
            losses['hard_target'] = self.hard_target_loss(student_output, ground_truth)
        
        # Physics loss (domain-specific)
        if 'physics' in self.weights and self.physics_loss is not None and inputs is not None:
            losses['physics'] = self.physics_loss(student_output, inputs, student_model)
        
        # Eikonal loss (special case for SDFs)
        if 'eikonal' in self.weights and inputs is not None:
            eikonal = EikonalLoss()
            losses['eikonal'] = eikonal(student_output, inputs, student_model)
        
        # Feature matching loss
        if 'feature_match' in self.weights and self.feature_matching is not None:
            losses['feature_match'] = self.feature_matching()
        
        # Surface consistency (extra weight near zero-crossing for SDFs)
        if 'surface_consistency' in self.weights:
            # Extra penalty near the surface (|SDF| < threshold)
            surface_mask = (torch.abs(teacher_output) < 0.1).float()
            surface_loss = F.mse_loss(
                student_output * surface_mask, 
                teacher_output * surface_mask
            )
            losses['surface_consistency'] = surface_loss
        
        # Edge preservation (cloth)
        if 'edge_preservation' in self.weights and inputs is not None:
            edge_loss = EdgePreservationLoss()
            losses['edge_preservation'] = edge_loss(student_output, inputs, student_model)
        
        # Compute weighted total
        total_loss = torch.tensor(0.0, device=student_output.device)
        for name, loss_value in losses.items():
            if name in self.weights:
                total_loss = total_loss + self.weights[name] * loss_value
        
        return total_loss, {k: v.detach() for k, v in losses.items()}


# =============================================================================
# Loss Factory
# =============================================================================

def create_physics_loss(
    loss_type: str,
    **kwargs
) -> PhysicsLoss | None:
    """
    Factory function to create physics loss by name.
    
    Args:
        loss_type: "eikonal", "navier_stokes", "edge_preservation", or "none"
        **kwargs: Additional arguments for the loss
        
    Returns:
        PhysicsLoss instance or None
    """
    loss_map = {
        'eikonal': EikonalLoss,
        'navier_stokes': NavierStokesLoss,
        'edge_preservation': EdgePreservationLoss,
    }
    
    if loss_type == 'none' or loss_type is None:
        return None
    
    if loss_type not in loss_map:
        logger.warning(f"Unknown physics loss type: {loss_type}")
        return None
    
    return loss_map[loss_type](**kwargs)
