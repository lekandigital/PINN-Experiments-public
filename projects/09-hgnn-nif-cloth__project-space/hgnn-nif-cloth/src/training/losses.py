"""
Physics-Based Loss Functions for Cloth Simulation

Implements multi-task losses combining:
1. Edge Spring Loss - enforces graph-based dynamics consistency
2. SDF Reconstruction Loss - ensures surface accuracy
3. Eikonal Loss - regularizes SDF gradients

Reference equations from the framework design document.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def edge_spring_loss(
    pred_pos: torch.Tensor,
    edge_index: torch.Tensor,
    rest_lengths: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute edge spring loss to enforce elastic constraints.
    
    Penalizes deviation from rest lengths:
    L_spring = (1/|E|) * Σ (||x_i - x_j|| - l_ij^0)²
    
    Args:
        pred_pos: Predicted node positions (B, N, 3) or (N, 3)
        edge_index: Edge indices (2, E) with [source, target]
        rest_lengths: Rest lengths per edge (E,). If None, computed from current positions.
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Spring loss scalar or tensor
    """
    batched = pred_pos.dim() == 3
    if not batched:
        pred_pos = pred_pos.unsqueeze(0)
        
    B, N, _ = pred_pos.shape
    src, tgt = edge_index[0], edge_index[1]
    
    # Compute current edge lengths for each sample in batch
    losses = []
    for b in range(B):
        pos = pred_pos[b]  # (N, 3)
        edge_vecs = pos[src] - pos[tgt]  # (E, 3)
        curr_lengths = torch.norm(edge_vecs, dim=-1)  # (E,)
        
        if rest_lengths is None:
            # Use current as rest (for initial setup)
            rest = curr_lengths.detach()
        else:
            rest = rest_lengths
            
        # Spring energy: squared deviation from rest length
        stretch = curr_lengths - rest
        loss = stretch ** 2
        
        if reduction == 'mean':
            losses.append(loss.mean())
        elif reduction == 'sum':
            losses.append(loss.sum())
        else:
            losses.append(loss)
            
    result = torch.stack(losses) if reduction != 'none' else losses
    
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    return result


def sdf_reconstruction_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    weight_near_surface: float = 10.0,
    surface_threshold: float = 0.02,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute SDF reconstruction loss with surface-aware weighting.
    
    L_SDF = (1/N) * Σ w_k * (φ_pred(p_k) - φ_gt(p_k))²
    
    Points near the surface (|SDF| < threshold) are weighted more heavily.
    
    Args:
        pred_sdf: Predicted SDF values (B, M) or (M,)
        gt_sdf: Ground truth SDF values (B, M) or (M,)
        weight_near_surface: Weight multiplier for near-surface points
        surface_threshold: Distance threshold to consider "near surface"
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        SDF loss scalar or tensor
    """
    # Squared error
    sq_error = (pred_sdf - gt_sdf) ** 2
    
    # Weight points near surface more heavily
    near_surface = torch.abs(gt_sdf) < surface_threshold
    weights = torch.ones_like(gt_sdf)
    weights[near_surface] = weight_near_surface
    
    weighted_error = weights * sq_error
    
    if reduction == 'mean':
        return weighted_error.mean()
    elif reduction == 'sum':
        return weighted_error.sum()
    return weighted_error


def eikonal_loss(
    sdf_gradients: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Eikonal regularization loss for valid SDFs.
    
    SDFs should satisfy ||∇φ|| = 1 almost everywhere.
    L_eikonal = (||∇φ|| - 1)²
    
    Args:
        sdf_gradients: Gradient of SDF w.r.t. coordinates (B, M, 3) or (M, 3)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Eikonal loss scalar or tensor
    """
    grad_norm = torch.norm(sdf_gradients, dim=-1)
    loss = (grad_norm - 1) ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def bending_loss(
    pred_pos: torch.Tensor,
    edge_index: torch.Tensor,
    rest_angles: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Bending resistance loss for cloth.
    
    Penalizes deviation from rest dihedral angles between adjacent faces.
    Simplified version using edge angle approximation.
    
    Args:
        pred_pos: Predicted node positions (B, N, 3)
        edge_index: Edge indices (2, E)
        rest_angles: Rest angles per edge pair
        
    Returns:
        Bending loss scalar
    """
    # Simplified: penalize local curvature by comparing edge vectors
    batched = pred_pos.dim() == 3
    if not batched:
        pred_pos = pred_pos.unsqueeze(0)
        
    B, N, _ = pred_pos.shape
    src, tgt = edge_index[0], edge_index[1]
    
    losses = []
    for b in range(B):
        pos = pred_pos[b]
        edge_vecs = pos[tgt] - pos[src]  # (E, 3)
        edge_vecs_norm = F.normalize(edge_vecs, dim=-1)
        
        # Compare adjacent edges (simplified)
        # For each vertex, compute angle between incoming edges
        # This is an approximation of bending energy
        
        # Compute variance of edge directions (should be low for flat cloth)
        mean_dir = edge_vecs_norm.mean(dim=0, keepdim=True)
        dir_variance = ((edge_vecs_norm - mean_dir) ** 2).sum(dim=-1).mean()
        losses.append(dir_variance)
        
    return torch.stack(losses).mean()


class PhysicsLoss(nn.Module):
    """
    Combined multi-task physics loss for HGNN-NIF-Cloth.
    
    L_total = λ_spring * L_spring + λ_sdf * L_SDF + λ_eik * L_eikonal
    
    Args:
        lambda_spring: Weight for edge spring loss (default: 0.1)
        lambda_sdf: Weight for SDF reconstruction loss (default: 1.0)
        lambda_eikonal: Weight for Eikonal regularization (default: 0.01)
        lambda_bending: Weight for bending loss (default: 0.0)
        surface_weight: Weight multiplier for near-surface SDF points
        surface_threshold: Distance to consider "near surface"
    """
    
    def __init__(
        self,
        lambda_spring: float = 0.1,
        lambda_sdf: float = 1.0,
        lambda_eikonal: float = 0.01,
        lambda_bending: float = 0.0,
        surface_weight: float = 10.0,
        surface_threshold: float = 0.02
    ):
        super().__init__()
        self.lambda_spring = lambda_spring
        self.lambda_sdf = lambda_sdf
        self.lambda_eikonal = lambda_eikonal
        self.lambda_bending = lambda_bending
        self.surface_weight = surface_weight
        self.surface_threshold = surface_threshold
        
        # For tracking individual losses
        self.last_losses = {}
        
    def forward(
        self,
        pred_pos: torch.Tensor,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None,
        sdf_gradients: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined physics loss.
        
        Args:
            pred_pos: Predicted node positions (B, N, 3)
            pred_sdf: Predicted SDF values (B, M)
            gt_sdf: Ground truth SDF values (B, M)
            edge_index: Edge indices (2, E)
            rest_lengths: Optional rest lengths for spring loss
            sdf_gradients: Optional SDF gradients for Eikonal loss
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total = 0.0
        
        # Edge spring loss
        if self.lambda_spring > 0:
            spring = edge_spring_loss(pred_pos, edge_index, rest_lengths)
            loss_dict['spring'] = spring
            total = total + self.lambda_spring * spring
            
        # SDF reconstruction loss
        if self.lambda_sdf > 0:
            sdf_loss = sdf_reconstruction_loss(
                pred_sdf, gt_sdf,
                weight_near_surface=self.surface_weight,
                surface_threshold=self.surface_threshold
            )
            loss_dict['sdf'] = sdf_loss
            total = total + self.lambda_sdf * sdf_loss
            
        # Eikonal regularization
        if self.lambda_eikonal > 0 and sdf_gradients is not None:
            eik = eikonal_loss(sdf_gradients)
            loss_dict['eikonal'] = eik
            total = total + self.lambda_eikonal * eik
            
        # Bending loss
        if self.lambda_bending > 0:
            bend = bending_loss(pred_pos, edge_index)
            loss_dict['bending'] = bend
            total = total + self.lambda_bending * bend
            
        loss_dict['total'] = total
        self.last_losses = {k: v.item() for k, v in loss_dict.items()}
        
        return total, loss_dict
    
    def get_last_losses(self) -> Dict[str, float]:
        """Return the last computed loss values."""
        return self.last_losses


class CurriculumWeightScheduler:
    """
    Scheduler for curriculum learning loss weights.
    
    Gradually adjusts loss weights over training to implement
    curriculum learning (easy → hard).
    
    Stages:
        0: Focus on SDF reconstruction (low physics weight)
        1: Add physics constraints
        2: Full training with all losses
    """
    
    def __init__(
        self,
        loss_fn: PhysicsLoss,
        total_epochs: int = 150,
        stage_fractions: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    ):
        self.loss_fn = loss_fn
        self.total_epochs = total_epochs
        self.stage_fractions = stage_fractions
        
        # Store initial weights
        self.init_lambda_spring = loss_fn.lambda_spring
        self.init_lambda_sdf = loss_fn.lambda_sdf
        self.init_lambda_eikonal = loss_fn.lambda_eikonal
        
        # Compute epoch boundaries
        self.stage1_end = int(total_epochs * stage_fractions[0])
        self.stage2_end = int(total_epochs * (stage_fractions[0] + stage_fractions[1]))
        
    def step(self, epoch: int) -> Dict[str, float]:
        """
        Update loss weights based on current epoch.
        
        Returns current weight values.
        """
        if epoch < self.stage1_end:
            # Stage 0: Focus on SDF, minimal physics
            self.loss_fn.lambda_spring = self.init_lambda_spring * 0.1
            self.loss_fn.lambda_sdf = self.init_lambda_sdf
            self.loss_fn.lambda_eikonal = self.init_lambda_eikonal * 0.1
            stage = 0
            
        elif epoch < self.stage2_end:
            # Stage 1: Ramp up physics
            progress = (epoch - self.stage1_end) / (self.stage2_end - self.stage1_end)
            self.loss_fn.lambda_spring = self.init_lambda_spring * (0.1 + 0.9 * progress)
            self.loss_fn.lambda_sdf = self.init_lambda_sdf
            self.loss_fn.lambda_eikonal = self.init_lambda_eikonal * (0.1 + 0.9 * progress)
            stage = 1
            
        else:
            # Stage 2: Full training
            self.loss_fn.lambda_spring = self.init_lambda_spring
            self.loss_fn.lambda_sdf = self.init_lambda_sdf
            self.loss_fn.lambda_eikonal = self.init_lambda_eikonal
            stage = 2
            
        return {
            'stage': stage,
            'lambda_spring': self.loss_fn.lambda_spring,
            'lambda_sdf': self.loss_fn.lambda_sdf,
            'lambda_eikonal': self.loss_fn.lambda_eikonal
        }
