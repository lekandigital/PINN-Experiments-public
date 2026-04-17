"""
Differentiable collision losses for training cloth models.

These losses penalize cloth-body penetration and encourage realistic
contact behavior during training.
"""

from __future__ import annotations
from typing import Dict, Optional, Callable
import torch
import torch.nn.functional as F

from .sdf_field import SDFField
from .config import CollisionConfig


def penetration_loss(
    cloth_vertices: torch.Tensor,
    body_sdf: SDFField,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute penetration loss: penalize vertices inside the body.
    
    L_penetration = mean(relu(-sdf(v))²)
    
    Squared to more aggressively penalize deep penetrations.
    Only activates for negative SDF values (inside the body).
    
    Args:
        cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
        body_sdf: SDFField representing the body surface
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Scalar loss tensor (or per-vertex if reduction='none')
    """
    sdf_values = body_sdf.query(cloth_vertices)
    
    # relu(-sdf) = max(0, -sdf) activates only when sdf < 0 (inside)
    penetration = F.relu(-sdf_values)
    loss = penetration ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def proximity_loss(
    cloth_vertices: torch.Tensor,
    body_sdf: SDFField,
    margin: float = 0.01,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute proximity loss: soft repulsion from body surface.
    
    L_proximity = mean(relu(margin - sdf(v))²)
    
    Creates a soft buffer zone around the body that cloth is gently
    pushed away from. Prevents cloth from getting so close that
    small perturbations cause penetration.
    
    Args:
        cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
        body_sdf: SDFField representing the body surface
        margin: Buffer zone distance (normalized coordinates)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Scalar loss tensor
    """
    sdf_values = body_sdf.query(cloth_vertices)
    
    # relu(margin - sdf) activates when sdf < margin (too close)
    proximity = F.relu(margin - sdf_values)
    loss = proximity ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def contact_loss(
    cloth_vertices: torch.Tensor,
    body_sdf: SDFField,
    contact_mask: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute contact loss: keep specified vertices on the body surface.
    
    L_contact = mean(sdf(contact_vertices)²)
    
    For vertices that should be in contact with the body (e.g., shirt
    collar on shoulders), penalize distance from the surface.
    
    Args:
        cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
        body_sdf: SDFField representing the body surface
        contact_mask: (V,) or (B, V) bool tensor marking contact vertices
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Scalar loss tensor
    """
    sdf_values = body_sdf.query(cloth_vertices)
    
    # Apply mask to get only contact vertices
    if contact_mask.any():
        contact_sdf = sdf_values[contact_mask]
        loss = contact_sdf ** 2
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    else:
        return torch.tensor(0.0, device=cloth_vertices.device)


def eikonal_loss(
    sdf_gradients: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute Eikonal loss: enforce |∇SDF| = 1.
    
    L_eikonal = mean((|∇φ| - 1)²)
    
    This regularization ensures the SDF represents valid signed distances.
    
    Args:
        sdf_gradients: (N, 3) or (B, N, 3) SDF gradient vectors
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Scalar loss tensor
    
    Note:
        Extracted and generalized from Project 09's implementation.
    """
    grad_norm = torch.norm(sdf_gradients, dim=-1)
    loss = (grad_norm - 1) ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def self_collision_loss(
    cloth_vertices: torch.Tensor,
    cloth_faces: torch.Tensor,
    min_distance: float = 0.005,
    sample_ratio: float = 0.1,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute self-collision loss using sampling-based detection.
    
    Penalizes cloth vertices that are too close to non-adjacent faces.
    
    Args:
        cloth_vertices: (V, 3) cloth vertex positions
        cloth_faces: (F, 3) cloth face indices
        min_distance: Minimum allowed distance
        sample_ratio: Fraction of pairs to sample
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Scalar loss tensor
        
    Note:
        Extracted from Project 12's implementation.
    """
    device = cloth_vertices.device
    num_vertices = cloth_vertices.shape[0]
    num_faces = cloth_faces.shape[0]
    
    # Sample random vertex-face pairs
    num_samples = int(num_vertices * num_faces * sample_ratio)
    num_samples = max(num_samples, 1000)
    
    vertex_indices = torch.randint(0, num_vertices, (num_samples,), device=device)
    face_indices = torch.randint(0, num_faces, (num_samples,), device=device)
    
    # Get sampled vertices and face centroids
    sampled_vertices = cloth_vertices[vertex_indices]
    face_verts = cloth_vertices[cloth_faces[face_indices]]
    face_centroids = face_verts.mean(dim=1)
    
    # Compute distances
    distances = torch.norm(sampled_vertices - face_centroids, dim=-1)
    
    # Penalize distances below threshold
    penetration = F.relu(min_distance - distances)
    loss = penetration ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class CollisionLoss(torch.nn.Module):
    """
    Combined collision loss module with configurable weights.
    
    Combines penetration, proximity, contact, and eikonal losses
    with configurable weights for training cloth models.
    
    Example:
        >>> loss_fn = CollisionLoss(weights={'penetration': 10.0, 'proximity': 1.0})
        >>> loss = loss_fn(cloth_vertices, body_sdf)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        proximity_margin: float = 0.01,
        config: Optional[CollisionConfig] = None,
    ):
        """
        Initialize collision loss.
        
        Args:
            weights: Dict mapping loss names to weights
            proximity_margin: Buffer zone for proximity loss
            config: Full collision config (overrides other params)
        """
        super().__init__()
        
        if config is not None:
            self.weights = config.loss_weights.copy()
            self.proximity_margin = config.proximity_margin
        else:
            self.weights = weights or {
                'penetration': 10.0,
                'proximity': 1.0,
                'contact': 0.5,
                'eikonal': 0.1,
            }
            self.proximity_margin = proximity_margin
    
    def forward(
        self,
        cloth_vertices: torch.Tensor,
        body_sdf: SDFField,
        contact_mask: Optional[torch.Tensor] = None,
        compute_eikonal: bool = False,
        eikonal_sample_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined collision loss.
        
        Args:
            cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
            body_sdf: SDFField representing the body surface
            contact_mask: Optional (V,) mask for contact loss
            compute_eikonal: Whether to compute eikonal loss
            eikonal_sample_points: Points for eikonal loss (if different from cloth vertices)
            
        Returns:
            Scalar loss tensor
        """
        total_loss = torch.tensor(0.0, device=cloth_vertices.device)
        
        # Penetration loss
        if self.weights.get('penetration', 0) > 0:
            pen_loss = penetration_loss(cloth_vertices, body_sdf)
            total_loss = total_loss + self.weights['penetration'] * pen_loss
        
        # Proximity loss
        if self.weights.get('proximity', 0) > 0:
            prox_loss = proximity_loss(cloth_vertices, body_sdf, margin=self.proximity_margin)
            total_loss = total_loss + self.weights['proximity'] * prox_loss
        
        # Contact loss
        if contact_mask is not None and self.weights.get('contact', 0) > 0:
            cont_loss = contact_loss(cloth_vertices, body_sdf, contact_mask)
            total_loss = total_loss + self.weights['contact'] * cont_loss
        
        # Eikonal loss
        if compute_eikonal and self.weights.get('eikonal', 0) > 0:
            sample_points = eikonal_sample_points if eikonal_sample_points is not None else cloth_vertices
            gradients = body_sdf.gradient(sample_points)
            eik_loss = eikonal_loss(gradients)
            total_loss = total_loss + self.weights['eikonal'] * eik_loss
        
        return total_loss
    
    def forward_detailed(
        self,
        cloth_vertices: torch.Tensor,
        body_sdf: SDFField,
        contact_mask: Optional[torch.Tensor] = None,
        compute_eikonal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual loss components for logging.
        
        Returns:
            Dict mapping loss names to values
        """
        losses = {}
        
        if self.weights.get('penetration', 0) > 0:
            losses['penetration'] = penetration_loss(cloth_vertices, body_sdf)
        
        if self.weights.get('proximity', 0) > 0:
            losses['proximity'] = proximity_loss(cloth_vertices, body_sdf, margin=self.proximity_margin)
        
        if contact_mask is not None and self.weights.get('contact', 0) > 0:
            losses['contact'] = contact_loss(cloth_vertices, body_sdf, contact_mask)
        
        if compute_eikonal and self.weights.get('eikonal', 0) > 0:
            gradients = body_sdf.gradient(cloth_vertices)
            losses['eikonal'] = eikonal_loss(gradients)
        
        # Compute weighted total
        losses['total'] = sum(
            self.weights.get(name, 0) * value
            for name, value in losses.items()
            if name != 'total'
        )
        
        return losses
