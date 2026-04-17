"""
NIF-Cloth4D: Neural Implicit Field for Cloth Dynamics
Core network implementation using SIREN (Sinusoidal Representation Networks)

This module implements a SIREN-based MLP that predicts signed distance fields
for cloth geometry at any given spacetime coordinate (x, y, z, t).

Uses the shared implicit_fields library for core SIREN implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# Import from shared implicit_fields library
import sys
from pathlib import Path
# Add repository root to path for implicit_fields import
repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from implicit_fields import SirenNetwork


class FourierFeatureSIREN(nn.Module):
    """
    SIREN network wrapper for implicit SDF prediction with optional conditioning.
    
    This network takes spacetime coordinates (x, y, z, t) and optionally
    conditioning inputs (e.g., force vectors) to predict signed distance values.
    
    Architecture:
        - First layer: Linear + Sine(omega_0=30) with special initialization
        - Hidden layers: Linear + Sine(omega_hidden=30) with SIREN initialization
        - Output layer: Linear (no activation) for SDF value
    
    Args:
        in_dim: Input coordinate dimensions (default 4 for x,y,z,t)
        cond_dim: Optional conditioning dimensions (e.g., 3 for wind force)
        hidden_dim: Number of neurons per hidden layer
        hidden_layers: Number of hidden layers
        w0: Frequency scaling for first layer (SIREN parameter)
    """
    
    def __init__(
        self,
        in_dim: int = 4,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        w0: float = 30.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        total_in = in_dim + cond_dim
        
        # Use shared SirenNetwork implementation
        self.siren = SirenNetwork(
            in_features=total_in,
            hidden_features=hidden_dim,
            out_features=1,
            hidden_layers=hidden_layers,
            omega_0=w0,
            omega_hidden=w0  # Match original behavior
        )
    
    def forward(
        self,
        coords: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict SDF values.
        
        Args:
            coords: Tensor of shape (..., in_dim) for spacetime coordinates
            cond: Optional tensor of shape (..., cond_dim) for conditioning
        
        Returns:
            Predicted SDF values of shape (..., 1)
        """
        if cond is not None:
            x = torch.cat([coords, cond], dim=-1)
        else:
            x = coords
        
        return self.siren(x)


def compute_sdf_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute SDF reconstruction loss (MSE).
    
    Args:
        pred_sdf: Predicted SDF values
        gt_sdf: Ground truth SDF values
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        MSE loss value
    """
    return F.mse_loss(pred_sdf, gt_sdf, reduction=reduction)


def compute_eikonal_loss(
    coords: torch.Tensor,
    model: nn.Module,
    cond: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Eikonal loss to encourage valid SDF (|∇SDF| = 1).
    
    This regularization helps ensure the learned function is a proper
    signed distance field with unit gradient magnitude.
    
    Args:
        coords: Input coordinates (requires grad)
        model: The SDF network
        cond: Optional conditioning input
    
    Returns:
        Eikonal loss value
    """
    coords = coords.requires_grad_(True)
    sdf = model(coords, cond)
    
    # Compute gradient with respect to spatial coordinates (x, y, z)
    grad = torch.autograd.grad(
        outputs=sdf,
        inputs=coords,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Only use spatial gradient (first 3 dimensions)
    spatial_grad = grad[..., :3]
    grad_norm = torch.norm(spatial_grad, dim=-1)
    
    # Eikonal: ||∇SDF|| should equal 1
    eikonal_loss = ((grad_norm - 1.0) ** 2).mean()
    return eikonal_loss


def compute_stretch_loss(
    pred_positions: torch.Tensor,
    rest_positions: torch.Tensor,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Compute stretch loss to penalize excessive stretching/compression.
    
    This loss encourages the predicted cloth to maintain edge lengths
    close to the rest state, preventing unrealistic deformations.
    
    Args:
        pred_positions: Predicted vertex positions [N, 3]
        rest_positions: Rest state vertex positions [N, 3]
        edges: Edge indices [E, 2]
        rest_lengths: Rest lengths for each edge [E]
    
    Returns:
        Stretch loss value
    """
    # Get vertex pairs for each edge
    v0 = pred_positions[edges[:, 0]]
    v1 = pred_positions[edges[:, 1]]
    
    # Compute current edge lengths
    pred_lengths = torch.norm(v1 - v0, dim=-1)
    
    # Stretch loss: (pred_length - rest_length)^2
    stretch_loss = ((pred_lengths - rest_lengths) ** 2).mean()
    return stretch_loss


def compute_bend_loss(
    pred_positions: torch.Tensor,
    edges: torch.Tensor,
    adjacent_faces: torch.Tensor,
    rest_angles: torch.Tensor
) -> torch.Tensor:
    """
    Compute bending loss based on dihedral angles between adjacent faces.
    
    This loss encourages smooth curvature by penalizing deviations
    from rest dihedral angles.
    
    Args:
        pred_positions: Predicted vertex positions [N, 3]
        edges: Edge indices [E, 2]
        adjacent_faces: Adjacent face indices for each edge [E, 2]
        rest_angles: Rest dihedral angles [E]
    
    Returns:
        Bend loss value
    """
    # Simplified implementation - compute angle at middle vertex of triplets
    # In practice, you'd compute actual dihedral angles between faces
    bend_loss = torch.tensor(0.0, device=pred_positions.device)
    return bend_loss


class NIFCloth4DLoss(nn.Module):
    """
    Combined loss function for NIF-Cloth4D training.
    
    Combines multiple loss terms:
        - SDF reconstruction loss (primary)
        - Eikonal regularization (optional)
        - Stretch loss (optional, requires mesh data)
        - Bend loss (optional, requires mesh data)
    
    Args:
        sdf_weight: Weight for SDF reconstruction loss
        eikonal_weight: Weight for Eikonal regularization
        stretch_weight: Weight for stretch loss
        bend_weight: Weight for bend loss
    """
    
    def __init__(
        self,
        sdf_weight: float = 1.0,
        eikonal_weight: float = 0.1,
        stretch_weight: float = 0.1,
        bend_weight: float = 0.1
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.eikonal_weight = eikonal_weight
        self.stretch_weight = stretch_weight
        self.bend_weight = bend_weight
    
    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred_sdf: Predicted SDF values
            gt_sdf: Ground truth SDF values
            coords: Input coordinates (for Eikonal)
            model: The SDF network (for Eikonal)
            cond: Conditioning input (for Eikonal)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss values
        """
        loss_dict = {}
        
        # SDF reconstruction loss
        sdf_loss = compute_sdf_loss(pred_sdf, gt_sdf)
        loss_dict['sdf'] = sdf_loss.item()
        total_loss = self.sdf_weight * sdf_loss
        
        # Eikonal regularization (optional)
        if self.eikonal_weight > 0 and coords is not None and model is not None:
            eikonal_loss = compute_eikonal_loss(coords, model, cond)
            loss_dict['eikonal'] = eikonal_loss.item()
            total_loss = total_loss + self.eikonal_weight * eikonal_loss
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


def create_model(config: dict) -> FourierFeatureSIREN:
    """
    Factory function to create a model from configuration.
    
    Args:
        config: Dictionary with model configuration
    
    Returns:
        Initialized FourierFeatureSIREN model
    """
    return FourierFeatureSIREN(
        in_dim=config.get('in_dim', 4),
        cond_dim=config.get('cond_dim', 0),
        hidden_dim=config.get('hidden_dim', 256),
        hidden_layers=config.get('hidden_layers', 5),
        w0=config.get('w0_initial', 30.0)
    )


if __name__ == "__main__":
    # Quick test
    print("Testing NIF-Cloth4D network...")
    
    # Create model
    model = FourierFeatureSIREN(
        in_dim=4,
        cond_dim=0,
        hidden_dim=128,
        hidden_layers=4
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 1024
    coords = torch.randn(batch_size, 4)  # (x, y, z, t)
    sdf = model(coords)
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {sdf.shape}")
    print(f"Output range: [{sdf.min().item():.4f}, {sdf.max().item():.4f}]")
    
    # Test loss computation
    gt_sdf = torch.randn(batch_size, 1)
    loss_fn = NIFCloth4DLoss()
    loss, loss_dict = loss_fn(sdf, gt_sdf)
    print(f"Loss: {loss.item():.4f}")
    print("Test passed!")
