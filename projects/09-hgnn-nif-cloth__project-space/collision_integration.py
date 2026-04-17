"""
Collision integration for Project 09 (HGNN-NIF-Cloth).

This is the flagship cloth model with position-based dynamics.
Integration adds collision loss to training and position correction at inference.

Usage:
    # Training
    from collision_integration import CollisionTrainer
    trainer = CollisionTrainer(model, body_mesh, config)
    loss = trainer.train_step(cloth_batch, optimizer)
    
    # Inference
    from collision_integration import CollisionInference
    inference = CollisionInference(model, body_mesh)
    corrected_cloth = inference.forward_with_collision(input_features)
"""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any

from shared.collision import (
    DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss, SDFField
)
from shared.collision.config import CollisionConfig


class CollisionTrainer:
    """
    Wraps HGNN-NIF-Cloth training with collision loss.
    
    Example:
        >>> trainer = CollisionTrainer(model, body_vertices, body_faces)
        >>> for batch in dataloader:
        ...     loss = trainer.train_step(batch, optimizer)
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_weight: float = 1.0,
        sdf_resolution: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize collision-aware trainer.
        
        Args:
            model: HGNN-NIF-Cloth model
            body_vertices: (V, 3) body rest pose vertices
            body_faces: (F, 3) body mesh faces
            collision_weight: Weight for collision loss relative to reconstruction loss
            sdf_resolution: Resolution for body SDF
            device: Target device
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Initialize collision system
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        self.collision_loss = CollisionLoss(
            weights={
                'penetration': 10.0,
                'proximity': 1.0,
                'contact': 0.0,  # Set > 0 if you have contact vertex annotations
                'eikonal': 0.0,
            }
        )
        
        self.collision_weight = collision_weight
    
    def update_body(self, body_vertices: torch.Tensor) -> None:
        """Update body mesh (e.g., from PEGNN output)."""
        self.body.update_from_deformation(body_vertices)
    
    def compute_collision_loss(
        self,
        pred_cloth_vertices: torch.Tensor,
        contact_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute collision loss for predicted cloth vertices.
        
        Args:
            pred_cloth_vertices: (B, V, 3) or (V, 3) predicted cloth positions
            contact_mask: Optional mask for contact loss
            
        Returns:
            Scalar collision loss
        """
        return self.collision_loss(
            pred_cloth_vertices,
            self.body.get_sdf(),
            contact_mask=contact_mask,
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        body_vertices: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step with collision loss.
        
        Args:
            batch: Dict containing 'coords', 'latent', 'target_sdf' etc.
            optimizer: PyTorch optimizer
            body_vertices: Optional updated body vertices for this batch
            
        Returns:
            Dict of loss values for logging
        """
        # Update body if new vertices provided
        if body_vertices is not None:
            self.update_body(body_vertices)
        
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass - adapt this to your model's interface
        coords = batch['coords'].to(self.device)
        latent = batch.get('latent', None)
        if latent is not None:
            latent = latent.to(self.device)
        
        # Model forward (adjust based on your HGNN-NIF-Cloth architecture)
        pred_sdf = self.model(coords, latent) if latent is not None else self.model(coords)
        
        # Reconstruction loss
        target_sdf = batch['target_sdf'].to(self.device)
        recon_loss = nn.functional.mse_loss(pred_sdf, target_sdf)
        
        # Collision loss on predicted cloth surface
        # Extract surface vertices from predicted SDF (or use provided mesh vertices)
        if 'cloth_vertices' in batch:
            cloth_vertices = batch['cloth_vertices'].to(self.device)
            coll_loss = self.compute_collision_loss(cloth_vertices)
        else:
            coll_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = recon_loss + self.collision_weight * coll_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'collision_loss': coll_loss.item(),
        }


class CollisionInference:
    """
    Wraps HGNN-NIF-Cloth inference with collision resolution.
    
    Example:
        >>> inference = CollisionInference(model, body_vertices, body_faces)
        >>> # Each frame:
        >>> inference.update_body(new_body_vertices)
        >>> corrected_cloth = inference.forward_with_collision(cloth_input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        sdf_resolution: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        self.detector = CollisionDetector(proximity_threshold=0.005)
        self.response = CollisionResponse(
            stiffness=1000.0,
            friction=0.3,
            damping=0.1,
        )
    
    def update_body(self, body_vertices: torch.Tensor) -> None:
        """Update body for new frame."""
        self.body.update_from_deformation(body_vertices)
    
    @torch.no_grad()
    def forward_with_collision(
        self,
        cloth_input: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        cloth_velocities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run model and resolve collisions.
        
        Args:
            cloth_input: Input to cloth model (coords or features)
            latent: Optional latent code
            cloth_velocities: Optional velocities for friction/damping
            
        Returns:
            Tuple of (corrected_vertices, adjusted_velocities)
        """
        cloth_input = cloth_input.to(self.device)
        if latent is not None:
            latent = latent.to(self.device)
        
        # Get model prediction (adapt to your model's interface)
        if latent is not None:
            pred = self.model(cloth_input, latent)
        else:
            pred = self.model(cloth_input)
        
        # If model outputs SDF, extract vertices (or use direct vertex prediction)
        # This depends on your model - adjust accordingly
        cloth_vertices = pred  # Assuming direct vertex output
        
        # Detect and resolve collisions
        collisions = self.detector.detect(cloth_vertices, self.body.get_sdf())
        
        if collisions.has_collisions:
            corrected_vertices, adjusted_velocities = self.response.full_response(
                cloth_vertices,
                cloth_velocities.to(self.device) if cloth_velocities is not None else None,
                collisions,
            )
            return corrected_vertices, adjusted_velocities
        
        return cloth_vertices, cloth_velocities


# Convenience functions for quick integration
def add_collision_loss_to_training(
    pred_cloth_vertices: torch.Tensor,
    body_sdf: SDFField,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Quick way to add collision loss to existing training loop.
    
    Args:
        pred_cloth_vertices: (B, V, 3) predicted cloth vertices
        body_sdf: SDFField for body mesh
        weight: Loss weight
        
    Returns:
        Weighted collision loss
    """
    loss_fn = CollisionLoss(weights={'penetration': 10.0, 'proximity': 1.0})
    return weight * loss_fn(pred_cloth_vertices, body_sdf)


def resolve_collisions(
    cloth_vertices: torch.Tensor,
    body_sdf: SDFField,
) -> torch.Tensor:
    """
    Quick collision resolution for inference.
    
    Args:
        cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
        body_sdf: SDFField for body mesh
        
    Returns:
        Corrected vertex positions
    """
    detector = CollisionDetector()
    response = CollisionResponse()
    
    result = detector.detect(cloth_vertices, body_sdf)
    return response.resolve_positions(cloth_vertices, result)
