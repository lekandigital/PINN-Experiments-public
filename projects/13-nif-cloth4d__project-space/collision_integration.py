"""
Collision integration for Project 13 (NIF-Cloth4D).

Ultra-compact 66K parameter model. Uses lightweight collision:
- Position-based correction only (no friction/damping)
- 64³ SDF resolution to minimize overhead
- Minimal memory footprint

Usage:
    from collision_integration import NIFCloth4DWithCollision
    
    model = NIFCloth4DWithCollision(base_model, body_mesh)
    corrected = model.forward_with_collision(coords, time)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional, Tuple

from shared.collision import (
    DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss
)


class NIFCloth4DWithCollision(nn.Module):
    """
    Lightweight collision wrapper for NIF-Cloth4D.
    
    Designed to maintain the ultra-compact footprint:
    - 64³ SDF (1MB) instead of 128³ (8MB)
    - Position correction only, no velocity tracking
    - Minimal collision config
    """
    
    def __init__(
        self,
        nif_cloth4d_model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        sdf_resolution: int = 64,  # Keep low for this lightweight model
        device: Optional[torch.device] = None,
    ):
        """
        Initialize lightweight collision wrapper.
        
        Args:
            nif_cloth4d_model: Base NIF-Cloth4D SIREN model
            body_vertices: (V, 3) body mesh vertices
            body_faces: (F, 3) body mesh faces
            sdf_resolution: SDF grid resolution (default 64 for speed)
            device: Target device
        """
        super().__init__()
        self.nif_model = nif_cloth4d_model
        self.device = device or next(nif_cloth4d_model.parameters()).device
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        # Lightweight detector - no proximity tracking
        self.detector = CollisionDetector(
            proximity_threshold=0.0,  # Disable proximity
            enable_proximity=False,
        )
        
        # Simple position correction only
        self.response = CollisionResponse(
            stiffness=1000.0,
            friction=0.0,  # No friction
            damping=0.0,   # No damping
            max_correction=0.05,
        )
    
    def update_body(self, body_vertices: torch.Tensor) -> None:
        """Update body mesh."""
        self.body.update_from_deformation(body_vertices)
    
    def forward(
        self,
        coords: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (no collision - for training without collision).
        
        Args:
            coords: (N, 3) spatial coordinates
            time: Optional (N, 1) or scalar time value
            
        Returns:
            (N, 1) SDF values
        """
        if time is not None:
            # Concatenate time to coords for 4D query
            if time.dim() == 0:
                time = time.expand(coords.shape[0], 1)
            elif time.dim() == 1:
                time = time.unsqueeze(-1)
            input_coords = torch.cat([coords, time], dim=-1)
        else:
            input_coords = coords
        
        return self.nif_model(input_coords)
    
    def forward_with_collision(
        self,
        cloth_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with collision correction.
        
        For inference when you have extracted cloth vertices
        and need to ensure they don't penetrate the body.
        
        Args:
            cloth_vertices: (V, 3) cloth vertex positions
            
        Returns:
            (V, 3) corrected vertex positions
        """
        collisions = self.detector.detect(
            cloth_vertices, self.body.get_sdf(),
            compute_normals=True,
        )
        
        if collisions.has_collisions:
            return self.response.resolve_positions(cloth_vertices, collisions)
        
        return cloth_vertices


class LightweightCollisionLoss(nn.Module):
    """
    Minimal collision loss for NIF-Cloth4D training.
    
    Only penetration loss, no proximity/contact/eikonal.
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        cloth_vertices: torch.Tensor,
        body_sdf,  # SDFField
    ) -> torch.Tensor:
        """
        Compute simple penetration loss.
        
        Args:
            cloth_vertices: (V, 3) or (B, V, 3) cloth vertices
            body_sdf: SDFField for body
            
        Returns:
            Scalar loss
        """
        sdf_values = body_sdf.query(cloth_vertices)
        penetration = torch.relu(-sdf_values)  # Only when inside (sdf < 0)
        return self.weight * (penetration ** 2).mean()


class NIFCloth4DTrainer:
    """
    Training utilities for NIF-Cloth4D with minimal collision overhead.
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_weight: float = 0.1,  # Lower weight for this simple model
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=64,
            device=self.device,
        )
        
        self.collision_loss = LightweightCollisionLoss(weight=collision_weight)
    
    def train_step(
        self,
        coords: torch.Tensor,
        target_sdf: torch.Tensor,
        cloth_vertices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Training step with lightweight collision loss.
        
        Args:
            coords: (N, 4) spacetime coordinates (x, y, z, t)
            target_sdf: (N, 1) target SDF values
            cloth_vertices: (V, 3) cloth mesh vertices for collision
            optimizer: Optimizer
            
        Returns:
            Dict of loss values
        """
        self.model.train()
        optimizer.zero_grad()
        
        coords = coords.to(self.device)
        target_sdf = target_sdf.to(self.device)
        cloth_vertices = cloth_vertices.to(self.device)
        
        # Forward
        pred_sdf = self.model(coords)
        
        # SDF reconstruction loss
        recon_loss = nn.functional.mse_loss(pred_sdf, target_sdf)
        
        # Collision loss
        coll_loss = self.collision_loss(cloth_vertices, self.body.get_sdf())
        
        total_loss = recon_loss + coll_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'collision': coll_loss.item(),
        }


# Quick utility for inference
def apply_collision_correction(
    cloth_vertices: torch.Tensor,
    body_vertices: torch.Tensor,
    body_faces: torch.Tensor,
) -> torch.Tensor:
    """
    One-shot collision correction for inference.
    
    Simple function for quick integration without full wrapper.
    
    Args:
        cloth_vertices: (V, 3) cloth vertices
        body_vertices: (V_body, 3) body vertices
        body_faces: (F, 3) body faces
        
    Returns:
        (V, 3) corrected cloth vertices
    """
    from shared.collision import SDFField
    
    device = cloth_vertices.device
    body_sdf = SDFField.from_mesh(
        body_vertices.to(device),
        body_faces.to(device),
        resolution=64,
    )
    
    detector = CollisionDetector(enable_proximity=False)
    response = CollisionResponse(friction=0.0, damping=0.0)
    
    result = detector.detect(cloth_vertices, body_sdf)
    return response.resolve_positions(cloth_vertices, result)
