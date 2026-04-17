"""
Collision integration for Project 08 (HGNN-ClothDyn).

This project uses physics-encoded EdgeForceConv with Hooke's law.
Integration adds collision forces alongside the spring forces in message passing.

Usage:
    from collision_integration import EdgeForceConvWithCollision
    
    conv = EdgeForceConvWithCollision(in_channels, out_channels, body_mesh)
    forces = conv(x, edge_index, edge_attr, positions)
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


class EdgeForceConvWithCollision(nn.Module):
    """
    EdgeForceConv layer augmented with collision forces.
    
    The base EdgeForceConv computes Hooke's law spring forces.
    This wrapper adds collision repulsion forces to the total force computation.
    """
    
    def __init__(
        self,
        base_edge_force_conv: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_stiffness: float = 1000.0,
        sdf_resolution: int = 64,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize collision-aware EdgeForceConv.
        
        Args:
            base_edge_force_conv: Original EdgeForceConv layer
            body_vertices: (V, 3) body mesh vertices
            body_faces: (F, 3) body mesh faces
            collision_stiffness: Collision response stiffness
            sdf_resolution: SDF grid resolution
            device: Target device
        """
        super().__init__()
        self.edge_force_conv = base_edge_force_conv
        self.device = device or next(base_edge_force_conv.parameters()).device
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        self.detector = CollisionDetector(proximity_threshold=0.01)
        self.response = CollisionResponse(stiffness=collision_stiffness)
    
    def update_body(self, body_vertices: torch.Tensor) -> None:
        """Update body mesh."""
        self.body.update_from_deformation(body_vertices)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: spring forces + collision forces.
        
        Args:
            x: (V, F) node features
            edge_index: (2, E) edge connectivity
            edge_attr: (E, F_edge) edge attributes (k, L0)
            positions: (V, 3) current positions (if not in x)
            
        Returns:
            (V, 3) total forces (spring + collision)
        """
        # Get spring forces from base EdgeForceConv
        spring_forces = self.edge_force_conv(x, edge_index, edge_attr)
        
        # Get positions for collision detection
        if positions is None:
            positions = x[:, :3]  # Assume first 3 channels are positions
        
        # Compute collision forces
        collisions = self.detector.detect(positions, self.body.get_sdf())
        collision_forces = self.response.resolve_forces(positions, collisions)
        
        # Combine forces
        total_forces = spring_forces + collision_forces
        
        return total_forces


class HGNNClothDynWithCollision(nn.Module):
    """
    Full HGNN-ClothDyn model with collision handling at each hierarchy level.
    """
    
    def __init__(
        self,
        hgnn_model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_stiffness: float = 1000.0,
        friction: float = 0.3,
        sdf_resolution: int = 64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hgnn = hgnn_model
        self.device = device or next(hgnn_model.parameters()).device
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        self.detector = CollisionDetector(proximity_threshold=0.01)
        self.response = CollisionResponse(
            stiffness=collision_stiffness,
            friction=friction,
            damping=0.1,
        )
    
    def update_body(self, body_vertices: torch.Tensor) -> None:
        """Update body for new frame."""
        self.body.update_from_deformation(body_vertices)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with collision handling.
        
        Args:
            x: (V, F) node features
            edge_index: (2, E) edges
            edge_attr: (E, F_edge) edge attributes
            batch: Optional batch indices
            
        Returns:
            Tuple of (corrected_positions, forces)
        """
        # Run base HGNN
        forces = self.hgnn(x, edge_index, edge_attr, batch)
        
        # Extract positions
        positions = x[:, :3]
        
        # Add collision forces
        collisions = self.detector.detect(positions, self.body.get_sdf())
        collision_forces = self.response.resolve_forces(positions, collisions)
        total_forces = forces + collision_forces
        
        # Simple integration
        dt = 1.0 / 30.0
        velocities = x[:, 3:6] if x.shape[1] >= 6 else torch.zeros_like(positions)
        new_velocities = velocities + dt * total_forces
        new_positions = positions + dt * new_velocities
        
        # Position correction for any remaining penetrations
        new_collisions = self.detector.detect(new_positions, self.body.get_sdf())
        if new_collisions.has_collisions:
            new_positions = self.response.resolve_positions(new_positions, new_collisions)
        
        return new_positions, total_forces


class CollisionTrainingWrapper:
    """
    Training wrapper that adds collision loss to HGNN-ClothDyn training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=64,
            device=self.device,
        )
        
        self.collision_loss = CollisionLoss(
            weights={'penetration': 10.0, 'proximity': 1.0}
        )
        self.collision_weight = collision_weight
    
    def train_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Training step with collision loss.
        
        Args:
            batch: Dict with 'x', 'edge_index', 'edge_attr', 'target'
            optimizer: Optimizer
            
        Returns:
            Dict of loss values
        """
        self.model.train()
        optimizer.zero_grad()
        
        x = batch['x'].to(self.device)
        edge_index = batch['edge_index'].to(self.device)
        edge_attr = batch['edge_attr'].to(self.device)
        target = batch['target'].to(self.device)
        
        # Forward
        pred_positions, _ = self.model(x, edge_index, edge_attr)
        
        # Losses
        recon_loss = nn.functional.mse_loss(pred_positions, target)
        coll_loss = self.collision_loss(pred_positions, self.body.get_sdf())
        
        total_loss = recon_loss + self.collision_weight * coll_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'collision': coll_loss.item(),
        }
