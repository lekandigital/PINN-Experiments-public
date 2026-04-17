"""
Collision integration for Project 05 (ClothGNN).

ClothGNN uses force-based dynamics with Encoder-GRU-Decoder architecture.
Integration adds collision forces to the physics simulation.

Usage:
    from collision_integration import ClothGNNWithCollision
    
    model = ClothGNNWithCollision(base_model, body_mesh)
    new_positions, new_velocities = model.forward_with_collision(
        node_features, edge_index, edge_attr, velocities
    )
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


class ClothGNNWithCollision(nn.Module):
    """
    Wraps ClothGNN with force-based collision handling.
    
    Since ClothGNN uses explicit force-based dynamics, we add collision forces
    rather than position-based correction.
    """
    
    def __init__(
        self,
        cloth_gnn_model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_stiffness: float = 1000.0,
        friction: float = 0.3,
        sdf_resolution: int = 64,  # Lower for real-time
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cloth_gnn = cloth_gnn_model
        self.device = device or next(cloth_gnn_model.parameters()).device
        
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
        """Update body mesh for new frame."""
        self.body.update_from_deformation(body_vertices)
    
    def compute_collision_forces(
        self,
        cloth_positions: torch.Tensor,
        cloth_velocities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute collision forces for cloth vertices.
        
        Args:
            cloth_positions: (V, 3) or (B, V, 3) cloth vertex positions
            cloth_velocities: Optional velocities for friction
            
        Returns:
            Tuple of (collision_forces, adjusted_velocities)
        """
        collisions = self.detector.detect(cloth_positions, self.body.get_sdf())
        
        # Get repulsive forces
        forces = self.response.resolve_forces(cloth_positions, collisions)
        
        # Apply friction to velocities if provided
        if cloth_velocities is not None and collisions.has_collisions:
            adjusted_velocities = self.response.apply_friction(
                cloth_velocities, collisions
            )
            adjusted_velocities = self.response.apply_damping(
                adjusted_velocities, collisions
            )
        else:
            adjusted_velocities = cloth_velocities
        
        return forces, adjusted_velocities
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with collision forces added.
        
        Assumes node_features contains positions and velocities.
        Collision forces are added to the force prediction.
        
        Args:
            node_features: (V, F) node features including positions
            edge_index: (2, E) edge connectivity
            edge_attr: (E, F_edge) edge attributes
            hidden: Optional GRU hidden state
            
        Returns:
            Tuple of (new_positions, new_hidden)
        """
        # Extract positions and velocities from node features
        # Adjust indices based on your actual feature layout
        positions = node_features[:, :3]
        velocities = node_features[:, 3:6] if node_features.shape[1] >= 6 else None
        
        # Run base ClothGNN to get force predictions
        if hidden is not None:
            forces_pred, new_hidden = self.cloth_gnn(
                node_features, edge_index, edge_attr, hidden
            )
        else:
            forces_pred = self.cloth_gnn(node_features, edge_index, edge_attr)
            new_hidden = None
        
        # Add collision forces
        collision_forces, adjusted_velocities = self.compute_collision_forces(
            positions, velocities
        )
        total_forces = forces_pred + collision_forces
        
        # Integrate (simple Euler - your model may do this internally)
        # If ClothGNN already does integration, you may need to modify this
        dt = 1.0 / 30.0  # Adjust based on your timestep
        if velocities is not None:
            new_velocities = adjusted_velocities + dt * total_forces
            new_positions = positions + dt * new_velocities
        else:
            new_positions = positions + dt * dt * total_forces  # Assume mass=1
        
        return new_positions, new_hidden


class CollisionAwareTraining:
    """
    Training utilities for ClothGNN with collision loss.
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
    
    def compute_loss(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined reconstruction + collision loss.
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(pred_positions, target_positions)
        
        # Collision loss
        coll_loss = self.collision_loss(pred_positions, self.body.get_sdf())
        
        total = recon_loss + self.collision_weight * coll_loss
        
        return total, {
            'total': total.item(),
            'reconstruction': recon_loss.item(),
            'collision': coll_loss.item(),
        }
