"""
Collision integration for Project 11 (NIF-Cloth3D).

NIF-Cloth3D uses SIREN for static cloth SDF representation.
Integration adds collision awareness for body interaction.

Usage:
    from collision_integration import NIFCloth3DWithCollision
    
    model = NIFCloth3DWithCollision(siren_model, body_mesh)
    cloth_sdf = model(query_coords)
    corrected_mesh = model.extract_and_correct_mesh()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional, Tuple

from shared.collision import (
    DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss, SDFField
)
from shared.collision.mesh_utils import marching_cubes_mesh


class NIFCloth3DWithCollision(nn.Module):
    """
    NIF-Cloth3D with collision-aware mesh extraction.
    
    The SIREN model predicts cloth SDF. This wrapper:
    1. Extracts mesh using marching cubes
    2. Corrects mesh vertices against body collision
    3. Optionally adds collision loss during training
    """
    
    def __init__(
        self,
        siren_model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        sdf_resolution: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize collision-aware NIF-Cloth3D.
        
        Args:
            siren_model: Base SIREN model for cloth SDF
            body_vertices: (V, 3) body mesh vertices
            body_faces: (F, 3) body mesh faces
            sdf_resolution: Resolution for body SDF
            device: Target device
        """
        super().__init__()
        self.siren = siren_model
        self.device = device or next(siren_model.parameters()).device
        
        self.body = DeformableBody(
            body_vertices, body_faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        self.detector = CollisionDetector(proximity_threshold=0.005)
        self.response = CollisionResponse(
            stiffness=1000.0,
            friction=0.0,  # Static mesh, no friction needed
            damping=0.0,
        )
    
    def update_body(self, body_vertices: torch.Tensor) -> None:
        """Update body mesh."""
        self.body.update_from_deformation(body_vertices)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - predict cloth SDF.
        
        Args:
            coords: (N, 3) query coordinates
            
        Returns:
            (N, 1) SDF values
        """
        return self.siren(coords.to(self.device))
    
    @torch.no_grad()
    def extract_mesh(
        self,
        resolution: int = 128,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract cloth mesh from predicted SDF using marching cubes.
        
        Args:
            resolution: Grid resolution for marching cubes
            bounds: Spatial bounds (min, max)
            
        Returns:
            Tuple of (vertices, faces)
        """
        # Build SDF from SIREN predictions
        cloth_sdf = self._build_cloth_sdf(resolution, bounds)
        
        # Extract mesh using marching cubes
        vertices, faces = marching_cubes_mesh(
            cloth_sdf._data.grid,
            cloth_sdf._data.bbox_min,
            cloth_sdf._data.bbox_max,
        )
        
        return vertices, faces
    
    @torch.no_grad()
    def extract_and_correct_mesh(
        self,
        resolution: int = 128,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract cloth mesh and correct collisions with body.
        
        Args:
            resolution: Grid resolution
            bounds: Spatial bounds
            
        Returns:
            Tuple of (corrected_vertices, faces)
        """
        vertices, faces = self.extract_mesh(resolution, bounds)
        
        # Detect and resolve collisions
        collisions = self.detector.detect(vertices, self.body.get_sdf())
        
        if collisions.has_collisions:
            vertices = self.response.resolve_positions(vertices, collisions)
        
        return vertices, faces
    
    def _build_cloth_sdf(
        self,
        resolution: int,
        bounds: Tuple[float, float],
    ) -> SDFField:
        """Build SDF from SIREN predictions."""
        from shared.collision.sdf_field import SDFFieldData
        
        # Create grid
        linspace = torch.linspace(bounds[0], bounds[1], resolution, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        
        # Query SIREN in chunks to avoid OOM
        chunk_size = 32768
        sdf_values = []
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords[i:i + chunk_size]
            sdf_chunk = self.siren(chunk)
            sdf_values.append(sdf_chunk)
        
        sdf_values = torch.cat(sdf_values, dim=0)
        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
        
        # Create SDFField with proper data structure
        data = SDFFieldData(
            grid=sdf_grid,
            bbox_min=torch.tensor([bounds[0]] * 3, device=self.device),
            bbox_max=torch.tensor([bounds[1]] * 3, device=self.device),
            resolution=resolution,
            device=self.device,
        )
        return SDFField(data)


class NIFCloth3DCollisionLoss(nn.Module):
    """
    Collision loss for NIF-Cloth3D training.
    
    Penalizes cloth SDF predictions that would result in body penetration.
    """
    
    def __init__(
        self,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.weight = weight
        self.device = device or body_vertices.device
        
        # Pre-build body SDF
        self.body_sdf = SDFField.from_mesh(
            body_vertices, body_faces,
            resolution=64,  # Lower res for training efficiency
            device=self.device,
        )
    
    def forward(
        self,
        coords: torch.Tensor,
        pred_cloth_sdf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute collision loss.
        
        Args:
            coords: (N, 3) query coordinates
            pred_cloth_sdf: (N, 1) predicted cloth SDF values
            
        Returns:
            Scalar collision loss
        """
        # Find points on cloth surface (SDF ≈ 0)
        surface_mask = torch.abs(pred_cloth_sdf.squeeze()) < 0.01
        
        if not surface_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        surface_points = coords[surface_mask]
        
        # Check these points against body SDF
        body_sdf_values = self.body_sdf.query(surface_points)
        
        # Penalize penetration (negative body SDF)
        penetration = torch.relu(-body_sdf_values)
        
        return self.weight * (penetration ** 2).mean()


class NIFCloth3DTrainer:
    """
    Training wrapper for NIF-Cloth3D with collision awareness.
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices: torch.Tensor,
        body_faces: torch.Tensor,
        collision_weight: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.collision_loss = NIFCloth3DCollisionLoss(
            body_vertices, body_faces,
            weight=collision_weight,
            device=self.device,
        )
    
    def train_step(
        self,
        coords: torch.Tensor,
        target_sdf: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Training step with collision loss.
        
        Args:
            coords: (N, 3) query coordinates
            target_sdf: (N, 1) target SDF values
            optimizer: Optimizer
            
        Returns:
            Dict of loss values
        """
        self.model.train()
        optimizer.zero_grad()
        
        coords = coords.to(self.device)
        target_sdf = target_sdf.to(self.device)
        
        # Forward
        pred_sdf = self.model(coords)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(pred_sdf, target_sdf)
        
        # Collision loss
        coll_loss = self.collision_loss(coords, pred_sdf)
        
        total_loss = recon_loss + coll_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'collision': coll_loss.item(),
        }
