"""
Collision integration for Project 12 (NIF-Cloth4D-Temporal).

Extends NIF-Cloth4D with temporal coherence via GRU latent dynamics.
Collision integration handles time-varying body deformation.

Usage:
    from collision_integration import TemporalCollisionHandler
    
    handler = TemporalCollisionHandler(model, body_sequence)
    for t in range(num_frames):
        cloth_state = handler.step(t, input_coords)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from shared.collision import (
    DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss, SDFField
)


class TemporalCollisionHandler:
    """
    Handles collision for time-varying cloth and body.
    
    Key features:
    - Tracks body deformation over time
    - Velocity-aware collision response for temporal coherence
    - Caches body SDFs when body animation repeats
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices_sequence: torch.Tensor,  # (T, V, 3)
        body_faces: torch.Tensor,
        sdf_resolution: int = 64,
        cache_body_sdfs: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize temporal collision handler.
        
        Args:
            model: NIF-Cloth4D-Temporal model
            body_vertices_sequence: (T, V, 3) body animation
            body_faces: (F, 3) body mesh faces
            sdf_resolution: Resolution for body SDF
            cache_body_sdfs: Whether to pre-compute all body SDFs
            device: Target device
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.body_sequence = body_vertices_sequence.to(self.device)
        self.body_faces = body_faces.to(self.device)
        self.num_frames = body_vertices_sequence.shape[0]
        self.sdf_resolution = sdf_resolution
        
        # Collision components
        self.detector = CollisionDetector(proximity_threshold=0.01)
        self.response = CollisionResponse(
            stiffness=1000.0,
            friction=0.3,
            damping=0.1,
        )
        
        # State tracking
        self._current_frame = 0
        self._prev_cloth_positions: Optional[torch.Tensor] = None
        self._cloth_velocities: Optional[torch.Tensor] = None
        
        # SDF caching
        self._sdf_cache: List[Optional[SDFField]] = [None] * self.num_frames
        if cache_body_sdfs:
            self._precompute_body_sdfs()
    
    def _precompute_body_sdfs(self) -> None:
        """Pre-compute SDFs for all body frames."""
        print(f"Pre-computing {self.num_frames} body SDFs...")
        for t in range(self.num_frames):
            self._sdf_cache[t] = SDFField.from_mesh(
                self.body_sequence[t],
                self.body_faces,
                resolution=self.sdf_resolution,
                device=self.device,
            )
        print("Done.")
    
    def get_body_sdf(self, t: int) -> SDFField:
        """Get body SDF for frame t."""
        t = t % self.num_frames  # Loop animation
        
        if self._sdf_cache[t] is None:
            self._sdf_cache[t] = SDFField.from_mesh(
                self.body_sequence[t],
                self.body_faces,
                resolution=self.sdf_resolution,
                device=self.device,
            )
        
        return self._sdf_cache[t]
    
    @torch.no_grad()
    def step(
        self,
        t: int,
        input_coords: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        dt: float = 1.0 / 30.0,
    ) -> torch.Tensor:
        """
        Advance one frame with collision handling.
        
        Args:
            t: Frame index
            input_coords: (N, 3) query coordinates
            latent: Optional latent code for this frame
            dt: Time step
            
        Returns:
            (N, 3) corrected cloth positions
        """
        input_coords = input_coords.to(self.device)
        
        # Get model prediction
        if latent is not None:
            cloth_positions = self.model(input_coords, latent.to(self.device))
        else:
            cloth_positions = self.model(input_coords, t)
        
        # Estimate velocity from previous frame
        if self._prev_cloth_positions is not None:
            self._cloth_velocities = (cloth_positions - self._prev_cloth_positions) / dt
        else:
            self._cloth_velocities = torch.zeros_like(cloth_positions)
        
        # Get body SDF for this frame
        body_sdf = self.get_body_sdf(t)
        
        # Detect collisions
        collisions = self.detector.detect(cloth_positions, body_sdf)
        
        if collisions.has_collisions:
            # Full response with velocity update
            cloth_positions, self._cloth_velocities = self.response.full_response(
                cloth_positions,
                self._cloth_velocities,
                collisions,
            )
        
        # Store for next frame
        self._prev_cloth_positions = cloth_positions.detach()
        self._current_frame = t
        
        return cloth_positions
    
    def reset(self) -> None:
        """Reset temporal state."""
        self._current_frame = 0
        self._prev_cloth_positions = None
        self._cloth_velocities = None


class TemporalCollisionLoss(nn.Module):
    """
    Collision loss with temporal consistency for training.
    
    Adds:
    - Standard penetration loss
    - Velocity consistency loss (reduces jitter)
    - Temporal smoothness loss
    """
    
    def __init__(
        self,
        body_vertices_sequence: torch.Tensor,
        body_faces: torch.Tensor,
        penetration_weight: float = 10.0,
        velocity_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.penetration_weight = penetration_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight
        self.device = device or body_vertices_sequence.device
        
        # Store body sequence for SDF computation
        self.body_sequence = body_vertices_sequence.to(self.device)
        self.body_faces = body_faces.to(self.device)
        
        # Cache
        self._sdf_cache = {}
    
    def _get_body_sdf(self, t: int) -> SDFField:
        """Get or compute body SDF for frame t."""
        if t not in self._sdf_cache:
            self._sdf_cache[t] = SDFField.from_mesh(
                self.body_sequence[t],
                self.body_faces,
                resolution=64,
                device=self.device,
            )
        return self._sdf_cache[t]
    
    def forward(
        self,
        pred_positions: torch.Tensor,  # (T, N, 3) or list of (N, 3)
        frame_indices: torch.Tensor,   # (T,) frame indices
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute temporal collision loss.
        
        Args:
            pred_positions: Predicted positions for multiple frames
            frame_indices: Body frame indices for each prediction
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if isinstance(pred_positions, list):
            pred_positions = torch.stack(pred_positions, dim=0)
        
        T = pred_positions.shape[0]
        
        # Penetration loss per frame
        penetration_losses = []
        for t in range(T):
            body_sdf = self._get_body_sdf(frame_indices[t].item())
            sdf_values = body_sdf.query(pred_positions[t])
            penetration = torch.relu(-sdf_values)
            penetration_losses.append((penetration ** 2).mean())
        
        penetration_loss = torch.stack(penetration_losses).mean()
        
        # Velocity consistency loss (penalize large accelerations)
        if T > 2:
            velocities = pred_positions[1:] - pred_positions[:-1]
            accelerations = velocities[1:] - velocities[:-1]
            velocity_loss = (accelerations ** 2).mean()
        else:
            velocity_loss = torch.tensor(0.0, device=self.device)
        
        # Temporal smoothness (penalize sudden changes)
        if T > 1:
            smoothness_loss = ((pred_positions[1:] - pred_positions[:-1]) ** 2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = (
            self.penetration_weight * penetration_loss +
            self.velocity_weight * velocity_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'penetration': penetration_loss.item(),
            'velocity': velocity_loss.item(),
            'smoothness': smoothness_loss.item(),
        }


class NIFCloth4DTemporalTrainer:
    """
    Training wrapper for NIF-Cloth4D-Temporal with collision.
    """
    
    def __init__(
        self,
        model: nn.Module,
        body_vertices_sequence: torch.Tensor,
        body_faces: torch.Tensor,
        collision_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.collision_loss = TemporalCollisionLoss(
            body_vertices_sequence,
            body_faces,
            device=self.device,
        )
        self.collision_weight = collision_weight
    
    def train_step(
        self,
        coords_sequence: List[torch.Tensor],  # List of (N, 3)
        target_sequence: List[torch.Tensor],   # List of (N, 3)
        frame_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Training step on sequence with collision loss.
        
        Args:
            coords_sequence: Input coordinates per frame
            target_sequence: Target positions per frame
            frame_indices: Body frame indices
            optimizer: Optimizer
            
        Returns:
            Dict of loss values
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass for all frames
        predictions = []
        for t, coords in enumerate(coords_sequence):
            pred = self.model(coords.to(self.device), frame_indices[t])
            predictions.append(pred)
        
        # Reconstruction loss
        recon_loss = 0.0
        for pred, target in zip(predictions, target_sequence):
            recon_loss += nn.functional.mse_loss(pred, target.to(self.device))
        recon_loss /= len(predictions)
        
        # Collision loss
        coll_loss, coll_dict = self.collision_loss(
            predictions, frame_indices.to(self.device)
        )
        
        total_loss = recon_loss + self.collision_weight * coll_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'collision': coll_dict['total'],
            'penetration': coll_dict['penetration'],
            'velocity': coll_dict['velocity'],
            'smoothness': coll_dict['smoothness'],
        }
