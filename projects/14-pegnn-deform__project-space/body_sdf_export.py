"""
Body SDF export for Project 14 (PEGNN-Deform).

This project produces deformed body meshes that cloth collides against.
This module wraps PEGNN output to produce DeformableBody objects with SDF.

Usage:
    from body_sdf_export import PEGNNCollisionBridge
    
    bridge = PEGNNCollisionBridge(pegnn_model, rest_body_mesh)
    
    # Each frame
    body = bridge.step(node_features, edge_index, edge_attr)
    sdf = body.get_sdf()  # Pass to cloth collision system
"""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from shared.collision import DeformableBody, SDFField
from shared.collision.body_interface import PEGNNBodyAdapter


class PEGNNCollisionBridge:
    """
    Bridge between PEGNN-Deform model and collision system.
    
    Takes PEGNN model outputs and produces DeformableBody with updated SDF
    for cloth collision queries.
    
    Example:
        >>> bridge = PEGNNCollisionBridge(pegnn_model, rest_vertices, faces)
        >>> 
        >>> for frame in animation:
        ...     body = bridge.step(node_features, edge_index, edge_attr)
        ...     # Pass body.get_sdf() to cloth collision system
        ...     cloth_collisions = detector.detect(cloth_verts, body.get_sdf())
    """
    
    def __init__(
        self,
        pegnn_model: nn.Module,
        rest_vertices: torch.Tensor,
        faces: torch.Tensor,
        sdf_resolution: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize PEGNN collision bridge.
        
        Args:
            pegnn_model: Trained PEGNN-Deform model
            rest_vertices: (V, 3) body mesh rest pose
            faces: (F, 3) body mesh face indices
            sdf_resolution: Resolution for SDF grid (64, 128, or 256)
            device: Target device
        """
        self.device = device or next(pegnn_model.parameters()).device
        self.pegnn_model = pegnn_model.to(self.device)
        self.pegnn_model.eval()
        
        self.rest_vertices = rest_vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.sdf_resolution = sdf_resolution
        
        # Initialize deformable body
        self.body = DeformableBody(
            self.rest_vertices,
            self.faces,
            sdf_resolution=sdf_resolution,
            device=self.device,
        )
        
        # Hidden state for GRU temporal continuity
        self._hidden = None
    
    @torch.no_grad()
    def step(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dt: float = 1.0 / 30.0,
    ) -> DeformableBody:
        """
        Run one PEGNN step and update body SDF.
        
        Args:
            node_features: (V, F_node) node input features
            edge_index: (2, E) edge connectivity
            edge_attr: (E, F_edge) edge attributes (stiffness, rest length)
            dt: Time step for velocity estimation
            
        Returns:
            DeformableBody with updated SDF
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        
        # Run PEGNN forward pass
        # Note: Adjust this based on your actual PEGNN model interface
        if hasattr(self.pegnn_model, 'forward_with_hidden'):
            deformed_vertices, self._hidden = self.pegnn_model.forward_with_hidden(
                node_features, edge_index, edge_attr, hidden=self._hidden
            )
        else:
            # Standard forward - may need to track hidden state separately
            deformed_vertices = self.pegnn_model(node_features, edge_index, edge_attr)
        
        # Update body mesh and SDF
        self.body.update_from_deformation(deformed_vertices, dt=dt)
        
        return self.body
    
    @torch.no_grad()
    def step_from_vertices(
        self,
        deformed_vertices: torch.Tensor,
        dt: float = 1.0 / 30.0,
    ) -> DeformableBody:
        """
        Update body directly from vertex positions (skip PEGNN forward).
        
        Useful when you already have the deformed vertices from elsewhere.
        
        Args:
            deformed_vertices: (V, 3) new vertex positions
            dt: Time step
            
        Returns:
            DeformableBody with updated SDF
        """
        self.body.update_from_deformation(deformed_vertices.to(self.device), dt=dt)
        return self.body
    
    def get_sdf(self) -> SDFField:
        """Get current body SDF for collision queries."""
        return self.body.get_sdf()
    
    def get_body(self) -> DeformableBody:
        """Get the DeformableBody object."""
        return self.body
    
    def get_velocity_field(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Get body surface velocity at query points.
        
        Useful for friction computation in cloth simulation.
        
        Args:
            query_points: (N, 3) query positions
            
        Returns:
            (N, 3) velocity vectors
        """
        return self.body.get_velocity_field(query_points)
    
    def reset(self) -> None:
        """Reset to initial state (rest pose)."""
        self._hidden = None
        self.body.reset()


class BatchedPEGNNCollisionBridge:
    """
    Batched version for processing multiple bodies in parallel.
    
    Useful for training with multiple body configurations.
    """
    
    def __init__(
        self,
        pegnn_model: nn.Module,
        rest_vertices: torch.Tensor,  # (B, V, 3) or (V, 3)
        faces: torch.Tensor,
        sdf_resolution: int = 64,  # Lower res for batched
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(pegnn_model.parameters()).device
        self.pegnn_model = pegnn_model.to(self.device)
        
        # Handle both batched and single rest vertices
        if rest_vertices.dim() == 2:
            rest_vertices = rest_vertices.unsqueeze(0)
        
        self.rest_vertices = rest_vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.batch_size = rest_vertices.shape[0]
        self.sdf_resolution = sdf_resolution
        
        # Create SDFs for each body in batch
        self._sdfs: Optional[list] = None
        self._current_vertices = self.rest_vertices.clone()
    
    @torch.no_grad()
    def step(
        self,
        node_features: torch.Tensor,  # (B, V, F)
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> list:
        """
        Process batch of body deformations.
        
        Returns:
            List of SDFField objects, one per batch element
        """
        # Run PEGNN (batched)
        deformed_vertices = self.pegnn_model(
            node_features.to(self.device),
            edge_index.to(self.device),
            edge_attr.to(self.device),
        )  # (B, V, 3)
        
        self._current_vertices = deformed_vertices
        
        # Build SDFs (this could be parallelized with proper batching)
        self._sdfs = []
        for b in range(self.batch_size):
            sdf = SDFField.from_mesh(
                deformed_vertices[b],
                self.faces,
                resolution=self.sdf_resolution,
                device=self.device,
            )
            self._sdfs.append(sdf)
        
        return self._sdfs
    
    def get_sdfs(self) -> list:
        """Get list of SDFField objects."""
        if self._sdfs is None:
            # Build initial SDFs from rest pose
            self._sdfs = []
            for b in range(self.batch_size):
                sdf = SDFField.from_mesh(
                    self.rest_vertices[b],
                    self.faces,
                    resolution=self.sdf_resolution,
                    device=self.device,
                )
                self._sdfs.append(sdf)
        return self._sdfs


# Convenience function for simple usage
def create_body_sdf_from_pegnn(
    pegnn_model: nn.Module,
    rest_vertices: torch.Tensor,
    faces: torch.Tensor,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    resolution: int = 128,
) -> SDFField:
    """
    One-shot function to get body SDF from PEGNN.
    
    For simple single-frame usage without tracking.
    
    Args:
        pegnn_model: PEGNN-Deform model
        rest_vertices: (V, 3) rest pose
        faces: (F, 3) face indices
        node_features: (V, F) input features
        edge_index: (2, E) edges
        edge_attr: (E, F_edge) edge features
        resolution: SDF grid resolution
        
    Returns:
        SDFField for the deformed body
    """
    device = next(pegnn_model.parameters()).device
    
    with torch.no_grad():
        deformed = pegnn_model(
            node_features.to(device),
            edge_index.to(device),
            edge_attr.to(device),
        )
    
    return SDFField.from_mesh(
        deformed, faces.to(device),
        resolution=resolution,
        device=device,
    )
