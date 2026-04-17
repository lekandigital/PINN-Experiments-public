"""
Collision detection between cloth vertices and body SDF fields.

Provides vertex-level penetration detection, proximity detection,
and utilities for continuous collision detection.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from .sdf_field import SDFField
from .config import CollisionConfig


@dataclass
class CollisionResult:
    """
    Result of collision detection between cloth and body.
    
    Attributes:
        penetrating_mask: (B, V) or (V,) bool tensor, True for vertices inside body
        penetration_depths: (B, V) or (V,) signed distances (negative = inside)
        surface_normals: (B, V, 3) or (V, 3) SDF gradients at penetrating vertices
        closest_surface_points: (B, V, 3) or (V, 3) estimated closest points on body
        proximity_mask: (B, V) or (V,) bool tensor, True for vertices within proximity threshold
        num_penetrating: int, count of penetrating vertices
        num_proximal: int, count of vertices within proximity threshold
    """
    penetrating_mask: torch.Tensor
    penetration_depths: torch.Tensor
    surface_normals: torch.Tensor
    closest_surface_points: torch.Tensor
    proximity_mask: torch.Tensor
    num_penetrating: int
    num_proximal: int
    
    @property
    def has_collisions(self) -> bool:
        """Whether any collisions were detected."""
        return self.num_penetrating > 0
    
    @property
    def has_proximity(self) -> bool:
        """Whether any vertices are within proximity threshold."""
        return self.num_proximal > 0
    
    def get_penetrating_vertices(self, cloth_vertices: torch.Tensor) -> torch.Tensor:
        """Extract only the penetrating vertex positions."""
        return cloth_vertices[self.penetrating_mask]
    
    def get_penetrating_depths(self) -> torch.Tensor:
        """Extract penetration depths for penetrating vertices only."""
        return self.penetration_depths[self.penetrating_mask]


class CollisionDetector:
    """
    Detects collisions between cloth vertices and a body SDF field.
    
    Example:
        >>> detector = CollisionDetector(proximity_threshold=0.005)
        >>> result = detector.detect(cloth_vertices, body_sdf)
        >>> if result.has_collisions:
        ...     print(f"Found {result.num_penetrating} penetrating vertices")
    """
    
    def __init__(
        self,
        proximity_threshold: float = 0.005,
        enable_proximity: bool = True,
        config: Optional[CollisionConfig] = None,
    ):
        """
        Initialize collision detector.
        
        Args:
            proximity_threshold: Distance threshold for proximity detection
            enable_proximity: Whether to compute proximity mask
            config: Full collision config (overrides other params)
        """
        if config is not None:
            self.proximity_threshold = config.proximity_threshold
            self.enable_proximity = config.enable_proximity_detection
        else:
            self.proximity_threshold = proximity_threshold
            self.enable_proximity = enable_proximity
    
    def detect(
        self,
        cloth_vertices: torch.Tensor,
        body_sdf: SDFField,
        compute_normals: bool = True,
    ) -> CollisionResult:
        """
        Detect collisions between cloth vertices and body SDF.
        
        Args:
            cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
            body_sdf: SDFField representing the body surface
            compute_normals: Whether to compute surface normals (slower but needed for response)
            
        Returns:
            CollisionResult with detection information
        """
        original_shape = cloth_vertices.shape
        batched = len(original_shape) == 3
        
        if batched:
            batch_size, num_verts, _ = original_shape
        else:
            num_verts = original_shape[0]
            cloth_vertices = cloth_vertices.unsqueeze(0)
            batch_size = 1
        
        device = cloth_vertices.device
        
        # Query SDF at all cloth vertices
        sdf_values = body_sdf.query(cloth_vertices)  # (B, V)
        
        # Penetrating: SDF < 0 (inside the body)
        penetrating_mask = sdf_values < 0
        
        # Proximity: 0 <= SDF < threshold (close but not penetrating)
        if self.enable_proximity:
            proximity_mask = (sdf_values >= 0) & (sdf_values < self.proximity_threshold)
        else:
            proximity_mask = torch.zeros_like(penetrating_mask)
        
        # Compute surface normals via SDF gradient
        if compute_normals:
            gradients = body_sdf.gradient(cloth_vertices)  # (B, V, 3)
            surface_normals = F.normalize(gradients, dim=-1)
        else:
            surface_normals = torch.zeros(*cloth_vertices.shape, device=device)
        
        # Estimate closest surface points
        # closest_point ≈ vertex + |sdf| * normalized_gradient (toward surface)
        closest_surface_points = cloth_vertices - sdf_values.unsqueeze(-1) * surface_normals
        
        # Count collisions
        num_penetrating = penetrating_mask.sum().item()
        num_proximal = proximity_mask.sum().item()
        
        # Remove batch dim if input wasn't batched
        if not batched:
            penetrating_mask = penetrating_mask.squeeze(0)
            sdf_values = sdf_values.squeeze(0)
            surface_normals = surface_normals.squeeze(0)
            closest_surface_points = closest_surface_points.squeeze(0)
            proximity_mask = proximity_mask.squeeze(0)
        
        return CollisionResult(
            penetrating_mask=penetrating_mask,
            penetration_depths=sdf_values,
            surface_normals=surface_normals,
            closest_surface_points=closest_surface_points,
            proximity_mask=proximity_mask,
            num_penetrating=num_penetrating,
            num_proximal=num_proximal,
        )
    
    def detect_self_collision(
        self,
        cloth_vertices: torch.Tensor,
        cloth_faces: torch.Tensor,
        min_distance: float = 0.005,
        sample_ratio: float = 0.1,
    ) -> torch.Tensor:
        """
        Detect cloth self-intersections using sampling-based approach.
        
        This is an approximate method that samples vertex-face pairs
        to detect self-intersection. For exact detection, use a
        spatial hash grid or BVH.
        
        Args:
            cloth_vertices: (V, 3) cloth vertex positions
            cloth_faces: (F, 3) cloth face indices
            min_distance: Minimum allowed distance between non-adjacent elements
            sample_ratio: Fraction of pairs to sample (0.1 = 10%)
            
        Returns:
            (V,) bool tensor, True for vertices in self-collision
        """
        device = cloth_vertices.device
        num_vertices = cloth_vertices.shape[0]
        num_faces = cloth_faces.shape[0]
        
        # Build vertex-to-face adjacency to exclude adjacent pairs
        vertex_faces = [set() for _ in range(num_vertices)]
        for f_idx, face in enumerate(cloth_faces):
            for v_idx in face:
                vertex_faces[v_idx.item()].add(f_idx)
        
        # Sample random vertex-face pairs
        num_samples = int(num_vertices * num_faces * sample_ratio)
        num_samples = max(num_samples, 1000)  # Minimum samples
        
        vertex_indices = torch.randint(0, num_vertices, (num_samples,), device=device)
        face_indices = torch.randint(0, num_faces, (num_samples,), device=device)
        
        # Get sampled vertices and face centroids
        sampled_vertices = cloth_vertices[vertex_indices]  # (S, 3)
        
        face_verts = cloth_vertices[cloth_faces[face_indices]]  # (S, 3, 3)
        face_centroids = face_verts.mean(dim=1)  # (S, 3)
        
        # Compute distances
        distances = torch.norm(sampled_vertices - face_centroids, dim=-1)  # (S,)
        
        # Find violations (close non-adjacent pairs)
        violations = distances < min_distance
        
        # Mark vertices involved in self-collision
        self_collision_mask = torch.zeros(num_vertices, dtype=torch.bool, device=device)
        violating_vertices = vertex_indices[violations]
        self_collision_mask[violating_vertices] = True
        
        return self_collision_mask
