"""
Signed Distance Field representation and queries.

Provides GPU-accelerated SDF computation from triangle meshes with
differentiable point queries for training collision losses.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .config import SDFConfig


@dataclass
class SDFFieldData:
    """Internal data structure for SDF field storage."""
    grid: torch.Tensor  # (res, res, res) SDF values
    bbox_min: torch.Tensor  # (3,) minimum corner
    bbox_max: torch.Tensor  # (3,) maximum corner
    resolution: int
    device: torch.device


class SDFField:
    """
    Signed distance field computed from a triangle mesh.
    
    Supports both grid-based (voxel) and query-based (arbitrary point) evaluation.
    All operations are GPU-accelerated and differentiable for training.
    
    Example:
        >>> vertices = torch.randn(1000, 3)  # mesh vertices
        >>> faces = torch.randint(0, 1000, (2000, 3))  # triangle indices
        >>> sdf = SDFField.from_mesh(vertices, faces, resolution=128)
        >>> query_points = torch.randn(500, 3)
        >>> distances = sdf.query(query_points)  # (500,) signed distances
        >>> gradients = sdf.gradient(query_points)  # (500, 3) gradient vectors
    """
    
    def __init__(self, data: SDFFieldData):
        """Initialize from precomputed SDF data. Use from_mesh() for construction."""
        self._data = data
    
    @classmethod
    def from_mesh(
        cls,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        resolution: int = 128,
        padding: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> 'SDFField':
        """
        Construct SDF field from a triangle mesh.
        
        Args:
            vertices: (V, 3) mesh vertex positions
            faces: (F, 3) triangle face indices
            resolution: Grid resolution (64, 128, or 256)
            padding: Padding around mesh bbox as fraction of size
            device: Target device (default: same as vertices)
            
        Returns:
            SDFField instance ready for queries
        """
        device = device or vertices.device
        vertices = vertices.to(device)
        faces = faces.to(device)
        
        # Compute bounding box with padding
        bbox_min = vertices.min(dim=0).values
        bbox_max = vertices.max(dim=0).values
        bbox_size = bbox_max - bbox_min
        bbox_min = bbox_min - padding * bbox_size
        bbox_max = bbox_max + padding * bbox_size
        
        # Create grid coordinates
        grid = cls._compute_sdf_grid(
            vertices, faces, bbox_min, bbox_max, resolution, device
        )
        
        data = SDFFieldData(
            grid=grid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=resolution,
            device=device,
        )
        return cls(data)
    
    @classmethod
    def _compute_sdf_grid(
        cls,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        resolution: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute SDF values on a regular 3D grid.
        
        Uses angle-weighted pseudonormal method for robust sign computation,
        which handles non-watertight meshes better than ray casting.
        """
        # Create grid sample points
        lin = torch.linspace(0, 1, resolution, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (res, res, res, 3)
        grid_points = grid_points * (bbox_max - bbox_min) + bbox_min
        grid_points_flat = grid_points.reshape(-1, 3)  # (res^3, 3)
        
        # Get triangle data
        v0 = vertices[faces[:, 0]]  # (F, 3)
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Compute unsigned distances and closest points
        # Process in batches to manage memory
        batch_size = 100000
        n_points = grid_points_flat.shape[0]
        unsigned_distances = torch.zeros(n_points, device=device)
        closest_face_idx = torch.zeros(n_points, dtype=torch.long, device=device)
        closest_points = torch.zeros(n_points, 3, device=device)
        
        for i in range(0, n_points, batch_size):
            end_i = min(i + batch_size, n_points)
            batch_points = grid_points_flat[i:end_i]
            
            # Compute distance to each triangle
            dist_batch, closest_batch, face_idx_batch = cls._point_to_mesh_distance(
                batch_points, v0, v1, v2
            )
            
            unsigned_distances[i:end_i] = dist_batch
            closest_points[i:end_i] = closest_batch
            closest_face_idx[i:end_i] = face_idx_batch
        
        # Compute signs using angle-weighted pseudonormal method
        signs = cls._compute_signs_pseudonormal(
            grid_points_flat, closest_points, closest_face_idx,
            vertices, faces, v0, v1, v2
        )
        
        # Combine into signed distance
        sdf_flat = unsigned_distances * signs
        sdf_grid = sdf_flat.reshape(resolution, resolution, resolution)
        
        return sdf_grid
    
    @staticmethod
    def _point_to_mesh_distance(
        points: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute unsigned distance from points to triangles.
        
        Args:
            points: (N, 3) query points
            v0, v1, v2: (F, 3) triangle vertices
            
        Returns:
            distances: (N,) unsigned distances to closest triangle
            closest_points: (N, 3) closest points on mesh
            closest_face: (N,) index of closest face
        """
        n_points = points.shape[0]
        n_faces = v0.shape[0]
        
        # For memory efficiency, compute distances in chunks over faces
        # if there are many faces
        if n_faces > 10000:
            return SDFField._point_to_mesh_distance_chunked(points, v0, v1, v2)
        
        # Expand for broadcasting: (N, F, 3)
        p = points[:, None, :]  # (N, 1, 3)
        
        # Triangle edges
        e0 = v1 - v0  # (F, 3)
        e1 = v2 - v0
        
        # Vector from v0 to point
        v0p = p - v0[None, :, :]  # (N, F, 3)
        
        # Compute barycentric coordinates via dot products
        d00 = (e0 * e0).sum(dim=-1)  # (F,)
        d01 = (e0 * e1).sum(dim=-1)
        d11 = (e1 * e1).sum(dim=-1)
        d20 = (v0p * e0[None, :, :]).sum(dim=-1)  # (N, F)
        d21 = (v0p * e1[None, :, :]).sum(dim=-1)
        
        denom = d00 * d11 - d01 * d01 + 1e-10
        
        # Barycentric coordinates
        u = (d11 * d20 - d01 * d21) / denom  # (N, F)
        v = (d00 * d21 - d01 * d20) / denom
        
        # Clamp to triangle
        u = u.clamp(0, 1)
        v = v.clamp(0, 1)
        uv_sum = u + v
        mask = uv_sum > 1
        u = torch.where(mask, u / uv_sum, u)
        v = torch.where(mask, v / uv_sum, v)
        
        # Closest point on each triangle
        closest_on_tri = v0[None, :, :] + u[:, :, None] * e0[None, :, :] + v[:, :, None] * e1[None, :, :]  # (N, F, 3)
        
        # Distance to each triangle
        dist_to_tri = torch.norm(p - closest_on_tri, dim=-1)  # (N, F)
        
        # Find closest triangle per point
        min_dist, closest_face = dist_to_tri.min(dim=1)  # (N,), (N,)
        
        # Get actual closest points
        batch_idx = torch.arange(n_points, device=points.device)
        closest_points = closest_on_tri[batch_idx, closest_face]  # (N, 3)
        
        return min_dist, closest_points, closest_face
    
    @staticmethod
    def _point_to_mesh_distance_chunked(
        points: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        face_chunk_size: int = 5000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-efficient version that processes faces in chunks."""
        n_points = points.shape[0]
        n_faces = v0.shape[0]
        device = points.device
        
        min_dist = torch.full((n_points,), float('inf'), device=device)
        closest_face = torch.zeros(n_points, dtype=torch.long, device=device)
        closest_points = torch.zeros(n_points, 3, device=device)
        
        for f_start in range(0, n_faces, face_chunk_size):
            f_end = min(f_start + face_chunk_size, n_faces)
            
            v0_chunk = v0[f_start:f_end]
            v1_chunk = v1[f_start:f_end]
            v2_chunk = v2[f_start:f_end]
            
            # Same computation as above but for chunk
            p = points[:, None, :]
            e0 = v1_chunk - v0_chunk
            e1 = v2_chunk - v0_chunk
            v0p = p - v0_chunk[None, :, :]
            
            d00 = (e0 * e0).sum(dim=-1)
            d01 = (e0 * e1).sum(dim=-1)
            d11 = (e1 * e1).sum(dim=-1)
            d20 = (v0p * e0[None, :, :]).sum(dim=-1)
            d21 = (v0p * e1[None, :, :]).sum(dim=-1)
            
            denom = d00 * d11 - d01 * d01 + 1e-10
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            
            u = u.clamp(0, 1)
            v = v.clamp(0, 1)
            uv_sum = u + v
            mask = uv_sum > 1
            u = torch.where(mask, u / uv_sum, u)
            v = torch.where(mask, v / uv_sum, v)
            
            closest_on_tri = v0_chunk[None, :, :] + u[:, :, None] * e0[None, :, :] + v[:, :, None] * e1[None, :, :]
            dist_to_tri = torch.norm(p - closest_on_tri, dim=-1)
            
            chunk_min_dist, chunk_closest_idx = dist_to_tri.min(dim=1)
            
            # Update global minimums
            update_mask = chunk_min_dist < min_dist
            min_dist = torch.where(update_mask, chunk_min_dist, min_dist)
            closest_face = torch.where(update_mask, chunk_closest_idx + f_start, closest_face)
            
            batch_idx = torch.arange(n_points, device=device)
            chunk_closest_pts = closest_on_tri[batch_idx, chunk_closest_idx]
            closest_points = torch.where(update_mask[:, None], chunk_closest_pts, closest_points)
        
        return min_dist, closest_points, closest_face
    
    @staticmethod
    def _compute_signs_pseudonormal(
        query_points: torch.Tensor,
        closest_points: torch.Tensor,
        closest_face_idx: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute signs using angle-weighted pseudonormal method.
        
        This method is more robust than ray casting for non-watertight meshes.
        Sign is determined by whether the query point is on the positive or
        negative side of the closest triangle's plane.
        """
        n_points = query_points.shape[0]
        device = query_points.device
        
        # Get the closest face vertices
        closest_v0 = v0[closest_face_idx]  # (N, 3)
        closest_v1 = v1[closest_face_idx]
        closest_v2 = v2[closest_face_idx]
        
        # Compute face normals
        e0 = closest_v1 - closest_v0
        e1 = closest_v2 - closest_v0
        face_normals = torch.cross(e0, e1, dim=-1)  # (N, 3)
        face_normals = F.normalize(face_normals, dim=-1)
        
        # Vector from closest point to query point
        to_query = query_points - closest_points  # (N, 3)
        
        # Sign is determined by dot product with face normal
        # Positive = outside (same side as normal), Negative = inside
        dot_product = (to_query * face_normals).sum(dim=-1)  # (N,)
        
        signs = torch.sign(dot_product)
        # Handle points exactly on surface (dot_product ≈ 0)
        signs = torch.where(torch.abs(dot_product) < 1e-8, torch.ones_like(signs), signs)
        
        return signs
    
    def query(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query SDF values at arbitrary points via trilinear interpolation.
        
        Args:
            points: (N, 3) or (B, N, 3) query points in world coordinates
            
        Returns:
            (N,) or (B, N) signed distance values
            Negative = inside mesh, Positive = outside, Zero = on surface
        """
        original_shape = points.shape
        batched = len(original_shape) == 3
        
        if batched:
            batch_size, n_points, _ = original_shape
            points = points.reshape(-1, 3)
        
        points = points.to(self._data.device)
        
        # Normalize to [0, 1] grid coordinates
        normalized = (points - self._data.bbox_min) / (self._data.bbox_max - self._data.bbox_min)
        
        # Clamp to valid range (points outside bbox get boundary SDF values)
        normalized = normalized.clamp(0, 1)
        
        # Convert to grid sample format: expects (B, D, H, W, 3) input with coords in [-1, 1]
        grid_coords = normalized * 2 - 1  # [0,1] -> [-1,1]
        grid_coords = grid_coords[None, None, None, :, :]  # (1, 1, 1, N, 3)
        
        # Permute SDF grid to (1, 1, D, H, W) for grid_sample
        sdf_grid = self._data.grid[None, None, :, :, :]
        
        # Sample with trilinear interpolation
        sampled = F.grid_sample(
            sdf_grid,
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        
        # Extract values: (1, 1, 1, 1, N) -> (N,)
        result = sampled.squeeze()
        
        if batched:
            result = result.reshape(batch_size, n_points)
        
        return result
    
    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF gradient at query points.
        
        The gradient points in the direction of fastest increase (outward from surface).
        Normalized gradients approximate surface normals.
        
        Args:
            points: (N, 3) or (B, N, 3) query points
            
        Returns:
            (N, 3) or (B, N, 3) gradient vectors
        """
        points = points.clone().requires_grad_(True)
        sdf_values = self.query(points)
        
        # Compute gradient via autograd
        grad_outputs = torch.ones_like(sdf_values)
        gradients = torch.autograd.grad(
            outputs=sdf_values,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return gradients
    
    def update(self, vertices: torch.Tensor) -> None:
        """
        Update SDF from deformed mesh vertices (same topology).
        
        This recomputes the SDF grid. For real-time performance,
        consider using a lower resolution or the DeformableBody class
        which provides optimized updates.
        
        Args:
            vertices: (V, 3) new vertex positions
        """
        # Note: faces are not stored in SDFField, so we need to create
        # a new field. This method is a convenience for the DeformableBody
        # class which maintains face information.
        raise NotImplementedError(
            "SDFField.update() requires face information. "
            "Use DeformableBody for deforming meshes, or create a new SDFField."
        )
    
    @property
    def resolution(self) -> int:
        """Grid resolution."""
        return self._data.resolution
    
    @property
    def device(self) -> torch.device:
        """Device where SDF is stored."""
        return self._data.device
    
    @property
    def bbox(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bounding box (min, max) corners."""
        return self._data.bbox_min, self._data.bbox_max
    
    @property
    def grid(self) -> torch.Tensor:
        """Raw SDF grid tensor (resolution, resolution, resolution)."""
        return self._data.grid
    
    def to(self, device: torch.device) -> 'SDFField':
        """Move SDF field to device."""
        new_data = SDFFieldData(
            grid=self._data.grid.to(device),
            bbox_min=self._data.bbox_min.to(device),
            bbox_max=self._data.bbox_max.to(device),
            resolution=self._data.resolution,
            device=device,
        )
        return SDFField(new_data)
