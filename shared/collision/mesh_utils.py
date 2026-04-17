"""
Mesh processing utilities: marching cubes extraction, smoothing, etc.

Extracted and generalized from Project 04 (ClothGeom-NIF).
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import numpy as np


def marching_cubes_mesh(
    sdf_grid: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    iso_level: float = 0.0,
    return_numpy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract triangle mesh from SDF grid using marching cubes.
    
    Args:
        sdf_grid: (D, H, W) SDF values on regular grid
        bbox_min: (3,) minimum corner of bounding box
        bbox_max: (3,) maximum corner of bounding box
        iso_level: Isosurface level (0 for surface)
        return_numpy: Whether to return numpy arrays instead of tensors
        
    Returns:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle face indices
        
    Note:
        Requires scikit-image for marching_cubes implementation.
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            "scikit-image is required for marching cubes. "
            "Install with: pip install scikit-image"
        )
    
    device = sdf_grid.device
    sdf_np = sdf_grid.cpu().numpy()
    
    # Pad to handle boundary cases
    sdf_padded = np.pad(sdf_np, pad_width=1, mode='constant', constant_values=1.0)
    
    # Run marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            sdf_padded,
            level=iso_level,
            spacing=(1.0, 1.0, 1.0),
            gradient_direction='descent',
        )
    except ValueError:
        # No surface found (all values same sign)
        if return_numpy:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)
    
    # Adjust for padding
    vertices = vertices - 1.0
    
    # Scale to world coordinates
    resolution = sdf_grid.shape[0]
    bbox_min_np = bbox_min.cpu().numpy()
    bbox_max_np = bbox_max.cpu().numpy()
    
    vertices = vertices / (resolution - 1)  # [0, 1]
    vertices = vertices * (bbox_max_np - bbox_min_np) + bbox_min_np
    
    if return_numpy:
        return vertices, faces
    
    return (
        torch.from_numpy(vertices).float().to(device),
        torch.from_numpy(faces.copy()).long().to(device),
    )


def extract_mesh_multires(
    sdf_fn,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    resolutions: Tuple[int, ...] = (64, 128, 256),
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract mesh with progressive resolution refinement.
    
    Starts at low resolution for fast preview, then refines.
    
    Args:
        sdf_fn: Callable that takes (N, 3) points and returns (N,) SDF values
        bbox_min: (3,) minimum corner
        bbox_max: (3,) maximum corner
        resolutions: Tuple of resolutions to use (low to high)
        device: Target device
        
    Returns:
        vertices: (V, 3) mesh vertices at highest resolution
        faces: (F, 3) triangle faces
    """
    device = device or bbox_min.device
    final_vertices, final_faces = None, None
    
    for res in resolutions:
        # Create grid
        lin = torch.linspace(0, 1, res, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        grid_points = grid_points * (bbox_max - bbox_min) + bbox_min
        grid_points_flat = grid_points.reshape(-1, 3)
        
        # Query SDF
        with torch.no_grad():
            sdf_values = sdf_fn(grid_points_flat)
        sdf_grid = sdf_values.reshape(res, res, res)
        
        # Extract mesh
        final_vertices, final_faces = marching_cubes_mesh(
            sdf_grid, bbox_min, bbox_max
        )
        
        if final_vertices.shape[0] > 0:
            # Successfully extracted mesh
            pass
    
    return final_vertices, final_faces


def laplacian_smooth(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    iterations: int = 3,
    lambda_factor: float = 0.5,
) -> torch.Tensor:
    """
    Apply Laplacian smoothing to mesh vertices.
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle faces
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing strength [0, 1]
        
    Returns:
        (V, 3) smoothed vertices
    """
    device = vertices.device
    num_vertices = vertices.shape[0]
    
    # Build adjacency from faces
    edges = set()
    for face in faces:
        for i in range(3):
            v0, v1 = face[i].item(), face[(i + 1) % 3].item()
            edges.add((min(v0, v1), max(v0, v1)))
    
    # Build neighbor lists
    neighbors = [[] for _ in range(num_vertices)]
    for v0, v1 in edges:
        neighbors[v0].append(v1)
        neighbors[v1].append(v0)
    
    smoothed = vertices.clone()
    
    for _ in range(iterations):
        new_positions = smoothed.clone()
        
        for v_idx in range(num_vertices):
            if len(neighbors[v_idx]) > 0:
                neighbor_positions = smoothed[neighbors[v_idx]]
                centroid = neighbor_positions.mean(dim=0)
                new_positions[v_idx] = (
                    (1 - lambda_factor) * smoothed[v_idx] +
                    lambda_factor * centroid
                )
        
        smoothed = new_positions
    
    return smoothed


def compute_mesh_sdf_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int = 128,
    padding: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SDF grid from mesh (convenience wrapper).
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle faces
        resolution: Grid resolution
        padding: Bounding box padding
        
    Returns:
        sdf_grid: (res, res, res) SDF values
        bbox_min: (3,) minimum corner
        bbox_max: (3,) maximum corner
    """
    from .sdf_field import SDFField
    
    sdf = SDFField.from_mesh(vertices, faces, resolution=resolution, padding=padding)
    return sdf.grid, sdf.bbox[0], sdf.bbox[1]


def compute_face_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-face normals.
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle faces
        
    Returns:
        (F, 3) unit face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    e0 = v1 - v0
    e1 = v2 - v0
    
    normals = torch.cross(e0, e1, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    
    return normals


def compute_vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-vertex normals (area-weighted average of adjacent face normals).
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle faces
        
    Returns:
        (V, 3) unit vertex normals
    """
    device = vertices.device
    num_vertices = vertices.shape[0]
    
    face_normals = compute_face_normals(vertices, faces)
    
    # Accumulate face normals at vertices
    vertex_normals = torch.zeros(num_vertices, 3, device=device)
    
    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], face_normals)
    
    # Normalize
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
    
    return vertex_normals
