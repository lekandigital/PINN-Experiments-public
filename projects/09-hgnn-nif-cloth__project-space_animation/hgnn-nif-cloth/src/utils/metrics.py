"""
Utility Functions: Metrics and Visualization

Provides:
- Chamfer distance computation (PyTorch)
- Grid point generation for SDF queries
- Mesh extraction utilities
"""

import torch
import numpy as np
from typing import Tuple, Optional


def chamfer_distance(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    reduce: str = 'mean'
) -> torch.Tensor:
    """
    Compute Chamfer distance between two point sets.
    
    Chamfer distance is the sum of nearest-neighbor distances in both directions:
    CD(A, B) = (1/|A|) Σ min_b ||a - b|| + (1/|B|) Σ min_a ||a - b||
    
    Args:
        points_a: First point set (N, 3) or (B, N, 3)
        points_b: Second point set (M, 3) or (B, M, 3)
        reduce: 'mean', 'sum', or 'none'
        
    Returns:
        Chamfer distance (scalar or per-batch tensor)
    """
    batched = points_a.dim() == 3
    if not batched:
        points_a = points_a.unsqueeze(0)
        points_b = points_b.unsqueeze(0)
        
    B, N, _ = points_a.shape
    _, M, _ = points_b.shape
    
    # Compute pairwise distances: (B, N, M)
    # Using efficient computation: ||a-b||² = ||a||² + ||b||² - 2<a,b>
    a_sq = (points_a ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    b_sq = (points_b ** 2).sum(dim=-1, keepdim=True)  # (B, M, 1)
    
    dists_sq = a_sq + b_sq.transpose(1, 2) - 2 * torch.bmm(points_a, points_b.transpose(1, 2))
    dists_sq = dists_sq.clamp(min=0)  # Numerical stability
    
    # Nearest neighbor in each direction
    nn_a_to_b = dists_sq.min(dim=2).values.sqrt()  # (B, N)
    nn_b_to_a = dists_sq.min(dim=1).values.sqrt()  # (B, M)
    
    # Chamfer distance per sample
    cd_per_sample = nn_a_to_b.mean(dim=1) + nn_b_to_a.mean(dim=1)  # (B,)
    
    if not batched:
        cd_per_sample = cd_per_sample.squeeze(0)
        
    if reduce == 'mean':
        return cd_per_sample.mean()
    elif reduce == 'sum':
        return cd_per_sample.sum()
    return cd_per_sample


def chamfer_distance_numpy(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> float:
    """
    Compute Chamfer distance using NumPy (for evaluation without PyTorch).
    
    Args:
        points_a: First point set (N, 3)
        points_b: Second point set (M, 3)
        
    Returns:
        Chamfer distance scalar
    """
    # A to B
    dists_a_to_b = np.linalg.norm(
        points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :],
        axis=-1
    )
    min_a_to_b = dists_a_to_b.min(axis=1).mean()
    
    # B to A
    min_b_to_a = dists_a_to_b.min(axis=0).mean()
    
    return float(min_a_to_b + min_b_to_a)


def generate_grid_points(
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    resolution: int = 32,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate 3D grid points for SDF queries.
    
    Args:
        bounds_min: Minimum bounds (3,)
        bounds_max: Maximum bounds (3,)
        resolution: Grid resolution per axis
        device: Device to create tensor on
        
    Returns:
        Grid points of shape (resolution³, 3)
    """
    # Create 1D linspaces
    x = torch.linspace(bounds_min[0], bounds_max[0], resolution, device=device)
    y = torch.linspace(bounds_min[1], bounds_max[1], resolution, device=device)
    z = torch.linspace(bounds_min[2], bounds_max[2], resolution, device=device)
    
    # Create 3D meshgrid
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Stack and flatten
    grid = torch.stack([xx, yy, zz], dim=-1)  # (R, R, R, 3)
    points = grid.reshape(-1, 3)  # (R³, 3)
    
    return points


def generate_grid_points_from_mesh(
    vertices: torch.Tensor,
    padding: float = 0.2,
    resolution: int = 32,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate grid points based on mesh bounding box.
    
    Args:
        vertices: Mesh vertices (N, 3)
        padding: Padding around bounding box
        resolution: Grid resolution per axis
        device: Device for tensors
        
    Returns:
        points: Grid points (R³, 3)
        bounds_min: Minimum bounds (3,)
        bounds_max: Maximum bounds (3,)
    """
    bounds_min = vertices.min(dim=0).values - padding
    bounds_max = vertices.max(dim=0).values + padding
    
    points = generate_grid_points(bounds_min, bounds_max, resolution, device)
    
    return points, bounds_min, bounds_max


def extract_mesh_from_sdf(
    sdf_volume: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from SDF using marching cubes.
    
    Requires scikit-image.
    
    Args:
        sdf_volume: SDF values (R, R, R)
        bounds_min: Minimum bounds (3,)
        bounds_max: Maximum bounds (3,)
        level: Iso-level to extract (0 for surface)
        
    Returns:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (F, 3)
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image required for marching cubes. Install with: pip install scikit-image")
        
    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        sdf_volume,
        level=level,
        spacing=(
            (bounds_max[0] - bounds_min[0]) / sdf_volume.shape[0],
            (bounds_max[1] - bounds_min[1]) / sdf_volume.shape[1],
            (bounds_max[2] - bounds_min[2]) / sdf_volume.shape[2],
        )
    )
    
    # Offset vertices to world coordinates
    verts = verts + bounds_min
    
    return verts, faces


def sdf_to_point_cloud(
    sdf_volume: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.01,
    max_points: int = 5000
) -> torch.Tensor:
    """
    Extract surface points from SDF by thresholding.
    
    A simple alternative to marching cubes.
    
    Args:
        sdf_volume: SDF values (R, R, R)
        bounds_min: Minimum bounds (3,)
        bounds_max: Maximum bounds (3,)
        threshold: Distance threshold for surface points
        max_points: Maximum number of points to return
        
    Returns:
        Surface points (M, 3)
    """
    R = sdf_volume.shape[0]
    device = sdf_volume.device
    
    # Generate grid coordinates
    grid = generate_grid_points(bounds_min, bounds_max, R, device)
    sdf_flat = sdf_volume.reshape(-1)
    
    # Find near-surface points
    near_surface = torch.abs(sdf_flat) < threshold
    surface_points = grid[near_surface]
    
    # Subsample if too many
    if len(surface_points) > max_points:
        indices = torch.randperm(len(surface_points))[:max_points]
        surface_points = surface_points[indices]
        
    return surface_points


def compute_mesh_stats(vertices: torch.Tensor, edges: torch.Tensor) -> dict:
    """
    Compute statistics about a mesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        edges: Edge indices (E, 2) or (2, E)
        
    Returns:
        Dictionary of mesh statistics
    """
    if edges.shape[0] == 2:
        edges = edges.T
        
    # Bounding box
    bounds_min = vertices.min(dim=0).values
    bounds_max = vertices.max(dim=0).values
    bbox_size = bounds_max - bounds_min
    
    # Edge lengths
    src, tgt = edges[:, 0], edges[:, 1]
    edge_vecs = vertices[src] - vertices[tgt]
    edge_lengths = torch.norm(edge_vecs, dim=-1)
    
    return {
        'num_vertices': len(vertices),
        'num_edges': len(edges),
        'bounds_min': bounds_min.tolist(),
        'bounds_max': bounds_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'edge_length_mean': edge_lengths.mean().item(),
        'edge_length_std': edge_lengths.std().item(),
        'edge_length_min': edge_lengths.min().item(),
        'edge_length_max': edge_lengths.max().item(),
    }


class Timer:
    """Simple timer for benchmarking."""
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.times = []
        
    def start(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.t0:
            self.t0.record()
        else:
            import time
            self._start = time.time()
            
    def stop(self) -> float:
        if self.t1:
            self.t1.record()
            torch.cuda.synchronize()
            elapsed = self.t0.elapsed_time(self.t1) / 1000  # ms to seconds
        else:
            import time
            elapsed = time.time() - self._start
        self.times.append(elapsed)
        return elapsed
    
    @property
    def mean(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    @property
    def fps(self) -> float:
        return 1.0 / self.mean if self.mean > 0 else 0.0
