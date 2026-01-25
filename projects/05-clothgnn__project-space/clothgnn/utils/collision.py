"""
Collision detection using voxelized occupancy grids.
Integrates obstacle detection with graph node features.
"""
import torch
import torch.nn as nn


class OccupancyGrid(nn.Module):
    """
    Generates 3D occupancy grid for collision detection.
    Fuses occupancy features with graph node features.
    
    Args:
        grid_dim: Resolution of the voxel grid (default: 32)
        bounds: Tuple of (x, y, z) bounds as ((min, max), (min, max), (min, max))
    """
    def __init__(self, grid_dim=32, bounds=((-1, 1), (-1, 2), (-1, 1))):
        super().__init__()
        self.grid_dim = grid_dim
        self.bounds = bounds
        
        # Precompute voxel centers for efficient SDF queries
        x = torch.linspace(bounds[0][0], bounds[0][1], grid_dim)
        y = torch.linspace(bounds[1][0], bounds[1][1], grid_dim)
        z = torch.linspace(bounds[2][0], bounds[2][1], grid_dim)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        self.register_buffer("voxel_centers", torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3))
        
        # Voxel size for each dimension
        self.voxel_size = torch.tensor([
            (bounds[0][1] - bounds[0][0]) / grid_dim,
            (bounds[1][1] - bounds[1][0]) / grid_dim,
            (bounds[2][1] - bounds[2][0]) / grid_dim
        ])
        
    def forward(self, node_positions, obstacle_sdf_fn):
        """
        Compute per-node occupancy features based on obstacle SDF.
        
        Args:
            node_positions: (N, 3) tensor of node positions
            obstacle_sdf_fn: Function that takes (N, 3) points and returns (N,) SDF values
        
        Returns:
            occupancy: (N, 1) binary tensor indicating collision proximity
        """
        # Query SDF at node positions
        sdf_values = obstacle_sdf_fn(node_positions)
        
        # Nodes with SDF < threshold are in collision
        threshold = 0.05  # 5cm collision margin
        occupancy = (sdf_values < threshold).float().unsqueeze(-1)
        
        return occupancy
    
    def compute_grid_occupancy(self, obstacle_sdf_fn):
        """
        Compute full 3D occupancy grid from obstacle SDF.
        
        Args:
            obstacle_sdf_fn: SDF function for the obstacle
            
        Returns:
            grid: (grid_dim, grid_dim, grid_dim) binary occupancy grid
        """
        sdf_values = obstacle_sdf_fn(self.voxel_centers)
        grid = (sdf_values < 0).float().reshape(self.grid_dim, self.grid_dim, self.grid_dim)
        return grid
    
    def get_node_voxel_indices(self, node_positions):
        """
        Map node positions to voxel grid indices.
        
        Args:
            node_positions: (N, 3) tensor of positions
            
        Returns:
            indices: (N, 3) tensor of integer voxel indices
        """
        # Normalize positions to [0, 1] within bounds
        mins = torch.tensor([b[0] for b in self.bounds], device=node_positions.device)
        maxs = torch.tensor([b[1] for b in self.bounds], device=node_positions.device)
        normalized = (node_positions - mins) / (maxs - mins)
        
        # Convert to grid indices
        indices = (normalized * self.grid_dim).long().clamp(0, self.grid_dim - 1)
        return indices


def sphere_sdf(points, center=None, radius=0.3):
    """
    Compute signed distance field for a sphere obstacle.
    
    Args:
        points: (N, 3) tensor of query points
        center: (3,) sphere center (default: [0.0, 0.5, 0.0])
        radius: Sphere radius
        
    Returns:
        sdf: (N,) signed distance values (negative inside, positive outside)
    """
    if center is None:
        center = torch.tensor([0.0, 0.5, 0.0], device=points.device)
    else:
        center = center.to(points.device)
    
    distances = torch.norm(points - center, dim=1)
    return distances - radius


def box_sdf(points, center=None, half_extents=None):
    """
    Compute signed distance field for an axis-aligned box obstacle.
    
    Args:
        points: (N, 3) tensor of query points
        center: (3,) box center
        half_extents: (3,) half-sizes in each dimension
        
    Returns:
        sdf: (N,) signed distance values
    """
    if center is None:
        center = torch.tensor([0.0, 0.5, 0.0], device=points.device)
    else:
        center = center.to(points.device)
        
    if half_extents is None:
        half_extents = torch.tensor([0.2, 0.2, 0.2], device=points.device)
    else:
        half_extents = half_extents.to(points.device)
    
    # Transform to box local space
    q = torch.abs(points - center) - half_extents
    
    # Distance computation
    outside_dist = torch.norm(torch.clamp(q, min=0), dim=1)
    inside_dist = torch.clamp(q.max(dim=1).values, max=0)
    
    return outside_dist + inside_dist


def ground_sdf(points, height=0.0):
    """
    Compute signed distance field for a ground plane.
    
    Args:
        points: (N, 3) tensor of query points
        height: Y-coordinate of the ground plane
        
    Returns:
        sdf: (N,) signed distance values (negative below ground)
    """
    return points[:, 1] - height
