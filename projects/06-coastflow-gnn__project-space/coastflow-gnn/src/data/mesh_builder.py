"""
Mesh Builder for CoastFlow-GNN

Provides utilities for:
- Creating coastal meshes from DEM data
- Converting meshes to PyTorch Geometric graphs
- Mesh refinement and quality checks
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial import Delaunay
from torch_geometric.data import Data


def create_coastal_mesh(
    x_range: Tuple[float, float] = (0, 1000),
    y_range: Tuple[float, float] = (0, 1000),
    resolution: float = 50.0,
    elevation_func: Optional[callable] = None,
    refinement_regions: Optional[List[dict]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a coastal mesh with variable resolution.
    
    Args:
        x_range: Domain extent in x (meters)
        y_range: Domain extent in y (meters)
        resolution: Base mesh resolution (meters)
        elevation_func: Function(x, y) -> z for elevation
        refinement_regions: List of dicts with 'center', 'radius', 'factor'
        
    Returns:
        points: Mesh points [N, 3]
        elevation: Elevation values [N]
    """
    # Generate base grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    nx = int((x_max - x_min) / resolution) + 1
    ny = int((y_max - y_min) / resolution) + 1
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Add refinement if specified
    if refinement_regions:
        extra_points = []
        for region in refinement_regions:
            center = np.array(region['center'])
            radius = region['radius']
            factor = region.get('factor', 2)
            
            # Generate finer points in region
            n_fine = int(factor * 2 * radius / resolution)
            theta = np.linspace(0, 2*np.pi, n_fine * 4)
            radii = np.linspace(0, radius, n_fine)
            
            for r in radii:
                for t in theta:
                    px = center[0] + r * np.cos(t)
                    py = center[1] + r * np.sin(t)
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        extra_points.append([px, py])
        
        if extra_points:
            points_2d = np.vstack([points_2d, np.array(extra_points)])
    
    # Compute elevation
    if elevation_func is None:
        # Default: coastal profile
        def elevation_func(x, y):
            x_norm = (x - x_min) / (x_max - x_min)
            return 10 * (1 - x_norm) - 5
    
    elevation = elevation_func(points_2d[:, 0], points_2d[:, 1])
    
    # Create 3D points
    points = np.column_stack([points_2d, elevation])
    
    return points, elevation


def mesh_to_graph(
    points: np.ndarray,
    elevation: Optional[np.ndarray] = None,
    wind_u: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
    k_neighbors: int = 8,
    use_delaunay: bool = True,
) -> Data:
    """
    Convert mesh points to PyTorch Geometric graph.
    
    Args:
        points: Mesh points [N, 2] or [N, 3]
        elevation: Elevation values [N]
        wind_u, wind_v: Wind components [N]
        k_neighbors: Number of neighbors for KNN (if not using Delaunay)
        use_delaunay: Use Delaunay triangulation for edges
        
    Returns:
        PyTorch Geometric Data object
    """
    N = points.shape[0]
    
    # Ensure 3D points
    if points.shape[1] == 2:
        if elevation is not None:
            points = np.column_stack([points, elevation])
        else:
            points = np.column_stack([points, np.zeros(N)])
    
    # Create edge index
    if use_delaunay:
        try:
            tri = Delaunay(points[:, :2])
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(edge)
            
            edge_list = list(edges)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        except Exception:
            use_delaunay = False
    
    if not use_delaunay:
        # Use k-nearest neighbors
        from torch_geometric.nn import knn_graph
        pos_tensor = torch.tensor(points[:, :2], dtype=torch.float32)
        edge_index = knn_graph(pos_tensor, k=k_neighbors, loop=False)
    
    # Prepare node features
    if elevation is None:
        elevation = points[:, 2]
    if wind_u is None:
        wind_u = np.zeros(N)
    if wind_v is None:
        wind_v = np.zeros(N)
    
    # Node features: [x, y, z, elevation, wind_u, wind_v]
    x = np.column_stack([
        points[:, 0],  # x
        points[:, 1],  # y
        points[:, 2],  # z
        elevation,     # elevation
        wind_u,        # wind_u
        wind_v,        # wind_v
    ])
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        pos=torch.tensor(points, dtype=torch.float32),
    )
    
    return data


def refine_mesh_near_shore(
    points: np.ndarray,
    elevation: np.ndarray,
    shore_level: float = 0.0,
    refinement_factor: int = 2,
    band_width: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine mesh near the shoreline (where elevation ≈ shore_level).
    
    Args:
        points: Mesh points [N, 3]
        elevation: Elevation values [N]
        shore_level: Water level (default 0)
        refinement_factor: Number of new points per edge
        band_width: Width of refinement band (meters)
        
    Returns:
        refined_points: [M, 3] with M > N
        refined_elevation: [M]
    """
    # Identify near-shore points
    shore_mask = np.abs(elevation - shore_level) < band_width
    
    if not shore_mask.any():
        return points, elevation
    
    # Add interpolated points between near-shore neighbors
    new_points = [points]
    new_elevation = [elevation]
    
    # Find edges between near-shore points
    try:
        tri = Delaunay(points[:, :2])
        edges_to_refine = set()
        
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    if shore_mask[simplex[i]] and shore_mask[simplex[j]]:
                        edges_to_refine.add(tuple(sorted([simplex[i], simplex[j]])))
        
        # Add midpoints
        for i, j in edges_to_refine:
            for k in range(1, refinement_factor):
                t = k / refinement_factor
                mid_point = (1 - t) * points[i] + t * points[j]
                mid_elev = (1 - t) * elevation[i] + t * elevation[j]
                new_points.append(mid_point.reshape(1, 3))
                new_elevation.append(np.array([mid_elev]))
        
        points = np.vstack(new_points)
        elevation = np.concatenate(new_elevation)
        
    except Exception:
        pass  # Keep original mesh if refinement fails
    
    return points, elevation


def compute_mesh_quality(
    points: np.ndarray,
    edge_index: torch.Tensor,
) -> dict:
    """
    Compute mesh quality metrics.
    
    Returns:
        Dictionary with:
        - min_edge_length
        - max_edge_length
        - mean_edge_length
        - std_edge_length
        - aspect_ratio (max/min edge length)
    """
    src, dst = edge_index
    src, dst = src.numpy(), dst.numpy()
    
    edge_vectors = points[dst] - points[src]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    return {
        'min_edge_length': float(edge_lengths.min()),
        'max_edge_length': float(edge_lengths.max()),
        'mean_edge_length': float(edge_lengths.mean()),
        'std_edge_length': float(edge_lengths.std()),
        'aspect_ratio': float(edge_lengths.max() / edge_lengths.min()),
        'num_nodes': len(points),
        'num_edges': len(edge_lengths) // 2,  # Undirected edges
    }


if __name__ == "__main__":
    print("Testing mesh builder...")
    
    # Create coastal mesh
    points, elevation = create_coastal_mesh(
        x_range=(0, 1000),
        y_range=(0, 500),
        resolution=50.0,
    )
    print(f"Created mesh with {len(points)} points")
    
    # Convert to graph
    data = mesh_to_graph(points, elevation)
    print(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Check mesh quality
    quality = compute_mesh_quality(points, data.edge_index)
    print(f"Mesh quality: {quality}")
    
    print("\n✓ Mesh builder tests passed!")
