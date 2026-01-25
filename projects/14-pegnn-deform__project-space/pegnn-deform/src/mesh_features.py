"""
PEGNN-Deform: Geometric Feature Extraction Module

Computes node and edge features for mesh graphs:
- Laplacian coordinates: node position relative to neighbor average
- Cotangent weights: discrete Laplace-Beltrami edge weights
- Principal curvatures: k1, k2 from discrete Gaussian and mean curvature

Author: PEGNN-Deform Team
"""

import torch
import math
from typing import Tuple, Optional


def compute_mesh_features(
    pos: torch.Tensor,
    faces: torch.Tensor,
    compute_curvatures: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-vertex and per-edge geometric features for a 3D triangle mesh.
    
    This is a vectorized implementation for efficiency on GPU.
    
    Args:
        pos: Vertex coordinates [N, 3]
        faces: Triangle indices [F, 3], 0-indexed
        compute_curvatures: Whether to compute principal curvatures (slower)
        
    Returns:
        node_features: [N, 5] - Laplacian coords (3) + principal curvatures (2)
        edge_index: [2, E] - Directed edge indices for PyG
        edge_attr: [E, 1] - Cotangent edge weights
    """
    N = pos.size(0)
    F = faces.size(0)
    device = pos.device
    dtype = pos.dtype
    
    # Extract triangle vertices
    v0 = pos[faces[:, 0]]  # [F, 3]
    v1 = pos[faces[:, 1]]  # [F, 3]
    v2 = pos[faces[:, 2]]  # [F, 3]
    
    # Compute edge vectors
    e0 = v2 - v1  # opposite to vertex 0
    e1 = v0 - v2  # opposite to vertex 1
    e2 = v1 - v0  # opposite to vertex 2
    
    # Compute triangle areas (for curvature normalization)
    cross_01 = torch.cross(e2, -e1, dim=1)  # v1-v0 x v2-v0
    tri_areas = torch.norm(cross_01, dim=1, keepdim=True) * 0.5  # [F, 1]
    
    # Compute cotangent weights for each edge in each triangle
    # cot(angle) = dot(adjacent_edges) / ||cross(adjacent_edges)||
    def compute_cot(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        """Cotangent of angle between -ea and eb"""
        dot = ((-ea) * eb).sum(dim=1)
        cross = torch.norm(torch.cross(-ea, eb, dim=1), dim=1)
        return dot / (cross + 1e-8)
    
    cot0 = compute_cot(e1, e2)  # angle at vertex 0
    cot1 = compute_cot(e2, e0)  # angle at vertex 1
    cot2 = compute_cot(e0, e1)  # angle at vertex 2
    
    # Build edge list with cotangent weights
    # Each triangle contributes 3 edges (undirected, we'll symmetrize)
    edges_i = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]], dim=0)
    edges_j = torch.cat([faces[:, 2], faces[:, 0], faces[:, 1]], dim=0)
    edge_cots = torch.cat([cot0, cot1, cot2], dim=0)
    
    # Create symmetric edge index
    edge_index_full = torch.stack([
        torch.cat([edges_i, edges_j]),
        torch.cat([edges_j, edges_i])
    ], dim=0)
    edge_cots_full = torch.cat([edge_cots, edge_cots])
    
    # Aggregate duplicate edges by summing cotangent weights
    # Use scatter_add for efficiency
    edge_hash = edge_index_full[0] * N + edge_index_full[1]
    unique_hash, inverse_idx = torch.unique(edge_hash, return_inverse=True)
    
    # Scatter add cotangent weights
    edge_weights = torch.zeros(unique_hash.size(0), device=device, dtype=dtype)
    edge_weights.scatter_add_(0, inverse_idx, edge_cots_full)
    
    # Reconstruct unique edge_index
    edge_index = torch.stack([
        unique_hash // N,
        unique_hash % N
    ], dim=0).long()
    edge_attr = edge_weights.unsqueeze(1)  # [E, 1]
    
    # Compute Laplacian coordinates using scatter operations
    # L_i = pos_i - mean(pos_neighbors)
    src, dst = edge_index[0], edge_index[1]
    
    # Compute degree (number of neighbors per node)
    degree = torch.zeros(N, device=device, dtype=dtype)
    degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=dtype))
    degree = degree.clamp(min=1)  # avoid division by zero
    
    # Sum of neighbor positions
    neighbor_sum = torch.zeros(N, 3, device=device, dtype=dtype)
    neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), pos[src])
    
    # Laplacian coordinates
    neighbor_mean = neighbor_sum / degree.unsqueeze(1)
    L_coords = pos - neighbor_mean  # [N, 3]
    
    if compute_curvatures:
        # Compute principal curvatures from discrete curvatures
        # Gaussian curvature K = angle defect / area
        # Mean curvature H from Laplace-Beltrami
        
        # Compute angles at each vertex in each triangle
        def compute_angle(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
            """Angle between -ea and eb"""
            cos_val = ((-ea) * eb).sum(dim=1)
            sin_val = torch.norm(torch.cross(-ea, eb, dim=1), dim=1)
            return torch.atan2(sin_val, cos_val)
        
        angle0 = compute_angle(e1, e2)
        angle1 = compute_angle(e2, e0)
        angle2 = compute_angle(e0, e1)
        
        # Sum angles at each vertex
        angle_sum = torch.zeros(N, device=device, dtype=dtype)
        angle_sum.scatter_add_(0, faces[:, 0], angle0)
        angle_sum.scatter_add_(0, faces[:, 1], angle1)
        angle_sum.scatter_add_(0, faces[:, 2], angle2)
        
        # Sum one-ring area at each vertex (1/3 of each incident triangle)
        vertex_area = torch.zeros(N, device=device, dtype=dtype)
        tri_area_flat = tri_areas.squeeze(1)
        vertex_area.scatter_add_(0, faces[:, 0], tri_area_flat / 3)
        vertex_area.scatter_add_(0, faces[:, 1], tri_area_flat / 3)
        vertex_area.scatter_add_(0, faces[:, 2], tri_area_flat / 3)
        vertex_area = vertex_area.clamp(min=1e-8)
        
        # Gaussian curvature: angle defect / area
        K = (2 * math.pi - angle_sum) / vertex_area
        
        # Mean curvature: ||L|| / (4 * area) where L is cotangent-weighted Laplacian
        # Compute cotangent-weighted Laplacian vector
        cot_laplacian = torch.zeros(N, 3, device=device, dtype=dtype)
        edge_diff = pos[src] - pos[dst]  # [E, 3]
        weighted_diff = edge_attr * edge_diff  # [E, 3]
        cot_laplacian.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), weighted_diff)
        
        H = torch.norm(cot_laplacian, dim=1) / (4 * vertex_area + 1e-8)
        
        # Principal curvatures from H, K
        # H = (k1 + k2) / 2, K = k1 * k2
        # k1, k2 = H ± sqrt(H² - K)
        disc = (H**2 - K).clamp(min=0)
        k1 = H + torch.sqrt(disc)
        k2 = H - torch.sqrt(disc)
        
        curvature_feats = torch.stack([k1, k2], dim=1)  # [N, 2]
    else:
        curvature_feats = torch.zeros(N, 2, device=device, dtype=dtype)
    
    # Combine node features
    node_features = torch.cat([L_coords, curvature_feats], dim=1)  # [N, 5]
    
    return node_features, edge_index, edge_attr


def compute_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    stiffness: float = 1.0
) -> torch.Tensor:
    """
    Compute edge features for spring network.
    
    Args:
        pos: Vertex positions [N, 3]
        edge_index: Edge indices [2, E]
        stiffness: Default spring stiffness
        
    Returns:
        edge_attr: [E, 2] - (stiffness, rest_length) per edge
    """
    src, dst = edge_index[0], edge_index[1]
    
    # Rest length = initial distance
    rest_length = torch.norm(pos[src] - pos[dst], dim=1)
    
    # Uniform stiffness (can be made per-edge)
    k = torch.full_like(rest_length, stiffness)
    
    edge_attr = torch.stack([k, rest_length], dim=1)  # [E, 2]
    
    return edge_attr


def build_mesh_graph(
    pos: torch.Tensor,
    faces: torch.Tensor,
    stiffness: float = 1.0,
    compute_curvatures: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build complete mesh graph with node and edge features.
    
    Args:
        pos: Vertex positions [N, 3]
        faces: Triangle indices [F, 3]
        stiffness: Spring stiffness constant
        compute_curvatures: Whether to compute curvature features
        
    Returns:
        node_features: [N, 5]
        edge_index: [2, E]
        edge_attr: [E, 2] - (stiffness, rest_length)
    """
    node_features, edge_index, cot_weights = compute_mesh_features(
        pos, faces, compute_curvatures
    )
    
    # Compute spring edge features
    edge_attr = compute_edge_features(pos, edge_index, stiffness)
    
    return node_features, edge_index, edge_attr


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test mesh (tetrahedron)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=torch.long)
    
    node_feats, edge_idx, edge_attr = build_mesh_graph(pos, faces, stiffness=1.0)
    
    print(f"Node features shape: {node_feats.shape}")
    print(f"Edge index shape: {edge_idx.shape}")
    print(f"Edge attributes shape: {edge_attr.shape}")
    print(f"Number of edges: {edge_idx.shape[1]}")
    print("✓ Mesh features computed successfully")
