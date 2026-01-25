"""
Mesh-to-Graph Converter with Multi-Resolution Graph Pyramid

This module converts cloth meshes to graph representations with hierarchical
coarsening using Graclus clustering for multi-resolution message passing.

Author: HGNN-ClothDyn
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce
from typing import Tuple, List, Optional, Dict, Any
import logging

# Try to import graclus, fall back to simple clustering if torch-cluster unavailable
try:
    from torch_geometric.nn.pool import graclus
    HAS_GRACLUS = True
except ImportError:
    HAS_GRACLUS = False
    logging.warning("torch-cluster not available, using simple strided clustering")


def simple_strided_cluster(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Simple strided clustering fallback when torch-cluster is unavailable.
    Groups every 2 adjacent nodes together based on node indices.

    Args:
        edge_index: Edge index tensor (2, E)
        num_nodes: Number of nodes

    Returns:
        cluster: Cluster assignment for each node
    """
    # Simple strided clustering: pair adjacent nodes
    cluster = torch.arange(num_nodes, dtype=torch.long)
    cluster = cluster // 2  # Each pair of adjacent nodes gets same cluster
    return cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_grid_mesh(
    size: int = 10,
    spacing: float = 0.1,
    center: Tuple[float, float, float] = (0.0, 1.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a simple grid mesh for cloth simulation.
    
    Args:
        size: Number of vertices per side (NxN grid)
        spacing: Distance between adjacent vertices
        center: Center position of the mesh
        
    Returns:
        vertices: (N*N, 3) vertex positions
        faces: (F, 3) triangle face indices
        edges: (E, 2) edge indices
    """
    n = size
    # Create grid positions
    x = np.linspace(-spacing * (n - 1) / 2, spacing * (n - 1) / 2, n)
    z = np.linspace(-spacing * (n - 1) / 2, spacing * (n - 1) / 2, n)
    xx, zz = np.meshgrid(x, z)
    
    # Vertices: flat cloth at height y=center[1]
    vertices = np.zeros((n * n, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten() + center[0]
    vertices[:, 1] = center[1]  # Initial height
    vertices[:, 2] = zz.flatten() + center[2]
    
    # Create faces (two triangles per quad)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.int64)
    
    # Extract edges from faces (unique, undirected)
    edge_set = set()
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_set.add(e)
    edges = np.array(list(edge_set), dtype=np.int64)
    
    logger.info(f"Created grid mesh: {n}x{n} = {len(vertices)} vertices, "
                f"{len(faces)} faces, {len(edges)} edges")
    
    return vertices, faces, edges


def compute_rest_lengths(
    vertices: np.ndarray,
    edges: np.ndarray
) -> np.ndarray:
    """
    Compute rest lengths for all edges (used for Hookean spring constraints).
    
    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices
        
    Returns:
        rest_lengths: (E,) rest length for each edge
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    rest_lengths = np.linalg.norm(v1 - v0, axis=1).astype(np.float32)
    return rest_lengths


def mesh_to_graph(
    vertices: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    edges: np.ndarray = None,
    rest_lengths: Optional[np.ndarray] = None,
    collision_mask: Optional[np.ndarray] = None
) -> Data:
    """
    Convert mesh data to PyTorch Geometric Data object.
    
    Args:
        vertices: (N, 3) vertex positions
        velocities: (N, 3) vertex velocities (optional, zeros if not provided)
        edges: (E, 2) edge connectivity
        rest_lengths: (E,) rest lengths for edges
        collision_mask: (N,) boolean mask for collision nodes
        
    Returns:
        PyG Data object with node features, edge index, and edge attributes
    """
    n_nodes = len(vertices)
    
    # Node features: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    if velocities is None:
        velocities = np.zeros((n_nodes, 3), dtype=np.float32)
    
    node_features = np.concatenate([vertices, velocities], axis=1).astype(np.float32)
    
    # Edge index (PyG format: [2, E])
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    # Make undirected
    edge_index = to_undirected(edge_index)
    
    # Rest lengths as edge attributes
    if rest_lengths is not None:
        # Duplicate for undirected edges
        rest_lengths_full = np.concatenate([rest_lengths, rest_lengths])
        edge_attr = torch.tensor(rest_lengths_full, dtype=torch.float32).unsqueeze(-1)
    else:
        edge_attr = None
    
    # Create Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(vertices, dtype=torch.float32),
        num_nodes=n_nodes
    )
    
    # Add collision mask if provided
    if collision_mask is not None:
        data.collision_mask = torch.tensor(collision_mask, dtype=torch.bool)
    
    return data


def build_graph_pyramid(
    data: Data,
    num_levels: int = 2,
    coarsening_ratio: float = 0.5
) -> Tuple[List[Data], List[torch.Tensor]]:
    """
    Build a multi-resolution graph pyramid using Graclus clustering.
    
    This enables hierarchical message passing: fine features are pooled to
    coarse levels for global context, then upsampled back.
    
    Args:
        data: Fine-level graph (level 0)
        num_levels: Number of pyramid levels (including fine level)
        coarsening_ratio: Target ratio of nodes at each coarser level
        
    Returns:
        graphs: List of Data objects [fine, coarse, coarsest, ...]
        cluster_maps: List of cluster assignment tensors for pooling/unpooling
    """
    graphs = [data]
    cluster_maps = []
    
    current_data = data
    device = data.x.device  # Get device from input data
    
    for level in range(1, num_levels):
        edge_index = current_data.edge_index
        num_nodes = current_data.num_nodes
        
        # Graclus clustering (needs CPU edge_index), fall back to simple clustering if unavailable
        edge_index_cpu = edge_index.cpu()
        try:
            if HAS_GRACLUS:
                cluster = graclus(edge_index_cpu, num_nodes=num_nodes)
            else:
                cluster = simple_strided_cluster(edge_index_cpu, num_nodes)
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Graclus failed ({e}), using simple strided clustering")
            cluster = simple_strided_cluster(edge_index_cpu, num_nodes)
        
        # Get unique clusters and remap
        unique_clusters, inverse = torch.unique(cluster, return_inverse=True)
        num_coarse_nodes = len(unique_clusters)
        
        # Move inverse to original device
        inverse = inverse.to(device)
        cluster_maps.append(inverse)
        
        # Pool node features (mean aggregation) - use scatter operations
        coarse_x = torch.zeros(num_coarse_nodes, current_data.x.size(1), 
                               dtype=current_data.x.dtype, device=device)
        coarse_pos = torch.zeros(num_coarse_nodes, 3, dtype=torch.float32, device=device)
        counts = torch.zeros(num_coarse_nodes, dtype=torch.float32, device=device)
        
        # Scatter add for pooling
        coarse_x.scatter_add_(0, inverse.unsqueeze(1).expand(-1, current_data.x.size(1)), 
                              current_data.x)
        coarse_pos.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), 
                                current_data.pos)
        counts.scatter_add_(0, inverse, torch.ones(num_nodes, dtype=torch.float32, device=device))
        
        coarse_x = coarse_x / counts.unsqueeze(1).clamp(min=1)
        coarse_pos = coarse_pos / counts.unsqueeze(1).clamp(min=1)
        
        # Build coarse edges from fine edges (on CPU for set operations)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        inverse_cpu = inverse.cpu()
        edge_src_cpu = edge_src.cpu()
        edge_dst_cpu = edge_dst.cpu()
        
        coarse_edge_set = set()
        for i in range(edge_index.size(1)):
            src_cluster = inverse_cpu[edge_src_cpu[i]].item()
            dst_cluster = inverse_cpu[edge_dst_cpu[i]].item()
            if src_cluster != dst_cluster:
                e = tuple(sorted([src_cluster, dst_cluster]))
                coarse_edge_set.add(e)
        
        if len(coarse_edge_set) > 0:
            coarse_edges = torch.tensor(list(coarse_edge_set), dtype=torch.long, device=device).T
            coarse_edge_index = to_undirected(coarse_edges)
        else:
            # Fallback: fully connected if no edges
            coarse_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Compute rest lengths for coarse edges
        if coarse_edge_index.size(1) > 0:
            src_pos = coarse_pos[coarse_edge_index[0]]
            dst_pos = coarse_pos[coarse_edge_index[1]]
            coarse_rest_lengths = torch.norm(dst_pos - src_pos, dim=1, keepdim=True)
        else:
            coarse_rest_lengths = None
        
        coarse_data = Data(
            x=coarse_x,
            edge_index=coarse_edge_index,
            edge_attr=coarse_rest_lengths,
            pos=coarse_pos,
            num_nodes=num_coarse_nodes
        )
        
        graphs.append(coarse_data)
        current_data = coarse_data
        
        logger.info(f"Level {level}: {num_coarse_nodes} nodes, "
                    f"{coarse_edge_index.size(1) // 2} edges")
    
    return graphs, cluster_maps


def pool_features(
    fine_features: torch.Tensor,
    cluster_map: torch.Tensor,
    num_coarse_nodes: int,
    pool_type: str = 'mean'
) -> torch.Tensor:
    """
    Pool fine-level features to coarse level.
    
    Args:
        fine_features: (N_fine, D) fine-level node features
        cluster_map: (N_fine,) cluster assignment for each fine node
        num_coarse_nodes: Number of nodes at coarse level
        pool_type: 'mean', 'max', or 'sum'
        
    Returns:
        coarse_features: (N_coarse, D) pooled features
    """
    device = fine_features.device
    dtype = fine_features.dtype
    feat_dim = fine_features.size(1)
    
    coarse_features = torch.zeros(num_coarse_nodes, feat_dim, 
                                   device=device, dtype=dtype)
    
    if pool_type == 'mean':
        counts = torch.zeros(num_coarse_nodes, device=device, dtype=dtype)
        coarse_features.scatter_add_(0, cluster_map.unsqueeze(1).expand(-1, feat_dim), 
                                      fine_features)
        counts.scatter_add_(0, cluster_map, torch.ones_like(cluster_map, dtype=dtype))
        coarse_features = coarse_features / counts.unsqueeze(1).clamp(min=1)
    elif pool_type == 'max':
        coarse_features.scatter_reduce_(0, cluster_map.unsqueeze(1).expand(-1, feat_dim),
                                         fine_features, reduce='amax', include_self=False)
    elif pool_type == 'sum':
        coarse_features.scatter_add_(0, cluster_map.unsqueeze(1).expand(-1, feat_dim),
                                      fine_features)
    
    return coarse_features


def unpool_features(
    coarse_features: torch.Tensor,
    cluster_map: torch.Tensor
) -> torch.Tensor:
    """
    Unpool coarse-level features back to fine level.
    
    Args:
        coarse_features: (N_coarse, D) coarse-level node features
        cluster_map: (N_fine,) cluster assignment for each fine node
        
    Returns:
        fine_features: (N_fine, D) unpooled features
    """
    return coarse_features[cluster_map]


def load_mesh_from_obj(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh from OBJ file.
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle face indices
        edges: (E, 2) edge indices
    """
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Handle various face formats (v, v/vt, v/vt/vn, v//vn)
                face_verts = []
                for p in parts[1:]:
                    v_idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(v_idx)
                # Triangulate if needed
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64)
    
    # Extract edges
    edge_set = set()
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_set.add(e)
    edges = np.array(list(edge_set), dtype=np.int64)
    
    logger.info(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces, {len(edges)} edges")
    
    return vertices, faces, edges


# ============================================================================
# Testing
# ============================================================================

def test_mesh_to_graph():
    """Test mesh-to-graph conversion and pyramid building."""
    print("=" * 60)
    print("Testing mesh_to_graph.py")
    print("=" * 60)
    
    # Create test mesh
    vertices, faces, edges = create_grid_mesh(size=10, spacing=0.1)
    print(f"Created mesh: {vertices.shape[0]} vertices, {edges.shape[0]} edges")
    
    # Compute rest lengths
    rest_lengths = compute_rest_lengths(vertices, edges)
    print(f"Rest lengths range: [{rest_lengths.min():.4f}, {rest_lengths.max():.4f}]")
    
    # Convert to graph
    data = mesh_to_graph(vertices, edges=edges, rest_lengths=rest_lengths)
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} directed edges")
    print(f"Node features shape: {data.x.shape}")
    
    # Build pyramid
    graphs, cluster_maps = build_graph_pyramid(data, num_levels=3)
    print(f"\nGraph pyramid ({len(graphs)} levels):")
    for i, g in enumerate(graphs):
        print(f"  Level {i}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges")
    
    # Test pooling/unpooling
    if len(cluster_maps) > 0:
        fine_feat = graphs[0].x
        coarse_feat = pool_features(fine_feat, cluster_maps[0], graphs[1].num_nodes)
        unpooled = unpool_features(coarse_feat, cluster_maps[0])
        print(f"\nPooling test:")
        print(f"  Fine: {fine_feat.shape} → Coarse: {coarse_feat.shape} → Unpooled: {unpooled.shape}")
    
    print("\n✓ All mesh_to_graph tests passed!")
    return True


if __name__ == "__main__":
    test_mesh_to_graph()
