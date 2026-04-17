"""
Graph construction and coarsening for cloth mesh sequences.

Builds graph representations suitable for GNN-based cloth simulation models
(Projects 05, 08, 09).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from ..types import ClothSequence


@dataclass
class GraphData:
    """
    Graph representation of a cloth mesh.
    
    Attributes:
        edge_index: Edge connectivity in COO format. Shape: (2, num_edges)
        edge_features: Per-edge features. Shape: (num_edges, feature_dim)
        rest_edge_lengths: Rest-state edge lengths. Shape: (num_edges,)
        rest_edge_directions: Rest-state edge direction vectors. Shape: (num_edges, 3)
        dihedral_angles: Dihedral angles at each edge. Shape: (num_edges,)
        num_vertices: Number of vertices in the graph
        num_edges: Number of edges in the graph
        
        # Multi-level hierarchy (for hierarchical GNNs)
        hierarchy: List of coarsened graph levels
    """
    edge_index: np.ndarray
    edge_features: Optional[np.ndarray] = None
    rest_edge_lengths: Optional[np.ndarray] = None
    rest_edge_directions: Optional[np.ndarray] = None
    dihedral_angles: Optional[np.ndarray] = None
    num_vertices: int = 0
    num_edges: int = 0
    
    # Hierarchical data
    hierarchy: List['CoarseLevel'] = field(default_factory=list)
    
    def __post_init__(self):
        if self.num_edges == 0:
            self.num_edges = self.edge_index.shape[1]


@dataclass
class CoarseLevel:
    """One level in the graph hierarchy."""
    edge_index: np.ndarray  # (2, num_coarse_edges)
    cluster_assignment: np.ndarray  # (num_fine_vertices,) -> coarse vertex id
    num_vertices: int
    num_edges: int
    inter_level_edges: Optional[np.ndarray] = None  # Edges between levels


def build_graph(
    sequence: ClothSequence,
    bidirectional: bool = True,
) -> GraphData:
    """
    Build a graph from mesh face connectivity.
    
    Args:
        sequence: ClothSequence with mesh data
        bidirectional: If True, include edges in both directions
        
    Returns:
        GraphData with edge_index and basic structure
    """
    faces = sequence.faces
    
    if sequence.is_triangulated:
        # Extract edges from triangles
        edges_0_1 = faces[:, [0, 1]]
        edges_1_2 = faces[:, [1, 2]]
        edges_2_0 = faces[:, [2, 0]]
        all_edges = np.vstack([edges_0_1, edges_1_2, edges_2_0])
    else:
        # Extract edges from quads
        edges_0_1 = faces[:, [0, 1]]
        edges_1_2 = faces[:, [1, 2]]
        edges_2_3 = faces[:, [2, 3]]
        edges_3_0 = faces[:, [3, 0]]
        all_edges = np.vstack([edges_0_1, edges_1_2, edges_2_3, edges_3_0])
    
    # Sort each edge so smaller index comes first
    all_edges = np.sort(all_edges, axis=1)
    
    # Remove duplicates
    unique_edges = np.unique(all_edges, axis=0)
    
    if bidirectional:
        # Add reverse edges
        reverse_edges = unique_edges[:, [1, 0]]
        edge_index = np.vstack([unique_edges, reverse_edges]).T
    else:
        edge_index = unique_edges.T
    
    return GraphData(
        edge_index=edge_index.astype(np.int64),
        num_vertices=sequence.num_vertices,
        num_edges=edge_index.shape[1],
    )


def compute_edge_features(
    sequence: ClothSequence,
    graph: GraphData,
    rest_frame: int = 0,
    compute_dihedral: bool = True,
) -> GraphData:
    """
    Compute edge features for the graph.
    
    Features computed:
    - Rest-state edge lengths
    - Rest-state edge direction vectors (normalized)
    - Dihedral angles between adjacent faces
    
    Args:
        sequence: ClothSequence with vertex data
        graph: GraphData to augment with features
        rest_frame: Which frame to use as rest state
        compute_dihedral: If True, compute dihedral angles (slower)
        
    Returns:
        GraphData with edge features added
    """
    rest_vertices = sequence.vertices[rest_frame]
    edge_index = graph.edge_index
    
    # Get source and target vertices for each edge
    src_indices = edge_index[0]
    tgt_indices = edge_index[1]
    
    src_pos = rest_vertices[src_indices]
    tgt_pos = rest_vertices[tgt_indices]
    
    # Edge vectors
    edge_vectors = tgt_pos - src_pos
    
    # Edge lengths
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Normalized edge directions
    edge_directions = edge_vectors / np.maximum(edge_lengths[:, np.newaxis], 1e-10)
    
    # Compute dihedral angles
    dihedral_angles = None
    if compute_dihedral and sequence.is_triangulated:
        dihedral_angles = _compute_dihedral_angles(
            rest_vertices, sequence.faces, edge_index
        )
    
    # Combine into feature vector
    # Features: [length, direction_x, direction_y, direction_z, dihedral]
    if dihedral_angles is not None:
        edge_features = np.column_stack([
            edge_lengths,
            edge_directions,
            dihedral_angles,
        ])
    else:
        edge_features = np.column_stack([
            edge_lengths,
            edge_directions,
        ])
    
    graph.edge_features = edge_features.astype(np.float32)
    graph.rest_edge_lengths = edge_lengths.astype(np.float32)
    graph.rest_edge_directions = edge_directions.astype(np.float32)
    graph.dihedral_angles = dihedral_angles.astype(np.float32) if dihedral_angles is not None else None
    
    return graph


def _compute_dihedral_angles(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """
    Compute dihedral angles at each edge.
    
    The dihedral angle is the angle between the normals of the two
    faces sharing an edge.
    """
    num_edges = edge_index.shape[1]
    dihedral_angles = np.zeros(num_edges, dtype=np.float32)
    
    # Build edge-to-faces mapping
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    
    for face_idx, face in enumerate(faces):
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]])),
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
    
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / np.maximum(
        np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-10
    )
    
    # Compute dihedral angle for each edge
    for i in range(num_edges):
        src, tgt = edge_index[0, i], edge_index[1, i]
        edge_key = tuple(sorted([src, tgt]))
        
        adjacent_faces = edge_to_faces.get(edge_key, [])
        
        if len(adjacent_faces) == 2:
            n1 = face_normals[adjacent_faces[0]]
            n2 = face_normals[adjacent_faces[1]]
            
            # Angle between normals
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            dihedral_angles[i] = np.arccos(cos_angle)
        else:
            # Boundary edge or non-manifold
            dihedral_angles[i] = 0.0
    
    return dihedral_angles


def coarsen_graph(
    sequence: ClothSequence,
    graph: GraphData,
    num_levels: int = 2,
    method: str = 'greedy',
    coarsening_ratio: float = 0.5,
) -> GraphData:
    """
    Build a hierarchical graph by coarsening.
    
    Args:
        sequence: ClothSequence with mesh data
        graph: Fine-level GraphData
        num_levels: Number of coarsening levels to create
        method: Coarsening method ('greedy', 'graclus', or 'metis')
        coarsening_ratio: Target ratio of vertices at each level
        
    Returns:
        GraphData with hierarchy populated
    """
    if method == 'graclus':
        return _coarsen_graclus(sequence, graph, num_levels, coarsening_ratio)
    elif method == 'metis':
        return _coarsen_metis(sequence, graph, num_levels, coarsening_ratio)
    elif method == 'greedy':
        return _coarsen_greedy(sequence, graph, num_levels, coarsening_ratio)
    else:
        raise ValueError(f"Unknown coarsening method: {method}")


def _coarsen_greedy(
    sequence: ClothSequence,
    graph: GraphData,
    num_levels: int,
    coarsening_ratio: float,
) -> GraphData:
    """
    Greedy matching-based graph coarsening.
    
    This is a simple fallback when torch-cluster isn't available.
    
    Algorithm:
    1. Randomly permute vertices
    2. Greedily match each unmatched vertex with its lowest-degree unmatched neighbor
    3. Collapse matched pairs into supernodes
    4. Build coarsened graph
    """
    hierarchy = []
    current_edge_index = graph.edge_index.copy()
    current_num_vertices = graph.num_vertices
    
    for level in range(num_levels):
        target_vertices = max(int(current_num_vertices * coarsening_ratio), 4)
        
        # Build adjacency list
        adj_list = [[] for _ in range(current_num_vertices)]
        for i in range(current_edge_index.shape[1]):
            src, tgt = current_edge_index[0, i], current_edge_index[1, i]
            if src != tgt:
                adj_list[src].append(tgt)
        
        # Compute vertex degrees
        degrees = np.array([len(adj) for adj in adj_list])
        
        # Random permutation
        perm = np.random.permutation(current_num_vertices)
        
        # Greedy matching
        matched = np.zeros(current_num_vertices, dtype=bool)
        cluster_assignment = np.full(current_num_vertices, -1, dtype=np.int64)
        next_cluster = 0
        
        for v in perm:
            if matched[v]:
                continue
            
            # Find unmatched neighbor with minimum degree
            best_neighbor = -1
            best_degree = float('inf')
            
            for neighbor in adj_list[v]:
                if not matched[neighbor] and degrees[neighbor] < best_degree:
                    best_degree = degrees[neighbor]
                    best_neighbor = neighbor
            
            if best_neighbor >= 0:
                # Match v with best_neighbor
                cluster_assignment[v] = next_cluster
                cluster_assignment[best_neighbor] = next_cluster
                matched[v] = True
                matched[best_neighbor] = True
            else:
                # No unmatched neighbor, v becomes its own cluster
                cluster_assignment[v] = next_cluster
                matched[v] = True
            
            next_cluster += 1
        
        # Handle any remaining unmatched vertices
        for v in range(current_num_vertices):
            if cluster_assignment[v] < 0:
                cluster_assignment[v] = next_cluster
                next_cluster += 1
        
        num_coarse_vertices = next_cluster
        
        # Build coarsened edge index
        coarse_edges = set()
        for i in range(current_edge_index.shape[1]):
            src_cluster = cluster_assignment[current_edge_index[0, i]]
            tgt_cluster = cluster_assignment[current_edge_index[1, i]]
            if src_cluster != tgt_cluster:
                coarse_edges.add((src_cluster, tgt_cluster))
        
        if coarse_edges:
            coarse_edge_index = np.array(list(coarse_edges), dtype=np.int64).T
        else:
            coarse_edge_index = np.zeros((2, 0), dtype=np.int64)
        
        hierarchy.append(CoarseLevel(
            edge_index=coarse_edge_index,
            cluster_assignment=cluster_assignment,
            num_vertices=num_coarse_vertices,
            num_edges=coarse_edge_index.shape[1],
        ))
        
        # Update for next level
        current_edge_index = coarse_edge_index
        current_num_vertices = num_coarse_vertices
        
        if current_num_vertices <= target_vertices:
            break
    
    graph.hierarchy = hierarchy
    return graph


def _coarsen_graclus(
    sequence: ClothSequence,
    graph: GraphData,
    num_levels: int,
    coarsening_ratio: float,
) -> GraphData:
    """
    Graclus-based graph coarsening using torch-cluster.
    """
    try:
        import torch
        from torch_cluster import graclus_cluster
    except ImportError:
        import warnings
        warnings.warn("torch-cluster not available, falling back to greedy coarsening")
        return _coarsen_greedy(sequence, graph, num_levels, coarsening_ratio)
    
    hierarchy = []
    current_edge_index = torch.from_numpy(graph.edge_index).long()
    current_num_vertices = graph.num_vertices
    
    for level in range(num_levels):
        # Run Graclus clustering
        cluster = graclus_cluster(
            current_edge_index[0],
            current_edge_index[1],
            num_nodes=current_num_vertices,
        )
        
        cluster_assignment = cluster.numpy()
        num_coarse_vertices = cluster_assignment.max() + 1
        
        # Build coarsened edges
        src_clusters = cluster_assignment[current_edge_index[0].numpy()]
        tgt_clusters = cluster_assignment[current_edge_index[1].numpy()]
        
        # Remove self-loops and duplicates
        mask = src_clusters != tgt_clusters
        coarse_edges = np.stack([src_clusters[mask], tgt_clusters[mask]], axis=0)
        coarse_edges = np.unique(coarse_edges, axis=1)
        
        hierarchy.append(CoarseLevel(
            edge_index=coarse_edges,
            cluster_assignment=cluster_assignment,
            num_vertices=num_coarse_vertices,
            num_edges=coarse_edges.shape[1],
        ))
        
        current_edge_index = torch.from_numpy(coarse_edges).long()
        current_num_vertices = num_coarse_vertices
    
    graph.hierarchy = hierarchy
    return graph


def _coarsen_metis(
    sequence: ClothSequence,
    graph: GraphData,
    num_levels: int,
    coarsening_ratio: float,
) -> GraphData:
    """
    METIS-based graph partitioning for coarsening.
    """
    try:
        import pymetis
    except ImportError:
        import warnings
        warnings.warn("pymetis not available, falling back to greedy coarsening")
        return _coarsen_greedy(sequence, graph, num_levels, coarsening_ratio)
    
    hierarchy = []
    current_edge_index = graph.edge_index
    current_num_vertices = graph.num_vertices
    
    for level in range(num_levels):
        target_clusters = max(int(current_num_vertices * coarsening_ratio), 4)
        
        # Build adjacency list for METIS
        adj_list = [[] for _ in range(current_num_vertices)]
        for i in range(current_edge_index.shape[1]):
            src, tgt = current_edge_index[0, i], current_edge_index[1, i]
            if src != tgt:
                adj_list[src].append(tgt)
        
        # Run METIS partitioning
        _, cluster_assignment = pymetis.part_graph(
            target_clusters,
            adjacency=adj_list,
        )
        cluster_assignment = np.array(cluster_assignment, dtype=np.int64)
        
        num_coarse_vertices = cluster_assignment.max() + 1
        
        # Build coarsened edges
        src_clusters = cluster_assignment[current_edge_index[0]]
        tgt_clusters = cluster_assignment[current_edge_index[1]]
        
        mask = src_clusters != tgt_clusters
        coarse_edges = np.stack([src_clusters[mask], tgt_clusters[mask]], axis=0)
        coarse_edges = np.unique(coarse_edges, axis=1)
        
        hierarchy.append(CoarseLevel(
            edge_index=coarse_edges,
            cluster_assignment=cluster_assignment,
            num_vertices=num_coarse_vertices,
            num_edges=coarse_edges.shape[1],
        ))
        
        current_edge_index = coarse_edges
        current_num_vertices = num_coarse_vertices
    
    graph.hierarchy = hierarchy
    return graph


def get_coarse_vertices(
    fine_vertices: np.ndarray,
    hierarchy: List[CoarseLevel],
    level: int = 0,
) -> np.ndarray:
    """
    Compute coarse vertex positions by averaging clustered fine vertices.
    
    Args:
        fine_vertices: (N_fine, 3) fine-level vertex positions
        hierarchy: List of coarsening levels
        level: Which hierarchy level to compute (0 = first coarse level)
        
    Returns:
        (N_coarse, 3) coarse vertex positions
    """
    if level >= len(hierarchy):
        raise ValueError(f"Level {level} not available (have {len(hierarchy)} levels)")
    
    cluster_assignment = hierarchy[level].cluster_assignment
    num_coarse = hierarchy[level].num_vertices
    
    coarse_vertices = np.zeros((num_coarse, 3), dtype=np.float32)
    counts = np.zeros(num_coarse, dtype=np.int32)
    
    for fine_idx, coarse_idx in enumerate(cluster_assignment):
        coarse_vertices[coarse_idx] += fine_vertices[fine_idx]
        counts[coarse_idx] += 1
    
    # Average
    coarse_vertices = coarse_vertices / np.maximum(counts[:, np.newaxis], 1)
    
    return coarse_vertices
