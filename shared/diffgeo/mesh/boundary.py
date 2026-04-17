"""
Boundary Detection Utilities
============================

Functions for identifying boundary edges and vertices in triangle meshes.
Essential for handling open meshes (cloth panels, coastal domains).
"""

import numpy as np
from typing import Tuple, Set, List


def find_boundary_edges(
    faces: np.ndarray,
    num_vertices: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find boundary edges (edges adjacent to exactly one face).
    
    Args:
        faces: (F, 3) face vertex indices
        num_vertices: Total number of vertices (optional, for validation)
        
    Returns:
        boundary_edges: (B, 2) boundary edge vertex indices
        edge_face_count: Dict mapping edge tuple to adjacent face count
    """
    # Count face adjacency for each edge
    edge_face_count = {}
    
    for face in faces:
        i, j, k = face
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
    
    # Boundary edges have exactly 1 adjacent face
    boundary_edges = [
        list(edge) for edge, count in edge_face_count.items() if count == 1
    ]
    
    return np.array(boundary_edges, dtype=np.int64), edge_face_count


def find_boundary_vertices(
    faces: np.ndarray,
    num_vertices: int = None
) -> np.ndarray:
    """
    Find boundary vertices (vertices on boundary edges).
    
    Args:
        faces: (F, 3) face vertex indices
        num_vertices: Total number of vertices (optional)
        
    Returns:
        boundary_vertices: (Bv,) sorted boundary vertex indices
    """
    boundary_edges, _ = find_boundary_edges(faces, num_vertices)
    
    boundary_vertex_set = set()
    for edge in boundary_edges:
        boundary_vertex_set.add(edge[0])
        boundary_vertex_set.add(edge[1])
    
    return np.array(sorted(boundary_vertex_set), dtype=np.int64)


def find_boundary_loops(
    faces: np.ndarray,
    num_vertices: int = None
) -> List[np.ndarray]:
    """
    Find ordered boundary loops (closed chains of boundary edges).
    
    Returns a list of vertex index arrays, one per boundary loop.
    For a mesh with holes, returns multiple loops.
    
    Args:
        faces: (F, 3) face vertex indices
        num_vertices: Total number of vertices (optional)
        
    Returns:
        loops: List of (Li,) arrays, each containing ordered vertex indices
               of a boundary loop
    """
    boundary_edges, _ = find_boundary_edges(faces, num_vertices)
    
    if len(boundary_edges) == 0:
        return []  # Closed mesh, no boundaries
    
    # Build adjacency for boundary edges
    vertex_neighbors = {}
    for edge in boundary_edges:
        a, b = edge
        if a not in vertex_neighbors:
            vertex_neighbors[a] = []
        if b not in vertex_neighbors:
            vertex_neighbors[b] = []
        vertex_neighbors[a].append(b)
        vertex_neighbors[b].append(a)
    
    # Trace boundary loops
    visited = set()
    loops = []
    
    for start_vertex in vertex_neighbors:
        if start_vertex in visited:
            continue
        
        # Trace loop starting from this vertex
        loop = [start_vertex]
        visited.add(start_vertex)
        
        current = start_vertex
        while True:
            neighbors = vertex_neighbors[current]
            next_vertex = None
            
            for n in neighbors:
                if n not in visited:
                    next_vertex = n
                    break
            
            if next_vertex is None:
                # Check if we're back at start (closed loop)
                if start_vertex in neighbors and len(loop) > 2:
                    break
                else:
                    break
            
            loop.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex
        
        if len(loop) > 2:
            loops.append(np.array(loop, dtype=np.int64))
    
    return loops


def is_manifold(faces: np.ndarray) -> bool:
    """
    Check if mesh is a manifold (each edge has at most 2 adjacent faces).
    
    Args:
        faces: (F, 3) face vertex indices
        
    Returns:
        True if mesh is manifold, False otherwise
    """
    _, edge_face_count = find_boundary_edges(faces)
    
    for edge, count in edge_face_count.items():
        if count > 2:
            return False
    
    return True


def is_closed(faces: np.ndarray) -> bool:
    """
    Check if mesh is closed (watertight, no boundary edges).
    
    Args:
        faces: (F, 3) face vertex indices
        
    Returns:
        True if mesh has no boundary edges
    """
    boundary_edges, _ = find_boundary_edges(faces)
    return len(boundary_edges) == 0
