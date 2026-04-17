"""
Tangent Basis and Parallel Transport
====================================

Functions for computing tangent bases at mesh vertices and
parallel-transporting vectors between tangent planes.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_tangent_basis_np(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orthonormal tangent basis (t1, t2) from surface normals.
    
    Uses Gram-Schmidt orthonormalization with a reference vector.
    
    Args:
        normals: (N, 3) unit surface normals
        
    Returns:
        t1: (N, 3) first tangent vector
        t2: (N, 3) second tangent vector (computed as n × t1)
    """
    N = normals.shape[0]
    
    # Choose reference vectors not parallel to normals
    # Default to x-axis, switch to y-axis when normal is near x-axis
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    
    # Find normals nearly parallel to x-axis
    parallel_mask = np.abs(np.sum(normals * ref, axis=-1)) > 0.9
    ref[parallel_mask] = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt: t1 = ref - (ref·n)n, then normalize
    dot = np.sum(ref * normals, axis=-1, keepdims=True)
    t1 = ref - dot * normals
    t1_norm = np.linalg.norm(t1, axis=-1, keepdims=True)
    t1 = t1 / (t1_norm + 1e-10)
    
    # t2 = n × t1
    t2 = np.cross(normals, t1)
    
    return t1, t2


if HAS_TORCH:
    def compute_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthonormal tangent basis (t1, t2) from surface normals.
        
        PyTorch version - differentiable.
        
        Args:
            normals: (N, 3) unit surface normals
            
        Returns:
            t1: (N, 3) first tangent vector
            t2: (N, 3) second tangent vector (computed as n × t1)
        """
        device = normals.device
        dtype = normals.dtype
        N = normals.shape[0]
        
        # Choose reference vectors not parallel to normals
        ref = torch.zeros(N, 3, device=device, dtype=dtype)
        ref[:, 0] = 1.0  # x-axis
        
        # Find normals nearly parallel to x-axis
        parallel_mask = torch.abs(torch.sum(normals * ref, dim=-1)) > 0.9
        ref[parallel_mask, 0] = 0.0
        ref[parallel_mask, 1] = 1.0  # Use y-axis instead
        
        # Gram-Schmidt: t1 = ref - (ref·n)n, then normalize
        dot = torch.sum(ref * normals, dim=-1, keepdim=True)
        t1 = ref - dot * normals
        t1 = t1 / (torch.norm(t1, dim=-1, keepdim=True) + 1e-8)
        
        # t2 = n × t1
        t2 = torch.cross(normals, t1, dim=-1)
        
        return t1, t2
else:
    def compute_tangent_basis(normals):
        """Fallback to numpy implementation when torch not available."""
        t1_np, t2_np = compute_tangent_basis_np(np.asarray(normals))
        return t1_np, t2_np


def parallel_transport(
    vector: np.ndarray,
    normal_from: np.ndarray,
    normal_to: np.ndarray,
    t1_from: np.ndarray,
    t2_from: np.ndarray,
    t1_to: np.ndarray,
    t2_to: np.ndarray,
) -> np.ndarray:
    """
    Parallel-transport a tangent vector from one tangent plane to another.
    
    Uses the Levi-Civita connection on the surface. For nearby points,
    this reduces to rotating the vector by the angle between the
    tangent frames.
    
    Args:
        vector: (2,) or (N, 2) vector in source tangent plane coordinates
        normal_from: (3,) or (N, 3) normal at source
        normal_to: (3,) or (N, 3) normal at target
        t1_from, t2_from: Tangent basis at source
        t1_to, t2_to: Tangent basis at target
        
    Returns:
        transported: Vector in target tangent plane coordinates
    """
    # Handle single vector case
    single = vector.ndim == 1
    if single:
        vector = vector[None, :]
        normal_from = normal_from[None, :]
        normal_to = normal_to[None, :]
        t1_from = t1_from[None, :]
        t2_from = t2_from[None, :]
        t1_to = t1_to[None, :]
        t2_to = t2_to[None, :]
    
    N = vector.shape[0]
    
    # Convert vector to 3D (in source tangent plane)
    vec_3d = vector[:, 0:1] * t1_from + vector[:, 1:2] * t2_from  # (N, 3)
    
    # Compute rotation from normal_from to normal_to
    # Using Rodrigues' rotation formula
    
    # Axis of rotation: n_from × n_to
    axis = np.cross(normal_from, normal_to)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    
    # Handle nearly parallel normals (no rotation needed)
    small_angle_mask = axis_norm.squeeze() < 1e-8
    
    # Normalize axis where valid
    axis = np.where(axis_norm > 1e-8, axis / axis_norm, axis)
    
    # Angle of rotation
    cos_angle = np.clip(np.sum(normal_from * normal_to, axis=-1, keepdims=True), -1, 1)
    sin_angle = axis_norm
    
    # Rodrigues' formula: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    k_cross_v = np.cross(axis, vec_3d)
    k_dot_v = np.sum(axis * vec_3d, axis=-1, keepdims=True)
    
    transported_3d = (
        vec_3d * cos_angle +
        k_cross_v * sin_angle +
        axis * k_dot_v * (1 - cos_angle)
    )
    
    # Project back to target tangent plane coordinates
    transported = np.stack([
        np.sum(transported_3d * t1_to, axis=-1),
        np.sum(transported_3d * t2_to, axis=-1),
    ], axis=-1)
    
    if single:
        transported = transported[0]
    
    return transported


if HAS_TORCH:
    def parallel_transport_batch(
        vectors: torch.Tensor,
        normals_from: torch.Tensor,
        normals_to: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute parallel transport rotation angles for batch of edges.
        
        Returns the rotation matrix components that transform vectors
        from source tangent plane to target tangent plane.
        
        This is optimized for use in message passing where we need
        to transport many vectors efficiently.
        
        Args:
            vectors: (E, 2) vectors in source tangent planes
            normals_from: (E, 3) normals at source vertices
            normals_to: (E, 3) normals at target vertices
            
        Returns:
            transported: (E, 2) transported vectors
            rotation_angles: (E,) rotation angles (useful for caching)
        """
        E = vectors.shape[0]
        device = vectors.device
        dtype = vectors.dtype
        
        # Compute tangent bases
        t1_from, t2_from = compute_tangent_basis(normals_from)
        t1_to, t2_to = compute_tangent_basis(normals_to)
        
        # Convert to 3D
        vec_3d = vectors[:, 0:1] * t1_from + vectors[:, 1:2] * t2_from
        
        # Rotation axis and angle
        axis = torch.cross(normals_from, normals_to, dim=-1)
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)
        
        cos_angle = torch.clamp(
            torch.sum(normals_from * normals_to, dim=-1, keepdim=True),
            -1.0, 1.0
        )
        sin_angle = axis_norm
        
        # Rodrigues' formula
        k_cross_v = torch.cross(axis, vec_3d, dim=-1)
        k_dot_v = torch.sum(axis * vec_3d, dim=-1, keepdim=True)
        
        transported_3d = (
            vec_3d * cos_angle +
            k_cross_v * sin_angle +
            axis * k_dot_v * (1 - cos_angle)
        )
        
        # Project to target frame
        transported = torch.stack([
            torch.sum(transported_3d * t1_to, dim=-1),
            torch.sum(transported_3d * t2_to, dim=-1),
        ], dim=-1)
        
        # Rotation angle for caching
        rotation_angles = torch.atan2(sin_angle.squeeze(), cos_angle.squeeze())
        
        return transported, rotation_angles


def compute_connection_1form(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    vertex_normals: np.ndarray,
) -> np.ndarray:
    """
    Compute the discrete connection 1-form on edges.
    
    The connection 1-form measures how much the tangent frame rotates
    when moving along an edge. This is the discrete analog of the
    Christoffel symbols.
    
    Args:
        mesh_vertices: (V, 3) vertex positions
        mesh_faces: (F, 3) face indices
        vertex_normals: (V, 3) vertex normals
        
    Returns:
        connection: (E,) rotation angle per edge
    """
    # Build edge list
    edge_set = set()
    for face in mesh_faces:
        i, j, k = face
        for a, b in [(i, j), (j, k), (k, i)]:
            edge_set.add((min(a, b), max(a, b)))
    
    edges = np.array(list(edge_set))
    E = len(edges)
    
    # Compute tangent bases
    t1, t2 = compute_tangent_basis_np(vertex_normals)
    
    connection = np.zeros(E)
    
    for e_idx, (i, j) in enumerate(edges):
        # Get normals and tangent bases
        n_i, n_j = vertex_normals[i], vertex_normals[j]
        t1_i, t2_i = t1[i], t2[i]
        t1_j, t2_j = t1[j], t2[j]
        
        # Transport t1_i to vertex j and measure angle with t1_j
        test_vec = np.array([1.0, 0.0])  # t1 direction in local coords
        transported = parallel_transport(
            test_vec, n_i, n_j, t1_i, t2_i, t1_j, t2_j
        )
        
        # Angle of transported vector relative to t1_j
        connection[e_idx] = np.arctan2(transported[1], transported[0])
    
    return connection


def holonomy_around_vertex(
    vertex_idx: int,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    vertex_normals: np.ndarray,
    vertex_faces: list,
) -> float:
    """
    Compute holonomy (total rotation) around a vertex.
    
    On a flat surface, holonomy is 0 (parallel transport around a loop
    returns the original vector). On a curved surface, holonomy equals
    the integrated Gaussian curvature enclosed by the loop.
    
    This provides a discrete verification of the Gauss-Bonnet theorem.
    
    Args:
        vertex_idx: Central vertex index
        mesh_vertices: (V, 3) vertex positions
        mesh_faces: (F, 3) face indices
        vertex_normals: (V, 3) vertex normals
        vertex_faces: List of face indices adjacent to each vertex
        
    Returns:
        holonomy: Total rotation angle (radians)
    """
    # Get ordered neighbors around the vertex
    adjacent_faces = vertex_faces[vertex_idx]
    
    if len(adjacent_faces) == 0:
        return 0.0
    
    # Collect ordered neighbors
    neighbors = []
    for f_idx in adjacent_faces:
        face = mesh_faces[f_idx]
        for v in face:
            if v != vertex_idx and v not in neighbors:
                neighbors.append(v)
    
    if len(neighbors) < 2:
        return 0.0
    
    # Compute tangent bases
    t1, t2 = compute_tangent_basis_np(vertex_normals)
    
    # Transport a vector around the loop
    test_vec = np.array([1.0, 0.0])
    current_vec = test_vec.copy()
    
    for i in range(len(neighbors)):
        j = (i + 1) % len(neighbors)
        v_i, v_j = neighbors[i], neighbors[j]
        
        # Transport from v_i to v_j
        current_vec = parallel_transport(
            current_vec,
            vertex_normals[v_i], vertex_normals[v_j],
            t1[v_i], t2[v_i], t1[v_j], t2[v_j],
        )
    
    # Holonomy is the angle between original and transported vector
    holonomy = np.arctan2(current_vec[1], current_vec[0]) - np.arctan2(test_vec[1], test_vec[0])
    
    # Normalize to [-π, π]
    while holonomy > np.pi:
        holonomy -= 2 * np.pi
    while holonomy < -np.pi:
        holonomy += 2 * np.pi
    
    return holonomy
