"""
Vertex feature computation for cloth mesh sequences.

Computes per-vertex features like curvature and normals for
use in GNN-based models.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..types import ClothSequence


def compute_vertex_features(
    sequence: ClothSequence,
    features: list = ['position', 'normal', 'curvature'],
    rest_frame: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Compute various vertex features for all frames.
    
    Args:
        sequence: Input ClothSequence
        features: List of features to compute:
            - 'position': Raw vertex positions
            - 'displacement': Displacement from rest pose
            - 'normal': Vertex normals
            - 'curvature': Mean curvature
            - 'gaussian_curvature': Gaussian curvature
            - 'uv': UV coordinates (if available)
        rest_frame: Which frame to use as rest pose
        
    Returns:
        Dictionary mapping feature names to arrays
    """
    result = {}
    
    if 'position' in features:
        result['position'] = sequence.vertices.astype(np.float32)
    
    if 'displacement' in features:
        rest_pose = sequence.vertices[rest_frame]
        result['displacement'] = (sequence.vertices - rest_pose).astype(np.float32)
    
    if 'normal' in features:
        if sequence.normals is not None:
            result['normal'] = sequence.normals.astype(np.float32)
        else:
            result['normal'] = compute_normals(sequence)
    
    if 'curvature' in features or 'mean_curvature' in features:
        result['curvature'] = compute_curvature(sequence, method='mean')
    
    if 'gaussian_curvature' in features:
        result['gaussian_curvature'] = compute_curvature(sequence, method='gaussian')
    
    if 'uv' in features and sequence.uvs is not None:
        # UVs are per-vertex, constant across frames
        # Broadcast to all frames for consistency
        result['uv'] = np.broadcast_to(
            sequence.uvs[np.newaxis, :, :],
            (sequence.num_frames, sequence.num_vertices, 2)
        ).astype(np.float32)
    
    return result


def compute_normals(sequence: ClothSequence) -> np.ndarray:
    """
    Compute vertex normals for all frames.
    
    Uses area-weighted average of adjacent face normals.
    
    Args:
        sequence: Input ClothSequence
        
    Returns:
        Normals array of shape (F, N, 3)
    """
    if not sequence.is_triangulated:
        sequence = sequence.triangulate()
    
    faces = sequence.faces
    num_frames = sequence.num_frames
    num_vertices = sequence.num_vertices
    
    normals = np.zeros((num_frames, num_vertices, 3), dtype=np.float32)
    
    for f in range(num_frames):
        normals[f] = _compute_frame_normals(sequence.vertices[f], faces)
    
    return normals


def _compute_frame_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals for a single frame."""
    num_vertices = vertices.shape[0]
    vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)
    
    # Get face vertex positions
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Face normals (magnitude = 2 * area)
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # Accumulate to vertices
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-10)
    
    return vertex_normals


def compute_curvature(
    sequence: ClothSequence,
    method: str = 'mean',
) -> np.ndarray:
    """
    Compute vertex curvature for all frames.
    
    Args:
        sequence: Input ClothSequence
        method: 'mean' for mean curvature, 'gaussian' for Gaussian curvature
        
    Returns:
        Curvature array of shape (F, N)
    """
    if not sequence.is_triangulated:
        sequence = sequence.triangulate()
    
    faces = sequence.faces
    num_frames = sequence.num_frames
    num_vertices = sequence.num_vertices
    
    curvatures = np.zeros((num_frames, num_vertices), dtype=np.float32)
    
    for f in range(num_frames):
        if method == 'mean':
            curvatures[f] = _compute_mean_curvature(sequence.vertices[f], faces)
        elif method == 'gaussian':
            curvatures[f] = _compute_gaussian_curvature(sequence.vertices[f], faces)
        else:
            raise ValueError(f"Unknown curvature method: {method}")
    
    return curvatures


def _compute_mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute mean curvature using the cotangent Laplacian.
    
    Mean curvature H = 0.5 * |Δx| where Δ is the Laplace-Beltrami operator.
    """
    num_vertices = vertices.shape[0]
    
    # Build cotangent weights
    # For edge (i,j) in face (i,j,k), weight = cot(angle at k)
    
    laplacian = np.zeros((num_vertices, 3), dtype=np.float64)
    vertex_areas = np.zeros(num_vertices, dtype=np.float64)
    
    for face in faces:
        i, j, k = face
        
        vi = vertices[i]
        vj = vertices[j]
        vk = vertices[k]
        
        # Edge vectors
        eij = vj - vi
        ejk = vk - vj
        eki = vi - vk
        
        # Cotangent weights
        # cot(angle at i) = dot(eij, -eki) / |cross(eij, -eki)|
        cot_i = np.dot(-eki, eij) / (np.linalg.norm(np.cross(-eki, eij)) + 1e-10)
        cot_j = np.dot(-eij, ejk) / (np.linalg.norm(np.cross(-eij, ejk)) + 1e-10)
        cot_k = np.dot(-ejk, eki) / (np.linalg.norm(np.cross(-ejk, eki)) + 1e-10)
        
        # Clamp cotangent weights to avoid numerical issues
        cot_i = np.clip(cot_i, -1e6, 1e6)
        cot_j = np.clip(cot_j, -1e6, 1e6)
        cot_k = np.clip(cot_k, -1e6, 1e6)
        
        # Accumulate Laplacian
        # Δvi += cot_k * (vj - vi) + cot_j * (vk - vi)
        laplacian[i] += cot_k * (vj - vi) + cot_j * (vk - vi)
        laplacian[j] += cot_i * (vk - vj) + cot_k * (vi - vj)
        laplacian[k] += cot_j * (vi - vk) + cot_i * (vj - vk)
        
        # Face area (for normalization)
        face_area = 0.5 * np.linalg.norm(np.cross(eij, -eki))
        vertex_areas[i] += face_area / 3
        vertex_areas[j] += face_area / 3
        vertex_areas[k] += face_area / 3
    
    # Normalize by vertex area
    laplacian = laplacian / np.maximum(vertex_areas[:, np.newaxis], 1e-10)
    
    # Mean curvature magnitude
    mean_curvature = 0.5 * np.linalg.norm(laplacian, axis=1)
    
    return mean_curvature.astype(np.float32)


def _compute_gaussian_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute Gaussian curvature using the angle deficit method.
    
    Gaussian curvature K = (2π - Σθ) / A
    where θ are the angles at the vertex and A is the vertex area.
    """
    num_vertices = vertices.shape[0]
    
    angle_sum = np.zeros(num_vertices, dtype=np.float64)
    vertex_areas = np.zeros(num_vertices, dtype=np.float64)
    
    for face in faces:
        i, j, k = face
        
        vi = vertices[i]
        vj = vertices[j]
        vk = vertices[k]
        
        # Edge vectors from each vertex
        eij = vj - vi
        eik = vk - vi
        eji = vi - vj
        ejk = vk - vj
        eki = vi - vk
        ekj = vj - vk
        
        # Angles at each vertex
        cos_i = np.dot(eij, eik) / (np.linalg.norm(eij) * np.linalg.norm(eik) + 1e-10)
        cos_j = np.dot(eji, ejk) / (np.linalg.norm(eji) * np.linalg.norm(ejk) + 1e-10)
        cos_k = np.dot(eki, ekj) / (np.linalg.norm(eki) * np.linalg.norm(ekj) + 1e-10)
        
        angle_i = np.arccos(np.clip(cos_i, -1, 1))
        angle_j = np.arccos(np.clip(cos_j, -1, 1))
        angle_k = np.arccos(np.clip(cos_k, -1, 1))
        
        angle_sum[i] += angle_i
        angle_sum[j] += angle_j
        angle_sum[k] += angle_k
        
        # Face area
        face_area = 0.5 * np.linalg.norm(np.cross(eij, eik))
        vertex_areas[i] += face_area / 3
        vertex_areas[j] += face_area / 3
        vertex_areas[k] += face_area / 3
    
    # Gaussian curvature = angle deficit / area
    gaussian_curvature = (2 * np.pi - angle_sum) / np.maximum(vertex_areas, 1e-10)
    
    return gaussian_curvature.astype(np.float32)


def compute_edge_strain(
    sequence: ClothSequence,
    rest_frame: int = 0,
) -> np.ndarray:
    """
    Compute edge strain (stretch/compression) relative to rest pose.
    
    Strain = (current_length - rest_length) / rest_length
    
    Args:
        sequence: Input ClothSequence
        rest_frame: Frame to use as rest state
        
    Returns:
        Strain array of shape (F, E) where E is number of unique edges
    """
    if not sequence.is_triangulated:
        sequence = sequence.triangulate()
    
    faces = sequence.faces
    
    # Extract unique edges
    edges_0_1 = faces[:, [0, 1]]
    edges_1_2 = faces[:, [1, 2]]
    edges_2_0 = faces[:, [2, 0]]
    all_edges = np.vstack([edges_0_1, edges_1_2, edges_2_0])
    all_edges = np.sort(all_edges, axis=1)
    edges = np.unique(all_edges, axis=0)
    
    num_frames = sequence.num_frames
    num_edges = edges.shape[0]
    
    # Compute rest edge lengths
    rest_verts = sequence.vertices[rest_frame]
    rest_vectors = rest_verts[edges[:, 1]] - rest_verts[edges[:, 0]]
    rest_lengths = np.linalg.norm(rest_vectors, axis=1)
    
    # Compute strain for each frame
    strains = np.zeros((num_frames, num_edges), dtype=np.float32)
    
    for f in range(num_frames):
        verts = sequence.vertices[f]
        vectors = verts[edges[:, 1]] - verts[edges[:, 0]]
        lengths = np.linalg.norm(vectors, axis=1)
        strains[f] = (lengths - rest_lengths) / np.maximum(rest_lengths, 1e-10)
    
    return strains


def compute_local_frame(
    sequence: ClothSequence,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local coordinate frames at each vertex.
    
    Returns orthonormal (tangent1, tangent2, normal) at each vertex.
    
    Returns:
        Tuple of:
        - tangent1: (F, N, 3) first tangent direction
        - tangent2: (F, N, 3) second tangent direction  
        - normal: (F, N, 3) normal direction
    """
    normals = compute_normals(sequence)
    
    # Choose an arbitrary initial tangent direction
    # (will be orthogonalized to the normal)
    arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # For vertices where normal is parallel to arbitrary, use different direction
    dot = np.abs(np.sum(normals * arbitrary, axis=-1))
    parallel_mask = dot > 0.9
    
    tangent1 = np.zeros_like(normals)
    
    # Cross product to get first tangent
    tangent1[~parallel_mask] = np.cross(normals[~parallel_mask], arbitrary)
    tangent1[parallel_mask] = np.cross(normals[parallel_mask], [0, 1, 0])
    
    # Normalize
    tangent1 = tangent1 / np.maximum(np.linalg.norm(tangent1, axis=-1, keepdims=True), 1e-10)
    
    # Second tangent is cross of normal and first tangent
    tangent2 = np.cross(normals, tangent1)
    
    return tangent1, tangent2, normals
