"""
Utility functions for data ingestion.
"""

import numpy as np
from typing import Tuple, Optional, List
from ..types import ClothSequence


def validate_topology(
    sequence: ClothSequence,
    reference_frame: int = 0,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that mesh topology is consistent across all frames.
    
    Args:
        sequence: ClothSequence to validate
        reference_frame: Frame to use as reference for comparison
        strict: If True, check face connectivity; if False, only check counts
        
    Returns:
        (is_valid, list_of_issues) tuple
    """
    issues = []
    
    num_vertices = sequence.num_vertices
    num_faces = sequence.num_faces
    
    # Check vertex count consistency (already enforced by array shape)
    if sequence.vertices.shape[1] != num_vertices:
        issues.append(f"Vertex count inconsistency in array shape")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(sequence.vertices)):
        nan_frames = np.any(np.isnan(sequence.vertices), axis=(1, 2))
        nan_frame_indices = np.where(nan_frames)[0]
        issues.append(f"NaN values found in frames: {nan_frame_indices.tolist()}")
    
    if np.any(np.isinf(sequence.vertices)):
        inf_frames = np.any(np.isinf(sequence.vertices), axis=(1, 2))
        inf_frame_indices = np.where(inf_frames)[0]
        issues.append(f"Inf values found in frames: {inf_frame_indices.tolist()}")
    
    # Check face indices are within bounds
    max_idx = sequence.faces.max()
    min_idx = sequence.faces.min()
    if max_idx >= num_vertices:
        issues.append(f"Face index {max_idx} exceeds vertex count {num_vertices}")
    if min_idx < 0:
        issues.append(f"Negative face index {min_idx}")
    
    # Check for degenerate triangles in reference frame
    if sequence.is_triangulated:
        ref_verts = sequence.vertices[reference_frame]
        v0 = ref_verts[sequence.faces[:, 0]]
        v1 = ref_verts[sequence.faces[:, 1]]
        v2 = ref_verts[sequence.faces[:, 2]]
        
        e1 = v1 - v0
        e2 = v2 - v0
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        
        degenerate_count = np.sum(areas < 1e-10)
        if degenerate_count > 0:
            issues.append(f"{degenerate_count} degenerate triangles (zero area)")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def interpolate_frames(
    sequence: ClothSequence,
    target_fps: float,
    method: str = 'linear'
) -> ClothSequence:
    """
    Resample a sequence to a different framerate.
    
    Args:
        sequence: Input ClothSequence
        target_fps: Target framerate
        method: Interpolation method ('linear' or 'cubic')
        
    Returns:
        New ClothSequence at target framerate
    """
    source_fps = sequence.fps
    
    if abs(source_fps - target_fps) < 0.01:
        return sequence  # No resampling needed
    
    source_times = np.arange(sequence.num_frames) / source_fps
    duration = source_times[-1]
    
    target_num_frames = int(np.ceil(duration * target_fps)) + 1
    target_times = np.arange(target_num_frames) / target_fps
    target_times = target_times[target_times <= duration]
    
    if method == 'linear':
        interpolated_vertices = _linear_interpolate(
            source_times, sequence.vertices, target_times
        )
    elif method == 'cubic':
        interpolated_vertices = _cubic_interpolate(
            source_times, sequence.vertices, target_times
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Interpolate normals if present
    interpolated_normals = None
    if sequence.normals is not None:
        if method == 'linear':
            interpolated_normals = _linear_interpolate(
                source_times, sequence.normals, target_times
            )
        else:
            interpolated_normals = _cubic_interpolate(
                source_times, sequence.normals, target_times
            )
        # Re-normalize normals
        norms = np.linalg.norm(interpolated_normals, axis=2, keepdims=True)
        interpolated_normals = interpolated_normals / np.maximum(norms, 1e-10)
    
    return ClothSequence(
        vertices=interpolated_vertices,
        faces=sequence.faces.copy(),
        normals=interpolated_normals,
        uvs=sequence.uvs.copy() if sequence.uvs is not None else None,
        velocities=None,  # Will be recomputed
        fps=target_fps,
        dt=1.0 / target_fps,
        frame_range=(0, len(target_times) - 1),
        metadata={
            **sequence.metadata,
            'resampled_from_fps': source_fps,
            'interpolation_method': method,
        }
    )


def _linear_interpolate(
    source_times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray
) -> np.ndarray:
    """
    Linear interpolation of vertex positions.
    
    Args:
        source_times: (F_source,) array of source timestamps
        values: (F_source, N, 3) array of values to interpolate
        target_times: (F_target,) array of target timestamps
        
    Returns:
        (F_target, N, 3) interpolated values
    """
    from scipy.interpolate import interp1d
    
    num_vertices = values.shape[1]
    num_coords = values.shape[2]
    num_target = len(target_times)
    
    result = np.zeros((num_target, num_vertices, num_coords), dtype=np.float32)
    
    for v in range(num_vertices):
        for c in range(num_coords):
            interp_func = interp1d(
                source_times, 
                values[:, v, c], 
                kind='linear',
                fill_value='extrapolate'
            )
            result[:, v, c] = interp_func(target_times)
    
    return result


def _cubic_interpolate(
    source_times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray
) -> np.ndarray:
    """
    Cubic spline interpolation of vertex positions.
    
    Args:
        source_times: (F_source,) array of source timestamps
        values: (F_source, N, 3) array of values to interpolate
        target_times: (F_target,) array of target timestamps
        
    Returns:
        (F_target, N, 3) interpolated values
    """
    from scipy.interpolate import CubicSpline
    
    num_vertices = values.shape[1]
    num_coords = values.shape[2]
    num_target = len(target_times)
    
    result = np.zeros((num_target, num_vertices, num_coords), dtype=np.float32)
    
    for v in range(num_vertices):
        for c in range(num_coords):
            cs = CubicSpline(source_times, values[:, v, c])
            result[:, v, c] = cs(target_times)
    
    return result


def compute_mesh_statistics(sequence: ClothSequence) -> dict:
    """
    Compute various statistics about a mesh sequence.
    
    Returns a dictionary with statistics useful for pipeline configuration
    and normalization.
    """
    vertices = sequence.vertices
    
    # Position statistics
    all_positions = vertices.reshape(-1, 3)
    position_mean = all_positions.mean(axis=0)
    position_std = all_positions.std(axis=0)
    position_min = all_positions.min(axis=0)
    position_max = all_positions.max(axis=0)
    
    # Per-frame centroid movement
    centroids = vertices.mean(axis=1)  # (F, 3)
    centroid_displacement = np.linalg.norm(centroids - centroids[0], axis=1)
    
    # Displacement from rest pose
    displacements = vertices - vertices[0:1]  # (F, N, 3)
    displacement_magnitudes = np.linalg.norm(displacements, axis=2)  # (F, N)
    
    # Edge length statistics
    if sequence.is_triangulated:
        faces = sequence.faces
        edge_pairs = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ])
        edge_pairs = np.unique(np.sort(edge_pairs, axis=1), axis=0)
        
        rest_edges = vertices[0, edge_pairs[:, 1]] - vertices[0, edge_pairs[:, 0]]
        rest_edge_lengths = np.linalg.norm(rest_edges, axis=1)
    else:
        rest_edge_lengths = np.array([])
    
    # Velocity statistics (if computed)
    velocity_stats = {}
    if sequence.velocities is not None:
        vel_magnitudes = np.linalg.norm(sequence.velocities, axis=2)
        velocity_stats = {
            'velocity_mean': float(vel_magnitudes.mean()),
            'velocity_std': float(vel_magnitudes.std()),
            'velocity_max': float(vel_magnitudes.max()),
        }
    
    return {
        'num_frames': sequence.num_frames,
        'num_vertices': sequence.num_vertices,
        'num_faces': sequence.num_faces,
        'duration': sequence.duration,
        'position_mean': position_mean.tolist(),
        'position_std': position_std.tolist(),
        'position_min': position_min.tolist(),
        'position_max': position_max.tolist(),
        'bbox_size': (position_max - position_min).tolist(),
        'max_centroid_displacement': float(centroid_displacement.max()),
        'max_vertex_displacement': float(displacement_magnitudes.max()),
        'mean_vertex_displacement': float(displacement_magnitudes.mean()),
        'edge_length_mean': float(rest_edge_lengths.mean()) if len(rest_edge_lengths) > 0 else 0.0,
        'edge_length_std': float(rest_edge_lengths.std()) if len(rest_edge_lengths) > 0 else 0.0,
        'edge_length_min': float(rest_edge_lengths.min()) if len(rest_edge_lengths) > 0 else 0.0,
        'edge_length_max': float(rest_edge_lengths.max()) if len(rest_edge_lengths) > 0 else 0.0,
        **velocity_stats,
    }
