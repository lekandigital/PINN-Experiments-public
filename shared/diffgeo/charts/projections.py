"""
Chart Projection Functions
==========================

Various projection methods for mapping 3D manifold points to 2D local coordinates.
Each projection has different properties (distortion, accuracy, applicability).

Projections:
- Tangent plane: Simple orthogonal projection onto tangent plane
- Stereographic: Conformal (angle-preserving), good for sphere patches
- Gnomonic: Preserves directions from center, used in navigation
- Exponential map: Geodesic-based, most accurate for small regions
"""

import numpy as np
from typing import Tuple, Optional


def tangent_plane_projection(
    points: np.ndarray,
    center: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
) -> np.ndarray:
    """
    Project points onto tangent plane at center.
    
    Simple orthogonal projection - fast but introduces distortion
    for points far from the center.
    
    Args:
        points: (M, 3) points to project
        center: (3,) center of projection
        tangent1: (3,) first tangent basis vector
        tangent2: (3,) second tangent basis vector
        
    Returns:
        coords: (M, 2) local coordinates
    """
    diff = points - center
    u = diff @ tangent1
    v = diff @ tangent2
    return np.stack([u, v], axis=-1)


def stereographic_projection(
    points: np.ndarray,
    pole: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Stereographic projection from a pole.
    
    Conformal (preserves angles locally) and maps circles to circles.
    Standard for sphere patches and complex analysis on manifolds.
    
    Args:
        points: (M, 3) points on sphere to project
        pole: (3,) projection pole (south pole maps to plane at equator)
        tangent1: (3,) first tangent basis vector at equator
        tangent2: (3,) second tangent basis vector at equator
        radius: Sphere radius
        
    Returns:
        coords: (M, 2) stereographic coordinates
    """
    # Normalize pole
    pole = pole / (np.linalg.norm(pole) + 1e-10)
    
    # Distance from pole along sphere
    # Stereographic: project from pole through point to plane
    dot = np.sum(points * pole, axis=-1, keepdims=True)  # (M, 1)
    
    # Projection factor: 1 / (1 - cos(θ)) where θ is angle from pole
    # For points at position p on unit sphere, factor = 2R / (1 + p·pole)
    # if pole is at (0,0,-1) (south pole) and plane is at z=0
    factor = 2 * radius / (radius - dot / radius + 1e-10)  # (M, 1)
    
    # Project onto plane orthogonal to pole
    projected = points - dot * pole  # Remove pole component
    projected = projected * factor  # Scale by stereographic factor
    
    # Express in tangent basis
    u = np.sum(projected * tangent1, axis=-1)
    v = np.sum(projected * tangent2, axis=-1)
    
    return np.stack([u, v], axis=-1)


def gnomonic_projection(
    points: np.ndarray,
    center: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Gnomonic (central) projection.
    
    Projects from sphere center through points to tangent plane.
    Preserves great circles (geodesics become straight lines).
    Standard for navigation and coastal charts.
    
    Limited to hemisphere - undefined for points on opposite side.
    
    Args:
        points: (M, 3) points on sphere to project
        center: (3,) tangent point (where plane touches sphere)
        tangent1: (3,) first tangent basis vector
        tangent2: (3,) second tangent basis vector
        radius: Sphere radius
        
    Returns:
        coords: (M, 2) gnomonic coordinates
    """
    # Normalize center (it's on the sphere surface)
    center_norm = center / (np.linalg.norm(center) + 1e-10)
    
    # Compute angle from center for each point
    cos_theta = np.sum(points * center_norm, axis=-1, keepdims=True) / radius  # (M, 1)
    
    # Clamp to avoid division issues for points near opposite hemisphere
    cos_theta = np.clip(cos_theta, 1e-6, 1.0)
    
    # Gnomonic projection factor: 1 / cos(θ)
    factor = 1.0 / cos_theta  # (M, 1)
    
    # Project to tangent plane
    # The tangent plane passes through 'center' and is perpendicular to center_norm
    diff = points - center  # Vector from center to point
    
    # Project diff onto tangent plane and scale
    # First, remove component along normal
    normal_component = np.sum(diff * center_norm, axis=-1, keepdims=True) * center_norm
    tangent_diff = diff - normal_component
    
    # Scale by gnomonic factor
    projected = tangent_diff * factor
    
    # Express in tangent basis
    u = np.sum(projected * tangent1, axis=-1)
    v = np.sum(projected * tangent2, axis=-1)
    
    return np.stack([u, v], axis=-1)


def exponential_map_projection(
    points: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Exponential map (logarithm) projection.
    
    Computes geodesic distance and direction from center to each point.
    Most accurate representation of intrinsic geometry for small regions.
    
    For a sphere: uses analytical formula.
    For general manifolds: requires iterative computation.
    
    Args:
        points: (M, 3) points to project
        center: (3,) base point
        normal: (3,) surface normal at center
        tangent1: (3,) first tangent basis vector
        tangent2: (3,) second tangent basis vector
        max_iterations: For iterative computation (not used for spheres)
        
    Returns:
        coords: (M, 2) exponential map coordinates (direction + distance)
    """
    # Detect if this is a sphere (all points at same distance from origin)
    center_dist = np.linalg.norm(center)
    point_dists = np.linalg.norm(points, axis=-1)
    
    # Check if points lie on a sphere centered at origin
    if np.allclose(point_dists, center_dist, rtol=0.01):
        # Analytical exponential map for sphere
        return _exp_map_sphere(points, center, tangent1, tangent2, center_dist)
    else:
        # General case: approximate with tangent plane projection
        # (Full exponential map requires mesh connectivity for geodesics)
        return tangent_plane_projection(points, center, tangent1, tangent2)


def _exp_map_sphere(
    points: np.ndarray,
    center: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Analytical exponential map for sphere.
    
    On a sphere, the exponential map is:
    exp_p(v) = cos(|v|/R) * p + sin(|v|/R) * R * v/|v|
    
    The inverse (logarithm) gives local coordinates.
    """
    # Normalize positions to unit sphere
    center_unit = center / radius
    points_unit = points / radius
    
    # Compute angle between center and each point
    cos_angle = np.clip(np.sum(center_unit * points_unit, axis=-1), -1.0, 1.0)
    angle = np.arccos(cos_angle)  # Geodesic angle
    
    # Direction in tangent plane
    # v = points - (points · center) * center, then normalize
    proj = np.sum(points_unit * center_unit, axis=-1, keepdims=True) * center_unit
    direction = points_unit - proj
    dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.where(dir_norm > 1e-10, direction / dir_norm, direction)
    
    # Geodesic distance
    geodesic_dist = angle * radius
    
    # Express direction in tangent basis
    u_dir = np.sum(direction * tangent1, axis=-1)
    v_dir = np.sum(direction * tangent2, axis=-1)
    
    # Local coordinates = direction * distance
    u = u_dir * geodesic_dist
    v = v_dir * geodesic_dist
    
    return np.stack([u, v], axis=-1)


def inverse_gnomonic_projection(
    coords: np.ndarray,
    center: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Inverse gnomonic projection: 2D coords back to 3D sphere.
    
    Args:
        coords: (M, 2) gnomonic coordinates
        center: (3,) tangent point on sphere
        tangent1: (3,) first tangent basis vector
        tangent2: (3,) second tangent basis vector
        radius: Sphere radius
        
    Returns:
        points: (M, 3) points on sphere
    """
    u, v = coords[:, 0], coords[:, 1]
    
    # Point in tangent plane
    plane_point = center + u[:, None] * tangent1 + v[:, None] * tangent2
    
    # Project back to sphere
    dist = np.linalg.norm(plane_point, axis=-1, keepdims=True)
    points = plane_point * (radius / dist)
    
    return points


def compute_projection_distortion(
    points: np.ndarray,
    coords: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute metric distortion introduced by projection.
    
    Compares edge length ratios in 3D vs 2D coordinates.
    Returns per-vertex distortion measure (1.0 = no distortion).
    
    Args:
        points: (V, 3) original 3D positions
        coords: (V, 2) projected 2D coordinates
        faces: (F, 3) face indices
        
    Returns:
        distortion: (V,) distortion factor per vertex
    """
    # Collect edges from faces
    edges = set()
    for face in faces:
        i, j, k = face
        edges.add((min(i, j), max(i, j)))
        edges.add((min(j, k), max(j, k)))
        edges.add((min(k, i), max(k, i)))
    
    edges = np.array(list(edges))
    
    # Compute edge lengths in 3D and 2D
    len_3d = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=-1)
    len_2d = np.linalg.norm(coords[edges[:, 1]] - coords[edges[:, 0]], axis=-1)
    
    # Ratio of lengths (scale factor)
    ratios = len_2d / (len_3d + 1e-10)
    
    # Median ratio gives the expected scale
    median_ratio = np.median(ratios)
    
    # Distortion = deviation from median scale
    edge_distortion = np.abs(ratios - median_ratio) / median_ratio
    
    # Aggregate to vertices
    V = points.shape[0]
    vertex_distortion = np.zeros(V)
    vertex_count = np.zeros(V)
    
    for idx, (i, j) in enumerate(edges):
        vertex_distortion[i] += edge_distortion[idx]
        vertex_distortion[j] += edge_distortion[idx]
        vertex_count[i] += 1
        vertex_count[j] += 1
    
    vertex_distortion = vertex_distortion / (vertex_count + 1e-10)
    
    return vertex_distortion
