"""
SDF (Signed Distance Field) computation for cloth mesh sequences.

Handles the challenge of computing SDFs for open cloth surfaces
(thin sheets, not closed volumes) using two approaches:
1. Thickened mesh: Extrude the surface to create a closed shell
2. Unsigned distance: Distance to nearest surface point with sign from normal
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from scipy.spatial import KDTree

from ..types import ClothSequence


@dataclass
class SDFData:
    """
    SDF computation results.
    
    Attributes:
        volume: Dense SDF grid. Shape: (res, res, res) or (F, res, res, res)
        samples: Point cloud SDF samples. Shape: (N, 4) where columns are (x, y, z, sdf)
        surface_samples: Samples specifically near the surface
        volume_samples: Samples throughout the bounding volume
        bounds: (min_corner, max_corner) of the SDF domain
        resolution: Grid resolution
        method: 'thickened' or 'unsigned'
    """
    volume: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
    surface_samples: Optional[np.ndarray] = None
    volume_samples: Optional[np.ndarray] = None
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    resolution: int = 64
    method: str = 'thickened'


def compute_sdf_volume(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 64,
    method: str = 'thickened',
    thickness: float = 0.002,
    padding: float = 0.1,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> SDFData:
    """
    Compute a dense SDF volume for a cloth mesh.
    
    Args:
        vertices: (N, 3) vertex positions for a single frame
        faces: (F, 3) triangle face indices
        resolution: Grid resolution per dimension
        method: 'thickened' for closed-shell SDF, 'unsigned' for distance field
        thickness: Shell thickness for 'thickened' method (meters)
        padding: Relative padding around mesh bounding box
        bounds: Optional explicit (min, max) bounds for the SDF domain
        
    Returns:
        SDFData with volume field
    """
    # Compute or use provided bounds
    if bounds is None:
        mesh_min = vertices.min(axis=0)
        mesh_max = vertices.max(axis=0)
        mesh_size = mesh_max - mesh_min
        pad = mesh_size * padding
        bounds = (mesh_min - pad, mesh_max + pad)
    
    bbox_min, bbox_max = bounds
    
    # Create grid coordinates
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    
    # Create meshgrid of query points
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # Compute SDF values
    if method == 'thickened':
        sdf_values = _compute_thickened_sdf(
            query_points, vertices, faces, thickness
        )
    elif method == 'unsigned':
        sdf_values = _compute_unsigned_distance(
            query_points, vertices, faces
        )
    else:
        raise ValueError(f"Unknown SDF method: {method}")
    
    # Reshape to volume
    volume = sdf_values.reshape(resolution, resolution, resolution)
    
    return SDFData(
        volume=volume.astype(np.float32),
        bounds=bounds,
        resolution=resolution,
        method=method,
    )


def compute_sdf_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int = 50000,
    near_surface_ratio: float = 0.7,
    near_surface_distance: float = 0.01,
    method: str = 'thickened',
    thickness: float = 0.002,
    padding: float = 0.1,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    include_normals: bool = False,
) -> SDFData:
    """
    Compute SDF at randomly sampled points.
    
    This is more efficient than dense grids for training neural implicit fields.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle face indices
        num_samples: Total number of samples to generate
        near_surface_ratio: Fraction of samples near the surface (vs. volume)
        near_surface_distance: "Near surface" threshold in meters
        method: 'thickened' or 'unsigned'
        thickness: Shell thickness for 'thickened' method
        padding: Relative padding for bounding box
        bounds: Optional explicit bounds
        include_normals: If True, also return surface normals at nearest points
        
    Returns:
        SDFData with sample arrays
    """
    # Compute bounds
    if bounds is None:
        mesh_min = vertices.min(axis=0)
        mesh_max = vertices.max(axis=0)
        mesh_size = mesh_max - mesh_min
        pad = mesh_size * padding
        bounds = (mesh_min - pad, mesh_max + pad)
    
    bbox_min, bbox_max = bounds
    
    num_surface = int(num_samples * near_surface_ratio)
    num_volume = num_samples - num_surface
    
    # Generate near-surface samples
    surface_points = _sample_near_surface(
        vertices, faces, num_surface, near_surface_distance
    )
    
    # Generate volume samples
    volume_points = np.random.uniform(
        bbox_min, bbox_max, size=(num_volume, 3)
    ).astype(np.float32)
    
    # Combine all sample points
    all_points = np.vstack([surface_points, volume_points])
    
    # Compute SDF values
    if method == 'thickened':
        sdf_values = _compute_thickened_sdf(
            all_points, vertices, faces, thickness
        )
    else:
        sdf_values = _compute_unsigned_distance(
            all_points, vertices, faces
        )
    
    # Combine into (N, 4) array: x, y, z, sdf
    samples = np.column_stack([all_points, sdf_values]).astype(np.float32)
    
    # Split for separate access
    surface_samples = samples[:num_surface]
    volume_samples = samples[num_surface:]
    
    return SDFData(
        samples=samples,
        surface_samples=surface_samples,
        volume_samples=volume_samples,
        bounds=bounds,
        method=method,
    )


def _compute_thickened_sdf(
    query_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """
    Compute SDF for a thickened cloth surface.
    
    The cloth is treated as a thin shell with the given thickness.
    Points inside the shell have negative SDF, outside have positive.
    
    For a thin sheet:
    - Compute unsigned distance to the surface
    - The SDF is: |distance| - thickness/2
    - Sign is determined by which side of the surface the point is on
    """
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / np.maximum(
        np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-10
    )
    
    # Compute face centroids for KDTree
    face_centroids = (v0 + v1 + v2) / 3.0
    
    # Build KDTree on face centroids for fast nearest face lookup
    tree = KDTree(face_centroids)
    
    # For each query point, find nearest face
    distances, nearest_face_indices = tree.query(query_points, k=1)
    
    # Compute exact distance to nearest triangle
    sdf_values = np.zeros(len(query_points), dtype=np.float32)
    
    for i, (point, face_idx) in enumerate(zip(query_points, nearest_face_indices)):
        # Get the triangle vertices
        tri_v0 = vertices[faces[face_idx, 0]]
        tri_v1 = vertices[faces[face_idx, 1]]
        tri_v2 = vertices[faces[face_idx, 2]]
        
        # Compute exact point-to-triangle distance
        closest_point, dist = _point_to_triangle_distance(
            point, tri_v0, tri_v1, tri_v2
        )
        
        # Determine sign based on which side of the surface
        face_normal = face_normals[face_idx]
        to_point = point - closest_point
        dot = np.dot(to_point, face_normal)
        
        # For thickened shell: SDF = |dist| - thickness/2
        # If inside the shell (|dist| < thickness/2), SDF is negative
        shell_sdf = dist - thickness / 2.0
        
        sdf_values[i] = shell_sdf if dot >= 0 else -shell_sdf
    
    return sdf_values


def _compute_unsigned_distance(
    query_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute unsigned distance to the mesh surface.
    
    This is simpler than signed distance for open surfaces.
    """
    # Compute face centroids
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_centroids = (v0 + v1 + v2) / 3.0
    
    # Build KDTree
    tree = KDTree(face_centroids)
    
    # Find nearest faces
    _, nearest_face_indices = tree.query(query_points, k=1)
    
    # Compute exact distances
    distances = np.zeros(len(query_points), dtype=np.float32)
    
    for i, (point, face_idx) in enumerate(zip(query_points, nearest_face_indices)):
        tri_v0 = vertices[faces[face_idx, 0]]
        tri_v1 = vertices[faces[face_idx, 1]]
        tri_v2 = vertices[faces[face_idx, 2]]
        
        _, dist = _point_to_triangle_distance(point, tri_v0, tri_v1, tri_v2)
        distances[i] = dist
    
    return distances


def _point_to_triangle_distance(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute the closest point on a triangle to a query point.
    
    Returns (closest_point, distance).
    
    Uses the method from "Real-Time Collision Detection" by Christer Ericson.
    """
    # Compute vectors
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0
    
    # Check if P is in vertex region outside A
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        # Closest to v0
        return v0, np.linalg.norm(point - v0)
    
    # Check if P is in vertex region outside B
    bp = point - v1
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        # Closest to v1
        return v1, np.linalg.norm(point - v1)
    
    # Check if P is in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        closest = v0 + v * ab
        return closest, np.linalg.norm(point - closest)
    
    # Check if P is in vertex region outside C
    cp = point - v2
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        # Closest to v2
        return v2, np.linalg.norm(point - v2)
    
    # Check if P is in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        closest = v0 + w * ac
        return closest, np.linalg.norm(point - closest)
    
    # Check if P is in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = v1 + w * (v2 - v1)
        return closest, np.linalg.norm(point - closest)
    
    # P is inside the triangle
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    closest = v0 + ab * v + ac * w
    return closest, np.linalg.norm(point - closest)


def _sample_near_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    max_distance: float,
) -> np.ndarray:
    """
    Sample points uniformly near the mesh surface.
    
    Strategy:
    1. Sample points uniformly on the triangle surfaces
    2. Perturb them by random offsets within max_distance
    """
    # Compute face areas for weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Face areas (half the cross product magnitude)
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Normalize to get sampling probabilities
    probs = areas / areas.sum()
    
    # Sample faces according to their areas
    sampled_faces = np.random.choice(len(faces), size=num_samples, p=probs)
    
    # Sample random barycentric coordinates
    r1 = np.sqrt(np.random.rand(num_samples))
    r2 = np.random.rand(num_samples)
    
    # Barycentric coords: (1-r1, r1*(1-r2), r1*r2)
    bary_a = 1 - r1
    bary_b = r1 * (1 - r2)
    bary_c = r1 * r2
    
    # Get sampled triangle vertices
    sampled_v0 = vertices[faces[sampled_faces, 0]]
    sampled_v1 = vertices[faces[sampled_faces, 1]]
    sampled_v2 = vertices[faces[sampled_faces, 2]]
    
    # Compute surface points
    surface_points = (
        bary_a[:, np.newaxis] * sampled_v0 +
        bary_b[:, np.newaxis] * sampled_v1 +
        bary_c[:, np.newaxis] * sampled_v2
    )
    
    # Add random perturbation
    perturbation = np.random.uniform(-max_distance, max_distance, size=(num_samples, 3))
    near_surface_points = surface_points + perturbation
    
    return near_surface_points.astype(np.float32)


def compute_sdf_sequence(
    sequence: ClothSequence,
    resolution: int = 64,
    method: str = 'thickened',
    thickness: float = 0.002,
    padding: float = 0.1,
    frames: Optional[Union[str, list]] = None,
) -> np.ndarray:
    """
    Compute SDF volumes for multiple frames of a sequence.
    
    Args:
        sequence: ClothSequence to process
        resolution: Grid resolution
        method: SDF computation method
        thickness: Shell thickness
        padding: Bounding box padding
        frames: Which frames to process:
            - None or 'all': All frames
            - 'keyframes': First and last frame
            - List of ints: Specific frame indices
            - 'every_nth:N': Every Nth frame
            
    Returns:
        SDF volumes array of shape (num_selected_frames, res, res, res)
    """
    # Determine which frames to process
    if frames is None or frames == 'all':
        frame_indices = list(range(sequence.num_frames))
    elif frames == 'keyframes':
        frame_indices = [0, sequence.num_frames - 1]
    elif isinstance(frames, list):
        frame_indices = frames
    elif isinstance(frames, str) and frames.startswith('every_nth:'):
        n = int(frames.split(':')[1])
        frame_indices = list(range(0, sequence.num_frames, n))
    else:
        raise ValueError(f"Unknown frames specification: {frames}")
    
    # Compute global bounds
    bbox_min, bbox_max = sequence.get_bounding_box()
    mesh_size = bbox_max - bbox_min
    pad = mesh_size * padding
    bounds = (bbox_min - pad, bbox_max + pad)
    
    # Triangulate if needed
    seq = sequence.triangulate()
    
    # Compute SDF for each frame
    volumes = []
    for frame_idx in frame_indices:
        sdf_data = compute_sdf_volume(
            vertices=seq.vertices[frame_idx],
            faces=seq.faces,
            resolution=resolution,
            method=method,
            thickness=thickness,
            bounds=bounds,
        )
        volumes.append(sdf_data.volume)
    
    return np.stack(volumes, axis=0)
