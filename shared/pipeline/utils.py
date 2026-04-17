"""
Utility functions for the animation pipeline.

Includes quaternion math, mesh utilities, and transform helpers.
Both numpy and torch versions provided where needed.
"""

import numpy as np
from typing import Tuple, Optional, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Quaternion operations (numpy)
# ============================================================================

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (wxyz format).
    
    Args:
        q1: (4,) or (N, 4) quaternions
        q2: (4,) or (N, 4) quaternions
        
    Returns:
        Product quaternion(s) in wxyz format
    """
    if q1.ndim == 1:
        q1 = q1.reshape(1, 4)
        squeeze = True
    else:
        squeeze = False
    
    if q2.ndim == 1:
        q2 = q2.reshape(1, 4)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    result = np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=1)
    
    if squeeze:
        return result.squeeze()
    return result


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate."""
    if q.ndim == 1:
        return np.array([q[0], -q[1], -q[2], -q[3]])
    return np.concatenate([q[:, :1], -q[:, 1:]], axis=1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s)."""
    if q.ndim == 1:
        return q / (np.linalg.norm(q) + 1e-8)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: (4,) quaternion in wxyz format
        
    Returns:
        (3, 3) rotation matrix
    """
    q = quat_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (wxyz).
    
    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return quat_normalize(np.array([w, x, y, z]))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.
    
    Args:
        q0: Start quaternion (wxyz)
        q1: End quaternion (wxyz)
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated quaternion
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    
    dot = np.dot(q0, q1)
    
    # If negative dot, negate one quaternion to take shorter path
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return quat_normalize(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return quat_normalize(s0 * q0 + s1 * q1)


def euler_to_quat(euler: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    
    Args:
        euler: (3,) angles in radians [x, y, z]
        order: Rotation order (default 'XYZ')
        
    Returns:
        (4,) quaternion in wxyz format
    """
    x, y, z = euler[0] / 2, euler[1] / 2, euler[2] / 2
    
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    
    if order == 'XYZ':
        w = cx*cy*cz + sx*sy*sz
        qx = sx*cy*cz - cx*sy*sz
        qy = cx*sy*cz + sx*cy*sz
        qz = cx*cy*sz - sx*sy*cz
    elif order == 'ZYX':
        w = cx*cy*cz - sx*sy*sz
        qx = sx*cy*cz + cx*sy*sz
        qy = cx*sy*cz - sx*cy*sz
        qz = cx*cy*sz + sx*sy*cz
    else:
        raise ValueError(f"Unsupported Euler order: {order}")
    
    return quat_normalize(np.array([w, qx, qy, qz]))


def quat_to_euler(q: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """
    Convert quaternion to Euler angles.
    
    Args:
        q: (4,) quaternion in wxyz format
        order: Rotation order
        
    Returns:
        (3,) Euler angles in radians
    """
    q = quat_normalize(q)
    w, x, y, z = q
    
    if order == 'XYZ':
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        rx = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1, 1)
        ry = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        rz = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([rx, ry, rz])
    else:
        raise ValueError(f"Unsupported Euler order: {order}")


# ============================================================================
# Mesh utilities
# ============================================================================

def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals from mesh.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        
    Returns:
        (V, 3) normalized vertex normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Face normals (not normalized to weight by area)
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # Accumulate to vertices
    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norms + 1e-8)
    
    return vertex_normals


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-face normals.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        
    Returns:
        (F, 3) normalized face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    return normals / (norms + 1e-8)


def build_edge_list(faces: np.ndarray) -> np.ndarray:
    """
    Build edge list from faces.
    
    Args:
        faces: (F, 3) face indices
        
    Returns:
        (E, 2) unique edges (undirected)
    """
    edges = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edges.add(e)
    
    return np.array(list(edges), dtype=np.int64)


def compute_mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute total mesh surface area."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    return areas.sum()


# ============================================================================
# Transform utilities
# ============================================================================

def make_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Create 4x4 transform matrix from rotation and translation.
    
    Args:
        rotation: (3, 3) rotation matrix or (4,) quaternion
        translation: (3,) translation vector
        
    Returns:
        (4, 4) homogeneous transform matrix
    """
    T = np.eye(4)
    
    if rotation.shape == (4,):
        rotation = quat_to_rotation_matrix(rotation)
    
    T[:3, :3] = rotation
    T[:3, 3] = translation
    
    return T


def decompose_transform_matrix(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose 4x4 transform into rotation and translation.
    
    Returns:
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
    """
    return T[:3, :3], T[:3, 3]


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transform to points.
    
    Args:
        points: (N, 3) points
        T: (4, 4) transform matrix
        
    Returns:
        (N, 3) transformed points
    """
    R, t = decompose_transform_matrix(T)
    return points @ R.T + t


# ============================================================================
# Torch versions (if available)
# ============================================================================

if HAS_TORCH:
    def quat_multiply_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply quaternions (wxyz format) using PyTorch."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)
    
    def quat_to_rotation_matrix_torch(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix using PyTorch."""
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        return torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
        ], dim=-2)
    
    def compute_vertex_normals_torch(vertices: torch.Tensor, 
                                      faces: torch.Tensor) -> torch.Tensor:
        """Compute per-vertex normals using PyTorch."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.index_add_(0, faces[:, i], face_normals)
        
        norms = vertex_normals.norm(dim=1, keepdim=True)
        return vertex_normals / (norms + 1e-8)
