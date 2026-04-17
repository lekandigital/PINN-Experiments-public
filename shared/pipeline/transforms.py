"""
Coordinate and skinning transforms for the animation pipeline.

Includes:
- Coordinate system conversions (Y-up/Z-up, handedness)
- Forward kinematics for skeleton
- Linear Blend Skinning (LBS)
- Dual Quaternion Skinning (DQS)
"""

from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CoordinateSystem(Enum):
    """Common 3D coordinate system conventions."""
    Y_UP_RIGHT_HANDED = auto()   # OpenGL, Blender default
    Y_UP_LEFT_HANDED = auto()    # DirectX, Unity
    Z_UP_RIGHT_HANDED = auto()   # Blender (with Z-up), scientific
    Z_UP_LEFT_HANDED = auto()    # Unreal Engine


def convert_coordinates(points: np.ndarray,
                        source: CoordinateSystem,
                        target: CoordinateSystem) -> np.ndarray:
    """
    Convert points between coordinate systems.
    
    Args:
        points: (N, 3) or (3,) points
        source: Source coordinate system
        target: Target coordinate system
        
    Returns:
        Converted points with same shape
    """
    if source == target:
        return points.copy()
    
    # Normalize to Y_UP_RIGHT_HANDED first
    if source == CoordinateSystem.Z_UP_RIGHT_HANDED:
        # Z-up to Y-up: rotate -90 around X
        # (x, y, z) -> (x, z, -y)
        if points.ndim == 1:
            points = np.array([points[0], points[2], -points[1]])
        else:
            points = np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)
    elif source == CoordinateSystem.Y_UP_LEFT_HANDED:
        # Flip Z for handedness
        if points.ndim == 1:
            points = np.array([points[0], points[1], -points[2]])
        else:
            points = points.copy()
            points[:, 2] = -points[:, 2]
    elif source == CoordinateSystem.Z_UP_LEFT_HANDED:
        # Z-up to Y-up, then flip handedness
        if points.ndim == 1:
            points = np.array([points[0], points[2], points[1]])
        else:
            points = np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)
    
    # Now convert from Y_UP_RIGHT_HANDED to target
    if target == CoordinateSystem.Z_UP_RIGHT_HANDED:
        # Y-up to Z-up: rotate 90 around X
        if points.ndim == 1:
            points = np.array([points[0], -points[2], points[1]])
        else:
            points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    elif target == CoordinateSystem.Y_UP_LEFT_HANDED:
        if points.ndim == 1:
            points = np.array([points[0], points[1], -points[2]])
        else:
            points = points.copy()
            points[:, 2] = -points[:, 2]
    elif target == CoordinateSystem.Z_UP_LEFT_HANDED:
        if points.ndim == 1:
            points = np.array([points[0], points[2], points[1]])
        else:
            points = np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)
    
    return points


def forward_kinematics(local_rotations: np.ndarray,
                       local_positions: np.ndarray,
                       parent_indices: List[int],
                       root_position: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute global joint positions from local rotations.
    
    Args:
        local_rotations: (J, 3, 3) local rotation matrices
        local_positions: (J, 3) local position offsets (bone lengths)
        parent_indices: List of parent joint indices (-1 for root)
        root_position: (3,) root translation (optional)
        
    Returns:
        global_positions: (J, 3) global joint positions
    """
    num_joints = len(parent_indices)
    global_rotations = np.zeros((num_joints, 3, 3))
    global_positions = np.zeros((num_joints, 3))
    
    for j in range(num_joints):
        parent = parent_indices[j]
        
        if parent == -1:
            # Root joint
            global_rotations[j] = local_rotations[j]
            global_positions[j] = local_positions[j]
            if root_position is not None:
                global_positions[j] += root_position
        else:
            # Child joint
            global_rotations[j] = global_rotations[parent] @ local_rotations[j]
            global_positions[j] = (
                global_positions[parent] + 
                global_rotations[parent] @ local_positions[j]
            )
    
    return global_positions


def joint_positions_to_local_rotations(positions: np.ndarray,
                                        rest_positions: np.ndarray,
                                        parent_indices: List[int]) -> np.ndarray:
    """
    Compute local rotations from global joint positions (simple IK).
    
    Args:
        positions: (J, 3) current global positions
        rest_positions: (J, 3) rest pose positions
        parent_indices: Parent indices for each joint
        
    Returns:
        local_rotations: (J, 4) local quaternions (wxyz)
    """
    num_joints = positions.shape[0]
    rotations = np.zeros((num_joints, 4))
    rotations[:, 0] = 1.0  # Identity quaternions
    
    for j in range(1, num_joints):
        parent = parent_indices[j]
        if parent < 0:
            continue
        
        # Current bone direction
        current_dir = positions[j] - positions[parent]
        current_len = np.linalg.norm(current_dir)
        if current_len < 1e-6:
            continue
        current_dir = current_dir / current_len
        
        # Rest bone direction
        rest_dir = rest_positions[j] - rest_positions[parent]
        rest_len = np.linalg.norm(rest_dir)
        if rest_len < 1e-6:
            continue
        rest_dir = rest_dir / rest_len
        
        # Rotation from rest to current
        rotations[j] = _rotation_between_vectors(rest_dir, current_dir)
    
    return rotations


def _rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute quaternion that rotates v1 to v2."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    
    dot = np.dot(v1, v2)
    
    if dot > 0.9999:
        return np.array([1, 0, 0, 0])
    
    if dot < -0.9999:
        axis = np.cross(np.array([1, 0, 0]), v1)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0, 1, 0]), v1)
        axis = axis / np.linalg.norm(axis)
        return np.array([0, axis[0], axis[1], axis[2]])
    
    axis = np.cross(v1, v2)
    s = np.sqrt((1 + dot) * 2)
    
    q = np.array([s / 2, axis[0] / s, axis[1] / s, axis[2] / s])
    return q / (np.linalg.norm(q) + 1e-8)


def linear_blend_skinning(vertices: np.ndarray,
                          weights: np.ndarray,
                          joint_transforms: np.ndarray,
                          rest_transforms: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply Linear Blend Skinning to vertices.
    
    Args:
        vertices: (V, 3) rest pose vertices
        weights: (V, J) skinning weights (sum to 1 per vertex)
        joint_transforms: (J, 4, 4) current joint transforms
        rest_transforms: (J, 4, 4) rest pose transforms (inverse bind)
        
    Returns:
        (V, 3) deformed vertices
    """
    num_vertices = vertices.shape[0]
    num_joints = joint_transforms.shape[0]
    
    # Compute relative transforms if rest transforms provided
    if rest_transforms is not None:
        # T_relative = T_current @ T_rest^-1
        transforms = np.zeros_like(joint_transforms)
        for j in range(num_joints):
            transforms[j] = joint_transforms[j] @ np.linalg.inv(rest_transforms[j])
    else:
        transforms = joint_transforms
    
    # Homogeneous coordinates
    vertices_h = np.concatenate([vertices, np.ones((num_vertices, 1))], axis=1)
    
    # Blend transforms
    deformed = np.zeros((num_vertices, 3))
    
    for v in range(num_vertices):
        blended_transform = np.zeros((4, 4))
        for j in range(num_joints):
            if weights[v, j] > 0:
                blended_transform += weights[v, j] * transforms[j]
        
        deformed[v] = (blended_transform @ vertices_h[v])[:3]
    
    return deformed


def dual_quaternion_skinning(vertices: np.ndarray,
                             weights: np.ndarray,
                             joint_rotations: np.ndarray,
                             joint_translations: np.ndarray) -> np.ndarray:
    """
    Apply Dual Quaternion Skinning to vertices.
    
    DQS avoids the "candy wrapper" artifacts of LBS for twisting motions.
    
    Args:
        vertices: (V, 3) rest pose vertices
        weights: (V, J) skinning weights
        joint_rotations: (J, 4) joint quaternions (wxyz)
        joint_translations: (J, 3) joint translations
        
    Returns:
        (V, 3) deformed vertices
    """
    num_vertices = vertices.shape[0]
    num_joints = joint_rotations.shape[0]
    
    # Convert to dual quaternions
    # Real part: rotation quaternion
    # Dual part: 0.5 * t * r where t is pure translation quaternion
    dual_quats = np.zeros((num_joints, 8))
    
    for j in range(num_joints):
        q = joint_rotations[j]  # wxyz
        t = joint_translations[j]
        
        # Real part
        dual_quats[j, :4] = q
        
        # Dual part: 0.5 * (0, t) * q
        dual_quats[j, 4] = 0.5 * (-t[0]*q[1] - t[1]*q[2] - t[2]*q[3])
        dual_quats[j, 5] = 0.5 * (t[0]*q[0] + t[1]*q[3] - t[2]*q[2])
        dual_quats[j, 6] = 0.5 * (-t[0]*q[3] + t[1]*q[0] + t[2]*q[1])
        dual_quats[j, 7] = 0.5 * (t[0]*q[2] - t[1]*q[1] + t[2]*q[0])
    
    # Blend dual quaternions
    deformed = np.zeros((num_vertices, 3))
    
    for v in range(num_vertices):
        # Blend dual quaternion
        blended_dq = np.zeros(8)
        pivot_sign = 1.0
        
        for j in range(num_joints):
            w = weights[v, j]
            if w > 0:
                # Ensure consistent hemisphere
                if np.dot(dual_quats[j, :4], blended_dq[:4]) < 0:
                    w = -w
                blended_dq += w * dual_quats[j]
        
        # Normalize
        real_norm = np.linalg.norm(blended_dq[:4])
        if real_norm > 1e-8:
            blended_dq /= real_norm
        
        # Extract rotation and translation
        q = blended_dq[:4]
        d = blended_dq[4:]
        
        # Translation: 2 * d * q*
        t = 2 * np.array([
            -d[0]*q[1] + d[1]*q[0] - d[2]*q[3] + d[3]*q[2],
            -d[0]*q[2] + d[1]*q[3] + d[2]*q[0] - d[3]*q[1],
            -d[0]*q[3] - d[1]*q[2] + d[2]*q[1] + d[3]*q[0]
        ])
        
        # Rotate vertex then translate
        # v' = q * v * q* + t
        p = vertices[v]
        
        # Quaternion rotation
        w, x, y, z = q
        deformed[v] = np.array([
            (1 - 2*y*y - 2*z*z) * p[0] + (2*x*y - 2*w*z) * p[1] + (2*x*z + 2*w*y) * p[2],
            (2*x*y + 2*w*z) * p[0] + (1 - 2*x*x - 2*z*z) * p[1] + (2*y*z - 2*w*x) * p[2],
            (2*x*z - 2*w*y) * p[0] + (2*y*z + 2*w*x) * p[1] + (1 - 2*x*x - 2*y*y) * p[2]
        ]) + t
    
    return deformed


def generate_skinning_weights(vertices: np.ndarray,
                              joint_positions: np.ndarray,
                              max_influences: int = 4,
                              falloff: float = 5.0) -> np.ndarray:
    """
    Generate skinning weights based on distance to joints.
    
    Args:
        vertices: (V, 3) vertex positions
        joint_positions: (J, 3) joint positions
        max_influences: Maximum joints per vertex
        falloff: Distance falloff factor (higher = sharper)
        
    Returns:
        (V, J) skinning weights
    """
    num_vertices = vertices.shape[0]
    num_joints = joint_positions.shape[0]
    
    # Compute distances
    dists = np.zeros((num_vertices, num_joints))
    for j in range(num_joints):
        dists[:, j] = np.linalg.norm(vertices - joint_positions[j], axis=1)
    
    # Convert to weights
    weights = 1.0 / (dists + 0.1)
    weights = np.exp(-dists * falloff)
    
    # Keep only top-k per vertex
    if max_influences < num_joints:
        for v in range(num_vertices):
            indices = np.argsort(weights[v])[::-1]
            weights[v, indices[max_influences:]] = 0
    
    # Normalize
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    
    return weights


def sparsify_skinning_weights(weights: np.ndarray,
                              max_influences: int = 4,
                              threshold: float = 0.001) -> np.ndarray:
    """
    Sparsify skinning weights by keeping only top influences.
    
    Args:
        weights: (V, J) skinning weights
        max_influences: Maximum influences per vertex
        threshold: Minimum weight to keep
        
    Returns:
        (V, J) sparse weights
    """
    sparse = weights.copy()
    
    # Apply threshold
    sparse[sparse < threshold] = 0
    
    # Keep only top-k
    for v in range(weights.shape[0]):
        indices = np.argsort(sparse[v])[::-1]
        sparse[v, indices[max_influences:]] = 0
    
    # Renormalize
    sparse = sparse / (sparse.sum(axis=1, keepdims=True) + 1e-8)
    
    return sparse


# ============================================================================
# PyTorch versions
# ============================================================================

if HAS_TORCH:
    def linear_blend_skinning_torch(vertices: torch.Tensor,
                                     weights: torch.Tensor,
                                     joint_transforms: torch.Tensor) -> torch.Tensor:
        """
        Apply LBS using PyTorch (GPU-accelerated, differentiable).
        
        Args:
            vertices: (V, 3) rest pose vertices
            weights: (V, J) skinning weights
            joint_transforms: (J, 4, 4) joint transforms
            
        Returns:
            (V, 3) deformed vertices
        """
        num_vertices = vertices.shape[0]
        num_joints = joint_transforms.shape[0]
        
        # Homogeneous coordinates
        ones = torch.ones(num_vertices, 1, device=vertices.device, dtype=vertices.dtype)
        vertices_h = torch.cat([vertices, ones], dim=1)  # (V, 4)
        
        # Apply transforms: (V, J) @ (J, 4, 4) -> blended (V, 4, 4) 
        # Then (V, 4, 4) @ (V, 4, 1) -> (V, 4, 1)
        
        # Expand for broadcasting
        weights_exp = weights.unsqueeze(-1).unsqueeze(-1)  # (V, J, 1, 1)
        transforms_exp = joint_transforms.unsqueeze(0)  # (1, J, 4, 4)
        
        # Weighted sum of transforms
        blended = (weights_exp * transforms_exp).sum(dim=1)  # (V, 4, 4)
        
        # Apply to vertices
        deformed_h = torch.bmm(blended, vertices_h.unsqueeze(-1))  # (V, 4, 1)
        
        return deformed_h[:, :3, 0]
    
    def convert_coordinates_torch(points: torch.Tensor,
                                   source: CoordinateSystem,
                                   target: CoordinateSystem) -> torch.Tensor:
        """Convert coordinates using PyTorch."""
        if source == target:
            return points.clone()
        
        # Convert to numpy, transform, convert back
        np_points = points.cpu().numpy()
        converted = convert_coordinates(np_points, source, target)
        return torch.tensor(converted, device=points.device, dtype=points.dtype)
