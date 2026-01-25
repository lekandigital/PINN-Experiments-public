"""
Physics-informed loss functions for ClothGNN.
Includes position, edge-length, shear, and seam losses.

Based on MeshGraphNetRP auxiliary losses for cloth simulation.
"""
import torch
import torch.nn.functional as F


def position_loss(pred, target):
    """
    L2 loss on predicted vs. ground-truth displacements.
    
    L_pos = (1/N) * sum_i ||pred_i - target_i||^2
    
    Args:
        pred: (N, 3) predicted displacements
        target: (N, 3) ground-truth displacements
        
    Returns:
        loss: Scalar MSE loss
    """
    return F.mse_loss(pred, target)


def edge_length_loss(pred_pos, edge_index, rest_lengths):
    """
    Penalize deviation from rest edge lengths.
    Helps preserve cloth structure during deformation.
    
    L_edge = (1/|E|) * sum_{(i,j) in E} (||p_i - p_j|| - L_ij^0)^2
    
    Args:
        pred_pos: (N, 3) predicted positions (original + displacement)
        edge_index: (2, E) edge connectivity
        rest_lengths: (E,) rest lengths for each edge
        
    Returns:
        loss: Scalar edge length preservation loss
    """
    row, col = edge_index
    
    # Current edge vectors
    edge_vectors = pred_pos[col] - pred_pos[row]  # (E, 3)
    
    # Current edge lengths
    current_lengths = edge_vectors.norm(dim=1)  # (E,)
    
    # Deviation from rest length
    length_diff = current_lengths - rest_lengths
    
    return (length_diff ** 2).mean()


def shear_loss(pred_pos, triangles, rest_angles=None):
    """
    Penalize changes in triangle angles (shear deformation).
    Prevents super-elastic "rubber" behavior.
    
    L_shear = (1/|T|) * sum_t ||delta_theta_t - delta_theta_t^gt||^2
    
    Args:
        pred_pos: (N, 3) predicted positions
        triangles: (T, 3) triangle vertex indices
        rest_angles: (T, 3) rest angles for each triangle (optional)
        
    Returns:
        loss: Scalar shear loss
    """
    # Get triangle vertices
    v0 = pred_pos[triangles[:, 0]]  # (T, 3)
    v1 = pred_pos[triangles[:, 1]]  # (T, 3)
    v2 = pred_pos[triangles[:, 2]]  # (T, 3)
    
    # Compute edge vectors
    e01 = v1 - v0  # (T, 3)
    e02 = v2 - v0  # (T, 3)
    e12 = v2 - v1  # (T, 3)
    
    # Compute angles using dot products
    def compute_angle(a, b):
        """Compute angle between vectors a and b."""
        cos_angle = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
        return torch.acos(cos_angle.clamp(-1 + 1e-7, 1 - 1e-7))
    
    # Three angles per triangle
    angle0 = compute_angle(e01, e02)  # Angle at v0
    angle1 = compute_angle(-e01, e12)  # Angle at v1
    angle2 = compute_angle(-e02, -e12)  # Angle at v2
    
    current_angles = torch.stack([angle0, angle1, angle2], dim=1)  # (T, 3)
    
    if rest_angles is not None:
        # Penalize deviation from rest angles
        angle_diff = current_angles - rest_angles
        return (angle_diff ** 2).mean()
    else:
        # Without rest angles, penalize deviation from 60 degrees (equilateral)
        target_angle = torch.tensor(3.14159 / 3.0, device=pred_pos.device)
        angle_diff = current_angles - target_angle
        return (angle_diff ** 2).mean()


def seam_curvature_loss(pred_disp, seam_edges):
    """
    Encourage smooth seam deformation.
    Nearby seam vertices should move consistently.
    
    L_seam = (1/|S|) * sum_{(i,j) in S} ||u_i - u_j||^2
    
    Args:
        pred_disp: (N, 3) predicted displacements
        seam_edges: (S, 2) seam edge vertex pairs
        
    Returns:
        loss: Scalar seam smoothness loss
    """
    if seam_edges is None or len(seam_edges) == 0:
        return torch.tensor(0.0, device=pred_disp.device)
    
    # Get seam vertex displacements
    disp_i = pred_disp[seam_edges[:, 0]]  # (S, 3)
    disp_j = pred_disp[seam_edges[:, 1]]  # (S, 3)
    
    # Penalize displacement differences along seams
    disp_diff = disp_i - disp_j
    
    return (disp_diff ** 2).sum(dim=1).mean()


def compute_rest_lengths(positions, edge_index):
    """
    Compute rest edge lengths from initial positions.
    
    Args:
        positions: (N, 3) rest positions
        edge_index: (2, E) edge connectivity
        
    Returns:
        rest_lengths: (E,) rest length for each edge
    """
    row, col = edge_index
    edge_vectors = positions[col] - positions[row]
    return edge_vectors.norm(dim=1)


def compute_rest_angles(positions, triangles):
    """
    Compute rest triangle angles from initial positions.
    
    Args:
        positions: (N, 3) rest positions
        triangles: (T, 3) triangle indices
        
    Returns:
        rest_angles: (T, 3) rest angles for each triangle
    """
    v0 = positions[triangles[:, 0]]
    v1 = positions[triangles[:, 1]]
    v2 = positions[triangles[:, 2]]
    
    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1
    
    def compute_angle(a, b):
        cos_angle = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
        return torch.acos(cos_angle.clamp(-1 + 1e-7, 1 - 1e-7))
    
    angle0 = compute_angle(e01, e02)
    angle1 = compute_angle(-e01, e12)
    angle2 = compute_angle(-e02, -e12)
    
    return torch.stack([angle0, angle1, angle2], dim=1)


def total_loss(pred_disp, target_disp, pred_pos, edge_index, rest_lengths,
               triangles=None, rest_angles=None, seam_edges=None,
               lambda_p=1.0, lambda_e=0.1, lambda_s=0.05, lambda_c=0.02):
    """
    Weighted combination of all loss terms.
    
    L = lambda_p * L_pos + lambda_e * L_edge + lambda_s * L_shear + lambda_c * L_seam
    
    Args:
        pred_disp: (N, 3) predicted displacements
        target_disp: (N, 3) ground-truth displacements
        pred_pos: (N, 3) predicted positions
        edge_index: (2, E) edge connectivity
        rest_lengths: (E,) rest edge lengths
        triangles: (T, 3) triangle indices (optional)
        rest_angles: (T, 3) rest triangle angles (optional)
        seam_edges: (S, 2) seam edge pairs (optional)
        lambda_p: Position loss weight (default: 1.0)
        lambda_e: Edge length loss weight (default: 0.1)
        lambda_s: Shear loss weight (default: 0.05)
        lambda_c: Seam curvature loss weight (default: 0.02)
        
    Returns:
        total: Weighted total loss
        loss_dict: Dictionary of individual loss components
    """
    # Position loss (always computed)
    l_pos = position_loss(pred_disp, target_disp)
    
    # Edge length loss
    l_edge = edge_length_loss(pred_pos, edge_index, rest_lengths)
    
    # Shear loss (optional)
    if triangles is not None:
        l_shear = shear_loss(pred_pos, triangles, rest_angles)
    else:
        l_shear = torch.tensor(0.0, device=pred_disp.device)
    
    # Seam curvature loss (optional)
    l_seam = seam_curvature_loss(pred_disp, seam_edges)
    
    # Weighted combination
    total = (lambda_p * l_pos + 
             lambda_e * l_edge + 
             lambda_s * l_shear + 
             lambda_c * l_seam)
    
    loss_dict = {
        "position": l_pos.item(),
        "edge_length": l_edge.item(),
        "shear": l_shear.item() if triangles is not None else 0.0,
        "seam": l_seam.item(),
        "total": total.item()
    }
    
    return total, loss_dict
