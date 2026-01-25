"""
Physics-aware loss functions for cloth simulation.

Implements differentiable physics losses that encourage physically plausible
cloth behavior:
    - L_stretch: Edge length preservation (spring constraints)
    - L_bend: Dihedral angle consistency (bending energy)
    - L_momentum: Newton's 2nd law enforcement
    - L_collision: Penetration prevention via SDF

These losses are combined with data reconstruction loss during training
to learn physically consistent neural cloth representations.

References:
    - Baraff & Witkin, "Large Steps in Cloth Simulation", SIGGRAPH 1998
    - Grinspun et al., "Discrete Shells", SCA 2003
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def compute_edge_lengths(
    vertices: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """
    Compute edge lengths for all edges in mesh.
    
    Args:
        vertices: Vertex positions (batch, num_vertices, 3)
        edges: Edge indices (num_edges, 2) - pairs of vertex indices
        
    Returns:
        Edge lengths (batch, num_edges)
    """
    # Get vertices at each end of edges
    v0 = vertices[:, edges[:, 0], :]  # (batch, num_edges, 3)
    v1 = vertices[:, edges[:, 1], :]  # (batch, num_edges, 3)
    
    # Compute edge vectors and lengths
    edge_vectors = v1 - v0  # (batch, num_edges, 3)
    lengths = torch.norm(edge_vectors, dim=-1)  # (batch, num_edges)
    
    return lengths


def compute_stretch_loss(
    pred_vertices: torch.Tensor,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute stretch loss penalizing edge length deviation from rest state.
    
    L_stretch = Σ (||v_i - v_j|| - ℓ_ij^rest)²
    
    This enforces inextensibility constraints, preventing unrealistic
    stretching or compression of the cloth.
    
    Args:
        pred_vertices: Predicted vertex positions (batch, num_vertices, 3)
        edges: Edge indices (num_edges, 2)
        rest_lengths: Rest edge lengths (num_edges,) or (batch, num_edges)
        normalize: Whether to normalize by number of edges
        
    Returns:
        Scalar stretch loss
    """
    # Current edge lengths
    current_lengths = compute_edge_lengths(pred_vertices, edges)
    
    # Ensure rest_lengths has batch dimension
    if rest_lengths.dim() == 1:
        rest_lengths = rest_lengths.unsqueeze(0).expand(current_lengths.size(0), -1)
    
    # Strain = (current - rest) / rest (relative stretch)
    strain = (current_lengths - rest_lengths) / (rest_lengths + 1e-8)
    
    # Squared strain loss
    loss = (strain ** 2).sum(dim=-1).mean()
    
    if normalize:
        loss = loss / edges.size(0)
    
    return loss


def compute_dihedral_angles(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    edge_to_faces: torch.Tensor,
) -> torch.Tensor:
    """
    Compute dihedral angles between adjacent faces.
    
    The dihedral angle is the angle between face normals at shared edges,
    measuring the local bending of the cloth surface.
    
    Args:
        vertices: Vertex positions (batch, num_vertices, 3)
        faces: Face indices (num_faces, 3)
        edge_to_faces: For each interior edge, the two adjacent face indices
                      (num_interior_edges, 2)
                      
    Returns:
        Dihedral angles in radians (batch, num_interior_edges)
    """
    batch_size = vertices.size(0)
    num_edges = edge_to_faces.size(0)
    
    # Compute face normals for all faces
    v0 = vertices[:, faces[:, 0], :]  # (batch, num_faces, 3)
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    
    e01 = v1 - v0
    e02 = v2 - v0
    
    face_normals = F.normalize(torch.cross(e01, e02, dim=-1), dim=-1)
    
    # Get normals of adjacent faces for each edge
    n1 = face_normals[:, edge_to_faces[:, 0], :]  # (batch, num_edges, 3)
    n2 = face_normals[:, edge_to_faces[:, 1], :]
    
    # Dihedral angle from dot product
    cos_angle = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(cos_angle)
    
    return angles


def compute_bend_loss(
    pred_vertices: torch.Tensor,
    faces: torch.Tensor,
    edge_to_faces: torch.Tensor,
    rest_angles: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute bending loss penalizing dihedral angle deviation from rest state.
    
    L_bend = Σ (θ_dihedral - θ_rest)²
    
    Bending energy prevents unrealistic folding and maintains cloth stiffness.
    Lower λ_bend allows more flexible cloth; higher values create stiffer material.
    
    Args:
        pred_vertices: Predicted vertex positions (batch, num_vertices, 3)
        faces: Face indices (num_faces, 3)
        edge_to_faces: Adjacent face pairs for interior edges (num_interior_edges, 2)
        rest_angles: Rest dihedral angles (num_interior_edges,) or (batch, num_interior_edges)
        normalize: Whether to normalize by number of edges
        
    Returns:
        Scalar bending loss
    """
    # Current dihedral angles
    current_angles = compute_dihedral_angles(pred_vertices, faces, edge_to_faces)
    
    # Ensure rest_angles has batch dimension
    if rest_angles.dim() == 1:
        rest_angles = rest_angles.unsqueeze(0).expand(current_angles.size(0), -1)
    
    # Angular deviation
    angle_diff = current_angles - rest_angles
    
    # Squared difference
    loss = (angle_diff ** 2).sum(dim=-1).mean()
    
    if normalize:
        loss = loss / edge_to_faces.size(0)
    
    return loss


def compute_momentum_loss(
    v_curr: torch.Tensor,
    v_prev: torch.Tensor,
    a_pred: torch.Tensor,
    dt: float,
    mass: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute momentum conservation loss enforcing Newton's 2nd law.
    
    L_momentum = Σ ||v_curr - (v_prev + a * dt)||²
    
    Ensures temporal consistency by penalizing violations of the
    kinematic equation: velocity(t+dt) = velocity(t) + acceleration * dt
    
    Args:
        v_curr: Current frame vertex positions (batch, num_vertices, 3)
        v_prev: Previous frame vertex positions (batch, num_vertices, 3)
        a_pred: Predicted accelerations (batch, num_vertices, 3)
        dt: Time step size (seconds)
        mass: Optional per-vertex masses (num_vertices,) for weighted loss
        
    Returns:
        Scalar momentum loss
    """
    # Expected position change from acceleration
    expected_displacement = a_pred * (dt ** 2)  # d = 0.5 * a * t^2, simplified
    
    # Actual displacement
    actual_displacement = v_curr - v_prev
    
    # Residual
    residual = actual_displacement - expected_displacement
    
    # Squared residual
    if mass is not None:
        # Weight by mass (heavier vertices have stronger momentum)
        mass = mass.view(1, -1, 1)
        loss = (mass * residual ** 2).sum(dim=(-1, -2)).mean()
    else:
        loss = (residual ** 2).sum(dim=(-1, -2)).mean()
    
    return loss


def compute_velocity_consistency_loss(
    pos_curr: torch.Tensor,
    pos_prev: torch.Tensor,
    pos_next: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Alternative momentum loss using finite difference velocities.
    
    Penalizes abrupt velocity changes (acceleration spikes) for smooth motion.
    
    L = Σ ||v(t+dt) - v(t)||² where v(t) = (x(t+dt) - x(t)) / dt
    
    Args:
        pos_curr: Current positions (batch, num_vertices, 3)
        pos_prev: Previous positions
        pos_next: Next positions
        dt: Time step
        
    Returns:
        Velocity consistency loss
    """
    # Finite difference velocities
    vel_backward = (pos_curr - pos_prev) / dt
    vel_forward = (pos_next - pos_curr) / dt
    
    # Velocity change (acceleration * dt)
    vel_diff = vel_forward - vel_backward
    
    return (vel_diff ** 2).sum(dim=(-1, -2)).mean()


def compute_collision_loss(
    pred_vertices: torch.Tensor,
    obstacle_sdf_fn: Optional[callable] = None,
    ground_height: float = 0.0,
    margin: float = 0.01,
) -> torch.Tensor:
    """
    Compute collision penalty preventing penetration into obstacles.
    
    L_collision = Σ max(0, margin - sdf(v_i))²
    
    Uses SDF (signed distance field) to detect and penalize penetration.
    The margin adds a safety buffer to prevent near-collisions.
    
    Args:
        pred_vertices: Predicted vertex positions (batch, num_vertices, 3)
        obstacle_sdf_fn: Function that computes SDF for arbitrary points
                        sdf_fn(points: Tensor(N,3)) -> Tensor(N,)
                        Positive = outside, Negative = inside obstacle
        ground_height: Height of ground plane (simple floor collision)
        margin: Safety margin for collision detection
        
    Returns:
        Scalar collision loss
    """
    loss = torch.tensor(0.0, device=pred_vertices.device, dtype=pred_vertices.dtype)
    
    # Ground plane collision (y > ground_height)
    y_coords = pred_vertices[..., 1]  # (batch, num_vertices)
    ground_penetration = margin + ground_height - y_coords
    ground_loss = F.relu(ground_penetration) ** 2
    loss = loss + ground_loss.sum(dim=-1).mean()
    
    # Obstacle SDF collision (if provided)
    if obstacle_sdf_fn is not None:
        batch_size, num_vertices, _ = pred_vertices.shape
        
        # Flatten for SDF query
        flat_vertices = pred_vertices.reshape(-1, 3)
        
        # Query SDF
        sdf_values = obstacle_sdf_fn(flat_vertices)
        sdf_values = sdf_values.reshape(batch_size, num_vertices)
        
        # Penetration penalty (negative SDF = inside obstacle)
        penetration = margin - sdf_values
        obstacle_loss = F.relu(penetration) ** 2
        loss = loss + obstacle_loss.sum(dim=-1).mean()
    
    return loss


def compute_self_collision_loss(
    pred_vertices: torch.Tensor,
    faces: torch.Tensor,
    min_distance: float = 0.005,
    sample_ratio: float = 0.1,
) -> torch.Tensor:
    """
    Compute self-collision loss preventing cloth self-intersection.
    
    Uses sampling-based approximate collision detection for efficiency.
    Checks distance between non-adjacent vertex-face pairs.
    
    Args:
        pred_vertices: Vertex positions (batch, num_vertices, 3)
        faces: Face indices (num_faces, 3)
        min_distance: Minimum allowed distance between cloth layers
        sample_ratio: Fraction of pairs to sample (for efficiency)
        
    Returns:
        Self-collision loss
    """
    batch_size, num_vertices, _ = pred_vertices.shape
    num_faces = faces.size(0)
    
    # Sample vertex-face pairs to check
    num_samples = int(num_vertices * num_faces * sample_ratio)
    
    # Random vertex indices
    vertex_idx = torch.randint(0, num_vertices, (num_samples,), device=pred_vertices.device)
    
    # Random face indices
    face_idx = torch.randint(0, num_faces, (num_samples,), device=pred_vertices.device)
    
    # Get sample vertices and face vertices
    sample_vertices = pred_vertices[:, vertex_idx, :]  # (batch, num_samples, 3)
    
    face_v0 = pred_vertices[:, faces[face_idx, 0], :]  # (batch, num_samples, 3)
    face_v1 = pred_vertices[:, faces[face_idx, 1], :]
    face_v2 = pred_vertices[:, faces[face_idx, 2], :]
    
    # Compute face centroids
    face_centroids = (face_v0 + face_v1 + face_v2) / 3.0
    
    # Distance from vertex to face centroid (simplified check)
    distances = torch.norm(sample_vertices - face_centroids, dim=-1)
    
    # Penalize distances below threshold
    penetration = min_distance - distances
    loss = F.relu(penetration) ** 2
    
    return loss.sum(dim=-1).mean()


class PhysicsLossStack(nn.Module):
    """
    Combined physics-aware loss for cloth simulation.
    
    Aggregates multiple physics losses with configurable weights:
        L_total = L_data + λ_stretch * L_stretch + λ_bend * L_bend 
                  + λ_momentum * L_momentum + λ_collision * L_collision
    
    Args:
        lambda_stretch: Weight for stretch loss (default: 1.0)
        lambda_bend: Weight for bending loss (default: 0.1)
        lambda_momentum: Weight for momentum loss (default: 0.1)
        lambda_collision: Weight for collision loss (default: 10.0)
        lambda_self_collision: Weight for self-collision (default: 1.0)
        ground_height: Height of ground plane
        collision_margin: Safety margin for collision detection
    """
    
    def __init__(
        self,
        lambda_stretch: float = 1.0,
        lambda_bend: float = 0.1,
        lambda_momentum: float = 0.1,
        lambda_collision: float = 10.0,
        lambda_self_collision: float = 1.0,
        ground_height: float = 0.0,
        collision_margin: float = 0.01,
    ):
        super().__init__()
        
        # Loss weights
        self.lambda_stretch = lambda_stretch
        self.lambda_bend = lambda_bend
        self.lambda_momentum = lambda_momentum
        self.lambda_collision = lambda_collision
        self.lambda_self_collision = lambda_self_collision
        
        # Collision parameters
        self.ground_height = ground_height
        self.collision_margin = collision_margin

        # Mesh topology (set via set_mesh_topology)
        # Register as buffers with None initially so they can be updated later
        self.register_buffer('edges', None)
        self.register_buffer('faces', None)
        self.register_buffer('edge_to_faces', None)
        self.register_buffer('rest_lengths', None)
        self.register_buffer('rest_angles', None)
    
    def set_mesh_topology(
        self,
        edges: torch.Tensor,
        faces: torch.Tensor,
        edge_to_faces: torch.Tensor,
        rest_vertices: torch.Tensor,
    ) -> None:
        """
        Set mesh topology and compute rest state properties.
        
        Call this once with the rest mesh before training.
        
        Args:
            edges: Edge indices (num_edges, 2)
            faces: Face indices (num_faces, 3)
            edge_to_faces: Interior edge to face mapping (num_interior_edges, 2)
            rest_vertices: Rest position vertices (num_vertices, 3)
        """
        # Update existing buffers (already registered in __init__)
        self.edges = edges
        self.faces = faces
        self.edge_to_faces = edge_to_faces

        # Compute rest lengths
        rest_vertices_batch = rest_vertices.unsqueeze(0)
        rest_lengths = compute_edge_lengths(rest_vertices_batch, edges).squeeze(0)
        self.rest_lengths = rest_lengths

        # Compute rest dihedral angles
        rest_angles = compute_dihedral_angles(
            rest_vertices_batch, faces, edge_to_faces
        ).squeeze(0)
        self.rest_angles = rest_angles
    
    def forward(
        self,
        pred_vertices: torch.Tensor,
        target_vertices: Optional[torch.Tensor] = None,
        prev_vertices: Optional[torch.Tensor] = None,
        pred_acceleration: Optional[torch.Tensor] = None,
        dt: float = 1.0 / 30.0,
        obstacle_sdf_fn: Optional[callable] = None,
        compute_self_collision: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total physics loss.
        
        Args:
            pred_vertices: Predicted vertex positions (batch, num_vertices, 3)
            target_vertices: Ground truth positions for data loss
            prev_vertices: Previous frame positions for momentum loss
            pred_acceleration: Predicted accelerations
            dt: Time step for momentum computation
            obstacle_sdf_fn: Optional obstacle SDF function
            compute_self_collision: Whether to compute expensive self-collision
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=pred_vertices.dtype)
        
        # Data reconstruction loss
        if target_vertices is not None:
            data_loss = F.mse_loss(pred_vertices, target_vertices)
            loss_dict['data'] = data_loss
            total_loss = total_loss + data_loss
        
        # Stretch loss
        if self.lambda_stretch > 0 and self.edges is not None:
            stretch_loss = compute_stretch_loss(
                pred_vertices, self.edges, self.rest_lengths
            )
            loss_dict['stretch'] = stretch_loss
            total_loss = total_loss + self.lambda_stretch * stretch_loss
        
        # Bend loss
        if self.lambda_bend > 0 and self.edge_to_faces is not None:
            bend_loss = compute_bend_loss(
                pred_vertices, self.faces, self.edge_to_faces, self.rest_angles
            )
            loss_dict['bend'] = bend_loss
            total_loss = total_loss + self.lambda_bend * bend_loss
        
        # Momentum loss
        if self.lambda_momentum > 0 and prev_vertices is not None and pred_acceleration is not None:
            momentum_loss = compute_momentum_loss(
                pred_vertices, prev_vertices, pred_acceleration, dt
            )
            loss_dict['momentum'] = momentum_loss
            total_loss = total_loss + self.lambda_momentum * momentum_loss
        
        # Collision loss
        if self.lambda_collision > 0:
            collision_loss = compute_collision_loss(
                pred_vertices,
                obstacle_sdf_fn=obstacle_sdf_fn,
                ground_height=self.ground_height,
                margin=self.collision_margin,
            )
            loss_dict['collision'] = collision_loss
            total_loss = total_loss + self.lambda_collision * collision_loss
        
        # Self-collision loss (expensive, optional)
        if compute_self_collision and self.lambda_self_collision > 0 and self.faces is not None:
            self_collision_loss = compute_self_collision_loss(
                pred_vertices, self.faces
            )
            loss_dict['self_collision'] = self_collision_loss
            total_loss = total_loss + self.lambda_self_collision * self_collision_loss
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def create_grid_mesh_topology(
    grid_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create mesh topology for a regular grid cloth.
    
    Useful for testing with synthetic data.
    
    Args:
        grid_size: Number of vertices per side (grid_size x grid_size mesh)
        
    Returns:
        edges: Edge indices (num_edges, 2)
        faces: Face indices (num_faces, 3)
        edge_to_faces: Interior edge to face mapping
    """
    num_vertices = grid_size * grid_size
    
    # Generate edges (horizontal, vertical, and diagonal)
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            v = i * grid_size + j
            
            # Horizontal edge
            if j < grid_size - 1:
                edges.append([v, v + 1])
            
            # Vertical edge
            if i < grid_size - 1:
                edges.append([v, v + grid_size])
            
            # Diagonal (for triangulation)
            if i < grid_size - 1 and j < grid_size - 1:
                edges.append([v, v + grid_size + 1])
    
    edges = torch.tensor(edges, dtype=torch.long)
    
    # Generate faces (two triangles per grid cell)
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v00 = i * grid_size + j
            v01 = v00 + 1
            v10 = v00 + grid_size
            v11 = v10 + 1
            
            # Upper-left triangle
            faces.append([v00, v10, v01])
            # Lower-right triangle
            faces.append([v01, v10, v11])
    
    faces = torch.tensor(faces, dtype=torch.long)
    
    # Edge to faces mapping (simplified - just pairs of adjacent face indices)
    # For a proper implementation, build this from mesh connectivity
    num_faces = faces.size(0)
    edge_to_faces = []
    for i in range(0, num_faces - 1, 2):
        edge_to_faces.append([i, i + 1])
    
    edge_to_faces = torch.tensor(edge_to_faces, dtype=torch.long)
    
    return edges, faces, edge_to_faces
