"""
NIF-Cloth3D-Interactive: Physics-Informed Loss Functions

This module implements physics-based loss functions for training the neural cloth simulator.
The losses enforce physical constraints on the predicted deformations:

1. Stretch Loss: Preserves edge lengths (inextensibility constraint)
   L_stretch = Σ_(i,j)∈E (||u_i - u_j|| - d_ij)²
   where d_ij is the rest length between vertices i and j

2. Bend Loss: Encourages smooth curvature via Laplacian regularization
   L_bend = Σ_i ||Δu_i||²
   where Δ is the discrete Laplacian operator

3. Momentum Loss: Enforces Newton's second law (dynamic consistency)
   L_momentum = ||ρ∂²u/∂t² - f_net||²
   where f_net includes gravity, wind, and elastic forces

References:
- Bertiche et al., "Neural Cloth Simulation", SIGGRAPH 2022
- Li et al., "DiffCloth: Differentiable Cloth Simulation", SIGGRAPH 2022
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


class StretchLoss(nn.Module):
    """
    Stretch (inextensibility) loss for cloth simulation.
    
    Penalizes deviation from rest edge lengths to enforce inextensibility.
    Real cloth is nearly inextensible in the fiber directions.
    
    Mathematical formulation:
    L_stretch = (1/|E|) Σ_(i,j)∈E (||x_i - x_j|| - L_ij)² / L_ij²
    
    The normalization by L_ij² makes the loss scale-invariant.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        rest_positions: torch.Tensor,
        edges: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Compute stretch loss for predicted cloth positions.
        
        Args:
            pred_positions: Predicted vertex positions (N, 3)
            rest_positions: Rest state vertex positions (N, 3)
            edges: List of (i, j) vertex index pairs defining edges
        
        Returns:
            Scalar stretch loss
        """
        loss = torch.tensor(0.0, device=pred_positions.device)
        
        for i, j in edges:
            # Current edge vector and length
            edge_pred = pred_positions[i] - pred_positions[j]
            len_pred = torch.norm(edge_pred)
            
            # Rest edge length
            edge_rest = rest_positions[i] - rest_positions[j]
            len_rest = torch.norm(edge_rest)
            
            # Strain: relative length change
            if self.normalize and len_rest > 1e-8:
                strain = (len_pred - len_rest) / len_rest
            else:
                strain = len_pred - len_rest
            
            loss = loss + strain ** 2
        
        # Average over edges
        if len(edges) > 0:
            loss = loss / len(edges)
        
        return loss


class StretchLossVectorized(nn.Module):
    """
    Vectorized stretch loss for efficiency with large meshes.
    
    Uses pre-computed edge index tensors for batched computation.
    """
    
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        rest_positions: torch.Tensor,
        edge_indices: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Vectorized stretch loss computation.
        
        Args:
            pred_positions: Predicted positions (N, 3)
            rest_positions: Rest positions (N, 3)
            edge_indices: Edge index pairs (E, 2) as LongTensor
            rest_lengths: Pre-computed rest lengths (E,) - optional
        
        Returns:
            Scalar stretch loss
        """
        # Get edge endpoints
        i_idx = edge_indices[:, 0]
        j_idx = edge_indices[:, 1]
        
        # Compute current edge vectors and lengths
        edge_vectors = pred_positions[i_idx] - pred_positions[j_idx]
        pred_lengths = torch.norm(edge_vectors, dim=1)
        
        # Compute or use provided rest lengths
        if rest_lengths is None:
            rest_vectors = rest_positions[i_idx] - rest_positions[j_idx]
            rest_lengths = torch.norm(rest_vectors, dim=1)
        
        # Compute strain
        if self.normalize:
            strain = (pred_lengths - rest_lengths) / (rest_lengths + self.eps)
        else:
            strain = pred_lengths - rest_lengths
        
        # Mean squared strain
        loss = torch.mean(strain ** 2)
        
        return loss


class BendLoss(nn.Module):
    """
    Bending loss using discrete Laplacian regularization.
    
    Penalizes deviation from smooth curvature to prevent unrealistic sharp folds.
    Uses the cotangent Laplacian for quality on triangle meshes, or a simpler
    uniform Laplacian for regular grids.
    
    For a regular grid cloth:
    Δu_i ≈ (u_{i-1} - 2u_i + u_{i+1})  (1D case)
    
    L_bend = (1/N) Σ_i ||Δu_i||²
    """
    
    def __init__(self, laplacian_type: str = "uniform"):
        super().__init__()
        self.laplacian_type = laplacian_type
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        neighbors: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute bending loss using discrete Laplacian.
        
        Args:
            pred_positions: Predicted positions (N, 3)
            neighbors: List of neighbor indices for each vertex
        
        Returns:
            Scalar bending loss
        """
        N = pred_positions.shape[0]
        loss = torch.tensor(0.0, device=pred_positions.device)
        count = 0
        
        for i in range(N):
            if len(neighbors[i]) == 0:
                continue
            
            # Compute Laplacian: average of neighbors minus center
            neighbor_positions = pred_positions[neighbors[i]]
            mean_neighbor = neighbor_positions.mean(dim=0)
            laplacian = mean_neighbor - pred_positions[i]
            
            loss = loss + torch.sum(laplacian ** 2)
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class BendLossVectorized(nn.Module):
    """
    Vectorized bending loss for efficiency.
    
    Uses sparse Laplacian matrix for efficient computation.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self._laplacian = None
    
    def build_laplacian(
        self,
        n_vertices: int,
        edges: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Build sparse Laplacian matrix from edge connectivity."""
        # Create adjacency matrix
        adj = torch.zeros(n_vertices, n_vertices, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj[edges[:, 1], edges[:, 0]] = 1
        
        # Degree matrix
        degree = adj.sum(dim=1)
        
        # Normalized Laplacian: L = I - D^(-1) A
        D_inv = torch.diag(1.0 / (degree + self.eps))
        laplacian = torch.eye(n_vertices, device=device) - D_inv @ adj
        
        return laplacian
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        laplacian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bending loss using Laplacian matrix.
        
        Args:
            pred_positions: Predicted positions (N, 3)
            laplacian: Laplacian matrix (N, N)
        
        Returns:
            Scalar bending loss
        """
        # Apply Laplacian to each coordinate
        lap_positions = laplacian @ pred_positions  # (N, 3)
        
        # Mean squared Laplacian magnitude
        loss = torch.mean(torch.sum(lap_positions ** 2, dim=1))
        
        return loss


class MomentumLoss(nn.Module):
    """
    Momentum conservation loss based on Newton's second law.
    
    Enforces dynamic consistency: ma = f_net
    where:
    - m: vertex mass
    - a: acceleration (∂²u/∂t²)
    - f_net: net force (gravity + wind + elastic response)
    
    L_momentum = (1/N) Σ_i ||m_i * a_i - f_i||²
    
    For quasi-static equilibrium (low velocity), we can approximate:
    L_momentum ≈ ||f_elastic + f_external||²
    """
    
    def __init__(self, mass: float = 1.0, dt: float = 1.0 / 30.0):
        super().__init__()
        self.mass = mass
        self.dt = dt
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        prev_positions: torch.Tensor,
        prev_prev_positions: torch.Tensor,
        external_forces: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute momentum loss using finite difference acceleration.
        
        Args:
            pred_positions: Current predicted positions (N, 3)
            prev_positions: Previous timestep positions (N, 3)
            prev_prev_positions: Two timesteps ago positions (N, 3)
            external_forces: Applied forces per vertex (N, 3)
        
        Returns:
            Scalar momentum loss
        """
        # Finite difference acceleration: a = (u_t - 2*u_{t-1} + u_{t-2}) / dt²
        acceleration = (pred_positions - 2 * prev_positions + prev_prev_positions) / (self.dt ** 2)
        
        # Newton's law residual: m*a - f = 0
        residual = self.mass * acceleration - external_forces
        
        # Mean squared residual
        loss = torch.mean(torch.sum(residual ** 2, dim=1))
        
        return loss


class QuasiStaticMomentumLoss(nn.Module):
    """
    Simplified momentum loss for quasi-static equilibrium.
    
    When velocities are low, we only need forces to balance:
    L = ||f_gravity + f_wind + f_elastic||²
    """
    
    def __init__(self, gravity: torch.Tensor = None):
        super().__init__()
        if gravity is None:
            gravity = torch.tensor([0.0, 0.0, -9.81])
        self.register_buffer('gravity', gravity)
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        rest_positions: torch.Tensor,
        external_forces: torch.Tensor,
        stiffness: float = 1.0
    ) -> torch.Tensor:
        """
        Compute quasi-static equilibrium loss.
        
        Args:
            pred_positions: Predicted positions (N, 3)
            rest_positions: Rest positions (N, 3)
            external_forces: External forces (wind, etc.) per vertex (N, 3)
            stiffness: Elastic stiffness coefficient
        
        Returns:
            Scalar equilibrium loss
        """
        N = pred_positions.shape[0]

        # Gravity force (ensure on same device as input)
        f_gravity = self.gravity.to(pred_positions.device).expand(N, -1)
        
        # Simple elastic response (Hooke's law approximation)
        displacement = pred_positions - rest_positions
        f_elastic = -stiffness * displacement
        
        # Total force (should be zero at equilibrium)
        f_total = f_gravity + external_forces + f_elastic
        
        # Mean squared force magnitude
        loss = torch.mean(torch.sum(f_total ** 2, dim=1))
        
        return loss


class CollisionLoss(nn.Module):
    """
    Collision avoidance loss using signed distance field.
    
    Penalizes cloth vertices that penetrate a collision body (e.g., human body).
    Uses SDF values to compute penetration depth.
    
    L_collision = Σ_i max(0, -sdf(x_i))²
    
    Negative SDF means inside the body (penetration).
    """
    
    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        sdf_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute collision penalty from SDF values.
        
        Args:
            pred_positions: Predicted positions (N, 3) - for reference
            sdf_values: SDF values at predicted positions (N,)
                       Negative = inside collision body
        
        Returns:
            Scalar collision loss
        """
        # Penetration depth with margin
        penetration = torch.relu(-sdf_values - self.margin)
        
        # Mean squared penetration
        loss = torch.mean(penetration ** 2)
        
        return loss


class PhysicsLoss(nn.Module):
    """
    Combined physics-informed loss for cloth simulation.
    
    Aggregates stretch, bend, momentum, and collision losses with configurable weights.
    """
    
    def __init__(
        self,
        stretch_weight: float = 1.0,
        bend_weight: float = 0.5,
        momentum_weight: float = 0.3,
        collision_weight: float = 10.0,
        damping_weight: float = 0.01
    ):
        super().__init__()
        
        self.stretch_weight = stretch_weight
        self.bend_weight = bend_weight
        self.momentum_weight = momentum_weight
        self.collision_weight = collision_weight
        self.damping_weight = damping_weight
        
        self.stretch_loss = StretchLossVectorized()
        self.bend_loss = BendLossVectorized()
        self.quasi_static_loss = QuasiStaticMomentumLoss()
        self.collision_loss = CollisionLoss()
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        rest_positions: torch.Tensor,
        edge_indices: torch.Tensor,
        laplacian: torch.Tensor,
        external_forces: torch.Tensor,
        sdf_values: Optional[torch.Tensor] = None,
        rest_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined physics loss.
        
        Returns:
            Dictionary with individual loss components and total loss
        """
        losses = {}
        
        # Stretch loss
        L_stretch = self.stretch_loss(pred_positions, rest_positions, edge_indices, rest_lengths)
        losses['stretch'] = L_stretch
        
        # Bend loss
        L_bend = self.bend_loss(pred_positions, laplacian)
        losses['bend'] = L_bend
        
        # Quasi-static momentum
        L_momentum = self.quasi_static_loss(pred_positions, rest_positions, external_forces)
        losses['momentum'] = L_momentum
        
        # Damping (regularization)
        displacement = pred_positions - rest_positions
        L_damping = torch.mean(displacement ** 2)
        losses['damping'] = L_damping
        
        # Collision (if SDF provided)
        if sdf_values is not None:
            L_collision = self.collision_loss(pred_positions, sdf_values)
            losses['collision'] = L_collision
        else:
            L_collision = torch.tensor(0.0, device=pred_positions.device)
            losses['collision'] = L_collision
        
        # Weighted sum
        total = (
            self.stretch_weight * L_stretch +
            self.bend_weight * L_bend +
            self.momentum_weight * L_momentum +
            self.damping_weight * L_damping +
            self.collision_weight * L_collision
        )
        losses['total'] = total
        
        return losses


def compute_physics_loss(
    pred_disp: torch.Tensor,
    rest_pos: torch.Tensor,
    edges: List[Tuple[int, int]],
    stretch_weight: float = 1.0,
    bend_weight: float = 0.5,
    damping_weight: float = 0.01
) -> torch.Tensor:
    """
    Simple physics loss function (non-vectorized, for reference).
    
    Computes stretch and bend losses from predicted displacements.
    
    Args:
        pred_disp: Predicted displacements (N, 3)
        rest_pos: Rest positions (N, 3)
        edges: List of (i, j) edge tuples
        stretch_weight: Weight for stretch loss
        bend_weight: Weight for bend loss
        damping_weight: Weight for displacement regularization
    
    Returns:
        Total physics loss
    """
    N = rest_pos.shape[0]
    pred_pos = rest_pos + pred_disp
    
    # Stretch loss
    stretch_loss = torch.tensor(0.0, device=pred_disp.device)
    for (i, j) in edges:
        pred_len = torch.norm(pred_pos[i] - pred_pos[j])
        rest_len = torch.norm(rest_pos[i] - rest_pos[j])
        stretch_loss = stretch_loss + ((pred_len - rest_len) / rest_len) ** 2
    stretch_loss = stretch_loss / max(len(edges), 1)
    
    # Bend loss (Laplacian for interior vertices)
    bend_loss = torch.tensor(0.0, device=pred_disp.device)
    for i in range(1, N - 1):
        laplacian = pred_pos[i - 1] - 2 * pred_pos[i] + pred_pos[i + 1]
        bend_loss = bend_loss + torch.sum(laplacian ** 2)
    bend_loss = bend_loss / max(N - 2, 1)
    
    # Damping regularization
    damping = torch.mean(pred_disp ** 2)
    
    # Total loss
    total = stretch_weight * stretch_loss + bend_weight * bend_loss + damping_weight * damping
    
    return total


if __name__ == "__main__":
    # Test loss functions
    print("Testing physics losses...")
    
    N = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    rest_pos = torch.randn(N, 3, device=device)
    pred_pos = rest_pos + torch.randn(N, 3, device=device) * 0.1
    
    # Create edges (simple chain)
    edges = [(i, i + 1) for i in range(N - 1)]
    edge_indices = torch.tensor(edges, dtype=torch.long, device=device)
    
    # Test stretch loss
    stretch_fn = StretchLossVectorized()
    loss_stretch = stretch_fn(pred_pos, rest_pos, edge_indices)
    print(f"Stretch loss: {loss_stretch.item():.6f}")
    
    # Test bend loss
    bend_fn = BendLossVectorized()
    laplacian = bend_fn.build_laplacian(N, edge_indices, device)
    loss_bend = bend_fn(pred_pos, laplacian)
    print(f"Bend loss: {loss_bend.item():.6f}")
    
    # Test combined loss
    external_forces = torch.zeros(N, 3, device=device)
    physics_loss_fn = PhysicsLoss()
    losses = physics_loss_fn(
        pred_pos, rest_pos, edge_indices, laplacian, external_forces
    )
    print(f"Total physics loss: {losses['total'].item():.6f}")
    print("Components:", {k: v.item() for k, v in losses.items()})
