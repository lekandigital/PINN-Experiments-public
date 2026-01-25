"""
Physics-Informed Loss Functions for CoastFlow-GNN

Implements physics constraints based on:
- Continuity equation (incompressible flow): ∇·u = 0
- Navier-Stokes momentum equations
- k-ε turbulence model residuals
- Coastal boundary conditions (no-slip walls, free surface)

Uses graph-based differential operators computed via message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, degree
from typing import Optional, Dict, Tuple


class GraphDifferentialOperators:
    """
    Computes differential operators on graphs using finite differences
    and message passing.
    
    For a graph with nodes at positions x_i, the gradient is approximated
    using neighbors:
        ∂f/∂x ≈ Σ_j w_ij (f_j - f_i) (x_j - x_i) / |x_j - x_i|²
    """
    
    @staticmethod
    def compute_edge_vectors(
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge vectors and distances.
        
        Args:
            pos: Node positions [N, 3] (x, y, z)
            edge_index: Graph connectivity [2, E]
            
        Returns:
            edge_vec: Edge vectors [E, 3]
            edge_dist: Edge distances [E, 1]
        """
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]  # [E, 3]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]
        return edge_vec, edge_dist
    
    @staticmethod
    def compute_gradient(
        f: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate gradient of scalar field f using graph neighbors.
        
        Uses least-squares gradient estimation:
        ∇f_i ≈ (X^T X)^{-1} X^T Δf
        
        Simplified version using weighted averaging.
        
        Args:
            f: Scalar field values [N, 1] or [N]
            pos: Node positions [N, 3]
            edge_index: Graph connectivity [2, E]
            
        Returns:
            grad_f: Gradient [N, 3]
        """
        if f.dim() == 1:
            f = f.unsqueeze(-1)
            
        src, dst = edge_index
        N = pos.size(0)
        
        # Edge vectors and distances
        edge_vec = pos[dst] - pos[src]  # [E, 3]
        edge_dist_sq = (edge_vec ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]
        
        # Value differences
        df = f[dst] - f[src]  # [E, 1]
        
        # Weighted gradient contribution from each edge
        # grad_contribution = df * edge_vec / edge_dist_sq
        grad_contrib = df * edge_vec / edge_dist_sq  # [E, 3]
        
        # Aggregate at each node
        grad_f = scatter(grad_contrib, src, dim=0, dim_size=N, reduce='mean')
        
        return grad_f
    
    @staticmethod
    def compute_divergence(
        u: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate divergence of vector field u using graph neighbors.
        
        ∇·u = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
        
        Args:
            u: Vector field [N, 3]
            pos: Node positions [N, 3]
            edge_index: Graph connectivity [2, E]
            
        Returns:
            div_u: Divergence [N, 1]
        """
        src, dst = edge_index
        N = pos.size(0)
        
        # Edge vectors and distances
        edge_vec = pos[dst] - pos[src]  # [E, 3]
        edge_dist_sq = (edge_vec ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        edge_dist = edge_dist_sq.sqrt()
        
        # Normalized edge direction
        edge_dir = edge_vec / edge_dist  # [E, 3]
        
        # Velocity differences
        du = u[dst] - u[src]  # [E, 3]
        
        # Directional derivative along edge
        du_dn = (du * edge_dir).sum(dim=-1, keepdim=True)  # [E, 1]
        
        # Approximate divergence contribution
        div_contrib = du_dn / edge_dist  # [E, 1]
        
        # Aggregate
        div_u = scatter(div_contrib, src, dim=0, dim_size=N, reduce='mean')
        
        return div_u
    
    @staticmethod
    def compute_laplacian(
        f: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate Laplacian using graph Laplacian.
        
        ∇²f_i ≈ Σ_j w_ij (f_j - f_i)
        
        Args:
            f: Scalar or vector field [N, C]
            pos: Node positions [N, 3]
            edge_index: Graph connectivity [2, E]
            
        Returns:
            lap_f: Laplacian [N, C]
        """
        src, dst = edge_index
        N = f.size(0)
        
        # Distance-based weights
        edge_vec = pos[dst] - pos[src]
        edge_dist_sq = (edge_vec ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = 1.0 / edge_dist_sq  # [E, 1]
        
        # Weighted differences
        df = f[dst] - f[src]  # [E, C]
        weighted_df = weights * df  # [E, C]
        
        # Aggregate
        lap_f = scatter(weighted_df, src, dim=0, dim_size=N, reduce='sum')
        
        # Normalize by total weight
        weight_sum = scatter(weights, src, dim=0, dim_size=N, reduce='sum')
        lap_f = lap_f / weight_sum.clamp(min=1e-8)
        
        return lap_f


def compute_continuity_loss(
    u: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute continuity equation loss (incompressible flow).
    
    For incompressible flow: ∇·u = 0
    
    Args:
        u: Velocity field [N, 3] (u_x, u_y, u_z)
        pos: Node positions [N, 3]
        edge_index: Graph connectivity [2, E]
        reduction: Loss reduction ('mean', 'sum', 'none')
        
    Returns:
        loss: Continuity loss (scalar or [N])
    """
    div_u = GraphDifferentialOperators.compute_divergence(u, pos, edge_index)
    
    # L2 loss on divergence
    loss = div_u ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss.squeeze(-1)


def compute_momentum_loss(
    u: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    nu: float = 1e-6,
    rho: float = 1025.0,  # Seawater density kg/m³
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute Navier-Stokes momentum equation residual (steady state).
    
    ρ(u·∇)u = -∇p + μ∇²u + f
    
    For steady incompressible flow, simplified to:
    (u·∇)u - ν∇²u ≈ 0 (ignoring pressure gradient, external forces)
    
    Args:
        u: Velocity field [N, 3]
        pos: Node positions [N, 3]
        edge_index: Graph connectivity [2, E]
        nu: Kinematic viscosity (m²/s)
        rho: Fluid density (kg/m³)
        reduction: Loss reduction
        
    Returns:
        loss: Momentum residual loss
    """
    ops = GraphDifferentialOperators()
    
    # Viscous term: ν∇²u
    lap_u = ops.compute_laplacian(u, pos, edge_index)  # [N, 3]
    viscous_term = nu * lap_u
    
    # Convective term: (u·∇)u approximation
    # For each component: u_j ∂u_i/∂x_j
    src, dst = edge_index
    N = pos.size(0)
    
    edge_vec = pos[dst] - pos[src]
    edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True).clamp(min=1e-8)
    edge_dir = edge_vec / edge_dist
    
    # Velocity at source nodes
    u_src = u[src]  # [E, 3]
    
    # Velocity gradient along edge
    du = u[dst] - u[src]  # [E, 3]
    du_ds = du / edge_dist  # [E, 3]
    
    # Project onto edge direction and weight by velocity magnitude
    u_edge = (u_src * edge_dir).sum(dim=-1, keepdim=True)  # [E, 1]
    convective_contrib = u_edge * du_ds  # [E, 3]
    
    convective_term = scatter(convective_contrib, src, dim=0, dim_size=N, reduce='mean')
    
    # Residual: convective - viscous ≈ 0 (simplified)
    residual = convective_term - viscous_term
    loss = (residual ** 2).sum(dim=-1)  # [N]
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def compute_turbulence_loss(
    u: torch.Tensor,
    k: Optional[torch.Tensor],
    epsilon: Optional[torch.Tensor],
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    C_mu: float = 0.09,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute k-ε turbulence model residuals (simplified).
    
    k equation: Dk/Dt = P_k - ε + ∇·(ν_t/σ_k ∇k)
    ε equation: Dε/Dt = C_ε1 ε/k P_k - C_ε2 ε²/k + ∇·(ν_t/σ_ε ∇ε)
    
    Simplified loss: ensures k, ε are physically consistent.
    
    Args:
        u: Velocity field [N, 3]
        k: Turbulent kinetic energy [N] (optional, will estimate if None)
        epsilon: Dissipation rate [N] (optional, will estimate if None)
        pos: Node positions [N, 3]
        edge_index: Graph connectivity [2, E]
        C_mu: k-ε model constant
        reduction: Loss reduction
        
    Returns:
        loss: Turbulence model residual loss
    """
    ops = GraphDifferentialOperators()
    N = u.size(0)
    
    # If k and epsilon not provided, estimate from velocity gradients
    if k is None or epsilon is None:
        # Estimate turbulent kinetic energy from velocity fluctuations
        # Using velocity gradients as proxy
        src, dst = edge_index
        du = u[dst] - u[src]
        du_mag_sq = (du ** 2).sum(dim=-1)  # [E]
        k_estimate = scatter(du_mag_sq, src, dim=0, dim_size=N, reduce='mean')
        k = k_estimate.clamp(min=1e-10)
        
        # Estimate epsilon (simplified)
        epsilon = C_mu * k ** 1.5 / (pos.std() * 0.1).clamp(min=1e-6)
    
    # Physical constraints
    # 1. k >= 0, epsilon >= 0
    positivity_loss = F.relu(-k).mean() + F.relu(-epsilon).mean()
    
    # 2. Realizability: ν_t = C_μ k²/ε should be reasonable
    nu_t = C_mu * k ** 2 / epsilon.clamp(min=1e-10)
    # Penalize extreme values
    nu_t_normalized = nu_t / nu_t.mean().clamp(min=1e-10)
    realizability_loss = ((nu_t_normalized - 1) ** 2).mean()
    
    # 3. Production-dissipation balance (simplified)
    # In equilibrium, production ≈ dissipation
    # P_k ≈ ε → k should be stable
    lap_k = ops.compute_laplacian(k.unsqueeze(-1), pos, edge_index).squeeze(-1)
    diffusion_loss = (lap_k ** 2).mean()
    
    loss = positivity_loss + 0.1 * realizability_loss + 0.01 * diffusion_loss
    
    return loss


def compute_boundary_loss(
    u: torch.Tensor,
    wave_height: torch.Tensor,
    pos: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_type: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute boundary condition losses.
    
    Boundary types:
        0: Interior (no constraint)
        1: No-slip wall (u = 0)
        2: Free surface (pressure = 0, specific wave conditions)
        3: Inlet (prescribed velocity)
        4: Outlet (zero gradient)
    
    Args:
        u: Velocity field [N, 3]
        wave_height: Wave height predictions [N]
        pos: Node positions [N, 3]
        boundary_mask: Boolean mask for boundary nodes [N]
        boundary_type: Type of boundary condition [N]
        reduction: Loss reduction
        
    Returns:
        loss: Boundary condition violation loss
    """
    loss = torch.tensor(0.0, device=u.device, dtype=u.dtype)
    
    # No-slip walls (type 1): u = 0
    no_slip_mask = (boundary_type == 1) & boundary_mask
    if no_slip_mask.any():
        u_no_slip = u[no_slip_mask]
        loss = loss + (u_no_slip ** 2).sum(dim=-1).mean()
    
    # Free surface (type 2): vertical velocity constraint
    free_surface_mask = (boundary_type == 2) & boundary_mask
    if free_surface_mask.any():
        # At free surface, w ≈ ∂η/∂t (kinematic condition)
        # Simplified: penalize large vertical velocities
        w_surface = u[free_surface_mask, 2]  # z-component
        loss = loss + 0.1 * (w_surface ** 2).mean()
        
        # Wave height should be positive
        h_surface = wave_height[free_surface_mask]
        loss = loss + F.relu(-h_surface).mean()
    
    # Outlet (type 4): zero gradient (approximate)
    outlet_mask = (boundary_type == 4) & boundary_mask
    if outlet_mask.any():
        # Penalize sharp gradients at outlet
        # (Simplified: regularize velocity magnitude)
        u_out = u[outlet_mask]
        loss = loss + 0.01 * (u_out ** 2).sum(dim=-1).mean()
    
    return loss


def physics_informed_loss(
    data,
    predictions: torch.Tensor,
    lambda_data: float = 1.0,
    lambda_cont: float = 1.0,
    lambda_mom: float = 0.1,
    lambda_turb: float = 0.01,
    lambda_bc: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Combined physics-informed loss function.
    
    L = λ_data * L_data + λ_cont * L_cont + λ_mom * L_mom 
        + λ_turb * L_turb + λ_bc * L_bc
    
    Args:
        data: PyTorch Geometric Data object with:
            - x: Input features [N, 6]
            - y: Ground truth outputs [N, 4]
            - pos: Node positions [N, 3] (derived from x[:, :3])
            - edge_index: Graph connectivity [2, E]
            - boundary_mask: (optional) Boundary node mask
            - boundary_type: (optional) Boundary type labels
        predictions: Model predictions [N, 4] (u_x, u_y, u_z, wave_height)
        lambda_*: Loss weights
        
    Returns:
        Dictionary with:
            - total: Total combined loss
            - data: Data fidelity loss
            - continuity: Continuity equation loss
            - momentum: Momentum equation loss
            - turbulence: Turbulence model loss
            - boundary: Boundary condition loss
    """
    device = predictions.device
    
    # Extract predictions
    u_pred = predictions[:, :3]  # Velocity [N, 3]
    wave_height_pred = predictions[:, 3]  # Wave height [N]
    
    # Get positions from input features
    pos = data.x[:, :3].to(device)  # [N, 3] (x, y, z coordinates)
    edge_index = data.edge_index.to(device)
    
    # Data fidelity loss
    if hasattr(data, 'y') and data.y is not None:
        y = data.y.to(device)
        loss_data = F.mse_loss(predictions, y)
    else:
        loss_data = torch.tensor(0.0, device=device)
    
    # Continuity loss
    loss_cont = compute_continuity_loss(u_pred, pos, edge_index)
    
    # Momentum loss
    loss_mom = compute_momentum_loss(u_pred, pos, edge_index)
    
    # Turbulence loss (without explicit k, epsilon)
    loss_turb = compute_turbulence_loss(
        u_pred, k=None, epsilon=None,
        pos=pos, edge_index=edge_index
    )
    
    # Boundary loss
    if hasattr(data, 'boundary_mask') and data.boundary_mask is not None:
        loss_bc = compute_boundary_loss(
            u_pred, wave_height_pred,
            pos, data.boundary_mask.to(device),
            data.boundary_type.to(device)
        )
    else:
        loss_bc = torch.tensor(0.0, device=device)
    
    # Total loss
    loss_total = (
        lambda_data * loss_data
        + lambda_cont * loss_cont
        + lambda_mom * loss_mom
        + lambda_turb * loss_turb
        + lambda_bc * loss_bc
    )
    
    return {
        'total': loss_total,
        'data': loss_data,
        'continuity': loss_cont,
        'momentum': loss_mom,
        'turbulence': loss_turb,
        'boundary': loss_bc,
    }


class PhysicsInformedLoss(nn.Module):
    """
    Module wrapper for physics-informed loss.
    
    Convenient for use in training loops.
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_cont: float = 1.0,
        lambda_mom: float = 0.1,
        lambda_turb: float = 0.01,
        lambda_bc: float = 1.0,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_cont = lambda_cont
        self.lambda_mom = lambda_mom
        self.lambda_turb = lambda_turb
        self.lambda_bc = lambda_bc
    
    def forward(
        self,
        data,
        predictions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return physics_informed_loss(
            data, predictions,
            lambda_data=self.lambda_data,
            lambda_cont=self.lambda_cont,
            lambda_mom=self.lambda_mom,
            lambda_turb=self.lambda_turb,
            lambda_bc=self.lambda_bc,
        )


if __name__ == "__main__":
    print("Testing physics losses...")
    
    # Create dummy data
    N = 100
    E = 300
    
    # Random positions and velocities
    pos = torch.randn(N, 3)
    u = torch.randn(N, 3, requires_grad=True)
    edge_index = torch.randint(0, N, (2, E))
    
    # Test continuity loss
    loss_cont = compute_continuity_loss(u, pos, edge_index)
    print(f"Continuity loss: {loss_cont.item():.6f}")
    assert not torch.isnan(loss_cont), "Continuity loss is NaN"
    
    # Test momentum loss
    loss_mom = compute_momentum_loss(u, pos, edge_index)
    print(f"Momentum loss: {loss_mom.item():.6f}")
    assert not torch.isnan(loss_mom), "Momentum loss is NaN"
    
    # Test turbulence loss
    loss_turb = compute_turbulence_loss(u, None, None, pos, edge_index)
    print(f"Turbulence loss: {loss_turb.item():.6f}")
    assert not torch.isnan(loss_turb), "Turbulence loss is NaN"
    
    # Test gradient computation
    loss_cont.backward()
    print("✓ Backward pass successful")
    
    print("\n✓ All physics loss tests passed!")
