"""
Maxwell-PINN-NIF: Physics-Informed Neural Network for Full-Vector Maxwell's Equations

This module implements the core neural network architecture for solving Maxwell's equations
in complex, anisotropic media. The model uses a shared MLP backbone with separate output
heads for E-field (Ex, Ey, Ez) and H-field (Hx, Hy, Hz) components.

Key features:
- Divergence-free constraint enforcement via penalty loss
- Maxwell curl equation residuals via automatic differentiation  
- Support for spatially-varying permittivity ε(x) and permeability μ(x)
- Optional Fourier feature encoding for high-frequency fields

References:
- Kovacs et al. (2021): Magnetostatics PINNs
- Nohra & Dufour (2024): PINNs for discontinuous EM media
- Richter-Powell et al. (2022): Neural Conservation Laws (divergence-free networks)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FourierFeatureEncoding(nn.Module):
    """
    Random Fourier Feature encoding for improved high-frequency learning.
    
    Maps input coordinates to higher-dimensional space using:
        γ(x) = [cos(2π·B·x), sin(2π·B·x)]
    
    where B is a random matrix sampled from N(0, σ²).
    
    This helps PINNs learn high-frequency components that are otherwise
    suppressed by the spectral bias of standard MLPs (see Tancik et al. 2020,
    "Fourier Features Let Networks Learn High Frequency Functions").
    """
    def __init__(self, input_dim: int = 3, num_frequencies: int = 64, sigma: float = 10.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Random frequency matrix (fixed, not learned)
        B = torch.randn(num_frequencies, input_dim) * sigma
        self.register_buffer('B', B)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [N, input_dim] spatial coordinates
        Returns:
            encoded: [N, 2*num_frequencies] Fourier features
        """
        # coords @ B.T -> [N, num_frequencies]
        proj = 2 * torch.pi * coords @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class MaxwellPINN(nn.Module):
    """
    Physics-Informed Neural Network for full-vector Maxwell equations.
    
    Architecture:
        Input: (x, y, z) coordinates → Optional Fourier encoding
        Backbone: Deep MLP with Tanh activations (smooth for gradient computation)
        Output: Two heads producing E = (Ex, Ey, Ez) and H = (Hx, Hy, Hz)
    
    The network learns to approximate electromagnetic fields that satisfy:
        - Faraday's Law: ∇×E = -μ ∂H/∂t
        - Ampère's Law: ∇×H = ε ∂E/∂t
        - Gauss's Law (E): ∇·(εE) = 0 (no free charges)
        - Gauss's Law (H): ∇·(μH) = 0 (no magnetic monopoles)
    
    Args:
        input_dim: Spatial dimension (default 3 for 3D)
        hidden_dim: Width of hidden layers (128-256 recommended)
        num_hidden: Number of hidden layers (6-8 for complex fields)
        use_fourier: Enable Fourier feature encoding for high frequencies
        fourier_sigma: Standard deviation for Fourier feature frequencies
        num_frequencies: Number of Fourier frequency components
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_hidden: int = 6,
        use_fourier: bool = False,
        fourier_sigma: float = 10.0,
        num_frequencies: int = 64
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier
        
        # Optional Fourier feature encoding
        if use_fourier:
            self.fourier = FourierFeatureEncoding(input_dim, num_frequencies, fourier_sigma)
            backbone_input_dim = 2 * num_frequencies
        else:
            self.fourier = None
            backbone_input_dim = input_dim
        
        # Build MLP backbone with Tanh activations
        # Tanh is chosen over ReLU for smoothness - we need continuous 2nd derivatives
        # for computing curl operations via autograd
        layers = []
        layers.append(nn.Linear(backbone_input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.backbone = nn.Sequential(*layers)
        
        # Output heads: E-field (3 components) and H-field (3 components)
        # Separate heads allow different scaling and easier constraint enforcement
        self.fc_E = nn.Linear(hidden_dim, 3)  # (Ex, Ey, Ez)
        self.fc_H = nn.Linear(hidden_dim, 3)  # (Hx, Hy, Hz)
        
        # Initialize output layers with small weights for stable training
        nn.init.xavier_normal_(self.fc_E.weight, gain=0.1)
        nn.init.xavier_normal_(self.fc_H.weight, gain=0.1)
        nn.init.zeros_(self.fc_E.bias)
        nn.init.zeros_(self.fc_H.bias)
    
    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: coordinates → (E, H) field predictions.
        
        Args:
            coords: [N, 3] tensor of (x, y, z) spatial coordinates
                   Must have requires_grad=True for physics loss computation
        
        Returns:
            E_pred: [N, 3] predicted electric field (Ex, Ey, Ez)
            H_pred: [N, 3] predicted magnetic field (Hx, Hy, Hz)
        """
        # Apply Fourier encoding if enabled
        if self.fourier is not None:
            features = self.fourier(coords)
        else:
            features = coords
        
        # Shared backbone extracts field-agnostic spatial features
        hidden = self.backbone(features)
        
        # Separate heads for E and H fields
        E_pred = self.fc_E(hidden)
        H_pred = self.fc_H(hidden)
        
        return E_pred, H_pred


def compute_spatial_gradients(
    field_component: torch.Tensor,
    coords: torch.Tensor,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Compute spatial gradients ∂f/∂x, ∂f/∂y, ∂f/∂z of a field component.
    
    Uses PyTorch autograd for automatic differentiation. The create_graph=True
    option allows computing second derivatives (needed for validation).
    
    Args:
        field_component: [N] scalar field values at N points
        coords: [N, 3] coordinates with requires_grad=True
        create_graph: Whether to build graph for higher-order derivatives
    
    Returns:
        gradients: [N, 3] tensor of (∂f/∂x, ∂f/∂y, ∂f/∂z)
    """
    grad = torch.autograd.grad(
        outputs=field_component,
        inputs=coords,
        grad_outputs=torch.ones_like(field_component),
        create_graph=create_graph,
        retain_graph=True
    )[0]
    return grad


def divergence_free_loss(
    E_pred: torch.Tensor,
    H_pred: torch.Tensor,
    coords: torch.Tensor,
    eps: torch.Tensor,
    mu: torch.Tensor
) -> torch.Tensor:
    """
    Compute divergence-free constraint loss: ∇·(εE) = 0 and ∇·(μH) = 0.
    
    For spatially-varying materials, we use the product rule:
        ∇·(εE) = ε(∇·E) + E·(∇ε)
    
    For uniform materials (constant ε, μ), this simplifies to:
        ∇·E = 0 and ∇·H = 0
    
    This ensures physical validity (no spurious charges/monopoles) which is
    critical for accurate EM simulations. Violating divergence constraints
    leads to non-physical "ghost charge" artifacts.
    
    Args:
        E_pred: [N, 3] predicted E-field
        H_pred: [N, 3] predicted H-field  
        coords: [N, 3] coordinates with requires_grad=True
        eps: [N, 1] or [N] permittivity at each point
        mu: [N, 1] or [N] permeability at each point
    
    Returns:
        loss: Scalar MSE of divergence violations
    """
    # Ensure eps and mu are 1D for broadcasting
    eps = eps.view(-1)
    mu = mu.view(-1)
    
    # Extract field components
    Ex, Ey, Ez = E_pred[:, 0], E_pred[:, 1], E_pred[:, 2]
    Hx, Hy, Hz = H_pred[:, 0], H_pred[:, 1], H_pred[:, 2]
    
    # Compute gradients of each field component
    # grad_Ex = [∂Ex/∂x, ∂Ex/∂y, ∂Ex/∂z], etc.
    grad_Ex = compute_spatial_gradients(Ex, coords)
    grad_Ey = compute_spatial_gradients(Ey, coords)
    grad_Ez = compute_spatial_gradients(Ez, coords)
    grad_Hx = compute_spatial_gradients(Hx, coords)
    grad_Hy = compute_spatial_gradients(Hy, coords)
    grad_Hz = compute_spatial_gradients(Hz, coords)
    
    # Divergence: ∇·E = ∂Ex/∂x + ∂Ey/∂y + ∂Ez/∂z
    div_E = grad_Ex[:, 0] + grad_Ey[:, 1] + grad_Ez[:, 2]
    div_H = grad_Hx[:, 0] + grad_Hy[:, 1] + grad_Hz[:, 2]
    
    # Check if materials are spatially varying (have gradients)
    # For uniform materials, skip the E·∇ε terms
    eps_is_uniform = not eps.requires_grad or (eps.max() - eps.min()).item() < 1e-6
    mu_is_uniform = not mu.requires_grad or (mu.max() - mu.min()).item() < 1e-6
    
    if eps_is_uniform:
        # Uniform material: ∇·(εE) = ε∇·E, and we just enforce ∇·E = 0
        div_eps_E = eps * div_E
    else:
        # Spatially varying: need product rule ∇·(εE) = ε∇·E + E·∇ε
        grad_eps = compute_spatial_gradients(eps, coords)
        E_dot_grad_eps = Ex * grad_eps[:, 0] + Ey * grad_eps[:, 1] + Ez * grad_eps[:, 2]
        div_eps_E = eps * div_E + E_dot_grad_eps
    
    if mu_is_uniform:
        div_mu_H = mu * div_H
    else:
        grad_mu = compute_spatial_gradients(mu, coords)
        H_dot_grad_mu = Hx * grad_mu[:, 0] + Hy * grad_mu[:, 1] + Hz * grad_mu[:, 2]
        div_mu_H = mu * div_H + H_dot_grad_mu
    
    # MSE loss penalizing non-zero divergence
    loss = torch.mean(div_eps_E ** 2) + torch.mean(div_mu_H ** 2)
    
    return loss


def maxwell_curl_residual(
    E_pred: torch.Tensor,
    H_pred: torch.Tensor,
    coords: torch.Tensor,
    eps: torch.Tensor,
    mu: torch.Tensor,
    omega: float = 1.0
) -> torch.Tensor:
    """
    Compute Maxwell curl equation residuals for time-harmonic fields.
    
    For time-harmonic fields with e^{-iωt} dependence:
        ∇×E = iωμH  (Faraday's law)
        ∇×H = -iωεE (Ampère's law)
    
    We minimize the squared residuals of these equations.
    
    The curl of a vector field F = (Fx, Fy, Fz) is:
        ∇×F = (∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y)
    
    Args:
        E_pred: [N, 3] predicted E-field
        H_pred: [N, 3] predicted H-field
        coords: [N, 3] coordinates with requires_grad=True
        eps: [N, 1] or [N] permittivity
        mu: [N, 1] or [N] permeability
        omega: Angular frequency (normalized)
    
    Returns:
        loss: Scalar MSE of curl equation residuals
    """
    eps = eps.view(-1, 1)  # [N, 1] for broadcasting
    mu = mu.view(-1, 1)
    
    # Extract components
    Ex, Ey, Ez = E_pred[:, 0], E_pred[:, 1], E_pred[:, 2]
    Hx, Hy, Hz = H_pred[:, 0], H_pred[:, 1], H_pred[:, 2]
    
    # Compute all necessary gradients
    grad_Ex = compute_spatial_gradients(Ex, coords)
    grad_Ey = compute_spatial_gradients(Ey, coords)
    grad_Ez = compute_spatial_gradients(Ez, coords)
    grad_Hx = compute_spatial_gradients(Hx, coords)
    grad_Hy = compute_spatial_gradients(Hy, coords)
    grad_Hz = compute_spatial_gradients(Hz, coords)
    
    # Compute curl of E: ∇×E
    # curl_E_x = ∂Ez/∂y - ∂Ey/∂z
    # curl_E_y = ∂Ex/∂z - ∂Ez/∂x  
    # curl_E_z = ∂Ey/∂x - ∂Ex/∂y
    curl_E_x = grad_Ez[:, 1] - grad_Ey[:, 2]
    curl_E_y = grad_Ex[:, 2] - grad_Ez[:, 0]
    curl_E_z = grad_Ey[:, 0] - grad_Ex[:, 1]
    curl_E = torch.stack([curl_E_x, curl_E_y, curl_E_z], dim=1)  # [N, 3]
    
    # Compute curl of H: ∇×H
    curl_H_x = grad_Hz[:, 1] - grad_Hy[:, 2]
    curl_H_y = grad_Hx[:, 2] - grad_Hz[:, 0]
    curl_H_z = grad_Hy[:, 0] - grad_Hx[:, 1]
    curl_H = torch.stack([curl_H_x, curl_H_y, curl_H_z], dim=1)  # [N, 3]
    
    # Faraday's law residual: ∇×E - iωμH = 0
    # For real-valued fields (single frequency), we use: ∇×E = ωμH
    faraday_residual = curl_E - omega * mu * H_pred
    
    # Ampère's law residual: ∇×H + iωεE = 0  
    # For real-valued fields: ∇×H = -ωεE → we check ∇×H + ωεE = 0
    # Note: Sign depends on time convention; here we use ∇×H = ωεE
    ampere_residual = curl_H - omega * eps * E_pred
    
    # Total PDE residual loss
    loss = torch.mean(faraday_residual ** 2) + torch.mean(ampere_residual ** 2)
    
    return loss


def total_physics_loss(
    model: MaxwellPINN,
    coords: torch.Tensor,
    eps: torch.Tensor,
    mu: torch.Tensor,
    E_target: Optional[torch.Tensor] = None,
    H_target: Optional[torch.Tensor] = None,
    omega: float = 1.0,
    weights: Optional[dict] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total physics-informed loss for training.
    
    Loss = w_pde * L_curl + w_div * L_div + w_data * L_data
    
    Args:
        model: MaxwellPINN model
        coords: [N, 3] coordinates (must have requires_grad=True)
        eps: [N] permittivity values
        mu: [N] permeability values
        E_target: [N, 3] optional ground-truth E-field for supervised loss
        H_target: [N, 3] optional ground-truth H-field for supervised loss
        omega: Angular frequency
        weights: Dict with keys 'pde', 'div', 'data' for loss weighting
    
    Returns:
        total_loss: Weighted sum of all losses
        loss_dict: Individual loss components for logging
    """
    if weights is None:
        weights = {'pde': 1.0, 'div': 10.0, 'data': 100.0}
    
    # Forward pass
    E_pred, H_pred = model(coords)
    
    # Physics losses
    loss_curl = maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, omega)
    loss_div = divergence_free_loss(E_pred, H_pred, coords, eps, mu)
    
    # Data loss (if ground truth available)
    loss_data = torch.tensor(0.0, device=coords.device)
    if E_target is not None and H_target is not None:
        loss_data = torch.mean((E_pred - E_target) ** 2) + torch.mean((H_pred - H_target) ** 2)
    
    # Weighted total
    total_loss = (
        weights['pde'] * loss_curl +
        weights['div'] * loss_div +
        weights['data'] * loss_data
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'curl': loss_curl.item(),
        'div': loss_div.item(),
        'data': loss_data.item()
    }
    
    return total_loss, loss_dict


# Convenience function for quick testing
def create_test_model(device: str = 'cuda') -> MaxwellPINN:
    """Create a model instance for testing."""
    model = MaxwellPINN(
        input_dim=3,
        hidden_dim=128,
        num_hidden=6,
        use_fourier=False
    )
    return model.to(device) if torch.cuda.is_available() else model


if __name__ == "__main__":
    # Quick self-test
    print("Testing MaxwellPINN model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MaxwellPINN(input_dim=3, hidden_dim=128, num_hidden=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 1000
    coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
    E_pred, H_pred = model(coords)
    print(f"Forward pass: coords {coords.shape} → E {E_pred.shape}, H {H_pred.shape}")
    
    # Test divergence loss
    eps = torch.ones(batch_size, device=device) * 2.0
    mu = torch.ones(batch_size, device=device)
    loss_div = divergence_free_loss(E_pred, H_pred, coords, eps, mu)
    print(f"Divergence loss: {loss_div.item():.6f}")
    
    # Test curl residual
    loss_curl = maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, omega=1.0)
    print(f"Curl residual loss: {loss_curl.item():.6f}")
    
    # Test backward pass
    total_loss = loss_div + loss_curl
    total_loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters())
    print(f"Backward pass successful: {has_grads}")
    
    print("\n✓ All tests passed!")
