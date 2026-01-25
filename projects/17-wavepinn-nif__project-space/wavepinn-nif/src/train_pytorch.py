#!/usr/bin/env python3
"""
WavePINN-NIF-Scalar: PyTorch Implementation
Physics-Informed Neural Network for 2D Acoustic Wave Equation

This script trains a PINN to solve:
    u_tt = c²(x) * (u_xx + u_zz) + s(x,t)

Where:
    u(x, z, t): scalar wavefield
    c(x, z): spatially-varying velocity
    s(x, z, t): source term (Ricker wavelet)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional
import time
import argparse

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ricker_wavelet(t: torch.Tensor, f0: float = 25.0) -> torch.Tensor:
    """
    Compute Ricker (Mexican hat) wavelet.
    
    Args:
        t: Time values
        f0: Dominant frequency in Hz
        
    Returns:
        Wavelet amplitude at times t
    """
    t0 = 1.0 / f0
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1 - 2 * a) * torch.exp(-a)

def generate_slowness_map(
    nx: int, nz: int, 
    base_velocity: float = 2000.0,
    n_anomalies: int = 4,
    anomaly_strength: float = 0.3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2D slowness map with velocity anomalies.
    
    Returns:
        slowness: (nx, nz) tensor of slowness values (1/velocity)
        velocity: (nx, nz) tensor of velocity values
    """
    np.random.seed(seed)
    
    # Create base slowness
    base_slowness = 1.0 / base_velocity
    slowness = np.ones((nx, nz)) * base_slowness
    
    # Add Gaussian anomalies
    x = np.linspace(0, 1, nx)
    z = np.linspace(0, 1, nz)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    
    for _ in range(n_anomalies):
        cx = np.random.uniform(0.2, 0.8)
        cz = np.random.uniform(0.2, 0.8)
        sigma = np.random.uniform(0.05, 0.15)
        amp = np.random.uniform(-anomaly_strength, anomaly_strength) * base_slowness
        
        anomaly = amp * np.exp(-((xx - cx)**2 + (zz - cz)**2) / (2 * sigma**2))
        slowness += anomaly
    
    # Ensure positive slowness
    slowness = np.maximum(slowness, base_slowness * 0.5)
    velocity = 1.0 / slowness
    
    return torch.tensor(slowness, dtype=torch.float32), torch.tensor(velocity, dtype=torch.float32)

def sample_collocation_points(
    n_interior: int, 
    n_boundary: int, 
    n_initial: int,
    x_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 0.5),
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Sample collocation points for PINN training.
    
    Returns:
        Dict with 'interior', 'boundary', 'initial' coordinate tensors
    """
    # Interior points (x, z, t)
    interior = torch.rand(n_interior, 3, device=device)
    interior[:, 0] = interior[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    interior[:, 1] = interior[:, 1] * (z_range[1] - z_range[0]) + z_range[0]
    interior[:, 2] = interior[:, 2] * (t_range[1] - t_range[0]) + t_range[0]
    
    # Boundary points (4 edges)
    n_per_edge = n_boundary // 4
    
    # Left boundary (x=0)
    left = torch.zeros(n_per_edge, 3, device=device)
    left[:, 0] = x_range[0]
    left[:, 1] = torch.rand(n_per_edge, device=device) * (z_range[1] - z_range[0]) + z_range[0]
    left[:, 2] = torch.rand(n_per_edge, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Right boundary (x=1)
    right = torch.zeros(n_per_edge, 3, device=device)
    right[:, 0] = x_range[1]
    right[:, 1] = torch.rand(n_per_edge, device=device) * (z_range[1] - z_range[0]) + z_range[0]
    right[:, 2] = torch.rand(n_per_edge, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Top boundary (z=1)
    top = torch.zeros(n_per_edge, 3, device=device)
    top[:, 0] = torch.rand(n_per_edge, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    top[:, 1] = z_range[1]
    top[:, 2] = torch.rand(n_per_edge, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Bottom boundary (z=0)
    bottom = torch.zeros(n_per_edge, 3, device=device)
    bottom[:, 0] = torch.rand(n_per_edge, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    bottom[:, 1] = z_range[0]
    bottom[:, 2] = torch.rand(n_per_edge, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    boundary = torch.cat([left, right, top, bottom], dim=0)
    
    # Initial condition points (t=0)
    initial = torch.rand(n_initial, 3, device=device)
    initial[:, 0] = initial[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    initial[:, 1] = initial[:, 1] * (z_range[1] - z_range[0]) + z_range[0]
    initial[:, 2] = 0.0  # t = 0
    
    return {
        'interior': interior,
        'boundary': boundary,
        'initial': initial
    }

def interpolate_velocity(velocity_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation of velocity at coordinates.
    
    Args:
        velocity_grid: (nx, nz) velocity values
        coords: (N, 2) or (N, 3) coordinates in [0, 1]
        
    Returns:
        (N,) velocity values at coords
    """
    nx, nz = velocity_grid.shape
    x = coords[:, 0] * (nx - 1)
    z = coords[:, 1] * (nz - 1)
    
    x0 = torch.floor(x).long().clamp(0, nx - 2)
    z0 = torch.floor(z).long().clamp(0, nz - 2)
    x1 = x0 + 1
    z1 = z0 + 1
    
    wx = x - x0.float()
    wz = z - z0.float()
    
    v00 = velocity_grid[x0, z0]
    v01 = velocity_grid[x0, z1]
    v10 = velocity_grid[x1, z0]
    v11 = velocity_grid[x1, z1]
    
    v = (v00 * (1 - wx) * (1 - wz) +
         v01 * (1 - wx) * wz +
         v10 * wx * (1 - wz) +
         v11 * wx * wz)
    
    return v

# ============================================================================
# Neural Network Model
# ============================================================================

class FourierFeatures(nn.Module):
    """Fourier feature encoding to mitigate spectral bias."""
    
    def __init__(self, input_dim: int, num_features: int = 64, scale: float = 10.0):
        super().__init__()
        self.num_features = num_features
        # Random Gaussian basis matrix
        self.register_buffer('B', torch.randn(input_dim, num_features) * scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim)
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class WavePINN(nn.Module):
    """
    Physics-Informed Neural Network for Acoustic Wave Equation.
    
    Network: coords (x, z, t) -> u (wavefield scalar)
    """
    
    def __init__(
        self,
        hidden_dims: list = [256, 256, 128, 64],
        use_fourier: bool = True,
        num_fourier: int = 64,
        fourier_scale: float = 10.0
    ):
        super().__init__()
        
        self.use_fourier = use_fourier
        
        if use_fourier:
            self.fourier = FourierFeatures(3, num_fourier, fourier_scale)
            input_dim = num_fourier * 2  # cos + sin
        else:
            input_dim = 3
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.Tanh()
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))  # Output: u
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: (N, 3) tensor of (x, z, t) coordinates
            
        Returns:
            (N, 1) wavefield values
        """
        if self.use_fourier:
            x = self.fourier(coords)
        else:
            x = coords
        return self.mlp(x)

# ============================================================================
# PINN Loss Functions
# ============================================================================

def compute_pde_residual(
    model: WavePINN,
    coords: torch.Tensor,
    velocity: torch.Tensor
) -> torch.Tensor:
    """
    Compute PDE residual: u_tt - c²(u_xx + u_zz)
    
    Uses automatic differentiation to compute second derivatives.
    """
    coords.requires_grad_(True)
    
    # Forward pass
    u = model(coords)
    
    # First derivatives
    grad_u = torch.autograd.grad(
        u, coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    u_x = grad_u[:, 0:1]
    u_z = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]
    
    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x, coords,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]
    
    u_zz = torch.autograd.grad(
        u_z, coords,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True
    )[0][:, 1:2]
    
    u_tt = torch.autograd.grad(
        u_t, coords,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True
    )[0][:, 2:3]
    
    # PDE residual: u_tt = c² * (u_xx + u_zz)
    c_squared = (velocity ** 2).unsqueeze(-1)
    laplacian = u_xx + u_zz
    residual = u_tt - c_squared * laplacian
    
    return residual

def compute_source_term(
    coords: torch.Tensor,
    source_loc: Tuple[float, float] = (0.5, 0.1),
    f0: float = 25.0,
    sigma: float = 0.02
) -> torch.Tensor:
    """
    Compute Ricker wavelet source term at source location.
    
    Args:
        coords: (N, 3) coordinates
        source_loc: (x, z) source position
        f0: Dominant frequency
        sigma: Spatial spread of source
        
    Returns:
        (N, 1) source values
    """
    x, z, t = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Spatial envelope (Gaussian)
    dx = x - source_loc[0]
    dz = z - source_loc[1]
    spatial = torch.exp(-(dx**2 + dz**2) / (2 * sigma**2))
    
    # Temporal Ricker wavelet
    t0 = 1.0 / f0
    a = (np.pi * f0 * (t - t0)) ** 2
    temporal = (1 - 2 * a) * torch.exp(-a)
    
    return (spatial * temporal).unsqueeze(-1)

class WavePINNLoss:
    """Combined PINN loss for wave equation."""
    
    def __init__(
        self,
        model: WavePINN,
        velocity_grid: torch.Tensor,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        source_loc: Tuple[float, float] = (0.5, 0.1),
        f0: float = 25.0,
        device: str = 'cuda'
    ):
        self.model = model
        self.velocity_grid = velocity_grid.to(device)
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.source_loc = source_loc
        self.f0 = f0
        self.device = device
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total PINN loss.
        
        Returns:
            total_loss: scalar loss
            components: dict of loss components
        """
        interior = batch['interior']
        boundary = batch['boundary']
        initial = batch['initial']
        
        # Interpolate velocity at interior points
        velocity = interpolate_velocity(self.velocity_grid, interior[:, :2])
        
        # 1. PDE residual loss
        residual = compute_pde_residual(self.model, interior, velocity)
        source = compute_source_term(interior, self.source_loc, self.f0)
        pde_residual = residual - source  # u_tt - c²∇²u - s = 0
        loss_pde = torch.mean(pde_residual ** 2)
        
        # 2. Boundary condition loss (Dirichlet u=0)
        u_bc = self.model(boundary)
        loss_bc = torch.mean(u_bc ** 2)
        
        # 3. Initial condition loss (u=0, u_t=0 at t=0)
        initial.requires_grad_(True)
        u_ic = self.model(initial)
        loss_ic_u = torch.mean(u_ic ** 2)
        
        # Initial velocity condition (u_t = 0)
        grad_u_ic = torch.autograd.grad(
            u_ic, initial,
            grad_outputs=torch.ones_like(u_ic),
            create_graph=True
        )[0]
        u_t_ic = grad_u_ic[:, 2:3]
        loss_ic_ut = torch.mean(u_t_ic ** 2)
        
        loss_ic = loss_ic_u + loss_ic_ut
        
        # Total loss
        total = (self.lambda_pde * loss_pde + 
                 self.lambda_bc * loss_bc + 
                 self.lambda_ic * loss_ic)
        
        components = {
            'total': total.item(),
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item()
        }
        
        return total, components

# ============================================================================
# Training Loop
# ============================================================================

def train_wavepinn(
    model: WavePINN,
    velocity_grid: torch.Tensor,
    n_epochs: int = 500,
    lr: float = 1e-3,
    n_interior: int = 10000,
    n_boundary: int = 2000,
    n_initial: int = 1000,
    device: str = 'cuda',
    log_every: int = 50,
    resample_every: int = 100
) -> Dict[str, list]:
    """
    Train WavePINN model.
    
    Returns:
        History dict with training metrics
    """
    model = model.to(device)
    velocity_grid = velocity_grid.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    loss_fn = WavePINNLoss(model, velocity_grid, device=device)
    
    history = {'loss': [], 'pde': [], 'bc': [], 'ic': [], 'time': []}
    
    # Initial sampling
    batch = sample_collocation_points(n_interior, n_boundary, n_initial, device=device)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Resample collocation points periodically
        if epoch > 0 and epoch % resample_every == 0:
            batch = sample_collocation_points(n_interior, n_boundary, n_initial, device=device)
        
        optimizer.zero_grad()
        
        loss, components = loss_fn(batch)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - start_time
            history['loss'].append(components['total'])
            history['pde'].append(components['pde'])
            history['bc'].append(components['bc'])
            history['ic'].append(components['ic'])
            history['time'].append(elapsed)
            
            print(f"Epoch {epoch:5d}/{n_epochs} | "
                  f"Loss: {components['total']:.4e} | "
                  f"PDE: {components['pde']:.4e} | "
                  f"BC: {components['bc']:.4e} | "
                  f"IC: {components['ic']:.4e} | "
                  f"Time: {elapsed:.1f}s")
    
    return history

# ============================================================================
# Visualization
# ============================================================================

def predict_wavefield(
    model: WavePINN,
    t: float,
    nx: int = 100,
    nz: int = 100,
    device: str = 'cuda'
) -> np.ndarray:
    """Predict wavefield on a 2D grid at time t."""
    model.eval()
    
    x = torch.linspace(0, 1, nx, device=device)
    z = torch.linspace(0, 1, nz, device=device)
    xx, zz = torch.meshgrid(x, z, indexing='ij')
    
    coords = torch.stack([
        xx.flatten(),
        zz.flatten(),
        torch.full((nx * nz,), t, device=device)
    ], dim=-1)
    
    with torch.no_grad():
        u = model(coords).reshape(nx, nz)
    
    return u.cpu().numpy()

def save_results(model, velocity_grid, history, save_path: str = 'results'):
    """Save trained model and results."""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), f'{save_path}/wavepinn_model.pt')
    
    # Save history
    np.savez(f'{save_path}/history.npz', **{k: np.array(v) for k, v in history.items()})
    
    # Save velocity
    np.save(f'{save_path}/velocity.npy', velocity_grid.cpu().numpy())
    
    print(f"Results saved to {save_path}/")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train WavePINN-NIF-Scalar')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_interior', type=int, default=10000, help='Interior points')
    parser.add_argument('--n_boundary', type=int, default=2000, help='Boundary points')
    parser.add_argument('--n_initial', type=int, default=1000, help='Initial points')
    parser.add_argument('--hidden', nargs='+', type=int, default=[256, 256, 128, 64])
    parser.add_argument('--fourier', type=int, default=64, help='Fourier features')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str, default='results', help='Save path')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    set_seed(args.seed)
    
    # Generate velocity model
    print("\nGenerating velocity model...")
    slowness, velocity = generate_slowness_map(100, 100, seed=args.seed)
    print(f"Velocity range: [{velocity.min():.0f}, {velocity.max():.0f}] m/s")
    
    # Create model
    print("\nCreating WavePINN model...")
    model = WavePINN(
        hidden_dims=args.hidden,
        use_fourier=True,
        num_fourier=args.fourier
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\n" + "="*60)
    print("TRAINING WAVEPINN")
    print("="*60 + "\n")
    
    history = train_wavepinn(
        model,
        velocity,
        n_epochs=args.epochs,
        lr=args.lr,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        device=device
    )
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Initial loss: {history['loss'][0]:.4e}")
    print(f"Final loss:   {history['loss'][-1]:.4e}")
    reduction = (1 - history['loss'][-1] / history['loss'][0]) * 100
    print(f"Reduction:    {reduction:.1f}%")
    print(f"Total time:   {history['time'][-1]:.1f}s")
    
    # Save
    save_results(model, velocity, history, args.save)
    
    # Acceptance test
    print("\n" + "="*60)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*60)
    
    # 1. Check predictions are finite and bounded
    u_test = predict_wavefield(model, t=0.25, device=device)
    is_finite = np.all(np.isfinite(u_test))
    is_bounded = np.all(np.abs(u_test) < 10)
    print(f"✓ Predictions finite: {is_finite}")
    print(f"✓ Predictions bounded (|u| < 10): {is_bounded}")
    print(f"  Max |u|: {np.max(np.abs(u_test)):.4f}")
    
    # 2. Check loss reduction > 50%
    loss_reduced = reduction > 50
    print(f"✓ Loss reduced > 50%: {loss_reduced} ({reduction:.1f}%)")
    
    # Overall
    passed = is_finite and is_bounded and loss_reduced
    print(f"\n{'✓ ALL TESTS PASSED' if passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if passed else 1

if __name__ == '__main__':
    exit(main())
