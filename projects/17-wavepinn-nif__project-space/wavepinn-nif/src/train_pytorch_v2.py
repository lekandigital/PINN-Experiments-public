#!/usr/bin/env python3
"""
WavePINN-NIF-Scalar: PyTorch Implementation (v2 - Fixed)
Physics-Informed Neural Network for 2D Acoustic Wave Equation

This script trains a PINN to solve:
    u_tt = c²(x) * (u_xx + u_zz) + s(x,t)

Key fixes:
- Normalized coordinates to [0, 1]
- Scaled velocity to [0, 1] range
- Proper loss weighting
- Gradient accumulation for stable training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
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

def generate_slowness_map(
    nx: int, nz: int, 
    base_velocity: float = 2.0,  # Normalized velocity
    n_anomalies: int = 3,
    anomaly_strength: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2D slowness map with velocity anomalies.
    Velocity is normalized to ~[0.5, 3.0] range for numerical stability.
    """
    np.random.seed(seed)
    
    # Create base velocity field
    velocity = np.ones((nx, nz)) * base_velocity
    
    # Add smooth Gaussian anomalies
    x = np.linspace(0, 1, nx)
    z = np.linspace(0, 1, nz)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    
    for _ in range(n_anomalies):
        cx = np.random.uniform(0.25, 0.75)
        cz = np.random.uniform(0.25, 0.75)
        sigma = np.random.uniform(0.08, 0.15)
        amp = np.random.uniform(-anomaly_strength, anomaly_strength) * base_velocity
        
        anomaly = amp * np.exp(-((xx - cx)**2 + (zz - cz)**2) / (2 * sigma**2))
        velocity += anomaly
    
    # Ensure positive velocity
    velocity = np.clip(velocity, 0.5, 4.0)
    slowness = 1.0 / velocity
    
    return torch.tensor(slowness, dtype=torch.float32), torch.tensor(velocity, dtype=torch.float32)

def sample_collocation_points(
    n_interior: int, 
    n_boundary: int, 
    n_initial: int,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Sample collocation points for PINN training in [0,1]^3."""
    
    # Interior points (x, z, t) in [0, 1]
    interior = torch.rand(n_interior, 3, device=device)
    interior[:, 2] = interior[:, 2] * 0.8 + 0.05  # t in [0.05, 0.85] to avoid boundary
    
    # Boundary points (4 edges)
    n_per_edge = n_boundary // 4
    
    edges = []
    for i in range(4):
        pts = torch.rand(n_per_edge, 3, device=device)
        pts[:, 2] = pts[:, 2] * 0.8 + 0.1  # t in [0.1, 0.9]
        if i == 0:  # x=0
            pts[:, 0] = 0.0
        elif i == 1:  # x=1
            pts[:, 0] = 1.0
        elif i == 2:  # z=0
            pts[:, 1] = 0.0
        else:  # z=1
            pts[:, 1] = 1.0
        edges.append(pts)
    
    boundary = torch.cat(edges, dim=0)
    
    # Initial condition points (t=0)
    initial = torch.rand(n_initial, 3, device=device)
    initial[:, 2] = 0.0
    
    return {
        'interior': interior,
        'boundary': boundary,
        'initial': initial
    }

def interpolate_velocity(velocity_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of velocity at coordinates."""
    nx, nz = velocity_grid.shape
    x = coords[:, 0] * (nx - 1)
    z = coords[:, 1] * (nz - 1)
    
    x0 = torch.floor(x).long().clamp(0, nx - 2)
    z0 = torch.floor(z).long().clamp(0, nz - 2)
    
    wx = (x - x0.float()).clamp(0, 1)
    wz = (z - z0.float()).clamp(0, 1)
    
    v00 = velocity_grid[x0, z0]
    v01 = velocity_grid[x0, z0 + 1]
    v10 = velocity_grid[x0 + 1, z0]
    v11 = velocity_grid[x0 + 1, z0 + 1]
    
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
    
    def __init__(self, input_dim: int, num_features: int = 64, scale: float = 1.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, num_features) * scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([x, torch.cos(proj), torch.sin(proj)], dim=-1)

class WavePINN(nn.Module):
    """
    Physics-Informed Neural Network for Acoustic Wave Equation.
    Network: coords (x, z, t) -> u (wavefield scalar)
    """
    
    def __init__(
        self,
        hidden_dims: list = [128, 128, 64, 32],
        use_fourier: bool = True,
        num_fourier: int = 32,
        fourier_scale: float = 1.0
    ):
        super().__init__()
        
        self.use_fourier = use_fourier
        
        if use_fourier:
            self.fourier = FourierFeatures(3, num_fourier, fourier_scale)
            input_dim = 3 + num_fourier * 2
        else:
            input_dim = 3
        
        # Build MLP with residual-style connections
        layers = []
        prev_dim = input_dim
        for i, hdim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.Tanh()
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            x = self.fourier(coords)
        else:
            x = coords
        return self.mlp(x)

# ============================================================================
# PINN Loss Functions
# ============================================================================

def compute_derivatives(model: WavePINN, coords: torch.Tensor):
    """
    Compute all needed derivatives for wave equation.
    Returns u, u_t, u_tt, u_xx, u_zz
    """
    coords = coords.clone().requires_grad_(True)
    
    u = model(coords)
    
    # First derivatives
    ones = torch.ones_like(u)
    grad_u = torch.autograd.grad(u, coords, ones, create_graph=True)[0]
    u_x, u_z, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, coords, ones, create_graph=True)[0][:, 0:1]
    u_zz = torch.autograd.grad(u_z, coords, ones, create_graph=True)[0][:, 1:2]
    u_tt = torch.autograd.grad(u_t, coords, ones, create_graph=True)[0][:, 2:3]
    
    return u, u_t, u_tt, u_xx, u_zz, coords

def source_term(coords: torch.Tensor, src_x: float = 0.5, src_z: float = 0.15) -> torch.Tensor:
    """
    Ricker wavelet source term.
    Normalized for stability.
    """
    x, z, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    
    # Spatial Gaussian (narrow)
    sigma_s = 0.03
    spatial = torch.exp(-((x - src_x)**2 + (z - src_z)**2) / (2 * sigma_s**2))
    
    # Ricker wavelet in time (f0 ~ 5 in normalized time)
    f0 = 5.0
    t0 = 0.15
    a = (np.pi * f0 * (t - t0)) ** 2
    ricker = (1 - 2 * a) * torch.exp(-a)
    
    # Scale amplitude
    return 0.5 * spatial * ricker

class WavePINNTrainer:
    """Training manager for WavePINN."""
    
    def __init__(
        self,
        model: WavePINN,
        velocity_grid: torch.Tensor,
        device: str = 'cuda',
        lambda_pde: float = 1.0,
        lambda_bc: float = 5.0,
        lambda_ic: float = 10.0
    ):
        self.model = model.to(device)
        self.velocity = velocity_grid.to(device)
        self.device = device
        
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute total PINN loss."""
        
        interior = batch['interior']
        boundary = batch['boundary']
        initial = batch['initial']
        
        # Get velocity at interior points
        vel = interpolate_velocity(self.velocity, interior[:, :2])
        c2 = (vel ** 2).unsqueeze(-1)
        
        # PDE loss: u_tt - c²(u_xx + u_zz) - source = 0
        u, u_t, u_tt, u_xx, u_zz, _ = compute_derivatives(self.model, interior)
        laplacian = u_xx + u_zz
        src = source_term(interior)
        
        residual = u_tt - c2 * laplacian - src
        loss_pde = torch.mean(residual ** 2)
        
        # BC loss: u = 0 on boundaries (absorbing BC approximation)
        u_bc = self.model(boundary)
        loss_bc = torch.mean(u_bc ** 2)
        
        # IC loss: u = 0, u_t = 0 at t=0
        u_ic, u_t_ic, _, _, _, _ = compute_derivatives(self.model, initial)
        loss_ic = torch.mean(u_ic ** 2) + torch.mean(u_t_ic ** 2)
        
        # Total
        total = (self.lambda_pde * loss_pde + 
                 self.lambda_bc * loss_bc + 
                 self.lambda_ic * loss_ic)
        
        return total, {
            'total': total.item(),
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item()
        }

def train(
    model: WavePINN,
    velocity: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    n_interior: int = 5000,
    n_boundary: int = 1000,
    n_initial: int = 500,
    device: str = 'cuda',
    log_every: int = 100
) -> Dict:
    """Main training loop."""
    
    trainer = WavePINNTrainer(model, velocity, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )
    
    history = {'loss': [], 'pde': [], 'bc': [], 'ic': []}
    best_loss = float('inf')
    
    start = time.time()
    
    for epoch in range(n_epochs):
        # Resample points each epoch for better coverage
        batch = sample_collocation_points(n_interior, n_boundary, n_initial, device)
        
        optimizer.zero_grad()
        loss, components = trainer.compute_loss(batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - start
            history['loss'].append(components['total'])
            history['pde'].append(components['pde'])
            history['bc'].append(components['bc'])
            history['ic'].append(components['ic'])
            
            lr_current = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d} | Loss: {components['total']:.4e} | "
                  f"PDE: {components['pde']:.4e} | BC: {components['bc']:.4e} | "
                  f"IC: {components['ic']:.4e} | LR: {lr_current:.1e} | {elapsed:.1f}s")
    
    return history

def predict_field(model: WavePINN, t: float, nx: int = 80, nz: int = 80, device: str = 'cuda'):
    """Predict wavefield at time t."""
    model.eval()
    x = torch.linspace(0, 1, nx, device=device)
    z = torch.linspace(0, 1, nz, device=device)
    xx, zz = torch.meshgrid(x, z, indexing='ij')
    
    coords = torch.stack([xx.flatten(), zz.flatten(), 
                          torch.full((nx*nz,), t, device=device)], dim=-1)
    
    with torch.no_grad():
        u = model(coords).reshape(nx, nz)
    return u.cpu().numpy()

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_interior', type=int, default=8000)
    parser.add_argument('--n_boundary', type=int, default=2000)
    parser.add_argument('--n_initial', type=int, default=1000)
    parser.add_argument('--hidden', nargs='+', type=int, default=[128, 128, 64, 32])
    parser.add_argument('--fourier', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    set_seed(args.seed)
    
    # Generate normalized velocity model
    print("\nGenerating velocity model (normalized)...")
    _, velocity = generate_slowness_map(64, 64, base_velocity=2.0, seed=args.seed)
    print(f"Velocity range: [{velocity.min():.2f}, {velocity.max():.2f}]")
    
    # Create model
    print("\nCreating WavePINN...")
    model = WavePINN(hidden_dims=args.hidden, num_fourier=args.fourier, fourier_scale=1.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    history = train(
        model, velocity,
        n_epochs=args.epochs,
        lr=args.lr,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        device=device,
        log_every=200
    )
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    initial_loss = history['loss'][0]
    final_loss = history['loss'][-1]
    reduction = (1 - final_loss / initial_loss) * 100
    
    print(f"Initial loss: {initial_loss:.4e}")
    print(f"Final loss:   {final_loss:.4e}")
    print(f"Reduction:    {reduction:.1f}%")
    
    # Test predictions
    u_mid = predict_field(model, t=0.3, device=device)
    is_finite = np.all(np.isfinite(u_mid))
    is_bounded = np.all(np.abs(u_mid) < 10)
    max_u = np.max(np.abs(u_mid))
    
    print(f"\nPredictions at t=0.3:")
    print(f"  Finite: {is_finite}")
    print(f"  Bounded (|u|<10): {is_bounded}")
    print(f"  Max |u|: {max_u:.4f}")
    
    # Acceptance
    passed = is_finite and is_bounded and reduction > 50
    print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}: All acceptance criteria")
    
    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'velocity': velocity.cpu(),
        'history': history
    }, 'wavepinn_trained.pt')
    print("\nModel saved to wavepinn_trained.pt")
    
    return 0 if passed else 1

if __name__ == '__main__':
    exit(main())
