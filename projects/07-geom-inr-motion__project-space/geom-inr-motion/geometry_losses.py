#!/usr/bin/env python3
"""
geometry_losses.py - Differential Geometry Losses for Geom-INR-Motion

This module implements geometric regularization losses based on differential
geometry of curves. These losses encourage smooth, physically plausible
motion trajectories by penalizing:

1. **Curvature (κ)**: Measures how sharply a trajectory bends
   - Low curvature = smooth, gradual turns
   - High curvature = sharp, jerky movements

2. **Torsion (τ)**: Measures how a trajectory twists out of its osculating plane
   - Low torsion = motion stays in a plane
   - High torsion = rapid 3D twisting

Mathematical Background:
    For a parametric curve r(t) in 3D, the Frenet-Serret formulas define:
    
    Curvature: κ = |r' × r''| / |r'|³
    Torsion:   τ = (r' × r'') · r''' / |r' × r''|²
    
    where r', r'', r''' are the first, second, and third derivatives.

References:
    - Frenet-Serret formulas: https://en.wikipedia.org/wiki/Frenet–Serret_formulas
    - SIREN: Sitzmann et al. 2020 (implicit derivative computation)

Usage:
    from geometry_losses import GeometryLoss, compute_curvature_torsion
    
    # Compute losses on predicted trajectory
    trajectory = model.predict_trajectory(actor_id=0, times=times)  # (T, J, 3)
    
    geo_loss = GeometryLoss(curvature_weight=0.001, torsion_weight=0.0001)
    loss_dict = geo_loss(trajectory)
    
    total_loss = loss_dict['curvature'] + loss_dict['torsion']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


# ============================================================================
# Finite Difference Derivatives
# ============================================================================

def compute_derivatives(
    trajectory: torch.Tensor,
    dt: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute first, second, and third derivatives using finite differences.
    
    Uses central differences where possible for better accuracy.
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions over time
        dt: Time step between frames
    
    Returns:
        velocity: (T-1, J, 3) first derivative (r')
        acceleration: (T-2, J, 3) second derivative (r'')
        jerk: (T-3, J, 3) third derivative (r''')
    """
    # First derivative: r'(t) ≈ (r(t+1) - r(t)) / dt
    velocity = (trajectory[1:] - trajectory[:-1]) / dt  # (T-1, J, 3)
    
    # Second derivative: r''(t) ≈ (r'(t+1) - r'(t)) / dt
    acceleration = (velocity[1:] - velocity[:-1]) / dt  # (T-2, J, 3)
    
    # Third derivative: r'''(t) ≈ (r''(t+1) - r''(t)) / dt
    jerk = (acceleration[1:] - acceleration[:-1]) / dt  # (T-3, J, 3)
    
    return velocity, acceleration, jerk


def compute_derivatives_central(
    trajectory: torch.Tensor,
    dt: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives using central finite differences (more accurate).
    
    Central difference formulas:
        r'(t) ≈ (r(t+1) - r(t-1)) / (2*dt)
        r''(t) ≈ (r(t+1) - 2*r(t) + r(t-1)) / dt²
        r'''(t) ≈ (r(t+2) - 2*r(t+1) + 2*r(t-1) - r(t-2)) / (2*dt³)
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions over time
        dt: Time step between frames
    
    Returns:
        velocity: (T-2, J, 3) first derivative
        acceleration: (T-2, J, 3) second derivative
        jerk: (T-4, J, 3) third derivative
    """
    T = trajectory.shape[0]
    
    # Central difference for first derivative
    # r'[i] = (r[i+1] - r[i-1]) / (2*dt) for i in [1, T-2]
    velocity = (trajectory[2:] - trajectory[:-2]) / (2 * dt)  # (T-2, J, 3)
    
    # Central difference for second derivative
    # r''[i] = (r[i+1] - 2*r[i] + r[i-1]) / dt² for i in [1, T-2]
    acceleration = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / (dt ** 2)  # (T-2, J, 3)
    
    # Central difference for third derivative
    # r'''[i] = (r[i+2] - 2*r[i+1] + 2*r[i-1] - r[i-2]) / (2*dt³) for i in [2, T-3]
    if T >= 5:
        jerk = (trajectory[4:] - 2 * trajectory[3:-1] + 2 * trajectory[1:-3] - trajectory[:-4]) / (2 * dt ** 3)
    else:
        jerk = torch.zeros(0, trajectory.shape[1], 3, device=trajectory.device)
    
    # Align lengths: use minimum common length
    min_len = min(velocity.shape[0], acceleration.shape[0], jerk.shape[0]) if jerk.shape[0] > 0 else 0
    
    if min_len > 0:
        velocity = velocity[:min_len]
        acceleration = acceleration[:min_len]
        jerk = jerk[:min_len]
    
    return velocity, acceleration, jerk


# ============================================================================
# Curvature Computation
# ============================================================================

def compute_curvature(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute curvature κ from velocity and acceleration.
    
    Curvature formula: κ = |v × a| / |v|³
    
    Args:
        velocity: (N, J, 3) first derivative
        acceleration: (N, J, 3) second derivative
        eps: Small constant for numerical stability
    
    Returns:
        curvature: (N, J) curvature values
    """
    # Cross product: v × a
    cross = torch.cross(velocity, acceleration, dim=-1)  # (N, J, 3)
    
    # Magnitudes
    cross_norm = torch.norm(cross, dim=-1)  # (N, J)
    vel_norm = torch.norm(velocity, dim=-1)  # (N, J)
    
    # Curvature: |v × a| / |v|³
    curvature = cross_norm / (vel_norm ** 3 + eps)
    
    return curvature


def compute_curvature_loss(
    trajectory: torch.Tensor,
    eps: float = 1e-6,
    use_central: bool = True
) -> torch.Tensor:
    """
    Compute mean squared curvature loss over trajectory.
    
    L_κ = (1/NJ) Σ κ²
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
        eps: Numerical stability constant
        use_central: Whether to use central differences
    
    Returns:
        loss: Scalar curvature loss
    """
    if use_central:
        velocity, acceleration, _ = compute_derivatives_central(trajectory)
    else:
        velocity, acceleration, _ = compute_derivatives(trajectory)
    
    # Ensure matching lengths
    min_len = min(velocity.shape[0], acceleration.shape[0])
    if min_len < 2:
        return torch.tensor(0.0, device=trajectory.device)
    
    velocity = velocity[:min_len]
    acceleration = acceleration[:min_len]
    
    curvature = compute_curvature(velocity, acceleration, eps)
    
    # Mean squared curvature
    loss = (curvature ** 2).mean()
    
    return loss


# ============================================================================
# Torsion Computation
# ============================================================================

def compute_torsion(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute torsion τ from velocity, acceleration, and jerk.
    
    Torsion formula: τ = (v × a) · j / |v × a|²
    
    Args:
        velocity: (N, J, 3) first derivative
        acceleration: (N, J, 3) second derivative
        jerk: (N, J, 3) third derivative
        eps: Small constant for numerical stability
    
    Returns:
        torsion: (N, J) torsion values
    """
    # Cross product: v × a
    cross = torch.cross(velocity, acceleration, dim=-1)  # (N, J, 3)
    
    # Dot product: (v × a) · j
    numerator = (cross * jerk).sum(dim=-1)  # (N, J)
    
    # Denominator: |v × a|²
    denominator = (torch.norm(cross, dim=-1) ** 2) + eps  # (N, J)
    
    # Torsion
    torsion = numerator / denominator
    
    return torsion


def compute_torsion_loss(
    trajectory: torch.Tensor,
    eps: float = 1e-6,
    use_central: bool = True
) -> torch.Tensor:
    """
    Compute mean squared torsion loss over trajectory.
    
    L_τ = (1/NJ) Σ τ²
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
        eps: Numerical stability constant
        use_central: Whether to use central differences
    
    Returns:
        loss: Scalar torsion loss
    """
    if use_central:
        velocity, acceleration, jerk = compute_derivatives_central(trajectory)
    else:
        velocity, acceleration, jerk = compute_derivatives(trajectory)
    
    # Ensure matching lengths
    min_len = min(velocity.shape[0], acceleration.shape[0], jerk.shape[0])
    if min_len < 2:
        return torch.tensor(0.0, device=trajectory.device)
    
    velocity = velocity[:min_len]
    acceleration = acceleration[:min_len]
    jerk = jerk[:min_len]
    
    torsion = compute_torsion(velocity, acceleration, jerk, eps)
    
    # Mean squared torsion
    loss = (torsion ** 2).mean()
    
    return loss


# ============================================================================
# Combined Curvature-Torsion Computation
# ============================================================================

def compute_curvature_torsion(
    trajectory: torch.Tensor,
    eps: float = 1e-6,
    use_central: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both curvature and torsion efficiently.
    
    This is more efficient than calling compute_curvature and compute_torsion
    separately as it shares the derivative computation.
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
        eps: Numerical stability constant
        use_central: Whether to use central differences
    
    Returns:
        curvature: (N, J) curvature values
        torsion: (N, J) torsion values
    """
    if use_central:
        velocity, acceleration, jerk = compute_derivatives_central(trajectory)
    else:
        velocity, acceleration, jerk = compute_derivatives(trajectory)
    
    # Ensure matching lengths
    min_len = min(velocity.shape[0], acceleration.shape[0], jerk.shape[0])
    if min_len < 1:
        device = trajectory.device
        return torch.zeros(0, trajectory.shape[1], device=device), \
               torch.zeros(0, trajectory.shape[1], device=device)
    
    velocity = velocity[:min_len]
    acceleration = acceleration[:min_len]
    jerk = jerk[:min_len]
    
    # Cross product: v × a
    cross = torch.cross(velocity, acceleration, dim=-1)  # (N, J, 3)
    cross_norm = torch.norm(cross, dim=-1)  # (N, J)
    vel_norm = torch.norm(velocity, dim=-1)  # (N, J)
    
    # Curvature: |v × a| / |v|³
    curvature = cross_norm / (vel_norm ** 3 + eps)
    
    # Torsion: (v × a) · j / |v × a|²
    numerator = (cross * jerk).sum(dim=-1)  # (N, J)
    torsion = numerator / (cross_norm ** 2 + eps)
    
    return curvature, torsion


# ============================================================================
# Additional Smoothness Metrics
# ============================================================================

def compute_jerk_loss(trajectory: torch.Tensor, use_central: bool = True) -> torch.Tensor:
    """
    Compute jerk (third derivative) loss for smoothness.
    
    Lower jerk = smoother motion with less sudden changes in acceleration.
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
        use_central: Whether to use central differences
    
    Returns:
        loss: Mean squared jerk magnitude
    """
    if use_central:
        _, _, jerk = compute_derivatives_central(trajectory)
    else:
        _, _, jerk = compute_derivatives(trajectory)
    
    if jerk.shape[0] < 1:
        return torch.tensor(0.0, device=trajectory.device)
    
    # Mean squared jerk magnitude
    jerk_magnitude = torch.norm(jerk, dim=-1)  # (N, J)
    loss = (jerk_magnitude ** 2).mean()
    
    return loss


def compute_acceleration_loss(trajectory: torch.Tensor, use_central: bool = True) -> torch.Tensor:
    """
    Compute acceleration magnitude loss.
    
    Penalizes high accelerations for smoother motion.
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
        use_central: Whether to use central differences
    
    Returns:
        loss: Mean squared acceleration magnitude
    """
    if use_central:
        _, acceleration, _ = compute_derivatives_central(trajectory)
    else:
        _, acceleration, _ = compute_derivatives(trajectory)
    
    if acceleration.shape[0] < 1:
        return torch.tensor(0.0, device=trajectory.device)
    
    accel_magnitude = torch.norm(acceleration, dim=-1)
    loss = (accel_magnitude ** 2).mean()
    
    return loss


def compute_bone_length_loss(
    trajectory: torch.Tensor,
    bone_pairs: Optional[torch.Tensor] = None,
    target_lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute bone length consistency loss.
    
    Encourages consistent bone lengths across the sequence.
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
        bone_pairs: (B, 2) tensor of joint index pairs defining bones
        target_lengths: (B,) target bone lengths (if None, use mean from sequence)
    
    Returns:
        loss: Bone length variance loss
    """
    T, J, _ = trajectory.shape
    device = trajectory.device
    
    if bone_pairs is None:
        # Default: simple chain skeleton
        bone_pairs = torch.tensor([[i, i+1] for i in range(min(J-1, 23))], device=device)
    
    B = bone_pairs.shape[0]
    
    # Compute bone vectors for each frame
    start_joints = trajectory[:, bone_pairs[:, 0], :]  # (T, B, 3)
    end_joints = trajectory[:, bone_pairs[:, 1], :]    # (T, B, 3)
    bone_vectors = end_joints - start_joints           # (T, B, 3)
    bone_lengths = torch.norm(bone_vectors, dim=-1)    # (T, B)
    
    if target_lengths is None:
        # Use mean length across sequence as target
        target_lengths = bone_lengths.mean(dim=0, keepdim=True)  # (1, B)
    
    # Loss: variance from target
    loss = ((bone_lengths - target_lengths) ** 2).mean()
    
    return loss


# ============================================================================
# Combined Geometry Loss Module
# ============================================================================

class GeometryLoss(nn.Module):
    """
    Combined geometry loss module for Geom-INR-Motion.
    
    Computes weighted sum of:
    - Curvature loss (κ²)
    - Torsion loss (τ²)
    - Jerk loss (optional)
    - Bone length loss (optional)
    
    Usage:
        geo_loss = GeometryLoss(curvature_weight=0.001, torsion_weight=0.0001)
        loss_dict = geo_loss(trajectory)
        total = loss_dict['total']
    """
    
    def __init__(
        self,
        curvature_weight: float = 0.001,
        torsion_weight: float = 0.0001,
        jerk_weight: float = 0.0,
        bone_length_weight: float = 0.0,
        use_central: bool = True,
        eps: float = 1e-6
    ):
        """
        Initialize geometry loss.
        
        Args:
            curvature_weight: Weight for curvature loss (λ_κ)
            torsion_weight: Weight for torsion loss (λ_τ)
            jerk_weight: Weight for jerk loss (0 = disabled)
            bone_length_weight: Weight for bone length loss (0 = disabled)
            use_central: Whether to use central differences
            eps: Numerical stability constant
        """
        super().__init__()
        
        self.curvature_weight = curvature_weight
        self.torsion_weight = torsion_weight
        self.jerk_weight = jerk_weight
        self.bone_length_weight = bone_length_weight
        self.use_central = use_central
        self.eps = eps
    
    def forward(
        self,
        trajectory: torch.Tensor,
        bone_pairs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute geometry losses on trajectory.
        
        Args:
            trajectory: (T, J, 3) tensor of joint positions
            bone_pairs: Optional bone definitions for bone length loss
        
        Returns:
            Dictionary with individual and total losses
        """
        device = trajectory.device
        losses = {}
        total = torch.tensor(0.0, device=device)
        
        # Curvature and torsion (computed together for efficiency)
        if self.curvature_weight > 0 or self.torsion_weight > 0:
            curvature, torsion = compute_curvature_torsion(
                trajectory, self.eps, self.use_central
            )
            
            if curvature.numel() > 0:
                curv_loss = (curvature ** 2).mean()
                tors_loss = (torsion ** 2).mean()
            else:
                curv_loss = torch.tensor(0.0, device=device)
                tors_loss = torch.tensor(0.0, device=device)
            
            losses['curvature'] = curv_loss
            losses['torsion'] = tors_loss
            
            total = total + self.curvature_weight * curv_loss
            total = total + self.torsion_weight * tors_loss
        
        # Jerk loss
        if self.jerk_weight > 0:
            jerk_loss = compute_jerk_loss(trajectory, self.use_central)
            losses['jerk'] = jerk_loss
            total = total + self.jerk_weight * jerk_loss
        
        # Bone length loss
        if self.bone_length_weight > 0:
            bone_loss = compute_bone_length_loss(trajectory, bone_pairs)
            losses['bone_length'] = bone_loss
            total = total + self.bone_length_weight * bone_loss
        
        losses['total'] = total
        
        return losses
    
    def __repr__(self):
        return (
            f"GeometryLoss(curvature_weight={self.curvature_weight}, "
            f"torsion_weight={self.torsion_weight}, "
            f"jerk_weight={self.jerk_weight}, "
            f"bone_length_weight={self.bone_length_weight})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def analyze_trajectory_geometry(trajectory: torch.Tensor) -> Dict[str, float]:
    """
    Analyze geometric properties of a trajectory.
    
    Args:
        trajectory: (T, J, 3) tensor of joint positions
    
    Returns:
        Dictionary of statistics
    """
    curvature, torsion = compute_curvature_torsion(trajectory)
    
    if curvature.numel() == 0:
        return {
            'curvature_mean': 0.0,
            'curvature_max': 0.0,
            'curvature_std': 0.0,
            'torsion_mean': 0.0,
            'torsion_max': 0.0,
            'torsion_std': 0.0,
        }
    
    return {
        'curvature_mean': curvature.mean().item(),
        'curvature_max': curvature.max().item(),
        'curvature_std': curvature.std().item(),
        'torsion_mean': torsion.abs().mean().item(),
        'torsion_max': torsion.abs().max().item(),
        'torsion_std': torsion.std().item(),
    }


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Geometry Losses Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test 1: Smooth trajectory (low curvature/torsion)
    print("\n[1] Testing smooth trajectory (sine wave)...")
    T, J = 100, 24
    t = torch.linspace(0, 2 * np.pi, T, device=device)
    
    # Create smooth sinusoidal motion
    smooth_traj = torch.zeros(T, J, 3, device=device)
    for j in range(J):
        phase = j * 0.1
        smooth_traj[:, j, 0] = t / (2 * np.pi)  # Linear x
        smooth_traj[:, j, 1] = 0.1 * torch.sin(t + phase)  # Smooth y
        smooth_traj[:, j, 2] = 0.1 * torch.cos(t + phase)  # Smooth z
    
    smooth_stats = analyze_trajectory_geometry(smooth_traj)
    print(f"    Curvature: mean={smooth_stats['curvature_mean']:.4f}, max={smooth_stats['curvature_max']:.4f}")
    print(f"    Torsion:   mean={smooth_stats['torsion_mean']:.4f}, max={smooth_stats['torsion_max']:.4f}")
    
    # Test 2: Noisy trajectory (high curvature/torsion)
    print("\n[2] Testing noisy trajectory (random noise)...")
    noisy_traj = smooth_traj + 0.1 * torch.randn_like(smooth_traj)
    
    noisy_stats = analyze_trajectory_geometry(noisy_traj)
    print(f"    Curvature: mean={noisy_stats['curvature_mean']:.4f}, max={noisy_stats['curvature_max']:.4f}")
    print(f"    Torsion:   mean={noisy_stats['torsion_mean']:.4f}, max={noisy_stats['torsion_max']:.4f}")
    
    # Verify noisy has higher curvature
    assert noisy_stats['curvature_mean'] > smooth_stats['curvature_mean'], \
        "Noisy trajectory should have higher curvature"
    print("    ✓ Noisy trajectory correctly has higher curvature")
    
    # Test 3: GeometryLoss module
    print("\n[3] Testing GeometryLoss module...")
    geo_loss = GeometryLoss(
        curvature_weight=0.001,
        torsion_weight=0.0001,
        jerk_weight=0.00001
    )
    print(f"    {geo_loss}")
    
    smooth_losses = geo_loss(smooth_traj)
    noisy_losses = geo_loss(noisy_traj)
    
    print(f"\n    Smooth trajectory losses:")
    for k, v in smooth_losses.items():
        print(f"      {k}: {v.item():.6f}")
    
    print(f"\n    Noisy trajectory losses:")
    for k, v in noisy_losses.items():
        print(f"      {k}: {v.item():.6f}")
    
    assert noisy_losses['total'] > smooth_losses['total'], \
        "Noisy trajectory should have higher total loss"
    print("\n    ✓ Noisy trajectory correctly has higher total loss")
    
    # Test 4: Gradient flow
    print("\n[4] Testing gradient flow...")
    traj_grad = smooth_traj.clone().requires_grad_(True)
    losses = geo_loss(traj_grad)
    losses['total'].backward()
    
    grad_norm = traj_grad.grad.norm().item()
    print(f"    Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradients should be non-zero"
    assert not torch.isnan(traj_grad.grad).any(), "Gradients should not be NaN"
    print("    ✓ Gradients flow correctly")
    
    # Test 5: Bone length loss
    print("\n[5] Testing bone length loss...")
    bone_loss = compute_bone_length_loss(smooth_traj)
    print(f"    Bone length loss (smooth): {bone_loss.item():.6f}")
    
    # Stretch one bone over time
    stretched_traj = smooth_traj.clone()
    stretched_traj[:, 1, 0] += torch.linspace(0, 0.5, T, device=device)
    bone_loss_stretched = compute_bone_length_loss(stretched_traj)
    print(f"    Bone length loss (stretched): {bone_loss_stretched.item():.6f}")
    
    assert bone_loss_stretched > bone_loss, \
        "Stretched trajectory should have higher bone length variance"
    print("    ✓ Bone length loss correctly penalizes inconsistent bones")
    
    # Test 6: Edge cases
    print("\n[6] Testing edge cases...")
    
    # Very short trajectory
    short_traj = torch.randn(5, J, 3, device=device)
    short_losses = geo_loss(short_traj)
    print(f"    Short trajectory (T=5): total={short_losses['total'].item():.6f}")
    
    # Single frame (should not crash)
    single_traj = torch.randn(1, J, 3, device=device)
    single_losses = geo_loss(single_traj)
    print(f"    Single frame (T=1): total={single_losses['total'].item():.6f}")
    
    print("\n✓ Geometry losses test completed!")
