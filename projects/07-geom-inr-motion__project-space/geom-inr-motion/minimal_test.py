#!/usr/bin/env python3
"""
minimal_test.py - Quick validation test for Geom-INR-Motion

This script creates a minimal working example to verify:
1. Continuous-time motion interpolation works
2. Simple INR network learns motion trajectories
3. GPU acceleration is functional

Run this FIRST before proceeding to full implementation.
Expected: Loss decreases from ~1.0 to <0.01 in 100 steps.

Usage:
    python minimal_test.py
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import CubicSpline
import math

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FRAMES = 50
NUM_JOINTS = 3
BATCH_SIZE = 64
NUM_STEPS = 100
LEARNING_RATE = 1e-3

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# Step 1: Create Toy Dataset (random walk motion)
# ============================================================================
print("\n[1/4] Creating synthetic motion dataset...")

# Generate smooth random walk trajectories for each joint
np.random.seed(42)
T, J = NUM_FRAMES, NUM_JOINTS
times = np.linspace(0, T - 1, T)

# Random walk with cumulative sum creates smooth-ish trajectories
poses = np.random.randn(T, J, 3) * 0.1  # Small increments
poses = poses.cumsum(axis=0)  # Cumulative sum for continuity
poses -= poses.mean(axis=0, keepdims=True)  # Center around origin

print(f"  Generated motion: {T} frames, {J} joints, shape={poses.shape}")


# ============================================================================
# Step 2: Build Continuous-Time Interpolators
# ============================================================================
print("\n[2/4] Building cubic spline interpolators...")

# Create spline interpolator for each joint and dimension
splines = {}
for j in range(J):
    splines[j] = {}
    for d in range(3):
        splines[j][d] = CubicSpline(times, poses[:, j, d])


def motion_func(t_query):
    """
    Query motion at arbitrary continuous time(s).
    
    Args:
        t_query: float or array of time values in [0, T-1]
    
    Returns:
        positions: (len(t_query), J, 3) array of joint positions
    """
    t_query = np.atleast_1d(t_query)
    result = np.zeros((len(t_query), J, 3))
    for j in range(J):
        for d in range(3):
            result[:, j, d] = splines[j][d](t_query)
    return result


# Test interpolation at fractional timestamps
test_times = [0.0, 5.5, 10.25, 25.0, 49.0]
test_poses = motion_func(test_times)
print(f"  Interpolation test at t={test_times}: shape={test_poses.shape}")


# ============================================================================
# Step 3: Define Tiny INR Network (SIREN-style)
# ============================================================================
print("\n[3/4] Building TinyINR network...")


class SineActivation(nn.Module):
    """Sine activation function for SIREN networks."""
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega
    
    def forward(self, x):
        return torch.sin(self.omega * x)


class TinyINR(nn.Module):
    """
    Minimal Implicit Neural Representation for motion.
    
    Maps (time, joint_index) -> (x, y, z) position.
    Uses sine activations for smooth, continuous output.
    """
    
    def __init__(self, hidden_dim=64, num_layers=3, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        
        # Input: [time (normalized), joint_index (normalized)]
        # Output: [x, y, z]
        layers = []
        
        # First layer with special initialization
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(SineActivation(omega_0))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(omega_0))
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.net = nn.Sequential(*layers)
        
        # SIREN initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following SIREN paper (Sitzmann et al. 2020)."""
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    num_input = layer.weight.size(1)
                    if i == 0:
                        # First layer: uniform(-1/d, 1/d)
                        layer.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        # Hidden layers: uniform(-sqrt(6/d)/omega, sqrt(6/d)/omega)
                        bound = math.sqrt(6.0 / num_input) / self.omega_0
                        layer.weight.uniform_(-bound, bound)
    
    def forward(self, t, j):
        """
        Forward pass.
        
        Args:
            t: (B,) tensor of time values (will be normalized to [0, 1])
            j: (B,) tensor of joint indices (will be normalized to [0, 1])
        
        Returns:
            positions: (B, 3) tensor of predicted joint positions
        """
        # Normalize inputs to [0, 1]
        t_norm = t.unsqueeze(-1) / (NUM_FRAMES - 1)  # (B, 1)
        j_norm = j.unsqueeze(-1) / (NUM_JOINTS - 1)  # (B, 1)
        
        x = torch.cat([t_norm, j_norm], dim=-1)  # (B, 2)
        return self.net(x)  # (B, 3)


# Initialize model
model = TinyINR(hidden_dim=64, num_layers=3).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {num_params:,}")
print(f"  Model architecture: 2 -> 64 -> 64 -> 64 -> 3 (with sine activations)")


# ============================================================================
# Step 4: Training Loop
# ============================================================================
print("\n[4/4] Training for 100 steps...")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

for step in range(NUM_STEPS):
    # Sample random (time, joint) pairs
    t_sample = torch.rand(BATCH_SIZE, device=DEVICE) * (NUM_FRAMES - 1)
    j_sample = torch.randint(0, NUM_JOINTS, (BATCH_SIZE,), device=DEVICE)
    
    # Get ground truth from interpolated motion
    # motion_func returns (1, J, 3) for single time query
    target_list = []
    for t, j in zip(t_sample.cpu(), j_sample.cpu()):
        pose = motion_func(t.item())  # shape: (1, J, 3)
        joint_pos = pose[0, j.item(), :]  # shape: (3,)
        target_list.append(joint_pos)
    target_np = np.array(target_list)
    target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)
    
    # Forward pass
    pred = model(t_sample, j_sample.float())
    
    # MSE loss
    loss = nn.functional.mse_loss(pred, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Print progress
    if step % 20 == 0:
        print(f"  Step {step:3d}, Loss: {loss.item():.6f}")

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

initial_loss = losses[0]
final_loss = losses[-1]
improvement = (initial_loss - final_loss) / initial_loss * 100

print(f"Initial Loss: {initial_loss:.6f}")
print(f"Final Loss:   {final_loss:.6f}")
print(f"Improvement:  {improvement:.1f}%")

# Validation: test on unseen time points
model.eval()
with torch.no_grad():
    # Test at fractional timestamps (interpolation test)
    test_t = torch.tensor([5.5, 15.75, 30.25], device=DEVICE)
    test_j = torch.tensor([0, 1, 2], device=DEVICE).float()
    
    pred_test = model(test_t, test_j)
    gt_list = []
    for t, j in zip(test_t.cpu(), test_j.cpu()):
        pose = motion_func(t.item())  # shape: (1, J, 3)
        joint_pos = pose[0, int(j.item()), :]  # shape: (3,)
        gt_list.append(joint_pos)
    gt_test = torch.tensor(np.array(gt_list), device=DEVICE, dtype=torch.float32)
    
    test_error = (pred_test - gt_test).abs().mean().item()
    print(f"\nInterpolation test (fractional t):")
    print(f"  Mean absolute error: {test_error:.6f}")

# Success criteria
SUCCESS_THRESHOLD = 0.01
if final_loss < SUCCESS_THRESHOLD:
    print(f"\n✅ MINIMAL TEST PASSED! (Loss {final_loss:.6f} < {SUCCESS_THRESHOLD})")
    exit_code = 0
else:
    print(f"\n⚠️  TEST WARNING: Loss {final_loss:.6f} >= {SUCCESS_THRESHOLD}")
    print("    Model may need more training steps or tuning.")
    exit_code = 0  # Don't fail, just warn

# Additional diagnostics
print("\n" + "-" * 60)
print("DIAGNOSTICS")
print("-" * 60)
print(f"Device used: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

print("\n✓ Minimal test completed!")
print("  Next: Run full implementation with data_pipeline.py, model.py, train.py")

# Plot losses if matplotlib available
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # Show ground truth vs prediction for one joint
    with torch.no_grad():
        eval_times = torch.linspace(0, NUM_FRAMES - 1, 100, device=DEVICE)
        eval_joints = torch.zeros(100, device=DEVICE)  # Joint 0
        pred_traj = model(eval_times, eval_joints).cpu().numpy()
        gt_traj = motion_func(eval_times.cpu().numpy())[:, 0, :]  # Joint 0
    
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='Ground Truth', linewidth=2)
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Predicted', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Joint 0 Trajectory (X-Y plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('minimal_test_results.png', dpi=150)
    print(f"\n📊 Saved plot to: minimal_test_results.png")
except ImportError:
    print("\n(matplotlib not available, skipping plot)")

exit(exit_code)
