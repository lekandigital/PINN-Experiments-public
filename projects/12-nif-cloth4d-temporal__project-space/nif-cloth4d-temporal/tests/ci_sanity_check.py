#!/usr/bin/env python3
"""
CI Sanity Check for NIF-Cloth4D-Temporal.

Minimal 100-iteration test for CI/CD validation.
Runs in ~2 minutes on CPU, validates core functionality.

Checks:
    1. Model initialization works
    2. Forward pass produces valid outputs
    3. Loss computation doesn't error
    4. Backward pass completes
    5. Optimizer step updates parameters
    6. Loss decreases over 100 steps

Usage:
    python tests/ci_sanity_check.py
    python -m pytest tests/ci_sanity_check.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from src.models import FourierFeatureMLP
from src.losses import PhysicsLossStack
from src.utils import set_seed


def test_model_initialization():
    """Test that model initializes correctly."""
    print("\n[1/6] Testing model initialization...")
    
    model = FourierFeatureMLP(
        in_dim=4,
        hidden_dim=64,
        out_dim=1,
        num_layers=3,
        num_freqs=8,
        use_gru=False,
    )
    
    assert model is not None, "Model is None"
    assert model.count_parameters() > 0, "Model has no parameters"
    
    # Test with GRU
    model_gru = FourierFeatureMLP(
        in_dim=4,
        hidden_dim=64,
        out_dim=1,
        num_layers=3,
        num_freqs=8,
        use_gru=True,
        gru_hidden=32,
    )
    
    assert model_gru.use_gru, "GRU not enabled"
    assert model_gru.temporal_gru is not None, "GRU module is None"
    
    print(f"  ✓ Model initialized with {model.count_parameters():,} parameters")
    print(f"  ✓ GRU model initialized with {model_gru.count_parameters():,} parameters")
    return True


def test_forward_pass():
    """Test that forward pass produces valid outputs."""
    print("\n[2/6] Testing forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FourierFeatureMLP(
        hidden_dim=64,
        num_layers=3,
        use_gru=True,
        gru_hidden=32,
    ).to(device)
    model.eval()
    
    # Test input
    batch_size = 16
    xyz = torch.rand(batch_size, 3, device=device)
    t = torch.rand(batch_size, 1, device=device)
    
    # Forward pass
    with torch.no_grad():
        sdf, hidden = model(xyz, t, hidden_state=None)
    
    # Check output shape
    assert sdf.shape == (batch_size, 1), f"Wrong output shape: {sdf.shape}"
    
    # Check hidden state shape (if using GRU)
    if model.use_gru:
        assert hidden is not None, "Hidden state is None with GRU enabled"
        assert hidden.shape[1] == batch_size, f"Wrong hidden batch size: {hidden.shape}"
    
    # Check for NaN/Inf
    assert not torch.isnan(sdf).any(), "Output contains NaN"
    assert not torch.isinf(sdf).any(), "Output contains Inf"
    
    print(f"  ✓ Output shape: {sdf.shape}")
    print(f"  ✓ Output range: [{sdf.min().item():.4f}, {sdf.max().item():.4f}]")
    print(f"  ✓ Hidden state shape: {hidden.shape if hidden is not None else 'N/A'}")
    return True


def test_loss_computation():
    """Test that loss functions compute without errors."""
    print("\n[3/6] Testing loss computation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create physics loss stack
    loss_stack = PhysicsLossStack(
        lambda_stretch=1.0,
        lambda_bend=0.1,
        lambda_momentum=0.1,
        lambda_collision=10.0,
    ).to(device)
    
    # Create dummy mesh data
    batch_size = 4
    num_vertices = 16  # 4x4 grid
    
    pred_vertices = torch.rand(batch_size, num_vertices, 3, device=device)
    target_vertices = torch.rand(batch_size, num_vertices, 3, device=device)
    
    # Basic MSE loss
    mse_loss = F.mse_loss(pred_vertices, target_vertices)
    assert not torch.isnan(mse_loss), "MSE loss is NaN"
    
    # Collision loss (ground plane)
    from src.losses.physics_losses import compute_collision_loss
    collision_loss = compute_collision_loss(pred_vertices, ground_height=0.0)
    assert not torch.isnan(collision_loss), "Collision loss is NaN"
    
    print(f"  ✓ MSE loss: {mse_loss.item():.6f}")
    print(f"  ✓ Collision loss: {collision_loss.item():.6f}")
    return True


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("\n[4/6] Testing backward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FourierFeatureMLP(
        hidden_dim=64,
        num_layers=3,
        use_gru=False,
    ).to(device)
    model.train()
    
    # Forward pass
    xyz = torch.rand(16, 3, device=device)
    t = torch.rand(16, 1, device=device)
    target = torch.zeros(16, 1, device=device)
    
    sdf, _ = model(xyz, t)
    loss = F.mse_loss(sdf, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    assert has_gradients, "No gradients computed"
    
    print(f"  ✓ Loss: {loss.item():.6f}")
    print(f"  ✓ Gradients computed successfully")
    return True


def test_optimizer_step():
    """Test that optimizer updates parameters."""
    print("\n[5/6] Testing optimizer step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FourierFeatureMLP(
        hidden_dim=64,
        num_layers=3,
    ).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Store initial parameters
    initial_params = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }
    
    # Training step
    xyz = torch.rand(16, 3, device=device)
    t = torch.rand(16, 1, device=device)
    target = torch.zeros(16, 1, device=device)
    
    optimizer.zero_grad()
    sdf, _ = model(xyz, t)
    loss = F.mse_loss(sdf, target)
    loss.backward()
    optimizer.step()
    
    # Check parameters changed
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            params_changed = True
            break
    
    assert params_changed, "Parameters did not change after optimizer step"
    
    print(f"  ✓ Parameters updated successfully")
    return True


def test_100_step_training():
    """
    Minimal 100-iteration training test.
    
    This is the main CI sanity check - validates that:
    1. Training loop runs without errors
    2. Loss decreases over 100 steps
    3. Model learns something meaningful
    """
    print("\n[6/6] Running 100-step training test...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    
    # Small model for fast testing
    model = FourierFeatureMLP(
        hidden_dim=64,
        num_layers=3,
        num_freqs=8,
        use_gru=True,
        gru_hidden=32,
    ).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if use_amp else None
    
    losses = []
    
    for step in range(100):
        # Random input
        xyz = torch.rand(16, 3, device=device)
        t = torch.rand(16, 1, device=device)
        
        # Target: SDF should be small near surface (z~0 plane for simplicity)
        target = xyz[:, 1:2]  # Distance from y=0 plane
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                pred, _ = model(xyz, t)
                loss = F.mse_loss(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, _ = model(xyz, t)
            loss = F.mse_loss(pred, target)
            
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 25 == 0:
            print(f"    Step {step + 1}/100: loss = {loss.item():.6f}")
    
    # Verify loss decreased
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10
    
    print(f"\n  Initial loss (avg first 10): {initial_loss:.6f}")
    print(f"  Final loss (avg last 10):    {final_loss:.6f}")
    print(f"  Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    assert final_loss < initial_loss, \
        f"Loss did not decrease! Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
    
    print(f"\n  ✓ CI test passed: loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
    return True


def run_all_tests():
    """Run all CI sanity checks."""
    print("=" * 60)
    print("NIF-Cloth4D-Temporal CI Sanity Check")
    print("=" * 60)
    
    set_seed(42)
    
    tests = [
        test_model_initialization,
        test_forward_pass,
        test_loss_computation,
        test_backward_pass,
        test_optimizer_step,
        test_100_step_training,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if failed > 0:
        print("❌ CI CHECK FAILED")
        return False
    else:
        print("✅ CI CHECK PASSED")
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
