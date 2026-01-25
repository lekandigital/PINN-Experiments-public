"""
Sphere PDE Integration Test

Tests the complete GeoPINN pipeline by solving a simple PDE on the unit sphere:

    Δ_S u = f

where:
    u_true(x, y, z) = x * y  (spherical harmonic l=2 mode)
    f(x, y, z) = -6 * x * y  (known from Δ_S(xy) = -6xy for l=2)

The eigenvalue for spherical harmonics is Δ_S Y_l^m = -l(l+1) Y_l^m.
For l=2: Δ_S(xy) = -2(2+1)xy = -6xy

This test validates:
1. Laplace-Beltrami computation via autograd
2. PINN training convergence
3. L2 error vs analytic solution

Success criteria:
- Training completes without errors
- Loss decreases over epochs
- Final L2 error < 0.1
"""

import sys
import os
import argparse
import time
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopinn.training.sphere_trainer import (
    SphereMLP,
    SpherePINNTrainer,
    compute_laplace_beltrami_sphere,
    sample_sphere_torch
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def true_solution(points: torch.Tensor) -> torch.Tensor:
    """
    Analytic solution: u(x, y, z) = x * y
    
    This is a spherical harmonic (l=2 mode).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (x * y).unsqueeze(-1)


def source_term(points: torch.Tensor) -> torch.Tensor:
    """
    Source term: f(x, y, z) = -6 * x * y
    
    This satisfies Δ_S(xy) = -6xy on the unit sphere.
    (eigenvalue for l=2 spherical harmonic: -l(l+1) = -6)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (-6 * x * y).unsqueeze(-1)


def verify_laplace_beltrami():
    """
    Verify Laplace-Beltrami computation on known solution.
    
    For u = xy on unit sphere (l=2 spherical harmonic):
        Δ_S u = -6xy
    """
    print("\n" + "="*50)
    print("Verifying Laplace-Beltrami Implementation")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Sample points
    n_test = 100
    points = sample_sphere_torch(n_test, device).requires_grad_(True)
    
    # Create model that outputs exactly xy
    class ExactSolution(torch.nn.Module):
        def forward(self, x):
            return (x[:, 0] * x[:, 1]).unsqueeze(-1)
    
    model = ExactSolution()
    
    # Compute Laplace-Beltrami (need create_graph=True for second derivatives)
    laplacian = compute_laplace_beltrami_sphere(model, points, create_graph=True)
    
    # Expected: -6xy (eigenvalue for l=2 spherical harmonic)
    expected = -6 * points[:, 0] * points[:, 1]
    
    # Compare
    error = torch.abs(laplacian.squeeze() - expected)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    print(f"Points tested: {n_test}")
    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    
    if max_error < 0.01:
        print("✓ Laplace-Beltrami verification PASSED")
        return True
    else:
        print("✗ Laplace-Beltrami verification FAILED")
        return False


def run_training_test(use_amp: bool = False, n_epochs: int = 100):
    """
    Run the main PINN training test.
    """
    print("\n" + "="*50)
    print("Sphere PDE Training Test")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = SphereMLP(
        in_dim=3,
        out_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation='tanh'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = SpherePINNTrainer(
        model=model,
        source_fn=source_term,
        device=device,
        lr=1e-3,
        use_amp=use_amp
    )
    
    # Training parameters
    n_points = 1000
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'test_sphere_model.pth')
    
    print(f"Collocation points: {n_points}")
    print(f"Epochs: {n_epochs}")
    print(f"AMP: {use_amp}")
    print("")
    
    # Train
    start_time = time.time()
    
    history = trainer.train(
        n_points=n_points,
        n_epochs=n_epochs,
        log_every=20,
        true_solution_fn=true_solution,
        checkpoint_path=checkpoint_path,
        checkpoint_every=50
    )
    
    total_time = time.time() - start_time
    
    # Results
    print("\n" + "-"*50)
    print("RESULTS")
    print("-"*50)
    
    initial_loss = history['loss'][0]
    final_loss = history['loss'][-1]
    final_l2_error = history['l2_error'][-1] if history['l2_error'] else None
    
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Time per epoch: {total_time/n_epochs*1000:.2f} ms")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    if final_l2_error is not None:
        print(f"Final L2 error: {final_l2_error:.6f}")
    
    # GPU stats
    if device == 'cuda':
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1e6:.1f} MB")
    
    # Verify success criteria
    print("\n" + "="*50)
    print("SUCCESS CRITERIA CHECK")
    print("="*50)
    
    success = True
    
    # Check 1: Loss decreased
    if final_loss < initial_loss:
        print("✓ Loss decreased during training")
    else:
        print("✗ Loss did not decrease")
        success = False
    
    # Check 2: L2 error threshold
    if final_l2_error is not None and final_l2_error < 0.1:
        print(f"✓ L2 error ({final_l2_error:.4f}) < 0.1 threshold")
    elif final_l2_error is not None:
        print(f"✗ L2 error ({final_l2_error:.4f}) >= 0.1 threshold")
        success = False
    
    # Check 3: Training completed
    print("✓ Training completed without errors")
    
    # Check 4: Checkpoint saved
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint saved to {checkpoint_path}")
    else:
        print("✗ Checkpoint not saved")
        success = False
    
    if success:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
    
    return success, history


def test_integration():
    """
    Run complete integration test.
    """
    print("\n" + "#"*60)
    print("# GeoPINN-Manifold Integration Test")
    print("# Sphere Laplace-Beltrami PDE")
    print("#"*60)
    
    # Verify implementation
    lb_ok = verify_laplace_beltrami()
    
    if not lb_ok:
        print("\nAborting: Laplace-Beltrami implementation incorrect")
        return False
    
    # Run training
    success, history = run_training_test(use_amp=False, n_epochs=100)
    
    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sphere PDE Test')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Run tests
    if args.amp or args.epochs != 100:
        print(f"Running with AMP={args.amp}, epochs={args.epochs}")
        success, _ = run_training_test(use_amp=args.amp, n_epochs=args.epochs)
    else:
        success = test_integration()
    
    # Exit code
    sys.exit(0 if success else 1)
