#!/usr/bin/env python3
"""
Quick Validation Test Suite for Maxwell-PINN-NIF

This script runs a comprehensive set of tests to validate that the
Maxwell-PINN-NIF implementation is working correctly. Tests include:

1. GPU/Device Check - Verify CUDA availability and GPU specs
2. Model Initialization - Create model and count parameters
3. Forward Pass - Test inference on random coordinates
4. Divergence Loss - Verify ∇·(εE) and ∇·(μH) computation
5. Curl Residual - Test Maxwell equation PDE residual
6. PML Loss - Verify absorbing boundary condition module
7. Backward Pass - Ensure gradients flow through entire model
8. Memory Usage - Check GPU memory consumption
9. Training Step - Run single optimization step
10. Convergence Test - Short training to verify loss decreases

Usage:
    python experiments/quick_test.py
    
All tests should pass for a correctly configured environment.
"""

import sys
import time
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with status indicator."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


class TestResults:
    """Collect and summarize test results."""
    
    def __init__(self):
        self.tests = []
    
    def add(self, name: str, passed: bool, details: str = ""):
        self.tests.append({'name': name, 'passed': passed, 'details': details})
        print_result(name, passed, details)
    
    def summary(self):
        passed = sum(1 for t in self.tests if t['passed'])
        total = len(self.tests)
        print_header("TEST SUMMARY")
        print(f"  Passed: {passed}/{total}")
        if passed == total:
            print("\n  🎉 ALL TESTS PASSED!")
        else:
            print("\n  ⚠️  Some tests failed:")
            for t in self.tests:
                if not t['passed']:
                    print(f"     - {t['name']}: {t['details']}")
        return passed == total


def main():
    results = TestResults()
    
    print_header("Maxwell-PINN-NIF Quick Test Suite")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    # ========================================================================
    # Test 1: GPU/Device Check
    # ========================================================================
    print_header("1. GPU/Device Check")
    
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Device: {gpu_name}")
            print(f"  Memory: {gpu_memory:.2f} GB")
            results.add("GPU Detection", True, f"{gpu_name}, {gpu_memory:.1f}GB")
        else:
            device = torch.device('cpu')
            print("  Running on CPU (no CUDA)")
            results.add("GPU Detection", True, "Running on CPU")
    except Exception as e:
        device = torch.device('cpu')
        results.add("GPU Detection", False, str(e))
    
    # ========================================================================
    # Test 2: Model Initialization
    # ========================================================================
    print_header("2. Model Initialization")
    
    try:
        from pinn_model import MaxwellPINN
        
        model = MaxwellPINN(
            input_dim=3,
            hidden_dim=128,
            num_hidden=6,
            use_fourier=False
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Architecture: 3 → 128×6 → (3, 3)")
        print(f"  Parameters: {n_params:,}")
        
        # Check parameter count is reasonable (should be ~100k-500k)
        param_ok = 50000 < n_params < 5000000
        results.add("Model Creation", param_ok, f"{n_params:,} parameters")
        
    except Exception as e:
        results.add("Model Creation", False, str(e))
        return False
    
    # ========================================================================
    # Test 3: Forward Pass
    # ========================================================================
    print_header("3. Forward Pass Test")
    
    try:
        batch_size = 1000
        coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
        
        start_time = time.time()
        E_pred, H_pred = model(coords)
        forward_time = (time.time() - start_time) * 1000  # ms
        
        print(f"  Input:  coords {list(coords.shape)}")
        print(f"  Output: E {list(E_pred.shape)}, H {list(H_pred.shape)}")
        print(f"  Time:   {forward_time:.2f} ms")
        
        shape_ok = E_pred.shape == (batch_size, 3) and H_pred.shape == (batch_size, 3)
        time_ok = forward_time < 1000  # Should be <1 second
        
        results.add("Forward Pass Shape", shape_ok, f"E: {list(E_pred.shape)}, H: {list(H_pred.shape)}")
        results.add("Forward Pass Speed", time_ok, f"{forward_time:.2f} ms for {batch_size} points")
        
    except Exception as e:
        results.add("Forward Pass", False, str(e))
    
    # ========================================================================
    # Test 4: Divergence-Free Loss
    # ========================================================================
    print_header("4. Divergence-Free Constraint Test")
    
    try:
        from pinn_model import divergence_free_loss
        
        eps = torch.ones(batch_size, device=device) * 2.0
        mu = torch.ones(batch_size, device=device) * 1.0
        
        loss_div = divergence_free_loss(E_pred, H_pred, coords, eps, mu)
        
        print(f"  ε values: constant 2.0")
        print(f"  μ values: constant 1.0")
        print(f"  Divergence loss: {loss_div.item():.6e}")
        
        # Loss should be computed (not nan or inf)
        loss_valid = torch.isfinite(loss_div) and loss_div.item() >= 0
        results.add("Divergence Loss Computation", loss_valid, f"L_div = {loss_div.item():.4e}")
        
    except Exception as e:
        results.add("Divergence Loss", False, str(e))
    
    # ========================================================================
    # Test 5: Maxwell Curl Residual
    # ========================================================================
    print_header("5. Maxwell Curl Equation Residual Test")
    
    try:
        from pinn_model import maxwell_curl_residual
        
        # Need fresh forward pass for this test
        coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
        E_pred, H_pred = model(coords)
        
        loss_curl = maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, omega=1.0)
        
        print(f"  ω (angular frequency): 1.0")
        print(f"  Curl residual loss: {loss_curl.item():.6e}")
        
        loss_valid = torch.isfinite(loss_curl) and loss_curl.item() >= 0
        results.add("Curl Residual Computation", loss_valid, f"L_curl = {loss_curl.item():.4e}")
        
    except Exception as e:
        results.add("Curl Residual", False, str(e))
    
    # ========================================================================
    # Test 6: PML Loss Module
    # ========================================================================
    print_header("6. PML (Absorbing Boundary) Loss Test")
    
    try:
        from pml_loss import PMLLoss
        
        domain_bounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        pml_thickness = 0.1
        
        pml = PMLLoss(domain_bounds, pml_thickness, sigma_max=1.0, order=2).to(device)
        
        # Fresh forward pass
        coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
        E_pred, H_pred = model(coords)
        
        loss_pml = pml(coords, E_pred, H_pred)
        
        # Check how many points are in PML region
        pml_mask = pml.pml_region.get_pml_mask(coords)
        n_pml = pml_mask.sum().item()
        
        print(f"  Domain: [0,1]³")
        print(f"  PML thickness: {pml_thickness}")
        print(f"  Points in PML: {n_pml}/{batch_size}")
        print(f"  PML loss: {loss_pml.item():.6e}")
        
        loss_valid = torch.isfinite(loss_pml)
        results.add("PML Loss Computation", loss_valid.item(), f"L_pml = {loss_pml.item():.4e}")
        
    except Exception as e:
        results.add("PML Loss", False, str(e))
    
    # ========================================================================
    # Test 7: Backward Pass
    # ========================================================================
    print_header("7. Backward Pass (Gradient Flow) Test")
    
    try:
        # Combine all losses
        total_loss = loss_div + loss_curl + loss_pml
        
        # Clear existing gradients
        model.zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist for all parameters
        grads_exist = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        grads_nonzero = any(p.grad.abs().sum() > 0 for p in model.parameters() if p.grad is not None)
        grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        
        print(f"  Total loss: {total_loss.item():.6e}")
        print(f"  Gradients exist: {grads_exist}")
        print(f"  Gradients non-zero: {grads_nonzero}")
        print(f"  Gradients finite: {grads_finite}")
        
        results.add("Gradient Computation", grads_exist and grads_finite, 
                   f"All params have {'valid' if grads_finite else 'invalid'} gradients")
        
    except Exception as e:
        results.add("Backward Pass", False, str(e))
    
    # ========================================================================
    # Test 8: Memory Usage
    # ========================================================================
    print_header("8. GPU Memory Usage Test")
    
    try:
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            max_memory = torch.cuda.max_memory_allocated(0) / 1e9
            
            print(f"  Allocated: {allocated:.3f} GB")
            print(f"  Reserved:  {reserved:.3f} GB")
            print(f"  Peak:      {max_memory:.3f} GB")
            
            # Memory should be reasonable (<10GB for 1000 points)
            memory_ok = max_memory < 10.0
            results.add("Memory Usage", memory_ok, f"Peak: {max_memory:.2f} GB")
        else:
            print("  Skipped (CPU mode)")
            results.add("Memory Usage", True, "Skipped (CPU)")
            
    except Exception as e:
        results.add("Memory Usage", False, str(e))
    
    # ========================================================================
    # Test 9: Single Training Step
    # ========================================================================
    print_header("9. Single Training Step Test")
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Fresh forward-backward pass
        coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
        eps = torch.ones(batch_size, device=device) * 2.0
        mu = torch.ones(batch_size, device=device)
        
        optimizer.zero_grad()
        
        E_pred, H_pred = model(coords)
        loss = (divergence_free_loss(E_pred, H_pred, coords, eps, mu) +
                maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, omega=1.0))
        
        loss_before = loss.item()
        
        loss.backward()
        optimizer.step()
        
        # Check loss after step
        E_pred, H_pred = model(coords)
        loss_after = (divergence_free_loss(E_pred, H_pred, coords, eps, mu).item() +
                     maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, omega=1.0).item())
        
        print(f"  Loss before: {loss_before:.6e}")
        print(f"  Loss after:  {loss_after:.6e}")
        
        # Loss might not decrease in single step, but should be computed
        step_ok = torch.isfinite(torch.tensor(loss_after))
        results.add("Training Step Execution", step_ok, 
                   f"{loss_before:.2e} → {loss_after:.2e}")
        
    except Exception as e:
        results.add("Training Step", False, str(e))
    
    # ========================================================================
    # Test 10: Short Training Convergence
    # ========================================================================
    print_header("10. Short Training Convergence Test")
    
    try:
        # Create fresh model for this test
        model_test = MaxwellPINN(
            input_dim=3, hidden_dim=64, num_hidden=4
        ).to(device)
        
        optimizer = torch.optim.Adam(model_test.parameters(), lr=1e-3)
        
        n_iters = 50
        losses = []
        
        print(f"  Running {n_iters} training iterations...")
        
        for i in range(n_iters):
            coords = torch.rand(500, 3, device=device, requires_grad=True)
            eps = torch.ones(500, device=device) * 2.0
            mu = torch.ones(500, device=device)
            
            optimizer.zero_grad()
            E_pred, H_pred = model_test(coords)
            
            loss = (divergence_free_loss(E_pred, H_pred, coords, eps, mu) +
                   0.1 * maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, omega=1.0))
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"  Initial loss (avg first 5):  {initial_loss:.4e}")
        print(f"  Final loss (avg last 5):     {final_loss:.4e}")
        print(f"  Reduction: {reduction:.1f}%")
        
        # Loss should decrease by at least 10% over 50 iterations
        converging = final_loss < initial_loss * 0.9 or reduction > 10
        results.add("Training Convergence", converging, 
                   f"Reduced by {reduction:.1f}%")
        
    except Exception as e:
        results.add("Training Convergence", False, str(e))
    
    # ========================================================================
    # Summary
    # ========================================================================
    all_passed = results.summary()
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
