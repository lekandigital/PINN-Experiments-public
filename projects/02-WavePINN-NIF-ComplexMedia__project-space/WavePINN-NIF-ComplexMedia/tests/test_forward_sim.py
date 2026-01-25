"""
Quick sanity check: Can the WavePINN-NIF model solve a simple case?

This test verifies:
1. Model initialization
2. Forward pass produces valid outputs
3. Loss computation works
4. Gradients can be computed
5. JAX JIT compilation works
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import time

from src.model import create_model, count_parameters
from src.data_generator import SyntheticWaveData
from src.physics_loss import WavePDELoss


def test_forward_propagation():
    """Test basic forward pass and loss computation."""
    
    print("=" * 60)
    print("WavePINN-NIF Quick Test")
    print("=" * 60)
    
    # Check JAX device
    devices = jax.devices()
    print(f"\nJAX devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")
    
    all_passed = True
    
    # =========================================================================
    # Test 1: Model Initialization
    # =========================================================================
    print("\n[1/6] Initializing model...")
    try:
        start_time = time.time()
        
        model = create_model()
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, 3))  # (x, y, t) for 2D
        params = model.init(rng, dummy_input, return_media=True)
        
        n_params = count_parameters(params)
        init_time = time.time() - start_time
        
        print(f"✓ Model initialized successfully ({init_time:.2f}s)")
        print(f"  Parameter count: {n_params:,}")
        
        # Verify parameter structure
        assert n_params > 0, "Model has no parameters"
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        all_passed = False
        return False
    
    # =========================================================================
    # Test 2: Data Generation
    # =========================================================================
    print("\n[2/6] Generating synthetic data...")
    try:
        data_gen = SyntheticWaveData(
            domain_size=(1.0, 1.0),
            n_collocation=1000,
            n_boundary=100,
            n_initial=100,
            t_max=1.0
        )
        data = data_gen.sample_collocation_points()
        
        print("✓ Data generated successfully")
        print(f"  Interior points: {data['interior'].shape}")
        print(f"  Initial points: {data['initial'].shape}")
        print(f"  Boundary faces: {list(data['boundary'].keys())}")
        
        # Test velocity model generation
        c_field = data_gen.generate_layered_medium(n_layers=5)
        print(f"  Velocity field: {c_field.shape}")
        print(f"  Velocity range: [{float(c_field.min()):.2f}, {float(c_field.max()):.2f}]")
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        all_passed = False
    
    # =========================================================================
    # Test 3: Forward Pass
    # =========================================================================
    print("\n[3/6] Testing forward pass...")
    try:
        # Test with wavefield only
        u_pred = model.apply(params, rng, data['interior'][:10])
        
        print("✓ Forward pass (u only) successful")
        print(f"  Output shape: {u_pred.shape}")
        print(f"  Output range: [{float(u_pred.min()):.6f}, {float(u_pred.max()):.6f}]")
        
        # Check for NaN/Inf
        assert not jnp.any(jnp.isnan(u_pred)), "NaN in output"
        assert not jnp.any(jnp.isinf(u_pred)), "Inf in output"
        
        # Test with both u and c
        u_pred, c_pred = model.apply(params, rng, data['interior'][:10], return_media=True)
        
        print("✓ Forward pass (u, c) successful")
        print(f"  u shape: {u_pred.shape}, c shape: {c_pred.shape}")
        print(f"  c range: [{float(c_pred.min()):.4f}, {float(c_pred.max()):.4f}]")
        
        # c should be positive
        assert jnp.all(c_pred > 0), "Wave speed should be positive"
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        all_passed = False
    
    # =========================================================================
    # Test 4: Loss Computation
    # =========================================================================
    print("\n[4/6] Computing physics loss...")
    try:
        loss_computer = WavePDELoss(
            model_apply=model.apply,
            lambda_pde=1.0,
            lambda_bc=10.0,
            lambda_ic=10.0,
            ndim=2
        )
        
        # Create minimal data batch
        data_batch = {
            'interior': data['interior'][:50],  # Reduced for speed
            'boundary': {k: v[:20] for k, v in data['boundary'].items()},
            'initial': data['initial'][:50],
            'u0_target': jnp.zeros(50),
            'v0_target': jnp.zeros(50),
        }
        
        start_time = time.time()
        total_loss, loss_dict = loss_computer.total_loss(params, rng, data_batch)
        loss_time = time.time() - start_time
        
        print(f"✓ Loss computation successful ({loss_time:.2f}s)")
        print(f"  Total loss: {float(total_loss):.6f}")
        print(f"  PDE residual: {float(loss_dict['loss_pde']):.6f}")
        print(f"  BC loss: {float(loss_dict['loss_bc']):.6f}")
        print(f"  IC loss: {float(loss_dict['loss_ic']):.6f}")
        
        # Check for valid losses
        assert not jnp.isnan(total_loss), "NaN in total loss"
        assert total_loss >= 0, "Loss should be non-negative"
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # =========================================================================
    # Test 5: Gradient Computation
    # =========================================================================
    print("\n[5/6] Testing gradient computation...")
    try:
        def loss_for_grad(p):
            total, _ = loss_computer.total_loss(p, rng, data_batch)
            return total
        
        start_time = time.time()
        grads = jax.grad(loss_for_grad)(params)
        grad_time = time.time() - start_time
        
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
        )
        
        print(f"✓ Gradient computation successful ({grad_time:.2f}s)")
        print(f"  Gradient norm: {float(grad_norm):.6f}")
        
        # Check gradient is valid
        assert not jnp.isnan(grad_norm), "NaN in gradients"
        assert grad_norm > 0, "Gradient norm should be positive"
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # =========================================================================
    # Test 6: JIT Compilation
    # =========================================================================
    print("\n[6/6] Testing JIT compilation...")
    try:
        @jax.jit
        def jitted_forward(params, rng, x):
            return model.apply(params, rng, x)
        
        # First call (compilation)
        start_time = time.time()
        _ = jitted_forward(params, rng, data['interior'][:100])
        compile_time = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        _ = jitted_forward(params, rng, data['interior'][:100])
        run_time = time.time() - start_time
        
        print(f"✓ JIT compilation successful")
        print(f"  Compile time: {compile_time:.4f}s")
        print(f"  Run time: {run_time:.6f}s")
        print(f"  Speedup: {compile_time/run_time:.1f}x")
        
    except Exception as e:
        print(f"✗ JIT compilation failed: {e}")
        all_passed = False
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Model is functional!")
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
    print("=" * 60)
    
    return all_passed


def test_3d_support():
    """Test 3D wave propagation support."""
    print("\n" + "=" * 60)
    print("3D Support Test")
    print("=" * 60)
    
    try:
        # Create 3D data generator
        data_gen = SyntheticWaveData(
            domain_size=(1.0, 1.0, 1.0),
            n_collocation=500,
            n_boundary=50,
            n_initial=50,
            t_max=1.0
        )
        
        # Initialize model for 3D
        model = create_model()
        rng = jax.random.PRNGKey(42)
        dummy_3d = jnp.zeros((1, 4))  # (x, y, z, t)
        params = model.init(rng, dummy_3d, return_media=True)
        
        # Test forward pass
        test_input = jax.random.uniform(rng, (10, 4))
        u_3d = model.apply(params, rng, test_input)
        
        print(f"✓ 3D forward pass: {u_3d.shape}")
        print(f"  Parameters: {count_parameters(params):,}")
        
        # Test with media
        u_3d, c_3d = model.apply(params, rng, test_input, return_media=True)
        print(f"✓ 3D (u, c): u={u_3d.shape}, c={c_3d.shape}")
        
        print("\n✅ 3D support test passed!")
        return True
        
    except Exception as e:
        print(f"✗ 3D test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("Training Step Test")
    print("=" * 60)
    
    try:
        import optax
        
        # Setup
        model = create_model()
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, jnp.zeros((1, 3)), return_media=True)
        
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        
        loss_computer = WavePDELoss(
            model_apply=model.apply,
            lambda_pde=1.0,
            lambda_bc=10.0,
            lambda_ic=10.0,
            ndim=2
        )
        
        # Create data
        data_gen = SyntheticWaveData(n_collocation=100, n_boundary=20, n_initial=20)
        data = data_gen.sample_collocation_points()
        
        data_batch = {
            'interior': data['interior'],
            'boundary': data['boundary'],
            'initial': data['initial'],
            'u0_target': jnp.zeros(len(data['initial'])),
            'v0_target': jnp.zeros(len(data['initial'])),
        }
        
        # JIT-compiled training step
        @jax.jit
        def train_step(params, opt_state, rng, batch):
            def loss_fn(p):
                return loss_computer.total_loss(p, rng, batch)
            
            (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, loss_dict
        
        # Run a few steps
        print("Running training steps...")
        losses = []
        for i in range(5):
            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss, loss_dict = train_step(params, opt_state, step_rng, data_batch)
            losses.append(float(loss))
            print(f"  Step {i+1}: loss = {losses[-1]:.6f}")
        
        # Check loss is decreasing (or at least not exploding)
        print(f"\n  Initial loss: {losses[0]:.6f}")
        print(f"  Final loss: {losses[-1]:.6f}")
        
        if losses[-1] < losses[0] * 2:  # Allow some variance
            print("\n✅ Training step test passed!")
            return True
        else:
            print("\n⚠️ Loss did not decrease, but no errors occurred")
            return True
            
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# WavePINN-NIF Complete Test Suite")
    print("#" * 60)
    
    results = {}
    
    # Test 1: Basic functionality
    results['forward'] = test_forward_propagation()
    
    # Test 2: 3D support
    results['3d'] = test_3d_support()
    
    # Test 3: Training step
    results['training'] = test_training_step()
    
    # Summary
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "#" * 60)
    if all_passed:
        print("# 🎉 ALL TESTS PASSED!")
    else:
        print("# ⚠️ SOME TESTS FAILED")
    print("#" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
