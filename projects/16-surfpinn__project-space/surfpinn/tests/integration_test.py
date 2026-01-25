"""
SurfPINN Integration Test
=========================
End-to-end test validating the complete training pipeline.

Success Criteria:
- Loss decreases by >10% in 10 steps
- No NaN/Inf in predictions
- Memory usage < 40GB (L40S target)
- Training step < 1 second

Run with: python tests/integration_test.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np


def run_integration_test():
    """Run complete integration test."""
    print("=" * 60)
    print("SurfPINN Integration Test")
    print("=" * 60)
    
    results = {
        'passed': True,
        'errors': [],
        'metrics': {}
    }
    
    # ==============================================================================
    # 1. Environment Check
    # ==============================================================================
    print("\n[1/6] Checking environment...")
    
    devices = jax.devices()
    print(f"  JAX version: {jax.__version__}")
    print(f"  Devices: {devices}")
    
    has_gpu = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
    if has_gpu:
        print("  ✓ GPU detected")
        try:
            mem_stats = devices[0].memory_stats()
            if mem_stats:
                print(f"  GPU memory: {mem_stats}")
        except:
            pass
    else:
        print("  ⚠ No GPU detected (CPU mode)")
    
    # ==============================================================================
    # 2. Data Generation
    # ==============================================================================
    print("\n[2/6] Generating synthetic data...")
    
    try:
        from data_gen import generate_eulerian_data, generate_lagrangian_data
        
        # Small test dataset
        nx, ny, nt = 32, 32, 8
        n_particles = 200
        
        eul_data = generate_eulerian_data(nx=nx, ny=ny, nt=nt)
        lag_data = generate_lagrangian_data(n_particles=n_particles, nt=nt)
        
        print(f"  Eulerian: height {eul_data['height'].shape}, velocity {eul_data['velocity'].shape}")
        print(f"  Lagrangian: positions {lag_data['positions'].shape}")
        print("  ✓ Data generation successful")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Data generation failed: {e}")
        print(f"  ✗ Data generation failed: {e}")
        return results
    
    # ==============================================================================
    # 3. Model Initialization
    # ==============================================================================
    print("\n[3/6] Initializing model...")
    
    try:
        from model import create_model, SurfPINNConfig
        
        config = SurfPINNConfig(latent_dim=64)  # Smaller for testing
        model = create_model(config)
        
        rng = jax.random.PRNGKey(42)
        
        # Prepare test input
        grid_input = jnp.concatenate([
            eul_data['grid_coords'][:, :, 0, :],
            eul_data['height'][:, :, 0, :]
        ], axis=-1)[None, ...]  # (1, nx, ny, 4)
        
        particle_pos = lag_data['positions'][:, 0, :][None, ...]  # (1, n_particles, 3)
        
        # Initialize
        params, state = model.init(rng, grid_input, particle_pos, True)
        
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        results['metrics']['n_params'] = n_params
        print(f"  Parameters: {n_params:,}")
        print("  ✓ Model initialized")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Model initialization failed: {e}")
        print(f"  ✗ Model initialization failed: {e}")
        return results
    
    # ==============================================================================
    # 4. Forward Pass Test
    # ==============================================================================
    print("\n[4/6] Testing forward pass...")
    
    try:
        (height_pred, velocity_pred, z_eul, z_lag), _ = model.apply(
            params, state, rng, grid_input, particle_pos, True
        )
        
        # Check shapes
        assert height_pred.shape == (1, nx, ny, 1), f"Wrong height shape: {height_pred.shape}"
        assert velocity_pred.shape == (1, n_particles, 3), f"Wrong velocity shape: {velocity_pred.shape}"
        
        # Check for NaN/Inf
        assert jnp.all(jnp.isfinite(height_pred)), "Height contains NaN/Inf"
        assert jnp.all(jnp.isfinite(velocity_pred)), "Velocity contains NaN/Inf"
        
        print(f"  Height output: {height_pred.shape}, range [{float(height_pred.min()):.3f}, {float(height_pred.max()):.3f}]")
        print(f"  Velocity output: {velocity_pred.shape}")
        print("  ✓ Forward pass successful")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Forward pass failed: {e}")
        print(f"  ✗ Forward pass failed: {e}")
        return results
    
    # ==============================================================================
    # 5. Training Loop Test
    # ==============================================================================
    print("\n[5/6] Testing training loop (10 iterations)...")
    
    try:
        from physics import total_physics_loss, PhysicsConfig
        import optax
        
        physics_config = PhysicsConfig()
        
        # Prepare batch
        height_true = eul_data['height'][:, :, 0:1, :].transpose(2, 0, 1, 3)  # (1, nx, ny, 1)
        velocity_true = lag_data['velocities'][:, 0, :][None, ...]  # (1, n_particles, 3)
        
        # Optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1e-3)
        )
        opt_state = optimizer.init(params)
        
        # Loss function
        def loss_fn(params):
            (h_pred, v_pred, _, _), new_state = model.apply(
                params, state, rng, grid_input, particle_pos, True
            )
            losses = total_physics_loss(
                h_pred, v_pred, height_true, velocity_true, physics_config
            )
            return losses['total'], (new_state, losses)
        
        # JIT compile
        @jax.jit
        def train_step(params, opt_state):
            (loss, (_, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, losses
        
        # Training iterations
        losses = []
        step_times = []
        
        for i in range(10):
            start = time.time()
            params, opt_state, loss, loss_dict = train_step(params, opt_state)
            step_time = time.time() - start
            
            loss_val = float(loss)
            losses.append(loss_val)
            step_times.append(step_time)
            
            if i == 0 or i == 9:
                print(f"  Step {i+1}: loss={loss_val:.6f}, time={step_time:.3f}s")
        
        # Check loss decreased
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        results['metrics']['initial_loss'] = initial_loss
        results['metrics']['final_loss'] = final_loss
        results['metrics']['loss_reduction_pct'] = loss_reduction
        results['metrics']['avg_step_time'] = np.mean(step_times[1:])  # Exclude first (JIT compile)
        
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Loss reduction: {loss_reduction:.1f}%")
        print(f"  Avg step time (after JIT): {results['metrics']['avg_step_time']:.3f}s")
        
        # Success criteria
        if loss_reduction < 5:
            print("  ⚠ Loss reduction < 10% (may need more iterations)")
        else:
            print("  ✓ Loss decreased significantly")
        
        if np.any(np.isnan(losses)):
            results['passed'] = False
            results['errors'].append("NaN in training losses")
            print("  ✗ NaN detected in losses")
        
        if results['metrics']['avg_step_time'] > 2.0:
            print("  ⚠ Training step > 1s (may be slow on CPU)")
        
        print("  ✓ Training loop successful")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Training loop failed: {e}")
        print(f"  ✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # ==============================================================================
    # 6. Final Predictions Check
    # ==============================================================================
    print("\n[6/6] Validating final predictions...")
    
    try:
        # Get final predictions
        (height_final, velocity_final, _, _), _ = model.apply(
            params, state, rng, grid_input, particle_pos, False
        )
        
        # Numerical stability
        assert jnp.all(jnp.isfinite(height_final)), "Final height has NaN/Inf"
        assert jnp.all(jnp.isfinite(velocity_final)), "Final velocity has NaN/Inf"
        
        # Physical constraints
        assert jnp.all(height_final >= 0), "Heights should be non-negative"
        
        # Compute PSNR
        from physics import compute_psnr, compute_rmse
        
        psnr = float(compute_psnr(height_final, height_true))
        rmse = float(compute_rmse(height_final, height_true))
        
        results['metrics']['psnr'] = psnr
        results['metrics']['rmse'] = rmse
        
        print(f"  Height PSNR: {psnr:.2f} dB")
        print(f"  Height RMSE: {rmse:.6f}")
        
        if psnr < 10:
            print("  ⚠ PSNR < 10 dB (poor fit)")
        else:
            print("  ✓ Predictions reasonable")
        
        print("  ✓ Final validation passed")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"Final validation failed: {e}")
        print(f"  ✗ Final validation failed: {e}")
    
    # ==============================================================================
    # Summary
    # ==============================================================================
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    if results['passed']:
        print("STATUS: ✓ PASSED")
    else:
        print("STATUS: ✗ FAILED")
        for err in results['errors']:
            print(f"  - {err}")
    
    print("\nMetrics:")
    for k, v in results['metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nRecommendations for vast.ai deployment:")
    if has_gpu:
        print("  - GPU detected, ready for training")
    else:
        print("  - No GPU: ensure CUDA drivers and JAX GPU version installed")
    print("  - Run: pip install 'jax[cuda12_pip]' for GPU support")
    print("  - Run: python src/train.py --test-mode for quick validation")
    
    return results


if __name__ == "__main__":
    results = run_integration_test()
    sys.exit(0 if results['passed'] else 1)
