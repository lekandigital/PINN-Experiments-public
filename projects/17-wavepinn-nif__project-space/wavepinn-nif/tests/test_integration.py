"""
End-to-end integration tests for WavePINN-NIF-Scalar.

These tests verify the complete pipeline from data generation
through training to prediction.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from src.data_gen import generate_training_dataset, sample_collocation_points_2d
from src.model import WavePINN, WavePINNConfig, create_simple_pinn, predict_on_grid
from src.training import (
    TrainingConfig, Trainer, init_train_state,
    create_optimizer, make_train_step
)


class TestEndToEndForward:
    """End-to-end tests for forward problem."""
    
    def test_complete_forward_pipeline(self):
        """
        Test complete forward solve pipeline:
        1. Generate synthetic data
        2. Initialize model
        3. Train for a few epochs
        4. Verify loss decreases
        """
        print("\n=== Testing Complete Forward Pipeline ===")
        
        # 1. Generate synthetic data
        print("Generating training data...")
        data = generate_training_dataset(
            seed=42,
            nx=50,
            nz=50,
            n_interior=1000,
            n_boundary=200,
            n_initial=200
        )
        
        assert 'slowness' in data
        assert 'collocation' in data
        print(f"  Slowness shape: {data['slowness'].shape}")
        print(f"  Interior points: {data['collocation']['interior'].shape}")
        
        # 2. Initialize model
        print("Initializing model...")
        config = WavePINNConfig(
            hidden_dims=[64, 64, 32],
            use_fourier_features=True,
            num_fourier_features=32
        )
        pinn = WavePINN(config, seed=42)
        
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"  Model parameters: {n_params:,}")
        
        # Prepare batch
        batch = {
            'interior': data['collocation']['interior'],
            'boundary': data['collocation']['boundary'],
            'initial': data['collocation']['initial'],
            'velocity': data['velocity_at_interior']
        }
        
        # 3. Initial loss
        initial_loss, initial_components = pinn.total_loss(
            params, batch, return_components=True
        )
        print(f"  Initial loss: {initial_loss:.4e}")
        print(f"    PDE: {initial_components['pde']:.4e}")
        print(f"    BC: {initial_components['bc']:.4e}")
        print(f"    IC: {initial_components['ic_u']:.4e}")
        
        # 4. Training loop
        print("Training for 50 epochs...")
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        
        @jax.jit
        def train_step(params, opt_state, batch):
            loss_fn = lambda p: pinn.total_loss(p, batch)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        losses = []
        for epoch in range(50):
            params, opt_state, loss = train_step(params, opt_state, batch)
            losses.append(float(loss))
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.4e}")
        
        final_loss = losses[-1]
        print(f"  Final loss: {final_loss:.4e}")
        
        # 5. Assertions
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4e} -> {final_loss:.4e}"
        
        loss_reduction = (initial_loss - final_loss) / initial_loss
        print(f"  Loss reduction: {loss_reduction * 100:.1f}%")
        
        # Check loss decreased by at least 20%
        assert loss_reduction > 0.2, \
            f"Loss reduction too small: {loss_reduction * 100:.1f}%"
        
        print("✓ Forward pipeline test passed!")
    
    def test_forward_prediction_quality(self):
        """Test that trained model produces bounded predictions."""
        print("\n=== Testing Forward Prediction Quality ===")
        
        # Quick training
        data = generate_training_dataset(
            seed=42, nx=30, nz=30,
            n_interior=500, n_boundary=100, n_initial=100
        )
        
        pinn = create_simple_pinn(hidden_dims=[32, 32], use_fourier=True)
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        batch = {
            'interior': data['collocation']['interior'],
            'boundary': data['collocation']['boundary'],
            'initial': data['collocation']['initial'],
            'velocity': data['velocity_at_interior']
        }
        
        # Train briefly
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        
        @jax.jit
        def train_step(params, opt_state):
            loss_fn = lambda p: pinn.total_loss(p, batch)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        for _ in range(30):
            params, opt_state, _ = train_step(params, opt_state)
        
        # Predict on grid
        u_grid = predict_on_grid(pinn, params, nx=20, nz=20, t=0.1)
        
        # Check predictions are bounded
        max_u = float(jnp.max(jnp.abs(u_grid)))
        print(f"  Max |u|: {max_u:.4f}")
        
        assert jnp.isfinite(u_grid).all(), "Predictions contain NaN/Inf"
        assert max_u < 100, f"Predictions too large: max |u| = {max_u}"
        
        print("✓ Prediction quality test passed!")


class TestTrainingInfrastructure:
    """Tests for training infrastructure."""
    
    def test_trainer_initialization(self):
        """Test Trainer class initialization."""
        print("\n=== Testing Trainer Initialization ===")
        
        config = WavePINNConfig(hidden_dims=[32, 32])
        pinn = WavePINN(config)
        
        train_config = TrainingConfig(
            n_epochs=10,
            learning_rate=1e-3
        )
        
        # Create trainer (don't log to disk for test)
        trainer = Trainer(pinn, train_config, log_dir=None, use_wandb=False)
        
        assert trainer.pinn is pinn
        assert trainer.config == train_config
        assert trainer.optimizer is not None
        
        print("✓ Trainer initialization test passed!")
    
    def test_train_state_creation(self):
        """Test training state initialization."""
        print("\n=== Testing Train State Creation ===")
        
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        optimizer = create_optimizer(TrainingConfig())
        key = jr.PRNGKey(42)
        
        state = init_train_state(pinn, optimizer, key)
        
        assert state.params is not None
        assert state.opt_state is not None
        assert state.step == 0
        
        print("✓ Train state creation test passed!")
    
    def test_single_train_step(self):
        """Test a single training step."""
        print("\n=== Testing Single Train Step ===")
        
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        config = TrainingConfig(use_mixed_precision=False)
        optimizer = create_optimizer(config)
        key = jr.PRNGKey(42)
        
        state = init_train_state(pinn, optimizer, key)
        train_step = make_train_step(pinn, optimizer, config)
        
        # Create batch
        batch = {
            'interior': jr.uniform(key, (100, 3)),
            'boundary': jr.uniform(key, (20, 3)),
            'initial': jnp.column_stack([
                jr.uniform(key, (20,)),
                jr.uniform(key, (20,)),
                jnp.zeros(20)
            ]),
            'velocity': jnp.ones(100) * 2000.0
        }
        
        new_state, metrics = train_step(state, batch)
        
        assert new_state.step == 1
        assert 'loss' in metrics
        assert jnp.isfinite(metrics['loss'])
        
        print(f"  Step 1 loss: {metrics['loss']:.4e}")
        print("✓ Single train step test passed!")


class TestMixedPrecision:
    """Tests for mixed-precision training."""
    
    def test_mixed_precision_training(self):
        """Test that mixed precision doesn't cause numerical issues."""
        print("\n=== Testing Mixed Precision Training ===")
        
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        config = TrainingConfig(use_mixed_precision=True, n_epochs=20)
        optimizer = create_optimizer(config)
        key = jr.PRNGKey(42)
        
        state = init_train_state(pinn, optimizer, key)
        train_step = make_train_step(pinn, optimizer, config)
        
        batch = {
            'interior': jr.uniform(key, (100, 3)),
            'boundary': jr.uniform(key, (20, 3)),
            'initial': jnp.column_stack([
                jr.uniform(key, (20,)),
                jr.uniform(key, (20,)),
                jnp.zeros(20)
            ]),
            'velocity': jnp.ones(100) * 2000.0
        }
        
        losses = []
        for i in range(20):
            state, metrics = train_step(state, batch)
            losses.append(float(metrics['loss']))
        
        # Check no NaN/Inf
        assert all(jnp.isfinite(l) for l in losses), "Mixed precision produced NaN/Inf"
        
        # Check loss decreased
        assert losses[-1] < losses[0], "Loss did not decrease with mixed precision"
        
        print(f"  Loss: {losses[0]:.4e} -> {losses[-1]:.4e}")
        print("✓ Mixed precision test passed!")


class TestMinimalAcceptance:
    """
    Minimal acceptance tests that must pass for the implementation
    to be considered working.
    """
    
    def test_acceptance_criteria(self):
        """
        Acceptance test: verify all success criteria are met.
        
        Criteria:
        1. ✅ Data generation produces valid slowness maps and collocation points
        2. ✅ Model forward pass returns finite wavefield values
        3. ✅ PDE loss is computable via autodiff without errors
        4. ✅ Training loop runs for 50 epochs and reduces loss by >50%
        5. ✅ Prediction at unseen coordinates returns physically plausible values
        """
        print("\n" + "=" * 60)
        print("ACCEPTANCE TEST: WavePINN-NIF-Scalar")
        print("=" * 60)
        
        # ===== Criterion 1: Data Generation =====
        print("\n[1/5] Testing data generation...")
        data = generate_training_dataset(
            seed=42,
            nx=50, nz=50,
            n_interior=1000,
            n_boundary=200,
            n_initial=200
        )
        
        assert data['slowness'].shape == (50, 50)
        assert jnp.all(data['slowness'] > 0)
        assert data['collocation']['interior'].shape == (1000, 3)
        print("  ✓ Data generation produces valid output")
        
        # ===== Criterion 2: Forward Pass =====
        print("\n[2/5] Testing model forward pass...")
        pinn = create_simple_pinn(hidden_dims=[64, 64, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        test_coords = jnp.array([[0.5, 0.5, 0.1], [0.3, 0.7, 0.2]])
        u = pinn.forward(params, test_coords)
        
        assert jnp.isfinite(u).all()
        assert u.shape == (2,)
        print("  ✓ Forward pass returns finite values")
        
        # ===== Criterion 3: PDE Loss Computation =====
        print("\n[3/5] Testing PDE loss computation...")
        batch = {
            'interior': data['collocation']['interior'][:100],
            'boundary': data['collocation']['boundary'][:50],
            'initial': data['collocation']['initial'][:50],
            'velocity': data['velocity_at_interior'][:100]
        }
        
        loss, components = pinn.total_loss(params, batch, return_components=True)
        assert jnp.isfinite(loss)
        assert jnp.isfinite(components['pde'])
        print(f"  ✓ PDE loss computed: {components['pde']:.4e}")
        
        # ===== Criterion 4: Training Loop =====
        print("\n[4/5] Testing training loop (50 epochs)...")
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        
        # Full batch
        full_batch = {
            'interior': data['collocation']['interior'],
            'boundary': data['collocation']['boundary'],
            'initial': data['collocation']['initial'],
            'velocity': data['velocity_at_interior']
        }
        
        @jax.jit
        def train_step(params, opt_state, batch):
            loss_fn = lambda p: pinn.total_loss(p, batch)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        initial_loss = float(pinn.total_loss(params, full_batch))
        print(f"  Initial loss: {initial_loss:.4e}")
        
        for epoch in range(50):
            params, opt_state, loss = train_step(params, opt_state, full_batch)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {float(loss):.4e}")
        
        final_loss = float(loss)
        print(f"  Final loss: {final_loss:.4e}")
        
        reduction = (initial_loss - final_loss) / initial_loss
        print(f"  Loss reduction: {reduction * 100:.1f}%")
        
        assert final_loss < initial_loss * 0.5, \
            f"Loss must decrease by >50%, got {reduction * 100:.1f}%"
        print("  ✓ Loss decreased by >50%")
        
        # ===== Criterion 5: Prediction Quality =====
        print("\n[5/5] Testing prediction quality...")
        test_grid = predict_on_grid(pinn, params, nx=30, nz=30, t=0.1)
        
        assert jnp.isfinite(test_grid).all()
        max_u = float(jnp.max(jnp.abs(test_grid)))
        assert max_u < 10, f"Predictions too large: |u|_max = {max_u}"
        print(f"  ✓ Predictions are finite and bounded (|u|_max = {max_u:.4f})")
        
        # ===== Final Result =====
        print("\n" + "=" * 60)
        print("✓ ALL ACCEPTANCE CRITERIA PASSED")
        print("✓ WavePINN-NIF-Scalar is operational!")
        print("=" * 60)


if __name__ == "__main__":
    # Run acceptance test directly
    test = TestMinimalAcceptance()
    test.test_acceptance_criteria()
    
    print("\nRunning full test suite...")
    pytest.main([__file__, "-v"])
