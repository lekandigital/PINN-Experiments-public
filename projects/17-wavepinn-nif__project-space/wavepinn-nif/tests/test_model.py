"""
Unit tests for the WavePINN model module.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from src.model import (
    WavePINN,
    WavePINNConfig,
    create_simple_pinn,
    predict_on_grid,
    predict_time_series,
)
from src.data_gen import generate_training_dataset


class TestWavePINNConfig:
    """Tests for model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WavePINNConfig()
        
        assert config.hidden_dims == [128, 128, 64]
        assert config.activation == "tanh"
        assert config.use_fourier_features == True
        assert config.spatial_dim == 2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WavePINNConfig(
            hidden_dims=[64, 64],
            activation="sin",
            lambda_pde=10.0
        )
        
        assert config.hidden_dims == [64, 64]
        assert config.activation == "sin"
        assert config.lambda_pde == 10.0


class TestWavePINNInit:
    """Tests for model initialization."""
    
    def test_init_params(self):
        """Test parameter initialization."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        # Check params is a dict (Haiku format)
        assert isinstance(params, dict)
        
        # Check params are not empty
        leaves = jax.tree_util.tree_leaves(params)
        assert len(leaves) > 0
        
        # Check all params are finite
        for leaf in leaves:
            assert jnp.isfinite(leaf).all()
    
    def test_fourier_basis_created(self):
        """Test that Fourier basis is created when enabled."""
        config = WavePINNConfig(use_fourier_features=True, num_fourier_features=32)
        pinn = WavePINN(config)
        
        assert pinn.fourier_B is not None
        assert pinn.fourier_B.shape[1] == 32
    
    def test_no_fourier_basis_when_disabled(self):
        """Test that Fourier basis is None when disabled."""
        config = WavePINNConfig(use_fourier_features=False)
        pinn = WavePINN(config)
        
        assert pinn.fourier_B is None


class TestWavePINNForward:
    """Tests for forward pass."""
    
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        # Single point
        coords = jnp.array([[0.5, 0.5, 0.1]])
        u = pinn.forward(params, coords)
        assert u.shape == (1,), f"Expected (1,), got {u.shape}"
        
        # Batch
        coords_batch = jnp.ones((100, 3)) * 0.5
        u_batch = pinn.forward(params, coords_batch)
        assert u_batch.shape == (100,)
    
    def test_forward_output_finite(self):
        """Test that forward pass produces finite values."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        coords = jr.uniform(key, (50, 3))
        u = pinn.forward(params, coords)
        
        assert jnp.isfinite(u).all(), "Forward pass produced non-finite values"
    
    def test_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        coords = jnp.array([[0.5, 0.5, 0.1]])
        u1 = pinn.forward(params, coords)
        u2 = pinn.forward(params, coords)
        
        assert jnp.allclose(u1, u2)


class TestPDEResidual:
    """Tests for PDE residual computation."""
    
    def test_pde_residual_shape(self):
        """Test PDE residual output shape."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        n_points = 50
        coords = jr.uniform(key, (n_points, 3))
        velocity = jnp.ones(n_points) * 2000.0
        
        residual = pinn.compute_pde_residual_2d_efficient(params, coords, velocity)
        
        assert residual.shape == (n_points,)
    
    def test_pde_residual_finite(self):
        """Test PDE residual produces finite values."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        n_points = 20
        coords = jr.uniform(key, (n_points, 3)) * 0.5 + 0.25  # Avoid boundaries
        velocity = jnp.ones(n_points) * 2000.0
        
        residual = pinn.compute_pde_residual_2d_efficient(params, coords, velocity)
        
        assert jnp.isfinite(residual).all(), "PDE residual contains non-finite values"
    
    def test_pde_residual_with_source(self):
        """Test PDE residual with source term."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        n_points = 20
        coords = jr.uniform(key, (n_points, 3))
        velocity = jnp.ones(n_points) * 2000.0
        source = jnp.ones(n_points) * 0.1
        
        residual = pinn.compute_pde_residual_2d_efficient(params, coords, velocity, source)
        
        assert residual.shape == (n_points,)
        assert jnp.isfinite(residual).all()


class TestBoundaryLoss:
    """Tests for boundary condition loss."""
    
    def test_boundary_loss_positive(self):
        """Test that boundary loss is non-negative."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        boundary_coords = jr.uniform(key, (50, 3))
        loss = pinn.boundary_loss(params, boundary_coords)
        
        assert loss >= 0
    
    def test_boundary_loss_finite(self):
        """Test that boundary loss is finite."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        boundary_coords = jr.uniform(key, (50, 3))
        loss = pinn.boundary_loss(params, boundary_coords)
        
        assert jnp.isfinite(loss)


class TestInitialLoss:
    """Tests for initial condition loss."""
    
    def test_initial_loss_shape(self):
        """Test initial condition loss returns two scalars."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        # Initial points at t=0
        initial_coords = jnp.column_stack([
            jr.uniform(key, (50,)),  # x
            jr.uniform(key, (50,)),  # z
            jnp.zeros(50)  # t=0
        ])
        
        loss_u, loss_dt = pinn.initial_loss(params, initial_coords)
        
        assert loss_u.shape == ()
        assert loss_dt.shape == ()
    
    def test_initial_loss_nonnegative(self):
        """Test that initial losses are non-negative."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        initial_coords = jnp.column_stack([
            jr.uniform(key, (50,)),
            jr.uniform(key, (50,)),
            jnp.zeros(50)
        ])
        
        loss_u, loss_dt = pinn.initial_loss(params, initial_coords)
        
        assert loss_u >= 0
        assert loss_dt >= 0


class TestTotalLoss:
    """Tests for total physics-informed loss."""
    
    def test_total_loss_computation(self):
        """Test total loss computation."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        # Create batch
        n_interior, n_boundary, n_initial = 50, 20, 20
        k1, k2, k3 = jr.split(key, 3)
        
        batch = {
            'interior': jr.uniform(k1, (n_interior, 3)),
            'boundary': jr.uniform(k2, (n_boundary, 3)),
            'initial': jnp.column_stack([
                jr.uniform(k3, (n_initial,)),
                jr.uniform(k3, (n_initial,)),
                jnp.zeros(n_initial)
            ]),
            'velocity': jnp.ones(n_interior) * 2000.0
        }
        
        loss = pinn.total_loss(params, batch)
        
        assert jnp.isfinite(loss)
        assert loss >= 0
    
    def test_total_loss_with_components(self):
        """Test total loss with component breakdown."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        batch = {
            'interior': jr.uniform(key, (50, 3)),
            'boundary': jr.uniform(key, (20, 3)),
            'initial': jnp.column_stack([
                jr.uniform(key, (20,)),
                jr.uniform(key, (20,)),
                jnp.zeros(20)
            ]),
            'velocity': jnp.ones(50) * 2000.0
        }
        
        total, components = pinn.total_loss(params, batch, return_components=True)
        
        assert 'pde' in components
        assert 'bc' in components
        assert 'ic_u' in components
        assert 'total' in components
        
        # All components should be non-negative
        for name, val in components.items():
            assert val >= 0, f"Component {name} is negative: {val}"


class TestPredictionUtilities:
    """Tests for prediction utility functions."""
    
    def test_predict_on_grid(self):
        """Test grid prediction."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        nx, nz = 20, 25
        u_grid = predict_on_grid(pinn, params, nx, nz, t=0.1)
        
        assert u_grid.shape == (nx, nz)
        assert jnp.isfinite(u_grid).all()
    
    def test_predict_time_series(self):
        """Test time series prediction."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        times = jnp.linspace(0, 0.5, 50)
        u_series = predict_time_series(pinn, params, location=(0.5, 0.5), times=times)
        
        assert u_series.shape == (50,)
        assert jnp.isfinite(u_series).all()


class TestGradientComputation:
    """Tests for gradient computation (for training)."""
    
    def test_loss_gradient_finite(self):
        """Test that loss gradients are finite."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        batch = {
            'interior': jr.uniform(key, (30, 3)),
            'boundary': jr.uniform(key, (10, 3)),
            'initial': jnp.column_stack([
                jr.uniform(key, (10,)),
                jr.uniform(key, (10,)),
                jnp.zeros(10)
            ]),
            'velocity': jnp.ones(30) * 2000.0
        }
        
        loss_fn = lambda p: pinn.total_loss(p, batch)
        grads = jax.grad(loss_fn)(params)
        
        # Check all gradients are finite
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.isfinite(leaf).all(), "Gradient contains non-finite values"
    
    def test_gradient_nonzero(self):
        """Test that gradients are not all zero (learning signal exists)."""
        pinn = create_simple_pinn(hidden_dims=[32, 32])
        key = jr.PRNGKey(42)
        params = pinn.init_params(key)
        
        batch = {
            'interior': jr.uniform(key, (30, 3)),
            'boundary': jr.uniform(key, (10, 3)),
            'initial': jnp.column_stack([
                jr.uniform(key, (10,)),
                jr.uniform(key, (10,)),
                jnp.zeros(10)
            ]),
            'velocity': jnp.ones(30) * 2000.0
        }
        
        loss_fn = lambda p: pinn.total_loss(p, batch)
        grads = jax.grad(loss_fn)(params)
        
        # At least some gradients should be non-zero
        total_grad_norm = sum(jnp.sum(jnp.abs(g)) for g in jax.tree_util.tree_leaves(grads))
        assert total_grad_norm > 0, "All gradients are zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
