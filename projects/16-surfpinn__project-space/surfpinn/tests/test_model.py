"""
SurfPINN Unit Tests
===================
Tests for model architecture, physics losses, and data generation.

Run with: pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import (
    SurfPINN, SurfPINNConfig, create_model, init_model,
    EulerianEncoder, LagrangianEncoder
)
from physics import (
    gradient_x, gradient_y, divergence_2d, laplacian_2d,
    mean_curvature_loss, continuity_loss, data_loss,
    total_physics_loss, PhysicsConfig, compute_psnr, compute_rmse
)
from data_gen import (
    shallow_water_dam_break, generate_eulerian_data,
    generate_lagrangian_data, generate_dataset
)


class TestModel:
    """Tests for SurfPINN model architecture."""
    
    def test_model_init(self):
        """Test model initialization."""
        rng = jax.random.PRNGKey(0)
        config = SurfPINNConfig(latent_dim=64)
        
        params, state = init_model(
            rng, config,
            grid_shape=(32, 32, 4),
            n_particles=100
        )
        
        assert params is not None
        assert state is not None
        
        # Check parameter count
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        assert n_params > 0
        print(f"Model initialized with {n_params:,} parameters")
    
    def test_forward_pass_shapes(self):
        """Test output shapes from forward pass."""
        rng = jax.random.PRNGKey(0)
        config = SurfPINNConfig(latent_dim=64)
        model = create_model(config)
        
        # Test inputs
        batch_size = 2
        nx, ny = 32, 32
        n_particles = 100
        
        grid_input = jnp.ones((batch_size, nx, ny, 4))
        particle_pos = jnp.ones((batch_size, n_particles, 3))
        
        # Initialize
        params, state = model.init(rng, grid_input, particle_pos, True)
        
        # Forward pass
        (height, velocity, z_eul, z_lag), _ = model.apply(
            params, state, rng, grid_input, particle_pos, True
        )
        
        # Check shapes
        assert height.shape == (batch_size, nx, ny, 1), f"Height shape: {height.shape}"
        assert velocity.shape == (batch_size, n_particles, 3), f"Velocity shape: {velocity.shape}"
        # Eulerian latent has shape (batch, nx, ny, channels) where channels = eul_channels[-1]
        assert z_eul.shape[-1] == config.eul_channels[-1], f"Eulerian latent: {z_eul.shape}"
        assert z_lag.shape == (batch_size, n_particles, config.lag_hidden[-1]), f"Lagrangian latent: {z_lag.shape}"
        
        print("✓ Forward pass shapes correct")
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        rng = jax.random.PRNGKey(0)
        config = SurfPINNConfig(latent_dim=32)
        model = create_model(config)
        
        grid_input = jnp.ones((1, 16, 16, 4))
        particle_pos = jnp.ones((1, 50, 3))
        
        params, state = model.init(rng, grid_input, particle_pos, True)
        
        def loss_fn(params):
            (height, velocity, _, _), _ = model.apply(
                params, state, rng, grid_input, particle_pos, True
            )
            return jnp.mean(height ** 2) + jnp.mean(velocity ** 2)
        
        grads = jax.grad(loss_fn)(params)
        
        # Check all gradients are non-zero
        grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grads)
        all_norms = jax.tree_util.tree_leaves(grad_norms)
        
        assert all(n > 0 for n in all_norms), "Some gradients are zero!"
        print("✓ Gradients flow through all layers")
    
    def test_output_values_reasonable(self):
        """Test that outputs are numerically reasonable (no NaN/Inf)."""
        rng = jax.random.PRNGKey(42)
        config = SurfPINNConfig()
        model = create_model(config)
        
        # Random inputs
        grid_input = jax.random.normal(rng, (1, 32, 32, 4))
        particle_pos = jax.random.uniform(rng, (1, 100, 3))
        
        params, state = model.init(rng, grid_input, particle_pos, True)
        (height, velocity, _, _), _ = model.apply(
            params, state, rng, grid_input, particle_pos, True
        )
        
        assert jnp.all(jnp.isfinite(height)), "Height contains NaN/Inf"
        assert jnp.all(jnp.isfinite(velocity)), "Velocity contains NaN/Inf"
        assert jnp.all(height >= 0), "Height should be non-negative (softplus)"
        
        print("✓ Outputs are numerically stable")


class TestPhysics:
    """Tests for physics loss functions."""
    
    def test_gradient_x(self):
        """Test x-gradient computation."""
        # Linear function: f(x,y) = 2x
        x = jnp.linspace(0, 1, 32)
        y = jnp.linspace(0, 1, 32)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        f = 2 * X
        
        dx = x[1] - x[0]
        df_dx = gradient_x(f, dx)
        
        # Gradient should be ~2 everywhere (except boundaries)
        interior = df_dx[1:-1, 1:-1]
        assert jnp.allclose(interior, 2.0, atol=0.1), f"Expected ~2, got {interior.mean()}"
        print("✓ X-gradient correct")
    
    def test_gradient_y(self):
        """Test y-gradient computation."""
        x = jnp.linspace(0, 1, 32)
        y = jnp.linspace(0, 1, 32)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        f = 3 * Y
        
        dy = y[1] - y[0]
        df_dy = gradient_y(f, dy)
        
        interior = df_dy[1:-1, 1:-1]
        assert jnp.allclose(interior, 3.0, atol=0.1), f"Expected ~3, got {interior.mean()}"
        print("✓ Y-gradient correct")
    
    def test_divergence_zero(self):
        """Test divergence of incompressible field is zero."""
        # u = -y, v = x (rotational, divergence-free)
        x = jnp.linspace(-1, 1, 32)
        y = jnp.linspace(-1, 1, 32)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        u = -Y
        v = X
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        div = divergence_2d(u, v, dx, dy)
        interior = div[2:-2, 2:-2]
        
        assert jnp.allclose(interior, 0.0, atol=0.1), f"Divergence should be ~0, got {interior.mean()}"
        print("✓ Divergence computation correct")
    
    def test_laplacian(self):
        """Test Laplacian of known function."""
        # f(x,y) = x² + y² => ∇²f = 4
        x = jnp.linspace(-1, 1, 32)
        y = jnp.linspace(-1, 1, 32)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        f = X**2 + Y**2
        
        dx = x[1] - x[0]
        lap = laplacian_2d(f, dx, dx)
        interior = lap[2:-2, 2:-2]
        
        assert jnp.allclose(interior, 4.0, atol=0.3), f"Expected ~4, got {interior.mean()}"
        print("✓ Laplacian computation correct")
    
    def test_curvature_loss_flat_surface(self):
        """Test curvature loss is zero for flat surface."""
        # Flat surface: h = constant
        height = jnp.ones((32, 32, 1))
        
        curv_loss = mean_curvature_loss(height, dx=0.1, dy=0.1)
        
        assert curv_loss < 1e-6, f"Curvature of flat surface should be ~0, got {curv_loss}"
        print("✓ Curvature loss correct for flat surface")
    
    def test_curvature_loss_curved_surface(self):
        """Test curvature loss is non-zero for curved surface."""
        # Curved surface: h = sin(x) * sin(y)
        x = jnp.linspace(0, 2*jnp.pi, 32)
        y = jnp.linspace(0, 2*jnp.pi, 32)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        height = jnp.sin(X) * jnp.sin(Y)
        height = height[..., None]
        
        curv_loss = mean_curvature_loss(height, dx=x[1]-x[0], dy=y[1]-y[0])
        
        assert curv_loss > 0.01, f"Curved surface should have non-zero curvature, got {curv_loss}"
        print("✓ Curvature loss detects curved surfaces")
    
    def test_continuity_loss(self):
        """Test continuity loss computation."""
        # Divergent field: u = x, v = y => div = 2
        x = jnp.linspace(-1, 1, 32)
        y = jnp.linspace(-1, 1, 32)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        u = X
        v = Y
        
        loss = continuity_loss(u, v, dx=x[1]-x[0], dy=y[1]-y[0])
        
        assert loss > 1.0, f"Divergent field should have high continuity loss, got {loss}"
        print("✓ Continuity loss detects divergent fields")
    
    def test_data_loss(self):
        """Test data loss computation."""
        pred = jnp.array([1.0, 2.0, 3.0])
        true = jnp.array([1.1, 2.2, 2.9])
        
        h_loss, v_loss = data_loss(pred, true, pred, true)
        
        expected = jnp.mean((pred - true) ** 2)
        assert jnp.isclose(h_loss, expected), f"Expected {expected}, got {h_loss}"
        print("✓ Data loss correct")
    
    def test_total_physics_loss(self):
        """Test combined physics loss."""
        height_pred = jnp.ones((1, 32, 32, 1)) * 0.5
        height_true = jnp.ones((1, 32, 32, 1)) * 0.5
        velocity_pred = jnp.zeros((1, 100, 3))
        velocity_true = jnp.zeros((1, 100, 3))
        
        losses = total_physics_loss(
            height_pred, velocity_pred,
            height_true, velocity_true,
            PhysicsConfig()
        )
        
        assert 'total' in losses
        assert 'data' in losses
        assert 'curvature' in losses
        assert jnp.isfinite(losses['total'])
        print("✓ Total physics loss computes correctly")


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_psnr(self):
        """Test PSNR computation."""
        true = jnp.array([1.0, 2.0, 3.0, 4.0])
        pred = true + 0.1  # Small error
        
        psnr = compute_psnr(pred, true)
        
        assert psnr > 20, f"PSNR should be high for small error, got {psnr}"
        print(f"✓ PSNR = {psnr:.2f} dB")
    
    def test_rmse(self):
        """Test RMSE computation."""
        true = jnp.array([1.0, 2.0, 3.0])
        pred = jnp.array([1.0, 2.0, 4.0])  # Error of 1 on last element
        
        rmse = compute_rmse(pred, true)
        expected = jnp.sqrt(1/3)
        
        assert jnp.isclose(rmse, expected, atol=0.01), f"Expected {expected}, got {rmse}"
        print("✓ RMSE correct")


class TestDataGen:
    """Tests for synthetic data generation."""
    
    def test_dam_break_initial(self):
        """Test dam break initial condition."""
        x = np.linspace(0, 1, 100)
        h = shallow_water_dam_break(x, t=0.0)
        
        # Left side should be high, right side low
        assert np.mean(h[x < 0.3]) > 0.6, "Left side should be high"
        assert np.mean(h[x > 0.5]) < 0.5, "Right side should be low"
        print("✓ Dam break initial condition correct")
    
    def test_dam_break_evolution(self):
        """Test dam break evolves over time."""
        x = np.linspace(0, 1, 100)
        h0 = shallow_water_dam_break(x, t=0.0)
        h1 = shallow_water_dam_break(x, t=0.5)
        
        # Wave should propagate rightward
        assert not np.allclose(h0, h1), "Height field should change over time"
        print("✓ Dam break evolution works")
    
    def test_eulerian_data_shapes(self):
        """Test Eulerian data generation shapes."""
        data = generate_eulerian_data(nx=32, ny=32, nt=8)
        
        assert data['grid_coords'].shape == (32, 32, 8, 3)
        assert data['height'].shape == (32, 32, 8, 1)
        assert data['velocity'].shape == (32, 32, 8, 2)
        print("✓ Eulerian data shapes correct")
    
    def test_lagrangian_data_shapes(self):
        """Test Lagrangian data generation shapes."""
        data = generate_lagrangian_data(n_particles=500, nt=8)
        
        assert data['positions'].shape == (500, 8, 3)
        assert data['velocities'].shape == (500, 8, 3)
        print("✓ Lagrangian data shapes correct")
    
    def test_data_values_physical(self):
        """Test generated data has physical values."""
        eul = generate_eulerian_data(nx=32, ny=32, nt=8)
        lag = generate_lagrangian_data(n_particles=100, nt=8)
        
        # Heights should be positive
        assert np.all(eul['height'] >= 0), "Heights should be non-negative"
        
        # Particles should stay in domain
        assert np.all(lag['positions'][:, :, 0] >= 0), "Particles should stay in domain"
        assert np.all(lag['positions'][:, :, 0] <= 1), "Particles should stay in domain"
        
        # Velocities should be finite
        assert np.all(np.isfinite(eul['velocity'])), "Velocities should be finite"
        assert np.all(np.isfinite(lag['velocities'])), "Velocities should be finite"
        
        print("✓ Generated data is physically reasonable")


class TestMemory:
    """Tests for memory usage (important for GPU deployment)."""
    
    def test_model_memory_estimate(self):
        """Estimate model memory usage."""
        rng = jax.random.PRNGKey(0)
        config = SurfPINNConfig(latent_dim=128)
        
        params, state = init_model(
            rng, config,
            grid_shape=(64, 64, 4),
            n_particles=1000
        )
        
        # Estimate parameter memory (float32 = 4 bytes)
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        param_memory_mb = n_params * 4 / (1024 ** 2)
        
        # For training: params + gradients + optimizer state (~3-4x params)
        estimated_train_memory_mb = param_memory_mb * 4
        
        print(f"  Parameters: {n_params:,}")
        print(f"  Parameter memory: {param_memory_mb:.1f} MB")
        print(f"  Estimated training memory: {estimated_train_memory_mb:.1f} MB")
        
        # Should fit easily in 48GB (L40S) or 24GB (3090)
        assert estimated_train_memory_mb < 1000, "Model too large for target GPUs"
        print("✓ Model fits in GPU memory")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
