"""
Unit tests for data generation module.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Import the module to test
from src.data_gen import (
    generate_slowness_map_2d,
    generate_slowness_map_3d,
    generate_layered_velocity_2d,
    ricker_wavelet,
    ricker_derivative,
    sample_collocation_points_2d,
    interpolate_velocity_at_coords,
    generate_training_dataset,
)


class TestSlownessMaps:
    """Tests for slowness map generation."""
    
    def test_generate_slowness_map_2d_shape(self):
        """Test that 2D slowness map has correct shape."""
        nx, nz = 50, 60
        slowness, velocity, coords = generate_slowness_map_2d(nx, nz, seed=42)
        
        assert slowness.shape == (nx, nz), f"Expected {(nx, nz)}, got {slowness.shape}"
        assert velocity.shape == (nx, nz)
        assert coords.shape == (nx * nz, 2)
    
    def test_slowness_is_positive(self):
        """Test that slowness values are positive."""
        slowness, _, _ = generate_slowness_map_2d(50, 50, seed=42)
        
        assert jnp.all(slowness > 0), "Slowness must be positive"
    
    def test_velocity_range(self):
        """Test that velocity is in reasonable range."""
        _, velocity, _ = generate_slowness_map_2d(
            50, 50,
            base_velocity=2000.0,
            anomaly_strength=0.3,
            seed=42
        )
        
        # With 30% anomaly strength, velocity should be ~1400-2600 m/s
        assert jnp.min(velocity) > 500, f"Min velocity too low: {jnp.min(velocity)}"
        assert jnp.max(velocity) < 5000, f"Max velocity too high: {jnp.max(velocity)}"
    
    def test_slowness_velocity_inverse(self):
        """Test that slowness = 1/velocity."""
        slowness, velocity, _ = generate_slowness_map_2d(50, 50, seed=42)
        
        reconstructed = 1.0 / velocity
        assert jnp.allclose(slowness, reconstructed, rtol=1e-5)
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        s1, v1, _ = generate_slowness_map_2d(50, 50, seed=42)
        s2, v2, _ = generate_slowness_map_2d(50, 50, seed=42)
        
        assert jnp.allclose(s1, s2)
        assert jnp.allclose(v1, v2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds give different results."""
        s1, _, _ = generate_slowness_map_2d(50, 50, seed=42)
        s2, _, _ = generate_slowness_map_2d(50, 50, seed=123)
        
        assert not jnp.allclose(s1, s2)
    
    def test_3d_slowness_map_shape(self):
        """Test 3D slowness map generation."""
        nx, ny, nz = 20, 25, 30
        slowness, velocity, coords = generate_slowness_map_3d(nx, ny, nz, seed=42)
        
        assert slowness.shape == (nx, ny, nz)
        assert velocity.shape == (nx, ny, nz)
        assert coords.shape == (nx * ny * nz, 3)
    
    def test_layered_velocity_model(self):
        """Test layered velocity model."""
        slowness, velocity, _ = generate_layered_velocity_2d(
            100, 100,
            layer_velocities=[1500, 2000, 2500],
            layer_depths=[0.3, 0.7, 1.0]
        )
        
        # Check that we have distinct layers
        unique_vels = jnp.unique(velocity)
        assert len(unique_vels) == 3, f"Expected 3 layers, got {len(unique_vels)}"


class TestRickerWavelet:
    """Tests for Ricker wavelet generation."""
    
    def test_wavelet_shape(self):
        """Test wavelet output shape."""
        t = jnp.linspace(0, 1, 500)
        wavelet = ricker_wavelet(t, f0=25.0)
        
        assert wavelet.shape == t.shape
    
    def test_wavelet_zero_at_boundaries(self):
        """Test that wavelet decays to near-zero at boundaries."""
        t = jnp.linspace(0, 0.5, 500)
        wavelet = ricker_wavelet(t, f0=25.0)
        
        # Should be small at early and late times
        assert jnp.abs(wavelet[0]) < 0.1
        assert jnp.abs(wavelet[-1]) < 0.1
    
    def test_wavelet_has_peak(self):
        """Test that wavelet has a clear peak near delay time."""
        t = jnp.linspace(0, 0.2, 500)
        f0 = 25.0
        t0 = 1.5 / f0  # Default delay
        wavelet = ricker_wavelet(t, f0=f0)
        
        # Peak should be near t0
        peak_idx = jnp.argmax(wavelet)
        peak_time = t[peak_idx]
        
        assert jnp.abs(peak_time - t0) < 0.01, f"Peak at {peak_time}, expected near {t0}"
    
    def test_wavelet_dominant_frequency(self):
        """Test wavelet has correct dominant frequency via FFT."""
        dt = 0.001  # 1 ms sampling
        t = jnp.arange(0, 0.5, dt)
        f0 = 25.0
        wavelet = ricker_wavelet(t, f0=f0)
        
        # FFT
        fft = jnp.fft.rfft(wavelet)
        freqs = jnp.fft.rfftfreq(len(t), dt)
        
        # Find peak frequency
        peak_freq_idx = jnp.argmax(jnp.abs(fft[1:]))  # Skip DC
        peak_freq = freqs[peak_freq_idx + 1]
        
        # Peak should be close to f0
        assert jnp.abs(peak_freq - f0) < 5.0, f"Peak at {peak_freq} Hz, expected {f0} Hz"
    
    def test_ricker_derivative(self):
        """Test Ricker derivative computation."""
        t = jnp.linspace(0, 0.2, 500)
        deriv = ricker_derivative(t, f0=25.0)
        
        assert deriv.shape == t.shape
        assert jnp.isfinite(deriv).all()


class TestCollocationPoints:
    """Tests for collocation point sampling."""
    
    def test_sample_points_2d_shapes(self):
        """Test that sampled points have correct shapes."""
        key = jr.PRNGKey(42)
        n_interior, n_boundary, n_initial = 1000, 400, 200
        
        points = sample_collocation_points_2d(
            key,
            n_interior=n_interior,
            n_boundary=n_boundary,
            n_initial=n_initial,
            domain_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 0.5)
        )
        
        assert points['interior'].shape == (n_interior, 3)
        assert points['boundary'].shape == (n_boundary, 3)
        assert points['initial'].shape == (n_initial, 3)
    
    def test_interior_points_in_domain(self):
        """Test that interior points are within domain bounds."""
        key = jr.PRNGKey(42)
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.5)
        
        points = sample_collocation_points_2d(
            key, 1000, 400, 200, domain_bounds=bounds
        )
        
        interior = points['interior']
        
        assert jnp.all(interior[:, 0] >= bounds[0])  # x >= x_min
        assert jnp.all(interior[:, 0] <= bounds[1])  # x <= x_max
        assert jnp.all(interior[:, 1] >= bounds[2])  # z >= z_min
        assert jnp.all(interior[:, 1] <= bounds[3])  # z <= z_max
        assert jnp.all(interior[:, 2] >= bounds[4])  # t >= t_min
        assert jnp.all(interior[:, 2] <= bounds[5])  # t <= t_max
    
    def test_boundary_points_on_edges(self):
        """Test that boundary points lie on domain edges."""
        key = jr.PRNGKey(42)
        x_min, x_max = 0.0, 1.0
        z_min, z_max = 0.0, 1.0
        
        points = sample_collocation_points_2d(
            key, 1000, 400, 200,
            domain_bounds=(x_min, x_max, z_min, z_max, 0.0, 0.5)
        )
        
        boundary = points['boundary']
        
        # Each point should be on at least one edge
        on_left = jnp.isclose(boundary[:, 0], x_min)
        on_right = jnp.isclose(boundary[:, 0], x_max)
        on_bottom = jnp.isclose(boundary[:, 1], z_min)
        on_top = jnp.isclose(boundary[:, 1], z_max)
        
        on_edge = on_left | on_right | on_bottom | on_top
        assert jnp.all(on_edge), "All boundary points should be on an edge"
    
    def test_initial_points_at_t0(self):
        """Test that initial points are at t=0."""
        key = jr.PRNGKey(42)
        t_min = 0.0
        
        points = sample_collocation_points_2d(
            key, 1000, 400, 200,
            domain_bounds=(0.0, 1.0, 0.0, 1.0, t_min, 0.5)
        )
        
        initial = points['initial']
        
        assert jnp.allclose(initial[:, 2], t_min), "Initial points should have t = t_min"


class TestVelocityInterpolation:
    """Tests for velocity field interpolation."""
    
    def test_interpolation_at_grid_points(self):
        """Test that interpolation is exact at grid points."""
        nx, nz = 10, 10
        _, velocity, coords = generate_slowness_map_2d(nx, nz, seed=42)
        
        # Interpolate at grid points
        interp_vel = interpolate_velocity_at_coords(
            velocity, coords,
            x_range=(0.0, 1.0), z_range=(0.0, 1.0)
        )
        
        assert jnp.allclose(interp_vel, velocity.ravel(), rtol=1e-3)
    
    def test_interpolation_bounds(self):
        """Test that interpolation stays within velocity bounds."""
        nx, nz = 50, 50
        _, velocity, _ = generate_slowness_map_2d(nx, nz, seed=42)
        
        # Random test points
        key = jr.PRNGKey(42)
        test_coords = jr.uniform(key, (100, 2))
        
        interp_vel = interpolate_velocity_at_coords(
            velocity, test_coords,
            x_range=(0.0, 1.0), z_range=(0.0, 1.0)
        )
        
        assert jnp.all(interp_vel >= velocity.min() - 1e-5)
        assert jnp.all(interp_vel <= velocity.max() + 1e-5)


class TestTrainingDataset:
    """Tests for complete training dataset generation."""
    
    def test_generate_training_dataset(self):
        """Test complete dataset generation."""
        data = generate_training_dataset(
            seed=42,
            nx=50,
            nz=50,
            n_interior=1000,
            n_boundary=200,
            n_initial=100
        )
        
        # Check all required keys exist
        assert 'slowness' in data
        assert 'velocity' in data
        assert 'collocation' in data
        assert 'velocity_at_interior' in data
        assert 'source_params' in data
        assert 'domain_bounds' in data
        
        # Check shapes
        assert data['slowness'].shape == (50, 50)
        assert data['velocity'].shape == (50, 50)
        assert data['collocation']['interior'].shape == (1000, 3)
        assert data['collocation']['boundary'].shape[1] == 3
        assert data['velocity_at_interior'].shape == (1000,)
    
    def test_velocity_at_interior_matches_grid(self):
        """Test that interpolated velocity is consistent with grid."""
        data = generate_training_dataset(
            seed=42, nx=50, nz=50, n_interior=500
        )
        
        vel_interp = data['velocity_at_interior']
        vel_grid = data['velocity']
        
        # Interpolated values should be within grid range
        assert jnp.min(vel_interp) >= jnp.min(vel_grid) * 0.9
        assert jnp.max(vel_interp) <= jnp.max(vel_grid) * 1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
