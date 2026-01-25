"""
Unit Tests for NACA Generator
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_generation.naca_generator import (
    generate_naca4,
    naca4_thickness,
    naca4_camber,
    naca_designation,
    generate_parametric_sweep
)


class TestNACAGenerator:
    """Test suite for NACA airfoil generator."""
    
    def test_naca4_thickness_at_zero(self):
        """Thickness should be zero at trailing edge (x=1)."""
        t = 0.12  # 12% thickness
        x = np.array([0.0, 0.5, 1.0])
        thickness = naca4_thickness(t, x)
        
        # Trailing edge should be closed (near zero)
        assert thickness[-1] < 0.01
    
    def test_naca4_thickness_max(self):
        """Maximum thickness occurs around x=0.3 for NACA 4-digit."""
        t = 0.12
        x = np.linspace(0, 1, 100)
        thickness = naca4_thickness(t, x)
        
        max_idx = np.argmax(thickness)
        max_x = x[max_idx]
        
        # Max thickness should be around 0.3 chord
        assert 0.2 < max_x < 0.4
        
        # Max thickness should be close to t/2
        assert abs(thickness[max_idx] - t/2) < 0.02
    
    def test_naca4_camber_symmetric(self):
        """Symmetric airfoil (m=0) should have zero camber."""
        m, p = 0.0, 0.4
        x = np.linspace(0, 1, 100)
        yc, dyc = naca4_camber(m, p, x)
        
        assert np.allclose(yc, 0.0)
        assert np.allclose(dyc, 0.0)
    
    def test_naca4_camber_nonsymmetric(self):
        """Cambered airfoil should have non-zero camber line."""
        m, p = 0.04, 0.4  # NACA 4412
        x = np.linspace(0, 1, 100)
        yc, dyc = naca4_camber(m, p, x)
        
        # Maximum camber should be at p
        max_idx = np.argmax(yc)
        max_x = x[max_idx]
        
        assert abs(max_x - p) < 0.05
        assert abs(yc[max_idx] - m) < 0.01
    
    def test_generate_naca4_basic(self):
        """Basic NACA 0012 airfoil generation."""
        m, p, t = 0.0, 0.0, 0.12
        x, y = generate_naca4(m, p, t, n_points=100)
        
        # Should return 199 points (100 upper + 99 lower, shared LE)
        assert len(x) == 199
        assert len(y) == len(x)
        
        # Symmetric airfoil should have points above and below y=0
        assert np.any(y > 0)
        assert np.any(y < 0)
    
    def test_generate_naca4_cambered(self):
        """NACA 2412 airfoil generation."""
        m, p, t = 0.02, 0.4, 0.12
        x, y = generate_naca4(m, p, t, n_points=100)
        
        # Check coordinate ranges
        assert np.min(x) >= -0.01
        assert np.max(x) <= 1.01
    
    def test_generate_naca4_cosine_spacing(self):
        """Cosine spacing should concentrate points at leading edge."""
        m, p, t = 0.0, 0.0, 0.12
        
        x_cosine, _ = generate_naca4(m, p, t, n_points=100, cosine_spacing=True)
        x_uniform, _ = generate_naca4(m, p, t, n_points=100, cosine_spacing=False)
        
        # Count points in first 10% of chord
        n_cosine_le = np.sum(x_cosine < 0.1)
        n_uniform_le = np.sum(x_uniform < 0.1)
        
        # Cosine spacing should have more points near LE
        assert n_cosine_le > n_uniform_le
    
    def test_naca_designation(self):
        """Test NACA designation string generation."""
        assert naca_designation(0.02, 0.4, 0.12) == "NACA2412"
        assert naca_designation(0.0, 0.0, 0.12) == "NACA0012"
        assert naca_designation(0.04, 0.4, 0.15) == "NACA4415"
    
    def test_parametric_sweep(self):
        """Test parametric sweep generation."""
        samples = generate_parametric_sweep(
            m_range=(0.01, 0.04),
            p_range=(0.2, 0.6),
            t_range=(0.08, 0.18),
            n_samples=10,
            seed=42
        )
        
        assert len(samples) == 10
        
        for sample in samples:
            assert 0.01 <= sample['m'] <= 0.04
            assert 0.2 <= sample['p'] <= 0.6
            assert 0.08 <= sample['t'] <= 0.18
            assert 'designation' in sample
    
    def test_parametric_sweep_reproducibility(self):
        """Same seed should produce same samples."""
        samples1 = generate_parametric_sweep(n_samples=5, seed=42)
        samples2 = generate_parametric_sweep(n_samples=5, seed=42)
        
        for s1, s2 in zip(samples1, samples2):
            assert s1['m'] == s2['m']
            assert s1['p'] == s2['p']
            assert s1['t'] == s2['t']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
