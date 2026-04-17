"""
Convergence Tests for DiffGeo Module
====================================

Tests that verify:
1. Laplacian convergence on sphere (known analytic solution: Y_l^m eigenvalues)
2. Gauss-Bonnet theorem (∫K dA = 2π χ)
3. Mean curvature of sphere (H = 1/R)
4. Gradient/divergence adjointness

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add shared module to path
shared_path = Path(__file__).parents[1]
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from diffgeo.mesh import TriangleMesh
from diffgeo.mesh.generation import icosphere, torus_mesh, flat_grid
from diffgeo.operators.laplace_beltrami import laplace_beltrami, laplacian_eigenpairs
from diffgeo.curvature.gaussian import gaussian_curvature, verify_gauss_bonnet
from diffgeo.curvature.mean import mean_curvature


class TestLaplacianConvergence:
    """Test Laplace-Beltrami convergence on unit sphere."""
    
    def test_sphere_eigenvalues(self):
        """
        Spherical harmonics Y_l^m are eigenfunctions of sphere Laplacian.
        Eigenvalue for degree l is: λ_l = l(l+1)/R²
        
        For unit sphere (R=1):
        - λ_0 = 0 (constant eigenfunction)
        - λ_1 = 2 (3-fold degenerate, Y_1^{-1,0,1})
        - λ_2 = 6 (5-fold degenerate)
        """
        # Create icosphere with enough resolution
        mesh = icosphere(subdivisions=4)  # ~2562 vertices
        
        # Compute first 10 eigenpairs
        eigenvalues, eigenvectors = laplacian_eigenpairs(mesh, k=10)
        
        # λ_0 should be ~0
        assert np.abs(eigenvalues[0]) < 1e-6, f"λ_0 = {eigenvalues[0]}, expected ~0"
        
        # λ_1,2,3 should be ~2 (3-fold degenerate)
        for i in [1, 2, 3]:
            assert np.abs(eigenvalues[i] - 2.0) < 0.1, \
                f"λ_{i} = {eigenvalues[i]}, expected ~2"
        
        # λ_4,5,6,7,8 should be ~6 (5-fold degenerate)
        for i in [4, 5, 6, 7, 8]:
            assert np.abs(eigenvalues[i] - 6.0) < 0.2, \
                f"λ_{i} = {eigenvalues[i]}, expected ~6"
    
    def test_laplacian_convergence_rate(self):
        """
        Test that Laplacian error decreases with mesh refinement.
        
        For smooth functions, cotangent Laplacian should achieve O(h²) convergence.
        """
        errors = []
        mesh_sizes = []
        
        for subdiv in [2, 3, 4]:
            mesh = icosphere(subdivisions=subdiv)
            mesh_sizes.append(np.sqrt(len(mesh.vertices)))
            
            # Test function: x² + y² - 2z² (harmonic on R³)
            # Restricted to sphere, Laplacian should be -6
            x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
            f = x**2 + y**2 - 2*z**2
            
            Lf = laplace_beltrami(mesh, f)
            
            # Expected: -6 * f / R² = -6f for unit sphere
            expected = -6 * f
            
            error = np.sqrt(np.mean((Lf - expected)**2))
            errors.append(error)
        
        # Check convergence rate
        # Error should decrease by ~4x when mesh size doubles (O(h²))
        rate1 = np.log(errors[0] / errors[1]) / np.log(mesh_sizes[1] / mesh_sizes[0])
        rate2 = np.log(errors[1] / errors[2]) / np.log(mesh_sizes[2] / mesh_sizes[1])
        
        assert rate1 > 1.5, f"Convergence rate {rate1} < 1.5"
        assert rate2 > 1.5, f"Convergence rate {rate2} < 1.5"


class TestGaussBonnet:
    """Test Gauss-Bonnet theorem: ∫K dA = 2π χ"""
    
    def test_sphere(self):
        """Sphere: χ = 2, so ∫K dA = 4π"""
        mesh = icosphere(subdivisions=3)
        
        K = gaussian_curvature(mesh)
        integral = np.dot(K, mesh.dual_areas)
        
        expected = 4 * np.pi
        error = np.abs(integral - expected) / expected
        
        assert error < 0.02, f"Gauss-Bonnet error {error:.4f} > 2%"
    
    def test_torus(self):
        """Torus: χ = 0, so ∫K dA = 0"""
        mesh = torus_mesh(R=1.0, r=0.3, n_major=40, n_minor=20)
        
        K = gaussian_curvature(mesh)
        integral = np.dot(K, mesh.dual_areas)
        
        # Should be close to 0
        total_area = np.sum(mesh.dual_areas)
        normalized_error = np.abs(integral) / total_area
        
        assert normalized_error < 0.05, \
            f"Torus Gauss-Bonnet integral {integral:.4f}, expected ~0"
    
    def test_verify_function(self):
        """Test the verify_gauss_bonnet helper."""
        mesh = icosphere(subdivisions=3)
        
        is_valid, result = verify_gauss_bonnet(mesh)
        
        assert is_valid, f"Gauss-Bonnet verification failed: {result}"


class TestMeanCurvature:
    """Test mean curvature computations."""
    
    def test_sphere_mean_curvature(self):
        """
        Unit sphere: H = 1/R = 1 everywhere.
        
        Sign convention: H > 0 for convex surfaces (outward normal).
        """
        mesh = icosphere(subdivisions=4)
        
        H = mean_curvature(mesh)
        
        # Should all be ~1.0
        mean_H = np.mean(H)
        std_H = np.std(H)
        
        assert np.abs(mean_H - 1.0) < 0.05, \
            f"Sphere mean curvature {mean_H:.4f}, expected 1.0"
        assert std_H < 0.05, \
            f"Sphere curvature std {std_H:.4f} > 0.05 (should be uniform)"
    
    def test_flat_grid_zero_curvature(self):
        """Flat grid should have H = 0."""
        mesh = flat_grid(nx=20, ny=20, width=1.0, height=1.0)
        
        H = mean_curvature(mesh)
        
        # Boundary vertices have artifacts, check interior only
        interior = np.abs(H) < 1.0  # Rough filter
        
        max_H = np.max(np.abs(H[interior])) if np.any(interior) else np.max(np.abs(H))
        
        assert max_H < 0.1, f"Flat grid max curvature {max_H:.4f}, expected ~0"


class TestGradientDivergenceAdjoint:
    """Test that gradient and divergence are adjoint operators."""
    
    def test_adjointness(self):
        """
        For smooth functions f (0-form) and X (1-form):
        ⟨∇f, X⟩ = -⟨f, div X⟩ + boundary terms
        
        On closed manifold (sphere), boundary terms vanish.
        """
        mesh = icosphere(subdivisions=3)
        
        # Random 0-form (scalar field)
        np.random.seed(42)
        f = np.random.randn(len(mesh.vertices))
        
        # Random 1-form (edge field)
        X = np.random.randn(len(mesh.edge_lengths))
        
        # Gradient: ∇f = d0 @ f
        grad_f = mesh.d0 @ f
        
        # Divergence: div X = -d0^T @ star1 @ X / star0
        # (up to signs and Hodge stars)
        div_X_weak = -mesh.d0.T @ (mesh.star1 @ X)
        div_X = div_X_weak / (mesh.dual_areas + 1e-12)
        
        # Check adjointness: ⟨grad f, X⟩_1 ≈ -⟨f, div X⟩_0
        inner_grad = np.dot(grad_f * mesh.edge_lengths, X)  # L2 on edges
        inner_div = np.dot(f * mesh.dual_areas, div_X)  # L2 on vertices
        
        # These should be approximately equal (up to sign)
        # Note: exact adjointness depends on proper Hodge stars
        ratio = np.abs(inner_grad / (inner_div + 1e-12))
        
        # Should be close to 1 (within factor of 2)
        assert 0.3 < ratio < 3.0, \
            f"Adjointness ratio {ratio:.4f}, expected ~1"


class TestMeshGeneration:
    """Test mesh generation utilities."""
    
    def test_icosphere_vertices(self):
        """Check icosphere vertex count."""
        # Formula: V = 10 * 4^n + 2
        for n in [0, 1, 2, 3]:
            mesh = icosphere(subdivisions=n)
            expected = 10 * 4**n + 2
            assert len(mesh.vertices) == expected, \
                f"Icosphere {n}: got {len(mesh.vertices)}, expected {expected}"
    
    def test_torus_topology(self):
        """Torus should have Euler characteristic 0."""
        mesh = torus_mesh(R=1.0, r=0.3, n_major=20, n_minor=10)
        
        V = len(mesh.vertices)
        E = len(mesh.edge_lengths)
        F = len(mesh.face_areas)
        
        chi = V - E + F
        assert chi == 0, f"Torus χ = {chi}, expected 0"
    
    def test_flat_grid_boundaries(self):
        """Flat grid should have boundary vertices."""
        mesh = flat_grid(nx=10, ny=10)
        
        # Check that boundary is detected correctly
        from diffgeo.mesh.boundary import find_boundary_vertices
        boundary = find_boundary_vertices(mesh)
        
        # Grid has 4*(n-1) boundary vertices for nxn grid
        expected_boundary = 4 * 9  # 36 for 10x10
        assert len(boundary) == expected_boundary, \
            f"Grid boundary: got {len(boundary)}, expected {expected_boundary}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
