"""
Integration tests for GeoPINN modules.

Tests all major components can be imported and instantiated correctly.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_layers():
    """Test that all layer modules can be imported."""
    from geopinn.layers import (
        TangentMessagePassing,
        SpectralGraphConv,
        build_dec_operators,
        DECLaplacian,
        Atlas,
        ChartMLP
    )
    print("✓ All layer modules imported successfully")


def test_import_data():
    """Test that all data modules can be imported."""
    from geopinn.data import (
        generate_sphere_swe_data,
        sample_sphere_points,
        generate_shell_elasticity_data,
        generate_torus_reaction_diffusion_data
    )
    print("✓ All data modules imported successfully")


def test_import_training():
    """Test that all training modules can be imported."""
    from geopinn.training import (
        SpherePINNTrainer,
        MeshPINNTrainer,
        adaptive_refine
    )
    print("✓ All training modules imported successfully")


def test_sphere_sampling():
    """Test sphere point sampling."""
    from geopinn.data.sphere_swe import sample_sphere_points
    
    # Test different methods
    for method in ['fibonacci', 'random', 'grid']:
        points = sample_sphere_points(1000, method=method)
        assert points.shape == (1000, 3) or len(points) >= 900, f"Shape: {points.shape}"
        
        # Check points are on unit sphere
        norms = np.linalg.norm(points, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Not on sphere: {norms.min()}, {norms.max()}"
    
    print("✓ Sphere sampling works correctly")


def test_tangent_message_passing():
    """Test TangentMessagePassing layer."""
    import torch
    from geopinn.layers import TangentMessagePassing
    
    # Create small test graph on sphere
    N = 10
    E = 30
    
    # Random points on sphere
    points = torch.randn(N, 3)
    points = points / points.norm(dim=1, keepdim=True)
    
    # Normals (same as points for unit sphere)
    normals = points.clone()
    
    # Random edges
    src = torch.randint(0, N, (E,))
    tgt = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, tgt])
    
    # Random features
    node_feats = torch.randn(N, 16)
    
    # Create layer
    layer = TangentMessagePassing(in_features=16, out_features=32)
    
    # Forward pass
    out = layer(node_feats, points, normals, edge_index)
    
    assert out.shape == (N, 32), f"Output shape: {out.shape}"
    print("✓ TangentMessagePassing forward pass works")


def test_dec_operators():
    """Test DEC operator construction."""
    from geopinn.layers import build_dec_operators
    
    # Simple tetrahedron mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ], dtype=np.int32)
    
    # Build operators
    ops = build_dec_operators(vertices, faces, return_tensors=False)
    
    # Check B0 shape (E x V)
    n_edges = ops['B0'].shape[0]
    assert ops['B0'].shape == (n_edges, 4), f"B0 shape: {ops['B0'].shape}"
    
    # Check B1 shape (F x E)
    assert ops['B1'].shape == (4, n_edges), f"B1 shape: {ops['B1'].shape}"
    
    # Check Hodge operators
    assert len(ops['Hodge0']) == 4, f"Hodge0 length: {len(ops['Hodge0'])}"
    assert len(ops['Hodge1']) == n_edges, f"Hodge1 length: {len(ops['Hodge1'])}"
    assert len(ops['Hodge2']) == 4, f"Hodge2 length: {len(ops['Hodge2'])}"
    
    print(f"✓ DEC operators built correctly ({n_edges} edges)")


def test_spectral_conv():
    """Test spectral convolution layer."""
    import torch
    from geopinn.layers import SpectralGraphConv
    
    N = 50
    k = 10
    
    # Mock eigenpairs
    eigenvals = torch.linspace(0, 2, k)
    eigenvecs = torch.randn(N, k)
    eigenvecs = eigenvecs / eigenvecs.norm(dim=0, keepdim=True)
    
    # Create layer
    layer = SpectralGraphConv(
        eigenvals=eigenvals,
        eigenvecs=eigenvecs,
        in_channels=4,
        out_channels=8,
        learnable_filter=True
    )
    
    # Test forward
    x = torch.randn(N, 4)
    out = layer(x)
    
    assert out.shape == (N, 8), f"Output shape: {out.shape}"
    print("✓ SpectralGraphConv forward pass works")


def test_atlas():
    """Test Atlas chart construction."""
    from geopinn.layers import Atlas
    from geopinn.data import sample_sphere_points
    
    # Sample sphere points
    points = sample_sphere_points(500, method='fibonacci')
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Create atlas
    atlas = Atlas(points, normals, num_charts=6)
    
    assert atlas.num_charts == 6
    assert atlas.chart_centers.shape == (6, 3)
    assert atlas.chart_normals.shape == (6, 3)
    
    # Test projection
    test_points = points[:10]
    coords = atlas.project_to_chart(test_points, chart_idx=0)
    assert coords.shape == (10, 2), f"Projection shape: {coords.shape}"
    
    # Test blending weights
    weights = atlas.compute_chart_weights(test_points)
    assert weights.shape == (10, 6), f"Weights shape: {weights.shape}"
    assert np.allclose(weights.sum(axis=1), 1.0), "Weights don't sum to 1"
    
    print("✓ Atlas construction and operations work")


def test_sphere_swe_data():
    """Test SWE data generation."""
    from geopinn.data import generate_sphere_swe_data
    
    data = generate_sphere_swe_data(n_points=100)
    
    assert 'collocation_points' in data
    assert 'initial_condition' in data
    assert data['collocation_points'].shape == (100, 3)
    assert data['initial_condition'].shape == (100, 4)
    
    print("✓ SWE data generation works")


def test_shell_data():
    """Test shell elasticity data generation."""
    from geopinn.data import generate_shell_elasticity_data
    
    data = generate_shell_elasticity_data(n_radial=10, n_angular=10)
    
    assert 'collocation_points' in data
    assert 'faces' in data
    assert 'solution_displacement' in data
    
    n_vertices = data['collocation_points'].shape[0]
    assert data['solution_displacement'].shape == (n_vertices, 3)
    
    print(f"✓ Shell data generation works ({n_vertices} vertices)")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("GeoPINN-Manifold Integration Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Import Layers", test_import_layers),
        ("Import Data", test_import_data),
        ("Import Training", test_import_training),
        ("Sphere Sampling", test_sphere_sampling),
        ("Tangent Message Passing", test_tangent_message_passing),
        ("DEC Operators", test_dec_operators),
        ("Spectral Convolution", test_spectral_conv),
        ("Atlas", test_atlas),
        ("SWE Data", test_sphere_swe_data),
        ("Shell Data", test_shell_data),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        print("-" * 40)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
