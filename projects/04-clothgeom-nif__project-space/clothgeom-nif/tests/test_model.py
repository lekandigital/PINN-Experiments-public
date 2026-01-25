#!/usr/bin/env python3
"""
Unit Tests for ClothGeom-NIF Model

Tests model architecture, forward pass, SDF properties,
latent interpolation, and mesh extraction.

Usage:
    python -m pytest tests/test_model.py -v
    python tests/test_model.py  # Run directly
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siren import SineActivation, SirenLinear, SirenLayer, SirenNetwork
from models.nif_decoder import NIFDecoder, create_nif_decoder


class TestSIREN(unittest.TestCase):
    """Tests for SIREN components."""
    
    def test_sine_activation(self):
        """Test sine activation with omega scaling."""
        act = SineActivation(omega_0=30.0)
        x = torch.linspace(-1, 1, 100)
        y = act(x)
        
        # Output should be bounded in [-1, 1]
        self.assertTrue(y.min() >= -1.0)
        self.assertTrue(y.max() <= 1.0)
        
        # Should be periodic
        self.assertTrue(torch.allclose(torch.sin(30 * x), y))
    
    def test_siren_linear_init(self):
        """Test SIREN linear layer initialization."""
        # First layer
        layer_first = SirenLinear(3, 256, omega_0=30.0, is_first=True)
        weights = layer_first.linear.weight.data
        
        # First layer should have uniform init in [-1/in, 1/in]
        bound = 1.0 / 3
        self.assertTrue(weights.min() >= -bound - 0.01)
        self.assertTrue(weights.max() <= bound + 0.01)
        
        # Hidden layer
        layer_hidden = SirenLinear(256, 256, omega_0=30.0, is_first=False)
        weights_h = layer_hidden.linear.weight.data
        
        # Hidden layers should have smaller init
        expected_bound = np.sqrt(6.0 / 256) / 30.0
        self.assertTrue(weights_h.abs().max() <= expected_bound + 0.01)
    
    def test_siren_layer_forward(self):
        """Test complete SIREN layer."""
        layer = SirenLayer(3, 256, omega_0=30.0, is_first=True)
        x = torch.randn(32, 3)
        y = layer(x)
        
        self.assertEqual(y.shape, (32, 256))
        self.assertTrue(torch.isfinite(y).all())
    
    def test_siren_network(self):
        """Test complete SIREN network."""
        net = SirenNetwork(
            in_features=3,
            hidden_features=128,
            out_features=1,
            num_layers=4
        )
        
        x = torch.randn(64, 3)
        y = net(x)
        
        self.assertEqual(y.shape, (64, 1))
        self.assertTrue(torch.isfinite(y).all())


class TestNIFDecoder(unittest.TestCase):
    """Tests for NIF decoder model."""
    
    def setUp(self):
        """Set up test model."""
        self.latent_dim = 128
        self.model = create_nif_decoder({
            'latent_dim': self.latent_dim,
            'hidden_dim': 128,  # Smaller for testing
            'num_layers': 4
        })
        self.device = torch.device('cpu')
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        batch_size = 32
        coords = torch.randn(batch_size, 3)
        latent = torch.randn(batch_size, self.latent_dim)
        
        sdf, variance = self.model(coords, latent)
        
        self.assertEqual(sdf.shape, (batch_size, 1))
        self.assertEqual(variance.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(sdf).all())
        self.assertTrue(torch.isfinite(variance).all())
    
    def test_forward_without_variance(self):
        """Test forward pass without variance computation."""
        coords = torch.randn(16, 3)
        latent = torch.randn(16, self.latent_dim)
        
        sdf, variance = self.model(coords, latent, return_variance=False)
        
        self.assertEqual(sdf.shape, (16, 1))
        self.assertIsNone(variance)
    
    def test_sdf_reasonable_range(self):
        """Test that SDF values are in reasonable range."""
        # Normalized coords in [-1, 1]
        coords = torch.rand(1000, 3) * 2 - 1
        latent = torch.randn(1000, self.latent_dim)
        latent = latent / latent.norm(dim=1, keepdim=True)
        
        sdf, _ = self.model(coords, latent)
        
        # SDF should mostly be in reasonable range
        # (depends on initialization, but shouldn't be huge)
        self.assertTrue(sdf.abs().mean() < 10.0)
    
    def test_variance_positive(self):
        """Test that variance output is always positive."""
        coords = torch.randn(100, 3)
        latent = torch.randn(100, self.latent_dim)
        
        _, variance = self.model(coords, latent)
        
        self.assertTrue((variance > 0).all())
    
    def test_gradient_computation(self):
        """Test SDF gradient computation."""
        coords = torch.randn(16, 3, requires_grad=True)
        latent = torch.randn(16, self.latent_dim)
        
        sdf, gradient = self.model.compute_gradient(coords, latent)
        
        self.assertEqual(gradient.shape, (16, 3))
        self.assertTrue(torch.isfinite(gradient).all())
        
        # Gradient norm should be reasonably close to 1 for good SDF
        grad_norm = gradient.norm(dim=1)
        # Untrained model won't have grad norm = 1, just check finite
        self.assertTrue(torch.isfinite(grad_norm).all())
    
    def test_batched_inference(self):
        """Test memory-efficient batched inference."""
        n_points = 10000
        coords = torch.randn(n_points, 3)
        latent = torch.randn(self.latent_dim)
        
        sdf, variance = self.model.forward_batched(
            coords, latent, batch_size=1000
        )
        
        self.assertEqual(sdf.shape, (n_points, 1))
        self.assertEqual(variance.shape, (n_points, 1))
    
    def test_config_roundtrip(self):
        """Test model config save/load."""
        config = self.model.get_config()
        
        # Create new model from config
        new_model = NIFDecoder.from_config(config)
        
        # Should have same architecture
        self.assertEqual(new_model.latent_dim, self.model.latent_dim)
        self.assertEqual(new_model.hidden_dim, self.model.hidden_dim)
        self.assertEqual(new_model.num_layers, self.model.num_layers)
    
    def test_no_nan_in_output(self):
        """Test that there are no NaN values in output."""
        # Test with various input magnitudes
        for scale in [0.01, 1.0, 10.0]:
            coords = torch.randn(32, 3) * scale
            latent = torch.randn(32, self.latent_dim)
            
            sdf, variance = self.model(coords, latent)
            
            self.assertFalse(torch.isnan(sdf).any(), f"NaN in SDF at scale {scale}")
            self.assertFalse(torch.isnan(variance).any(), f"NaN in variance at scale {scale}")
            self.assertFalse(torch.isinf(sdf).any(), f"Inf in SDF at scale {scale}")
            self.assertFalse(torch.isinf(variance).any(), f"Inf in variance at scale {scale}")


class TestLatentInterpolation(unittest.TestCase):
    """Tests for latent space interpolation."""
    
    def setUp(self):
        """Set up test model."""
        self.model = create_nif_decoder({
            'latent_dim': 128,
            'hidden_dim': 64,
            'num_layers': 3
        })
    
    def test_smooth_interpolation(self):
        """Test that interpolation between latents is smooth."""
        latent1 = torch.randn(128)
        latent2 = torch.randn(128)
        
        # Normalize
        latent1 = latent1 / latent1.norm()
        latent2 = latent2 / latent2.norm()
        
        # Sample point
        coord = torch.tensor([[0.0, 0.0, 0.0]])
        
        # Interpolate
        t_values = torch.linspace(0, 1, 10)
        sdf_values = []
        
        for t in t_values:
            latent_interp = (1 - t) * latent1 + t * latent2
            latent_interp = latent_interp / latent_interp.norm()  # Stay on sphere
            
            sdf, _ = self.model(coord, latent_interp.unsqueeze(0))
            sdf_values.append(sdf.item())
        
        sdf_values = np.array(sdf_values)
        
        # Check smoothness: differences between consecutive values should be small
        diffs = np.abs(np.diff(sdf_values))
        max_diff = diffs.max()
        
        # Untrained model may have large variations, just ensure no jumps > 10
        self.assertTrue(max_diff < 10.0, f"Large jump in interpolation: {max_diff}")


class TestMeshExtraction(unittest.TestCase):
    """Tests for mesh extraction."""
    
    def setUp(self):
        """Set up test model."""
        self.model = create_nif_decoder({
            'latent_dim': 128,
            'hidden_dim': 64,
            'num_layers': 3
        })
    
    def test_sdf_grid_evaluation(self):
        """Test SDF evaluation on grid."""
        from inference.mesh_extractor import MeshExtractor, MeshExtractionConfig
        
        config = MeshExtractionConfig(resolution=16)  # Low res for speed
        extractor = MeshExtractor(self.model, config=config)
        
        latent = torch.randn(128)
        sdf_volume = extractor.evaluate_sdf_grid(latent)
        
        self.assertEqual(sdf_volume.shape, (16, 16, 16))
        self.assertTrue(np.isfinite(sdf_volume).all())
    
    def test_mesh_extraction(self):
        """Test mesh extraction produces valid output."""
        try:
            from skimage import measure
        except ImportError:
            self.skipTest("scikit-image not available")
        
        from inference.mesh_extractor import MeshExtractor, MeshExtractionConfig
        
        config = MeshExtractionConfig(resolution=16)
        extractor = MeshExtractor(self.model, config=config)
        
        latent = torch.randn(128)
        vertices, faces, normals = extractor.extract_mesh(latent)
        
        # Should return arrays (may be empty if no surface)
        self.assertIsInstance(vertices, np.ndarray)
        self.assertIsInstance(faces, np.ndarray)
        
        if len(vertices) > 0:
            self.assertEqual(vertices.shape[1], 3)
            self.assertEqual(faces.shape[1], 3)


class TestModelFP16(unittest.TestCase):
    """Tests for FP16 compatibility."""
    
    def test_fp16_forward(self):
        """Test forward pass with FP16."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = create_nif_decoder({
            'latent_dim': 128,
            'hidden_dim': 64,
            'num_layers': 3
        }).cuda().half()
        
        coords = torch.randn(32, 3, dtype=torch.float16, device='cuda')
        latent = torch.randn(32, 128, dtype=torch.float16, device='cuda')
        
        sdf, variance = model(coords, latent)
        
        self.assertEqual(sdf.dtype, torch.float16)
        self.assertTrue(torch.isfinite(sdf).all())


def run_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSIREN))
    suite.addTests(loader.loadTestsFromTestCase(TestNIFDecoder))
    suite.addTests(loader.loadTestsFromTestCase(TestLatentInterpolation))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestModelFP16))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
