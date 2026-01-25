#!/usr/bin/env python3
"""
Unit Tests for ClothGeom-NIF Training Pipeline

Tests data loading, loss computation, gradient flow,
and checkpoint handling.

Usage:
    python -m pytest tests/test_training.py -v
    python tests/test_training.py  # Run directly
"""

import sys
import os
import unittest
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_nif_decoder
from data.synthetic_cloth_generator import SyntheticClothGenerator, generate_dataset
from data.cloth_dataset import ClothSDFDataset, create_dataloader


class TestDataGeneration(unittest.TestCase):
    """Tests for synthetic data generation."""
    
    def setUp(self):
        """Set up generator."""
        self.generator = SyntheticClothGenerator(
            sdf_resolution=32,
            seed=42
        )
    
    def test_rest_mesh_generation(self):
        """Test rest mesh creation."""
        nodes = self.generator.create_rest_mesh()
        
        # Check shape: 32x32 grid = 1024 nodes
        self.assertEqual(nodes.shape, (1024, 3))
        
        # Should be mostly flat (Z ≈ 0)
        self.assertTrue(np.allclose(nodes[:, 2], 0))
    
    def test_deformation_application(self):
        """Test mesh deformation."""
        rest_nodes = self.generator.create_rest_mesh()
        deformed = self.generator.apply_deformation(rest_nodes, 'sine')
        
        # Should have same shape
        self.assertEqual(deformed.shape, rest_nodes.shape)
        
        # Should be different from rest
        self.assertFalse(np.allclose(deformed, rest_nodes))
        
        # Z values should have variation
        self.assertTrue(deformed[:, 2].std() > 0.01)
    
    def test_strain_computation(self):
        """Test edge strain calculation."""
        rest_nodes = self.generator.create_rest_mesh()
        deformed = self.generator.apply_deformation(rest_nodes, 'sine')
        strains = self.generator.compute_edge_strains(deformed, rest_nodes)
        
        # Should have correct shape
        self.assertEqual(strains.shape[0], self.generator.num_edges)
        self.assertEqual(strains.shape[1], 1)
        
        # Strains should be mostly reasonable (< 5.0 for synthetic deformations)
        self.assertTrue(np.abs(strains).mean() < 5.0)
    
    def test_latent_generation(self):
        """Test latent code creation."""
        rest_nodes = self.generator.create_rest_mesh()
        deformed = self.generator.apply_deformation(rest_nodes)
        strains = self.generator.compute_edge_strains(deformed, rest_nodes)
        
        latent = self.generator.nodes_to_latent(deformed, strains, latent_dim=128)
        
        self.assertEqual(latent.shape, (128,))
        self.assertTrue(np.isfinite(latent).all())
        
        # Should be normalized
        norm = np.linalg.norm(latent)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_sdf_computation(self):
        """Test SDF volume generation."""
        rest_nodes = self.generator.create_rest_mesh()
        deformed = self.generator.apply_deformation(rest_nodes)
        
        sdf = self.generator.compute_sdf(deformed)
        
        # Check shape
        self.assertEqual(sdf.shape, (32, 32, 32))
        
        # Should have both positive and negative values (surface exists)
        self.assertTrue(sdf.min() < 0)
        self.assertTrue(sdf.max() > 0)
    
    def test_sample_generation(self):
        """Test complete sample generation."""
        sample = self.generator.generate_sample(latent_dim=128)
        
        # Check all keys exist
        self.assertIn('nodes', sample)
        self.assertIn('strains', sample)
        self.assertIn('latent', sample)
        self.assertIn('sdf', sample)
        
        # Check dtypes
        self.assertEqual(sample['latent'].dtype, np.float32)
        self.assertEqual(sample['sdf'].dtype, np.float32)


class TestDataset(unittest.TestCase):
    """Tests for PyTorch dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Create temporary dataset."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_path = os.path.join(cls.temp_dir, 'test_dataset.h5')
        
        # Generate small dataset
        generate_dataset(
            cls.data_path,
            num_samples=5,
            sdf_resolution=32,
            latent_dim=64,
            grid_size=16,
            seed=42
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dataset_loading(self):
        """Test dataset initialization."""
        dataset = ClothSDFDataset(
            self.data_path,
            num_points_per_sample=100,
            cache_in_memory=True
        )
        
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.latent_dim, 64)
        self.assertEqual(dataset.sdf_resolution, 32)
    
    def test_dataset_getitem(self):
        """Test dataset sample retrieval."""
        dataset = ClothSDFDataset(
            self.data_path,
            num_points_per_sample=100,
            augment=False
        )
        
        sample = dataset[0]
        
        self.assertIn('coords', sample)
        self.assertIn('latent', sample)
        self.assertIn('sdf', sample)
        
        self.assertEqual(sample['coords'].shape, (100, 3))
        self.assertEqual(sample['latent'].shape[0], 64)
        self.assertEqual(sample['sdf'].shape, (100, 1))
    
    def test_surface_sampling(self):
        """Test importance sampling near surface."""
        dataset = ClothSDFDataset(
            self.data_path,
            num_points_per_sample=1000,
            surface_ratio=0.8,
            augment=False
        )
        
        sample = dataset[0]
        
        # Get full volume for comparison
        full = dataset.get_full_volume(0)
        
        # Points should be valid coordinates
        self.assertTrue((sample['coords'] >= -1.5).all())
        self.assertTrue((sample['coords'] <= 1.5).all())
    
    def test_dataloader(self):
        """Test DataLoader creation."""
        loader = create_dataloader(
            self.data_path,
            batch_size=2,
            num_workers=0,
            num_points=50
        )
        
        batch = next(iter(loader))
        
        # Batch should have expected shapes
        self.assertEqual(batch['coords'].shape[0], 2)  # batch size
        self.assertEqual(batch['coords'].shape[1], 50)  # num points
        self.assertEqual(batch['coords'].shape[2], 3)  # xyz


class TestLossFunction(unittest.TestCase):
    """Tests for loss computation."""
    
    def test_sdf_loss(self):
        """Test SDF loss computation."""
        # Import from train.py
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train import SDFLoss
        
        criterion = SDFLoss(
            lambda_sdf=1.0,
            lambda_eikonal=0.1,
            lambda_variance=0.01
        )
        
        pred_sdf = torch.randn(100, 1)
        gt_sdf = torch.randn(100, 1)
        pred_var = torch.abs(torch.randn(100, 1)) + 0.1
        gradient = torch.randn(100, 3)
        
        losses = criterion(pred_sdf, gt_sdf, pred_var, gradient)
        
        self.assertIn('total', losses)
        self.assertIn('sdf', losses)
        self.assertIn('eikonal', losses)
        self.assertIn('variance', losses)
        
        # All losses should be non-negative
        for key, value in losses.items():
            self.assertTrue(value >= 0, f"{key} loss is negative")
    
    def test_eikonal_loss_zero_for_unit_gradient(self):
        """Test Eikonal loss is zero when gradient magnitude is 1."""
        from train import SDFLoss
        
        criterion = SDFLoss(lambda_eikonal=1.0)
        
        # Create unit-norm gradients
        gradient = torch.randn(100, 3)
        gradient = gradient / gradient.norm(dim=1, keepdim=True)
        
        pred_sdf = torch.zeros(100, 1)
        gt_sdf = torch.zeros(100, 1)
        
        losses = criterion(pred_sdf, gt_sdf, None, gradient)
        
        # Eikonal loss should be very small
        self.assertTrue(losses['eikonal'] < 1e-5)
    
    def test_loss_gradient_flow(self):
        """Test that gradients flow through loss."""
        from train import SDFLoss
        
        model = create_nif_decoder({
            'latent_dim': 64,
            'hidden_dim': 32,
            'num_layers': 2
        })
        
        criterion = SDFLoss()
        
        coords = torch.randn(16, 3, requires_grad=True)
        latent = torch.randn(16, 64)
        gt_sdf = torch.randn(16, 1)
        
        pred_sdf, pred_var = model(coords, latent)
        losses = criterion(pred_sdf, gt_sdf, pred_var, None)
        
        # Backward should work
        losses['total'].backward()
        
        # Model should have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)
        
        # No NaN gradients
        for p in model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())


class TestCheckpointing(unittest.TestCase):
    """Tests for model checkpointing."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        model = create_nif_decoder({
            'latent_dim': 64,
            'hidden_dim': 32,
            'num_layers': 2
        })
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, 'test.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model.get_config()
        }, checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        new_model = create_nif_decoder(checkpoint['model_config'])
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check outputs match
        test_input = (torch.randn(4, 3), torch.randn(4, 64))
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            out1, _ = model(*test_input)
            out2, _ = new_model(*test_input)
        
        self.assertTrue(torch.allclose(out1, out2))


class TestTrainingStep(unittest.TestCase):
    """Tests for training step."""
    
    def test_single_training_step(self):
        """Test one forward-backward step."""
        model = create_nif_decoder({
            'latent_dim': 64,
            'hidden_dim': 32,
            'num_layers': 2
        })
        
        from train import SDFLoss
        criterion = SDFLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        coords = torch.randn(32, 3)
        latent = torch.randn(32, 64)
        gt_sdf = torch.randn(32, 1)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        pred_sdf, pred_var = model(coords, latent)
        losses = criterion(pred_sdf, gt_sdf, pred_var, None)
        
        initial_loss = losses['total'].item()
        
        losses['total'].backward()
        optimizer.step()
        
        # Loss should be computable and finite
        self.assertTrue(np.isfinite(initial_loss))
    
    def test_training_reduces_loss(self):
        """Test that training reduces loss over multiple steps."""
        model = create_nif_decoder({
            'latent_dim': 64,
            'hidden_dim': 64,
            'num_layers': 3
        })
        
        from train import SDFLoss
        criterion = SDFLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Fixed target (model should learn this)
        coords = torch.randn(64, 3)
        latent = torch.randn(64, 64)
        gt_sdf = torch.zeros(64, 1)  # Target: all zeros
        
        initial_loss = None
        final_loss = None
        
        model.train()
        for step in range(50):
            optimizer.zero_grad()
            pred_sdf, pred_var = model(coords, latent)
            losses = criterion(pred_sdf, gt_sdf, pred_var, None)
            losses['total'].backward()
            optimizer.step()
            
            if step == 0:
                initial_loss = losses['total'].item()
            if step == 49:
                final_loss = losses['total'].item()
        
        # Loss should decrease
        self.assertTrue(final_loss < initial_loss)


def run_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointing))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingStep))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
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
