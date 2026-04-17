"""
Tests for NIFDecoder and inverse design components.

Ported from Project 04 (ClothGeom-NIF) and adapted for P09.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNIFDecoder:
    """Tests for the NIFDecoder model."""
    
    def test_import(self):
        """NIFDecoder can be imported."""
        from src.models import NIFDecoder, create_nif_decoder
        assert NIFDecoder is not None
        assert create_nif_decoder is not None
    
    def test_create_default(self):
        """Can create NIFDecoder with default config."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder()
        assert decoder.latent_dim == 128
        assert decoder.hidden_dim == 256
        assert decoder.num_layers == 8
    
    def test_create_custom(self):
        """Can create NIFDecoder with custom config."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({
            'latent_dim': 64,
            'hidden_dim': 128,
            'num_layers': 4,
        })
        assert decoder.latent_dim == 64
        assert decoder.hidden_dim == 128
        assert decoder.num_layers == 4
    
    def test_forward_shape(self):
        """Forward pass produces correct output shapes."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({'latent_dim': 64, 'hidden_dim': 128, 'num_layers': 4})
        
        coords = torch.randn(100, 3)
        latent = torch.randn(100, 64)
        
        sdf, var = decoder(coords, latent)
        
        assert sdf.shape == (100, 1)
        assert var.shape == (100, 1)
    
    def test_forward_no_variance(self):
        """Forward pass can skip variance computation."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({'latent_dim': 64, 'hidden_dim': 128, 'num_layers': 4})
        
        coords = torch.randn(100, 3)
        latent = torch.randn(100, 64)
        
        sdf, var = decoder(coords, latent, return_variance=False)
        
        assert sdf.shape == (100, 1)
        assert var is None
    
    def test_forward_with_time(self):
        """Forward pass works with time input."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({
            'latent_dim': 64,
            'hidden_dim': 128,
            'num_layers': 4,
            'use_time': True,
        })
        
        coords = torch.randn(100, 3)
        latent = torch.randn(100, 64)
        time = torch.randn(100, 1)
        
        sdf, var = decoder(coords, latent, time)
        
        assert sdf.shape == (100, 1)
        assert var.shape == (100, 1)
    
    def test_compute_gradient(self):
        """Gradient computation produces correct shapes."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({'latent_dim': 64, 'hidden_dim': 128, 'num_layers': 4})
        
        coords = torch.randn(10, 3)
        latent = torch.randn(10, 64)
        
        sdf, gradient = decoder.compute_gradient(coords, latent)
        
        assert sdf.shape == (10, 1)
        assert gradient.shape == (10, 3)
    
    def test_batched_forward(self):
        """Batched forward pass works correctly."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({'latent_dim': 64, 'hidden_dim': 128, 'num_layers': 4})
        
        coords = torch.randn(1000, 3)
        latent = torch.randn(64)  # Single latent, will be expanded
        
        sdf, var = decoder.forward_batched(coords, latent, batch_size=256)
        
        assert sdf.shape == (1000, 1)
        assert var.shape == (1000, 1)
    
    def test_get_config(self):
        """Can retrieve model configuration."""
        from src.models import NIFDecoder
        decoder = NIFDecoder(latent_dim=64, hidden_dim=128, num_layers=4)
        
        config = decoder.get_config()
        
        assert config['latent_dim'] == 64
        assert config['hidden_dim'] == 128
        assert config['num_layers'] == 4
    
    def test_from_config(self):
        """Can create model from config dict."""
        from src.models import NIFDecoder
        
        config = {
            'latent_dim': 64,
            'hidden_dim': 128,
            'num_layers': 4,
        }
        
        decoder = NIFDecoder.from_config(config)
        
        assert decoder.latent_dim == 64
        assert decoder.hidden_dim == 128
    
    def test_skip_connection(self):
        """Skip connection is applied at correct layer."""
        from src.models import NIFDecoder
        
        decoder = NIFDecoder(latent_dim=64, hidden_dim=128, num_layers=8, use_skip_connection=True)
        assert decoder.skip_layer == 4  # num_layers // 2
        
        decoder_no_skip = NIFDecoder(latent_dim=64, hidden_dim=128, num_layers=8, use_skip_connection=False)
        assert decoder_no_skip.skip_layer == -1
    
    def test_variance_positive(self):
        """Variance output is always positive."""
        from src.models import create_nif_decoder
        decoder = create_nif_decoder({'latent_dim': 64, 'hidden_dim': 128, 'num_layers': 4})
        
        coords = torch.randn(100, 3)
        latent = torch.randn(100, 64)
        
        _, var = decoder(coords, latent)
        
        assert (var > 0).all()


class TestInverseDesign:
    """Tests for inverse design optimization."""
    
    def test_import(self):
        """Inverse design components can be imported."""
        from src.inverse import InverseDesignConfig, PoseMatchingOptimizer
        assert InverseDesignConfig is not None
        assert PoseMatchingOptimizer is not None
    
    def test_config_defaults(self):
        """InverseDesignConfig has sensible defaults."""
        from src.inverse import InverseDesignConfig
        config = InverseDesignConfig()
        
        assert config.num_steps == 500
        assert config.lr == 0.02
        assert config.num_points == 8192
        assert config.optimizer == 'adam'
    
    def test_config_custom(self):
        """Can create custom InverseDesignConfig."""
        from src.inverse import InverseDesignConfig
        config = InverseDesignConfig(num_steps=100, lr=0.01)
        
        assert config.num_steps == 100
        assert config.lr == 0.01
    
    def test_interpolate_latents_linear(self):
        """Linear latent interpolation works."""
        from src.inverse import interpolate_latents
        
        latent_a = torch.zeros(1, 64)
        latent_b = torch.ones(1, 64)
        
        latents = interpolate_latents(latent_a, latent_b, n_steps=5, method='linear')
        
        assert len(latents) == 5
        assert torch.allclose(latents[0], latent_a)
        assert torch.allclose(latents[-1], latent_b)
        assert torch.allclose(latents[2], torch.full((1, 64), 0.5))  # Midpoint
    
    def test_interpolate_latents_spherical(self):
        """Spherical latent interpolation works."""
        from src.inverse import interpolate_latents
        
        latent_a = torch.randn(1, 64)
        latent_b = torch.randn(1, 64)
        
        latents = interpolate_latents(latent_a, latent_b, n_steps=5, method='spherical')
        
        assert len(latents) == 5


class TestMeshExtractor:
    """Tests for MeshExtractor integration."""
    
    def test_import_from_implicit_fields(self):
        """MeshExtractor can be imported from implicit_fields."""
        from implicit_fields import MeshExtractor, MeshExtractionConfig
        assert MeshExtractor is not None
        assert MeshExtractionConfig is not None
    
    def test_config_defaults(self):
        """MeshExtractionConfig has sensible defaults."""
        from implicit_fields import MeshExtractionConfig
        config = MeshExtractionConfig()
        
        assert config.resolution == 128
        assert config.iso_level == 0.0
        assert config.device == 'cuda'
    
    def test_config_custom(self):
        """Can create custom MeshExtractionConfig."""
        from implicit_fields import MeshExtractionConfig
        config = MeshExtractionConfig(resolution=64, smooth_iterations=3, device='cpu')
        
        assert config.resolution == 64
        assert config.smooth_iterations == 3
        assert config.device == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
