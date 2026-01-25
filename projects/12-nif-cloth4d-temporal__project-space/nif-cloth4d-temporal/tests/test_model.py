"""
Unit tests for FourierFeatureMLP model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from src.models import FourierFeatureMLP, SIRENLayer, SIRENNetwork, TemporalGRU
from src.models.fourier_mlp import FourierFeatureEmbedding


class TestFourierFeatureEmbedding:
    """Tests for Fourier feature embedding."""
    
    def test_output_shape(self):
        """Test output dimension is correct."""
        num_freqs = 16
        in_dim = 4
        
        embed = FourierFeatureEmbedding(in_dim=in_dim, num_freqs=num_freqs)
        
        x = torch.randn(32, in_dim)
        out = embed(x)
        
        assert out.shape == (32, num_freqs * 2)
    
    def test_output_range(self):
        """Test output is bounded by sin/cos range."""
        embed = FourierFeatureEmbedding(in_dim=4, num_freqs=16)
        
        x = torch.randn(100, 4)
        out = embed(x)
        
        assert out.min() >= -1.0
        assert out.max() <= 1.0
    
    def test_deterministic(self):
        """Test that embedding is deterministic."""
        embed = FourierFeatureEmbedding(in_dim=4, num_freqs=16)
        
        x = torch.randn(10, 4)
        out1 = embed(x)
        out2 = embed(x)
        
        assert torch.allclose(out1, out2)


class TestSIRENLayer:
    """Tests for SIREN layer."""
    
    def test_output_shape(self):
        """Test output dimension."""
        layer = SIRENLayer(in_features=32, out_features=64)
        
        x = torch.randn(16, 32)
        out = layer(x)
        
        assert out.shape == (16, 64)
    
    def test_sine_activation(self):
        """Test that output is bounded by sine range."""
        layer = SIRENLayer(in_features=32, out_features=64)
        
        x = torch.randn(100, 32)
        out = layer(x)
        
        assert out.min() >= -1.0
        assert out.max() <= 1.0
    
    def test_initialization_first_layer(self):
        """Test first layer initialization bounds."""
        in_features = 32
        layer = SIRENLayer(in_features=in_features, out_features=64, is_first=True)
        
        # First layer should have weights in [-1/in, 1/in]
        bound = 1.0 / in_features
        assert layer.linear.weight.min() >= -bound - 0.01  # Small tolerance
        assert layer.linear.weight.max() <= bound + 0.01


class TestSIRENNetwork:
    """Tests for complete SIREN network."""
    
    def test_output_shape(self):
        """Test network output shape."""
        net = SIRENNetwork(
            in_features=4,
            hidden_features=64,
            out_features=1,
            num_layers=3,
        )
        
        x = torch.randn(32, 4)
        out = net(x)
        
        assert out.shape == (32, 1)
    
    def test_gradient_computation(self):
        """Test gradient can be computed."""
        net = SIRENNetwork(
            in_features=4,
            hidden_features=64,
            out_features=1,
            num_layers=3,
        )
        
        x = torch.randn(32, 4, requires_grad=True)
        out, grad = net.forward_with_gradient(x)
        
        assert out.shape == (32, 1)
        assert grad.shape == (32, 4)


class TestTemporalGRU:
    """Tests for temporal GRU module."""
    
    def test_output_shape(self):
        """Test output and hidden state shapes."""
        gru = TemporalGRU(input_dim=64, hidden_dim=32)
        
        x = torch.randn(16, 64)
        out, hidden = gru(x)
        
        assert out.shape == (16, 64)  # Same as input (residual)
        assert hidden.shape == (1, 16, 32)  # (num_layers, batch, hidden)
    
    def test_hidden_state_persistence(self):
        """Test that hidden state is properly passed."""
        gru = TemporalGRU(input_dim=64, hidden_dim=32)
        
        x1 = torch.randn(16, 64)
        x2 = torch.randn(16, 64)
        
        # First pass
        out1, hidden1 = gru(x1)
        
        # Second pass with hidden state
        out2, hidden2 = gru(x2, hidden1)
        
        # Hidden states should differ
        assert not torch.allclose(hidden1, hidden2)
    
    def test_init_hidden(self):
        """Test hidden state initialization."""
        gru = TemporalGRU(input_dim=64, hidden_dim=32)
        
        hidden = gru.init_hidden(batch_size=8, device=torch.device('cpu'))
        
        assert hidden.shape == (1, 8, 32)
        assert (hidden == 0).all()


class TestFourierFeatureMLP:
    """Tests for main FourierFeatureMLP model."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        model = FourierFeatureMLP(
            hidden_dim=64,
            num_layers=3,
            use_gru=False,
        )
        
        xyz = torch.randn(32, 3)
        t = torch.randn(32, 1)
        
        out, hidden = model(xyz, t)
        
        assert out.shape == (32, 1)
        assert hidden is None  # No GRU
    
    def test_forward_with_gru(self):
        """Test forward pass with GRU."""
        model = FourierFeatureMLP(
            hidden_dim=64,
            num_layers=3,
            use_gru=True,
            gru_hidden=32,
        )
        
        xyz = torch.randn(32, 3)
        t = torch.randn(32, 1)
        
        out, hidden = model(xyz, t)
        
        assert out.shape == (32, 1)
        assert hidden is not None
        assert hidden.shape[1] == 32  # Batch size
    
    def test_temporal_rollout(self):
        """Test multiple timestep rollout with GRU."""
        model = FourierFeatureMLP(
            hidden_dim=64,
            num_layers=3,
            use_gru=True,
            gru_hidden=32,
        )
        
        xyz = torch.randn(16, 3)
        hidden = None
        
        outputs = []
        for frame in range(10):
            t = torch.full((16, 1), frame / 30.0)
            out, hidden = model(xyz, t, hidden)
            outputs.append(out)
        
        # Outputs should vary across time (GRU effect)
        assert not torch.allclose(outputs[0], outputs[-1])
    
    def test_gradient_computation(self):
        """Test gradient computation for normals."""
        model = FourierFeatureMLP(
            hidden_dim=64,
            num_layers=3,
            use_gru=False,
        )
        
        xyz = torch.randn(32, 3)
        t = torch.randn(32, 1)
        
        sdf, gradient, _ = model.compute_gradient(xyz, t)
        
        assert sdf.shape == (32, 1)
        assert gradient.shape == (32, 3)
    
    def test_parameter_count(self):
        """Test parameter count method."""
        model = FourierFeatureMLP(
            hidden_dim=256,
            num_layers=5,
            use_gru=True,
        )
        
        count = model.count_parameters()
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_mixed_precision(self):
        """Test model works with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        model = FourierFeatureMLP(hidden_dim=64, num_layers=3).to(device)
        
        xyz = torch.randn(32, 3, device=device)
        t = torch.randn(32, 1, device=device)
        
        with torch.cuda.amp.autocast():
            out, _ = model(xyz, t)
        
        assert out.dtype == torch.float16 or out.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
