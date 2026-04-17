"""
Tests for Fourier Feature encoding implementation.

Tests cover:
- Output dimensionality
- Deterministic encoding with fixed seed
- include_input toggle
- FourierFeatureMLP end-to-end
"""

import math

import pytest
import torch

from implicit_fields.fourier_features import (
    FourierFeatureEncoding,
    FourierFeatureMLP,
    MultiScaleFourierFeatures,
)


class TestFourierFeatureEncoding:
    """Tests for FourierFeatureEncoding module."""
    
    def test_output_dim_with_input(self):
        """Test output dimensionality with include_input=True."""
        in_features = 3
        num_frequencies = 128
        
        encoder = FourierFeatureEncoding(
            in_features=in_features,
            num_frequencies=num_frequencies,
            include_input=True,
        )
        
        # Expected: in_features + 2 * num_frequencies
        expected_dim = in_features + 2 * num_frequencies
        assert encoder.output_dim == expected_dim
        
        x = torch.randn(100, in_features)
        encoded = encoder(x)
        assert encoded.shape == (100, expected_dim)
    
    def test_output_dim_without_input(self):
        """Test output dimensionality with include_input=False."""
        in_features = 3
        num_frequencies = 128
        
        encoder = FourierFeatureEncoding(
            in_features=in_features,
            num_frequencies=num_frequencies,
            include_input=False,
        )
        
        # Expected: 2 * num_frequencies (no raw input)
        expected_dim = 2 * num_frequencies
        assert encoder.output_dim == expected_dim
        
        x = torch.randn(100, in_features)
        encoded = encoder(x)
        assert encoded.shape == (100, expected_dim)
    
    def test_deterministic_with_seed(self):
        """Test that encoding is deterministic with fixed seed."""
        torch.manual_seed(42)
        encoder1 = FourierFeatureEncoding(3, 64, scale=10.0)
        
        torch.manual_seed(42)
        encoder2 = FourierFeatureEncoding(3, 64, scale=10.0)
        
        # B matrices should be identical
        assert torch.allclose(encoder1.B, encoder2.B)
        
        # Encodings should be identical
        x = torch.randn(100, 3)
        assert torch.allclose(encoder1(x), encoder2(x))
    
    def test_different_seeds_different_encoding(self):
        """Test that different seeds produce different encodings."""
        torch.manual_seed(42)
        encoder1 = FourierFeatureEncoding(3, 64, scale=10.0)
        
        torch.manual_seed(123)
        encoder2 = FourierFeatureEncoding(3, 64, scale=10.0)
        
        # B matrices should differ
        assert not torch.allclose(encoder1.B, encoder2.B)
    
    def test_scale_affects_frequencies(self):
        """Test that scale parameter affects frequency distribution."""
        torch.manual_seed(42)
        encoder_low = FourierFeatureEncoding(3, 64, scale=1.0)
        
        torch.manual_seed(42)
        encoder_high = FourierFeatureEncoding(3, 64, scale=100.0)
        
        # Higher scale should have larger B values
        assert encoder_high.B.abs().mean() > encoder_low.B.abs().mean() * 10
    
    def test_learnable_frequencies(self):
        """Test learnable frequency matrix."""
        encoder = FourierFeatureEncoding(3, 64, learnable=True)
        
        # B should be a parameter
        assert isinstance(encoder.B, torch.nn.Parameter)
        
        # Should have gradients during backward
        x = torch.randn(10, 3)
        encoded = encoder(x)
        loss = encoded.sum()
        loss.backward()
        
        assert encoder.B.grad is not None
    
    def test_non_learnable_frequencies(self):
        """Test non-learnable (buffer) frequency matrix."""
        encoder = FourierFeatureEncoding(3, 64, learnable=False)
        
        # B should be a buffer, not parameter
        assert not isinstance(encoder.B, torch.nn.Parameter)
        assert 'B' in dict(encoder.named_buffers())
    
    def test_output_bounded(self):
        """Test that sin/cos outputs are bounded."""
        encoder = FourierFeatureEncoding(3, 128, scale=10.0, include_input=False)
        
        x = torch.randn(1000, 3) * 10  # Large inputs
        encoded = encoder(x)
        
        # Sin and cos are bounded in [-1, 1]
        assert encoded.min() >= -1.0
        assert encoded.max() <= 1.0
    
    def test_various_input_dims(self):
        """Test encoding works for various input dimensions."""
        for in_dim in [1, 2, 3, 4, 8, 16]:
            encoder = FourierFeatureEncoding(in_dim, 64)
            x = torch.randn(50, in_dim)
            encoded = encoder(x)
            
            expected_dim = in_dim + 2 * 64
            assert encoded.shape == (50, expected_dim), f"Failed for in_dim={in_dim}"


class TestFourierFeatureMLP:
    """Tests for FourierFeatureMLP module."""
    
    def test_output_shape(self):
        """Test output shape."""
        model = FourierFeatureMLP(
            in_features=4,
            hidden_features=256,
            hidden_layers=4,
            out_features=1,
        )
        
        x = torch.randn(100, 4)
        y = model(x)
        assert y.shape == (100, 1)
    
    def test_different_activations(self):
        """Test various activation functions."""
        activations = ['relu', 'gelu', 'tanh', 'leaky_relu', 'elu']
        
        for act_name in activations:
            model = FourierFeatureMLP(
                in_features=3,
                hidden_features=64,
                hidden_layers=2,
                out_features=1,
                activation=act_name,
            )
            
            x = torch.randn(10, 3)
            y = model(x)
            assert y.shape == (10, 1), f"Failed for activation={act_name}"
    
    def test_sine_activation_hybrid(self):
        """Test hybrid Fourier + SIREN with sine activation."""
        model = FourierFeatureMLP(
            in_features=3,
            hidden_features=64,
            hidden_layers=2,
            out_features=1,
            activation='sine',
        )
        
        x = torch.randn(10, 3)
        y = model(x)
        assert y.shape == (10, 1)
    
    def test_gradient_flow(self):
        """Test gradients flow through network."""
        model = FourierFeatureMLP(
            in_features=3,
            hidden_features=64,
            hidden_layers=2,
            out_features=1,
        )
        
        x = torch.randn(10, 3, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_dropout(self):
        """Test dropout affects output variance."""
        model = FourierFeatureMLP(
            in_features=3,
            hidden_features=64,
            hidden_layers=2,
            out_features=1,
            dropout=0.5,
        )
        
        model.train()
        x = torch.randn(100, 3)
        
        # Multiple forward passes should differ
        y1 = model(x)
        y2 = model(x)
        assert not torch.allclose(y1, y2)
    
    def test_learnable_frequencies_in_mlp(self):
        """Test MLP with learnable Fourier frequencies."""
        model = FourierFeatureMLP(
            in_features=3,
            hidden_features=64,
            hidden_layers=2,
            out_features=1,
            learnable_frequencies=True,
        )
        
        # B should be a parameter
        assert isinstance(model.encoding.B, torch.nn.Parameter)


class TestMultiScaleFourierFeatures:
    """Tests for MultiScaleFourierFeatures module."""
    
    def test_output_dim(self):
        """Test output dimensionality."""
        in_features = 3
        num_freq_per_scale = 32
        scales = [1.0, 4.0, 16.0]
        
        encoder = MultiScaleFourierFeatures(
            in_features=in_features,
            num_frequencies_per_scale=num_freq_per_scale,
            scales=scales,
            include_input=True,
        )
        
        # Expected: in_features + 2 * num_freq_per_scale * num_scales
        expected = in_features + 2 * num_freq_per_scale * len(scales)
        assert encoder.output_dim == expected
        
        x = torch.randn(100, in_features)
        encoded = encoder(x)
        assert encoded.shape == (100, expected)
    
    def test_default_scales(self):
        """Test default scale values."""
        encoder = MultiScaleFourierFeatures(in_features=3)
        
        # Default scales are [1.0, 2.0, 4.0, 8.0]
        assert encoder.scales == [1.0, 2.0, 4.0, 8.0]
    
    def test_custom_scales(self):
        """Test custom scale values."""
        custom_scales = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        encoder = MultiScaleFourierFeatures(
            in_features=3,
            scales=custom_scales,
        )
        
        assert encoder.scales == custom_scales


class TestFourierFeaturesNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_inputs(self):
        """Test encoding with large input values."""
        encoder = FourierFeatureEncoding(3, 64, scale=10.0)
        
        # Large inputs
        x = torch.randn(100, 3) * 1000
        encoded = encoder(x)
        
        # Should not have NaN or Inf
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_small_inputs(self):
        """Test encoding with very small inputs."""
        encoder = FourierFeatureEncoding(3, 64, scale=10.0)
        
        x = torch.randn(100, 3) * 1e-6
        encoded = encoder(x)
        
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_zero_inputs(self):
        """Test encoding with zero inputs."""
        encoder = FourierFeatureEncoding(3, 64, scale=10.0, include_input=True)
        
        x = torch.zeros(100, 3)
        encoded = encoder(x)
        
        assert not torch.isnan(encoded).any()
        
        # First 3 elements should be zero (raw input)
        assert torch.allclose(encoded[:, :3], x)


class TestFourierFeaturesCPU:
    """Test Fourier features work on CPU."""
    
    def test_cpu_encoding(self):
        """Test encoding on CPU."""
        encoder = FourierFeatureEncoding(3, 64)
        encoder = encoder.cpu()
        
        x = torch.randn(100, 3)
        encoded = encoder(x)
        
        assert encoded.device.type == 'cpu'
    
    def test_cpu_mlp(self):
        """Test MLP on CPU."""
        model = FourierFeatureMLP(3, 64, 2, 1)
        model = model.cpu()
        
        x = torch.randn(100, 3)
        y = model(x)
        
        assert y.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
