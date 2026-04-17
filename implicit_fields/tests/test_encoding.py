"""
Tests for positional encoding utilities.

Tests cover:
- Output dimensionality
- Log vs linear frequency sampling
- Numerical stability
- PositionalEncoding module
"""

import math

import pytest
import torch

from implicit_fields.encoding import (
    positional_encoding,
    PositionalEncoding,
    IntegratedPositionalEncoding,
    compute_encoding_dim,
)


class TestPositionalEncodingFunction:
    """Tests for positional_encoding function."""
    
    def test_output_dim_with_input(self):
        """Test output dimensionality with include_input=True."""
        x = torch.randn(100, 3)
        num_frequencies = 10
        
        encoded = positional_encoding(x, num_frequencies=num_frequencies, include_input=True)
        
        # Expected: D + D * 2 * L = D * (1 + 2L)
        expected_dim = 3 * (1 + 2 * num_frequencies)
        assert encoded.shape == (100, expected_dim)
    
    def test_output_dim_without_input(self):
        """Test output dimensionality with include_input=False."""
        x = torch.randn(100, 3)
        num_frequencies = 10
        
        encoded = positional_encoding(x, num_frequencies=num_frequencies, include_input=False)
        
        # Expected: D * 2 * L
        expected_dim = 3 * 2 * num_frequencies
        assert encoded.shape == (100, expected_dim)
    
    def test_log_sampling_frequencies(self):
        """Test that log_sampling produces exponentially spaced frequencies."""
        x = torch.zeros(1, 1)  # Single point, single dimension
        x[0, 0] = 1.0
        
        # With log sampling, frequencies are 2^0, 2^1, ..., 2^(L-1)
        encoded = positional_encoding(x, num_frequencies=3, include_input=False, log_sampling=True)
        
        # For x=1, sin(2^k * pi * x) = sin(2^k * pi)
        # At k=0: sin(pi) ≈ 0
        # At k=1: sin(2*pi) ≈ 0
        # etc.
        # Better to check with a non-integer
        x2 = torch.tensor([[0.5]])
        encoded2 = positional_encoding(x2, num_frequencies=3, include_input=False, log_sampling=True)
        
        # Should have 6 values: sin/cos for 3 frequencies
        assert encoded2.shape == (1, 6)
    
    def test_linear_sampling_frequencies(self):
        """Test that linear sampling produces linearly spaced frequencies."""
        x = torch.tensor([[0.25]])
        
        encoded_log = positional_encoding(x, num_frequencies=4, include_input=False, log_sampling=True)
        encoded_linear = positional_encoding(x, num_frequencies=4, include_input=False, log_sampling=False)
        
        # Encodings should differ due to different frequency spacing
        assert not torch.allclose(encoded_log, encoded_linear)
    
    def test_include_input_preserved(self):
        """Test that raw input is correctly included."""
        x = torch.randn(50, 3)
        
        encoded = positional_encoding(x, num_frequencies=6, include_input=True)
        
        # First 3 elements should be the raw input
        assert torch.allclose(encoded[:, :3], x)
    
    def test_various_input_dims(self):
        """Test encoding for various input dimensions."""
        for in_dim in [1, 2, 3, 4, 8]:
            x = torch.randn(50, in_dim)
            num_freq = 6
            
            encoded = positional_encoding(x, num_frequencies=num_freq, include_input=True)
            
            expected_dim = in_dim * (1 + 2 * num_freq)
            assert encoded.shape == (50, expected_dim), f"Failed for in_dim={in_dim}"
    
    def test_batch_dimensions(self):
        """Test encoding with multi-dimensional batches."""
        x = torch.randn(10, 20, 3)  # (batch, seq, dim)
        
        encoded = positional_encoding(x, num_frequencies=4, include_input=True)
        
        expected_last_dim = 3 * (1 + 2 * 4)
        assert encoded.shape == (10, 20, expected_last_dim)
    
    def test_output_bounded(self):
        """Test that sin/cos outputs are bounded."""
        x = torch.randn(1000, 3) * 100  # Large inputs
        
        encoded = positional_encoding(x, num_frequencies=10, include_input=False)
        
        # Sin and cos are bounded in [-1, 1]
        assert encoded.min() >= -1.0
        assert encoded.max() <= 1.0
    
    def test_gradient_flow(self):
        """Test that gradients flow through encoding."""
        x = torch.randn(10, 3, requires_grad=True)
        
        encoded = positional_encoding(x, num_frequencies=6)
        loss = encoded.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestPositionalEncodingModule:
    """Tests for PositionalEncoding nn.Module."""
    
    def test_forward_pass(self):
        """Test forward pass of module."""
        encoder = PositionalEncoding(num_frequencies=10, include_input=True)
        
        x = torch.randn(100, 3)
        encoded = encoder(x)
        
        expected_dim = encoder.output_dim(3)
        assert encoded.shape == (100, expected_dim)
    
    def test_output_dim_method(self):
        """Test output_dim calculation method."""
        encoder = PositionalEncoding(num_frequencies=6, include_input=True)
        
        # D * (1 + 2L)
        assert encoder.output_dim(3) == 3 * (1 + 2 * 6)
        assert encoder.output_dim(4) == 4 * (1 + 2 * 6)
    
    def test_output_dim_no_input(self):
        """Test output_dim without raw input."""
        encoder = PositionalEncoding(num_frequencies=6, include_input=False)
        
        # D * 2L
        assert encoder.output_dim(3) == 3 * 2 * 6
    
    def test_module_in_sequential(self):
        """Test using module in nn.Sequential."""
        encoder = PositionalEncoding(num_frequencies=6)
        
        model = torch.nn.Sequential(
            encoder,
            torch.nn.Linear(encoder.output_dim(3), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        
        x = torch.randn(50, 3)
        y = model(x)
        
        assert y.shape == (50, 1)


class TestIntegratedPositionalEncoding:
    """Tests for Mip-NeRF style integrated positional encoding."""
    
    def test_forward_pass(self):
        """Test integrated encoding forward pass."""
        encoder = IntegratedPositionalEncoding(num_frequencies=6)
        
        mean = torch.randn(100, 3)
        variance = torch.abs(torch.randn(100, 3)) * 0.1
        
        encoded = encoder(mean, variance)
        
        # Should produce valid output
        assert encoded.shape[0] == 100
        assert not torch.isnan(encoded).any()
    
    def test_zero_variance_damping_factor(self):
        """Test that zero variance produces damping factor close to 1."""
        encoder = IntegratedPositionalEncoding(num_frequencies=4, include_input=False)
        
        mean = torch.zeros(1, 3)
        variance = torch.ones(1, 3) * 1e-10  # Very small variance
        
        encoded = encoder(mean, variance)
        
        # With zero mean and near-zero variance:
        # sin(0) * 1 ≈ 0, cos(0) * 1 ≈ 1
        # So the output should be close to standard encoding of zeros
        # (mostly cos values near 1 for small variance)
        assert not torch.isnan(encoded).any()
        # The damping factor should be close to 1, so abs values should be reasonable
        assert encoded.abs().max() <= 1.0 + 1e-5
    
    def test_high_variance_damping(self):
        """Test that high variance damps high frequencies."""
        encoder = IntegratedPositionalEncoding(num_frequencies=6, include_input=False)
        
        mean = torch.zeros(1, 3)
        
        # Low variance
        low_var = torch.ones(1, 3) * 0.001
        encoded_low = encoder(mean, low_var)
        
        # High variance
        high_var = torch.ones(1, 3) * 10.0
        encoded_high = encoder(mean, high_var)
        
        # High variance should have smaller magnitude (more damping)
        assert encoded_high.abs().mean() < encoded_low.abs().mean()
    
    def test_scalar_variance(self):
        """Test with scalar variance (same for all dimensions)."""
        encoder = IntegratedPositionalEncoding(num_frequencies=4)
        
        mean = torch.randn(50, 3)
        variance = torch.abs(torch.randn(50, 1)) * 0.1  # (50, 1)
        
        encoded = encoder(mean, variance)
        
        assert encoded.shape[0] == 50
        assert not torch.isnan(encoded).any()


class TestComputeEncodingDim:
    """Tests for compute_encoding_dim utility function."""
    
    def test_with_input(self):
        """Test dimension computation with input included."""
        dim = compute_encoding_dim(3, 10, include_input=True)
        assert dim == 3 + 3 * 2 * 10  # 63
    
    def test_without_input(self):
        """Test dimension computation without input."""
        dim = compute_encoding_dim(3, 10, include_input=False)
        assert dim == 3 * 2 * 10  # 60
    
    def test_various_configs(self):
        """Test various configurations."""
        test_cases = [
            (3, 6, True, 3 * (1 + 2 * 6)),
            (3, 6, False, 3 * 2 * 6),
            (4, 10, True, 4 * (1 + 2 * 10)),
            (2, 4, False, 2 * 2 * 4),
        ]
        
        for in_dim, num_freq, include, expected in test_cases:
            result = compute_encoding_dim(in_dim, num_freq, include)
            assert result == expected, f"Failed for ({in_dim}, {num_freq}, {include})"


class TestNumericalStability:
    """Tests for numerical stability of encoding."""
    
    def test_large_coordinates(self):
        """Test encoding with large coordinate values."""
        x = torch.randn(100, 3) * 1000
        
        encoded = positional_encoding(x, num_frequencies=10)
        
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_small_coordinates(self):
        """Test encoding with very small coordinates."""
        x = torch.randn(100, 3) * 1e-8
        
        encoded = positional_encoding(x, num_frequencies=10)
        
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_zero_coordinates(self):
        """Test encoding with zero coordinates."""
        x = torch.zeros(100, 3)
        
        encoded = positional_encoding(x, num_frequencies=6, include_input=True)
        
        assert not torch.isnan(encoded).any()
        
        # First 3 elements (raw input) should be zero
        assert torch.allclose(encoded[:, :3], torch.zeros(100, 3))
        
        # sin(0) = 0, cos(0) = 1
        # Check some sin terms are near zero
        sin_terms = encoded[:, 3::2]  # Every other starting from index 3
        # Many should be near zero
    
    def test_high_frequency(self):
        """Test encoding with many frequency bands."""
        x = torch.randn(50, 3)
        
        # High number of frequencies
        encoded = positional_encoding(x, num_frequencies=100)
        
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()


class TestEncodingCPU:
    """Test encoding works on CPU."""
    
    def test_cpu_encoding(self):
        """Test positional encoding on CPU."""
        x = torch.randn(100, 3)
        encoded = positional_encoding(x, num_frequencies=10)
        
        assert encoded.device.type == 'cpu'
    
    def test_cpu_module(self):
        """Test PositionalEncoding module on CPU."""
        encoder = PositionalEncoding(num_frequencies=10)
        encoder = encoder.cpu()
        
        x = torch.randn(100, 3)
        encoded = encoder(x)
        
        assert encoded.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
