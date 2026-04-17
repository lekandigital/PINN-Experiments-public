"""
Tests for SIREN (Sinusoidal Representation Networks) implementation.

Tests cover:
- Forward pass output shapes
- Weight initialization distributions
- Gradient flow (backward pass)
- LatentConditionedSiren functionality
- Reproducibility with fixed seeds
"""

import math

import pytest
import torch
import torch.nn as nn

from implicit_fields.siren import (
    SineActivation,
    SirenLayer,
    SirenNetwork,
    LatentConditionedSiren,
    ModulatedSirenLayer,
)


class TestSineActivation:
    """Tests for SineActivation module."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        act = SineActivation(omega=30.0)
        x = torch.randn(100, 256)
        y = act(x)
        assert y.shape == x.shape
    
    def test_omega_scaling(self):
        """Test that omega scales the frequency."""
        x = torch.linspace(-1, 1, 100)
        
        act_low = SineActivation(omega=1.0)
        act_high = SineActivation(omega=30.0)
        
        y_low = act_low(x)
        y_high = act_high(x)
        
        # Higher omega should produce more oscillations
        # Count zero crossings as a proxy
        crossings_low = (y_low[:-1] * y_low[1:] < 0).sum()
        crossings_high = (y_high[:-1] * y_high[1:] < 0).sum()
        
        assert crossings_high > crossings_low
    
    def test_range(self):
        """Test that output is bounded in [-1, 1]."""
        act = SineActivation(omega=30.0)
        x = torch.randn(1000, 256) * 10  # Large inputs
        y = act(x)
        
        assert y.min() >= -1.0
        assert y.max() <= 1.0


class TestSirenLayer:
    """Tests for SirenLayer module."""
    
    def test_output_shape(self):
        """Test output dimensions."""
        layer = SirenLayer(3, 256, omega=30.0, is_first=True)
        x = torch.randn(100, 3)
        y = layer(x)
        assert y.shape == (100, 256)
    
    def test_first_layer_init(self):
        """Test first layer weight initialization: uniform[-1/n, 1/n]."""
        in_features = 3
        layer = SirenLayer(in_features, 256, omega=30.0, is_first=True)
        
        weights = layer.linear.weight.data.flatten()
        expected_bound = 1.0 / in_features
        
        # Check weights are within expected bounds
        assert weights.min() >= -expected_bound - 1e-6
        assert weights.max() <= expected_bound + 1e-6
        
        # Check distribution is roughly uniform (mean near 0, reasonable variance)
        assert abs(weights.mean()) < 0.1
        # Variance of uniform[-a,a] is a²/3
        expected_var = (expected_bound ** 2) / 3
        assert abs(weights.var() - expected_var) < expected_var * 0.5  # Within 50%
    
    def test_hidden_layer_init(self):
        """Test hidden layer weight initialization: uniform[-sqrt(6/n)/omega, sqrt(6/n)/omega]."""
        in_features = 256
        omega = 30.0
        layer = SirenLayer(in_features, 256, omega=omega, is_first=False)
        
        weights = layer.linear.weight.data.flatten()
        expected_bound = math.sqrt(6.0 / in_features) / omega
        
        # Check weights are within expected bounds
        assert weights.min() >= -expected_bound - 1e-6
        assert weights.max() <= expected_bound + 1e-6
        
        # Check variance
        expected_var = (expected_bound ** 2) / 3
        actual_var = weights.var().item()
        assert abs(actual_var - expected_var) < expected_var * 0.5
    
    def test_bias_init(self):
        """Test bias initialization matches weight initialization."""
        in_features = 3
        layer = SirenLayer(in_features, 256, omega=30.0, is_first=True)
        
        biases = layer.linear.bias.data
        expected_bound = 1.0 / in_features
        
        assert biases.min() >= -expected_bound - 1e-6
        assert biases.max() <= expected_bound + 1e-6
    
    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = SirenLayer(3, 256, omega=30.0, is_first=True)
        x = torch.randn(10, 3, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert x.grad is not None
        assert layer.linear.weight.grad is not None
        assert x.grad.abs().sum() > 0
        assert layer.linear.weight.grad.abs().sum() > 0


class TestSirenNetwork:
    """Tests for SirenNetwork module."""
    
    def test_output_shape_various_configs(self):
        """Test output shape for various configurations."""
        configs = [
            (3, 128, 3, 1),    # 3D SDF
            (4, 256, 5, 1),    # 4D SDF
            (3, 128, 3, 3),    # 3D displacement
            (8, 512, 4, 1),    # 8D input
        ]
        
        for in_f, hidden, layers, out_f in configs:
            model = SirenNetwork(in_f, hidden, layers, out_f)
            x = torch.randn(100, in_f)
            y = model(x)
            assert y.shape == (100, out_f), f"Failed for config {(in_f, hidden, layers, out_f)}"
    
    def test_parameter_count(self):
        """Test that parameter count is as expected."""
        model = SirenNetwork(
            in_features=3,
            hidden_features=128,
            hidden_layers=3,
            out_features=1,
        )
        
        # First layer: 3*128 + 128 = 512
        # Hidden 1: 128*128 + 128 = 16512
        # Hidden 2: 128*128 + 128 = 16512
        # Hidden 3: 128*128 + 128 = 16512
        # Final: 128*1 + 1 = 129
        # Total = 512 + 3*16512 + 129 = 50177
        expected = 3*128 + 128 + 3*(128*128 + 128) + 128*1 + 1
        actual = sum(p.numel() for p in model.parameters())
        assert actual == expected
    
    def test_omega_parameters(self):
        """Test different omega_0 and omega_hidden values."""
        model = SirenNetwork(
            in_features=3,
            hidden_features=128,
            hidden_layers=2,
            out_features=1,
            omega_0=60.0,  # Double default
            omega_hidden=1.0,  # Much smaller
        )
        
        # Check first layer uses omega_0
        first_layer = model.layers[0]
        assert first_layer.omega == 60.0
        
        # Check hidden layers use omega_hidden
        # layers[1] is second SirenLayer (index 1 in sequential)
        hidden_layer = model.layers[1]
        assert hidden_layer.omega == 1.0
    
    def test_final_activation(self):
        """Test custom final activation."""
        model = SirenNetwork(
            in_features=3,
            hidden_features=128,
            hidden_layers=2,
            out_features=1,
            final_activation=nn.Tanh(),
        )
        
        x = torch.randn(100, 3) * 10
        y = model(x)
        
        # Output should be bounded by tanh
        assert y.min() >= -1.0
        assert y.max() <= 1.0
    
    def test_dropout(self):
        """Test dropout is applied correctly."""
        model = SirenNetwork(
            in_features=3,
            hidden_features=128,
            hidden_layers=2,
            out_features=1,
            dropout=0.5,
        )
        
        model.train()
        x = torch.randn(100, 3)
        
        # Run multiple times - outputs should differ due to dropout
        y1 = model(x)
        y2 = model(x)
        assert not torch.allclose(y1, y2)
        
        # In eval mode, outputs should be deterministic
        model.eval()
        y3 = model(x)
        y4 = model(x)
        assert torch.allclose(y3, y4)
    
    def test_gradient_flow_full_network(self):
        """Test gradients flow through entire network."""
        model = SirenNetwork(3, 128, 3, 1)
        x = torch.randn(10, 3, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # Check all layers received gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    def test_forward_with_activations(self):
        """Test forward_with_activations returns intermediate values."""
        model = SirenNetwork(3, 128, 2, 1)
        x = torch.randn(10, 3)
        
        y, activations = model.forward_with_activations(x)
        
        # Should have: input, layer1, layer2, output
        assert len(activations) >= 3
        assert activations[0].shape == x.shape
        assert activations[-1].shape == y.shape
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        torch.manual_seed(42)
        model1 = SirenNetwork(3, 128, 2, 1)
        
        torch.manual_seed(42)
        model2 = SirenNetwork(3, 128, 2, 1)
        
        # Weights should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
        
        # Outputs should be identical
        x = torch.randn(10, 3)
        assert torch.allclose(model1(x), model2(x))


class TestLatentConditionedSiren:
    """Tests for LatentConditionedSiren module."""
    
    def test_output_shape(self):
        """Test output shape with latent conditioning."""
        model = LatentConditionedSiren(
            in_features=3,
            latent_dim=64,
            hidden_features=128,
            hidden_layers=3,
            out_features=1,
        )
        
        coords = torch.randn(100, 3)
        latent = torch.randn(100, 64)
        y = model(coords, latent)
        
        assert y.shape == (100, 1)
    
    def test_latent_dim_stored(self):
        """Test that coord_dim and latent_dim are stored correctly."""
        model = LatentConditionedSiren(
            in_features=3,
            latent_dim=64,
            hidden_features=128,
            hidden_layers=2,
            out_features=1,
        )
        
        assert model.coord_dim == 3
        assert model.latent_dim == 64
        assert model.in_features == 3 + 64
    
    def test_different_latents_different_outputs(self):
        """Test that different latent codes produce different outputs."""
        model = LatentConditionedSiren(
            in_features=3,
            latent_dim=64,
            hidden_features=128,
            hidden_layers=2,
            out_features=1,
        )
        
        coords = torch.randn(100, 3)
        latent1 = torch.randn(100, 64)
        latent2 = torch.randn(100, 64)
        
        y1 = model(coords, latent1)
        y2 = model(coords, latent2)
        
        assert not torch.allclose(y1, y2)
    
    def test_same_latent_same_output(self):
        """Test deterministic output with same inputs."""
        model = LatentConditionedSiren(
            in_features=3,
            latent_dim=64,
            hidden_features=128,
            hidden_layers=2,
            out_features=1,
        )
        model.eval()
        
        coords = torch.randn(100, 3)
        latent = torch.randn(100, 64)
        
        y1 = model(coords, latent)
        y2 = model(coords, latent)
        
        assert torch.allclose(y1, y2)


class TestModulatedSirenLayer:
    """Tests for ModulatedSirenLayer module."""
    
    def test_output_shape(self):
        """Test output shape."""
        layer = ModulatedSirenLayer(256, 256, omega=30.0)
        x = torch.randn(100, 256)
        y = layer(x)
        assert y.shape == (100, 256)
    
    def test_modulation_effect(self):
        """Test that gamma and beta affect output."""
        layer = ModulatedSirenLayer(256, 256, omega=30.0)
        x = torch.randn(100, 256)
        
        # No modulation
        y_base = layer(x)
        
        # Scale modulation
        gamma = torch.ones(100, 256) * 2.0
        y_scaled = layer(x, gamma=gamma)
        
        # Shift modulation
        beta = torch.ones(100, 256) * 0.5
        y_shifted = layer(x, beta=beta)
        
        assert not torch.allclose(y_base, y_scaled)
        assert not torch.allclose(y_base, y_shifted)
    
    def test_identity_modulation(self):
        """Test that gamma=1, beta=0 gives same result as no modulation."""
        layer = ModulatedSirenLayer(256, 256, omega=30.0)
        x = torch.randn(100, 256)
        
        y_base = layer(x)
        
        gamma = torch.ones(100, 256)
        beta = torch.zeros(100, 256)
        y_modulated = layer(x, gamma=gamma, beta=beta)
        
        assert torch.allclose(y_base, y_modulated)


class TestSirenInitializationStatistics:
    """Statistical tests for SIREN weight initialization."""
    
    def test_first_layer_uniform_distribution(self):
        """Statistical test that first layer weights follow uniform distribution."""
        # Create many layers and aggregate statistics
        all_weights = []
        in_features = 3
        
        for _ in range(100):
            layer = SirenLayer(in_features, 256, omega=30.0, is_first=True)
            all_weights.append(layer.linear.weight.data.flatten())
        
        weights = torch.cat(all_weights)
        bound = 1.0 / in_features
        
        # Check bounds
        assert weights.min() >= -bound - 1e-6
        assert weights.max() <= bound + 1e-6
        
        # Check mean is near zero
        assert abs(weights.mean()) < 0.01
        
        # Check variance matches uniform distribution
        expected_var = (bound ** 2) / 3
        actual_var = weights.var().item()
        assert abs(actual_var - expected_var) / expected_var < 0.1  # Within 10%
    
    def test_hidden_layer_uniform_distribution(self):
        """Statistical test for hidden layer initialization."""
        all_weights = []
        in_features = 256
        omega = 30.0
        
        for _ in range(100):
            layer = SirenLayer(in_features, 256, omega=omega, is_first=False)
            all_weights.append(layer.linear.weight.data.flatten())
        
        weights = torch.cat(all_weights)
        bound = math.sqrt(6.0 / in_features) / omega
        
        # Check bounds
        assert weights.min() >= -bound - 1e-6
        assert weights.max() <= bound + 1e-6
        
        # Check variance
        expected_var = (bound ** 2) / 3
        actual_var = weights.var().item()
        assert abs(actual_var - expected_var) / expected_var < 0.1


class TestSirenCPU:
    """Test SIREN works on CPU without CUDA."""
    
    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        model = SirenNetwork(3, 128, 2, 1)
        model = model.cpu()
        
        x = torch.randn(100, 3)
        y = model(x)
        
        assert y.device.type == 'cpu'
        assert y.shape == (100, 1)
    
    def test_cpu_backward(self):
        """Test backward pass on CPU."""
        model = SirenNetwork(3, 128, 2, 1)
        model = model.cpu()
        
        x = torch.randn(10, 3, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
