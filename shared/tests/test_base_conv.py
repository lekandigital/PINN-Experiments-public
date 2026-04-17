"""
Unit tests for PhysicsEncodedConv base class.

Tests:
    - Physics method has no learnable parameters
    - Correction MLP initializes near zero
    - Physics fraction tracking works correctly
    - Gradient flow through both branches
    - Aggregation modes work correctly
"""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.physics_conv.base import (
    PhysicsEncodedConv,
    PhysicsEncodedConvBase,
    PhysicsConvConfig,
    CorrectionMLP,
    get_activation,
)


class SimplePhysicsConv(PhysicsEncodedConvBase):
    """Simple concrete implementation for testing."""
    
    def __init__(self, config=None):
        super().__init__(config or PhysicsConvConfig())
        # Build correction MLP: x_i(3) + x_j(3) + edge(1) + physics(3) = 10
        self._build_correction_mlp(10)
    
    def compute_edge_physics(self, x_i, x_j, edge_attr=None, **kwargs):
        """Simple physics: difference vector (like spring direction)."""
        return x_j - x_i
    
    def physics_name(self):
        return "Simple Difference"


class TestCorrectionMLP:
    """Tests for the CorrectionMLP helper class."""
    
    def test_output_shape(self):
        """Test that MLP produces correct output shape."""
        mlp = CorrectionMLP(input_dim=10, output_dim=3, hidden_dim=32, num_layers=2)
        x = torch.randn(100, 10)
        out = mlp(x)
        assert out.shape == (100, 3)
    
    def test_initializes_near_zero(self):
        """Test that initial outputs are near zero."""
        mlp = CorrectionMLP(
            input_dim=10, 
            output_dim=3, 
            hidden_dim=32, 
            num_layers=2,
            scale_init=0.01
        )
        
        # Multiple random inputs
        for _ in range(10):
            x = torch.randn(100, 10)
            out = mlp(x)
            
            # Output should be small (< 0.1 on average)
            assert out.abs().mean() < 0.5, f"Initial output too large: {out.abs().mean()}"
    
    def test_final_bias_is_zero(self):
        """Test that final layer bias is initialized to zero."""
        mlp = CorrectionMLP(input_dim=10, output_dim=3)
        final_layer = mlp.net[-1]
        assert torch.allclose(final_layer.bias, torch.zeros_like(final_layer.bias))


class TestPhysicsEncodedConvBase:
    """Tests for the base PhysicsEncodedConv class."""
    
    def test_physics_has_no_learnable_params(self):
        """Verify compute_edge_physics doesn't add parameters."""
        conv = SimplePhysicsConv()
        
        # Count params before
        params_before = sum(p.numel() for p in conv.parameters())
        
        # Call physics computation
        x_i = torch.randn(50, 3)
        x_j = torch.randn(50, 3)
        physics_msg = conv.compute_edge_physics(x_i, x_j)
        
        # Count params after
        params_after = sum(p.numel() for p in conv.parameters())
        
        assert params_before == params_after, "compute_edge_physics added parameters!"
    
    def test_correction_initializes_near_zero(self):
        """At initialization, correction should be much smaller than physics."""
        conv = SimplePhysicsConv()
        
        # Create test data
        x_i = torch.randn(100, 3)
        x_j = torch.randn(100, 3)
        edge_attr = torch.randn(100, 1)
        
        physics_msg = conv.compute_edge_physics(x_i, x_j)
        correction_msg = conv.compute_edge_correction(x_i, x_j, edge_attr, physics_msg)
        
        physics_norm = physics_msg.norm()
        correction_norm = correction_msg.norm()
        
        # Correction should be < 10% of physics initially
        assert correction_norm < 0.1 * physics_norm, \
            f"Correction too large at init: {correction_norm:.3f} vs physics {physics_norm:.3f}"
    
    def test_physics_fraction_at_init(self):
        """physics_fraction() should return ~1.0 at initialization."""
        conv = SimplePhysicsConv()
        conv.train()
        
        # Run a forward pass
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 1)
        
        _ = conv(x, edge_index, edge_attr)
        
        # Check physics fraction
        pf = conv.physics_fraction()
        assert pf > 0.9, f"Physics fraction at init should be >0.9, got {pf:.3f}"
    
    def test_combine_modes(self):
        """Test both additive and multiplicative combination."""
        physics_msg = torch.randn(100, 3)
        correction_msg = torch.randn(100, 3) * 0.1  # Small correction
        
        # Additive
        config_add = PhysicsConvConfig(combine_mode="additive")
        conv_add = SimplePhysicsConv(config_add)
        combined_add = conv_add.combine_physics_and_correction(physics_msg, correction_msg)
        expected_add = physics_msg + correction_msg
        assert torch.allclose(combined_add, expected_add)
        
        # Multiplicative
        config_mul = PhysicsConvConfig(combine_mode="multiplicative")
        conv_mul = SimplePhysicsConv(config_mul)
        combined_mul = conv_mul.combine_physics_and_correction(physics_msg, correction_msg)
        expected_mul = physics_msg * (1.0 + correction_msg)
        assert torch.allclose(combined_mul, expected_mul)
    
    def test_gradient_flow(self):
        """Gradients should flow through both physics and correction branches."""
        conv = SimplePhysicsConv()
        
        # Input that requires grad
        x = torch.randn(20, 3, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 1)
        
        # Forward
        out = conv(x, edge_index, edge_attr)
        loss = out.sum()
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "No gradient on input x"
        assert not torch.all(x.grad == 0), "Gradient on x is all zeros"
        
        # Check correction MLP has gradients
        for name, param in conv.correction_mlp.named_parameters():
            assert param.grad is not None, f"No gradient on {name}"
    
    def test_aggregation_sum(self):
        """Test sum aggregation."""
        config = PhysicsConvConfig(aggregation="sum")
        conv = SimplePhysicsConv(config)
        
        messages = torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
        index = torch.tensor([0, 0, 1])  # Two to node 0, one to node 1
        
        result = conv.aggregate(messages, index, dim_size=2)
        
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([3.0, 0, 0]))  # 1+2
        assert torch.allclose(result[1], torch.tensor([3.0, 0, 0]))  # 3
    
    def test_aggregation_mean(self):
        """Test mean aggregation."""
        config = PhysicsConvConfig(aggregation="mean")
        conv = SimplePhysicsConv(config)
        
        messages = torch.tensor([[1.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]])
        index = torch.tensor([0, 0, 1])
        
        result = conv.aggregate(messages, index, dim_size=2)
        
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([2.0, 0, 0]))  # (1+3)/2
        assert torch.allclose(result[1], torch.tensor([6.0, 0, 0]))  # 6/1
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        conv = SimplePhysicsConv()
        
        num_nodes = 50
        num_edges = 150
        
        x = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 1)
        
        out = conv(x, edge_index, edge_attr)
        
        assert out.shape == (num_nodes, 3)


class TestActivations:
    """Test activation function helper."""
    
    def test_known_activations(self):
        """Test that known activations are returned correctly."""
        assert isinstance(get_activation("silu"), nn.SiLU)
        assert isinstance(get_activation("relu"), nn.ReLU)
        assert isinstance(get_activation("tanh"), nn.Tanh)
        assert isinstance(get_activation("gelu"), nn.GELU)
        assert isinstance(get_activation("elu"), nn.ELU)
    
    def test_case_insensitive(self):
        """Test that activation lookup is case-insensitive."""
        assert isinstance(get_activation("SILU"), nn.SiLU)
        assert isinstance(get_activation("ReLU"), nn.ReLU)
    
    def test_default_fallback(self):
        """Test that unknown activations fall back to SiLU."""
        assert isinstance(get_activation("unknown"), nn.SiLU)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
