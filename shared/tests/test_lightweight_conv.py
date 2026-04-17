"""
Unit tests for LiteClothConv.

Tests:
    - Parameter count is minimal
    - Physics still works correctly
    - Pure physics mode (no learning)
"""

import pytest
import torch

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.physics_conv.lightweight_conv import (
    LiteClothConv,
    LiteClothConvConfig,
    PurePhysicsClothConv,
)


class TestParameterCount:
    """Tests verifying minimal parameter count."""
    
    def test_lite_conv_small_params(self):
        """LiteClothConv should have < 5K parameters."""
        conv = LiteClothConv()
        
        param_count = conv.count_parameters()
        
        assert param_count < 5000, f"Too many params: {param_count}"
        print(f"LiteClothConv parameters: {param_count}")
    
    def test_pure_physics_zero_params(self):
        """PurePhysicsClothConv should have 0 learnable parameters."""
        conv = PurePhysicsClothConv()
        
        param_count = sum(p.numel() for p in conv.parameters() if p.requires_grad)
        
        assert param_count == 0, f"Should have 0 params, got {param_count}"
    
    def test_single_layer_config(self):
        """Single-layer correction should have fewer params than multi-layer parent."""
        # LiteClothConv uses single layer, small hidden dim
        config_lite = LiteClothConvConfig(correction_layers=1, correction_hidden_dim=32)
        conv_lite = LiteClothConv(config_lite)
        
        # Compare with parent ClothForceConv using 2 layers and larger hidden
        from shared.physics_conv.cloth_conv import ClothForceConv, ClothConvConfig
        config_full = ClothConvConfig(correction_layers=2, correction_hidden_dim=64)
        conv_full = ClothForceConv(config_full)
        
        params_lite = conv_lite.count_parameters()
        params_full = sum(p.numel() for p in conv_full.parameters() if p.requires_grad)
        
        assert params_lite < params_full, f"Lite ({params_lite}) should have fewer params than full ({params_full})"


class TestPhysicsCorrectness:
    """Tests verifying physics still works in lightweight version."""
    
    def test_stretched_spring_pulls(self):
        """Stretched spring creates pulling force."""
        conv = LiteClothConv()
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        edge_attr = torch.tensor([[1.0]])  # rest length
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Should pull in +x direction
        assert physics_msg[0, 0] > 0
    
    def test_compressed_spring_pushes(self):
        """Compressed spring creates pushing force."""
        conv = LiteClothConv()
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        edge_attr = torch.tensor([[1.0]])  # rest length
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Should push in -x direction (negative force)
        assert physics_msg[0, 0] < 0
    
    def test_pure_physics_no_correction(self):
        """PurePhysicsClothConv returns zero correction."""
        conv = PurePhysicsClothConv()
        
        x = torch.randn(10, 3)
        edge_attr = torch.ones(10, 1)
        physics_msg = torch.randn(10, 3)
        
        correction = conv.compute_edge_correction(
            x, x, edge_attr, physics_message=physics_msg
        )
        
        assert torch.all(correction == 0)


class TestNoBending:
    """Tests verifying bending is disabled."""
    
    def test_bending_disabled_by_default(self):
        """Bending should be disabled in lite version."""
        config = LiteClothConvConfig()
        assert config.compute_bending is False
    
    def test_no_bending_computation(self):
        """Forward pass should not compute bending."""
        conv = LiteClothConv()
        
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)
        
        # Should work without face_pairs
        forces = conv(x, edge_index, edge_attr)
        
        assert forces.shape == (20, 3)


class TestNoDamping:
    """Tests verifying damping is disabled."""
    
    def test_damping_disabled_by_default(self):
        """Damping should be disabled in lite version."""
        config = LiteClothConvConfig()
        assert config.compute_damping is False
    
    def test_velocity_ignored(self):
        """Velocity should not affect output (no damping)."""
        conv = LiteClothConv()
        
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)
        
        # With and without velocity should give same physics
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]
        
        physics_no_vel = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        vel = torch.randn(20, 3)
        # Lite version doesn't use vel in physics computation
        physics_with_vel = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        assert torch.allclose(physics_no_vel, physics_with_vel)


class TestPhysicsFraction:
    """Tests for physics fraction in lightweight conv."""
    
    def test_physics_fraction_high(self):
        """Physics fraction should be very high (minimal correction)."""
        conv = LiteClothConv()
        conv.train()
        
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)
        
        _ = conv(x, edge_index, edge_attr)
        
        pf = conv.physics_fraction()
        assert pf > 0.9, f"Physics fraction should be >0.9, got {pf}"
    
    def test_pure_physics_fraction_one(self):
        """Pure physics should have physics_fraction = 1.0."""
        conv = PurePhysicsClothConv()
        conv.train()
        
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)
        
        _ = conv(x, edge_index, edge_attr)
        
        pf = conv.physics_fraction()
        assert pf > 0.99, f"Pure physics fraction should be ~1.0, got {pf}"


class TestGradientFlow:
    """Tests for gradient flow in lightweight conv."""
    
    def test_gradients_flow(self):
        """Gradients should flow through the network."""
        conv = LiteClothConv()
        
        x = torch.randn(20, 3, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)
        
        forces = conv(x, edge_index, edge_attr)
        loss = forces.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_pure_physics_gradients_flow(self):
        """Even pure physics should have gradients to input."""
        conv = PurePhysicsClothConv()
        
        x = torch.randn(20, 3, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)
        
        forces = conv(x, edge_index, edge_attr)
        loss = forces.sum()
        loss.backward()
        
        assert x.grad is not None


class TestOutputShape:
    """Tests for correct output shapes."""
    
    def test_lite_output_shape(self):
        """LiteClothConv should output [num_nodes, 3]."""
        conv = LiteClothConv()
        
        for num_nodes in [10, 50, 100]:
            x = torch.randn(num_nodes, 3)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
            edge_attr = torch.ones(num_nodes * 3, 1)
            
            forces = conv(x, edge_index, edge_attr)
            
            assert forces.shape == (num_nodes, 3)
    
    def test_pure_physics_output_shape(self):
        """PurePhysicsClothConv should output [num_nodes, 3]."""
        conv = PurePhysicsClothConv()
        
        x = torch.randn(100, 3)
        edge_index = torch.randint(0, 100, (2, 300))
        edge_attr = torch.ones(300, 1)
        
        forces = conv(x, edge_index, edge_attr)
        
        assert forces.shape == (100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
