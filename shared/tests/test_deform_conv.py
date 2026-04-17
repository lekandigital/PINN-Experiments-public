"""
Unit tests for ElasticForceConv.

Tests:
    - 3D Hooke's law implementation
    - Material property handling
    - VelocityGRU integration
    - Time integration
"""

import pytest
import torch

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.physics_conv.deform_conv import (
    ElasticForceConv,
    ElasticForceConvWithIntegration,
    ElasticConvConfig,
    VelocityGRU,
)


class TestElasticHookesLaw:
    """Tests verifying elastic Hooke's law implementation."""
    
    def test_stretched_bond_creates_force(self):
        """A stretched bond should create attractive force."""
        config = ElasticConvConfig()
        conv = ElasticForceConv(config)
        
        # Two nodes connected by an edge
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Stretched from rest length 1.0
        ])
        edge_index = torch.tensor([[0], [1]])
        # edge_attr: [rest_length, stiffness]
        edge_attr = torch.tensor([[1.0, 1000.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Stretched by 1.0, stiffness 1000 -> force = 1000 * 1.0 = 1000
        # Direction: from 0 to 1 is [1,0,0]
        assert physics_msg[0, 0] > 0, "Should pull node 0 toward node 1"
        assert abs(physics_msg[0, 0].item() - 1000.0) < 1.0
    
    def test_compressed_bond_creates_repulsion(self):
        """A compressed bond should create repulsive force."""
        config = ElasticConvConfig()
        conv = ElasticForceConv(config)
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Compressed from rest length 1.0
        ])
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[1.0, 1000.0]])  # rest=1, k=1000
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Compressed by 0.5, force = 1000 * (-0.5) = -500
        assert physics_msg[0, 0] < 0, "Should push node 0 away from node 1"
    
    def test_no_force_at_rest(self):
        """No force when at rest length."""
        config = ElasticConvConfig()
        conv = ElasticForceConv(config)
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        edge_attr = torch.tensor([[1.0, 1000.0]])  # rest=1, k=1000
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        assert physics_msg.abs().max() < 1e-5


class TestMaterialProperties:
    """Tests for material property handling."""
    
    def test_per_edge_stiffness(self):
        """Test that per-edge stiffness is used."""
        config = ElasticConvConfig(stiffness_mode="per_edge")
        conv = ElasticForceConv(config)
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Stretched
        ])
        
        # Test with different stiffnesses
        for k in [100.0, 500.0, 1000.0]:
            edge_attr = torch.tensor([[1.0, k]])
            x_i, x_j = x[0:1], x[1:2]
            physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
            
            expected_force = k * 1.0  # k * displacement
            assert abs(physics_msg[0, 0].item() - expected_force) < 1.0
    
    def test_uses_youngs_modulus_fallback(self):
        """When no edge stiffness, use config's Young's modulus."""
        config = ElasticConvConfig(youngs_modulus=500.0)
        conv = ElasticForceConv(config)
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        # Only rest length, no stiffness
        edge_attr = torch.tensor([[1.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Should use youngs_modulus as stiffness
        expected = 500.0 * 1.0
        assert abs(physics_msg[0, 0].item() - expected) < 1.0


class TestVelocityGRU:
    """Tests for VelocityGRU module."""
    
    def test_output_shape(self):
        """Test GRU output shape."""
        gru = VelocityGRU(input_size=3, hidden_size=64, output_size=3)
        
        vel_update = torch.randn(100, 3)
        refined, hidden = gru(vel_update)
        
        assert refined.shape == (100, 3)
        assert hidden.shape == (100, 64)
    
    def test_hidden_state_persistence(self):
        """Test that hidden state is updated."""
        gru = VelocityGRU(input_size=3, hidden_size=64, output_size=3)
        
        vel = torch.randn(50, 3)
        
        # First pass
        out1, hidden1 = gru(vel)
        
        # Second pass with hidden
        out2, hidden2 = gru(vel, hidden1)
        
        # Outputs should be different due to hidden state
        assert not torch.allclose(out1, out2)
    
    def test_init_hidden(self):
        """Test hidden state initialization."""
        gru = VelocityGRU(hidden_size=32)
        
        hidden = gru.init_hidden(100, torch.device('cpu'), torch.float32)
        
        assert hidden.shape == (100, 32)
        assert torch.all(hidden == 0)


class TestElasticConvWithGRU:
    """Tests for ElasticForceConv with VelocityGRU."""
    
    def test_gru_integration(self):
        """Test that GRU integrates with ElasticForceConv."""
        config = ElasticConvConfig(use_velocity_gru=True, gru_hidden_dim=32)
        conv = ElasticForceConv(config)
        
        assert conv.velocity_gru is not None
        
        # Forward pass with velocity
        x = torch.randn(20, 3)
        vel = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.rand(50, 2)  # [rest_length, stiffness]
        edge_attr[:, 0] = 0.1  # Small rest lengths
        edge_attr[:, 1] = 100.0  # Moderate stiffness
        
        # Should return refined velocity
        result = conv(x, edge_index, edge_attr, vel=vel, dt=0.01)
        
        assert result.shape == (20, 3)
    
    def test_return_hidden(self):
        """Test returning hidden state."""
        config = ElasticConvConfig(use_velocity_gru=True)
        conv = ElasticForceConv(config)
        
        x = torch.randn(20, 3)
        vel = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.rand(50, 2)
        edge_attr[:, 0] = 0.1
        edge_attr[:, 1] = 100.0
        
        result, hidden = conv(x, edge_index, edge_attr, vel=vel, dt=0.01, return_hidden=True)
        
        assert result.shape == (20, 3)
        assert hidden.shape == (20, 64)  # Default hidden dim


class TestElasticConvWithIntegration:
    """Tests for ElasticForceConvWithIntegration."""
    
    def test_returns_pos_vel(self):
        """Test that integration returns new positions and velocities."""
        config = ElasticConvConfig()
        conv = ElasticForceConvWithIntegration(config, integrator="semi_implicit")
        
        x = torch.randn(20, 3)
        vel = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.rand(50, 2)
        edge_attr[:, 0] = 0.1
        edge_attr[:, 1] = 100.0
        
        new_pos, new_vel, new_hidden = conv(x, edge_index, edge_attr, vel=vel, dt=0.01)
        
        assert new_pos.shape == x.shape
        assert new_vel.shape == vel.shape
    
    def test_different_integrators(self):
        """Test with different integrators."""
        config = ElasticConvConfig()
        
        x = torch.randn(20, 3)
        vel = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.rand(50, 2)
        edge_attr[:, 0] = 0.1
        edge_attr[:, 1] = 100.0
        
        results = {}
        for integrator in ["euler", "semi_implicit", "verlet"]:
            conv = ElasticForceConvWithIntegration(config, integrator=integrator)
            new_pos, new_vel, _ = conv(x.clone(), edge_index, edge_attr, vel=vel.clone(), dt=0.01)
            results[integrator] = (new_pos, new_vel)
        
        # Different integrators should give (slightly) different results
        # except for very small dt where they converge
        # Here we just verify they all work
        for name, (pos, vel) in results.items():
            assert pos.shape == x.shape, f"{name} position shape mismatch"
            assert vel.shape == x.shape, f"{name} velocity shape mismatch"


class TestPhysicsFraction:
    """Tests for physics fraction in elastic conv."""
    
    def test_physics_dominates(self):
        """Physics should dominate at initialization."""
        conv = ElasticForceConv()
        conv.train()
        
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.rand(50, 2)
        edge_attr[:, 0] = 0.1
        edge_attr[:, 1] = 100.0
        
        _ = conv(x, edge_index, edge_attr)
        
        pf = conv.physics_fraction()
        assert pf > 0.7, f"Physics fraction should be >0.7, got {pf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
