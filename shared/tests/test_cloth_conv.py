"""
Unit tests for ClothForceConv.

Tests:
    - Hooke's law implementation
    - Strain computation
    - Force direction
    - Damping forces
    - Known equilibrium solutions
"""

import pytest
import torch
import math

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.physics_conv.cloth_conv import (
    ClothForceConv,
    ClothForceConvWithBending,
    ClothConvConfig,
)


class TestHookesLaw:
    """Tests verifying Hooke's law implementation."""
    
    def test_stretched_spring_pulls(self):
        """A stretched spring should create a pulling force."""
        config = ClothConvConfig(
            stretch_stiffness=100.0,
            compute_damping=False,
        )
        conv = ClothForceConv(config)
        
        # Two nodes, edge from 0 to 1
        # Rest length = 1.0, current length = 2.0 (stretched)
        x = torch.tensor([
            [0.0, 0.0, 0.0],  # Node 0
            [2.0, 0.0, 0.0],  # Node 1 (stretched to 2.0)
        ])
        edge_index = torch.tensor([[0], [1]])  # Edge from 0 to 1
        edge_attr = torch.tensor([[1.0]])  # Rest length = 1.0
        
        forces = conv(x, edge_index, edge_attr)
        
        # Force on node 1 should point toward node 0 (negative x)
        # because the spring is pulling it back
        # Actually, force on node 1 is aggregated from edge (0->1)
        # The edge message is force on source (node 0) from the spring
        # Since we aggregate to target, node 1 receives force computed at edge
        
        # Let's check the physics message directly
        x_i = x[0:1]  # Node 0
        x_j = x[1:2]  # Node 1
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Strain = (2 - 1) / 1 = 1.0
        # Force magnitude = k * strain = 100 * 1 = 100
        # Direction from i to j: [1, 0, 0]
        # So force = 100 * [1, 0, 0] (positive, pulling node 0 toward node 1)
        
        assert physics_msg[0, 0] > 0, "Force should pull node 0 toward node 1"
        expected_magnitude = 100.0 * 1.0  # k * strain
        assert abs(physics_msg[0, 0].item() - expected_magnitude) < 1e-5
    
    def test_compressed_spring_pushes(self):
        """A compressed spring should create a pushing force."""
        config = ClothConvConfig(
            stretch_stiffness=100.0,
            compute_damping=False,
        )
        conv = ClothForceConv(config)
        
        # Rest length = 2.0, current length = 1.0 (compressed)
        x = torch.tensor([
            [0.0, 0.0, 0.0],  # Node 0
            [1.0, 0.0, 0.0],  # Node 1 (compressed)
        ])
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[2.0]])  # Rest length = 2.0
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Strain = (1 - 2) / 2 = -0.5
        # Force is negative (compressed spring pushes)
        # Direction is still [1,0,0], so force on node 0 is negative (pushed away)
        assert physics_msg[0, 0] < 0, "Compressed spring should push"
    
    def test_no_force_at_rest(self):
        """No force when spring is at rest length."""
        config = ClothConvConfig(
            stretch_stiffness=100.0,
            compute_damping=False,
            correction_scale_init=0.0,  # No correction
        )
        conv = ClothForceConv(config)
        
        # At rest length
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[1.0]])  # Rest length = current length
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Should be zero (or very close)
        assert physics_msg.abs().max() < 1e-6
    
    def test_force_scales_with_stiffness(self):
        """Force should scale linearly with stiffness."""
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        edge_attr = torch.tensor([[1.0]])  # Rest length
        
        forces = []
        for k in [10.0, 100.0, 1000.0]:
            config = ClothConvConfig(stretch_stiffness=k, compute_damping=False)
            conv = ClothForceConv(config)
            
            x_i, x_j = x[0:1], x[1:2]
            physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
            forces.append(physics_msg[0, 0].item())
        
        # Forces should scale linearly: f1/f0 = k1/k0
        assert abs(forces[1] / forces[0] - 10.0) < 0.1
        assert abs(forces[2] / forces[1] - 10.0) < 0.1


class TestStrainClamping:
    """Tests for strain clamping (numerical stability)."""
    
    def test_extreme_stretch_is_clamped(self):
        """Extreme stretching should be clamped."""
        config = ClothConvConfig(
            stretch_stiffness=100.0,
            strain_clamp=2.0,
            compute_damping=False,
        )
        conv = ClothForceConv(config)
        
        # Extreme stretch: rest=1, current=100 -> strain=99
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        edge_attr = torch.tensor([[1.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # With strain_clamp=2.0, effective strain should be 2.0
        # Force = k * strain_clamped = 100 * 2.0 = 200
        assert abs(physics_msg[0, 0].item() - 200.0) < 1e-3


class TestDamping:
    """Tests for damping force computation."""
    
    def test_damping_opposes_stretching_velocity(self):
        """Damping should oppose relative velocity along edge."""
        config = ClothConvConfig(
            stretch_stiffness=0.0,  # No spring force
            damping_coefficient=1.0,
            compute_damping=True,
        )
        conv = ClothForceConv(config)
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        # Node 1 moving away from node 0
        vel = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Moving in +x direction
        ])
        edge_attr = torch.tensor([[1.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        vel_i, vel_j = vel[0:1], vel[1:2]
        
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr, vel_i, vel_j)
        
        # Relative velocity is [1,0,0], along edge direction [1,0,0]
        # Damping force = -d * (v_rel · dir) * dir = -1 * 1 * [1,0,0] = [-1,0,0]
        # This acts on node 0, trying to catch up to node 1
        assert physics_msg[0, 0] < 0, "Damping should oppose separation"
    
    def test_no_damping_for_perpendicular_velocity(self):
        """No damping for velocity perpendicular to edge."""
        config = ClothConvConfig(
            stretch_stiffness=0.0,
            damping_coefficient=1.0,
            compute_damping=True,
        )
        conv = ClothForceConv(config)
        
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Edge along x
        ])
        vel = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Velocity along y (perpendicular)
        ])
        edge_attr = torch.tensor([[1.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        vel_i, vel_j = vel[0:1], vel[1:2]
        
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr, vel_i, vel_j)
        
        # Perpendicular velocity has no component along edge
        assert physics_msg.abs().max() < 1e-6


class TestEquilibrium:
    """Tests for known equilibrium configurations."""
    
    def test_hanging_chain_equilibrium(self):
        """A vertical chain under gravity should reach equilibrium."""
        config = ClothConvConfig(
            stretch_stiffness=1000.0,
            compute_damping=False,
        )
        conv = ClothForceConv(config)
        
        # Two nodes connected vertically
        # Top at y=1, bottom at y=0 (stretched from rest length 0.9)
        x = torch.tensor([
            [0.0, 1.0, 0.0],  # Top node (index 0)
            [0.0, 0.0, 0.0],  # Bottom node (index 1)
        ])
        # Edge from top (0) to bottom (1)
        edge_index = torch.tensor([[0], [1]])
        rest_length = 0.9  # Stretched from 0.9 to 1.0
        edge_attr = torch.tensor([[rest_length]])
        
        # Get the edge physics message
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Edge direction: from top to bottom is [0, -1, 0]
        # Strain is positive (stretched): (1.0 - 0.9) / 0.9 > 0
        # Force = stiffness * strain * direction = k * (+strain) * [0, -1, 0]
        # So force on source node (top) is in -y direction (pulled down)
        # which means the spring is pulling correctly
        assert physics_msg[0, 1] < 0, "Spring pulls top node toward bottom"
        
        # Test with pure physics (no correction) to verify Newton's 3rd law
        from shared.physics_conv import PurePhysicsClothConv
        pure_conv = PurePhysicsClothConv(stiffness=1000.0)
        
        edge_index_bi = torch.tensor([[0, 1], [1, 0]])
        edge_attr_bi = torch.tensor([[rest_length], [rest_length]])
        forces = pure_conv(x, edge_index_bi, edge_attr_bi)
        
        # For internal forces with pure physics, they should sum to zero (Newton's 3rd law)
        total_force = forces.sum(dim=0)
        assert total_force.abs().max() < 1e-3, f"Internal forces should cancel: {total_force}"


class TestPhysicsFraction:
    """Tests for physics fraction diagnostic."""
    
    def test_physics_dominates_initially(self):
        """Physics fraction should be high at initialization."""
        conv = ClothForceConv()
        conv.train()
        
        x = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.ones(50, 1)  # Rest lengths
        
        _ = conv(x, edge_index, edge_attr)
        
        pf = conv.physics_fraction()
        assert pf > 0.9, f"Physics fraction should be >0.9, got {pf}"


class TestOutputShape:
    """Tests for correct output shapes."""
    
    def test_output_shape_matches_nodes(self):
        """Output should have shape [num_nodes, 3]."""
        conv = ClothForceConv()
        
        num_nodes = 100
        num_edges = 300
        
        x = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.ones(num_edges, 1)
        
        forces = conv(x, edge_index, edge_attr)
        
        assert forces.shape == (num_nodes, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
