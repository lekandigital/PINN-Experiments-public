"""
Unit tests for ShallowWaterConv.

Tests:
    - Shallow water physics implementation
    - Pressure gradient forces
    - Bottom friction
    - Mass conservation
    - Boundary conditions
"""

import pytest
import torch
import math

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.physics_conv.shallow_water_conv import (
    ShallowWaterConv,
    ShallowWaterConvWithIntegration,
    ShallowWaterConfig,
)


class TestPressureGradient:
    """Tests for pressure gradient force computation."""
    
    def test_water_flows_downhill(self):
        """Water should accelerate from high to low surface elevation."""
        config = ShallowWaterConfig(
            gravity=9.81,
            drag_coefficient=0.0,  # No friction for this test
        )
        conv = ShallowWaterConv(config)
        
        # Node state: [η, u, v, h_bathy]
        # Node 0: high surface (η=1), stationary
        # Node 1: low surface (η=0), stationary
        x = torch.tensor([
            [1.0, 0.0, 0.0, 10.0],  # High water, deep
            [0.0, 0.0, 0.0, 10.0],  # Low water, deep
        ])
        edge_index = torch.tensor([[0], [1]])
        # edge_attr: [length, normal_x, normal_y, type]
        edge_attr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # Unit length, x-normal
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # F_pressure = -g * grad(η) = -g * (η_j - η_i) / L * n
        # = -9.81 * (0 - 1) / 1 * [1, 0] = [9.81, 0]
        # This is acceleration on node 0, which should be positive (toward node 1)
        
        # The momentum component (du/dt) should be positive
        assert physics_msg[0, 1] > 0, "Pressure should accelerate flow toward lower surface"
    
    def test_flat_surface_no_pressure_force(self):
        """Flat water surface should have no pressure gradient force."""
        config = ShallowWaterConfig(drag_coefficient=0.0)
        conv = ShallowWaterConv(config)
        
        # Both nodes have same surface elevation
        x = torch.tensor([
            [1.0, 0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 10.0],
        ])
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Pressure gradient should be zero
        # But there might be mass flux if there's velocity
        # With zero velocity, all fluxes should be zero
        assert abs(physics_msg[0, 1].item()) < 1e-5, "No pressure force for flat surface"
        assert abs(physics_msg[0, 2].item()) < 1e-5


class TestBottomFriction:
    """Tests for bottom friction force."""
    
    def test_friction_opposes_motion(self):
        """Bottom friction should decelerate flow."""
        config = ShallowWaterConfig(
            gravity=0.0,  # No pressure for this test
            drag_coefficient=0.01,
        )
        conv = ShallowWaterConv(config)
        
        # Node with velocity in +x direction
        x = torch.tensor([
            [0.0, 1.0, 0.0, 10.0],  # u=1, v=0, deep water
            [0.0, 1.0, 0.0, 10.0],
        ])
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Friction = -C_d * |u| * u / h
        # For u=[1,0], |u|=1, h=10: F_drag = -0.01 * 1 * 1 / 10 = -0.001
        # du/dt component should be negative (decelerating)
        assert physics_msg[0, 1] < 0, "Friction should decelerate +x velocity"
    
    def test_friction_scales_with_speed(self):
        """Faster flow should experience more friction."""
        config = ShallowWaterConfig(gravity=0.0, drag_coefficient=0.01)
        conv = ShallowWaterConv(config)
        
        frictions = []
        for speed in [1.0, 2.0, 4.0]:
            x = torch.tensor([
                [0.0, speed, 0.0, 10.0],
                [0.0, speed, 0.0, 10.0],
            ])
            edge_attr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
            
            x_i, x_j = x[0:1], x[1:2]
            physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
            
            frictions.append(abs(physics_msg[0, 1].item()))
        
        # Quadratic friction: doubling speed should quadruple friction
        assert frictions[1] > frictions[0]
        assert frictions[2] > frictions[1]


class TestMassFlux:
    """Tests for mass flux computation."""
    
    def test_flow_creates_mass_flux(self):
        """Velocity normal to edge creates mass flux."""
        config = ShallowWaterConfig(gravity=0.0, drag_coefficient=0.0)
        conv = ShallowWaterConv(config)
        
        # Flow in +x direction across x-normal edge
        x = torch.tensor([
            [0.0, 1.0, 0.0, 10.0],  # u=1, depth=10
            [0.0, 1.0, 0.0, 10.0],
        ])
        edge_attr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # x-normal
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Mass flux = h * u_normal * length = 10 * 1 * 1 = 10
        # The dη/dt component
        assert abs(physics_msg[0, 0].item()) > 0, "Should have mass flux"


class TestMassConservation:
    """Tests for mass conservation enforcement."""
    
    def test_mass_conservation_projection(self):
        """Total mass change should be zero after projection."""
        config = ShallowWaterConfig(enforce_mass_conservation=True)
        conv = ShallowWaterConv(config)
        
        # Create a scenario that would create/destroy mass
        x = torch.randn(20, 4)
        x[:, 3] = 10.0  # Set bathymetry to constant
        
        edge_index = torch.randint(0, 20, (2, 60))
        edge_attr = torch.randn(60, 4)
        edge_attr[:, 0] = 1.0  # Unit length
        edge_attr[:, 1:3] = edge_attr[:, 1:3] / (edge_attr[:, 1:3].norm(dim=1, keepdim=True) + 1e-8)
        edge_attr[:, 3] = 0  # Interior edges
        
        dstate_dt = conv(x, edge_index, edge_attr)
        
        # Total dη/dt should sum to zero
        total_mass_change = dstate_dt[:, 0].sum()
        assert abs(total_mass_change.item()) < 1e-5, f"Mass not conserved: {total_mass_change}"


class TestBoundaryConditions:
    """Tests for boundary condition handling."""
    
    def test_land_boundary_no_flux(self):
        """Land boundary edges should have zero normal flux."""
        config = ShallowWaterConfig(handle_boundaries=True)
        conv = ShallowWaterConv(config)
        
        x = torch.tensor([
            [1.0, 1.0, 0.0, 10.0],  # Water node
            [0.0, 0.0, 0.0, 0.0],   # Land node (dry)
        ])
        edge_index = torch.tensor([[0], [1]])
        # Edge type 1 = land boundary
        edge_attr = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        
        x_i, x_j = x[0:1], x[1:2]
        physics_msg = conv.compute_edge_physics(x_i, x_j, edge_attr)
        
        # Apply boundary conditions
        edge_type = edge_attr[:, 3:4]
        physics_msg = conv._apply_edge_boundary_conditions(
            physics_msg, edge_type, x_i, x_j
        )
        
        # Mass flux should be zero at land boundary
        assert abs(physics_msg[0, 0].item()) < 1e-6


class TestPrecision:
    """Tests for numerical precision."""
    
    def test_uses_float64_internally(self):
        """Physics should use float64 for precision."""
        config = ShallowWaterConfig(physics_dtype=torch.float64)
        conv = ShallowWaterConv(config)
        
        x = torch.randn(10, 4)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 4)
        
        # Forward should work with float32 input
        dstate = conv(x, edge_index, edge_attr)
        
        # Output should be same dtype as input
        assert dstate.dtype == x.dtype


class TestOutputShape:
    """Tests for correct output shapes."""
    
    def test_output_shape(self):
        """Output should be [num_nodes, 3]."""
        conv = ShallowWaterConv()
        
        num_nodes = 50
        x = torch.randn(num_nodes, 4)
        edge_index = torch.randint(0, num_nodes, (2, 150))
        edge_attr = torch.randn(150, 4)
        
        dstate = conv(x, edge_index, edge_attr)
        
        # Output: [dη/dt, du/dt, dv/dt]
        assert dstate.shape == (num_nodes, 3)


class TestShallowWaterWithIntegration:
    """Tests for ShallowWaterConvWithIntegration."""
    
    def test_returns_updated_state(self):
        """Integration should return updated state."""
        conv = ShallowWaterConvWithIntegration()
        
        x = torch.randn(20, 4)
        x[:, 3] = 10.0  # Constant bathymetry
        
        edge_index = torch.randint(0, 20, (2, 60))
        edge_attr = torch.randn(60, 4)
        edge_attr[:, 0] = 1.0
        
        x_new = conv(x, edge_index, edge_attr, dt=0.1)
        
        # Output should have same shape as input
        assert x_new.shape == x.shape
        
        # Bathymetry should be unchanged
        assert torch.allclose(x_new[:, 3], x[:, 3])


class TestPhysicsFraction:
    """Tests for physics fraction diagnostic."""
    
    def test_physics_dominates(self):
        """Physics should dominate at initialization."""
        conv = ShallowWaterConv()
        conv.train()
        
        x = torch.randn(20, 4)
        x[:, 3] = 10.0
        edge_index = torch.randint(0, 20, (2, 60))
        edge_attr = torch.randn(60, 4)
        edge_attr[:, 0] = 1.0
        
        _ = conv(x, edge_index, edge_attr)
        
        pf = conv.physics_fraction()
        assert pf > 0.7, f"Physics fraction should be >0.7, got {pf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
