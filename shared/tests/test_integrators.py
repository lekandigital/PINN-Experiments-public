"""
Unit tests for time integrators.

Tests:
    - Correct update formulas
    - Energy behavior (conservation/dissipation)
    - Batch handling
    - Factory creation
"""

import pytest
import torch
import math

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.physics_conv.integrators import (
    ExplicitEuler,
    SemiImplicitEuler,
    VelocityVerlet,
    IntegratorFactory,
)


class TestExplicitEuler:
    """Tests for ExplicitEuler integrator."""
    
    def test_basic_step(self):
        """Test basic Euler step formula."""
        integrator = ExplicitEuler(damping=1.0)
        
        pos = torch.tensor([[0.0, 0.0, 0.0]])
        vel = torch.tensor([[1.0, 0.0, 0.0]])
        acc = torch.tensor([[0.0, -9.81, 0.0]])
        dt = 0.1
        
        new_pos, new_vel = integrator.step(pos, vel, acc, dt)
        
        # x' = x + dt * v = [0.1, 0, 0]
        assert torch.allclose(new_pos, torch.tensor([[0.1, 0.0, 0.0]]))
        
        # v' = v + dt * a = [1, -0.981, 0]
        expected_vel = vel + dt * acc
        assert torch.allclose(new_vel, expected_vel)
    
    def test_damping(self):
        """Test that damping reduces velocity."""
        integrator = ExplicitEuler(damping=0.9)
        
        pos = torch.zeros(1, 3)
        vel = torch.tensor([[10.0, 0.0, 0.0]])
        acc = torch.zeros(1, 3)
        dt = 0.01
        
        new_pos, new_vel = integrator.step(pos, vel, acc, dt)
        
        # Velocity should be damped
        assert new_vel[0, 0] < vel[0, 0]
        assert torch.allclose(new_vel, vel * 0.9)
    
    def test_batched_input(self):
        """Test with batched inputs."""
        integrator = ExplicitEuler()
        
        batch_size = 5
        num_nodes = 10
        
        pos = torch.randn(batch_size, num_nodes, 3)
        vel = torch.randn(batch_size, num_nodes, 3)
        acc = torch.randn(batch_size, num_nodes, 3)
        
        new_pos, new_vel = integrator.step(pos, vel, acc, dt=0.01)
        
        assert new_pos.shape == pos.shape
        assert new_vel.shape == vel.shape
    
    def test_name(self):
        """Test name property."""
        assert ExplicitEuler().name == "ExplicitEuler"


class TestSemiImplicitEuler:
    """Tests for SemiImplicitEuler integrator."""
    
    def test_uses_new_velocity(self):
        """Test that position update uses NEW velocity."""
        integrator = SemiImplicitEuler(damping=1.0)
        
        pos = torch.zeros(1, 3)
        vel = torch.tensor([[1.0, 0.0, 0.0]])
        acc = torch.tensor([[1.0, 0.0, 0.0]])  # Accelerating in x
        dt = 0.1
        
        new_pos, new_vel = integrator.step(pos, vel, acc, dt)
        
        # v' = v + dt * a = 1.1
        expected_vel = vel + dt * acc
        assert torch.allclose(new_vel, expected_vel)
        
        # x' = x + dt * v' = 0.11 (uses new velocity 1.1, not old velocity 1.0)
        expected_pos = pos + dt * new_vel
        assert torch.allclose(new_pos, expected_pos)
        
        # Compare with explicit Euler which would give 0.1
        explicit = ExplicitEuler(damping=1.0)
        exp_pos, _ = explicit.step(pos, vel, acc, dt)
        assert not torch.allclose(new_pos, exp_pos)  # Should be different
    
    def test_harmonic_oscillator_energy(self):
        """Test energy behavior for harmonic oscillator.
        
        Semi-implicit Euler should have bounded energy oscillation.
        """
        integrator = SemiImplicitEuler(damping=1.0)
        
        # Simple harmonic oscillator: a = -k * x
        k = 1.0
        dt = 0.01
        
        pos = torch.tensor([[1.0, 0.0, 0.0]])  # Initial displacement
        vel = torch.zeros(1, 3)
        
        energies = []
        for _ in range(1000):
            # E = 0.5 * k * x^2 + 0.5 * v^2
            energy = 0.5 * k * (pos ** 2).sum() + 0.5 * (vel ** 2).sum()
            energies.append(energy.item())
            
            acc = -k * pos
            pos, vel = integrator.step(pos, vel, acc, dt)
        
        # Energy should be roughly conserved (bounded, not growing)
        energies = torch.tensor(energies)
        initial_energy = energies[0]
        
        # Allow some oscillation but no systematic growth
        assert energies.max() < 1.5 * initial_energy, "Energy grew too much"
        assert energies.min() > 0.5 * initial_energy, "Energy decayed too much"
    
    def test_name(self):
        assert SemiImplicitEuler().name == "SemiImplicitEuler"


class TestVelocityVerlet:
    """Tests for VelocityVerlet integrator."""
    
    def test_second_order_accuracy(self):
        """Verify second-order accuracy with constant acceleration."""
        integrator = VelocityVerlet(damping=1.0)
        
        pos = torch.zeros(1, 3)
        vel = torch.tensor([[1.0, 0.0, 0.0]])
        acc = torch.tensor([[2.0, 0.0, 0.0]])  # Constant acceleration
        dt = 0.1
        
        new_pos, new_vel = integrator.step(pos, vel, acc, dt)
        
        # Exact solution for constant acceleration:
        # x = x0 + v0*t + 0.5*a*t^2 = 0 + 1*0.1 + 0.5*2*0.01 = 0.11
        expected_pos = pos + vel * dt + 0.5 * acc * dt * dt
        assert torch.allclose(new_pos, expected_pos)
    
    def test_half_step_velocity(self):
        """Test half-step velocity computation."""
        integrator = VelocityVerlet()
        
        vel = torch.tensor([[1.0, 0.0, 0.0]])
        acc = torch.tensor([[2.0, 0.0, 0.0]])
        dt = 0.1
        
        v_half = integrator.half_step_velocity(vel, acc, dt)
        
        # v_half = v + 0.5 * a * dt = 1 + 0.5 * 2 * 0.1 = 1.1
        expected = vel + 0.5 * acc * dt
        assert torch.allclose(v_half, expected)
    
    def test_better_energy_conservation(self):
        """Verlet should conserve energy better than Euler for oscillator."""
        k = 1.0
        dt = 0.05
        steps = 500
        
        # Run Verlet
        verlet = VelocityVerlet(damping=1.0)
        pos_v = torch.tensor([[1.0, 0.0, 0.0]])
        vel_v = torch.zeros(1, 3)
        
        verlet_energies = []
        for _ in range(steps):
            energy = 0.5 * k * (pos_v ** 2).sum() + 0.5 * (vel_v ** 2).sum()
            verlet_energies.append(energy.item())
            acc = -k * pos_v
            pos_v, vel_v = verlet.step(pos_v, vel_v, acc, dt)
        
        # Run Explicit Euler for comparison
        euler = ExplicitEuler(damping=1.0)
        pos_e = torch.tensor([[1.0, 0.0, 0.0]])
        vel_e = torch.zeros(1, 3)
        
        euler_energies = []
        for _ in range(steps):
            energy = 0.5 * k * (pos_e ** 2).sum() + 0.5 * (vel_e ** 2).sum()
            euler_energies.append(energy.item())
            acc = -k * pos_e
            pos_e, vel_e = euler.step(pos_e, vel_e, acc, dt)
        
        # Verlet energy variance should be smaller
        verlet_var = torch.tensor(verlet_energies).var()
        euler_var = torch.tensor(euler_energies).var()
        
        assert verlet_var < euler_var, \
            f"Verlet energy variance ({verlet_var}) should be less than Euler ({euler_var})"
    
    def test_order_property(self):
        """Test that order returns 2."""
        assert VelocityVerlet().order == 2
    
    def test_name(self):
        assert VelocityVerlet().name == "VelocityVerlet"


class TestIntegratorFactory:
    """Tests for IntegratorFactory."""
    
    def test_create_by_name(self):
        """Test creating integrators by name."""
        euler = IntegratorFactory.create("euler")
        assert isinstance(euler, ExplicitEuler)
        
        semi = IntegratorFactory.create("semi_implicit")
        assert isinstance(semi, SemiImplicitEuler)
        
        verlet = IntegratorFactory.create("verlet")
        assert isinstance(verlet, VelocityVerlet)
    
    def test_case_and_formatting(self):
        """Test that names are flexible with case and formatting."""
        assert isinstance(IntegratorFactory.create("EULER"), ExplicitEuler)
        assert isinstance(IntegratorFactory.create("explicit-euler"), ExplicitEuler)
        assert isinstance(IntegratorFactory.create("Velocity_Verlet"), VelocityVerlet)
        assert isinstance(IntegratorFactory.create("semi implicit"), SemiImplicitEuler)
    
    def test_passes_kwargs(self):
        """Test that kwargs are passed to constructor."""
        euler = IntegratorFactory.create("euler", damping=0.5)
        assert euler.damping == 0.5
    
    def test_unknown_raises(self):
        """Test that unknown integrator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown integrator"):
            IntegratorFactory.create("unknown_integrator")
    
    def test_available(self):
        """Test available() returns list of names."""
        available = IntegratorFactory.available()
        assert "euler" in available
        assert "verlet" in available
        assert "semi_implicit" in available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
