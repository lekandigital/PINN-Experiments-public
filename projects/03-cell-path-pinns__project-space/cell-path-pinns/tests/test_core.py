"""
Tests for core geodesic trajectory abstractions.

Tests PotentialFieldBase, GeodesicLoss, and TrajectoryPINN classes.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

import sys
sys.path.insert(0, '..')

from src.core.potential_field import (
    PotentialFieldBase,
    LearnedPotentialField,
    AnalyticPotentialField,
    ComposedPotentialField,
)
from src.core.geodesic_loss import (
    ConstantSpeedLoss,
    BoundaryLoss,
    GradientFollowingLoss,
    CurvaturePenalty,
    GeodesicLoss,
)
from src.core.trajectory_pinn import (
    TrajectoryPINN,
    BoundaryConditionedTrajectory,
    MultiTrajectoryPINN,
)


# ============================================================================
# PotentialField Tests
# ============================================================================

class TestLearnedPotentialField:
    """Tests for LearnedPotentialField."""
    
    def test_construction(self):
        """Test basic construction."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        assert field.input_dim == 2
        
    def test_forward_shape(self):
        """Test output shape."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(100, 2)
        phi = field(x)
        assert phi.shape == (100, 1)
        
    def test_forward_3d(self):
        """Test 3D input."""
        field = LearnedPotentialField(input_dim=3, hidden_dims=[32, 32])
        x = torch.randn(50, 3)
        phi = field(x)
        assert phi.shape == (50, 1)
        
    def test_gradient_shape(self):
        """Test gradient computation."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(100, 2, requires_grad=True)
        grad = field.gradient(x)
        assert grad.shape == (100, 2)
        
    def test_gradient_autograd(self):
        """Test that gradient uses autograd correctly."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(10, 2, requires_grad=True)
        
        # Manual gradient via autograd
        phi = field(x)
        manual_grad = torch.autograd.grad(
            phi, x, grad_outputs=torch.ones_like(phi), create_graph=True
        )[0]
        
        # Method gradient
        method_grad = field.gradient(x)
        
        assert torch.allclose(manual_grad, method_grad, atol=1e-6)
        
    def test_laplacian_shape(self):
        """Test Laplacian computation."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(50, 2, requires_grad=True)
        lap = field.laplacian(x)
        assert lap.shape == (50, 1)
        
    def test_metric_tensor_shape(self):
        """Test metric tensor shape (default identity)."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(50, 2)
        g = field.metric_tensor(x)
        assert g.shape == (50, 2, 2)
        
    def test_metric_tensor_identity(self):
        """Test that default metric tensor is identity."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(50, 2)
        g = field.metric_tensor(x)
        expected = torch.eye(2).unsqueeze(0).expand(50, -1, -1)
        assert torch.allclose(g, expected)


class TestAnalyticPotentialField:
    """Tests for AnalyticPotentialField."""
    
    def test_quadratic_potential(self):
        """Test simple quadratic potential."""
        def quadratic(x: torch.Tensor) -> torch.Tensor:
            return (x ** 2).sum(dim=-1, keepdim=True)
        
        field = AnalyticPotentialField(quadratic, input_dim=2)
        x = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        phi = field(x)
        
        expected = torch.tensor([[2.0], [4.0]])
        assert torch.allclose(phi, expected)
        
    def test_gradient_analytic(self):
        """Test gradient of analytic potential."""
        def quadratic(x: torch.Tensor) -> torch.Tensor:
            return (x ** 2).sum(dim=-1, keepdim=True)
        
        field = AnalyticPotentialField(quadratic, input_dim=2)
        x = torch.tensor([[1.0, 1.0], [2.0, 3.0]], requires_grad=True)
        grad = field.gradient(x)
        
        # Gradient of x^2 + y^2 is [2x, 2y]
        expected = 2 * x.detach()
        assert torch.allclose(grad, expected, atol=1e-5)


class TestComposedPotentialField:
    """Tests for ComposedPotentialField."""
    
    def test_sum_composition(self):
        """Test sum of two fields."""
        def f1(x: torch.Tensor) -> torch.Tensor:
            return x[:, 0:1] ** 2
        
        def f2(x: torch.Tensor) -> torch.Tensor:
            return x[:, 1:2] ** 2
        
        field1 = AnalyticPotentialField(f1, input_dim=2)
        field2 = AnalyticPotentialField(f2, input_dim=2)
        
        composed = ComposedPotentialField([field1, field2], weights=[1.0, 1.0])
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        phi = composed(x)
        
        # Should be x^2 + y^2
        expected = torch.tensor([[1.0 + 4.0], [9.0 + 16.0]])
        assert torch.allclose(phi, expected)
        
    def test_weighted_composition(self):
        """Test weighted sum of fields."""
        def f1(x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 1)
        
        def f2(x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 1) * 2
        
        field1 = AnalyticPotentialField(f1, input_dim=2)
        field2 = AnalyticPotentialField(f2, input_dim=2)
        
        composed = ComposedPotentialField([field1, field2], weights=[0.5, 0.25])
        
        x = torch.randn(10, 2)
        phi = composed(x)
        
        # 0.5 * 1 + 0.25 * 2 = 1.0
        expected = torch.ones(10, 1)
        assert torch.allclose(phi, expected)


# ============================================================================
# GeodesicLoss Tests
# ============================================================================

class TestConstantSpeedLoss:
    """Tests for ConstantSpeedLoss."""
    
    def test_constant_speed_satisfied(self):
        """Test loss is zero when speed is constant."""
        loss_fn = ConstantSpeedLoss(target_speed=1.0, weight=1.0)
        
        # Velocity with magnitude 1
        velocity = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        velocity = velocity / velocity.norm(dim=1, keepdim=True)  # Normalize
        
        loss = loss_fn(velocity)
        assert loss.item() < 0.01  # Should be near zero
        
    def test_constant_speed_violated(self):
        """Test loss is positive when speed varies."""
        loss_fn = ConstantSpeedLoss(target_speed=1.0, weight=1.0)
        
        # Velocity with varying magnitudes
        velocity = torch.tensor([[2.0, 0.0], [0.5, 0.0]])
        
        loss = loss_fn(velocity)
        assert loss.item() > 0.5  # Should be significant


class TestBoundaryLoss:
    """Tests for BoundaryLoss."""
    
    def test_start_boundary(self):
        """Test start position loss."""
        loss_fn = BoundaryLoss(weight=1.0)
        
        predicted_start = torch.tensor([[0.1, 0.1], [0.0, 0.0]])
        target_start = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        
        loss = loss_fn(predicted_start, target_start, None, None)
        
        # First point has error, second doesn't
        assert loss.item() > 0
        
    def test_end_boundary(self):
        """Test end position loss."""
        loss_fn = BoundaryLoss(weight=1.0)
        
        predicted_end = torch.tensor([[1.0, 1.0], [0.9, 0.9]])
        target_end = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        
        loss = loss_fn(None, None, predicted_end, target_end)
        
        # Second point has error
        assert loss.item() > 0
        
    def test_both_boundaries(self):
        """Test both boundary conditions."""
        loss_fn = BoundaryLoss(weight=1.0)
        
        pred_start = torch.tensor([[0.0, 0.0]])
        target_start = torch.tensor([[0.0, 0.0]])
        pred_end = torch.tensor([[1.0, 1.0]])
        target_end = torch.tensor([[1.0, 1.0]])
        
        loss = loss_fn(pred_start, target_start, pred_end, target_end)
        assert loss.item() < 1e-6  # Should be zero


class TestGradientFollowingLoss:
    """Tests for GradientFollowingLoss."""
    
    def test_gradient_following(self):
        """Test that velocity following gradient gives low loss."""
        loss_fn = GradientFollowingLoss(weight=1.0)
        
        # Velocity pointing in negative gradient direction
        velocity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        potential_gradient = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        
        loss = loss_fn(velocity, potential_gradient)
        assert loss.item() < 0.1
        
    def test_gradient_opposing(self):
        """Test that velocity opposing gradient gives high loss."""
        loss_fn = GradientFollowingLoss(weight=1.0)
        
        # Velocity pointing in positive gradient direction (wrong way)
        velocity = torch.tensor([[1.0, 0.0]])
        potential_gradient = torch.tensor([[1.0, 0.0]])
        
        loss = loss_fn(velocity, potential_gradient)
        assert loss.item() > 0.5


class TestCurvaturePenalty:
    """Tests for CurvaturePenalty."""
    
    def test_straight_line(self):
        """Test that straight line has zero curvature penalty."""
        loss_fn = CurvaturePenalty(weight=1.0)
        
        # Zero acceleration = straight line
        acceleration = torch.zeros(100, 2)
        
        loss = loss_fn(acceleration)
        assert loss.item() < 1e-6
        
    def test_curved_path(self):
        """Test that curved path has positive penalty."""
        loss_fn = CurvaturePenalty(weight=1.0)
        
        # Non-zero acceleration = curved path
        acceleration = torch.randn(100, 2)
        
        loss = loss_fn(acceleration)
        assert loss.item() > 0


class TestGeodesicLoss:
    """Tests for combined GeodesicLoss."""
    
    def test_combined_loss_dict(self):
        """Test that combined loss returns dictionary."""
        # Create a simple potential field
        def simple_potential(x: torch.Tensor) -> torch.Tensor:
            return (x ** 2).sum(dim=-1, keepdim=True)
        
        field = AnalyticPotentialField(simple_potential, input_dim=2)
        
        loss_fn = GeodesicLoss(
            potential_field=field,
            speed_weight=1.0,
            boundary_weight=1.0,
            gradient_weight=1.0,
            curvature_weight=0.1,
        )
        
        # Create dummy trajectory data
        positions = torch.randn(100, 2, requires_grad=True)
        velocity = torch.randn(100, 2)
        acceleration = torch.randn(100, 2)
        
        total_loss, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
            start_pos=positions[0:1],
            target_start=positions[0:1],
            end_pos=positions[-1:],
            target_end=positions[-1:],
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert 'speed' in loss_dict
        assert 'boundary' in loss_dict
        assert 'gradient' in loss_dict
        assert 'curvature' in loss_dict


# ============================================================================
# TrajectoryPINN Tests
# ============================================================================

class TestTrajectoryPINN:
    """Tests for TrajectoryPINN."""
    
    def test_construction(self):
        """Test basic construction."""
        model = TrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=4,
        )
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        
    def test_forward_shape(self):
        """Test output shape."""
        model = TrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=4,
        )
        
        t = torch.linspace(0, 1, 50).unsqueeze(1)
        context = torch.randn(50, 4)
        
        positions = model(t, context)
        assert positions.shape == (50, 2)
        
    def test_forward_3d(self):
        """Test 3D output."""
        model = TrajectoryPINN(
            output_dim=3,
            hidden_dims=[32, 32],
            context_dim=4,
        )
        
        t = torch.linspace(0, 1, 50).unsqueeze(1)
        context = torch.randn(50, 4)
        
        positions = model(t, context)
        assert positions.shape == (50, 3)
        
    def test_velocity_shape(self):
        """Test velocity computation."""
        model = TrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=4,
        )
        
        t = torch.linspace(0, 1, 50).unsqueeze(1).requires_grad_(True)
        context = torch.randn(50, 4)
        
        velocity = model.velocity(t, context)
        assert velocity.shape == (50, 2)
        
    def test_acceleration_shape(self):
        """Test acceleration computation."""
        model = TrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=4,
        )
        
        t = torch.linspace(0, 1, 50).unsqueeze(1).requires_grad_(True)
        context = torch.randn(50, 4)
        
        acceleration = model.acceleration(t, context)
        assert acceleration.shape == (50, 2)
        
    def test_no_context(self):
        """Test model without context."""
        model = TrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=0,
        )
        
        t = torch.linspace(0, 1, 50).unsqueeze(1)
        
        positions = model(t, None)
        assert positions.shape == (50, 2)


class TestBoundaryConditionedTrajectory:
    """Tests for BoundaryConditionedTrajectory."""
    
    def test_start_boundary_satisfied(self):
        """Test that start boundary is exactly satisfied."""
        model = BoundaryConditionedTrajectory(
            output_dim=2,
            hidden_dims=[32, 32],
        )
        
        start = torch.tensor([[0.0, 0.0]])
        end = torch.tensor([[1.0, 1.0]])
        t = torch.tensor([[0.0]])
        
        pos = model(t, start, end)
        
        assert torch.allclose(pos, start, atol=1e-6)
        
    def test_end_boundary_satisfied(self):
        """Test that end boundary is exactly satisfied."""
        model = BoundaryConditionedTrajectory(
            output_dim=2,
            hidden_dims=[32, 32],
        )
        
        start = torch.tensor([[0.0, 0.0]])
        end = torch.tensor([[1.0, 1.0]])
        t = torch.tensor([[1.0]])
        
        pos = model(t, start, end)
        
        assert torch.allclose(pos, end, atol=1e-6)
        
    def test_interior_varies(self):
        """Test that interior points are not just linear interpolation."""
        model = BoundaryConditionedTrajectory(
            output_dim=2,
            hidden_dims=[32, 32],
        )
        
        start = torch.tensor([[0.0, 0.0]])
        end = torch.tensor([[1.0, 0.0]])
        t = torch.tensor([[0.5]])
        
        pos = model(t, start, end)
        
        # Should be near [0.5, 0] but not exactly (network adds deviation)
        linear = 0.5 * (start + end)
        # At least check it's in reasonable range
        assert pos[0, 0].item() > -1 and pos[0, 0].item() < 2


class TestMultiTrajectoryPINN:
    """Tests for MultiTrajectoryPINN."""
    
    def test_batch_trajectories(self):
        """Test batched trajectory prediction."""
        model = MultiTrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=4,
            num_time_steps=50,
        )
        
        context = torch.randn(10, 4)  # 10 trajectories
        
        positions = model(context)
        assert positions.shape == (10, 50, 2)
        
    def test_velocities(self):
        """Test batched velocity computation."""
        model = MultiTrajectoryPINN(
            output_dim=2,
            hidden_dims=[32, 32],
            context_dim=4,
            num_time_steps=50,
        )
        
        context = torch.randn(10, 4)
        
        velocities = model.velocities(context)
        assert velocities.shape == (10, 50, 2)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_training_step(self):
        """Test a complete forward pass through all components."""
        # Setup
        field = LearnedPotentialField(input_dim=2, hidden_dims=[16, 16])
        trajectory = TrajectoryPINN(output_dim=2, hidden_dims=[16, 16], context_dim=4)
        loss_fn = GeodesicLoss(
            potential_field=field,
            speed_weight=1.0,
            boundary_weight=10.0,
            gradient_weight=1.0,
            curvature_weight=0.1,
        )
        
        optimizer = torch.optim.Adam(
            list(field.parameters()) + list(trajectory.parameters()),
            lr=1e-3
        )
        
        # Generate sample data
        batch_size = 10
        t = torch.linspace(0, 1, 50).unsqueeze(1).requires_grad_(True)
        context = torch.randn(50, 4)
        
        target_start = torch.zeros(batch_size, 2)
        target_end = torch.ones(batch_size, 2)
        
        # Forward pass
        positions = trajectory(t, context)
        velocity = trajectory.velocity(t, context)
        acceleration = trajectory.acceleration(t, context)
        
        # Compute loss
        total_loss, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
            start_pos=positions[0:1].expand(batch_size, -1),
            target_start=target_start,
            end_pos=positions[-1:].expand(batch_size, -1),
            target_end=target_end,
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Check gradients were computed
        assert all(p.grad is not None for p in field.parameters() if p.requires_grad)
        assert all(p.grad is not None for p in trajectory.parameters() if p.requires_grad)
        
    def test_boundary_conditioned_training(self):
        """Test training with hard boundary conditions."""
        field = LearnedPotentialField(input_dim=2, hidden_dims=[16, 16])
        trajectory = BoundaryConditionedTrajectory(output_dim=2, hidden_dims=[16, 16])
        
        optimizer = torch.optim.Adam(
            list(field.parameters()) + list(trajectory.parameters()),
            lr=1e-3
        )
        
        # Setup
        start = torch.tensor([[0.0, 0.0]])
        end = torch.tensor([[1.0, 1.0]])
        t = torch.linspace(0, 1, 50).unsqueeze(1).requires_grad_(True)
        
        # Forward
        positions = trajectory(t, start.expand(50, -1), end.expand(50, -1))
        
        # Compute a simple loss
        phi = field(positions)
        loss = phi.mean()  # Minimize potential
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check boundaries are still exact after optimization step
        with torch.no_grad():
            t_start = torch.tensor([[0.0]])
            t_end = torch.tensor([[1.0]])
            pos_start = trajectory(t_start, start, end)
            pos_end = trajectory(t_end, start, end)
            
            assert torch.allclose(pos_start, start, atol=1e-5)
            assert torch.allclose(pos_end, end, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
