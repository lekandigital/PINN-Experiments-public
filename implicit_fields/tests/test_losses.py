"""
Tests for physics-informed loss functions.

Tests cover:
- Eikonal loss (unit gradient constraint)
- Collision loss (SDF interpenetration)
- Gradient utility function
- Divergence-free loss
"""

import pytest
import torch
import torch.nn as nn

from implicit_fields.losses import (
    gradient,
    eikonal_loss,
    sdf_collision_loss,
    sdf_boundary_loss,
    divergence_free_loss,
    laplacian_loss,
    CombinedSDFLoss,
)


class TestGradientFunction:
    """Tests for the gradient utility function."""
    
    def test_basic_gradient(self):
        """Test basic gradient computation."""
        x = torch.randn(100, 3, requires_grad=True)
        # y = sum(x^2) for each sample -> gradient should be 2x
        y = (x ** 2).sum(dim=-1, keepdim=True)
        
        grads = gradient(y, x)
        
        # Gradient of x^2 is 2x
        expected = 2 * x
        assert torch.allclose(grads, expected, atol=1e-5)
    
    def test_gradient_shape(self):
        """Test gradient output shape."""
        x = torch.randn(100, 3, requires_grad=True)
        y = x.sum(dim=-1, keepdim=True)  # (100, 1)
        
        grads = gradient(y, x)
        assert grads.shape == x.shape
    
    def test_gradient_with_network(self):
        """Test gradient computation through a neural network."""
        model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        x = torch.randn(50, 3, requires_grad=True)
        y = model(x)
        
        grads = gradient(y, x)
        assert grads.shape == (50, 3)
        assert not torch.isnan(grads).any()
    
    def test_gradient_create_graph(self):
        """Test that create_graph=True allows second derivatives."""
        x = torch.randn(10, 3, requires_grad=True)
        y = (x ** 3).sum(dim=-1, keepdim=True)  # y = sum(x^3)
        
        # First gradient: 3x^2
        grads = gradient(y, x, create_graph=True)
        
        # Second gradient should be possible
        second_grads = gradient(grads.sum(), x, create_graph=False)
        
        # Gradient of 3x^2 is 6x
        expected = 6 * x
        assert torch.allclose(second_grads, expected, atol=1e-4)


class TestEikonalLoss:
    """Tests for eikonal loss function."""
    
    def test_unit_gradient_zero_loss(self):
        """Test that unit norm gradients give zero loss."""
        # Create gradients with unit norm
        gradients = torch.randn(100, 3)
        gradients = gradients / gradients.norm(dim=-1, keepdim=True)
        
        loss = eikonal_loss(gradients)
        
        assert loss < 1e-6
    
    def test_non_unit_gradient_positive_loss(self):
        """Test that non-unit gradients give positive loss."""
        # Gradients with norm 2.0
        gradients = torch.randn(100, 3)
        gradients = 2.0 * gradients / gradients.norm(dim=-1, keepdim=True)
        
        loss = eikonal_loss(gradients)
        
        # Expected: mean((2 - 1)^2) = 1
        assert abs(loss - 1.0) < 0.1
    
    def test_reduction_none(self):
        """Test reduction='none' returns per-sample losses."""
        gradients = torch.randn(100, 3)
        gradients = gradients / gradients.norm(dim=-1, keepdim=True)
        
        loss = eikonal_loss(gradients, reduction='none')
        
        assert loss.shape == (100,)
    
    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses."""
        gradients = torch.randn(100, 3)
        
        loss_none = eikonal_loss(gradients, reduction='none')
        loss_sum = eikonal_loss(gradients, reduction='sum')
        
        assert torch.allclose(loss_sum, loss_none.sum())
    
    def test_eikonal_gradient_flow(self):
        """Test that eikonal loss gradients flow correctly."""
        model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        x = torch.randn(50, 3, requires_grad=True)
        y = model(x)
        
        grads = gradient(y, x, create_graph=True)
        loss = eikonal_loss(grads)
        
        # Should be able to backprop through eikonal loss
        loss.backward()
        
        # At least one model parameter should have gradients (bias might not always)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "At least one parameter should have gradients"
    
    def test_numerical_stability_small_gradients(self):
        """Test numerical stability with near-zero gradients."""
        gradients = torch.randn(100, 3) * 1e-8
        
        loss = eikonal_loss(gradients)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestSDFCollisionLoss:
    """Tests for SDF collision loss."""
    
    def test_no_collision_zero_loss(self):
        """Test zero loss when objects don't interpenetrate."""
        # SDF_a positive (outside A), SDF_b positive (outside B)
        sdf_a = torch.ones(100, 1)  # All outside
        sdf_b = torch.ones(100, 1)
        
        loss = sdf_collision_loss(sdf_a, sdf_b)
        assert loss == 0.0
    
    def test_collision_positive_loss(self):
        """Test positive loss when objects interpenetrate."""
        # Both negative = inside both objects = collision
        sdf_a = -torch.ones(100, 1)  # All inside A
        sdf_b = -torch.ones(100, 1)  # All inside B
        
        loss = sdf_collision_loss(sdf_a, sdf_b)
        
        # Expected: mean(1 * 1) = 1
        assert abs(loss - 1.0) < 1e-6
    
    def test_partial_collision(self):
        """Test partial collision (some points inside, some outside)."""
        sdf_a = torch.cat([
            -torch.ones(50, 1),   # Inside A
            torch.ones(50, 1),    # Outside A
        ])
        sdf_b = torch.cat([
            -torch.ones(50, 1),   # Inside B (collision with first 50)
            -torch.ones(50, 1),   # Inside B but A is outside (no collision)
        ])
        
        loss = sdf_collision_loss(sdf_a, sdf_b)
        
        # Only first 50 points contribute: mean of [1*1, ..., 0*1, ...]
        # = (50 * 1 + 50 * 0) / 100 = 0.5
        assert abs(loss - 0.5) < 1e-6
    
    def test_1d_input(self):
        """Test with 1D tensors instead of 2D."""
        sdf_a = -torch.ones(100)  # Shape (100,)
        sdf_b = -torch.ones(100)
        
        loss = sdf_collision_loss(sdf_a, sdf_b)
        assert abs(loss - 1.0) < 1e-6
    
    def test_gradient_flow(self):
        """Test gradient flow through collision loss."""
        sdf_a = torch.randn(50, 1, requires_grad=True)
        sdf_b = torch.randn(50, 1, requires_grad=True)
        
        loss = sdf_collision_loss(sdf_a, sdf_b)
        loss.backward()
        
        assert sdf_a.grad is not None
        assert sdf_b.grad is not None


class TestSDFBoundaryLoss:
    """Tests for SDF boundary loss."""
    
    def test_perfect_prediction(self):
        """Test zero loss for perfect predictions."""
        pred = torch.zeros(100, 1)  # Predicted as surface
        target = torch.zeros(100, 1)  # Ground truth surface
        
        loss = sdf_boundary_loss(pred, target)
        assert loss < 1e-10
    
    def test_positive_loss(self):
        """Test positive loss for wrong predictions."""
        pred = torch.ones(100, 1)  # Predicted outside
        target = torch.zeros(100, 1)  # Should be on surface
        
        loss = sdf_boundary_loss(pred, target)
        
        # MSE of (1 - 0)^2 = 1
        assert abs(loss - 1.0) < 1e-6


class TestDivergenceFreeLoss:
    """Tests for divergence-free loss."""
    
    def test_divergence_free_field(self):
        """Test that a divergence-free field has low loss."""
        # Use identity function: field = coords, which has divergence 3
        # Instead test a known divergence-free field: f(x) = (y, -x, 0)
        # has div = ∂y/∂x + ∂(-x)/∂y + 0 = 0 + 0 + 0 = 0
        
        class DivFreeField(nn.Module):
            def forward(self, x):
                # f(x,y,z) = (y, -x, 0) which is divergence-free
                return torch.stack([x[:, 1], -x[:, 0], torch.zeros_like(x[:, 0])], dim=-1)
        
        model = DivFreeField()
        coords = torch.randn(50, 3, requires_grad=True)
        field = model(coords)
        
        loss = divergence_free_loss(field, coords)
        
        # Should be near zero
        assert loss < 1e-4
    
    def test_non_divergence_free_field(self):
        """Test that a field with divergence has positive loss."""
        # field = (x, y, z) has divergence = 1 + 1 + 1 = 3
        coords = torch.randn(50, 3, requires_grad=True)
        field = coords.clone()  # f(x) = x
        
        loss = divergence_free_loss(field, coords)
        
        # Divergence = 3, loss = 9
        assert loss > 1.0
    
    def test_dimension_mismatch_error(self):
        """Test error when field and coords have different dimensions."""
        coords = torch.randn(50, 3, requires_grad=True)
        field = torch.randn(50, 2)  # Wrong dimension
        
        with pytest.raises(ValueError):
            divergence_free_loss(field, coords)

class TestLaplacianLoss:
    """Tests for Laplacian regularization loss."""
    
    def test_quadratic_function_nonzero_laplacian(self):
        """Test that quadratic functions have non-zero Laplacian."""
        # f(x) = x^2 + y^2 + z^2 has Laplacian = 2 + 2 + 2 = 6
        coords = torch.randn(30, 3, requires_grad=True)
        sdf = (coords ** 2).sum(dim=-1, keepdim=True)
        
        loss = laplacian_loss(sdf, coords)
        
        # Laplacian = 6, loss = 36
        assert loss > 10.0
    
    def test_laplacian_loss_computes(self):
        """Test that laplacian loss can be computed through a network."""
        model = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        
        coords = torch.randn(20, 3, requires_grad=True)
        sdf = model(coords)
        
        loss = laplacian_loss(sdf, coords)
        
        # Should produce a finite loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0


class TestCombinedSDFLoss:
    """Tests for CombinedSDFLoss module."""
    
    def test_basic_usage(self):
        """Test basic usage of combined loss."""
        loss_fn = CombinedSDFLoss(w_data=1.0, w_eikonal=0.1)
        
        model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        coords = torch.randn(50, 3, requires_grad=True)
        sdf_pred = model(coords)
        sdf_target = torch.zeros(50, 1)
        
        total_loss, loss_dict = loss_fn(sdf_pred, sdf_target, coords)
        
        assert 'total' in loss_dict
        assert 'data' in loss_dict
        assert 'eikonal' in loss_dict
        assert total_loss == loss_dict['total']
    
    def test_weights_applied(self):
        """Test that weights are applied correctly."""
        loss_fn = CombinedSDFLoss(w_data=2.0, w_eikonal=0.5)
        
        coords = torch.randn(50, 3, requires_grad=True)
        sdf_pred = coords[:, 0:1]  # Simple function that depends on coords
        sdf_target = torch.zeros(50, 1)
        
        total, losses = loss_fn(sdf_pred, sdf_target, coords)
        
        expected = 2.0 * losses['data'] + 0.5 * losses['eikonal']
        assert torch.allclose(total, expected)
    
    def test_optional_collision_loss(self):
        """Test collision loss when obstacle SDF is provided."""
        loss_fn = CombinedSDFLoss(w_data=1.0, w_eikonal=0.0, w_collision=1.0)  # Disable eikonal for simpler test
        
        coords = torch.randn(50, 3)  # No grad needed since eikonal is disabled
        sdf_pred = -torch.ones(50, 1)  # Inside
        sdf_target = torch.zeros(50, 1)
        sdf_obstacle = -torch.ones(50, 1)  # Also inside = collision
        
        total, losses = loss_fn(sdf_pred, sdf_target, coords, sdf_obstacle)
        
        assert 'collision' in losses
        assert losses['collision'] > 0


class TestLossesCPU:
    """Test losses work on CPU without CUDA."""
    
    def test_eikonal_cpu(self):
        """Test eikonal loss on CPU."""
        gradients = torch.randn(100, 3)
        loss = eikonal_loss(gradients)
        
        assert loss.device.type == 'cpu'
    
    def test_collision_cpu(self):
        """Test collision loss on CPU."""
        sdf_a = torch.randn(100)
        sdf_b = torch.randn(100)
        loss = sdf_collision_loss(sdf_a, sdf_b)
        
        assert loss.device.type == 'cpu'
    
    def test_gradient_cpu(self):
        """Test gradient computation on CPU."""
        x = torch.randn(50, 3, requires_grad=True)
        y = (x ** 2).sum(dim=-1, keepdim=True)
        
        grads = gradient(y, x)
        
        assert grads.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
