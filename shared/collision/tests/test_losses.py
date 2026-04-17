"""Unit tests for collision losses."""

import pytest
import torch

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.collision import (
    SDFField, CollisionLoss,
    penetration_loss, proximity_loss, contact_loss, eikonal_loss,
)
from shared.collision.sdf_field import SDFFieldData


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_sphere_sdf(device):
    """Analytical SDF for unit sphere."""
    resolution = 64
    lin = torch.linspace(-1.5, 1.5, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
    distances = torch.sqrt(grid_x**2 + grid_y**2 + grid_z**2) - 1.0
    
    data = SDFFieldData(
        grid=distances,
        bbox_min=torch.tensor([-1.5, -1.5, -1.5], device=device),
        bbox_max=torch.tensor([1.5, 1.5, 1.5], device=device),
        resolution=resolution,
        device=device,
    )
    return SDFField(data)


class TestPenetrationLoss:
    def test_positive_for_penetrating_vertices(self, simple_sphere_sdf, device):
        cloth_vertices = torch.tensor([
            [0.0, 0.0, 0.0],   # inside
            [0.5, 0.0, 0.0],   # inside
        ], device=device)
        
        loss = penetration_loss(cloth_vertices, simple_sphere_sdf)
        
        assert loss > 0
    
    def test_zero_for_outside_vertices(self, simple_sphere_sdf, device):
        cloth_vertices = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ], device=device)
        
        loss = penetration_loss(cloth_vertices, simple_sphere_sdf)
        
        assert loss < 1e-5
    
    def test_gradient_flows_to_vertices(self, simple_sphere_sdf, device):
        cloth_vertices = torch.tensor([
            [0.5, 0.0, 0.0],
        ], device=device, requires_grad=True)
        
        loss = penetration_loss(cloth_vertices, simple_sphere_sdf)
        loss.backward()
        
        assert cloth_vertices.grad is not None
        assert torch.norm(cloth_vertices.grad) > 0


class TestProximityLoss:
    def test_positive_within_margin(self, simple_sphere_sdf, device):
        # Vertex just outside surface but within margin
        cloth_vertices = torch.tensor([[1.05, 0.0, 0.0]], device=device)
        
        loss = proximity_loss(cloth_vertices, simple_sphere_sdf, margin=0.1)
        
        assert loss > 0
    
    def test_zero_far_from_surface(self, simple_sphere_sdf, device):
        cloth_vertices = torch.tensor([[2.0, 0.0, 0.0]], device=device)
        
        loss = proximity_loss(cloth_vertices, simple_sphere_sdf, margin=0.1)
        
        assert loss < 1e-5


class TestContactLoss:
    def test_zero_when_on_surface(self, simple_sphere_sdf, device):
        cloth_vertices = torch.tensor([
            [1.0, 0.0, 0.0],   # on surface
            [0.0, 1.0, 0.0],   # on surface
        ], device=device)
        contact_mask = torch.ones(2, dtype=torch.bool, device=device)
        
        loss = contact_loss(cloth_vertices, simple_sphere_sdf, contact_mask)
        
        # Should be close to zero (within interpolation error)
        assert loss < 0.05
    
    def test_positive_when_away_from_surface(self, simple_sphere_sdf, device):
        cloth_vertices = torch.tensor([
            [1.5, 0.0, 0.0],   # away from surface
        ], device=device)
        contact_mask = torch.ones(1, dtype=torch.bool, device=device)
        
        loss = contact_loss(cloth_vertices, simple_sphere_sdf, contact_mask)
        
        assert loss > 0


class TestEikonalLoss:
    def test_zero_for_unit_gradients(self, device):
        # Gradients with magnitude 1
        torch.manual_seed(42)
        gradients = torch.randn(100, 3, device=device)
        gradients = gradients / torch.norm(gradients, dim=-1, keepdim=True)
        
        loss = eikonal_loss(gradients)
        
        assert loss < 1e-5
    
    def test_positive_for_non_unit_gradients(self, device):
        # Gradients with magnitude 0.5
        torch.manual_seed(42)
        gradients = torch.randn(100, 3, device=device)
        gradients = 0.5 * gradients / torch.norm(gradients, dim=-1, keepdim=True)
        
        loss = eikonal_loss(gradients)
        
        assert loss > 0.2  # (0.5 - 1)^2 = 0.25


class TestCollisionLossModule:
    def test_combined_loss(self, simple_sphere_sdf, device):
        loss_fn = CollisionLoss(weights={'penetration': 10.0, 'proximity': 1.0})
        
        # Mixed vertices
        cloth_vertices = torch.tensor([
            [0.5, 0.0, 0.0],   # inside
            [1.05, 0.0, 0.0],  # just outside
        ], device=device)
        
        loss = loss_fn(cloth_vertices, simple_sphere_sdf)
        
        assert loss > 0
    
    def test_detailed_losses(self, simple_sphere_sdf, device):
        loss_fn = CollisionLoss()
        
        cloth_vertices = torch.tensor([
            [0.5, 0.0, 0.0],
        ], device=device)
        
        losses = loss_fn.forward_detailed(cloth_vertices, simple_sphere_sdf)
        
        assert 'penetration' in losses
        assert 'proximity' in losses
        assert 'total' in losses


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
