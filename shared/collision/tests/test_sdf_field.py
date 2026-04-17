"""Unit tests for SDFField."""

import pytest
import torch
import math

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.collision import SDFField
from shared.collision.sdf_field import SDFFieldData


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def unit_sphere_mesh(device):
    """Create an icosphere approximation of unit sphere."""
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    
    # Icosahedron vertices
    vertices = torch.tensor([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=torch.float32, device=device)
    
    # Normalize to unit sphere
    vertices = vertices / torch.norm(vertices, dim=1, keepdim=True)
    
    # Icosahedron faces
    faces = torch.tensor([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=torch.long, device=device)
    
    return vertices, faces


@pytest.fixture
def simple_sphere_sdf(device):
    """Analytical SDF for unit sphere (more accurate for testing)."""
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


class TestSDFFieldConstruction:
    def test_from_mesh_creates_valid_sdf(self, unit_sphere_mesh, device):
        vertices, faces = unit_sphere_mesh
        sdf = SDFField.from_mesh(vertices, faces, resolution=32)
        
        assert sdf.resolution == 32
        assert sdf.device == device
        assert sdf.grid.shape == (32, 32, 32)
    
    def test_bbox_contains_mesh(self, unit_sphere_mesh, device):
        vertices, faces = unit_sphere_mesh
        sdf = SDFField.from_mesh(vertices, faces, resolution=32, padding=0.1)
        
        bbox_min, bbox_max = sdf.bbox
        assert (vertices >= bbox_min).all()
        assert (vertices <= bbox_max).all()


class TestSDFFieldQueries:
    def test_center_of_sphere_is_inside(self, simple_sphere_sdf, device):
        center = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        distance = simple_sphere_sdf.query(center)
        
        # Center should be inside (negative SDF), roughly -1.0 for unit sphere
        assert distance.item() < 0
        assert abs(distance.item() + 1.0) < 0.1  # Allow some discretization error
    
    def test_point_outside_sphere_is_positive(self, simple_sphere_sdf, device):
        outside = torch.tensor([[0.0, 0.0, 2.0]], device=device)
        distance = simple_sphere_sdf.query(outside)
        
        # Point at distance 2 from center, sphere radius ~1, so SDF ~1
        assert distance.item() > 0
        assert abs(distance.item() - 1.0) < 0.2
    
    def test_point_on_surface_is_near_zero(self, simple_sphere_sdf, device):
        on_surface = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        distance = simple_sphere_sdf.query(on_surface)
        
        # Should be close to zero
        assert abs(distance.item()) < 0.1
    
    def test_batched_queries(self, simple_sphere_sdf, device):
        # Batched points
        points = torch.tensor([
            [0.0, 0.0, 0.0],  # inside
            [0.0, 0.0, 2.0],  # outside
            [1.0, 0.0, 0.0],  # on surface
        ], device=device)
        
        distances = simple_sphere_sdf.query(points)
        
        assert distances.shape == (3,)
        assert distances[0] < 0  # inside
        assert distances[1] > 0  # outside


class TestSDFGradient:
    def test_gradient_points_outward_for_sphere(self, simple_sphere_sdf, device):
        # Points near surface
        points = torch.tensor([
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ], device=device)
        
        gradients = simple_sphere_sdf.gradient(points)
        
        # For a sphere, gradient should point radially outward
        # i.e., gradient / |gradient| ≈ point / |point|
        normalized_grads = gradients / torch.norm(gradients, dim=-1, keepdim=True)
        normalized_points = points / torch.norm(points, dim=-1, keepdim=True)
        
        # Should be roughly aligned (dot product close to 1)
        dots = (normalized_grads * normalized_points).sum(dim=-1)
        assert (dots > 0.8).all()
    
    def test_gradient_magnitude_near_one(self, simple_sphere_sdf, device):
        """Eikonal property: |∇SDF| ≈ 1"""
        # Random points in bbox
        torch.manual_seed(42)
        points = torch.rand(100, 3, device=device) * 2 - 1
        gradients = simple_sphere_sdf.gradient(points)
        
        grad_norms = torch.norm(gradients, dim=-1)
        
        # Most gradients should be close to 1
        # (some discretization error at boundaries is acceptable)
        assert (grad_norms > 0.5).float().mean() > 0.8


class TestSDFDeviceMovement:
    def test_to_device(self, unit_sphere_mesh):
        vertices, faces = unit_sphere_mesh
        sdf = SDFField.from_mesh(vertices.cpu(), faces.cpu(), resolution=32)
        
        sdf_cpu = sdf.to(torch.device('cpu'))
        assert sdf_cpu.device == torch.device('cpu')
        assert sdf_cpu.grid.device == torch.device('cpu')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
