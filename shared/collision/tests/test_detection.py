"""Unit tests for CollisionDetector."""

import pytest
import torch

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.collision import SDFField, CollisionDetector, CollisionResult
from shared.collision.sdf_field import SDFFieldData


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_sphere_sdf(device):
    """Create SDF for unit sphere centered at origin."""
    resolution = 64
    lin = torch.linspace(-1.5, 1.5, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
    
    # Analytical SDF for unit sphere: |p| - 1
    distances = torch.sqrt(grid_x**2 + grid_y**2 + grid_z**2) - 1.0
    
    data = SDFFieldData(
        grid=distances,
        bbox_min=torch.tensor([-1.5, -1.5, -1.5], device=device),
        bbox_max=torch.tensor([1.5, 1.5, 1.5], device=device),
        resolution=resolution,
        device=device,
    )
    return SDFField(data)


class TestCollisionDetection:
    def test_detects_penetrating_vertices(self, simple_sphere_sdf, device):
        detector = CollisionDetector(proximity_threshold=0.1)
        
        # Vertices inside sphere (penetrating)
        cloth_vertices = torch.tensor([
            [0.0, 0.0, 0.0],   # center - deep inside
            [0.5, 0.0, 0.0],   # 0.5 from center - inside
            [0.0, 0.3, 0.0],   # inside
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        
        assert result.has_collisions
        assert result.num_penetrating == 3
        assert result.penetrating_mask.all()
    
    def test_non_penetrating_vertices(self, simple_sphere_sdf, device):
        detector = CollisionDetector(proximity_threshold=0.05)
        
        # Vertices outside sphere
        cloth_vertices = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.3],
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        
        assert not result.has_collisions
        assert result.num_penetrating == 0
        assert not result.penetrating_mask.any()
    
    def test_proximity_detection(self, simple_sphere_sdf, device):
        detector = CollisionDetector(proximity_threshold=0.2)
        
        # Vertices just outside sphere surface (within proximity)
        cloth_vertices = torch.tensor([
            [1.1, 0.0, 0.0],   # 0.1 outside surface
            [0.0, 1.05, 0.0],  # 0.05 outside surface
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        
        assert not result.has_collisions  # Not penetrating
        assert result.has_proximity        # But within proximity
        assert result.proximity_mask.any()
    
    def test_mixed_vertices(self, simple_sphere_sdf, device):
        detector = CollisionDetector(proximity_threshold=0.1)
        
        cloth_vertices = torch.tensor([
            [0.0, 0.0, 0.0],   # inside
            [2.0, 0.0, 0.0],   # outside far
            [1.05, 0.0, 0.0],  # outside close (proximity)
            [0.5, 0.0, 0.0],   # inside
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        
        assert result.num_penetrating == 2
        assert result.penetrating_mask[0] == True
        assert result.penetrating_mask[1] == False
        assert result.penetrating_mask[3] == True
    
    def test_surface_normals_point_outward(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        
        # Vertex inside sphere
        cloth_vertices = torch.tensor([
            [0.5, 0.0, 0.0],
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf, compute_normals=True)
        
        # Normal should point in +x direction (outward from center)
        normal = result.surface_normals[0]
        assert normal[0] > 0.9  # Mostly in +x direction


class TestCollisionResult:
    def test_get_penetrating_vertices(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        
        cloth_vertices = torch.tensor([
            [0.0, 0.0, 0.0],   # inside
            [2.0, 0.0, 0.0],   # outside
            [0.5, 0.0, 0.0],   # inside
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        penetrating = result.get_penetrating_vertices(cloth_vertices)
        
        assert penetrating.shape[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
