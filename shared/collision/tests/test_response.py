"""Unit tests for CollisionResponse."""

import pytest
import torch

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.collision import CollisionDetector, CollisionResponse, SDFField
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


class TestPositionCorrection:
    def test_pushes_vertex_to_surface(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        response = CollisionResponse(stiffness=1000.0)
        
        # Vertex inside sphere at x=0.5
        cloth_vertices = torch.tensor([[0.5, 0.0, 0.0]], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        corrected = response.resolve_positions(cloth_vertices, result)
        
        # Should be pushed to surface at approximately x=1.0
        assert corrected[0, 0] > 0.9
        # Y and Z should remain near zero
        assert abs(corrected[0, 1]) < 0.1
        assert abs(corrected[0, 2]) < 0.1
    
    def test_does_not_modify_outside_vertices(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        response = CollisionResponse()
        
        # Vertex outside sphere
        cloth_vertices = torch.tensor([[2.0, 0.0, 0.0]], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        corrected = response.resolve_positions(cloth_vertices, result)
        
        # Should be unchanged
        assert torch.allclose(corrected, cloth_vertices, atol=1e-5)
    
    def test_max_correction_limit(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        response = CollisionResponse(max_correction=0.1)
        
        # Vertex deep inside sphere (center)
        cloth_vertices = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        corrected = response.resolve_positions(cloth_vertices, result)
        
        # Correction should be limited
        displacement = torch.norm(corrected - cloth_vertices)
        assert displacement <= 0.1 + 1e-5


class TestForceCorrection:
    def test_force_direction(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        response = CollisionResponse(stiffness=1000.0)
        
        # Vertex inside sphere
        cloth_vertices = torch.tensor([[0.5, 0.0, 0.0]], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        forces = response.resolve_forces(cloth_vertices, result)
        
        # Force should point in +x direction (out of sphere)
        assert forces[0, 0] > 0
    
    def test_force_magnitude_proportional_to_penetration(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        response = CollisionResponse(stiffness=100.0)
        
        # Two vertices at different depths
        cloth_vertices = torch.tensor([
            [0.5, 0.0, 0.0],   # 0.5 inside
            [0.8, 0.0, 0.0],   # 0.2 inside
        ], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        forces = response.resolve_forces(cloth_vertices, result)
        
        # Deeper vertex should have larger force
        force_mag_0 = torch.norm(forces[0])
        force_mag_1 = torch.norm(forces[1])
        assert force_mag_0 > force_mag_1


class TestFriction:
    def test_friction_reduces_tangential_velocity(self, simple_sphere_sdf, device):
        detector = CollisionDetector()
        response = CollisionResponse(friction=0.5)
        
        # Vertex on surface with tangential velocity
        cloth_vertices = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        cloth_velocities = torch.tensor([[0.0, 1.0, 0.0]], device=device)  # tangential
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        # Mark as in contact for friction to apply
        result.penetrating_mask[0] = True
        
        adjusted = response.apply_friction(cloth_velocities, result)
        
        # Tangential velocity should be reduced
        assert torch.norm(adjusted) < torch.norm(cloth_velocities)


class TestDamping:
    def test_damping_reduces_velocity_near_surface(self, simple_sphere_sdf, device):
        detector = CollisionDetector(proximity_threshold=0.2)
        response = CollisionResponse(damping=0.5)
        
        # Vertex near surface
        cloth_vertices = torch.tensor([[1.1, 0.0, 0.0]], device=device)
        cloth_velocities = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        
        result = detector.detect(cloth_vertices, simple_sphere_sdf)
        damped = response.apply_damping(cloth_velocities, result)
        
        # Velocity should be reduced
        assert torch.norm(damped) < torch.norm(cloth_velocities)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
