"""Unit tests for DeformableBody interface."""

import pytest
import torch

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.collision import DeformableBody, SDFField
from shared.collision.sdf_field import SDFFieldData


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_cube_mesh(device):
    """Create a simple cube mesh."""
    vertices = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=torch.float32, device=device)
    
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 6, 5], [4, 7, 6],  # back
        [0, 4, 5], [0, 5, 1],  # bottom
        [2, 6, 7], [2, 7, 3],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=torch.long, device=device)
    
    return vertices, faces


class TestDeformableBodyInit:
    def test_creates_sdf_on_init(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=32, device=device)
        
        sdf = body.get_sdf()
        assert sdf is not None
        assert sdf.resolution == 32
    
    def test_stores_vertices_and_faces(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, device=device)
        
        assert body.num_vertices == 8
        assert torch.allclose(body.vertices, vertices)


class TestDeformableBodyUpdate:
    def test_update_from_deformation(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=32, device=device)
        
        # Move vertices
        new_vertices = vertices + torch.tensor([1, 0, 0], device=device)
        body.update_from_deformation(new_vertices)
        
        assert torch.allclose(body.vertices, new_vertices)
    
    def test_velocity_estimation(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=32, device=device)
        
        # Move vertices by 1 unit in x direction over dt=1
        new_vertices = vertices + torch.tensor([1, 0, 0], device=device)
        body.update_from_deformation(new_vertices, dt=1.0)
        
        # Velocity should be approximately [1, 0, 0]
        velocity = body.velocity
        assert velocity is not None
        assert velocity[0, 0] > 0.5  # x velocity positive


class TestDeformableBodySDF:
    def test_sdf_updates_after_deformation(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=32, device=device)
        
        # Query a point that's outside the initial cube
        point = torch.tensor([[3.0, 0.0, 0.0]], device=device)
        sdf_before = body.get_sdf().query(point).item()
        
        # Move cube toward the point
        new_vertices = vertices + torch.tensor([2, 0, 0], device=device)
        body.update_from_deformation(new_vertices)
        
        sdf_after = body.get_sdf().query(point).item()
        
        # Point should be closer to (or inside) the body after moving
        assert sdf_after < sdf_before


class TestDeformableBodyVelocityField:
    def test_velocity_field_query(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=32, device=device)
        
        # Move vertices
        new_vertices = vertices + torch.tensor([1, 0, 0], device=device)
        body.update_from_deformation(new_vertices, dt=1.0)
        
        # Query velocity at a point
        query_points = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        velocities = body.get_velocity_field(query_points)
        
        assert velocities.shape == (1, 3)


class TestDeformableBodyReset:
    def test_reset_returns_to_rest_pose(self, simple_cube_mesh, device):
        vertices, faces = simple_cube_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=32, device=device)
        
        # Move vertices
        new_vertices = vertices + torch.tensor([5, 5, 5], device=device)
        body.update_from_deformation(new_vertices)
        
        # Reset
        body.reset()
        
        assert torch.allclose(body.vertices, vertices)
        assert body.velocity is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
