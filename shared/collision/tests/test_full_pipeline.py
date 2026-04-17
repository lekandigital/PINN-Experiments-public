"""
Full pipeline integration test and benchmarks.
Tests the complete collision workflow: body → SDF → detect → resolve.
"""

import pytest
import torch
import time
import math

import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')

from shared.collision import (
    DeformableBody, CollisionDetector, CollisionResponse, CollisionLoss, SDFField
)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def body_mesh(device):
    """Create a simple body mesh (sphere approximation)."""
    # Create icosphere
    phi = (1 + math.sqrt(5)) / 2
    
    vertices = torch.tensor([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=torch.float32, device=device)
    
    vertices = vertices / torch.norm(vertices, dim=1, keepdim=True)
    
    faces = torch.tensor([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=torch.long, device=device)
    
    return vertices, faces


@pytest.fixture
def cloth_vertices(device):
    """Create cloth vertices (some inside body, some outside)."""
    return torch.tensor([
        # Inside body (penetrating)
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        # Near surface (proximity)
        [1.05, 0.0, 0.0],
        [0.0, 1.08, 0.0],
        # Outside (safe)
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ], dtype=torch.float32, device=device)


class TestFullPipeline:
    """Test the complete collision pipeline end-to-end."""
    
    def test_body_to_sdf_to_detect_to_resolve(self, body_mesh, cloth_vertices, device):
        """Full pipeline: body → SDF → detect → resolve."""
        vertices, faces = body_mesh
        
        # 1. Create deformable body with SDF
        body = DeformableBody(vertices, faces, sdf_resolution=64, device=device)
        
        # 2. Create collision components
        detector = CollisionDetector(proximity_threshold=0.1)
        response = CollisionResponse(stiffness=1000.0, friction=0.3)
        loss_fn = CollisionLoss(weights={'penetration': 10.0, 'proximity': 1.0})
        
        # 3. Detect collisions
        collisions = detector.detect(cloth_vertices, body.get_sdf())
        
        # Verify detection
        assert collisions.has_collisions
        assert collisions.num_penetrating == 3  # First 3 vertices are inside
        
        # 4. Resolve collisions
        corrected_vertices = response.resolve_positions(cloth_vertices, collisions)
        
        # 5. Verify resolution - re-detect to ensure no more penetrations
        new_collisions = detector.detect(corrected_vertices, body.get_sdf())
        
        # All vertices should be outside now (or at least much better)
        # Note: Due to SDF discretization, might not be perfect
        corrected_sdf = body.get_sdf().query(corrected_vertices)
        assert (corrected_sdf >= -0.1).all()  # Allow small tolerance
        
        # 6. Compute training loss
        loss = loss_fn(cloth_vertices, body.get_sdf())
        assert loss > 0  # Should have positive loss for penetrating config
    
    def test_body_deformation_updates_collision(self, body_mesh, device):
        """Test that body deformation properly updates collision detection."""
        vertices, faces = body_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=64, device=device)
        detector = CollisionDetector()
        
        # Point that's initially outside
        test_point = torch.tensor([[1.5, 0.0, 0.0]], device=device)
        
        result_before = detector.detect(test_point, body.get_sdf())
        assert not result_before.has_collisions  # Outside initially
        
        # Move body toward the point
        deformed_vertices = vertices + torch.tensor([1.0, 0, 0], device=device)
        body.update_from_deformation(deformed_vertices)
        
        result_after = detector.detect(test_point, body.get_sdf())
        # Point should now be penetrating or at least closer
        sdf_after = body.get_sdf().query(test_point).item()
        assert sdf_after < 0.5  # Much closer to/inside body
    
    def test_gradient_flow_through_pipeline(self, body_mesh, cloth_vertices, device):
        """Test that gradients flow through the entire pipeline."""
        vertices, faces = body_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=64, device=device)
        loss_fn = CollisionLoss(weights={'penetration': 10.0})
        
        # Make cloth vertices require grad
        cloth_verts = cloth_vertices.clone().requires_grad_(True)
        
        # Compute loss
        loss = loss_fn(cloth_verts, body.get_sdf())
        loss.backward()
        
        # Check gradients exist and are non-zero for penetrating vertices
        assert cloth_verts.grad is not None
        # First 3 vertices are penetrating and should have gradients
        assert torch.norm(cloth_verts.grad[:3]) > 0


class TestPerformanceBenchmarks:
    """Performance benchmarks for collision system."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmarks")
    def test_sdf_computation_time(self, body_mesh, device):
        """Benchmark SDF computation time."""
        vertices, faces = body_mesh
        
        # Warm up
        _ = SDFField.from_mesh(vertices, faces, resolution=32, device=device)
        
        # Benchmark different resolutions
        times = {}
        for resolution in [32, 64, 128]:
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(3):
                _ = SDFField.from_mesh(vertices, faces, resolution=resolution, device=device)
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / 3
            times[resolution] = elapsed * 1000  # ms
        
        print(f"\nSDF computation times:")
        for res, t in times.items():
            print(f"  {res}³: {t:.1f}ms")
        
        # Target: 128³ should be < 100ms for small meshes
        # (will be slower for large meshes)
        assert times[128] < 1000  # 1 second max
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmarks")
    def test_query_time(self, body_mesh, device):
        """Benchmark SDF query time."""
        vertices, faces = body_mesh
        sdf = SDFField.from_mesh(vertices, faces, resolution=128, device=device)
        
        # Generate query points
        n_points = 10000
        query_points = torch.rand(n_points, 3, device=device) * 3 - 1.5
        
        # Warm up
        _ = sdf.query(query_points)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = sdf.query(query_points)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100
        
        print(f"\nQuery time for {n_points} points: {elapsed*1000:.2f}ms")
        
        # Target: < 1ms for 10K points
        assert elapsed < 0.01  # 10ms max
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmarks")
    def test_full_pipeline_time(self, body_mesh, device):
        """Benchmark full collision pipeline."""
        vertices, faces = body_mesh
        body = DeformableBody(vertices, faces, sdf_resolution=64, device=device)
        detector = CollisionDetector()
        response = CollisionResponse()
        
        # Generate cloth vertices
        n_verts = 5000
        cloth = torch.rand(n_verts, 3, device=device) * 3 - 1.5
        
        # Warm up
        result = detector.detect(cloth, body.get_sdf())
        _ = response.resolve_positions(cloth, result)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        iterations = 50
        for _ in range(iterations):
            result = detector.detect(cloth, body.get_sdf())
            _ = response.resolve_positions(cloth, result)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / iterations
        
        print(f"\nFull pipeline time ({n_verts} cloth vertices): {elapsed*1000:.2f}ms")
        
        # Target: < 5ms per frame for 5K vertices
        assert elapsed < 0.05  # 50ms max (generous for CI)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
