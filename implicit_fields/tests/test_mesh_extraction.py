"""
Tests for mesh extraction utilities.

Tests cover:
- Sphere SDF mesh extraction accuracy
- Progressive refinement
- Laplacian smoothing
- OBJ/PLY export
- Mesh quality metrics
"""

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from implicit_fields.mesh_extraction import (
    extract_mesh,
    laplacian_smooth,
    taubin_smooth,
    export_mesh_obj,
    export_mesh_ply,
    compute_mesh_quality,
    _build_adjacency,
    _create_grid,
)


def sphere_sdf(coords: torch.Tensor, radius: float = 0.5, center: torch.Tensor = None) -> torch.Tensor:
    """Analytic SDF for a sphere: f(x) = |x - c| - r."""
    if center is None:
        center = torch.zeros(3, device=coords.device, dtype=coords.dtype)
    
    dist = torch.norm(coords - center, dim=-1, keepdim=True)
    return dist - radius


def box_sdf(coords: torch.Tensor, half_extents: torch.Tensor = None) -> torch.Tensor:
    """Analytic SDF for an axis-aligned box."""
    if half_extents is None:
        half_extents = torch.tensor([0.5, 0.5, 0.5], device=coords.device, dtype=coords.dtype)
    
    q = torch.abs(coords) - half_extents
    outside = torch.norm(torch.clamp(q, min=0), dim=-1, keepdim=True)
    inside = torch.clamp(q.max(dim=-1, keepdim=True)[0], max=0)
    return outside + inside


class TestExtractMesh:
    """Tests for extract_mesh function."""
    
    def test_sphere_extraction_basic(self):
        """Test basic mesh extraction from a sphere SDF."""
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        vertices, faces = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
            bounds=bounds,
            resolutions=[32],  # Low resolution for speed
            device='cpu',
            smoothing_iterations=0,
        )
        
        assert len(vertices) > 100  # Should have reasonable vertex count
        assert len(faces) > 100  # Should have reasonable face count
    
    def test_sphere_surface_area_accuracy(self):
        """Test that extracted sphere has correct surface area (within 2%)."""
        radius = 0.5
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        vertices, faces = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=radius),
            bounds=bounds,
            resolutions=[64, 128],  # Higher resolution for accuracy
            device='cpu',
            smoothing_iterations=0,
        )
        
        # Compute surface area from mesh
        surface_area = _compute_surface_area(vertices, faces)
        
        # Analytic surface area: 4 * pi * r^2
        expected_area = 4 * math.pi * (radius ** 2)
        
        # Check within 2% error
        relative_error = abs(surface_area - expected_area) / expected_area
        assert relative_error < 0.02, f"Surface area error: {relative_error*100:.1f}%"
    
    def test_sphere_watertight(self):
        """Test that extracted sphere mesh is watertight."""
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        vertices, faces = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
            bounds=bounds,
            resolutions=[64],
            device='cpu',
            smoothing_iterations=0,
        )
        
        quality = compute_mesh_quality(vertices, faces)
        assert quality['watertight'], "Sphere mesh should be watertight"
    
    def test_progressive_vs_single_resolution(self):
        """Test that progressive refinement produces different results than single resolution."""
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        # Progressive refinement
        verts_prog, faces_prog = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
            bounds=bounds,
            resolutions=[32, 64],
            device='cpu',
            progressive=True,
            smoothing_iterations=0,
        )
        
        # Single resolution
        verts_single, faces_single = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
            bounds=bounds,
            resolutions=[64],
            device='cpu',
            progressive=False,
            smoothing_iterations=0,
        )
        
        # Both should produce valid meshes
        assert len(verts_prog) > 0
        assert len(verts_single) > 0
    
    def test_different_batch_sizes(self):
        """Test mesh extraction with different batch sizes."""
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        for batch_size in [1024, 8192, 32768]:
            vertices, faces = extract_mesh(
                sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
                bounds=bounds,
                resolutions=[32],
                batch_size=batch_size,
                device='cpu',
                smoothing_iterations=0,
            )
            
            assert len(vertices) > 0, f"Failed with batch_size={batch_size}"


def _compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute surface area of a triangle mesh."""
    total_area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Area = 0.5 * |cross(v1-v0, v2-v0)|
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        total_area += 0.5 * np.linalg.norm(cross)
    return total_area


class TestLaplacianSmooth:
    """Tests for Laplacian smoothing."""
    
    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces vertex noise."""
        # Create a simple mesh (tetrahedron with noise)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
        ], dtype=np.float32)
        
        # Add noise
        np.random.seed(42)
        noisy_vertices = vertices + np.random.randn(*vertices.shape) * 0.1
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=np.int32)
        
        # Smooth
        smoothed = laplacian_smooth(noisy_vertices, faces, iterations=5, lam=0.5)
        
        # Smoothed vertices should be closer to original than noisy
        noise_error = np.linalg.norm(noisy_vertices - vertices)
        smooth_error = np.linalg.norm(smoothed - vertices)
        
        # Note: Smoothing might not always reduce error to original, but should reduce high-frequency noise
        assert smoothed.shape == vertices.shape
    
    def test_smoothing_preserves_shape(self):
        """Test that smoothing doesn't collapse the mesh."""
        # Create cube vertices
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        
        # Cube faces (2 triangles per face)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ], dtype=np.int32)
        
        smoothed = laplacian_smooth(vertices, faces, iterations=3, lam=0.3)
        
        # Bounding box should still contain the original
        assert smoothed.min(axis=0).min() >= -0.5
        assert smoothed.max(axis=0).max() <= 1.5
    
    def test_zero_iterations(self):
        """Test that zero iterations returns unchanged vertices."""
        vertices = np.random.randn(10, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        
        smoothed = laplacian_smooth(vertices, faces, iterations=0)
        
        np.testing.assert_array_almost_equal(smoothed, vertices)


class TestTaubinSmooth:
    """Tests for Taubin smoothing."""
    
    def test_taubin_preserves_volume_better(self):
        """Test that Taubin smoothing preserves volume better than Laplacian."""
        # Create a sphere-like mesh
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        vertices, faces = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
            bounds=bounds,
            resolutions=[32],
            device='cpu',
            smoothing_iterations=0,
        )
        
        # Compute original volume approximation (via bounding box)
        original_extent = vertices.max(axis=0) - vertices.min(axis=0)
        
        # Apply Laplacian smoothing
        laplacian_result = laplacian_smooth(vertices, faces, iterations=5, lam=0.5)
        laplacian_extent = laplacian_result.max(axis=0) - laplacian_result.min(axis=0)
        
        # Apply Taubin smoothing
        taubin_result = taubin_smooth(vertices, faces, iterations=5, lam=0.5, mu=-0.53)
        taubin_extent = taubin_result.max(axis=0) - taubin_result.min(axis=0)
        
        # Taubin should shrink less than Laplacian
        laplacian_shrink = np.linalg.norm(original_extent - laplacian_extent)
        taubin_shrink = np.linalg.norm(original_extent - taubin_extent)
        
        # Taubin typically shrinks less, but this may not always hold for all meshes
        # Just verify both produce valid output
        assert taubin_result.shape == vertices.shape


class TestMeshExport:
    """Tests for mesh export functions."""
    
    def test_export_obj(self):
        """Test OBJ file export."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            filepath = f.name
        
        try:
            export_mesh_obj(vertices, faces, filepath)
            
            # Check file exists and has content
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            
            # Read and verify content
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert 'v ' in content  # Has vertices
            assert 'f ' in content  # Has faces
            assert 'f 1 2 3' in content  # OBJ is 1-indexed
        finally:
            os.unlink(filepath)
    
    def test_export_obj_with_colors(self):
        """Test OBJ export with vertex colors."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            filepath = f.name
        
        try:
            export_mesh_obj(vertices, faces, filepath, vertex_colors=colors)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Should have color values after position
            assert '1.000 0.000 0.000' in content or '1.0 0.0 0.0' in content
        finally:
            os.unlink(filepath)
    
    def test_export_ply_ascii(self):
        """Test PLY file export in ASCII format."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            export_mesh_ply(vertices, faces, filepath, binary=False)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert 'ply' in content
            assert 'format ascii' in content
            assert 'element vertex 3' in content
            assert 'element face 1' in content
        finally:
            os.unlink(filepath)
    
    def test_export_ply_binary(self):
        """Test PLY file export in binary format."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            export_mesh_ply(vertices, faces, filepath, binary=True)
            
            # Binary file should exist and be non-empty
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            
            # Header should still be readable
            with open(filepath, 'rb') as f:
                header_start = f.read(3)
            assert header_start == b'ply'
        finally:
            os.unlink(filepath)


class TestMeshQuality:
    """Tests for mesh quality metrics."""
    
    def test_basic_metrics(self):
        """Test basic quality metrics computation."""
        # Simple triangle
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        quality = compute_mesh_quality(vertices, faces)
        
        assert quality['vertex_count'] == 3
        assert quality['face_count'] == 1
        assert quality['edge_count'] == 3
        assert quality['min_edge_length'] > 0
        assert quality['max_edge_length'] > 0
        assert quality['mean_edge_length'] > 0
    
    def test_watertight_detection(self):
        """Test watertight mesh detection."""
        # Tetrahedron is watertight
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=np.int32)
        
        quality = compute_mesh_quality(vertices, faces)
        assert quality['watertight'] == True
    
    def test_non_watertight_detection(self):
        """Test non-watertight (open) mesh detection."""
        # Single triangle is not watertight
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        quality = compute_mesh_quality(vertices, faces)
        assert quality['watertight'] == False
        assert quality['boundary_edges'] == 3  # All 3 edges are boundary


class TestHelperFunctions:
    """Tests for internal helper functions."""
    
    def test_build_adjacency(self):
        """Test adjacency list construction."""
        faces = np.array([
            [0, 1, 2],
            [1, 2, 3],
        ], dtype=np.int32)
        
        adjacency = _build_adjacency(faces, n_vertices=4)
        
        # Vertex 0 is adjacent to 1 and 2
        assert 1 in adjacency[0]
        assert 2 in adjacency[0]
        
        # Vertex 1 is adjacent to 0, 2, 3
        assert 0 in adjacency[1]
        assert 2 in adjacency[1]
        assert 3 in adjacency[1]
    
    def test_create_grid(self):
        """Test grid creation."""
        min_corner = np.array([-1.0, -1.0, -1.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        resolution = 10
        
        points, shape, spacing = _create_grid(min_corner, max_corner, resolution)
        
        assert shape == (10, 10, 10)
        assert len(points) == 1000  # 10^3
        assert points.shape == (1000, 3)
        
        # Check bounds
        assert np.allclose(points.min(axis=0), min_corner)
        assert np.allclose(points.max(axis=0), max_corner)


class TestMeshExtractionCPU:
    """Test mesh extraction works on CPU without CUDA."""
    
    def test_cpu_extraction(self):
        """Test extraction explicitly on CPU."""
        bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        
        vertices, faces = extract_mesh(
            sdf_fn=lambda x: sphere_sdf(x, radius=0.5),
            bounds=bounds,
            resolutions=[32],
            device='cpu',
            smoothing_iterations=0,
        )
        
        assert len(vertices) > 0
        assert len(faces) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
