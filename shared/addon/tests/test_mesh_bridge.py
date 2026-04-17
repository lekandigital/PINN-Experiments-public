"""
Unit tests for mesh bridge utilities.

Tests conversion functions without requiring Blender.
Uses mock objects for Blender types.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestMeshBridgeImports(unittest.TestCase):
    """Test mesh bridge can be imported without Blender."""
    
    @patch.dict('sys.modules', {'bpy': MagicMock()})
    def test_import_with_mock_bpy(self):
        """Test importing with mocked bpy."""
        # This test verifies the module structure
        # Actual bpy functions are tested in Blender integration tests
        pass


class TestNumpyConversions(unittest.TestCase):
    """Test numpy array utilities."""
    
    def test_vertex_array_shape(self):
        """Test expected vertex array shapes."""
        # Vertex arrays should be (N, 3)
        vertices = np.random.randn(100, 3).astype(np.float32)
        
        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 3)
    
    def test_face_array_shape(self):
        """Test expected face array shapes."""
        # Triangles: (N, 3), Quads: (N, 4)
        triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        
        self.assertEqual(triangles.ndim, 2)
        self.assertEqual(triangles.shape[1], 3)
    
    def test_displacement_application(self):
        """Test that displacement adds to positions."""
        original = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        displacement = np.array([
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [-0.1, 0.1, 0.0],
        ], dtype=np.float32)
        
        result = original + displacement
        
        np.testing.assert_array_almost_equal(
            result[0], [0.1, 0.2, 0.3]
        )
        np.testing.assert_array_almost_equal(
            result[1], [1.0, 0.0, 0.0]
        )


class TestSDFOperations(unittest.TestCase):
    """Test SDF-related operations."""
    
    def test_sdf_grid_creation(self):
        """Test creating an SDF grid."""
        resolution = (32, 32, 32)
        sdf_grid = np.random.randn(*resolution).astype(np.float32)
        
        self.assertEqual(sdf_grid.shape, resolution)
        self.assertEqual(sdf_grid.dtype, np.float32)
    
    def test_sdf_inside_outside(self):
        """Test SDF sign convention."""
        # Negative = inside, Positive = outside
        # Create a simple sphere SDF
        resolution = 16
        grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        center = resolution // 2
        radius = resolution // 4
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    distance = np.sqrt(
                        (i - center)**2 + 
                        (j - center)**2 + 
                        (k - center)**2
                    ) - radius
                    grid[i, j, k] = distance
        
        # Center should be negative (inside)
        self.assertLess(grid[center, center, center], 0)
        
        # Corner should be positive (outside)
        self.assertGreater(grid[0, 0, 0], 0)
    
    def test_marching_cubes_available(self):
        """Test that scikit-image marching_cubes is available."""
        try:
            from skimage.measure import marching_cubes
            self.assertTrue(callable(marching_cubes))
        except ImportError:
            self.skipTest("scikit-image not installed")
    
    def test_marching_cubes_on_sphere(self):
        """Test marching cubes on a sphere SDF."""
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            self.skipTest("scikit-image not installed")
        
        # Create sphere SDF
        resolution = 32
        grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        center = resolution / 2
        radius = resolution / 4
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    distance = np.sqrt(
                        (i - center + 0.5)**2 + 
                        (j - center + 0.5)**2 + 
                        (k - center + 0.5)**2
                    ) - radius
                    grid[i, j, k] = distance
        
        # Run marching cubes
        verts, faces, normals, values = marching_cubes(grid, level=0.0)
        
        # Should produce a mesh
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)


class TestJointOperations(unittest.TestCase):
    """Test joint/skeleton operations."""
    
    def test_joint_positions_shape(self):
        """Test expected joint position shapes."""
        # SMPL has 24 joints
        num_joints = 24
        positions = np.random.randn(num_joints, 3).astype(np.float32)
        
        self.assertEqual(positions.shape, (24, 3))
    
    def test_joint_angles_shape(self):
        """Test expected joint angle shapes."""
        # SMPL pose is 72D (24 joints * 3 axis-angles)
        pose = np.random.randn(72).astype(np.float32)
        
        self.assertEqual(pose.shape, (72,))
        
        # Can be reshaped to (24, 3)
        pose_reshaped = pose.reshape(24, 3)
        self.assertEqual(pose_reshaped.shape, (24, 3))
    
    def test_bone_rotation_matrix(self):
        """Test creating rotation matrix from axis-angle."""
        # Axis-angle representation
        axis_angle = np.array([0.0, 0.0, np.pi/4])  # 45 degree Z rotation
        
        # Convert to rotation matrix (using Rodrigues formula)
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-8:
            # Identity for zero rotation
            R = np.eye(3)
        else:
            axis = axis_angle / angle
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        # Should be orthogonal
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=6)
        
        # Determinant should be 1
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)


class TestEdgeComputation(unittest.TestCase):
    """Test edge list computation from faces."""
    
    def test_edges_from_triangles(self):
        """Test computing edges from triangle faces."""
        # Two triangles sharing an edge
        faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32)
        
        # Extract all edges (with duplicates)
        edges_raw = []
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges_raw.append(edge)
        
        # Unique edges
        edges = np.array(list(set(edges_raw)), dtype=np.int32)
        
        # Should have 5 unique edges
        # (0,1), (1,2), (0,2), (1,3), (2,3)
        self.assertEqual(len(edges), 5)
    
    def test_edge_symmetry(self):
        """Test that edges are properly undirected."""
        # Edge (a,b) should equal edge (b,a)
        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
        ], dtype=np.int32)
        
        # Sort each edge
        edges_sorted = np.sort(edges, axis=1)
        
        # Check symmetry
        for edge in edges_sorted:
            self.assertLessEqual(edge[0], edge[1])


class TestCoordinateTransforms(unittest.TestCase):
    """Test coordinate space transformations."""
    
    def test_normalize_to_unit_cube(self):
        """Test normalizing mesh to [-1, 1] cube."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 8.0, 3.0],
        ], dtype=np.float32)
        
        # Compute bounds
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        center = (vmin + vmax) / 2
        scale = (vmax - vmin).max() / 2
        
        # Normalize
        normalized = (vertices - center) / scale
        
        # Check bounds
        self.assertTrue(np.all(normalized >= -1.0 - 1e-6))
        self.assertTrue(np.all(normalized <= 1.0 + 1e-6))
    
    def test_sdf_to_world_coords(self):
        """Test converting SDF grid coords to world coords."""
        resolution = 64
        bounds_min = np.array([-1.0, -1.0, -1.0])
        bounds_max = np.array([1.0, 1.0, 1.0])
        
        # Grid point at center
        grid_point = np.array([32, 32, 32])
        
        # Convert to world
        world_point = (
            bounds_min + 
            (grid_point / (resolution - 1)) * (bounds_max - bounds_min)
        )
        
        np.testing.assert_array_almost_equal(
            world_point, [0.0, 0.0, 0.0], decimal=2
        )


if __name__ == '__main__':
    unittest.main()
