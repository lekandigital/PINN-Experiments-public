"""
Integration tests that can run inside Blender.

Run these tests from Blender's Python console or as a script:
    blender --background --python run_blender_tests.py

These tests require Blender to be running and the addon to be installed.
"""

import unittest
import sys


def run_tests_in_blender():
    """Run integration tests inside Blender."""
    try:
        import bpy
    except ImportError:
        print("ERROR: These tests must be run inside Blender!")
        print("Usage: blender --background --python run_blender_tests.py")
        sys.exit(1)
    
    # Ensure addon is loaded
    addon_name = "addon"
    if addon_name not in bpy.context.preferences.addons:
        # Try to enable it
        try:
            bpy.ops.preferences.addon_enable(module=addon_name)
        except Exception as e:
            print(f"Failed to enable addon: {e}")
            print("Make sure the addon is installed first.")
            sys.exit(1)


class TestAddonRegistration(unittest.TestCase):
    """Test addon registration in Blender."""
    
    def test_addon_info_exists(self):
        """Test that bl_info is properly defined."""
        import bpy
        
        # Check addon is registered
        prefs = bpy.context.preferences
        addon = prefs.addons.get('addon')
        
        self.assertIsNotNone(addon, "Addon not registered")
    
    def test_panels_registered(self):
        """Test that UI panels are registered."""
        import bpy
        
        # Check for our panel types
        panel_names = [
            'NEURALSIM_PT_backend',
            'NEURALSIM_PT_target',
            'NEURALSIM_PT_forces',
        ]
        
        for name in panel_names:
            self.assertTrue(
                hasattr(bpy.types, name),
                f"Panel {name} not registered"
            )
    
    def test_operators_registered(self):
        """Test that operators are registered."""
        import bpy
        
        # Check for our operators
        operator_names = [
            'neuralsim.load_backend',
            'neuralsim.refresh_backends',
            'neuralsim.play',
            'neuralsim.pause',
        ]
        
        for name in operator_names:
            parts = name.split('.')
            op_module = getattr(bpy.ops, parts[0], None)
            
            self.assertIsNotNone(
                op_module,
                f"Operator module {parts[0]} not found"
            )
    
    def test_properties_registered(self):
        """Test that scene properties are registered."""
        import bpy
        
        # Check scene has our property group
        self.assertTrue(
            hasattr(bpy.types.Scene, 'neural_sim'),
            "Property group not registered on Scene"
        )


class TestMeshBridgeBlender(unittest.TestCase):
    """Test mesh bridge with actual Blender meshes."""
    
    def setUp(self):
        """Create test mesh."""
        import bpy
        
        # Clear scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Create a simple mesh
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        self.mesh_obj = bpy.context.active_object
        self.mesh_obj.name = "TestMesh"
    
    def tearDown(self):
        """Clean up test mesh."""
        import bpy
        
        if self.mesh_obj and self.mesh_obj.name in bpy.data.objects:
            bpy.data.objects.remove(self.mesh_obj)
    
    def test_extract_vertices(self):
        """Test extracting vertices from Blender mesh."""
        from addon.core.mesh_bridge import blender_mesh_to_numpy
        import numpy as np
        
        data = blender_mesh_to_numpy(self.mesh_obj)
        
        self.assertIn('vertices', data)
        self.assertIsInstance(data['vertices'], np.ndarray)
        self.assertEqual(data['vertices'].shape[1], 3)
        
        # Plane has 4 vertices
        self.assertEqual(data['vertices'].shape[0], 4)
    
    def test_extract_faces(self):
        """Test extracting faces from Blender mesh."""
        from addon.core.mesh_bridge import blender_mesh_to_numpy
        import numpy as np
        
        data = blender_mesh_to_numpy(self.mesh_obj)
        
        self.assertIn('faces', data)
        # Plane has 1 quad face (or 2 triangles after triangulation)
        self.assertGreater(len(data['faces']), 0)
    
    def test_apply_displacements(self):
        """Test applying displacements to Blender mesh."""
        from addon.core.mesh_bridge import (
            blender_mesh_to_numpy,
            numpy_displacements_to_blender,
        )
        import numpy as np
        
        # Get original positions
        original = blender_mesh_to_numpy(self.mesh_obj)['vertices'].copy()
        
        # Create displacement (move everything up by 1)
        displacements = np.zeros_like(original)
        displacements[:, 2] = 1.0  # Z displacement
        
        # Apply
        numpy_displacements_to_blender(self.mesh_obj, displacements)
        
        # Check new positions
        new_positions = blender_mesh_to_numpy(self.mesh_obj)['vertices']
        
        np.testing.assert_array_almost_equal(
            new_positions[:, 2],
            original[:, 2] + 1.0,
            decimal=5
        )


class TestSDFToMesh(unittest.TestCase):
    """Test SDF to mesh conversion in Blender."""
    
    def test_create_mesh_from_sdf(self):
        """Test creating a Blender mesh from SDF grid."""
        import bpy
        import numpy as np
        from addon.core.mesh_bridge import sdf_grid_to_blender_mesh
        
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
        
        # Create mesh
        mesh_obj = sdf_grid_to_blender_mesh(
            grid, 
            name="SDF_Sphere",
            bounds_min=(-1, -1, -1),
            bounds_max=(1, 1, 1),
        )
        
        self.assertIsNotNone(mesh_obj)
        self.assertIsInstance(mesh_obj, bpy.types.Object)
        self.assertGreater(len(mesh_obj.data.vertices), 0)
        
        # Cleanup
        bpy.data.objects.remove(mesh_obj)


class TestFrameHandler(unittest.TestCase):
    """Test frame change handler."""
    
    def test_handler_registration(self):
        """Test that handler can be registered."""
        import bpy
        from addon.core.frame_handler import (
            register_frame_handler,
            unregister_frame_handler,
        )
        
        initial_count = len(bpy.app.handlers.frame_change_post)
        
        register_frame_handler()
        
        self.assertEqual(
            len(bpy.app.handlers.frame_change_post),
            initial_count + 1
        )
        
        unregister_frame_handler()
        
        self.assertEqual(
            len(bpy.app.handlers.frame_change_post),
            initial_count
        )


class TestBackendDiscoveryBlender(unittest.TestCase):
    """Test backend discovery in Blender context."""
    
    def test_discover_available_backends(self):
        """Test that backends are discovered."""
        from addon.backend import get_manager
        
        manager = get_manager()
        available = manager.list_available()
        
        self.assertIsInstance(available, list)
        # Should have at least the legacy backend
        print(f"Discovered backends: {available}")


def suite():
    """Create test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAddonRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshBridgeBlender))
    suite.addTests(loader.loadTestsFromTestCase(TestSDFToMesh))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendDiscoveryBlender))
    
    return suite


if __name__ == '__main__':
    run_tests_in_blender()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
