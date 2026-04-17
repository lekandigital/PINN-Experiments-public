"""
Unit tests for the BackendManager.

Tests backend discovery, registration, and lifecycle management.
Does not require actual model files or Blender.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys


class TestManagerImports(unittest.TestCase):
    """Test manager module imports."""
    
    def test_import_manager(self):
        """Test importing the manager module."""
        from addon.backend.manager import BackendManager
        self.assertTrue(callable(BackendManager))
    
    def test_import_helpers(self):
        """Test importing helper functions."""
        from addon.backend import get_manager, reset_manager
        self.assertTrue(callable(get_manager))
        self.assertTrue(callable(reset_manager))


class TestBackendManagerSingleton(unittest.TestCase):
    """Test singleton behavior."""
    
    def setUp(self):
        """Reset singleton before each test."""
        from addon.backend import reset_manager
        reset_manager()
    
    def test_singleton_instance(self):
        """Verify get_manager returns same instance."""
        from addon.backend import get_manager
        
        manager1 = get_manager()
        manager2 = get_manager()
        
        self.assertIs(manager1, manager2)
    
    def test_reset_creates_new_instance(self):
        """Verify reset_manager creates a new instance."""
        from addon.backend import get_manager, reset_manager
        
        manager1 = get_manager()
        reset_manager()
        manager2 = get_manager()
        
        self.assertIsNot(manager1, manager2)


class TestBackendDiscovery(unittest.TestCase):
    """Test backend discovery functionality."""
    
    def setUp(self):
        from addon.backend import reset_manager
        reset_manager()
    
    def test_discover_backends_returns_list(self):
        """Verify discover_backends returns a list."""
        from addon.backend import get_manager
        
        manager = get_manager()
        # Set a valid adapters path
        adapters_path = Path(__file__).parent.parent / "backend" / "adapters"
        manager.set_adapters_path(adapters_path)
        
        backends = manager.discover_backends()
        self.assertIsInstance(backends, list)
    
    def test_list_available_backends(self):
        """Test listing available backend IDs."""
        from addon.backend import get_manager
        
        manager = get_manager()
        adapters_path = Path(__file__).parent.parent / "backend" / "adapters"
        manager.set_adapters_path(adapters_path)
        manager.discover_backends()
        
        available = manager.list_available()
        self.assertIsInstance(available, list)


class TestBackendActivation(unittest.TestCase):
    """Test backend activation and deactivation."""
    
    def setUp(self):
        from addon.backend import reset_manager
        reset_manager()
    
    def test_no_active_backend_initially(self):
        """Verify no backend is active initially."""
        from addon.backend import get_manager
        
        manager = get_manager()
        self.assertIsNone(manager.get_active_backend())
    
    def test_get_active_capabilities_none(self):
        """Test getting capabilities when no backend active."""
        from addon.backend import get_manager
        
        manager = get_manager()
        caps = manager.get_active_capabilities()
        self.assertIsNone(caps)


class TestMockBackendRegistration(unittest.TestCase):
    """Test manual backend registration with mocks."""
    
    def setUp(self):
        from addon.backend import reset_manager
        reset_manager()
    
    def test_register_mock_backend(self):
        """Test registering a mock backend."""
        from addon.backend import get_manager
        from addon.backend.interface import (
            ModelBackend,
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        # Create a mock backend
        mock_backend = Mock(spec=ModelBackend)
        mock_backend.get_capabilities.return_value = BackendCapabilities(
            name="Mock Backend",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        
        manager = get_manager()
        manager.register_backend("mock", mock_backend)
        
        available = manager.list_available()
        self.assertIn("mock", available)
    
    def test_activate_mock_backend(self):
        """Test activating a registered mock backend."""
        from addon.backend import get_manager
        from addon.backend.interface import (
            ModelBackend,
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        mock_backend = Mock(spec=ModelBackend)
        mock_backend.get_capabilities.return_value = BackendCapabilities(
            name="Mock Backend",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        mock_backend.load.return_value = True
        
        manager = get_manager()
        manager.register_backend("mock", mock_backend)
        
        success = manager.activate_backend("mock", "/fake/checkpoint.pt")
        
        self.assertTrue(success)
        mock_backend.load.assert_called_once_with("/fake/checkpoint.pt")
    
    def test_deactivate_backend(self):
        """Test deactivating a backend."""
        from addon.backend import get_manager
        from addon.backend.interface import (
            ModelBackend,
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        mock_backend = Mock(spec=ModelBackend)
        mock_backend.get_capabilities.return_value = BackendCapabilities(
            name="Mock Backend",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        mock_backend.load.return_value = True
        
        manager = get_manager()
        manager.register_backend("mock", mock_backend)
        manager.activate_backend("mock", "/fake/checkpoint.pt")
        
        manager.deactivate()
        
        mock_backend.unload.assert_called_once()
        self.assertIsNone(manager.get_active_backend())


class TestPredictionRouting(unittest.TestCase):
    """Test prediction routing through the manager."""
    
    def setUp(self):
        from addon.backend import reset_manager
        reset_manager()
    
    def test_predict_with_no_active_backend(self):
        """Test that predict returns None when no backend active."""
        from addon.backend import get_manager
        from addon.backend.interface import PredictionRequest, SimulationState
        import numpy as np
        
        manager = get_manager()
        
        request = PredictionRequest(
            vertices=np.zeros((10, 3), dtype=np.float32),
            state=SimulationState(frame=0, time=0.0),
        )
        
        result = manager.predict(request)
        self.assertIsNone(result)
    
    def test_predict_routes_to_backend(self):
        """Test that predict routes to active backend."""
        from addon.backend import get_manager
        from addon.backend.interface import (
            ModelBackend,
            BackendCapabilities,
            PredictionRequest,
            PredictionResult,
            SimulationState,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        import numpy as np
        
        # Create mock backend with predict
        mock_backend = Mock(spec=ModelBackend)
        mock_backend.get_capabilities.return_value = BackendCapabilities(
            name="Predictor",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        mock_backend.load.return_value = True
        
        expected_result = PredictionResult(
            output=np.ones((10, 3), dtype=np.float32),
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            new_state=SimulationState(frame=1, time=0.033),
        )
        mock_backend.predict.return_value = expected_result
        
        # Setup manager
        manager = get_manager()
        manager.register_backend("predictor", mock_backend)
        manager.activate_backend("predictor", "/fake/model.pt")
        
        # Make prediction
        request = PredictionRequest(
            vertices=np.zeros((10, 3), dtype=np.float32),
            state=SimulationState(frame=0, time=0.0),
        )
        
        result = manager.predict(request)
        
        self.assertIsNotNone(result)
        mock_backend.predict.assert_called_once_with(request)
        np.testing.assert_array_equal(result.output, expected_result.output)


class TestBackendLifecycle(unittest.TestCase):
    """Test backend lifecycle management."""
    
    def setUp(self):
        from addon.backend import reset_manager
        reset_manager()
    
    def test_switch_backends(self):
        """Test switching between backends."""
        from addon.backend import get_manager
        from addon.backend.interface import (
            ModelBackend,
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        # Create two mock backends
        mock1 = Mock(spec=ModelBackend)
        mock1.get_capabilities.return_value = BackendCapabilities(
            name="Backend 1",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        mock1.load.return_value = True
        
        mock2 = Mock(spec=ModelBackend)
        mock2.get_capabilities.return_value = BackendCapabilities(
            name="Backend 2",
            version="1.0.0",
            category=ModelCategory.SDF_CLOTH,
            output_format=OutputFormat.SDF_GRID,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        mock2.load.return_value = True
        
        # Register both
        manager = get_manager()
        manager.register_backend("backend1", mock1)
        manager.register_backend("backend2", mock2)
        
        # Activate first
        manager.activate_backend("backend1", "/path/1.pt")
        caps = manager.get_active_capabilities()
        self.assertEqual(caps.name, "Backend 1")
        
        # Switch to second
        manager.activate_backend("backend2", "/path/2.pt")
        
        # First should be unloaded
        mock1.unload.assert_called_once()
        
        # Second should be active
        caps = manager.get_active_capabilities()
        self.assertEqual(caps.name, "Backend 2")
    
    def test_reset_calls_unload(self):
        """Test that reset properly unloads backend."""
        from addon.backend import get_manager
        from addon.backend.interface import (
            ModelBackend,
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        mock_backend = Mock(spec=ModelBackend)
        mock_backend.get_capabilities.return_value = BackendCapabilities(
            name="Reset Test",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        mock_backend.load.return_value = True
        
        manager = get_manager()
        manager.register_backend("test", mock_backend)
        manager.activate_backend("test", "/test.pt")
        
        # Reset the backend simulation state
        manager.reset_simulation()
        
        mock_backend.reset.assert_called_once()


if __name__ == '__main__':
    unittest.main()
