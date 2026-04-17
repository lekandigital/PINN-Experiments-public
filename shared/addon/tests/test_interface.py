"""
Unit tests for the backend interface and data classes.

These tests do not require PyTorch or Blender, testing only
the interface definitions and data validation.
"""

import unittest
from dataclasses import asdict
import numpy as np


class TestInterfaceImports(unittest.TestCase):
    """Test that all interface components can be imported."""
    
    def test_import_enums(self):
        """Test enum imports."""
        from addon.backend.interface import (
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        # Verify enum values exist
        self.assertIsNotNone(ModelCategory.CLOTH)
        self.assertIsNotNone(OutputFormat.VERTEX_DISPLACEMENTS)
        self.assertIsNotNone(InputRequirement.MESH_VERTICES)
    
    def test_import_dataclasses(self):
        """Test dataclass imports."""
        from addon.backend.interface import (
            BackendCapabilities,
            SimulationState,
            PredictionRequest,
            PredictionResult,
        )
        
        # Verify classes can be instantiated
        self.assertTrue(callable(BackendCapabilities))
        self.assertTrue(callable(SimulationState))
        self.assertTrue(callable(PredictionRequest))
        self.assertTrue(callable(PredictionResult))
    
    def test_import_backend_abc(self):
        """Test ModelBackend ABC import."""
        from addon.backend.interface import ModelBackend
        from abc import ABC
        
        self.assertTrue(issubclass(ModelBackend, ABC))


class TestModelCategory(unittest.TestCase):
    """Test ModelCategory enum."""
    
    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        from addon.backend.interface import ModelCategory
        
        expected = ['CLOTH', 'BODY', 'MOTION', 'SDF_CLOTH']
        for name in expected:
            self.assertTrue(hasattr(ModelCategory, name),
                          f"Missing category: {name}")
    
    def test_category_values_are_strings(self):
        """Category values should be string identifiers."""
        from addon.backend.interface import ModelCategory
        
        for member in ModelCategory:
            self.assertIsInstance(member.value, str)


class TestOutputFormat(unittest.TestCase):
    """Test OutputFormat enum."""
    
    def test_all_formats_exist(self):
        """Verify all expected output formats."""
        from addon.backend.interface import OutputFormat
        
        expected = [
            'VERTEX_DISPLACEMENTS',
            'ABSOLUTE_POSITIONS',
            'SDF_GRID',
            'JOINT_ANGLES',
            'JOINT_POSITIONS',
        ]
        for name in expected:
            self.assertTrue(hasattr(OutputFormat, name),
                          f"Missing format: {name}")


class TestInputRequirement(unittest.TestCase):
    """Test InputRequirement enum."""
    
    def test_all_requirements_exist(self):
        """Verify all expected input requirements."""
        from addon.backend.interface import InputRequirement
        
        expected = [
            'MESH_VERTICES',
            'MESH_FACES',
            'MESH_EDGES',
            'TIME_COORDINATE',
            'BODY_POSE',
            'PREVIOUS_STATE',
            'MATERIAL_PARAMS',
            'EXTERNAL_FORCES',
        ]
        for name in expected:
            self.assertTrue(hasattr(InputRequirement, name),
                          f"Missing requirement: {name}")


class TestBackendCapabilities(unittest.TestCase):
    """Test BackendCapabilities dataclass."""
    
    def test_minimal_capabilities(self):
        """Test creating capabilities with minimal args."""
        from addon.backend.interface import (
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        caps = BackendCapabilities(
            name="Test Backend",
            version="1.0.0",
            category=ModelCategory.CLOTH,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            required_inputs={InputRequirement.MESH_VERTICES},
        )
        
        self.assertEqual(caps.name, "Test Backend")
        self.assertEqual(caps.version, "1.0.0")
        self.assertEqual(caps.category, ModelCategory.CLOTH)
        self.assertFalse(caps.supports_wind)  # Default
        self.assertFalse(caps.supports_material_params)  # Default
    
    def test_full_capabilities(self):
        """Test creating capabilities with all options."""
        from addon.backend.interface import (
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        caps = BackendCapabilities(
            name="Full Backend",
            version="2.0.0",
            category=ModelCategory.SDF_CLOTH,
            output_format=OutputFormat.SDF_GRID,
            required_inputs={
                InputRequirement.MESH_VERTICES,
                InputRequirement.TIME_COORDINATE,
            },
            optional_inputs={
                InputRequirement.EXTERNAL_FORCES,
                InputRequirement.MATERIAL_PARAMS,
            },
            supports_wind=True,
            supports_material_params=True,
            is_sequential=True,
            expected_vertex_count=1024,
            sdf_resolution=(64, 64, 64),
            min_fps=30,
            recommended_fps=60,
        )
        
        self.assertTrue(caps.supports_wind)
        self.assertTrue(caps.supports_material_params)
        self.assertTrue(caps.is_sequential)
        self.assertEqual(caps.expected_vertex_count, 1024)
        self.assertEqual(caps.sdf_resolution, (64, 64, 64))
    
    def test_capabilities_to_dict(self):
        """Test converting capabilities to dictionary."""
        from addon.backend.interface import (
            BackendCapabilities,
            ModelCategory,
            OutputFormat,
            InputRequirement,
        )
        
        caps = BackendCapabilities(
            name="Dict Test",
            version="1.0.0",
            category=ModelCategory.BODY,
            output_format=OutputFormat.ABSOLUTE_POSITIONS,
            required_inputs={InputRequirement.BODY_POSE},
        )
        
        data = asdict(caps)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "Dict Test")


class TestSimulationState(unittest.TestCase):
    """Test SimulationState dataclass."""
    
    def test_create_empty_state(self):
        """Test creating state with no history."""
        from addon.backend.interface import SimulationState
        
        state = SimulationState(
            frame=0,
            time=0.0,
        )
        
        self.assertEqual(state.frame, 0)
        self.assertEqual(state.time, 0.0)
        self.assertIsNone(state.hidden_state)
        self.assertIsNone(state.previous_positions)
        self.assertIsNone(state.previous_velocities)
    
    def test_create_state_with_history(self):
        """Test creating state with simulation history."""
        from addon.backend.interface import SimulationState
        
        positions = np.random.randn(100, 3).astype(np.float32)
        velocities = np.random.randn(100, 3).astype(np.float32)
        hidden = np.random.randn(1, 64).astype(np.float32)
        
        state = SimulationState(
            frame=10,
            time=0.333,
            hidden_state=hidden,
            previous_positions=positions,
            previous_velocities=velocities,
        )
        
        self.assertEqual(state.frame, 10)
        self.assertAlmostEqual(state.time, 0.333, places=3)
        self.assertEqual(state.hidden_state.shape, (1, 64))
        self.assertEqual(state.previous_positions.shape, (100, 3))


class TestPredictionRequest(unittest.TestCase):
    """Test PredictionRequest dataclass."""
    
    def test_create_basic_request(self):
        """Test creating a basic prediction request."""
        from addon.backend.interface import PredictionRequest, SimulationState
        
        vertices = np.random.randn(100, 3).astype(np.float32)
        state = SimulationState(frame=5, time=0.167)
        
        request = PredictionRequest(
            vertices=vertices,
            state=state,
        )
        
        self.assertEqual(request.vertices.shape, (100, 3))
        self.assertEqual(request.state.frame, 5)
        self.assertIsNone(request.faces)
        self.assertIsNone(request.wind_velocity)
    
    def test_create_full_request(self):
        """Test creating a request with all inputs."""
        from addon.backend.interface import PredictionRequest, SimulationState
        
        vertices = np.random.randn(100, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
        wind = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
        body_pose = np.zeros(72, dtype=np.float32)
        material = {
            'density': 0.2,
            'stiffness': 0.5,
            'bending_stiffness': 0.1,
        }
        
        request = PredictionRequest(
            vertices=vertices,
            state=SimulationState(frame=0, time=0.0),
            faces=faces,
            edges=edges,
            wind_velocity=wind,
            gravity=gravity,
            body_pose=body_pose,
            material_params=material,
        )
        
        self.assertEqual(request.faces.shape, (2, 3))
        self.assertEqual(request.edges.shape, (3, 2))
        np.testing.assert_array_equal(request.wind_velocity, wind)
        self.assertEqual(request.material_params['density'], 0.2)


class TestPredictionResult(unittest.TestCase):
    """Test PredictionResult dataclass."""
    
    def test_create_displacement_result(self):
        """Test creating a vertex displacement result."""
        from addon.backend.interface import (
            PredictionResult,
            SimulationState,
            OutputFormat,
        )
        
        displacements = np.random.randn(100, 3).astype(np.float32)
        
        result = PredictionResult(
            output=displacements,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            new_state=SimulationState(frame=1, time=0.033),
            confidence=0.95,
        )
        
        self.assertEqual(result.output.shape, (100, 3))
        self.assertEqual(result.output_format, OutputFormat.VERTEX_DISPLACEMENTS)
        self.assertEqual(result.confidence, 0.95)
    
    def test_create_sdf_result(self):
        """Test creating an SDF grid result."""
        from addon.backend.interface import (
            PredictionResult,
            SimulationState,
            OutputFormat,
        )
        
        sdf_grid = np.random.randn(64, 64, 64).astype(np.float32)
        
        result = PredictionResult(
            output=sdf_grid,
            output_format=OutputFormat.SDF_GRID,
            new_state=SimulationState(frame=1, time=0.033),
        )
        
        self.assertEqual(result.output.shape, (64, 64, 64))
        self.assertEqual(result.output_format, OutputFormat.SDF_GRID)
    
    def test_create_joint_result(self):
        """Test creating a joint position result."""
        from addon.backend.interface import (
            PredictionResult,
            SimulationState,
            OutputFormat,
        )
        
        # 24 joints with 3D positions
        joints = np.random.randn(24, 3).astype(np.float32)
        
        result = PredictionResult(
            output=joints,
            output_format=OutputFormat.JOINT_POSITIONS,
            new_state=SimulationState(frame=1, time=0.033),
            auxiliary_data={'bone_names': ['joint_0', 'joint_1']},
        )
        
        self.assertEqual(result.output.shape, (24, 3))
        self.assertIn('bone_names', result.auxiliary_data)


class TestModelBackendInterface(unittest.TestCase):
    """Test the ModelBackend abstract base class."""
    
    def test_cannot_instantiate_abc(self):
        """Verify ModelBackend cannot be directly instantiated."""
        from addon.backend.interface import ModelBackend
        
        with self.assertRaises(TypeError):
            ModelBackend()
    
    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""
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
        
        class DummyBackend(ModelBackend):
            def get_capabilities(self) -> BackendCapabilities:
                return BackendCapabilities(
                    name="Dummy",
                    version="0.0.1",
                    category=ModelCategory.CLOTH,
                    output_format=OutputFormat.VERTEX_DISPLACEMENTS,
                    required_inputs={InputRequirement.MESH_VERTICES},
                )
            
            def load(self, checkpoint_path: str) -> bool:
                return True
            
            def unload(self) -> None:
                pass
            
            def predict(self, request: PredictionRequest) -> PredictionResult:
                # Return input as output (no displacement)
                return PredictionResult(
                    output=np.zeros_like(request.vertices),
                    output_format=OutputFormat.VERTEX_DISPLACEMENTS,
                    new_state=SimulationState(
                        frame=request.state.frame + 1,
                        time=request.state.time + 0.033,
                    ),
                )
            
            def reset(self) -> None:
                pass
        
        backend = DummyBackend()
        caps = backend.get_capabilities()
        
        self.assertEqual(caps.name, "Dummy")
        self.assertTrue(backend.load("/fake/path"))


if __name__ == '__main__':
    unittest.main()
