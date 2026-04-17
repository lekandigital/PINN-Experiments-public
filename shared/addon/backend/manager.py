"""
Backend Manager - Discovers, loads, and switches between available backends.

The addon UI interacts with this manager, not with backends directly.
This provides a clean separation between the UI layer and the model implementations.
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Type
import time
import traceback

from .interface import (
    ModelBackend,
    BackendCapabilities,
    PredictionRequest,
    PredictionResult,
    SimulationState,
)


# Singleton instance
_manager_instance: Optional["BackendManager"] = None


def get_manager() -> "BackendManager":
    """Get the singleton BackendManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = BackendManager()
    return _manager_instance


def reset_manager() -> None:
    """Reset the singleton manager (useful for testing)."""
    global _manager_instance
    if _manager_instance is not None:
        _manager_instance.cleanup()
    _manager_instance = None


class BackendManager:
    """
    Singleton that manages all available backends.
    
    Responsibilities:
    - Discover available backends by scanning the adapters directory
    - Check dependencies for each backend
    - Load/unload backends on demand
    - Forward prediction requests to the active backend
    - Handle errors gracefully without crashing Blender
    """
    
    def __init__(self):
        self._backends: dict[str, Type[ModelBackend]] = {}
        self._backend_info: dict[str, BackendCapabilities] = {}
        self._active_backend: Optional[ModelBackend] = None
        self._active_name: Optional[str] = None
        self._adapters_path: Optional[Path] = None
        self._last_error: Optional[str] = None
        
        # Performance tracking
        self._total_predictions: int = 0
        self._total_inference_time_ms: float = 0.0
    
    def set_adapters_path(self, path: str | Path) -> None:
        """
        Set the path to the adapters directory.
        
        Args:
            path: Path to directory containing adapter_*.py files
        """
        self._adapters_path = Path(path)
    
    def discover_backends(self, force_refresh: bool = False) -> list[str]:
        """
        Scan the adapters directory and import all available backends.
        
        A backend is available if:
        1. Its adapter file exists in shared/addon/backend/adapters/
        2. Its required dependencies are installed (torch, torch_geometric, etc.)
        3. The adapter class can be instantiated
        
        Backends whose dependencies are missing are silently skipped
        (don't crash the addon because one project's deps aren't installed).
        
        Args:
            force_refresh: If True, rescan even if backends already discovered
            
        Returns:
            List of available backend names
        """
        if self._backends and not force_refresh:
            return list(self._backends.keys())
        
        self._backends.clear()
        self._backend_info.clear()
        
        # Determine adapters path
        if self._adapters_path is None:
            self._adapters_path = Path(__file__).parent / "adapters"
        
        if not self._adapters_path.exists():
            print(f"[NeuralSim] Adapters directory not found: {self._adapters_path}")
            return []
        
        # Add adapters path to sys.path if needed
        adapters_parent = str(self._adapters_path.parent)
        if adapters_parent not in sys.path:
            sys.path.insert(0, adapters_parent)
        
        # Scan for adapter files
        adapter_files = list(self._adapters_path.glob("adapter_*.py"))
        
        for adapter_file in adapter_files:
            module_name = adapter_file.stem
            backend_name = self._extract_backend_name(module_name)
            
            try:
                # Import the adapter module
                spec = importlib.util.spec_from_file_location(
                    f"adapters.{module_name}",
                    adapter_file
                )
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"adapters.{module_name}"] = module
                spec.loader.exec_module(module)
                
                # Look for the backend class (should be named *Backend)
                backend_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, ModelBackend) and 
                        attr is not ModelBackend):
                        backend_class = attr
                        break
                
                if backend_class is None:
                    print(f"[NeuralSim] No ModelBackend subclass found in {adapter_file}")
                    continue
                
                # Try to get capabilities (instantiate temporarily)
                try:
                    temp_instance = backend_class()
                    capabilities = temp_instance.get_capabilities()
                    
                    # Use display name as key if available
                    if capabilities.display_name:
                        backend_name = capabilities.display_name
                    
                    self._backends[backend_name] = backend_class
                    self._backend_info[backend_name] = capabilities
                    
                    print(f"[NeuralSim] Discovered backend: {backend_name}")
                    
                except Exception as e:
                    print(f"[NeuralSim] Backend {module_name} failed capability check: {e}")
                    continue
                    
            except ImportError as e:
                # Missing dependency — skip silently but log
                print(f"[NeuralSim] Backend {module_name} skipped (missing dependency): {e}")
                continue
            except Exception as e:
                print(f"[NeuralSim] Error loading {module_name}: {e}")
                traceback.print_exc()
                continue
        
        return list(self._backends.keys())
    
    def _extract_backend_name(self, module_name: str) -> str:
        """Extract a readable name from adapter module name."""
        # adapter_11_legacy -> 11_legacy -> 11 Legacy
        name = module_name.replace("adapter_", "")
        parts = name.split("_")
        return " ".join(p.capitalize() for p in parts)
    
    def get_available_backends(self) -> list[str]:
        """Return list of discovered backend names."""
        if not self._backends:
            self.discover_backends()
        return list(self._backends.keys())
    
    def get_backend_info(self, name: str) -> Optional[BackendCapabilities]:
        """
        Return capabilities for a backend without loading it.
        
        Args:
            name: Backend name
            
        Returns:
            BackendCapabilities or None if backend not found
        """
        if not self._backends:
            self.discover_backends()
        return self._backend_info.get(name)
    
    def get_all_backend_info(self) -> dict[str, BackendCapabilities]:
        """Return capabilities for all discovered backends."""
        if not self._backends:
            self.discover_backends()
        return dict(self._backend_info)
    
    def activate_backend(
        self, 
        name: str, 
        checkpoint_path: str, 
        device: str = "auto"
    ) -> BackendCapabilities:
        """
        Switch to a new backend. Cleans up the old one first.
        
        Args:
            name: Name of the backend to activate
            checkpoint_path: Path to model checkpoint file
            device: "cpu", "cuda", "cuda:N", or "auto" (auto-detect)
            
        Returns:
            The new backend's capabilities for UI configuration
            
        Raises:
            KeyError: If backend name is not found
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If backend fails to load
        """
        if not self._backends:
            self.discover_backends()
        
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not found. Available: {list(self._backends.keys())}")
        
        # Clean up existing backend
        if self._active_backend is not None:
            try:
                self._active_backend.cleanup()
            except Exception as e:
                print(f"[NeuralSim] Warning: cleanup failed for {self._active_name}: {e}")
            self._active_backend = None
            self._active_name = None
        
        # Instantiate new backend
        backend_class = self._backends[name]
        self._active_backend = backend_class()
        self._active_name = name
        
        # Resolve device
        if device == "auto":
            device = self._active_backend.get_device_recommendation()
        
        # Load checkpoint
        try:
            self._active_backend.load(checkpoint_path, device)
        except Exception as e:
            self._last_error = str(e)
            self._active_backend = None
            self._active_name = None
            raise RuntimeError(f"Failed to load backend '{name}': {e}")
        
        # Warmup with example request
        try:
            example = self._create_example_request()
            self._active_backend.warmup(example)
        except Exception as e:
            print(f"[NeuralSim] Warning: warmup failed: {e}")
        
        # Reset performance counters
        self._total_predictions = 0
        self._total_inference_time_ms = 0.0
        self._last_error = None
        
        return self._active_backend.get_capabilities()
    
    def _create_example_request(self) -> PredictionRequest:
        """Create an example request for warmup."""
        import numpy as np
        
        caps = self._active_backend.get_capabilities()
        
        request = PredictionRequest(
            time=0.0,
            frame=0,
            delta_time=1/30,
        )
        
        # Add mesh data if needed
        if any(req in caps.input_requirements for req in [
            InputRequirement.MESH_VERTICES,
            InputRequirement.MESH_TOPOLOGY,
        ]):
            # Simple 4-vertex quad
            request.vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
            ], dtype=np.float32)
            request.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            request.edges = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0], [0, 2]
            ], dtype=np.int32)
        
        # Add query points if needed
        if InputRequirement.QUERY_POINTS in caps.input_requirements:
            # Small 4x4x4 grid
            request.query_resolution = 4
            
        return request
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Forward prediction to active backend. Catches exceptions gracefully.
        
        Args:
            request: PredictionRequest with input data
            
        Returns:
            PredictionResult from backend, or safe fallback on error
        """
        if self._active_backend is None:
            return PredictionResult(
                state=request.state or SimulationState(),
                confidence=0.0,
                error_message="No backend is active",
            )
        
        start_time = time.perf_counter()
        
        try:
            result = self._active_backend.predict(request)
            
            # Track performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.inference_time_ms = elapsed_ms
            self._total_predictions += 1
            self._total_inference_time_ms += elapsed_ms
            
            return result
            
        except Exception as e:
            self._last_error = str(e)
            print(f"[NeuralSim] Backend error: {e}")
            traceback.print_exc()
            
            # Return safe fallback
            return PredictionResult(
                state=request.state or SimulationState(),
                confidence=0.0,
                error_message=str(e),
            )
    
    def reset(self) -> None:
        """Reset active backend's simulation state."""
        if self._active_backend is not None:
            try:
                self._active_backend.reset()
            except Exception as e:
                print(f"[NeuralSim] Reset error: {e}")
    
    def get_active_capabilities(self) -> Optional[BackendCapabilities]:
        """Return current backend's capabilities, or None if no backend active."""
        if self._active_backend is not None:
            return self._active_backend.get_capabilities()
        return None
    
    def get_active_name(self) -> Optional[str]:
        """Return the name of the currently active backend."""
        return self._active_name
    
    def is_active(self) -> bool:
        """Check if a backend is currently active and loaded."""
        return (self._active_backend is not None and 
                self._active_backend.is_loaded())
    
    def get_performance_stats(self) -> dict:
        """Return performance statistics."""
        avg_ms = 0.0
        if self._total_predictions > 0:
            avg_ms = self._total_inference_time_ms / self._total_predictions
        
        return {
            "total_predictions": self._total_predictions,
            "total_time_ms": self._total_inference_time_ms,
            "average_time_ms": avg_ms,
            "average_fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
        }
    
    def get_last_error(self) -> Optional[str]:
        """Return the last error message, if any."""
        return self._last_error
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        if self._active_backend is not None:
            try:
                self._active_backend.cleanup()
            except Exception as e:
                print(f"[NeuralSim] Cleanup error: {e}")
            self._active_backend = None
            self._active_name = None
        
        self._backends.clear()
        self._backend_info.clear()


# Import InputRequirement for the example request creation
from .interface import InputRequirement
