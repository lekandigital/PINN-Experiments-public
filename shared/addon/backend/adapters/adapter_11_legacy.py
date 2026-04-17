"""
Backend adapter for Project 11's original SIREN model (NIF-Cloth3D-Interactive).

Preserves backward compatibility — the existing model still works
through the new interface without any changes to its weights or architecture.

Model Details:
- Input: 8D tensor (x, y, z, t, fx, fy, fz, material_param)
- Output: 3D displacement (dx, dy, dz)
- Architecture: SIREN with sine activations, 6 layers, 256 hidden dim
- Checkpoint format: TorchScript (.pt) via torch.jit.save
"""

import os
import time
from pathlib import Path
from typing import Optional
import numpy as np

from ..interface import (
    ModelBackend,
    ModelCategory,
    OutputFormat,
    InputRequirement,
    BackendCapabilities,
    SimulationState,
    PredictionRequest,
    PredictionResult,
)


class LegacyNIFClothBackend(ModelBackend):
    """
    Adapter for Project 11's NIF-Cloth3D SIREN model.
    
    This wraps the existing model exactly as it works in the original addon,
    enabling backward compatibility with existing checkpoints.
    """
    
    def __init__(self):
        self._model = None
        self._device = None
        self._checkpoint_path: Optional[str] = None
        self._loaded = False
        
        # Project paths
        self._project_dir = self._find_project_dir()
    
    def _find_project_dir(self) -> Optional[Path]:
        """Find the Project 11 directory relative to this file."""
        # Navigate from shared/addon/backend/adapters/ to projects/
        current = Path(__file__).parent
        for _ in range(4):  # Go up to PINN-Experiments
            current = current.parent
        
        project_dir = current / "projects" / "11-nif-cloth3d__project-space"
        if project_dir.exists():
            return project_dir
        
        # Also try alternative naming
        project_dir = current / "projects" / "11-NIF-Cloth3D-Interactive"
        if project_dir.exists():
            return project_dir
            
        return None
    
    def get_capabilities(self) -> BackendCapabilities:
        """Return capabilities matching the original Project 11 addon."""
        return BackendCapabilities(
            category=ModelCategory.CLOTH_SIMULATION,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            input_requirements={
                InputRequirement.MESH_VERTICES,
                InputRequirement.TIME,
                InputRequirement.FORCES,
                InputRequirement.MATERIAL_PARAMS,
            },
            
            # Force support
            supports_wind=True,
            supports_gravity=True,
            supports_custom_forces=False,
            wind_dimensions=3,
            
            # Material support - single stiffness parameter
            supports_material_params=True,
            material_param_names=["material_type"],
            material_param_ranges={"material_type": (0.0, 4.0)},
            material_param_defaults={"material_type": 0.0},
            
            # Temporal behavior - stateless SIREN
            is_temporal=True,
            supports_continuous_time=True,  # SIREN can query any t
            needs_sequential_frames=False,  # No hidden state
            max_stable_frames=10000,
            
            # No resolution control (direct vertex displacement)
            supports_resolution_control=False,
            
            # Performance characteristics
            typical_fps=100.0,
            typical_memory_mb=50.0,
            parameter_count=400_000,  # ~400K params for 6-layer SIREN
            model_size_mb=1.6,
            
            # Mesh requirements
            needs_fixed_topology=False,
            max_vertices=100_000,
            
            # Metadata
            display_name="NIF-Cloth3D (Legacy)",
            description="Original SIREN-based cloth simulation from Project 11. "
                       "Supports wind, gravity, and material parameters. "
                       "Best for real-time interactive cloth with force control.",
            quality_tier="standard",
            icon="OUTLINER_OB_SURFACE",
            
            # Project reference
            project_id="11-nif-cloth3d",
            project_path="projects/11-nif-cloth3d__project-space",
        )
    
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """
        Load model weights from a TorchScript checkpoint.
        
        Args:
            checkpoint_path: Path to .pt TorchScript file
            device: "cpu" or "cuda" / "cuda:N"
        """
        # Lazy import torch to avoid import errors if not installed
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is required for NIF-Cloth3D backend. "
                "Please install: pip install torch"
            )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Setup device
        if device == "auto" or device == "cuda":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Load TorchScript model
        try:
            self._model = torch.jit.load(checkpoint_path, map_location=self._device)
            self._model.eval()
            self._model.to(self._device)
        except Exception as e:
            raise RuntimeError(f"Failed to load TorchScript model: {e}")
        
        self._checkpoint_path = checkpoint_path
        self._loaded = True
        
        print(f"[NIF-Cloth3D] Loaded model on {self._device}")
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Run inference: vertices + time + forces + material → displacements.
        
        Input tensor format: [x, y, z, t, fx, fy, fz, material_id] (8D)
        Output tensor format: [dx, dy, dz] (3D displacement)
        """
        import torch
        
        if not self._loaded or self._model is None:
            return PredictionResult(
                state=request.state or SimulationState(),
                confidence=0.0,
                error_message="Model not loaded",
            )
        
        start_time = time.perf_counter()
        
        try:
            # Get vertices (required)
            if request.vertices is None:
                return PredictionResult(
                    state=request.state or SimulationState(),
                    confidence=0.0,
                    error_message="No vertices provided",
                )
            
            num_vertices = len(request.vertices)
            
            # Prepare input components
            # Vertices: (N, 3)
            verts = torch.from_numpy(request.vertices).float().to(self._device)
            
            # Time: normalized to [0, 1] - broadcast to (N, 1)
            t_value = request.time
            t_tensor = torch.full((num_vertices, 1), t_value, 
                                  dtype=torch.float32, device=self._device)
            
            # Wind force: (N, 3)
            if request.wind_velocity is not None:
                wind = torch.from_numpy(request.wind_velocity).float().to(self._device)
                wind = wind.unsqueeze(0).expand(num_vertices, 3)
            else:
                wind = torch.zeros((num_vertices, 3), dtype=torch.float32, 
                                   device=self._device)
            
            # Material parameter: (N, 1)
            material_val = request.material_params.get("material_type", 0.0)
            material = torch.full((num_vertices, 1), material_val,
                                  dtype=torch.float32, device=self._device)
            
            # Concatenate to 8D input: [x, y, z, t, fx, fy, fz, material]
            model_input = torch.cat([verts, t_tensor, wind, material], dim=1)
            
            # Run inference
            with torch.no_grad():
                displacement = self._model(model_input)
            
            # Convert output to numpy
            displacement_np = displacement.cpu().numpy().astype(np.float32)
            
            # Compute inference time
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update state
            new_state = SimulationState(
                frame=request.frame,
                time=request.time,
            )
            
            return PredictionResult(
                displacements=displacement_np,
                state=new_state,
                inference_time_ms=inference_time_ms,
                confidence=1.0,
            )
            
        except Exception as e:
            return PredictionResult(
                state=request.state or SimulationState(),
                confidence=0.0,
                error_message=f"Inference error: {str(e)}",
            )
    
    def reset(self) -> None:
        """Reset state. No-op for stateless SIREN model."""
        pass  # SIREN is stateless, nothing to reset
    
    def warmup(self, example_request: PredictionRequest) -> None:
        """Run warmup inference to trigger CUDA kernel caching."""
        if self._loaded:
            _ = self.predict(example_request)
    
    def get_recommended_checkpoint(self) -> Optional[str]:
        """Return path to default checkpoint if available."""
        if self._project_dir is None:
            return None
        
        # Search for checkpoints in standard locations
        search_paths = [
            self._project_dir / "nif-cloth3d" / "checkpoints" / "best_model.pt",
            self._project_dir / "nif-cloth3d" / "checkpoints" / "model.pt",
            self._project_dir / "checkpoints" / "best_model.pt",
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_available_checkpoints(self) -> list[str]:
        """Find all available checkpoint files."""
        checkpoints = []
        
        if self._project_dir is None:
            return checkpoints
        
        # Search for .pt files
        for pattern in ["**/*.pt", "**/checkpoints/*.pt"]:
            for path in self._project_dir.glob(pattern):
                if path.is_file():
                    checkpoints.append(str(path))
        
        return sorted(set(checkpoints))
    
    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._loaded = False
        
        # Try to free CUDA memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self._model is not None
    
    def get_model_info(self) -> dict:
        """Return model architecture information."""
        return {
            "architecture": "SIREN (Sinusoidal Representation Network)",
            "input_dim": 8,
            "output_dim": 3,
            "hidden_dim": 256,
            "num_layers": 6,
            "activation": "sine",
            "omega_0": 30.0,
            "checkpoint_format": "TorchScript",
            "project": "11-NIF-Cloth3D-Interactive",
        }
