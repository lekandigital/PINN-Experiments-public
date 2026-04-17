"""
Backend adapter for Project 13: NIF-Cloth4D

Ultra-compact SIREN for fast cloth preview — 66K params, trains in 31s.
Output: SDF field that needs marching cubes to extract mesh.

Model Details:
- Input: 4D tensor (x, y, z, t)
- Output: Scalar SDF value
- Architecture: Compact SIREN with omega_0=30
- Use case: Fast preview, web deployment, quick iteration
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


class NIFCloth4DBackend(ModelBackend):
    """
    Adapter for Project 13's NIF-Cloth4D SIREN model.
    
    This is the simplest SDF-output backend — just (x,y,z,t) → SDF value.
    No force parameterization, no material control, pure function of spacetime.
    """
    
    def __init__(self):
        self._model = None
        self._device = None
        self._checkpoint_path: Optional[str] = None
        self._loaded = False
        self._config = None
        
        # Default bounds for SDF query grid
        self._default_bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        
        # Project paths
        self._project_dir = self._find_project_dir()
    
    def _find_project_dir(self) -> Optional[Path]:
        """Find the Project 13 directory."""
        current = Path(__file__).parent
        for _ in range(4):
            current = current.parent
        
        project_dir = current / "projects" / "13-nif-cloth4d__project-space"
        if project_dir.exists():
            return project_dir
            
        return None
    
    def get_capabilities(self) -> BackendCapabilities:
        """Return capabilities for the compact SDF model."""
        return BackendCapabilities(
            category=ModelCategory.CLOTH_SIMULATION,
            output_format=OutputFormat.SDF_FIELD,
            input_requirements={
                InputRequirement.TIME,
                InputRequirement.QUERY_POINTS,
            },
            
            # No force support - behavior is baked into training data
            supports_wind=False,
            supports_gravity=False,
            supports_custom_forces=False,
            
            # No material support
            supports_material_params=False,
            material_param_names=[],
            
            # Temporal behavior - stateless, continuous time
            is_temporal=True,
            supports_continuous_time=True,
            needs_sequential_frames=False,
            max_stable_frames=10000,
            
            # Resolution control for SDF grid
            supports_resolution_control=True,
            resolution_range=(32, 256),
            default_resolution=64,  # Lower default for speed
            
            # Performance - very fast
            typical_fps=1000.0,
            typical_memory_mb=10.0,
            parameter_count=66_000,
            model_size_mb=0.3,
            
            # No topology constraints
            needs_fixed_topology=False,
            max_vertices=500_000,  # After marching cubes
            
            # Metadata
            display_name="NIF-Cloth4D (Preview)",
            description="Ultra-fast cloth preview. Minimal 66K parameter SIREN "
                       "with 4D input (x,y,z,t). No force/material controls — "
                       "behavior is baked in. Best for quick iteration and previews.",
            quality_tier="preview",
            icon="MESH_ICOSPHERE",
            
            project_id="13-nif-cloth4d",
            project_path="projects/13-nif-cloth4d__project-space",
        )
    
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load the SIREN model from checkpoint."""
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is required for NIF-Cloth4D backend. "
                "Please install: pip install torch"
            )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Setup device
        if device == "auto" or device == "cuda":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                self._config = checkpoint.get("config", {})
            else:
                state_dict = checkpoint
                self._config = {}
            
            # Build model architecture
            self._model = self._build_model()
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            
        except Exception as e:
            # Try TorchScript format
            try:
                self._model = torch.jit.load(checkpoint_path, map_location=self._device)
                self._model.eval()
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e}, {e2}")
        
        self._checkpoint_path = checkpoint_path
        self._loaded = True
        
        print(f"[NIF-Cloth4D] Loaded model on {self._device}")
    
    def _build_model(self):
        """Build the SIREN model architecture."""
        import torch
        import torch.nn as nn
        
        class SineLayer(nn.Module):
            def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
                super().__init__()
                self.omega_0 = omega_0
                self.linear = nn.Linear(in_features, out_features)
                self.is_first = is_first
                self._init_weights()
            
            def _init_weights(self):
                with torch.no_grad():
                    if self.is_first:
                        self.linear.weight.uniform_(-1 / self.linear.in_features,
                                                     1 / self.linear.in_features)
                    else:
                        self.linear.weight.uniform_(
                            -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                            np.sqrt(6 / self.linear.in_features) / self.omega_0
                        )
            
            def forward(self, x):
                return torch.sin(self.omega_0 * self.linear(x))
        
        class ClothSIREN(nn.Module):
            def __init__(self, in_features=4, hidden_features=128, hidden_layers=3, out_features=1):
                super().__init__()
                
                layers = [SineLayer(in_features, hidden_features, omega_0=30.0, is_first=True)]
                for _ in range(hidden_layers):
                    layers.append(SineLayer(hidden_features, hidden_features, omega_0=30.0))
                
                self.net = nn.Sequential(*layers)
                self.final = nn.Linear(hidden_features, out_features)
                
                with torch.no_grad():
                    self.final.weight.uniform_(
                        -np.sqrt(6 / hidden_features) / 30.0,
                        np.sqrt(6 / hidden_features) / 30.0
                    )
            
            def forward(self, coords):
                return self.final(self.net(coords))
        
        # Use config if available, otherwise defaults
        config = self._config or {}
        return ClothSIREN(
            in_features=config.get("in_features", 4),
            hidden_features=config.get("hidden_features", 128),
            hidden_layers=config.get("hidden_layers", 3),
            out_features=config.get("out_features", 1),
        )
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate SDF grid for the given time.
        
        Creates a regular 3D grid of query points, evaluates the SIREN,
        and returns the SDF values for marching cubes.
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
            # Get resolution and bounds
            resolution = request.query_resolution or 64
            bounds = request.query_bounds or self._default_bounds
            min_bound, max_bound = bounds
            
            # Create 3D grid of query points
            x = np.linspace(min_bound[0], max_bound[0], resolution)
            y = np.linspace(min_bound[1], max_bound[1], resolution)
            z = np.linspace(min_bound[2], max_bound[2], resolution)
            
            # Create meshgrid and flatten
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
            
            # Add time dimension
            t_value = request.time
            t_column = np.full((len(points), 1), t_value, dtype=np.float32)
            xyzt = np.concatenate([points.astype(np.float32), t_column], axis=1)
            
            # Convert to tensor
            xyzt_tensor = torch.from_numpy(xyzt).to(self._device)
            
            # Run inference in batches to manage memory
            batch_size = 100_000
            sdf_values = []
            
            with torch.no_grad():
                for i in range(0, len(xyzt_tensor), batch_size):
                    batch = xyzt_tensor[i:i + batch_size]
                    sdf_batch = self._model(batch)
                    sdf_values.append(sdf_batch.cpu().numpy())
            
            # Combine and reshape to 3D grid
            sdf_flat = np.concatenate(sdf_values, axis=0).squeeze()
            sdf_grid = sdf_flat.reshape(resolution, resolution, resolution)
            
            # Compute inference time
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update state
            new_state = SimulationState(
                frame=request.frame,
                time=request.time,
            )
            
            return PredictionResult(
                sdf_grid=sdf_grid.astype(np.float32),
                sdf_bounds=bounds,
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
        pass
    
    def get_recommended_checkpoint(self) -> Optional[str]:
        """Return path to default checkpoint."""
        if self._project_dir is None:
            return None
        
        search_paths = [
            self._project_dir / "nif-cloth4d" / "checkpoints" / "best_model.pt",
            self._project_dir / "checkpoints" / "best_model.pt",
            self._project_dir / "nif-cloth4d" / "model.pt",
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
        
        for pattern in ["**/*.pt", "**/*.pth"]:
            for path in self._project_dir.glob(pattern):
                if path.is_file() and "checkpoint" not in path.stem.lower():
                    checkpoints.append(str(path))
        
        return sorted(set(checkpoints))
    
    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._loaded = False
        
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
            "architecture": "SIREN (Compact)",
            "input_dim": 4,
            "output_dim": 1,
            "hidden_dim": 128,
            "num_layers": 4,
            "activation": "sine",
            "omega_0": 30.0,
            "parameter_count": "~66K",
            "project": "13-NIF-Cloth4D",
        }
