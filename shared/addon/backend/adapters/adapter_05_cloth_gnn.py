"""
Backend adapter for Project 05: ClothGNN

Encoder-GRU-Decoder GNN for lightweight cloth simulation.
~70K params, 167 FPS.
Output: per-vertex 3D displacement vectors.

Model Details:
- Input: Cloth mesh graph (vertices + edges + features)
- Output: Per-vertex 3D displacement vectors
- Temporal: GRU hidden state carries frame-to-frame information
- Use case: Lightweight/mobile cloth, many simultaneous garments

Note: This backend requires torch_geometric for GNN operations.
If torch_geometric is not available, it will fail gracefully.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple
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


class ClothGNNBackend(ModelBackend):
    """
    Adapter for Project 05's ClothGNN model.
    
    Uses an Encoder-GRU-Decoder architecture with message passing
    to predict per-vertex displacements. Requires torch_geometric.
    """
    
    def __init__(self):
        self._model = None
        self._device = None
        self._checkpoint_path: Optional[str] = None
        self._loaded = False
        self._config = None
        
        # GRU hidden state
        self._hidden_state = None
        
        # Check for torch_geometric availability
        self._has_pyg = self._check_pyg()
        
        # Project paths
        self._project_dir = self._find_project_dir()
    
    def _check_pyg(self) -> bool:
        """Check if torch_geometric is available."""
        try:
            import torch_geometric
            return True
        except ImportError:
            return False
    
    def _find_project_dir(self) -> Optional[Path]:
        """Find the Project 05 directory."""
        current = Path(__file__).parent
        for _ in range(4):
            current = current.parent
        
        for name in ["05-clothgnn__project-space", "05-ClothGNN"]:
            project_dir = current / "projects" / name
            if project_dir.exists():
                return project_dir
        
        return None
    
    def get_capabilities(self) -> BackendCapabilities:
        """Return capabilities for the ClothGNN model."""
        return BackendCapabilities(
            category=ModelCategory.CLOTH_SIMULATION,
            output_format=OutputFormat.VERTEX_DISPLACEMENTS,
            input_requirements={
                InputRequirement.MESH_VERTICES,
                InputRequirement.MESH_TOPOLOGY,
                InputRequirement.TIME,
            },
            
            # Limited force support - forces baked into training
            supports_wind=False,
            supports_gravity=False,
            supports_custom_forces=False,
            
            # No material parameters
            supports_material_params=False,
            material_param_names=[],
            
            # Temporal behavior - GRU based
            is_temporal=True,
            supports_continuous_time=False,  # Discrete steps only
            needs_sequential_frames=True,    # GRU hidden state
            max_stable_frames=200,           # Rollout stability limit
            
            # No resolution control (direct vertex output)
            supports_resolution_control=False,
            
            # Performance
            typical_fps=167.0,
            typical_memory_mb=50.0,
            parameter_count=70_000,
            model_size_mb=0.3,
            
            # Mesh requirements
            needs_fixed_topology=True,  # GNN requires consistent graph
            max_vertices=20_000,
            
            # Metadata
            display_name="ClothGNN (Lightweight)",
            description="Lightweight GNN-based cloth simulation. "
                       "Encoder-GRU-Decoder architecture with ~70K parameters. "
                       "Best for mobile/web deployment and multiple garments. "
                       "Note: Forces are baked into training data, not controllable.",
            quality_tier="standard",
            icon="MESH_GRID",
            
            project_id="05-clothgnn",
            project_path="projects/05-clothgnn__project-space",
        )
    
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load the ClothGNN model from checkpoint."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required. Install: pip install torch")
        
        if not self._has_pyg:
            raise RuntimeError(
                "torch_geometric is required for ClothGNN backend. "
                "Install: pip install torch-geometric torch-scatter torch-sparse"
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
            
            if isinstance(checkpoint, dict):
                self._config = checkpoint.get("config", {})
                state_dict = checkpoint.get("model_state_dict", checkpoint)
            else:
                state_dict = checkpoint
                self._config = {}
            
            # Build model
            self._model = self._build_model()
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self._device)
            self._model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ClothGNN model: {e}")
        
        self._checkpoint_path = checkpoint_path
        self._loaded = True
        self._hidden_state = None
        
        print(f"[ClothGNN] Loaded model on {self._device}")
    
    def _build_model(self):
        """Build the ClothGNN model architecture."""
        import torch
        import torch.nn as nn
        
        try:
            from torch_geometric.nn import MessagePassing
            from torch_geometric.data import Data
        except ImportError:
            raise RuntimeError("torch_geometric is required")
        
        class EdgeConv(MessagePassing):
            """Edge convolution layer for cloth mesh."""
            def __init__(self, in_channels, out_channels):
                super().__init__(aggr='mean')
                self.mlp = nn.Sequential(
                    nn.Linear(2 * in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels),
                )
            
            def forward(self, x, edge_index):
                return self.propagate(edge_index, x=x)
            
            def message(self, x_i, x_j):
                return self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))
        
        class ClothGNNModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                
                node_in = config.get("node_features", 6)  # pos(3) + vel(3)
                hidden_dim = config.get("hidden_dim", 64)
                gru_dim = config.get("gru_dim", 64)
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(node_in, hidden_dim),
                    nn.ReLU(),
                )
                
                # Graph convolutions
                self.conv1 = EdgeConv(hidden_dim, hidden_dim)
                self.conv2 = EdgeConv(hidden_dim, hidden_dim)
                
                # GRU for temporal
                self.gru = nn.GRU(hidden_dim, gru_dim, batch_first=True)
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(gru_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3),  # displacement
                )
            
            def forward(self, x, edge_index, hidden=None):
                # Encode
                h = self.encoder(x)
                
                # Graph convolutions
                h = h + self.conv1(h, edge_index)
                h = h + self.conv2(h, edge_index)
                
                # GRU step
                h = h.unsqueeze(0)  # Add batch dim
                if hidden is None:
                    hidden = torch.zeros(1, h.size(1), self.gru.hidden_size, device=h.device)
                
                h, hidden = self.gru(h, hidden)
                h = h.squeeze(0)
                
                # Decode
                displacement = self.decoder(h)
                
                return displacement, hidden
        
        return ClothGNNModel(self._config or {})
    
    def _build_edge_index(self, faces: np.ndarray) -> np.ndarray:
        """Build edge index from faces."""
        edges = set()
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edges.add((min(v1, v2), max(v1, v2)))
        
        edge_list = list(edges)
        # Make bidirectional
        edge_index = []
        for e in edge_list:
            edge_index.append([e[0], e[1]])
            edge_index.append([e[1], e[0]])
        
        return np.array(edge_index, dtype=np.int64).T
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Predict per-vertex displacements from mesh state.
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
            # Get mesh data
            vertices = request.vertices
            if vertices is None:
                return PredictionResult(
                    state=request.state or SimulationState(),
                    confidence=0.0,
                    error_message="No vertices provided",
                )
            
            # Build node features (position + velocity placeholder)
            num_verts = len(vertices)
            
            # Get velocity from state if available
            prev_state = request.state
            if prev_state is not None and "_positions" in prev_state._backend_data:
                prev_pos = prev_state._backend_data["_positions"]
                velocity = (vertices - prev_pos) / request.delta_time
            else:
                velocity = np.zeros_like(vertices)
            
            node_features = np.concatenate([vertices, velocity], axis=1).astype(np.float32)
            node_tensor = torch.from_numpy(node_features).to(self._device)
            
            # Build edge index
            if request.edges is not None:
                edge_index = request.edges.T
            elif request.faces is not None:
                edge_index = self._build_edge_index(request.faces)
            else:
                return PredictionResult(
                    state=request.state or SimulationState(),
                    confidence=0.0,
                    error_message="No topology provided (edges or faces)",
                )
            
            edge_tensor = torch.from_numpy(edge_index.astype(np.int64)).to(self._device)
            
            # Get hidden state
            if prev_state is not None and prev_state.hidden is not None:
                hidden = prev_state.hidden
                if isinstance(hidden, np.ndarray):
                    hidden = torch.from_numpy(hidden).to(self._device)
            else:
                hidden = None
            
            # Run inference
            with torch.no_grad():
                displacement, new_hidden = self._model(node_tensor, edge_tensor, hidden)
            
            # Convert output
            displacement_np = displacement.cpu().numpy().astype(np.float32)
            new_hidden_np = new_hidden.cpu().numpy()
            
            # Update state
            new_state = SimulationState(
                frame=request.frame,
                time=request.time,
                hidden=new_hidden_np,
                _backend_data={"_positions": vertices.copy()},
            )
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            return PredictionResult(
                displacements=displacement_np,
                state=new_state,
                inference_time_ms=inference_time_ms,
                confidence=1.0,
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PredictionResult(
                state=request.state or SimulationState(),
                confidence=0.0,
                error_message=f"Inference error: {str(e)}",
            )
    
    def reset(self) -> None:
        """Reset GRU hidden state."""
        self._hidden_state = None
    
    def get_recommended_checkpoint(self) -> Optional[str]:
        """Return path to default checkpoint."""
        if self._project_dir is None:
            return None
        
        search_paths = [
            self._project_dir / "clothgnn" / "checkpoints" / "best_model.pt",
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
        
        for pattern in ["**/*.pt", "**/*.pth"]:
            for path in self._project_dir.glob(pattern):
                if path.is_file():
                    checkpoints.append(str(path))
        
        return sorted(set(checkpoints))
    
    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._hidden_state = None
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
            "architecture": "Encoder-GRU-Decoder GNN",
            "gnn_type": "EdgeConv",
            "temporal": "GRU",
            "has_pyg": self._has_pyg,
            "project": "05-ClothGNN",
        }
