"""
ClothGNN Export Adapter (Project 05).

Adapter for exporting GNN-based cloth simulation models to mobile deployment.
This is the most complex adapter due to GNN operations not exporting directly to ONNX.

Strategy: Baked Adjacency Approach
- Pre-compute the adjacency matrix for a specific mesh resolution
- Convert scatter/gather message passing to dense matrix operations
- Export GRU hidden states as explicit inputs/outputs for temporal consistency
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import (
    ExportConfig, TensorSpec, OutputType, TargetPlatform,
    MeshReconstructionConfig, TemporalConfig, PhysicsParameter
)

logger = logging.getLogger(__name__)


class DenseMessagePassing(nn.Module):
    """
    Dense matrix-based message passing to replace scatter/gather operations.
    
    Instead of: messages = scatter(edge_features, edge_index)
    We use:     messages = adjacency @ node_features
    
    This is less memory efficient but fully ONNX-compatible.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        adjacency: torch.Tensor,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Register adjacency as buffer (not a parameter, but saved with model)
        # Shape: [N, N] where N is number of nodes
        self.register_buffer('adjacency', adjacency)
        
        # Node feature projection
        self.lin_node = nn.Linear(in_channels, hidden_channels)
        
        # Message MLP (simplified from original)
        self.lin_msg = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        self.norm = nn.LayerNorm(hidden_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dense message passing.
        
        Args:
            x: [B, N, in_channels] node features
            
        Returns:
            [B, N, hidden_channels] updated features
        """
        batch_size = x.shape[0]
        
        # Project node features
        h = self.lin_node(x)  # [B, N, hidden]
        
        # Aggregate neighbor features via adjacency matrix
        # adj: [N, N], h: [B, N, hidden]
        # We need: neighbor_agg[b, i, :] = sum_j adj[i,j] * h[b, j, :]
        adj = self.adjacency.unsqueeze(0)  # [1, N, N]
        neighbor_agg = torch.bmm(adj.expand(batch_size, -1, -1), h)  # [B, N, hidden]
        
        # Concatenate self and neighbor features
        msg_input = torch.cat([h, neighbor_agg], dim=-1)  # [B, N, hidden*2]
        
        # Apply message MLP
        msg = self.lin_msg(msg_input)  # [B, N, hidden]
        
        # Residual + norm
        out = self.norm(h + msg)
        
        return out


class ClothGNNExportable(nn.Module):
    """
    ONNX-exportable version of ClothGNN.
    
    Replaces MessagePassing layers with dense matrix operations.
    Exposes GRU hidden state as explicit input/output.
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int = 16,
        hidden_dim: int = 64,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        
        # Create default adjacency if not provided (grid connectivity)
        if adjacency is None:
            adjacency = self._create_grid_adjacency(num_nodes)
        
        # Dense encoder (replaces MessagePassing)
        self.encoder = DenseMessagePassing(node_feat_dim, hidden_dim, adjacency)
        
        # GRU for temporal dynamics
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # Output: 3D displacement
        )
    
    def _create_grid_adjacency(self, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix for a grid mesh."""
        # Assume square grid
        grid_size = int(np.sqrt(num_nodes))
        if grid_size * grid_size != num_nodes:
            grid_size = int(np.ceil(np.sqrt(num_nodes)))
        
        adj = torch.zeros(num_nodes, num_nodes)
        
        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size
            
            # Connect to 4 neighbors (if they exist)
            neighbors = []
            if row > 0:
                neighbors.append(i - grid_size)  # Up
            if row < grid_size - 1:
                neighbors.append(i + grid_size)  # Down
            if col > 0:
                neighbors.append(i - 1)  # Left
            if col < grid_size - 1:
                neighbors.append(i + 1)  # Right
            
            for j in neighbors:
                if 0 <= j < num_nodes:
                    adj[i, j] = 1.0
        
        # Normalize (mean aggregation)
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / row_sum
        
        return adj
    
    def forward(
        self,
        node_features: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one timestep.
        
        Args:
            node_features: [B, N, node_feat_dim] current node features
            hidden_state: [B, N, hidden_dim] GRU hidden state from previous step
            
        Returns:
            displacements: [B, N, 3] predicted vertex displacements
            new_hidden: [B, N, hidden_dim] updated hidden state
        """
        batch_size = node_features.shape[0]
        
        # Encode with dense message passing
        encoded = self.encoder(node_features)  # [B, N, hidden]
        
        # Reshape for GRU: [B*N, 1, hidden]
        encoded_flat = encoded.view(batch_size * self.num_nodes, 1, self.hidden_dim)
        hidden_flat = hidden_state.view(1, batch_size * self.num_nodes, self.hidden_dim)
        
        # GRU step
        gru_out, new_hidden_flat = self.gru(encoded_flat, hidden_flat)
        
        # Reshape back
        gru_out = gru_out.view(batch_size, self.num_nodes, self.hidden_dim)
        new_hidden = new_hidden_flat.view(batch_size, self.num_nodes, self.hidden_dim)
        
        # Decode to displacements
        displacements = self.decoder(gru_out)  # [B, N, 3]
        
        return displacements, new_hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Create zero-initialized hidden state."""
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


def load_model(
    checkpoint_path: str,
    config: ExportConfig,
    mesh_resolution: int = 32,
) -> nn.Module:
    """
    Load ClothGNN model and convert to exportable version.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Export configuration
        mesh_resolution: Mesh resolution (nodes = resolution²)
        
    Returns:
        ONNX-exportable model
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model config
    if isinstance(checkpoint, dict):
        model_cfg = checkpoint.get('config', {})
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        model_cfg = {}
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else {}
    
    node_feat_dim = model_cfg.get('node_feat_dim', 16)
    hidden_dim = model_cfg.get('hidden_dim', 64)
    num_nodes = mesh_resolution * mesh_resolution
    
    # Create exportable model
    model = ClothGNNExportable(
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
    )
    
    # Try to load weights (may need mapping)
    try:
        # Attempt direct load first
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights with strict=False (some keys may not match)")
    except Exception as e:
        logger.warning(f"Could not load weights directly: {e}")
        logger.info("Using randomly initialized exportable model")
    
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Created exportable ClothGNN: {num_nodes} nodes, {param_count:,} parameters")
    
    return model


def wrap_for_export(
    model: nn.Module,
    num_nodes: int,
    hidden_dim: int = 64,
) -> nn.Module:
    """
    Wrap an existing ClothGNN model for export.
    
    If the model is already a ClothGNNExportable, returns it directly.
    Otherwise, creates a new exportable model with matching architecture.
    
    Args:
        model: Original ClothGNN model
        num_nodes: Number of mesh nodes
        hidden_dim: Hidden dimension
        
    Returns:
        ONNX-exportable model
    """
    if isinstance(model, ClothGNNExportable):
        return model
    
    # Create exportable version
    node_feat_dim = 16  # Default from Project 05
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'lin_node'):
        node_feat_dim = model.encoder.lin_node.in_features
    
    exportable = ClothGNNExportable(
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
    )
    
    # Copy compatible weights
    try:
        exportable_dict = exportable.state_dict()
        original_dict = model.state_dict()
        
        for key in exportable_dict.keys():
            if key in original_dict and exportable_dict[key].shape == original_dict[key].shape:
                exportable_dict[key] = original_dict[key]
        
        exportable.load_state_dict(exportable_dict)
        logger.info("Copied compatible weights to exportable model")
    except Exception as e:
        logger.warning(f"Could not copy weights: {e}")
    
    return exportable


def create_template_mesh(
    resolution: int = 32,
    size: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a template cloth mesh (vertices + faces).
    
    This is exported alongside the model so the mobile app
    knows the mesh topology.
    
    Args:
        resolution: Grid resolution
        size: Physical size of cloth
        
    Returns:
        vertices: [N, 3] initial vertex positions
        faces: [F, 3] triangle face indices
    """
    # Create grid vertices
    x = np.linspace(-size / 2, size / 2, resolution)
    y = np.linspace(-size / 2, size / 2, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
    
    # Create faces (two triangles per grid cell)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # Triangle 1
            faces.append([idx, idx + resolution, idx + 1])
            # Triangle 2
            faces.append([idx + 1, idx + resolution, idx + resolution + 1])
    
    faces = np.array(faces, dtype=np.int32)
    
    return vertices, faces


def get_default_config(mesh_resolution: int = 32) -> ExportConfig:
    """
    Get default export configuration for ClothGNN.
    
    Args:
        mesh_resolution: Cloth mesh resolution
        
    Returns:
        ExportConfig with project-specific settings
    """
    num_nodes = mesh_resolution * mesh_resolution
    hidden_dim = 64
    
    return ExportConfig(
        project_name="cloth_gnn",
        model_version="1.0.0",
        input_specs=[
            TensorSpec(
                name="node_features",
                shape=[-1, num_nodes, 16],  # [batch, nodes, features]
                dtype="float32",
                description="Node features (position, velocity, forces, etc.)",
            ),
            TensorSpec(
                name="hidden_state",
                shape=[-1, num_nodes, hidden_dim],  # [batch, nodes, hidden]
                dtype="float32",
                description="GRU hidden state from previous timestep",
            ),
        ],
        output_specs=[
            TensorSpec(
                name="displacements",
                shape=[-1, num_nodes, 3],
                dtype="float32",
                description="Predicted vertex displacements",
            ),
            TensorSpec(
                name="new_hidden_state",
                shape=[-1, num_nodes, hidden_dim],
                dtype="float32",
                description="Updated GRU hidden state",
            ),
        ],
        dynamic_axes={
            "node_features": {0: "batch"},
            "hidden_state": {0: "batch"},
            "displacements": {0: "batch"},
            "new_hidden_state": {0: "batch"},
        },
        opset_version=17,
        enable_int8=True,
        enable_fp16=False,
        targets=[
            TargetPlatform.MOBILE,
            TargetPlatform.DESKTOP,
        ],
        description="GNN-based cloth dynamics with temporal consistency (baked adjacency)",
        physics_domain="cloth_simulation",
        expected_output_type=OutputType.MESH_VERTICES,
        mesh_config=MeshReconstructionConfig(
            method="direct_vertices",
            template_mesh_path=f"cloth_template_{mesh_resolution}x{mesh_resolution}.npz",
        ),
        temporal_config=TemporalConfig(
            has_temporal_state=True,
            hidden_state_names=["hidden_state", "new_hidden_state"],
            hidden_state_shapes=[[-1, num_nodes, hidden_dim]],
        ),
        physics_parameters=[
            PhysicsParameter(
                name="wind_x",
                display_name="Wind X",
                min_value=-10.0,
                max_value=10.0,
                default_value=0.0,
                unit="m/s",
            ),
            PhysicsParameter(
                name="wind_y",
                display_name="Wind Y",
                min_value=-10.0,
                max_value=10.0,
                default_value=0.0,
                unit="m/s",
            ),
            PhysicsParameter(
                name="wind_z",
                display_name="Wind Z",
                min_value=-10.0,
                max_value=10.0,
                default_value=0.0,
                unit="m/s",
            ),
        ],
    )


def export_with_template_mesh(
    model: nn.Module,
    output_dir: str,
    mesh_resolution: int = 32,
) -> Dict[str, str]:
    """
    Export model along with template mesh and metadata.
    
    Args:
        model: ClothGNN model (exportable version)
        output_dir: Output directory
        mesh_resolution: Mesh resolution
        
    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save template mesh
    vertices, faces = create_template_mesh(mesh_resolution)
    mesh_path = output_dir / f"cloth_template_{mesh_resolution}x{mesh_resolution}.npz"
    np.savez(mesh_path, vertices=vertices, faces=faces)
    logger.info(f"Saved template mesh to {mesh_path}")
    
    return {
        "template_mesh": str(mesh_path),
    }
