"""
CoastFlow GNN Export Adapter (Project 06).

Adapter for exporting hierarchical GNN models for coastal flow simulation.
Uses pre-computed scenario approach due to TopK pooling complexity.

Strategy: Pre-computed Scenarios
- TopK pooling is input-dependent and hard to export
- Pre-compute outputs for a discrete set of parameter combinations
- Export simplified decoder for interpolation
- Store scenario embeddings for lookup
"""

import json
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


class ScenarioEncoder(nn.Module):
    """
    Simplified encoder that maps scenario parameters to latent embeddings.
    
    This replaces the full hierarchical GNN encoder for export.
    During offline preprocessing, we run the full encoder on sampled
    scenarios and train this simplified network to reproduce those embeddings.
    """
    
    def __init__(
        self,
        num_params: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_params = num_params
        self.embed_dim = embed_dim
        
        # MLP to encode scenario parameters
        self.encoder = nn.Sequential(
            nn.Linear(num_params, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Encode scenario parameters.
        
        Args:
            params: [B, num_params] scenario parameters
            
        Returns:
            [B, embed_dim] scenario embedding
        """
        return self.encoder(params)


class FieldDecoder(nn.Module):
    """
    Decoder that maps scenario embeddings + spatial queries to field values.
    
    This is a more exportable alternative that uses grid-based output
    instead of arbitrary point queries.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        grid_size: int = 64,
        output_channels: int = 3,  # velocity (u, v) + elevation (eta)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.output_channels = output_channels
        
        # Generate fixed grid queries
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [G*G, 2]
        self.register_buffer('query_grid', grid)
        
        # Per-query decoder (embedding + position -> values)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_channels),
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode field on fixed grid.
        
        Args:
            embedding: [B, embed_dim] scenario embedding
            
        Returns:
            [B, grid_size, grid_size, output_channels] field values on grid
        """
        batch_size = embedding.shape[0]
        num_queries = self.query_grid.shape[0]
        
        # Expand embedding for all query points
        emb_expanded = embedding.unsqueeze(1).expand(-1, num_queries, -1)  # [B, G*G, embed]
        grid_expanded = self.query_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, G*G, 2]
        
        # Concatenate
        decoder_input = torch.cat([emb_expanded, grid_expanded], dim=-1)  # [B, G*G, embed+2]
        
        # Decode
        output = self.decoder(decoder_input)  # [B, G*G, channels]
        
        # Reshape to grid
        output = output.view(batch_size, self.grid_size, self.grid_size, self.output_channels)
        
        return output


class CoastFlowExportable(nn.Module):
    """
    ONNX-exportable version of CoastFlow GNN.
    
    Uses scenario encoder + field decoder architecture that avoids
    TopK pooling and hierarchical message passing.
    """
    
    def __init__(
        self,
        num_params: int = 4,
        embed_dim: int = 64,
        grid_size: int = 64,
        output_channels: int = 3,
    ):
        super().__init__()
        self.num_params = num_params
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.output_channels = output_channels
        
        self.scenario_encoder = ScenarioEncoder(num_params, embed_dim)
        self.field_decoder = FieldDecoder(embed_dim, grid_size, output_channels)
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.
        
        Args:
            params: [B, num_params] scenario parameters
                   (e.g., inlet_velocity, wave_period, bathymetry_profile, time)
            
        Returns:
            [B, grid_size, grid_size, output_channels] predicted flow field
        """
        embedding = self.scenario_encoder(params)
        field = self.field_decoder(embedding)
        return field


class ScenarioLookupExportable(nn.Module):
    """
    Alternative export using pre-computed scenario lookup.
    
    For demos where we want exact outputs for specific scenarios,
    this stores pre-computed outputs and interpolates between them.
    """
    
    def __init__(
        self,
        scenario_params: torch.Tensor,  # [S, num_params]
        scenario_outputs: torch.Tensor,  # [S, H, W, C]
        num_interpolate: int = 4,  # Number of nearest scenarios to interpolate
    ):
        super().__init__()
        self.num_interpolate = num_interpolate
        
        # Store as buffers
        self.register_buffer('scenario_params', scenario_params)
        self.register_buffer('scenario_outputs', scenario_outputs)
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Lookup and interpolate from pre-computed scenarios.
        
        Args:
            params: [B, num_params] query parameters
            
        Returns:
            [B, H, W, C] interpolated output
        """
        batch_size = params.shape[0]
        num_scenarios = self.scenario_params.shape[0]
        
        # Compute distances to all scenarios
        # params: [B, P], scenario_params: [S, P]
        diff = params.unsqueeze(1) - self.scenario_params.unsqueeze(0)  # [B, S, P]
        distances = torch.norm(diff, dim=-1)  # [B, S]
        
        # Find K nearest scenarios
        _, indices = torch.topk(distances, self.num_interpolate, largest=False)  # [B, K]
        
        # Gather nearest outputs
        # scenario_outputs: [S, H, W, C]
        nearest_outputs = self.scenario_outputs[indices]  # [B, K, H, W, C]
        
        # Compute interpolation weights (inverse distance)
        nearest_distances = torch.gather(distances, 1, indices)  # [B, K]
        weights = 1.0 / (nearest_distances + 1e-6)  # [B, K]
        weights = weights / weights.sum(dim=-1, keepdim=True)  # [B, K]
        
        # Weighted sum
        weights = weights.view(batch_size, self.num_interpolate, 1, 1, 1)
        output = (nearest_outputs * weights).sum(dim=1)  # [B, H, W, C]
        
        return output


def load_model(
    checkpoint_path: str,
    config: ExportConfig,
    grid_size: int = 64,
) -> nn.Module:
    """
    Load CoastFlow model and create exportable version.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Export configuration
        grid_size: Output grid size
        
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
    
    num_params = model_cfg.get('num_params', 4)
    embed_dim = model_cfg.get('embed_dim', 64)
    output_channels = model_cfg.get('output_channels', 3)
    
    # Create exportable model
    model = CoastFlowExportable(
        num_params=num_params,
        embed_dim=embed_dim,
        grid_size=grid_size,
        output_channels=output_channels,
    )
    
    # Try to load weights
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights with strict=False")
    except Exception as e:
        logger.warning(f"Could not load weights: {e}")
        logger.info("Using randomly initialized exportable model")
    
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Created exportable CoastFlow: {grid_size}x{grid_size} grid, {param_count:,} parameters")
    
    return model


def generate_scenario_cache(
    original_model: nn.Module,
    param_ranges: Dict[str, Tuple[float, float]],
    samples_per_param: int = 10,
    output_dir: str = ".",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute outputs for a grid of scenarios.
    
    This is run offline to create the scenario cache for lookup-based export.
    
    Args:
        original_model: The full (non-exportable) model
        param_ranges: Dict of parameter name -> (min, max)
        samples_per_param: Number of samples per parameter
        output_dir: Directory to save cache
        
    Returns:
        scenario_params: [S, P] sampled parameters
        scenario_outputs: [S, H, W, C] corresponding outputs
    """
    import itertools
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter grid
    param_names = list(param_ranges.keys())
    param_grids = [
        np.linspace(v[0], v[1], samples_per_param)
        for v in param_ranges.values()
    ]
    
    # All combinations
    all_params = list(itertools.product(*param_grids))
    num_scenarios = len(all_params)
    
    logger.info(f"Generating cache for {num_scenarios} scenarios...")
    
    # Run model on all scenarios
    scenario_params = np.array(all_params, dtype=np.float32)
    
    original_model.eval()
    with torch.no_grad():
        # Process in batches
        batch_size = 32
        outputs = []
        
        for i in range(0, num_scenarios, batch_size):
            batch_params = torch.from_numpy(scenario_params[i:i+batch_size])
            batch_output = original_model(batch_params)
            outputs.append(batch_output.cpu().numpy())
    
    scenario_outputs = np.concatenate(outputs, axis=0)
    
    # Save cache
    cache_path = output_dir / "scenario_cache.npz"
    np.savez(
        cache_path,
        params=scenario_params,
        outputs=scenario_outputs,
        param_names=param_names,
        param_ranges=np.array([list(param_ranges.values())]),
    )
    logger.info(f"Saved scenario cache to {cache_path}")
    
    return scenario_params, scenario_outputs


def create_lookup_model(
    cache_path: str,
    num_interpolate: int = 4,
) -> ScenarioLookupExportable:
    """
    Create lookup-based exportable model from cache.
    
    Args:
        cache_path: Path to scenario cache
        num_interpolate: Number of scenarios to interpolate
        
    Returns:
        Lookup-based exportable model
    """
    cache = np.load(cache_path)
    
    scenario_params = torch.from_numpy(cache['params'])
    scenario_outputs = torch.from_numpy(cache['outputs'])
    
    model = ScenarioLookupExportable(
        scenario_params=scenario_params,
        scenario_outputs=scenario_outputs,
        num_interpolate=num_interpolate,
    )
    
    logger.info(f"Created lookup model with {scenario_params.shape[0]} scenarios")
    
    return model


def get_default_config(grid_size: int = 64) -> ExportConfig:
    """
    Get default export configuration for CoastFlow.
    
    Args:
        grid_size: Output grid resolution
        
    Returns:
        ExportConfig with project-specific settings
    """
    return ExportConfig(
        project_name="coastflow_gnn",
        model_version="1.0.0",
        input_specs=[
            TensorSpec(
                name="scenario_params",
                shape=[-1, 4],  # [batch, num_params]
                dtype="float32",
                description="Scenario parameters: inlet_velocity, wave_period, bathymetry, time",
            ),
        ],
        output_specs=[
            TensorSpec(
                name="flow_field",
                shape=[-1, grid_size, grid_size, 3],
                dtype="float32",
                description="Flow field: u velocity, v velocity, surface elevation",
            ),
        ],
        dynamic_axes={
            "scenario_params": {0: "batch"},
            "flow_field": {0: "batch"},
        },
        opset_version=17,
        enable_int8=False,  # Keep FP32 for physics accuracy
        enable_fp16=True,
        targets=[
            TargetPlatform.BROWSER,
            TargetPlatform.DESKTOP,
        ],
        description="Coastal flow simulation (simplified for web demo)",
        physics_domain="fluid_dynamics",
        expected_output_type=OutputType.VELOCITY_FIELD,
        mesh_config=MeshReconstructionConfig(
            method="structured_grid",
            resolution=(grid_size, grid_size),
        ),
        physics_parameters=[
            PhysicsParameter(
                name="inlet_velocity",
                display_name="Inlet Velocity",
                min_value=0.0,
                max_value=5.0,
                default_value=1.0,
                unit="m/s",
            ),
            PhysicsParameter(
                name="wave_period",
                display_name="Wave Period",
                min_value=1.0,
                max_value=20.0,
                default_value=8.0,
                unit="s",
            ),
            PhysicsParameter(
                name="bathymetry_profile",
                display_name="Bathymetry Profile",
                min_value=0.0,
                max_value=1.0,
                default_value=0.5,
                unit="normalized",
            ),
            PhysicsParameter(
                name="time",
                display_name="Simulation Time",
                min_value=0.0,
                max_value=100.0,
                default_value=0.0,
                unit="s",
            ),
        ],
    )


def create_coastline_mask(
    grid_size: int = 64,
    coastline_type: str = "simple",
) -> np.ndarray:
    """
    Create a coastline mask for visualization.
    
    Args:
        grid_size: Grid resolution
        coastline_type: Type of coastline geometry
        
    Returns:
        [H, W] binary mask (1 = water, 0 = land)
    """
    mask = np.ones((grid_size, grid_size), dtype=np.float32)
    
    if coastline_type == "simple":
        # Simple straight coastline on right side
        mask[:, -grid_size // 4:] = 0
        
    elif coastline_type == "bay":
        # Bay geometry
        center_y = grid_size // 2
        for i in range(grid_size):
            # Curved coastline
            x_coast = int(grid_size * 0.75 - 0.2 * grid_size * np.cos(np.pi * i / grid_size))
            mask[i, x_coast:] = 0
            
    elif coastline_type == "island":
        # Island in center
        cx, cy = grid_size // 2, grid_size // 2
        radius = grid_size // 6
        for i in range(grid_size):
            for j in range(grid_size):
                if (i - cx) ** 2 + (j - cy) ** 2 < radius ** 2:
                    mask[i, j] = 0
    
    return mask


def export_with_coastline(
    model: nn.Module,
    output_dir: str,
    grid_size: int = 64,
    coastline_type: str = "simple",
) -> Dict[str, str]:
    """
    Export model along with coastline geometry.
    
    Args:
        model: CoastFlow model (exportable version)
        output_dir: Output directory
        grid_size: Grid resolution
        coastline_type: Type of coastline
        
    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save coastline mask
    mask = create_coastline_mask(grid_size, coastline_type)
    mask_path = output_dir / f"coastline_mask_{coastline_type}.npy"
    np.save(mask_path, mask)
    logger.info(f"Saved coastline mask to {mask_path}")
    
    # Save metadata
    metadata = {
        "grid_size": grid_size,
        "coastline_type": coastline_type,
        "output_channels": ["u_velocity", "v_velocity", "surface_elevation"],
        "coordinate_system": {
            "x_range": [0, 1],
            "y_range": [0, 1],
            "z_scale": "meters",
        },
    }
    
    metadata_path = output_dir / "coastflow_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    return {
        "coastline_mask": str(mask_path),
        "metadata": str(metadata_path),
    }
