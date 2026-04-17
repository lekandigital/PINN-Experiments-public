"""
NIF-Cloth4D Export Adapter (Project 13).

Adapter for exporting SIREN-based 4D cloth SDF models to browser deployment.
Handles the specific requirements of SIREN networks including:
- Sine activation export to ONNX
- Query grid pre-computation
- Metadata for marching cubes reconstruction
"""

import logging
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import (
    ExportConfig, TensorSpec, OutputType, TargetPlatform,
    MeshReconstructionConfig, PhysicsParameter
)

logger = logging.getLogger(__name__)


class SineActivation(nn.Module):
    """Sinusoidal activation for SIREN networks."""
    
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class FourierFeatureSIREN(nn.Module):
    """
    Fourier-Feature SIREN network for implicit SDF prediction.
    
    Matches the architecture from Project 13's nif_cloth4d.py.
    """
    
    def __init__(
        self,
        in_dim: int = 4,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        w0: float = 30.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        total_in = in_dim + cond_dim
        
        # First layer with higher w0
        self.first_linear = nn.Linear(total_in, hidden_dim)
        self.first_act = SineActivation(w0=w0)
        
        # SIREN initialization
        with torch.no_grad():
            self.first_linear.weight.uniform_(-1.0 / total_in, 1.0 / total_in)
        
        # Hidden layers
        self.hidden = nn.ModuleList()
        for _ in range(hidden_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            self.hidden.append(lin)
            self.hidden.append(SineActivation(w0=1.0))
            
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_dim)
                lin.weight.uniform_(-bound, bound)
        
        # Output layer
        self.final_linear = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.final_linear.weight.fill_(0.0)
            self.final_linear.bias.fill_(0.0)
    
    def forward(
        self,
        coords: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is not None:
            x = torch.cat([coords, cond], dim=-1)
        else:
            x = coords
        
        x = self.first_linear(x)
        x = self.first_act(x)
        
        for layer in self.hidden:
            x = layer(x)
        
        sdf = self.final_linear(x)
        return sdf


class Cloth4DExportWrapper(nn.Module):
    """
    Wrapper that flattens grid input for ONNX export.
    
    The browser viewer will:
    1. Generate a 3D grid of query points
    2. Pass through this model to get SDF values
    3. Run marching cubes on the SDF grid
    
    This wrapper accepts flattened grid coordinates [B, N, 4] where N = res³
    and returns SDF values [B, N, 1].
    """
    
    def __init__(self, model: FourierFeatureSIREN):
        super().__init__()
        self.model = model
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for grid query.
        
        Args:
            coords: [B, N, 4] grid coordinates (x, y, z, t)
            
        Returns:
            [B, N, 1] SDF values
        """
        # Handle both [B, N, 4] and [N, 4] inputs
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        
        batch_size, num_points, dim = coords.shape
        
        # Flatten to [B*N, 4]
        flat_coords = coords.view(-1, dim)
        
        # Run model
        sdf = self.model(flat_coords)
        
        # Reshape to [B, N, 1]
        sdf = sdf.view(batch_size, num_points, 1)
        
        return sdf


def load_model(
    checkpoint_path: str,
    config: ExportConfig,
) -> nn.Module:
    """
    Load NIF-Cloth4D model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Export configuration
        
    Returns:
        Loaded and wrapped model ready for export
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Determine model config
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            model_cfg = checkpoint['config']
        else:
            # Try to infer from state dict
            model_cfg = {
                'in_dim': 4,
                'cond_dim': 0,
                'hidden_dim': 256,
                'hidden_layers': 5,
                'w0': 30.0,
            }
        
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        # Assume it's a model directly
        return wrap_for_export(checkpoint)
    
    # Create model
    model = FourierFeatureSIREN(
        in_dim=model_cfg.get('in_dim', 4),
        cond_dim=model_cfg.get('cond_dim', 0),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        hidden_layers=model_cfg.get('hidden_layers', 5),
        w0=model_cfg.get('w0', 30.0),
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded NIF-Cloth4D model: {param_count:,} parameters")
    
    return wrap_for_export(model)


def wrap_for_export(model: nn.Module) -> nn.Module:
    """
    Wrap model for ONNX export.
    
    Args:
        model: Original SIREN model
        
    Returns:
        Wrapped model ready for export
    """
    if isinstance(model, Cloth4DExportWrapper):
        return model
    return Cloth4DExportWrapper(model)


def generate_query_grid(
    resolution: int = 64,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    time_value: float = 0.0,
) -> np.ndarray:
    """
    Generate a 3D query grid for SDF evaluation.
    
    This grid is used by the browser viewer to query the model
    and reconstruct the mesh via marching cubes.
    
    Args:
        resolution: Grid resolution (res × res × res points)
        bounds: Spatial bounds (min, max)
        time_value: Time value to use for all points
        
    Returns:
        [res³, 4] array of query coordinates (x, y, z, t)
    """
    lin = np.linspace(bounds[0], bounds[1], resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
    
    # Flatten and stack
    coords = np.stack([
        xx.flatten(),
        yy.flatten(),
        zz.flatten(),
        np.full(resolution ** 3, time_value),
    ], axis=1).astype(np.float32)
    
    return coords


def save_query_grid(
    output_path: str,
    resolution: int = 64,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> str:
    """
    Save query grid to file for use by browser viewer.
    
    Args:
        output_path: Output path for grid file
        resolution: Grid resolution
        bounds: Spatial bounds
        
    Returns:
        Path to saved file
    """
    grid = generate_query_grid(resolution, bounds, time_value=0.0)
    np.save(output_path, grid)
    logger.info(f"Saved query grid ({resolution}³ = {len(grid)} points) to {output_path}")
    return output_path


def get_default_config(resolution: int = 64) -> ExportConfig:
    """
    Get default export configuration for NIF-Cloth4D.
    
    Args:
        resolution: Grid resolution for mesh reconstruction
        
    Returns:
        ExportConfig with project-specific settings
    """
    return ExportConfig(
        project_name="nif_cloth4d",
        model_version="1.0.0",
        input_specs=[
            TensorSpec(
                name="coords",
                shape=[-1, -1, 4],  # [batch, num_points, xyzt]
                dtype="float32",
                description="4D spacetime coordinates [x, y, z, t]",
                min_value=-1.0,
                max_value=1.0,
            )
        ],
        output_specs=[
            TensorSpec(
                name="sdf",
                shape=[-1, -1, 1],  # [batch, num_points, 1]
                dtype="float32",
                description="Signed distance field values",
            )
        ],
        dynamic_axes={
            "coords": {0: "batch", 1: "num_points"},
            "sdf": {0: "batch", 1: "num_points"},
        },
        opset_version=17,
        enable_int8=True,
        enable_fp16=False,  # FP16 can affect SIREN precision
        targets=[
            TargetPlatform.BROWSER,
            TargetPlatform.DESKTOP,
        ],
        description="Neural Implicit Field for 4D cloth dynamics using SIREN",
        physics_domain="cloth_simulation",
        expected_output_type=OutputType.MESH_SDF,
        mesh_config=MeshReconstructionConfig(
            method="marching_cubes",
            grid_resolution=resolution,
            iso_value=0.0,
            bounds=(-1.0, 1.0),
            smooth_iterations=2,
            decimate_target=50000,  # Limit faces for web
        ),
        physics_parameters=[
            PhysicsParameter(
                name="time",
                display_name="Time",
                input_index=3,  # t is 4th coordinate
                min_value=0.0,
                max_value=1.0,
                default_value=0.0,
                step=0.01,
                unit="s",
            ),
        ],
    )


def verify_sine_export(model: nn.Module, onnx_path: str) -> dict:
    """
    Verify that sine activations export correctly to ONNX.
    
    SIREN networks rely heavily on sine activations. This function
    verifies that torch.sin exports correctly and produces identical
    results in ONNX Runtime.
    
    Args:
        model: PyTorch SIREN model
        onnx_path: Path to exported ONNX model
        
    Returns:
        Verification results
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not available"}
    
    # Create test input
    test_input = torch.randn(1, 1000, 4)
    
    # PyTorch output
    model.eval()
    with torch.no_grad():
        torch_out = model(test_input).numpy()
    
    # ONNX output
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    onnx_out = sess.run([output_name], {input_name: test_input.numpy()})[0]
    
    # Compare
    max_diff = np.max(np.abs(torch_out - onnx_out))
    mean_diff = np.mean(np.abs(torch_out - onnx_out))
    
    # For SIREN, we expect very close results since sine is well-supported
    passed = max_diff < 1e-5
    
    return {
        "passed": passed,
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "note": "SIREN sine activations" + (" verified" if passed else " may have precision issues"),
    }
