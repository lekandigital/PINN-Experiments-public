"""
PINN-Lite-Foil Export Adapter.

Adapter for Project 15 - validates that the shared pipeline produces
identical artifacts to the original project-specific export code.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import ExportConfig, TensorSpec, OutputType, TargetPlatform

logger = logging.getLogger(__name__)


class NavierStokesPINN(nn.Module):
    """
    Simple MLP for airfoil flow prediction.
    
    This is a simplified version that matches the architecture from Project 15.
    Input: [x, y, aoa] (position + angle of attack)
    Output: [u, v, p] (velocity components + pressure)
    """
    
    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_units: int = 64,
        nu: float = 1e-3,
    ):
        super().__init__()
        self.nu = nu
        
        layers = []
        in_features = 3  # x, y, aoa
        
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.Tanh())
            in_features = hidden_units
        
        layers.append(nn.Linear(hidden_units, 3))  # u, v, p
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(
    checkpoint_path: str,
    config: ExportConfig,
    config_path: Optional[str] = None,
) -> nn.Module:
    """
    Load PINN-Lite-Foil model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        config: Export configuration
        config_path: Optional path to model config JSON
        
    Returns:
        Loaded PyTorch model
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Try to find config JSON
    if config_path is None:
        # Look for config in same directory
        possible_configs = [
            checkpoint_path.with_suffix('.json'),
            checkpoint_path.parent / 'distillation_config.json',
            checkpoint_path.parent / 'config.json',
        ]
        for cfg_path in possible_configs:
            if cfg_path.exists():
                config_path = str(cfg_path)
                break
    
    # Load model config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Determine architecture
        if 'student_layers' in model_config:
            hidden_layers = model_config['student_layers']
            hidden_units = model_config['student_units']
        else:
            hidden_layers = model_config.get('hidden_layers', 4)
            hidden_units = model_config.get('hidden_units', 64)
        
        nu = model_config.get('nu', 1e-3)
    else:
        # Default architecture
        hidden_layers = 4
        hidden_units = 64
        nu = 1e-3
        logger.warning("No config file found, using default architecture")
    
    # Create model
    model = NavierStokesPINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        nu=nu,
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Loaded PINN-Lite-Foil model: {hidden_layers} layers × {hidden_units} units")
    
    return model


def get_default_config() -> ExportConfig:
    """
    Get default export configuration for PINN-Lite-Foil.
    
    Returns:
        ExportConfig with project-specific settings
    """
    return ExportConfig(
        project_name="pinn_lite_foil",
        model_version="1.0.0",
        input_specs=[
            TensorSpec(
                name="input",
                shape=[-1, 3],
                dtype="float32",
                description="Spatial coordinates and angle of attack [x, y, aoa]",
                min_value=-1.0,
                max_value=1.0,
            )
        ],
        output_specs=[
            TensorSpec(
                name="output",
                shape=[-1, 3],
                dtype="float32",
                description="Flow quantities [u, v, p] (velocity components and pressure)",
            )
        ],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        enable_int8=True,
        enable_fp16=True,
        targets=[
            TargetPlatform.DESKTOP,
            TargetPlatform.BROWSER,
            TargetPlatform.MOBILE,
            TargetPlatform.EDGE_TENSORRT,
        ],
        description="Knowledge-distilled PINN for airfoil flow prediction",
        physics_domain="aerodynamics",
        expected_output_type=OutputType.FIELD_VECTOR,
    )


def validate_against_original(
    shared_onnx_path: str,
    original_onnx_path: str,
    num_samples: int = 100,
    tolerance: float = 1e-6,
) -> dict:
    """
    Validate that the shared pipeline produces identical results to original.
    
    Args:
        shared_onnx_path: Path to ONNX from shared pipeline
        original_onnx_path: Path to ONNX from original Project 15 pipeline
        num_samples: Number of test samples
        tolerance: Maximum allowed difference
        
    Returns:
        Validation results dictionary
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not available for validation")
        return {"error": "onnxruntime not available"}
    
    # Load both models
    shared_sess = ort.InferenceSession(shared_onnx_path, providers=['CPUExecutionProvider'])
    original_sess = ort.InferenceSession(original_onnx_path, providers=['CPUExecutionProvider'])
    
    shared_input = shared_sess.get_inputs()[0].name
    shared_output = shared_sess.get_outputs()[0].name
    original_input = original_sess.get_inputs()[0].name
    original_output = original_sess.get_outputs()[0].name
    
    max_diff = 0.0
    mean_diff = 0.0
    
    for _ in range(num_samples):
        # Random input
        test_input = np.random.randn(10, 3).astype(np.float32)
        
        # Run both
        shared_out = shared_sess.run([shared_output], {shared_input: test_input})[0]
        original_out = original_sess.run([original_output], {original_input: test_input})[0]
        
        diff = np.abs(shared_out - original_out)
        max_diff = max(max_diff, np.max(diff))
        mean_diff += np.mean(diff)
    
    mean_diff /= num_samples
    
    passed = max_diff < tolerance
    
    return {
        "passed": passed,
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "tolerance": tolerance,
        "num_samples": num_samples,
    }
