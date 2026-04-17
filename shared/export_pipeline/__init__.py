"""
Shared ONNX Export Pipeline for PINN-Experiments.

This package provides a unified, model-agnostic infrastructure for exporting
PyTorch models to ONNX format, with support for quantization, platform-specific
optimization, and comprehensive benchmarking.

Supported target platforms:
- Browser (ONNX Runtime Web, WebNN)
- Mobile (ONNX Runtime Mobile, .ort format)
- Edge - TensorRT (NVIDIA GPUs)
- Edge - OpenVINO (Intel hardware)
- Desktop (CPU/CUDA)

Usage:
    from shared.export_pipeline import ExportConfig, run_pipeline
    
    config = ExportConfig(
        project_name="cloth4d",
        model_version="1.0.0",
        input_specs=[TensorSpec("coords", [-1, 4], "float32")],
        output_specs=[TensorSpec("sdf", [-1, 1], "float32")],
        targets=["browser", "mobile"]
    )
    
    result = run_pipeline(model, config, output_dir="exports/")
"""

from .config import (
    ExportConfig,
    TensorSpec,
    MeshReconstructionConfig,
    QuantizationConfig,
)
from .exporter import ONNXExporter
from .quantize import quantize_model, QuantizationResult
from .optimize import optimize_for_platform
from .pipeline import run_pipeline, PipelineResult

__version__ = "0.1.0"
__all__ = [
    "ExportConfig",
    "TensorSpec", 
    "MeshReconstructionConfig",
    "QuantizationConfig",
    "ONNXExporter",
    "quantize_model",
    "QuantizationResult",
    "optimize_for_platform",
    "run_pipeline",
    "PipelineResult",
]
