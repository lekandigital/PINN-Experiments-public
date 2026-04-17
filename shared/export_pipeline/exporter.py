"""
ONNX Model Exporter.

Provides model-agnostic ONNX export functionality with metadata embedding,
verification against PyTorch, and support for various model architectures.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    from onnx import helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from .config import ExportConfig, TensorSpec, OutputType

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Exception raised during ONNX export."""
    pass


class VerificationError(Exception):
    """Exception raised when ONNX model output doesn't match PyTorch."""
    pass


def _ensure_dependencies():
    """Check that required dependencies are available."""
    if not ONNX_AVAILABLE:
        raise ImportError(
            "onnx package not found. Install with: pip install onnx"
        )
    if not ORT_AVAILABLE:
        raise ImportError(
            "onnxruntime package not found. Install with: pip install onnxruntime"
        )


def _create_dummy_input(
    input_specs: List[TensorSpec],
    batch_size: int = 1,
    device: torch.device = torch.device("cpu")
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Create dummy input tensors matching the input specifications.
    
    Args:
        input_specs: List of TensorSpec defining expected inputs
        batch_size: Batch size to use (replaces -1 in shapes)
        device: Device to create tensors on
        
    Returns:
        Single tensor or tuple of tensors
    """
    inputs = []
    for spec in input_specs:
        # Replace dynamic dims (-1) with concrete values
        shape = []
        for dim in spec.shape:
            if dim == -1:
                shape.append(batch_size)
            else:
                shape.append(dim)
        
        # Create tensor with appropriate dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        dtype = dtype_map.get(spec.dtype, torch.float32)
        
        # Generate random data within specified bounds
        if dtype in (torch.float32, torch.float16, torch.float64):
            tensor = torch.randn(shape, dtype=dtype, device=device)
            if spec.min_value is not None and spec.max_value is not None:
                # Scale to specified range
                tensor = tensor * (spec.max_value - spec.min_value) / 4 + (spec.max_value + spec.min_value) / 2
        else:
            tensor = torch.randint(0, 10, shape, dtype=dtype, device=device)
        
        inputs.append(tensor)
    
    if len(inputs) == 1:
        return inputs[0]
    return tuple(inputs)


def _embed_metadata(onnx_model: "onnx.ModelProto", metadata: Dict[str, Any]) -> "onnx.ModelProto":
    """
    Embed metadata into ONNX model properties.
    
    The metadata is stored as JSON in the model's metadata_props.
    This allows the viewer and other tools to understand how to
    interpret the model's output without external config files.
    
    Args:
        onnx_model: ONNX model to modify
        metadata: Dictionary of metadata to embed
        
    Returns:
        Modified ONNX model
    """
    # Clear existing metadata with our keys
    existing_props = {p.key: p.value for p in onnx_model.metadata_props 
                      if not p.key.startswith("pinn_export_")}
    
    # Add new metadata
    onnx_model.metadata_props.clear()
    
    # Re-add existing non-pinn props
    for key, value in existing_props.items():
        onnx_model.metadata_props.append(
            onnx.helper.make_opsetid(key, 0) if isinstance(value, int) 
            else onnx.StringStringEntryProto(key=key, value=str(value))
        )
    
    # Add our metadata as JSON
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="pinn_export_metadata",
            value=json.dumps(metadata)
        )
    )
    
    # Also add key fields as individual props for easy access
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="pinn_export_project",
            value=metadata.get("project_name", "")
        )
    )
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="pinn_export_version",
            value=metadata.get("model_version", "")
        )
    )
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="pinn_export_output_type",
            value=metadata.get("expected_output_type", "")
        )
    )
    
    return onnx_model


class ONNXExporter:
    """
    Model-agnostic ONNX exporter with verification and metadata embedding.
    
    This class handles the export of any PyTorch nn.Module to ONNX format,
    with automatic verification that the exported model produces outputs
    matching the original PyTorch model.
    
    Example:
        exporter = ONNXExporter(config)
        result = exporter.export(model, "model.onnx")
        print(f"Exported to {result.output_path}, size: {result.model_size_mb:.2f} MB")
    """
    
    def __init__(self, config: ExportConfig):
        """
        Initialize exporter with configuration.
        
        Args:
            config: Export configuration
        """
        _ensure_dependencies()
        self.config = config
    
    def _validate_model(self, model: nn.Module) -> None:
        """
        Validate that the model can be exported.
        
        Args:
            model: PyTorch model to validate
            
        Raises:
            ExportError: If model cannot be exported
        """
        # Check model is in eval mode
        if model.training:
            logger.warning("Model is in training mode, switching to eval mode")
            model.eval()
        
        # Try a forward pass with dummy input
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        dummy_input = _create_dummy_input(self.config.input_specs, batch_size=2, device=device)
        
        try:
            with torch.no_grad():
                if isinstance(dummy_input, tuple):
                    output = model(*dummy_input)
                else:
                    output = model(dummy_input)
        except Exception as e:
            raise ExportError(f"Model forward pass failed: {e}")
        
        # Validate output shape matches spec
        if isinstance(output, tuple):
            outputs = output
        else:
            outputs = (output,)
        
        if len(outputs) != len(self.config.output_specs):
            raise ExportError(
                f"Model has {len(outputs)} outputs but config specifies {len(self.config.output_specs)}"
            )
        
        for i, (out, spec) in enumerate(zip(outputs, self.config.output_specs)):
            expected_ndim = len(spec.shape)
            if out.ndim != expected_ndim:
                raise ExportError(
                    f"Output {i} ({spec.name}) has {out.ndim} dims but spec has {expected_ndim}"
                )
    
    def _verify_onnx_output(
        self,
        model: nn.Module,
        onnx_path: str,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Verify ONNX model output matches PyTorch.
        
        Args:
            model: Original PyTorch model
            onnx_path: Path to exported ONNX model
            num_samples: Number of random samples to test
            
        Returns:
            Dictionary with verification metrics
            
        Raises:
            VerificationError: If outputs differ beyond tolerance
        """
        # Load ONNX model
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_names = [inp.name for inp in sess.get_inputs()]
        output_names = [out.name for out in sess.get_outputs()]
        
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        
        max_abs_errors = []
        max_rel_errors = []
        mean_abs_errors = []
        
        for _ in range(num_samples):
            # Generate random input
            dummy_input = _create_dummy_input(
                self.config.input_specs,
                batch_size=np.random.randint(1, 10),
                device=device
            )
            
            # PyTorch forward
            model.eval()
            with torch.no_grad():
                if isinstance(dummy_input, tuple):
                    torch_output = model(*dummy_input)
                else:
                    torch_output = model(dummy_input)
            
            if not isinstance(torch_output, tuple):
                torch_output = (torch_output,)
            
            # Prepare ONNX input
            if isinstance(dummy_input, tuple):
                ort_inputs = {
                    name: inp.cpu().numpy() 
                    for name, inp in zip(input_names, dummy_input)
                }
            else:
                ort_inputs = {input_names[0]: dummy_input.cpu().numpy()}
            
            # ONNX forward
            ort_outputs = sess.run(output_names, ort_inputs)
            
            # Compare outputs
            for torch_out, ort_out in zip(torch_output, ort_outputs):
                torch_np = torch_out.cpu().numpy()
                
                abs_error = np.abs(torch_np - ort_out)
                max_abs_errors.append(np.max(abs_error))
                mean_abs_errors.append(np.mean(abs_error))
                
                # Relative error (avoid division by zero)
                denom = np.maximum(np.abs(torch_np), 1e-7)
                rel_error = abs_error / denom
                max_rel_errors.append(np.max(rel_error))
        
        metrics = {
            "max_abs_error": float(np.max(max_abs_errors)),
            "mean_abs_error": float(np.mean(mean_abs_errors)),
            "max_rel_error": float(np.max(max_rel_errors)),
            "num_samples": num_samples,
        }
        
        # Check against tolerances
        if metrics["max_abs_error"] > self.config.verification_atol:
            if metrics["max_rel_error"] > self.config.verification_rtol:
                raise VerificationError(
                    f"ONNX output differs from PyTorch: "
                    f"max_abs_error={metrics['max_abs_error']:.2e} > atol={self.config.verification_atol}, "
                    f"max_rel_error={metrics['max_rel_error']:.2e} > rtol={self.config.verification_rtol}"
                )
        
        return metrics
    
    def export(
        self,
        model: nn.Module,
        output_path: str,
        verify: bool = True,
        embed_metadata: bool = True,
    ) -> "ExportResult":
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            output_path: Path for output ONNX file
            verify: Whether to verify output matches PyTorch
            embed_metadata: Whether to embed export metadata in ONNX file
            
        Returns:
            ExportResult with export details
        """
        logger.info(f"Exporting {self.config.project_name} v{self.config.model_version} to ONNX")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate model
        self._validate_model(model)
        
        # Prepare model for export
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        
        # Create dummy input
        dummy_input = _create_dummy_input(self.config.input_specs, batch_size=1, device=device)
        
        # Build input/output names
        input_names = self.config.get_input_names()
        output_names = self.config.get_output_names()
        
        # Export to ONNX
        logger.info(f"Exporting with opset version {self.config.opset_version}")
        
        # Use temporary file first, then move
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                tmp_path,
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.config.dynamic_axes if self.config.dynamic_axes else None,
            )
            
            # Load and validate ONNX model
            onnx_model = onnx.load(tmp_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
            
            # Embed metadata
            if embed_metadata:
                metadata = self.config.to_metadata_dict()
                onnx_model = _embed_metadata(onnx_model, metadata)
            
            # Save final model
            onnx.save(onnx_model, str(output_path))
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
        
        # Get model size
        model_size_bytes = output_path.stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        logger.info(f"Exported model size: {model_size_mb:.2f} MB ({model_size_bytes:,} bytes)")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Verify output
        verification_metrics = None
        if verify:
            logger.info("Verifying ONNX model output...")
            try:
                verification_metrics = self._verify_onnx_output(
                    model, str(output_path), 
                    num_samples=self.config.verification_samples
                )
                logger.info(
                    f"Verification passed: max_abs_error={verification_metrics['max_abs_error']:.2e}, "
                    f"max_rel_error={verification_metrics['max_rel_error']:.2e}"
                )
            except VerificationError as e:
                logger.error(f"Verification failed: {e}")
                raise
        
        return ExportResult(
            output_path=str(output_path),
            model_size_bytes=model_size_bytes,
            model_size_mb=model_size_mb,
            param_count=param_count,
            opset_version=self.config.opset_version,
            input_names=input_names,
            output_names=output_names,
            verification_metrics=verification_metrics,
        )


class ExportResult:
    """Result of ONNX export operation."""
    
    def __init__(
        self,
        output_path: str,
        model_size_bytes: int,
        model_size_mb: float,
        param_count: int,
        opset_version: int,
        input_names: List[str],
        output_names: List[str],
        verification_metrics: Optional[Dict[str, float]] = None,
    ):
        self.output_path = output_path
        self.model_size_bytes = model_size_bytes
        self.model_size_mb = model_size_mb
        self.param_count = param_count
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names
        self.verification_metrics = verification_metrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": self.output_path,
            "model_size_bytes": self.model_size_bytes,
            "model_size_mb": self.model_size_mb,
            "param_count": self.param_count,
            "opset_version": self.opset_version,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "verification_metrics": self.verification_metrics,
        }
    
    def __repr__(self) -> str:
        return (
            f"ExportResult(path={self.output_path}, "
            f"size={self.model_size_mb:.2f}MB, "
            f"params={self.param_count:,})"
        )
