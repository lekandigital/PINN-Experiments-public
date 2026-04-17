"""
ONNX Export Engine

Exports PyTorch models to ONNX format with support for:
1. Direct export (standard tensor models like MLPs, SIRENs)
2. Flatten strategy (convert graph-structured GNNs to dense tensors)
3. Decoder-only export (export only non-graph portions)
4. TorchScript export (alternative to ONNX)

Also includes validation to ensure exported models match PyTorch outputs.

Extracted and generalized from Project 15 (PINN-Lite-Foil).
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Results from model export."""
    
    onnx_path: str | None
    torchscript_path: str | None
    
    # Export metadata
    strategy: str
    opset_version: int
    input_shape: list[int]
    output_shape: list[int]
    
    # Validation results
    validated: bool
    max_error: float
    mean_error: float
    
    # Model info
    file_size_bytes: int
    
    # Metadata embedded in model
    metadata: dict[str, Any]


class ONNXExporter:
    """
    Model-agnostic ONNX exporter.
    
    Supports multiple export strategies for different model architectures:
    - direct: Standard PyTorch → ONNX tracing (MLPs, SIRENs, CNNs)
    - flatten: Convert graph inputs to dense tensors (GNNs with fixed topology)
    - decoder_only: Export only the decoder portion (hybrid GNN+MLP models)
    - torchscript: Export as TorchScript instead of ONNX
    
    Usage:
        exporter = ONNXExporter(model, config)
        result = exporter.export(output_dir)
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: list[int] | None = None,
        output_shape: list[int] | None = None,
        strategy: str = "direct",
        opset_version: int = 17,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        max_nodes: int | None = None,
        validation_tolerance: float = 1e-5,
        num_validation_samples: int = 100,
    ):
        """
        Initialize the exporter.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (with batch dim)
            output_shape: Output tensor shape (with batch dim)
            strategy: Export strategy ("direct", "flatten", "decoder_only", "torchscript")
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            max_nodes: Maximum nodes for flatten strategy
            validation_tolerance: Maximum allowed error in validation
            num_validation_samples: Number of samples for validation
        """
        self.model = model
        self.model.eval()
        
        self.input_shape = input_shape or [1, 3]  # Default: batch of 3D vectors
        self.output_shape = output_shape
        self.strategy = strategy
        self.opset_version = opset_version
        self.max_nodes = max_nodes
        self.validation_tolerance = validation_tolerance
        self.num_validation_samples = num_validation_samples
        
        # Default dynamic axes: batch dimension is dynamic
        self.dynamic_axes = dynamic_axes or {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        # Metadata to embed in the model
        self.metadata: dict[str, Any] = {}
    
    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Set metadata to embed in the exported model."""
        self.metadata = metadata
    
    def export(
        self,
        output_dir: str | Path,
        model_name: str = "model",
        also_export_torchscript: bool = False,
    ) -> ExportResult:
        """
        Export the model using the configured strategy.
        
        Args:
            output_dir: Directory to save exported files
            model_name: Base name for output files
            also_export_torchscript: Also export TorchScript version
            
        Returns:
            ExportResult with paths and validation info
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        onnx_path = None
        torchscript_path = None
        
        if self.strategy == "direct":
            onnx_path = self._export_direct(output_dir, model_name)
        elif self.strategy == "flatten":
            onnx_path = self._export_flatten(output_dir, model_name)
        elif self.strategy == "decoder_only":
            onnx_path = self._export_decoder_only(output_dir, model_name)
        elif self.strategy == "torchscript":
            torchscript_path = self._export_torchscript(output_dir, model_name)
        else:
            raise ValueError(f"Unknown export strategy: {self.strategy}")
        
        # Also export TorchScript if requested
        if also_export_torchscript and self.strategy != "torchscript":
            torchscript_path = self._export_torchscript(output_dir, model_name)
        
        # Validate ONNX export
        validated = False
        max_error = float('inf')
        mean_error = float('inf')
        
        if onnx_path:
            validated, max_error, mean_error = self._validate_onnx(onnx_path)
        
        # Get file size
        file_size = 0
        if onnx_path:
            file_size = Path(onnx_path).stat().st_size
        elif torchscript_path:
            file_size = Path(torchscript_path).stat().st_size
        
        # Infer output shape if not provided
        output_shape = self.output_shape
        if output_shape is None:
            with torch.no_grad():
                example = torch.randn(*self.input_shape)
                output = self.model(example)
                output_shape = list(output.shape)
        
        # Save metadata
        metadata = {
            **self.metadata,
            'export_date': datetime.now().isoformat(),
            'export_strategy': self.strategy,
            'opset_version': self.opset_version,
            'input_shape': self.input_shape,
            'output_shape': output_shape,
            'validated': validated,
            'max_error': max_error,
            'mean_error': mean_error,
        }
        
        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return ExportResult(
            onnx_path=str(onnx_path) if onnx_path else None,
            torchscript_path=str(torchscript_path) if torchscript_path else None,
            strategy=self.strategy,
            opset_version=self.opset_version,
            input_shape=self.input_shape,
            output_shape=output_shape,
            validated=validated,
            max_error=max_error,
            mean_error=mean_error,
            file_size_bytes=file_size,
            metadata=metadata,
        )
    
    def _export_direct(self, output_dir: Path, model_name: str) -> Path:
        """
        Direct ONNX export via torch.onnx.export.
        
        Works for standard tensor-in, tensor-out models.
        """
        onnx_path = output_dir / f"{model_name}.onnx"
        
        # Create example input
        example_input = torch.randn(*self.input_shape)
        
        logger.info(f"Exporting model to ONNX (direct strategy)")
        logger.info(f"  Input shape: {self.input_shape}")
        logger.info(f"  Opset version: {self.opset_version}")
        
        torch.onnx.export(
            self.model,
            example_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=self.dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        
        # Optimize the exported model
        self._optimize_onnx(onnx_path)
        
        logger.info(f"Exported to {onnx_path}")
        return onnx_path
    
    def _export_flatten(self, output_dir: Path, model_name: str) -> Path:
        """
        Export GNN by flattening graph to dense tensors.
        
        Converts variable-sized graph inputs to fixed-size dense tensors
        with padding. Suitable for GNNs with bounded graph size.
        """
        if self.max_nodes is None:
            raise ValueError("max_nodes must be specified for flatten strategy")
        
        onnx_path = output_dir / f"{model_name}.onnx"
        
        logger.info(f"Exporting model to ONNX (flatten strategy)")
        logger.info(f"  Max nodes: {self.max_nodes}")
        
        # Create wrapper that accepts flattened inputs
        wrapper = GraphFlattenWrapper(self.model, self.max_nodes)
        
        # Example input: flattened graph
        # Assuming input is [node_features, adjacency_matrix]
        # This is a simplified version - real implementation depends on the GNN
        example_node_features = torch.randn(1, self.max_nodes, self.input_shape[-1])
        example_adjacency = torch.zeros(1, self.max_nodes, self.max_nodes)
        
        # Create dummy adjacency (random sparse)
        for i in range(self.max_nodes):
            for j in range(min(5, self.max_nodes)):  # ~5 neighbors per node
                example_adjacency[0, i, (i + j + 1) % self.max_nodes] = 1.0
        
        example_input = (example_node_features, example_adjacency)
        
        torch.onnx.export(
            wrapper,
            example_input,
            onnx_path,
            input_names=['node_features', 'adjacency'],
            output_names=['output'],
            dynamic_axes={
                'node_features': {0: 'batch_size'},
                'adjacency': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
        
        self._optimize_onnx(onnx_path)
        
        logger.info(f"Exported to {onnx_path}")
        return onnx_path
    
    def _export_decoder_only(self, output_dir: Path, model_name: str) -> Path:
        """
        Export only the decoder portion of a model.
        
        For hybrid models (GNN encoder + MLP decoder), exports just the decoder
        which takes embeddings and produces outputs. The encoder stays in PyTorch.
        """
        onnx_path = output_dir / f"{model_name}_decoder.onnx"
        
        logger.info(f"Exporting model decoder to ONNX (decoder_only strategy)")
        
        # Try to find a decoder submodule
        decoder = None
        for name, module in self.model.named_modules():
            if 'decoder' in name.lower() or 'output' in name.lower() or 'mlp' in name.lower():
                decoder = module
                break
        
        if decoder is None:
            # Assume the whole model can be treated as a decoder
            logger.warning("No decoder found, exporting full model")
            decoder = self.model
        
        # Infer decoder input shape by running a forward pass
        with torch.no_grad():
            # This is a heuristic - may need adjustment per model
            example = torch.randn(*self.input_shape)
            
            # Try to get intermediate output
            if hasattr(self.model, 'encode'):
                embedding = self.model.encode(example)
                decoder_input_shape = list(embedding.shape)
            elif hasattr(self.model, 'encoder'):
                embedding = self.model.encoder(example)
                decoder_input_shape = list(embedding.shape)
            else:
                # Use original input shape
                decoder_input_shape = self.input_shape
        
        example_decoder_input = torch.randn(*decoder_input_shape)
        
        torch.onnx.export(
            decoder,
            example_decoder_input,
            onnx_path,
            input_names=['embedding'],
            output_names=['output'],
            dynamic_axes={
                'embedding': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
        
        self._optimize_onnx(onnx_path)
        
        logger.info(f"Exported decoder to {onnx_path}")
        return onnx_path
    
    def _export_torchscript(self, output_dir: Path, model_name: str) -> Path:
        """
        Export model as TorchScript.
        
        Better for models with dynamic control flow that ONNX can't handle.
        Compatible with PyTorch Mobile and LibTorch (C++).
        """
        ts_path = output_dir / f"{model_name}.pt"
        
        logger.info(f"Exporting model to TorchScript")
        
        example_input = torch.randn(*self.input_shape)
        
        try:
            # Try scripting first (handles dynamic control flow)
            scripted = torch.jit.script(self.model)
        except Exception as e:
            logger.warning(f"Script failed ({e}), falling back to trace")
            # Fall back to tracing
            scripted = torch.jit.trace(self.model, example_input)
        
        # Optimize for mobile (optional)
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            scripted = optimize_for_mobile(scripted)
            logger.info("Applied mobile optimizations")
        except ImportError:
            pass
        
        scripted.save(ts_path)
        
        logger.info(f"Exported to {ts_path}")
        return ts_path
    
    def _optimize_onnx(self, onnx_path: Path) -> None:
        """Apply ONNX graph optimizations."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            model = onnx.load(onnx_path)
            
            # Basic optimizations
            optimized = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # Generic optimization
                opt_level=1,
            )
            optimized.save_model_to_file(str(onnx_path))
            
            logger.debug("Applied ONNX optimizations")
        except ImportError:
            logger.debug("onnxruntime.transformers not available, skipping optimization")
        except Exception as e:
            logger.debug(f"ONNX optimization failed: {e}")
        
        # Try onnx-simplifier
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(onnx_path)
            simplified, check = simplify(model)
            
            if check:
                onnx.save(simplified, onnx_path)
                logger.debug("Applied onnx-simplifier")
        except ImportError:
            logger.debug("onnxsim not available, skipping simplification")
        except Exception as e:
            logger.debug(f"ONNX simplification failed: {e}")
    
    def _validate_onnx(self, onnx_path: Path) -> tuple[bool, float, float]:
        """
        Validate ONNX export by comparing outputs.
        
        Returns:
            (validated, max_error, mean_error)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")
            return False, float('inf'), float('inf')
        
        logger.info("Validating ONNX export...")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        
        input_name = session.get_inputs()[0].name
        
        errors = []
        
        for _ in range(self.num_validation_samples):
            # Generate random input
            test_input = torch.randn(*self.input_shape)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = self.model(test_input).numpy()
            
            # ONNX inference
            onnx_output = session.run(
                None, 
                {input_name: test_input.numpy()}
            )[0]
            
            # Compute error
            error = np.abs(pytorch_output - onnx_output)
            errors.append(error)
        
        errors = np.concatenate([e.flatten() for e in errors])
        max_error = float(np.max(errors))
        mean_error = float(np.mean(errors))
        
        validated = max_error < self.validation_tolerance
        
        if validated:
            logger.info(f"Validation passed (max error: {max_error:.2e})")
        else:
            logger.warning(
                f"Validation failed: max error {max_error:.2e} > tolerance {self.validation_tolerance:.2e}"
            )
        
        return validated, max_error, mean_error


class GraphFlattenWrapper(nn.Module):
    """
    Wrapper that converts flattened graph inputs to PyG format.
    
    For ONNX export of GNN models that expect graph-structured input.
    """
    
    def __init__(self, model: nn.Module, max_nodes: int):
        super().__init__()
        self.model = model
        self.max_nodes = max_nodes
    
    def forward(
        self, 
        node_features: torch.Tensor, 
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flattened inputs to graph format and run model.
        
        Args:
            node_features: [batch, max_nodes, feature_dim]
            adjacency: [batch, max_nodes, max_nodes]
            
        Returns:
            Model output
        """
        batch_size = node_features.shape[0]
        outputs = []
        
        for b in range(batch_size):
            # Extract non-padded nodes (assuming zero padding)
            nodes = node_features[b]
            adj = adjacency[b]
            
            # Convert adjacency to edge_index
            edge_index = adj.nonzero(as_tuple=False).t()
            
            # Run model
            # This is simplified - actual implementation depends on the model
            if hasattr(self.model, 'forward_dense'):
                output = self.model.forward_dense(nodes, adj)
            else:
                output = self.model(nodes)
            
            outputs.append(output)
        
        return torch.stack(outputs)


def export_model(
    model: nn.Module,
    output_dir: str | Path,
    model_name: str = "model",
    strategy: str = "direct",
    input_shape: list[int] | None = None,
    opset_version: int = 17,
    metadata: dict[str, Any] | None = None,
    validate: bool = True,
    also_export_torchscript: bool = False,
    **kwargs
) -> ExportResult:
    """
    Convenience function to export a model.
    
    Args:
        model: PyTorch model to export
        output_dir: Output directory
        model_name: Base name for output files
        strategy: Export strategy
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        metadata: Metadata to embed
        validate: Whether to validate export
        also_export_torchscript: Also export TorchScript
        **kwargs: Additional arguments for ONNXExporter
        
    Returns:
        ExportResult
    """
    exporter = ONNXExporter(
        model=model,
        input_shape=input_shape,
        strategy=strategy,
        opset_version=opset_version,
        validation_tolerance=1e-5 if validate else float('inf'),
        **kwargs
    )
    
    if metadata:
        exporter.set_metadata(metadata)
    
    return exporter.export(
        output_dir=output_dir,
        model_name=model_name,
        also_export_torchscript=also_export_torchscript,
    )
