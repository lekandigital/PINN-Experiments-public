"""
Platform-Specific Optimizers.

Provides optimization functions for different deployment targets:
- Browser (ONNX Runtime Web)
- Mobile (ONNX Runtime Mobile / .ort format)
- Edge - TensorRT (NVIDIA)
- Edge - OpenVINO (Intel)
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import onnx
    from onnx import helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from .config import ExportConfig, TargetPlatform

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of platform optimization."""
    platform: TargetPlatform
    output_path: str
    original_size_bytes: int
    optimized_size_bytes: int
    success: bool
    warnings: List[str]
    unsupported_ops: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "output_path": self.output_path,
            "original_size_bytes": self.original_size_bytes,
            "optimized_size_bytes": self.optimized_size_bytes,
            "success": self.success,
            "warnings": self.warnings,
            "unsupported_ops": self.unsupported_ops,
        }


# ONNX operators supported by ONNX Runtime Web (WASM backend)
# This is a conservative list - some ops may work in newer versions
ONNX_WEB_SUPPORTED_OPS: Set[str] = {
    # Basic math
    "Abs", "Add", "Sub", "Mul", "Div", "Neg", "Pow", "Sqrt", "Exp", "Log",
    "Sin", "Cos", "Tan", "Sinh", "Cosh", "Tanh", "Asin", "Acos", "Atan",
    "Sigmoid", "Relu", "LeakyRelu", "Elu", "Selu", "Softmax", "LogSoftmax",
    "Softplus", "Softsign", "HardSigmoid", "HardSwish", "Gelu", "Erf",
    # Reduction
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
    # Matrix ops
    "MatMul", "Gemm", "Conv", "ConvTranspose", "BatchNormalization",
    "InstanceNormalization", "LayerNormalization", "GroupNormalization",
    # Pooling
    "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
    # Shape ops
    "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Concat",
    "Split", "Slice", "Gather", "GatherElements", "GatherND", "Scatter",
    "ScatterElements", "ScatterND", "Tile", "Expand", "Shape", "Size",
    # Element-wise
    "Clip", "Cast", "Floor", "Ceil", "Round", "Sign", "Reciprocal",
    "Where", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
    "And", "Or", "Not", "Xor", "Min", "Max", "Mean", "Sum",
    # RNN (limited support)
    "LSTM", "GRU", "RNN",
    # Misc
    "Dropout", "Identity", "Constant", "ConstantOfShape", "Range",
    "Pad", "Resize", "Upsample",
}

# WebNN has a more limited op set
WEBNN_SUPPORTED_OPS: Set[str] = {
    "Add", "Sub", "Mul", "Div", "Pow", "Abs", "Ceil", "Floor", "Neg",
    "Exp", "Log", "Sqrt", "Sigmoid", "Tanh", "Relu", "LeakyRelu", "Elu",
    "Softmax", "MatMul", "Gemm", "Conv", "ConvTranspose",
    "BatchNormalization", "LayerNormalization",
    "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
    "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Concat",
    "Split", "Slice", "Gather", "Pad", "Resize",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
    "Cast", "Clip", "Where", "Constant", "Identity",
}


def _get_model_ops(model_path: str) -> Set[str]:
    """Get set of operators used in ONNX model."""
    model = onnx.load(model_path)
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    return ops


def _check_op_compatibility(
    model_path: str, 
    supported_ops: Set[str]
) -> tuple[bool, List[str]]:
    """
    Check if all model ops are supported.
    
    Returns:
        Tuple of (all_supported, unsupported_ops_list)
    """
    model_ops = _get_model_ops(model_path)
    unsupported = model_ops - supported_ops
    return len(unsupported) == 0, list(unsupported)


def optimize_for_browser(
    input_path: str,
    output_path: str,
    config: ExportConfig,
    check_webnn: bool = False,
) -> OptimizationResult:
    """
    Optimize ONNX model for browser deployment (ONNX Runtime Web).
    
    Applies graph optimizations and checks operator compatibility with
    ONNX Runtime Web (WASM/WebGL/WebGPU backends) and optionally WebNN.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for optimized output
        config: Export configuration
        check_webnn: Also check WebNN compatibility
        
    Returns:
        OptimizationResult with compatibility info
    """
    if not ORT_AVAILABLE:
        raise ImportError("onnxruntime not available")
    
    logger.info(f"Optimizing {input_path} for browser deployment")
    
    original_size = Path(input_path).stat().st_size
    warnings = []
    unsupported_ops = []
    
    # Check op compatibility
    all_supported, unsupported = _check_op_compatibility(input_path, ONNX_WEB_SUPPORTED_OPS)
    if not all_supported:
        warnings.append(f"Potentially unsupported ops for ONNX Runtime Web: {unsupported}")
        unsupported_ops.extend(unsupported)
        logger.warning(f"Model uses ops that may not be supported: {unsupported}")
    
    if check_webnn:
        webnn_supported, webnn_unsupported = _check_op_compatibility(input_path, WEBNN_SUPPORTED_OPS)
        if not webnn_supported:
            warnings.append(f"Unsupported ops for WebNN: {webnn_unsupported}")
            logger.warning(f"Model uses ops not supported by WebNN: {webnn_unsupported}")
    
    # Run graph optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path
    
    # Create session to trigger optimization and save
    try:
        sess = ort.InferenceSession(
            input_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        del sess
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        # Fall back to copying original
        shutil.copy(input_path, output_path)
        warnings.append(f"Graph optimization failed: {e}")
    
    # Strip unnecessary metadata to reduce size
    try:
        model = onnx.load(output_path)
        # Remove training-related info
        model.graph.ClearField('initializer')
        # Keep our metadata, remove others
        # Actually, let's keep it as-is after optimization
        onnx.save(model, output_path)
    except Exception as e:
        logger.warning(f"Post-optimization cleanup failed: {e}")
    
    # Reload to get final model
    model = onnx.load(output_path)
    onnx.save(model, output_path)
    
    optimized_size = Path(output_path).stat().st_size
    
    logger.info(
        f"Browser optimization complete: {original_size / 1024:.1f} KB -> "
        f"{optimized_size / 1024:.1f} KB"
    )
    
    return OptimizationResult(
        platform=TargetPlatform.BROWSER,
        output_path=output_path,
        original_size_bytes=original_size,
        optimized_size_bytes=optimized_size,
        success=len(unsupported_ops) == 0,
        warnings=warnings,
        unsupported_ops=unsupported_ops,
    )


def optimize_for_mobile(
    input_path: str,
    output_dir: str,
    config: ExportConfig,
) -> OptimizationResult:
    """
    Optimize ONNX model for mobile deployment (ONNX Runtime Mobile).
    
    Converts to ORT format (.ort) which is optimized for mobile inference.
    
    Args:
        input_path: Path to input ONNX model
        output_dir: Directory for output files
        config: Export configuration
        
    Returns:
        OptimizationResult
    """
    logger.info(f"Optimizing {input_path} for mobile deployment")
    
    original_size = Path(input_path).stat().st_size
    warnings = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path
    model_name = Path(input_path).stem
    ort_path = output_dir / f"{model_name}.ort"
    
    # Try using ONNX Runtime tools for ORT format conversion
    try:
        # Method 1: Use python API if available
        result = subprocess.run(
            [
                "python", "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
                "--optimization_style", "Fixed",
                input_path,
            ],
            capture_output=True,
            text=True,
            cwd=str(output_dir),
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ORT conversion failed: {result.stderr}")
        
        # Find generated .ort file
        generated_ort = Path(input_path).with_suffix('.ort')
        if generated_ort.exists():
            shutil.move(str(generated_ort), str(ort_path))
        
    except Exception as e:
        logger.warning(f"ORT format conversion failed: {e}")
        warnings.append(f"ORT conversion failed, using optimized ONNX: {e}")
        
        # Fall back to optimized ONNX
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = str(ort_path.with_suffix('.onnx'))
        
        sess = ort.InferenceSession(
            input_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        del sess
        
        # Rename to .ort for consistency
        ort_path = ort_path.with_suffix('.onnx')
    
    if ort_path.exists():
        optimized_size = ort_path.stat().st_size
    else:
        optimized_size = original_size
        warnings.append("Output file not found")
    
    logger.info(
        f"Mobile optimization complete: {original_size / 1024:.1f} KB -> "
        f"{optimized_size / 1024:.1f} KB"
    )
    
    return OptimizationResult(
        platform=TargetPlatform.MOBILE,
        output_path=str(ort_path),
        original_size_bytes=original_size,
        optimized_size_bytes=optimized_size,
        success=ort_path.exists(),
        warnings=warnings,
        unsupported_ops=[],
    )


def optimize_for_tensorrt(
    input_path: str,
    output_path: str,
    config: ExportConfig,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 1024,
) -> OptimizationResult:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for TensorRT engine output
        config: Export configuration  
        fp16: Enable FP16 mode (nearly free on NVIDIA GPUs)
        int8: Enable INT8 mode (requires calibration)
        workspace_mb: TensorRT workspace size in MB
        
    Returns:
        OptimizationResult
    """
    logger.info(f"Converting {input_path} to TensorRT engine")
    
    original_size = Path(input_path).stat().st_size
    warnings = []
    
    # Check for trtexec
    trtexec_path = shutil.which("trtexec")
    
    if trtexec_path:
        # Use trtexec CLI
        cmd = [
            trtexec_path,
            f"--onnx={input_path}",
            f"--saveEngine={output_path}",
            f"--workspace={workspace_mb}",
        ]
        
        if fp16:
            cmd.append("--fp16")
        if int8:
            cmd.append("--int8")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"trtexec failed: {result.stderr}")
                warnings.append(f"trtexec failed: {result.stderr[:500]}")
            else:
                logger.info("TensorRT engine built successfully")
                
        except subprocess.TimeoutExpired:
            warnings.append("TensorRT conversion timed out")
        except Exception as e:
            warnings.append(f"TensorRT conversion error: {e}")
    else:
        # Try Python API
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:
                
                # Parse ONNX
                with open(input_path, 'rb') as f:
                    if not parser.parse(f.read()):
                        for i in range(parser.num_errors):
                            logger.error(f"TensorRT parser error: {parser.get_error(i)}")
                        raise RuntimeError("Failed to parse ONNX model")
                
                # Configure builder
                config_trt = builder.create_builder_config()
                config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
                
                if fp16:
                    config_trt.set_flag(trt.BuilderFlag.FP16)
                if int8:
                    config_trt.set_flag(trt.BuilderFlag.INT8)
                
                # Build engine
                engine = builder.build_serialized_network(network, config_trt)
                
                if engine:
                    with open(output_path, 'wb') as f:
                        f.write(engine)
                    logger.info("TensorRT engine built successfully")
                else:
                    warnings.append("Failed to build TensorRT engine")
                    
        except ImportError:
            warnings.append("TensorRT not available - skipping")
            logger.warning("TensorRT not installed, skipping TensorRT optimization")
        except Exception as e:
            warnings.append(f"TensorRT conversion failed: {e}")
            logger.error(f"TensorRT conversion failed: {e}")
    
    if Path(output_path).exists():
        optimized_size = Path(output_path).stat().st_size
        success = True
    else:
        optimized_size = 0
        success = False
    
    return OptimizationResult(
        platform=TargetPlatform.EDGE_TENSORRT,
        output_path=output_path,
        original_size_bytes=original_size,
        optimized_size_bytes=optimized_size,
        success=success,
        warnings=warnings,
        unsupported_ops=[],
    )


def optimize_for_openvino(
    input_path: str,
    output_dir: str,
    config: ExportConfig,
    compress_to_fp16: bool = True,
) -> OptimizationResult:
    """
    Convert ONNX model to OpenVINO IR format.
    
    Args:
        input_path: Path to input ONNX model
        output_dir: Directory for output files (.xml, .bin)
        config: Export configuration
        compress_to_fp16: Compress weights to FP16
        
    Returns:
        OptimizationResult
    """
    logger.info(f"Converting {input_path} to OpenVINO IR format")
    
    original_size = Path(input_path).stat().st_size
    warnings = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(input_path).stem
    xml_path = output_dir / f"{model_name}.xml"
    bin_path = output_dir / f"{model_name}.bin"
    
    try:
        from openvino.tools import mo
        from openvino.runtime import Core
        
        # Convert using Model Optimizer
        mo_args = [
            "--input_model", input_path,
            "--output_dir", str(output_dir),
            "--model_name", model_name,
        ]
        
        if compress_to_fp16:
            mo_args.extend(["--compress_to_fp16"])
        
        # Run model optimizer
        result = subprocess.run(
            ["mo", *mo_args],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            # Try Python API
            from openvino import convert_model, save_model
            
            ov_model = convert_model(input_path)
            save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)
        
        logger.info("OpenVINO IR conversion successful")
        
    except ImportError:
        warnings.append("OpenVINO not available - skipping")
        logger.warning("OpenVINO not installed, skipping OpenVINO optimization")
    except Exception as e:
        warnings.append(f"OpenVINO conversion failed: {e}")
        logger.error(f"OpenVINO conversion failed: {e}")
    
    if xml_path.exists() and bin_path.exists():
        optimized_size = xml_path.stat().st_size + bin_path.stat().st_size
        success = True
        output_path = str(xml_path)
    else:
        optimized_size = 0
        success = False
        output_path = str(xml_path)
    
    return OptimizationResult(
        platform=TargetPlatform.EDGE_OPENVINO,
        output_path=output_path,
        original_size_bytes=original_size,
        optimized_size_bytes=optimized_size,
        success=success,
        warnings=warnings,
        unsupported_ops=[],
    )


def optimize_for_platform(
    input_path: str,
    output_path: str,
    platform: TargetPlatform,
    config: ExportConfig,
) -> OptimizationResult:
    """
    Optimize model for specified target platform.
    
    This is the main entry point for platform-specific optimization.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for optimized output
        platform: Target platform
        config: Export configuration
        
    Returns:
        OptimizationResult
    """
    output_dir = Path(output_path).parent
    
    if platform == TargetPlatform.BROWSER:
        return optimize_for_browser(input_path, output_path, config)
    elif platform == TargetPlatform.MOBILE:
        return optimize_for_mobile(input_path, str(output_dir), config)
    elif platform == TargetPlatform.EDGE_TENSORRT:
        return optimize_for_tensorrt(input_path, output_path, config)
    elif platform == TargetPlatform.EDGE_OPENVINO:
        return optimize_for_openvino(input_path, str(output_dir), config)
    elif platform == TargetPlatform.DESKTOP:
        # Desktop just uses standard optimized ONNX
        return optimize_for_browser(input_path, output_path, config)
    else:
        raise ValueError(f"Unknown platform: {platform}")
