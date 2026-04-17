"""
Model Quantization Module.

Provides INT8 and FP16 quantization for ONNX models with accuracy verification
and size/performance reporting.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        CalibrationDataReader,
        QuantType,
        QuantFormat,
    )
    ORT_QUANTIZATION_AVAILABLE = True
except ImportError:
    ORT_QUANTIZATION_AVAILABLE = False

try:
    from onnxconverter_common import float16
    FP16_AVAILABLE = True
except ImportError:
    FP16_AVAILABLE = False

from .config import ExportConfig, QuantizationConfig, QuantizationMethod

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Result of quantization operation."""
    output_path: str
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    quantization_method: str
    accuracy_metrics: Dict[str, float]
    passed_accuracy_check: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": self.output_path,
            "original_size_bytes": self.original_size_bytes,
            "quantized_size_bytes": self.quantized_size_bytes,
            "compression_ratio": self.compression_ratio,
            "quantization_method": self.quantization_method,
            "accuracy_metrics": self.accuracy_metrics,
            "passed_accuracy_check": self.passed_accuracy_check,
        }


class NumpyCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for static quantization.
    
    Reads calibration data from numpy files or generates synthetic data
    based on input specifications.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        input_specs: Optional[List[Dict]] = None,
        num_samples: int = 100,
        input_names: Optional[List[str]] = None,
    ):
        """
        Initialize calibration data reader.
        
        Args:
            data_path: Path to directory containing .npy files
            input_specs: Input tensor specifications (used for synthetic data)
            num_samples: Number of calibration samples
            input_names: Names of input tensors
        """
        self.data_path = data_path
        self.input_specs = input_specs or []
        self.num_samples = num_samples
        self.input_names = input_names or []
        self.current_index = 0
        
        # Load or generate data
        self.data = self._load_or_generate_data()
    
    def _load_or_generate_data(self) -> List[Dict[str, np.ndarray]]:
        """Load calibration data from files or generate synthetic data."""
        data = []
        
        if self.data_path and Path(self.data_path).exists():
            # Load from files
            data_dir = Path(self.data_path)
            
            # Look for numbered sample files
            sample_idx = 0
            while True:
                sample = {}
                found_all = True
                
                for name in self.input_names:
                    file_path = data_dir / f"{name}_{sample_idx}.npy"
                    if file_path.exists():
                        sample[name] = np.load(file_path)
                    else:
                        found_all = False
                        break
                
                if found_all and sample:
                    data.append(sample)
                    sample_idx += 1
                else:
                    break
                
                if len(data) >= self.num_samples:
                    break
            
            logger.info(f"Loaded {len(data)} calibration samples from {data_dir}")
        
        # Generate synthetic data if needed
        if len(data) < self.num_samples and self.input_specs:
            logger.info(f"Generating {self.num_samples - len(data)} synthetic calibration samples")
            
            for _ in range(self.num_samples - len(data)):
                sample = {}
                for spec in self.input_specs:
                    name = spec.get("name", f"input_{len(sample)}")
                    shape = spec.get("shape", [1, 4])
                    dtype = spec.get("dtype", "float32")
                    
                    # Replace dynamic dims
                    concrete_shape = [8 if d == -1 else d for d in shape]
                    
                    np_dtype = getattr(np, dtype, np.float32)
                    
                    if np.issubdtype(np_dtype, np.floating):
                        min_val = spec.get("min_value", -1.0)
                        max_val = spec.get("max_value", 1.0)
                        arr = np.random.uniform(min_val, max_val, concrete_shape).astype(np_dtype)
                    else:
                        arr = np.random.randint(0, 10, concrete_shape).astype(np_dtype)
                    
                    sample[name] = arr
                
                data.append(sample)
        
        return data
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next calibration sample."""
        if self.current_index >= len(self.data):
            return None
        
        sample = self.data[self.current_index]
        self.current_index += 1
        return sample
    
    def rewind(self):
        """Reset to beginning of data."""
        self.current_index = 0


def _compare_models(
    original_path: str,
    quantized_path: str,
    input_specs: List[Dict],
    num_samples: int = 20,
) -> Dict[str, float]:
    """
    Compare outputs of original and quantized models.
    
    Args:
        original_path: Path to original ONNX model
        quantized_path: Path to quantized ONNX model
        input_specs: Input tensor specifications
        num_samples: Number of comparison samples
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Load both models
    orig_sess = ort.InferenceSession(original_path, providers=['CPUExecutionProvider'])
    quant_sess = ort.InferenceSession(quantized_path, providers=['CPUExecutionProvider'])
    
    orig_inputs = [inp.name for inp in orig_sess.get_inputs()]
    orig_outputs = [out.name for out in orig_sess.get_outputs()]
    quant_outputs = [out.name for out in quant_sess.get_outputs()]
    
    max_abs_errors = []
    max_rel_errors = []
    mean_abs_errors = []
    
    for _ in range(num_samples):
        # Generate random input
        ort_input = {}
        for spec in input_specs:
            name = spec.get("name")
            if name not in orig_inputs:
                continue
                
            shape = spec.get("shape", [1, 4])
            concrete_shape = [np.random.randint(1, 10) if d == -1 else d for d in shape]
            
            dtype = spec.get("dtype", "float32")
            np_dtype = getattr(np, dtype, np.float32)
            
            if np.issubdtype(np_dtype, np.floating):
                arr = np.random.randn(*concrete_shape).astype(np_dtype)
            else:
                arr = np.random.randint(0, 10, concrete_shape).astype(np_dtype)
            
            ort_input[name] = arr
        
        # Run both models
        orig_output = orig_sess.run(orig_outputs, ort_input)
        quant_output = quant_sess.run(quant_outputs, ort_input)
        
        # Compare outputs
        for orig_out, quant_out in zip(orig_output, quant_output):
            abs_error = np.abs(orig_out - quant_out)
            max_abs_errors.append(np.max(abs_error))
            mean_abs_errors.append(np.mean(abs_error))
            
            # Relative error
            denom = np.maximum(np.abs(orig_out), 1e-7)
            rel_error = abs_error / denom
            max_rel_errors.append(np.max(rel_error))
    
    return {
        "max_abs_error": float(np.max(max_abs_errors)),
        "mean_abs_error": float(np.mean(mean_abs_errors)),
        "max_rel_error": float(np.max(max_rel_errors)),
        "mean_rel_error": float(np.mean(max_rel_errors)),
        "num_samples": num_samples,
    }


def quantize_dynamic_int8(
    input_path: str,
    output_path: str,
    config: QuantizationConfig,
) -> QuantizationResult:
    """
    Apply dynamic INT8 quantization to ONNX model.
    
    Dynamic quantization quantizes weights to INT8 statically, while
    activations are quantized dynamically at runtime. This requires
    no calibration data and works well for most models.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for quantized output
        config: Quantization configuration
        
    Returns:
        QuantizationResult with metrics
    """
    if not ORT_QUANTIZATION_AVAILABLE:
        raise ImportError("onnxruntime quantization not available")
    
    logger.info(f"Applying dynamic INT8 quantization to {input_path}")
    
    # Get original size
    original_size = Path(input_path).stat().st_size
    
    # Quantize
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=config.per_channel,
        reduce_range=config.reduce_range,
    )
    
    # Get quantized size
    quantized_size = Path(output_path).stat().st_size
    compression_ratio = original_size / quantized_size
    
    logger.info(
        f"Quantization complete: {original_size / 1024:.1f} KB -> {quantized_size / 1024:.1f} KB "
        f"({compression_ratio:.2f}x compression)"
    )
    
    return QuantizationResult(
        output_path=output_path,
        original_size_bytes=original_size,
        quantized_size_bytes=quantized_size,
        compression_ratio=compression_ratio,
        quantization_method="dynamic_int8",
        accuracy_metrics={},  # Will be filled by caller
        passed_accuracy_check=True,
    )


def quantize_static_int8(
    input_path: str,
    output_path: str,
    config: QuantizationConfig,
    input_specs: List[Dict],
    input_names: List[str],
) -> QuantizationResult:
    """
    Apply static INT8 quantization to ONNX model.
    
    Static quantization quantizes both weights and activations to INT8.
    This requires calibration data to determine activation ranges.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for quantized output
        config: Quantization configuration
        input_specs: Input tensor specifications
        input_names: Names of input tensors
        
    Returns:
        QuantizationResult with metrics
    """
    if not ORT_QUANTIZATION_AVAILABLE:
        raise ImportError("onnxruntime quantization not available")
    
    logger.info(f"Applying static INT8 quantization to {input_path}")
    
    # Get original size
    original_size = Path(input_path).stat().st_size
    
    # Create calibration data reader
    calibration_reader = NumpyCalibrationDataReader(
        data_path=config.calibration_data_path,
        input_specs=input_specs,
        num_samples=config.calibration_num_samples,
        input_names=input_names,
    )
    
    # Quantize
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=config.per_channel,
        reduce_range=config.reduce_range,
    )
    
    # Get quantized size
    quantized_size = Path(output_path).stat().st_size
    compression_ratio = original_size / quantized_size
    
    logger.info(
        f"Quantization complete: {original_size / 1024:.1f} KB -> {quantized_size / 1024:.1f} KB "
        f"({compression_ratio:.2f}x compression)"
    )
    
    return QuantizationResult(
        output_path=output_path,
        original_size_bytes=original_size,
        quantized_size_bytes=quantized_size,
        compression_ratio=compression_ratio,
        quantization_method="static_int8",
        accuracy_metrics={},
        passed_accuracy_check=True,
    )


def quantize_fp16(
    input_path: str,
    output_path: str,
) -> QuantizationResult:
    """
    Convert ONNX model from FP32 to FP16.
    
    FP16 conversion reduces model size by ~50% with minimal accuracy loss.
    This is especially useful for GPU targets (WebGPU, TensorRT).
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for FP16 output
        
    Returns:
        QuantizationResult with metrics
    """
    if not FP16_AVAILABLE:
        raise ImportError(
            "onnxconverter-common not available. Install with: pip install onnxconverter-common"
        )
    
    logger.info(f"Converting {input_path} to FP16")
    
    # Get original size
    original_size = Path(input_path).stat().st_size
    
    # Load model
    model = onnx.load(input_path)
    
    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep input/output as FP32 for compatibility
    )
    
    # Save
    onnx.save(model_fp16, output_path)
    
    # Get new size
    quantized_size = Path(output_path).stat().st_size
    compression_ratio = original_size / quantized_size
    
    logger.info(
        f"FP16 conversion complete: {original_size / 1024:.1f} KB -> {quantized_size / 1024:.1f} KB "
        f"({compression_ratio:.2f}x compression)"
    )
    
    return QuantizationResult(
        output_path=output_path,
        original_size_bytes=original_size,
        quantized_size_bytes=quantized_size,
        compression_ratio=compression_ratio,
        quantization_method="fp16",
        accuracy_metrics={},
        passed_accuracy_check=True,
    )


def quantize_model(
    input_path: str,
    output_path: str,
    config: ExportConfig,
    method: Optional[QuantizationMethod] = None,
) -> QuantizationResult:
    """
    Quantize ONNX model based on configuration.
    
    This is the main entry point for model quantization. It selects
    the appropriate quantization method based on config and performs
    accuracy verification.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path for quantized output
        config: Export configuration
        method: Override quantization method (uses config default if None)
        
    Returns:
        QuantizationResult with metrics and accuracy info
    """
    quant_config = config.quantization_config or QuantizationConfig()
    method = method or quant_config.method
    
    input_specs = [s.to_dict() for s in config.input_specs]
    input_names = config.get_input_names()
    
    # Perform quantization
    if method == QuantizationMethod.DYNAMIC:
        result = quantize_dynamic_int8(input_path, output_path, quant_config)
    elif method == QuantizationMethod.STATIC:
        result = quantize_static_int8(
            input_path, output_path, quant_config, input_specs, input_names
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Compare accuracy
    logger.info("Comparing quantized model accuracy...")
    accuracy_metrics = _compare_models(
        input_path, output_path, input_specs, num_samples=20
    )
    result.accuracy_metrics = accuracy_metrics
    
    # Check accuracy threshold
    if accuracy_metrics["max_rel_error"] > quant_config.accuracy_threshold:
        logger.warning(
            f"Quantization accuracy degradation exceeds threshold: "
            f"max_rel_error={accuracy_metrics['max_rel_error']:.4f} > "
            f"threshold={quant_config.accuracy_threshold}"
        )
        result.passed_accuracy_check = False
    else:
        logger.info(
            f"Accuracy check passed: max_rel_error={accuracy_metrics['max_rel_error']:.4f}"
        )
        result.passed_accuracy_check = True
    
    return result
