"""
Model Quantization Engine

Post-training quantization to INT8 and FP16 for edge deployment.

Supports:
1. Dynamic quantization (quantize weights only, no calibration needed)
2. Static quantization (quantize weights and activations, needs calibration data)
3. FP16 conversion (half precision for GPU targets)

Also measures accuracy degradation from quantization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Results from model quantization."""
    
    # Output paths
    dynamic_int8_path: str | None = None
    static_int8_path: str | None = None
    fp16_path: str | None = None
    
    # Size comparison
    original_size_bytes: int = 0
    dynamic_int8_size_bytes: int = 0
    static_int8_size_bytes: int = 0
    fp16_size_bytes: int = 0
    
    # Accuracy degradation
    dynamic_int8_accuracy: dict[str, float] | None = None
    static_int8_accuracy: dict[str, float] | None = None
    fp16_accuracy: dict[str, float] | None = None


class CalibrationDataReader:
    """
    Provides calibration data for static quantization.
    
    Wraps a data generator to provide samples for determining
    quantization ranges.
    """
    
    def __init__(
        self,
        data_generator: Callable[[], np.ndarray] | Iterator[np.ndarray],
        num_samples: int = 100,
        input_name: str = "input",
    ):
        """
        Args:
            data_generator: Callable that returns input arrays, or iterator of arrays
            num_samples: Number of calibration samples
            input_name: Name of the input tensor in the ONNX model
        """
        self.input_name = input_name
        
        # Collect calibration data
        if callable(data_generator):
            self.data = [data_generator() for _ in range(num_samples)]
        else:
            self.data = []
            for i, sample in enumerate(data_generator):
                if i >= num_samples:
                    break
                self.data.append(sample)
        
        self.index = 0
    
    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get the next calibration sample."""
        if self.index >= len(self.data):
            return None
        
        result = {self.input_name: self.data[self.index]}
        self.index += 1
        return result
    
    def rewind(self) -> None:
        """Reset to the beginning."""
        self.index = 0


class ModelQuantizer:
    """
    ONNX model quantizer.
    
    Provides dynamic and static INT8 quantization, plus FP16 conversion.
    
    Usage:
        quantizer = ModelQuantizer(onnx_model_path)
        result = quantizer.quantize(
            output_dir,
            methods=["dynamic_int8", "static_int8", "fp16"],
            calibration_data=calibration_generator,
        )
    """
    
    def __init__(
        self,
        model_path: str | Path,
        per_channel: bool = True,
    ):
        """
        Args:
            model_path: Path to the ONNX model to quantize
            per_channel: Use per-channel quantization (better accuracy)
        """
        self.model_path = Path(model_path)
        self.per_channel = per_channel
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def quantize(
        self,
        output_dir: str | Path,
        methods: list[str] | None = None,
        calibration_data: Callable[[], np.ndarray] | Iterator[np.ndarray] | None = None,
        calibration_samples: int = 100,
        measure_accuracy: bool = True,
        test_data: Callable[[], np.ndarray] | None = None,
        num_test_samples: int = 100,
    ) -> QuantizationResult:
        """
        Quantize the model using specified methods.
        
        Args:
            output_dir: Directory for quantized models
            methods: List of methods: "dynamic_int8", "static_int8", "fp16"
            calibration_data: Data generator for static quantization
            calibration_samples: Number of calibration samples
            measure_accuracy: Whether to measure accuracy degradation
            test_data: Data generator for accuracy measurement
            num_test_samples: Number of samples for accuracy test
            
        Returns:
            QuantizationResult with paths and metrics
        """
        methods = methods or ["dynamic_int8"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = QuantizationResult()
        result.original_size_bytes = self.model_path.stat().st_size
        
        model_stem = self.model_path.stem
        
        # Dynamic INT8 quantization
        if "dynamic_int8" in methods:
            logger.info("Applying dynamic INT8 quantization...")
            output_path = output_dir / f"{model_stem}_int8_dynamic.onnx"
            self._quantize_dynamic(output_path)
            result.dynamic_int8_path = str(output_path)
            result.dynamic_int8_size_bytes = output_path.stat().st_size
            
            if measure_accuracy and test_data:
                result.dynamic_int8_accuracy = self._measure_accuracy(
                    output_path, test_data, num_test_samples
                )
        
        # Static INT8 quantization
        if "static_int8" in methods:
            if calibration_data is None:
                logger.warning("Static quantization requires calibration_data, skipping")
            else:
                logger.info("Applying static INT8 quantization...")
                output_path = output_dir / f"{model_stem}_int8_static.onnx"
                
                calibration_reader = CalibrationDataReader(
                    calibration_data, 
                    calibration_samples
                )
                self._quantize_static(output_path, calibration_reader)
                result.static_int8_path = str(output_path)
                result.static_int8_size_bytes = output_path.stat().st_size
                
                if measure_accuracy and test_data:
                    result.static_int8_accuracy = self._measure_accuracy(
                        output_path, test_data, num_test_samples
                    )
        
        # FP16 conversion
        if "fp16" in methods:
            logger.info("Converting to FP16...")
            output_path = output_dir / f"{model_stem}_fp16.onnx"
            self._convert_fp16(output_path)
            result.fp16_path = str(output_path)
            result.fp16_size_bytes = output_path.stat().st_size
            
            if measure_accuracy and test_data:
                result.fp16_accuracy = self._measure_accuracy(
                    output_path, test_data, num_test_samples
                )
        
        # Log summary
        self._log_summary(result)
        
        return result
    
    def _quantize_dynamic(self, output_path: Path) -> None:
        """Apply dynamic quantization (weights only)."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                str(self.model_path),
                str(output_path),
                weight_type=QuantType.QInt8,
            )
            
            logger.info(f"Dynamic quantization complete: {output_path}")
            
        except ImportError:
            logger.error("onnxruntime.quantization not available")
            raise
    
    def _quantize_static(
        self, 
        output_path: Path, 
        calibration_reader: CalibrationDataReader
    ) -> None:
        """Apply static quantization (weights and activations)."""
        try:
            from onnxruntime.quantization import (
                quantize_static,
                QuantFormat,
                QuantType,
                CalibrationMethod,
            )
            
            quantize_static(
                str(self.model_path),
                str(output_path),
                calibration_reader,
                quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format
                per_channel=self.per_channel,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
            
            logger.info(f"Static quantization complete: {output_path}")
            
        except ImportError:
            logger.error("onnxruntime.quantization not available")
            raise
    
    def _convert_fp16(self, output_path: Path) -> None:
        """Convert model to FP16 (half precision)."""
        try:
            import onnx
            from onnxruntime.transformers import float16
            
            model = onnx.load(str(self.model_path))
            
            # Convert to FP16, keeping IO types as FP32 for compatibility
            model_fp16 = float16.convert_float_to_float16(
                model, 
                keep_io_types=True
            )
            
            onnx.save(model_fp16, str(output_path))
            
            logger.info(f"FP16 conversion complete: {output_path}")
            
        except ImportError:
            logger.error("onnx or onnxruntime.transformers not available")
            raise
    
    def _measure_accuracy(
        self,
        quantized_path: Path,
        test_data: Callable[[], np.ndarray],
        num_samples: int,
    ) -> dict[str, float]:
        """
        Measure accuracy degradation from quantization.
        
        Compares quantized model output against original model output.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not available for accuracy measurement")
            return {}
        
        # Create sessions
        original_session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )
        quantized_session = ort.InferenceSession(
            str(quantized_path),
            providers=['CPUExecutionProvider']
        )
        
        input_name = original_session.get_inputs()[0].name
        
        errors = []
        relative_errors = []
        
        for _ in range(num_samples):
            # Get test input
            test_input = test_data()
            if not isinstance(test_input, np.ndarray):
                test_input = np.array(test_input, dtype=np.float32)
            
            # Run both models
            original_output = original_session.run(
                None, {input_name: test_input}
            )[0]
            quantized_output = quantized_session.run(
                None, {input_name: test_input}
            )[0]
            
            # Compute errors
            abs_error = np.abs(original_output - quantized_output)
            errors.append(abs_error)
            
            # Relative error (avoid division by zero)
            denom = np.abs(original_output) + 1e-8
            rel_error = abs_error / denom
            relative_errors.append(rel_error)
        
        errors = np.concatenate([e.flatten() for e in errors])
        relative_errors = np.concatenate([e.flatten() for e in relative_errors])
        
        return {
            'mean_absolute_error': float(np.mean(errors)),
            'max_absolute_error': float(np.max(errors)),
            'std_absolute_error': float(np.std(errors)),
            'mean_relative_error': float(np.mean(relative_errors)),
            'max_relative_error': float(np.max(relative_errors)),
            'p95_absolute_error': float(np.percentile(errors, 95)),
            'p99_absolute_error': float(np.percentile(errors, 99)),
        }
    
    def _log_summary(self, result: QuantizationResult) -> None:
        """Log a summary of quantization results."""
        original_mb = result.original_size_bytes / 1e6
        
        logger.info("Quantization Summary:")
        logger.info(f"  Original: {original_mb:.2f} MB")
        
        if result.dynamic_int8_path:
            size_mb = result.dynamic_int8_size_bytes / 1e6
            reduction = (1 - size_mb / original_mb) * 100
            logger.info(f"  Dynamic INT8: {size_mb:.2f} MB ({reduction:.1f}% reduction)")
            if result.dynamic_int8_accuracy:
                mae = result.dynamic_int8_accuracy.get('mean_absolute_error', 0)
                logger.info(f"    Mean absolute error: {mae:.2e}")
        
        if result.static_int8_path:
            size_mb = result.static_int8_size_bytes / 1e6
            reduction = (1 - size_mb / original_mb) * 100
            logger.info(f"  Static INT8: {size_mb:.2f} MB ({reduction:.1f}% reduction)")
            if result.static_int8_accuracy:
                mae = result.static_int8_accuracy.get('mean_absolute_error', 0)
                logger.info(f"    Mean absolute error: {mae:.2e}")
        
        if result.fp16_path:
            size_mb = result.fp16_size_bytes / 1e6
            reduction = (1 - size_mb / original_mb) * 100
            logger.info(f"  FP16: {size_mb:.2f} MB ({reduction:.1f}% reduction)")
            if result.fp16_accuracy:
                mae = result.fp16_accuracy.get('mean_absolute_error', 0)
                logger.info(f"    Mean absolute error: {mae:.2e}")


def quantize_model(
    model_path: str | Path,
    output_dir: str | Path,
    methods: list[str] | None = None,
    calibration_data: Callable[[], np.ndarray] | None = None,
    calibration_samples: int = 100,
    test_data: Callable[[], np.ndarray] | None = None,
    **kwargs
) -> QuantizationResult:
    """
    Convenience function to quantize a model.
    
    Args:
        model_path: Path to ONNX model
        output_dir: Output directory
        methods: Quantization methods to apply
        calibration_data: Data generator for static quantization
        calibration_samples: Number of calibration samples
        test_data: Data generator for accuracy testing
        **kwargs: Additional arguments for ModelQuantizer
        
    Returns:
        QuantizationResult
    """
    quantizer = ModelQuantizer(model_path, **kwargs)
    
    return quantizer.quantize(
        output_dir=output_dir,
        methods=methods,
        calibration_data=calibration_data,
        calibration_samples=calibration_samples,
        test_data=test_data,
    )
