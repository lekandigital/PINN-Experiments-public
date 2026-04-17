"""
Benchmark Harness Core Module

Provides framework-agnostic utilities for benchmarking physics-informed ML models.
"""

from .core import BenchmarkResult, BenchmarkConfig
from .base_adapter import ProjectAdapter
from .metrics import compute_rmse, compute_nrmse, compute_l2_relative_error
from .timing import TimingResult, time_inference_pytorch, time_inference_jax, time_inference_onnx
from .memory import MemoryResult, measure_memory_pytorch, measure_memory_jax
from .hardware import get_hardware_info
from .report import save_json_result, generate_markdown_tables, generate_html_dashboard

__all__ = [
    # Core
    'BenchmarkResult',
    'BenchmarkConfig',
    'ProjectAdapter',
    # Metrics
    'compute_rmse',
    'compute_nrmse', 
    'compute_l2_relative_error',
    # Timing
    'TimingResult',
    'time_inference_pytorch',
    'time_inference_jax',
    'time_inference_onnx',
    # Memory
    'MemoryResult',
    'measure_memory_pytorch',
    'measure_memory_jax',
    # Hardware
    'get_hardware_info',
    # Report
    'save_json_result',
    'generate_markdown_tables',
    'generate_html_dashboard',
]
