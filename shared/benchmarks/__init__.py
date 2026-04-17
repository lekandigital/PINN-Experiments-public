"""
Shared Benchmarks Package.

Provides comprehensive benchmarking for ONNX models across platforms.
"""

from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    run_benchmark,
    compare_benchmarks,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite", 
    "run_benchmark",
    "compare_benchmarks",
]
