"""
Hardware Benchmarking Harness

Benchmarks model inference across different hardware targets and backends.

Measures:
- Latency (mean, median, P95, P99)
- Throughput (samples/second)
- Memory usage (peak and current)
- Model size on disk
- Accuracy vs baseline

Supports backends:
- ONNX Runtime (CPU, CUDA, TensorRT)
- PyTorch native
- TorchScript

Extracted and generalized from Project 15 (PINN-Lite-Foil).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    num_runs: int


@dataclass
class ThroughputMetrics:
    """Throughput measurement results."""
    samples_per_second: float
    batches_per_second: float
    duration_seconds: float
    total_samples: int


@dataclass
class MemoryMetrics:
    """Memory usage results."""
    peak_mb: float
    current_mb: float
    device: str


@dataclass
class ModelMetrics:
    """Model size and parameter metrics."""
    file_size_bytes: int
    file_size_mb: float
    parameter_count: int | None


@dataclass
class AccuracyMetrics:
    """Accuracy comparison metrics."""
    mean_absolute_error: float
    max_absolute_error: float
    mean_relative_error: float
    correlation: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a single model."""
    
    model_name: str
    backend: str
    
    # Core metrics
    latency: LatencyMetrics | None = None
    throughput: ThroughputMetrics | None = None
    memory: MemoryMetrics | None = None
    model_info: ModelMetrics | None = None
    accuracy: AccuracyMetrics | None = None
    
    # System info
    system_info: dict[str, Any] = field(default_factory=dict)
    
    # Batch size used
    batch_size: int = 1
    
    # Target pass/fail
    targets_met: dict[str, bool] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Multi-backend benchmarking harness.
    
    Supports various backends for comparing inference performance.
    
    Usage:
        runner = BenchmarkRunner()
        
        # Benchmark ONNX model
        result = runner.benchmark_onnx(
            model_path="model.onnx",
            input_shape=[1, 4],
            backends=["cpu", "cuda"],
        )
        
        # Benchmark PyTorch model
        result = runner.benchmark_pytorch(
            model=my_model,
            input_shape=[1, 4],
        )
    """
    
    def __init__(
        self,
        num_warmup: int = 50,
        num_runs: int = 1000,
        throughput_duration: float = 10.0,
    ):
        """
        Args:
            num_warmup: Number of warmup iterations before timing
            num_runs: Number of timed iterations for latency
            throughput_duration: Duration in seconds for throughput test
        """
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.throughput_duration = throughput_duration
        
        self.system_info = self._collect_system_info()
    
    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        # CPU info
        try:
            import psutil
            info['cpu_count_logical'] = psutil.cpu_count(logical=True)
            info['cpu_count_physical'] = psutil.cpu_count(logical=False)
            info['ram_total_gb'] = psutil.virtual_memory().total / 1e9
            info['ram_available_gb'] = psutil.virtual_memory().available / 1e9
        except ImportError:
            pass
        
        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_name'] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info['gpu_memory_gb'] = props.total_memory / 1e9
                info['gpu_compute_capability'] = f"{props.major}.{props.minor}"
            else:
                info['cuda_available'] = False
        except ImportError:
            info['cuda_available'] = False
        
        # ONNX Runtime info
        try:
            import onnxruntime as ort
            info['onnxruntime_version'] = ort.__version__
            info['onnxruntime_providers'] = ort.get_available_providers()
        except ImportError:
            pass
        
        return info
    
    def benchmark_onnx(
        self,
        model_path: str | Path,
        input_shape: list[int],
        backends: list[str] | None = None,
        input_generator: Callable[[], np.ndarray] | None = None,
        baseline_outputs: np.ndarray | None = None,
    ) -> list[BenchmarkResult]:
        """
        Benchmark an ONNX model across multiple backends.
        
        Args:
            model_path: Path to ONNX model
            input_shape: Shape of input tensor
            backends: List of backends ("cpu", "cuda", "tensorrt")
            input_generator: Optional custom input generator
            baseline_outputs: Optional baseline for accuracy comparison
            
        Returns:
            List of BenchmarkResult for each backend
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed")
            return []
        
        model_path = Path(model_path)
        backends = backends or ["cpu"]
        
        # Map backend names to ONNX Runtime providers
        provider_map = {
            'cpu': 'CPUExecutionProvider',
            'cuda': 'CUDAExecutionProvider',
            'tensorrt': 'TensorrtExecutionProvider',
        }
        
        results = []
        
        for backend in backends:
            provider = provider_map.get(backend)
            if provider is None:
                logger.warning(f"Unknown backend: {backend}")
                continue
            
            # Check if provider is available
            available = ort.get_available_providers()
            if provider not in available:
                logger.warning(f"Provider {provider} not available, skipping")
                continue
            
            logger.info(f"Benchmarking ONNX model on {backend}...")
            
            # Create session
            session = ort.InferenceSession(
                str(model_path),
                providers=[provider]
            )
            
            input_name = session.get_inputs()[0].name
            
            # Input generator
            if input_generator is None:
                def input_generator():
                    return np.random.randn(*input_shape).astype(np.float32)
            
            # Benchmark function
            def run_inference(inputs: np.ndarray) -> np.ndarray:
                return session.run(None, {input_name: inputs})[0]
            
            result = self._benchmark_function(
                name=model_path.stem,
                backend=f"onnx_{backend}",
                inference_fn=run_inference,
                input_generator=input_generator,
                batch_size=input_shape[0],
                baseline_outputs=baseline_outputs,
            )
            
            # Add model info
            result.model_info = ModelMetrics(
                file_size_bytes=model_path.stat().st_size,
                file_size_mb=model_path.stat().st_size / 1e6,
                parameter_count=self._count_onnx_params(model_path),
            )
            
            results.append(result)
        
        return results
    
    def benchmark_pytorch(
        self,
        model,  # nn.Module
        input_shape: list[int],
        device: str = "cpu",
        input_generator: Callable | None = None,
        baseline_outputs=None,  # np.ndarray
        model_name: str = "pytorch_model",
    ) -> BenchmarkResult:
        """
        Benchmark a PyTorch model.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor
            device: Device to benchmark on
            input_generator: Optional custom input generator
            baseline_outputs: Optional baseline for accuracy comparison
            model_name: Name for the result
            
        Returns:
            BenchmarkResult
        """
        import torch
        
        model = model.to(device)
        model.eval()
        
        # Input generator
        if input_generator is None:
            def input_generator():
                return torch.randn(*input_shape, device=device)
        
        # Benchmark function
        def run_inference(inputs):
            with torch.no_grad():
                return model(inputs)
        
        # For latency measurement, we need to handle GPU sync
        is_cuda = device.startswith('cuda')
        
        result = self._benchmark_function(
            name=model_name,
            backend=f"pytorch_{device}",
            inference_fn=run_inference,
            input_generator=input_generator,
            batch_size=input_shape[0],
            baseline_outputs=baseline_outputs,
            cuda_sync=is_cuda,
        )
        
        # Add model info
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        result.model_info = ModelMetrics(
            file_size_bytes=param_size + buffer_size,
            file_size_mb=(param_size + buffer_size) / 1e6,
            parameter_count=param_count,
        )
        
        # Memory measurement for PyTorch
        if is_cuda:
            import torch
            torch.cuda.reset_peak_memory_stats()
            _ = run_inference(input_generator())
            torch.cuda.synchronize()
            
            result.memory = MemoryMetrics(
                peak_mb=torch.cuda.max_memory_allocated() / 1e6,
                current_mb=torch.cuda.memory_allocated() / 1e6,
                device=device,
            )
        
        return result
    
    def _benchmark_function(
        self,
        name: str,
        backend: str,
        inference_fn: Callable,
        input_generator: Callable,
        batch_size: int,
        baseline_outputs=None,
        cuda_sync: bool = False,
    ) -> BenchmarkResult:
        """
        Core benchmarking logic.
        
        Args:
            name: Model name
            backend: Backend identifier
            inference_fn: Function to benchmark
            input_generator: Function to generate inputs
            batch_size: Batch size being used
            baseline_outputs: Optional baseline for accuracy
            cuda_sync: Whether to synchronize CUDA
        """
        import torch
        
        result = BenchmarkResult(
            model_name=name,
            backend=backend,
            batch_size=batch_size,
            system_info=self.system_info,
        )
        
        # Warmup
        logger.debug(f"Running {self.num_warmup} warmup iterations...")
        for _ in range(self.num_warmup):
            inputs = input_generator()
            _ = inference_fn(inputs)
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Latency measurement
        logger.debug(f"Running {self.num_runs} timed iterations...")
        latencies = []
        
        for _ in range(self.num_runs):
            inputs = input_generator()
            
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter_ns()
            _ = inference_fn(inputs)
            
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)  # Convert to ms
        
        latencies = np.array(latencies)
        
        result.latency = LatencyMetrics(
            mean_ms=float(np.mean(latencies)),
            median_ms=float(np.median(latencies)),
            std_ms=float(np.std(latencies)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            num_runs=self.num_runs,
        )
        
        # Throughput measurement
        logger.debug(f"Running throughput test for {self.throughput_duration}s...")
        count = 0
        start = time.perf_counter()
        
        while time.perf_counter() - start < self.throughput_duration:
            inputs = input_generator()
            _ = inference_fn(inputs)
            count += 1
        
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        result.throughput = ThroughputMetrics(
            samples_per_second=count * batch_size / elapsed,
            batches_per_second=count / elapsed,
            duration_seconds=elapsed,
            total_samples=count * batch_size,
        )
        
        # Accuracy comparison
        if baseline_outputs is not None:
            logger.debug("Computing accuracy metrics...")
            test_outputs = []
            
            for _ in range(min(100, self.num_runs)):
                inputs = input_generator()
                output = inference_fn(inputs)
                
                # Convert to numpy if needed
                if hasattr(output, 'cpu'):
                    output = output.cpu().numpy()
                
                test_outputs.append(output)
            
            test_outputs = np.concatenate(test_outputs, axis=0)
            
            # Align shapes
            min_len = min(len(test_outputs), len(baseline_outputs))
            test_outputs = test_outputs[:min_len]
            baseline = baseline_outputs[:min_len]
            
            errors = np.abs(test_outputs - baseline)
            relative_errors = errors / (np.abs(baseline) + 1e-8)
            
            result.accuracy = AccuracyMetrics(
                mean_absolute_error=float(np.mean(errors)),
                max_absolute_error=float(np.max(errors)),
                mean_relative_error=float(np.mean(relative_errors)),
                correlation=float(np.corrcoef(
                    test_outputs.flatten(), 
                    baseline.flatten()
                )[0, 1]) if len(test_outputs.flatten()) > 1 else 1.0,
            )
        
        return result
    
    def _count_onnx_params(self, model_path: Path) -> int | None:
        """Count parameters in an ONNX model."""
        try:
            import onnx
            model = onnx.load(str(model_path))
            
            param_count = 0
            for initializer in model.graph.initializer:
                param_count += int(np.prod(initializer.dims))
            
            return param_count
        except Exception:
            return None
    
    def check_targets(
        self,
        result: BenchmarkResult,
        targets: dict[str, float],
    ) -> dict[str, bool]:
        """
        Check if benchmark results meet targets.
        
        Args:
            result: Benchmark result to check
            targets: Dictionary of target metrics
            
        Returns:
            Dictionary of target name -> whether met
        """
        met = {}
        
        if 'latency_ms' in targets and result.latency:
            met['latency_ms'] = result.latency.mean_ms <= targets['latency_ms']
        
        if 'model_size_mb' in targets and result.model_info:
            met['model_size_mb'] = result.model_info.file_size_mb <= targets['model_size_mb']
        
        if 'peak_memory_mb' in targets and result.memory:
            met['peak_memory_mb'] = result.memory.peak_mb <= targets['peak_memory_mb']
        
        if 'throughput_samples_per_sec' in targets and result.throughput:
            met['throughput_samples_per_sec'] = (
                result.throughput.samples_per_second >= targets['throughput_samples_per_sec']
            )
        
        if 'accuracy_relative' in targets and result.accuracy:
            met['accuracy_relative'] = (
                result.accuracy.mean_relative_error <= targets['accuracy_relative']
            )
        
        result.targets_met = met
        return met


def generate_benchmark_report(
    results: list[BenchmarkResult],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Generate a benchmark comparison report.
    
    Args:
        results: List of benchmark results
        output_path: Optional path to save JSON report
        
    Returns:
        Report dictionary
    """
    report = {
        'summary': {},
        'details': [],
        'comparison_table': [],
    }
    
    # Summary
    if results:
        report['summary'] = {
            'num_benchmarks': len(results),
            'backends_tested': list(set(r.backend for r in results)),
            'system_info': results[0].system_info,
        }
    
    # Details for each result
    for result in results:
        detail = {
            'model_name': result.model_name,
            'backend': result.backend,
            'batch_size': result.batch_size,
        }
        
        if result.latency:
            detail['latency'] = {
                'mean_ms': result.latency.mean_ms,
                'p95_ms': result.latency.p95_ms,
                'p99_ms': result.latency.p99_ms,
            }
        
        if result.throughput:
            detail['throughput'] = {
                'samples_per_second': result.throughput.samples_per_second,
            }
        
        if result.model_info:
            detail['model'] = {
                'size_mb': result.model_info.file_size_mb,
                'parameters': result.model_info.parameter_count,
            }
        
        if result.accuracy:
            detail['accuracy'] = {
                'mean_relative_error': result.accuracy.mean_relative_error,
                'correlation': result.accuracy.correlation,
            }
        
        if result.targets_met:
            detail['targets_met'] = result.targets_met
        
        report['details'].append(detail)
    
    # Comparison table
    for result in results:
        row = {
            'Model': result.model_name,
            'Backend': result.backend,
            'Params': f"{result.model_info.parameter_count:,}" if result.model_info and result.model_info.parameter_count else "N/A",
            'Size (MB)': f"{result.model_info.file_size_mb:.2f}" if result.model_info else "N/A",
            'Latency (ms)': f"{result.latency.mean_ms:.2f}" if result.latency else "N/A",
            'FPS': f"{result.throughput.samples_per_second:.0f}" if result.throughput else "N/A",
            'Rel. Error': f"{result.accuracy.mean_relative_error:.4f}" if result.accuracy else "N/A",
        }
        report['comparison_table'].append(row)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also generate markdown table
        md_path = output_path.with_suffix('.md')
        _generate_markdown_report(report, md_path)
    
    return report


def _generate_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    """Generate a markdown version of the benchmark report."""
    lines = [
        "# Benchmark Report\n",
        f"**Benchmarks:** {report['summary'].get('num_benchmarks', 0)}  ",
        f"**Backends:** {', '.join(report['summary'].get('backends_tested', []))}\n",
        "",
        "## Results\n",
    ]
    
    # Table header
    if report['comparison_table']:
        headers = list(report['comparison_table'][0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in report['comparison_table']:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    
    lines.append("\n## System Info\n")
    for key, value in report['summary'].get('system_info', {}).items():
        lines.append(f"- **{key}:** {value}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
