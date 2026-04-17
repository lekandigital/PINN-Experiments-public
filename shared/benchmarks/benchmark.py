"""
Benchmarking Framework for ONNX Models.

Provides comprehensive benchmarking across platforms with metrics including:
- Latency (median, p95, p99)
- Throughput (inferences/second)
- Memory usage
- Model size
- Accuracy vs PyTorch baseline
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    num_iterations: int
    warmup_iterations: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
        }


@dataclass
class ThroughputMetrics:
    """Throughput measurement results."""
    samples_per_second: float
    batches_per_second: float
    batch_size: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "samples_per_second": self.samples_per_second,
            "batches_per_second": self.batches_per_second,
            "batch_size": self.batch_size,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    model_size_mb: float
    peak_memory_mb: Optional[float] = None
    allocated_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_size_mb": self.model_size_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
        }


@dataclass
class AccuracyMetrics:
    """Accuracy comparison metrics."""
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    num_samples: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "max_rel_error": self.max_rel_error,
            "mean_rel_error": self.mean_rel_error,
            "num_samples": self.num_samples,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a model/platform combination."""
    model_path: str
    platform: str
    provider: str
    batch_size: int
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    memory: MemoryMetrics
    accuracy: Optional[AccuracyMetrics] = None
    first_inference_ms: Optional[float] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "platform": self.platform,
            "provider": self.provider,
            "batch_size": self.batch_size,
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "memory": self.memory.to_dict(),
            "accuracy": self.accuracy.to_dict() if self.accuracy else None,
            "first_inference_ms": self.first_inference_ms,
            "system_info": self.system_info,
            "timestamp": self.timestamp,
        }
    
    def save(self, path: str) -> None:
        """Save benchmark results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        """Load benchmark results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        data["latency"] = LatencyMetrics(**data["latency"])
        data["throughput"] = ThroughputMetrics(**data["throughput"])
        data["memory"] = MemoryMetrics(**data["memory"])
        if data.get("accuracy"):
            data["accuracy"] = AccuracyMetrics(**data["accuracy"])
        
        return cls(**data)


def _get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    import platform
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    
    if ORT_AVAILABLE:
        info["onnxruntime_version"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
    
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass
    
    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            info["gpu_name"] = gpu_info[0]
            info["gpu_memory_mb"] = int(gpu_info[1])
    except Exception:
        pass
    
    return info


def _create_random_input(
    session: "ort.InferenceSession",
    batch_size: int,
    input_specs: Optional[List[Dict]] = None,
) -> Dict[str, np.ndarray]:
    """Create random input tensors for benchmarking."""
    inputs = {}
    
    for inp in session.get_inputs():
        shape = list(inp.shape)
        
        # Replace dynamic dims
        for i, dim in enumerate(shape):
            if isinstance(dim, str) or dim is None or dim < 0:
                if i == 0:  # Batch dimension
                    shape[i] = batch_size
                else:
                    # Use spec if available
                    if input_specs:
                        for spec in input_specs:
                            if spec.get("name") == inp.name:
                                spec_shape = spec.get("shape", [])
                                if i < len(spec_shape) and spec_shape[i] > 0:
                                    shape[i] = spec_shape[i]
                                    break
                    if shape[i] < 0 or isinstance(shape[i], str):
                        shape[i] = 64  # Default
        
        # Create tensor
        dtype_str = inp.type.lower()
        if "float16" in dtype_str:
            dtype = np.float16
        elif "float" in dtype_str:
            dtype = np.float32
        elif "int64" in dtype_str:
            dtype = np.int64
        elif "int" in dtype_str:
            dtype = np.int32
        else:
            dtype = np.float32
        
        if np.issubdtype(dtype, np.floating):
            inputs[inp.name] = np.random.randn(*shape).astype(dtype)
        else:
            inputs[inp.name] = np.random.randint(0, 10, shape).astype(dtype)
    
    return inputs


def measure_latency(
    session: "ort.InferenceSession",
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
    num_iterations: int = 1000,
    warmup_iterations: int = 100,
) -> LatencyMetrics:
    """
    Measure inference latency.
    
    Args:
        session: ONNX Runtime session
        inputs: Input tensors
        output_names: Names of output tensors
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        LatencyMetrics with timing statistics
    """
    # Warmup
    for _ in range(warmup_iterations):
        session.run(output_names, inputs)
    
    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(output_names, inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return LatencyMetrics(
        mean_ms=float(np.mean(times)),
        std_ms=float(np.std(times)),
        min_ms=float(np.min(times)),
        max_ms=float(np.max(times)),
        median_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )


def measure_throughput(
    session: "ort.InferenceSession",
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
    duration_seconds: float = 10.0,
    batch_size: int = 1,
) -> ThroughputMetrics:
    """
    Measure sustained throughput.
    
    Args:
        session: ONNX Runtime session
        inputs: Input tensors
        output_names: Names of output tensors
        duration_seconds: Duration to measure over
        batch_size: Batch size being used
        
    Returns:
        ThroughputMetrics
    """
    num_batches = 0
    start_time = time.perf_counter()
    
    while True:
        session.run(output_names, inputs)
        num_batches += 1
        
        elapsed = time.perf_counter() - start_time
        if elapsed >= duration_seconds:
            break
    
    batches_per_second = num_batches / elapsed
    samples_per_second = batches_per_second * batch_size
    
    return ThroughputMetrics(
        samples_per_second=samples_per_second,
        batches_per_second=batches_per_second,
        batch_size=batch_size,
        duration_seconds=elapsed,
    )


def measure_first_inference(
    model_path: str,
    inputs: Dict[str, np.ndarray],
    providers: List[str],
) -> float:
    """
    Measure cold-start latency (time to first inference).
    
    This includes model loading time.
    
    Args:
        model_path: Path to ONNX model
        inputs: Input tensors
        providers: Execution providers
        
    Returns:
        Time to first inference in milliseconds
    """
    start = time.perf_counter()
    
    sess = ort.InferenceSession(model_path, providers=providers)
    output_names = [out.name for out in sess.get_outputs()]
    sess.run(output_names, inputs)
    
    end = time.perf_counter()
    return (end - start) * 1000


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for ONNX models.
    
    Runs benchmarks across different providers and batch sizes,
    collecting latency, throughput, memory, and accuracy metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        input_specs: Optional[List[Dict]] = None,
        reference_model_path: Optional[str] = None,
    ):
        """
        Initialize benchmark suite.
        
        Args:
            model_path: Path to ONNX model to benchmark
            input_specs: Optional input tensor specifications
            reference_model_path: Optional path to reference model for accuracy comparison
        """
        self.model_path = model_path
        self.input_specs = input_specs
        self.reference_model_path = reference_model_path
        self.model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    
    def run(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16],
        providers: Optional[List[str]] = None,
        num_iterations: int = 1000,
        warmup_iterations: int = 100,
        throughput_duration: float = 10.0,
        measure_accuracy: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run full benchmark suite.
        
        Args:
            batch_sizes: Batch sizes to test
            providers: Execution providers (auto-detect if None)
            num_iterations: Iterations for latency measurement
            warmup_iterations: Warmup iterations
            throughput_duration: Duration for throughput test
            measure_accuracy: Whether to measure accuracy vs reference
            
        Returns:
            List of BenchmarkResult for each provider/batch combination
        """
        if not ORT_AVAILABLE:
            raise ImportError("onnxruntime not available")
        
        # Auto-detect providers
        if providers is None:
            providers = []
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        
        results = []
        system_info = _get_system_info()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        for provider in providers:
            if provider not in ort.get_available_providers():
                logger.warning(f"Provider {provider} not available, skipping")
                continue
            
            logger.info(f"Benchmarking with {provider}")
            
            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            try:
                session = ort.InferenceSession(
                    self.model_path,
                    sess_options,
                    providers=[provider]
                )
            except Exception as e:
                logger.error(f"Failed to create session with {provider}: {e}")
                continue
            
            output_names = [out.name for out in session.get_outputs()]
            active_provider = session.get_providers()[0]
            
            for batch_size in batch_sizes:
                logger.info(f"  Batch size: {batch_size}")
                
                # Create input
                inputs = _create_random_input(session, batch_size, self.input_specs)
                
                # Measure latency
                latency = measure_latency(
                    session, inputs, output_names,
                    num_iterations, warmup_iterations
                )
                
                # Measure throughput
                throughput = measure_throughput(
                    session, inputs, output_names,
                    throughput_duration, batch_size
                )
                
                # Measure first inference (cold start)
                first_inference = measure_first_inference(
                    self.model_path, inputs, [provider]
                )
                
                # Memory metrics
                memory = MemoryMetrics(model_size_mb=self.model_size_mb)
                
                # Accuracy (if reference available)
                accuracy = None
                if measure_accuracy and self.reference_model_path:
                    accuracy = self._measure_accuracy(session, inputs, output_names)
                
                # Create result
                result = BenchmarkResult(
                    model_path=self.model_path,
                    platform="desktop",
                    provider=active_provider,
                    batch_size=batch_size,
                    latency=latency,
                    throughput=throughput,
                    memory=memory,
                    accuracy=accuracy,
                    first_inference_ms=first_inference,
                    system_info=system_info,
                    timestamp=timestamp,
                )
                results.append(result)
                
                logger.info(
                    f"    Latency: {latency.median_ms:.2f}ms (p99: {latency.p99_ms:.2f}ms), "
                    f"Throughput: {throughput.samples_per_second:.0f} samples/s"
                )
        
        return results
    
    def _measure_accuracy(
        self,
        session: "ort.InferenceSession",
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
        num_samples: int = 20,
    ) -> AccuracyMetrics:
        """Measure accuracy against reference model."""
        ref_session = ort.InferenceSession(
            self.reference_model_path,
            providers=['CPUExecutionProvider']
        )
        ref_output_names = [out.name for out in ref_session.get_outputs()]
        
        max_abs_errors = []
        max_rel_errors = []
        mean_abs_errors = []
        
        for _ in range(num_samples):
            # Run both models
            output = session.run(output_names, inputs)
            ref_output = ref_session.run(ref_output_names, inputs)
            
            for out, ref_out in zip(output, ref_output):
                abs_error = np.abs(out - ref_out)
                max_abs_errors.append(np.max(abs_error))
                mean_abs_errors.append(np.mean(abs_error))
                
                denom = np.maximum(np.abs(ref_out), 1e-7)
                rel_error = abs_error / denom
                max_rel_errors.append(np.max(rel_error))
        
        return AccuracyMetrics(
            max_abs_error=float(np.max(max_abs_errors)),
            mean_abs_error=float(np.mean(mean_abs_errors)),
            max_rel_error=float(np.max(max_rel_errors)),
            mean_rel_error=float(np.mean(max_rel_errors)),
            num_samples=num_samples,
        )


def run_benchmark(
    model_path: str,
    batch_size: int = 1,
    provider: str = "CPUExecutionProvider",
    num_iterations: int = 1000,
    input_specs: Optional[List[Dict]] = None,
) -> BenchmarkResult:
    """
    Convenience function to run a single benchmark.
    
    Args:
        model_path: Path to ONNX model
        batch_size: Batch size
        provider: Execution provider
        num_iterations: Number of iterations
        input_specs: Input tensor specifications
        
    Returns:
        BenchmarkResult
    """
    suite = BenchmarkSuite(model_path, input_specs)
    results = suite.run(
        batch_sizes=[batch_size],
        providers=[provider],
        num_iterations=num_iterations,
    )
    return results[0] if results else None


def compare_benchmarks(
    results: List[BenchmarkResult],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare benchmark results and generate summary.
    
    Args:
        results: List of benchmark results to compare
        output_path: Optional path to save comparison as JSON
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        "summary": [],
        "best_latency": None,
        "best_throughput": None,
    }
    
    best_latency = float('inf')
    best_throughput = 0
    
    for result in results:
        summary = {
            "model": Path(result.model_path).name,
            "platform": result.platform,
            "provider": result.provider,
            "batch_size": result.batch_size,
            "latency_median_ms": result.latency.median_ms,
            "latency_p99_ms": result.latency.p99_ms,
            "throughput_fps": result.throughput.samples_per_second,
            "model_size_mb": result.memory.model_size_mb,
        }
        
        if result.accuracy:
            summary["max_rel_error"] = result.accuracy.max_rel_error
        
        comparison["summary"].append(summary)
        
        if result.latency.median_ms < best_latency:
            best_latency = result.latency.median_ms
            comparison["best_latency"] = summary
        
        if result.throughput.samples_per_second > best_throughput:
            best_throughput = result.throughput.samples_per_second
            comparison["best_throughput"] = summary
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    return comparison


def generate_browser_benchmark_html(
    model_path: str,
    metadata_path: str,
    output_path: str,
) -> str:
    """
    Generate a standalone HTML page for browser benchmarking.
    
    The generated page loads the ONNX model via ONNX Runtime Web,
    runs benchmarks, and displays results.
    
    Args:
        model_path: Path to ONNX model (will be referenced, not embedded)
        metadata_path: Path to metadata JSON
        output_path: Output path for HTML file
        
    Returns:
        Path to generated HTML file
    """
    model_filename = Path(model_path).name
    metadata_filename = Path(metadata_path).name
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Model Browser Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{ color: #00d4ff; }}
        .card {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }}
        .metric-value {{
            font-weight: bold;
            color: #00d4ff;
        }}
        button {{
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 5px;
        }}
        button:disabled {{
            background: #666;
            cursor: not-allowed;
        }}
        #status {{
            padding: 10px;
            background: #0f3460;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .progress {{
            height: 4px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
        }}
        .progress-bar {{
            height: 100%;
            background: #00d4ff;
            width: 0%;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <h1>🔬 ONNX Model Browser Benchmark</h1>
    
    <div class="card">
        <h3>Model Information</h3>
        <div class="metric">
            <span>Model File:</span>
            <span class="metric-value" id="modelFile">{model_filename}</span>
        </div>
        <div class="metric">
            <span>Backend:</span>
            <span class="metric-value" id="backend">Detecting...</span>
        </div>
        <div class="metric">
            <span>Status:</span>
            <span class="metric-value" id="loadStatus">Not loaded</span>
        </div>
    </div>
    
    <div id="status">Click "Load Model" to begin</div>
    <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
    
    <button id="loadBtn" onclick="loadModel()">Load Model</button>
    <button id="benchBtn" onclick="runBenchmark()" disabled>Run Benchmark</button>
    <button id="exportBtn" onclick="exportResults()" disabled>Export Results</button>
    
    <div class="card" id="resultsCard" style="display: none;">
        <h3>📊 Benchmark Results</h3>
        <div class="metric">
            <span>Warmup Iterations:</span>
            <span class="metric-value" id="warmupIter">-</span>
        </div>
        <div class="metric">
            <span>Benchmark Iterations:</span>
            <span class="metric-value" id="benchIter">-</span>
        </div>
        <div class="metric">
            <span>Mean Latency:</span>
            <span class="metric-value" id="meanLatency">-</span>
        </div>
        <div class="metric">
            <span>Median Latency:</span>
            <span class="metric-value" id="medianLatency">-</span>
        </div>
        <div class="metric">
            <span>P95 Latency:</span>
            <span class="metric-value" id="p95Latency">-</span>
        </div>
        <div class="metric">
            <span>P99 Latency:</span>
            <span class="metric-value" id="p99Latency">-</span>
        </div>
        <div class="metric">
            <span>Throughput:</span>
            <span class="metric-value" id="throughput">-</span>
        </div>
        <div class="metric">
            <span>First Inference:</span>
            <span class="metric-value" id="firstInference">-</span>
        </div>
    </div>
    
    <script>
        let session = null;
        let metadata = null;
        let benchmarkResults = null;
        
        const MODEL_PATH = '{model_filename}';
        const METADATA_PATH = '{metadata_filename}';
        
        async function detectBackend() {{
            const backends = [];
            
            // Check WebGPU
            if (navigator.gpu) {{
                backends.push('webgpu');
            }}
            
            // WebGL is generally available
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            if (gl) {{
                backends.push('webgl');
            }}
            
            // WASM is always available
            backends.push('wasm');
            
            // Check WebNN
            if (navigator.ml) {{
                backends.push('webnn');
            }}
            
            return backends;
        }}
        
        async function loadModel() {{
            const status = document.getElementById('status');
            const loadBtn = document.getElementById('loadBtn');
            const benchBtn = document.getElementById('benchBtn');
            const loadStatus = document.getElementById('loadStatus');
            const backendEl = document.getElementById('backend');
            
            loadBtn.disabled = true;
            status.textContent = 'Loading model...';
            
            try {{
                // Detect available backends
                const backends = await detectBackend();
                backendEl.textContent = backends.join(', ');
                
                // Configure ONNX Runtime
                ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
                
                // Try backends in order of preference
                const executionProviders = [];
                if (backends.includes('webgpu')) {{
                    executionProviders.push('webgpu');
                }}
                if (backends.includes('webgl')) {{
                    executionProviders.push('webgl');
                }}
                executionProviders.push('wasm');
                
                const startTime = performance.now();
                
                // Load model
                session = await ort.InferenceSession.create(MODEL_PATH, {{
                    executionProviders: executionProviders,
                }});
                
                const loadTime = performance.now() - startTime;
                
                // Load metadata
                try {{
                    const response = await fetch(METADATA_PATH);
                    metadata = await response.json();
                }} catch (e) {{
                    console.warn('Could not load metadata:', e);
                }}
                
                loadStatus.textContent = `Loaded in ${{loadTime.toFixed(0)}}ms`;
                status.textContent = `Model loaded successfully. Active provider: ${{session.handler._ep || 'unknown'}}`;
                benchBtn.disabled = false;
                
            }} catch (error) {{
                status.textContent = `Error loading model: ${{error.message}}`;
                loadStatus.textContent = 'Failed';
                console.error(error);
            }}
            
            loadBtn.disabled = false;
        }}
        
        function createRandomInput(inputMeta) {{
            // Get shape, replacing dynamic dims
            const shape = inputMeta.dims.map((d, i) => {{
                if (d < 0 || d === null) {{
                    return i === 0 ? 1 : 64; // Batch=1, others=64
                }}
                return d;
            }});
            
            const size = shape.reduce((a, b) => a * b, 1);
            const data = new Float32Array(size);
            for (let i = 0; i < size; i++) {{
                data[i] = (Math.random() - 0.5) * 2;
            }}
            
            return new ort.Tensor('float32', data, shape);
        }}
        
        async function runBenchmark() {{
            const status = document.getElementById('status');
            const benchBtn = document.getElementById('benchBtn');
            const exportBtn = document.getElementById('exportBtn');
            const progressBar = document.getElementById('progressBar');
            const resultsCard = document.getElementById('resultsCard');
            
            benchBtn.disabled = true;
            status.textContent = 'Running benchmark...';
            
            const WARMUP = 100;
            const ITERATIONS = 1000;
            
            try {{
                // Create input
                const inputMeta = session.inputNames.map(name => session.handler._model.graph.inputs.find(i => i.name === name) || {{}});
                const feeds = {{}};
                for (const name of session.inputNames) {{
                    const meta = session.handler._model?.graph?.inputs?.find(i => i.name === name);
                    feeds[name] = createRandomInput(meta || {{ dims: [1, 4] }});
                }}
                
                // Measure first inference
                const firstStart = performance.now();
                await session.run(feeds);
                const firstInference = performance.now() - firstStart;
                
                // Warmup
                status.textContent = `Warmup: 0/${{WARMUP}}`;
                for (let i = 0; i < WARMUP; i++) {{
                    await session.run(feeds);
                    if (i % 10 === 0) {{
                        status.textContent = `Warmup: ${{i}}/${{WARMUP}}`;
                        progressBar.style.width = `${{(i / WARMUP) * 30}}%`;
                        await new Promise(r => setTimeout(r, 0));
                    }}
                }}
                
                // Benchmark
                const times = [];
                for (let i = 0; i < ITERATIONS; i++) {{
                    const start = performance.now();
                    await session.run(feeds);
                    times.push(performance.now() - start);
                    
                    if (i % 50 === 0) {{
                        status.textContent = `Benchmarking: ${{i}}/${{ITERATIONS}}`;
                        progressBar.style.width = `${{30 + (i / ITERATIONS) * 70}}%`;
                        await new Promise(r => setTimeout(r, 0));
                    }}
                }}
                
                // Calculate statistics
                times.sort((a, b) => a - b);
                const mean = times.reduce((a, b) => a + b) / times.length;
                const median = times[Math.floor(times.length / 2)];
                const p95 = times[Math.floor(times.length * 0.95)];
                const p99 = times[Math.floor(times.length * 0.99)];
                
                benchmarkResults = {{
                    warmupIterations: WARMUP,
                    benchmarkIterations: ITERATIONS,
                    firstInferenceMs: firstInference,
                    meanMs: mean,
                    medianMs: median,
                    p95Ms: p95,
                    p99Ms: p99,
                    throughputFps: 1000 / mean,
                    backend: session.handler._ep || 'wasm',
                    timestamp: new Date().toISOString(),
                    userAgent: navigator.userAgent,
                }};
                
                // Display results
                document.getElementById('warmupIter').textContent = WARMUP;
                document.getElementById('benchIter').textContent = ITERATIONS;
                document.getElementById('meanLatency').textContent = `${{mean.toFixed(2)}} ms`;
                document.getElementById('medianLatency').textContent = `${{median.toFixed(2)}} ms`;
                document.getElementById('p95Latency').textContent = `${{p95.toFixed(2)}} ms`;
                document.getElementById('p99Latency').textContent = `${{p99.toFixed(2)}} ms`;
                document.getElementById('throughput').textContent = `${{(1000 / mean).toFixed(1)}} FPS`;
                document.getElementById('firstInference').textContent = `${{firstInference.toFixed(2)}} ms`;
                
                resultsCard.style.display = 'block';
                progressBar.style.width = '100%';
                status.textContent = 'Benchmark complete!';
                exportBtn.disabled = false;
                
            }} catch (error) {{
                status.textContent = `Benchmark error: ${{error.message}}`;
                console.error(error);
            }}
            
            benchBtn.disabled = false;
        }}
        
        function exportResults() {{
            if (!benchmarkResults) return;
            
            const blob = new Blob([JSON.stringify(benchmarkResults, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'browser_benchmark_results.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // Auto-detect on load
        detectBackend().then(backends => {{
            document.getElementById('backend').textContent = backends.join(', ');
        }});
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Browser benchmark HTML generated: {output_path}")
    return output_path
