"""
Framework-aware timing utilities.

Provides accurate timing for PyTorch (with CUDA sync), JAX (with block_until_ready),
and ONNX Runtime inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
import numpy as np


@dataclass
class TimingResult:
    """Raw timing measurements before statistical aggregation."""
    times_ms: list[float]
    num_warmup: int
    num_runs: int
    device: str
    framework: str
    
    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms))
    
    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms))
    
    @property
    def median_ms(self) -> float:
        return float(np.median(self.times_ms))
    
    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.times_ms, 95))
    
    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.times_ms, 99))
    
    @property
    def min_ms(self) -> float:
        return float(np.min(self.times_ms))
    
    @property
    def max_ms(self) -> float:
        return float(np.max(self.times_ms))
    
    def to_timing_stats(self):
        """Convert to TimingStats dataclass from core module."""
        from .core import TimingStats
        return TimingStats(
            mean_ms=self.mean_ms,
            std_ms=self.std_ms,
            median_ms=self.median_ms,
            p95_ms=self.p95_ms,
            p99_ms=self.p99_ms,
            min_ms=self.min_ms,
            max_ms=self.max_ms,
            num_runs=self.num_runs,
        )


def time_inference_pytorch(
    model: Any,
    input_data: Any,
    num_warmup: int = 100,
    num_runs: int = 1000,
    device: str = "cuda",
    use_cuda_events: bool = True,
) -> TimingResult:
    """
    Time PyTorch model inference with proper CUDA synchronization.
    
    Uses CUDA events for GPU timing (most accurate) or perf_counter for CPU.
    
    Args:
        model: PyTorch model (already on device, in eval mode)
        input_data: Input tensor(s) - can be a tensor, tuple, or dict
        num_warmup: Number of warmup runs (discarded)
        num_runs: Number of timed runs
        device: "cuda" or "cpu"
        use_cuda_events: Use CUDA events for GPU timing (more accurate)
    
    Returns:
        TimingResult with all individual timing measurements
    """
    import torch
    
    model.eval()
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    
    # Helper to run inference handling different input types
    def run_inference():
        with torch.no_grad():
            if isinstance(input_data, dict):
                return model(**input_data)
            elif isinstance(input_data, (tuple, list)):
                return model(*input_data)
            else:
                return model(input_data)
    
    # Warmup
    for _ in range(num_warmup):
        _ = run_inference()
        if is_cuda:
            torch.cuda.synchronize()
    
    times_ms = []
    
    if is_cuda and use_cuda_events:
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for _ in range(num_runs):
            start_event.record()
            _ = run_inference()
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        # CPU timing or fallback
        if is_cuda:
            torch.cuda.synchronize()
        
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = run_inference()
            if is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)
    
    return TimingResult(
        times_ms=times_ms,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device,
        framework="pytorch",
    )


def time_inference_pytorch_batched(
    model: Any,
    prepare_input_fn: Callable[[int], Any],
    batch_sizes: list[int],
    num_warmup: int = 100,
    num_runs: int = 1000,
    device: str = "cuda",
) -> dict[int, TimingResult]:
    """
    Time PyTorch inference across multiple batch sizes.
    
    Args:
        model: PyTorch model
        prepare_input_fn: Function that takes batch_size and returns input data
        batch_sizes: List of batch sizes to test
        num_warmup: Warmup runs per batch size
        num_runs: Timed runs per batch size
        device: Device string
    
    Returns:
        Dict mapping batch_size to TimingResult
    """
    results = {}
    for batch_size in batch_sizes:
        input_data = prepare_input_fn(batch_size)
        results[batch_size] = time_inference_pytorch(
            model, input_data, num_warmup, num_runs, device
        )
    return results


def time_inference_jax(
    apply_fn: Callable,
    params: Any,
    input_data: Any,
    rng_key: Optional[Any] = None,
    num_warmup: int = 100,
    num_runs: int = 1000,
    use_jit: bool = True,
) -> TimingResult:
    """
    Time JAX model inference with proper JIT warmup and block_until_ready.
    
    Args:
        apply_fn: The model's apply function (e.g., model.apply for Haiku)
        params: Model parameters
        input_data: Input array(s)
        rng_key: Optional JAX PRNG key (for stochastic models)
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        use_jit: Whether to JIT compile the function
    
    Returns:
        TimingResult with timing measurements
    """
    import jax
    import jax.numpy as jnp
    
    # Determine if we have a GPU
    devices = jax.devices()
    device = "cuda" if any(d.platform == "gpu" for d in devices) else "cpu"
    
    # Prepare the inference function
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    def inference_fn():
        if isinstance(input_data, dict):
            result = apply_fn(params, rng_key, **input_data)
        elif isinstance(input_data, (tuple, list)):
            result = apply_fn(params, rng_key, *input_data)
        else:
            result = apply_fn(params, rng_key, input_data)
        return result
    
    # JIT compile if requested
    if use_jit:
        inference_fn = jax.jit(inference_fn)
    
    # Warmup (ensure JIT compilation completes)
    for _ in range(num_warmup):
        result = inference_fn()
        # Block until computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif isinstance(result, dict):
            for r in result.values():
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
    
    times_ms = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        result = inference_fn()
        # Block until computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif isinstance(result, dict):
            for r in result.values():
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    return TimingResult(
        times_ms=times_ms,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device,
        framework="jax",
    )


def time_inference_jax_haiku(
    model_fn: Callable,
    params: Any,
    state: Any,
    input_data: Any,
    rng_key: Optional[Any] = None,
    num_warmup: int = 100,
    num_runs: int = 1000,
    use_jit: bool = True,
    is_training: bool = False,
) -> TimingResult:
    """
    Time Haiku model inference (transform_with_state pattern).
    
    Args:
        model_fn: The transformed model's apply function
        params: Model parameters
        state: Model state (e.g., batch norm stats)
        input_data: Input array(s)
        rng_key: JAX PRNG key
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        use_jit: Whether to JIT compile
        is_training: Training mode flag
    
    Returns:
        TimingResult with timing measurements
    """
    import jax
    
    devices = jax.devices()
    device = "cuda" if any(d.platform == "gpu" for d in devices) else "cpu"
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    def inference_fn():
        output, _ = model_fn(params, state, rng_key, input_data, is_training=is_training)
        return output
    
    if use_jit:
        inference_fn = jax.jit(inference_fn)
    
    # Warmup
    for _ in range(num_warmup):
        result = inference_fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
    
    times_ms = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        result = inference_fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    return TimingResult(
        times_ms=times_ms,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device,
        framework="jax_haiku",
    )


def time_inference_onnx(
    session: Any,
    input_dict: dict[str, Any],
    output_names: Optional[list[str]] = None,
    num_warmup: int = 100,
    num_runs: int = 1000,
) -> TimingResult:
    """
    Time ONNX Runtime inference.
    
    Args:
        session: ONNXRuntime InferenceSession
        input_dict: Dict mapping input names to numpy arrays
        output_names: Optional list of output names (None = all outputs)
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
    
    Returns:
        TimingResult with timing measurements
    """
    # Determine device from session providers
    providers = session.get_providers()
    if 'CUDAExecutionProvider' in providers:
        device = "cuda"
    elif 'TensorrtExecutionProvider' in providers:
        device = "tensorrt"
    else:
        device = "cpu"
    
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(output_names, input_dict)
    
    times_ms = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(output_names, input_dict)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    return TimingResult(
        times_ms=times_ms,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device,
        framework="onnx",
    )


def time_inference_tensorflow(
    model: Any,
    input_data: Any,
    num_warmup: int = 100,
    num_runs: int = 1000,
) -> TimingResult:
    """
    Time TensorFlow/Keras model inference.
    
    Args:
        model: TensorFlow/Keras model
        input_data: Input tensor or numpy array
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
    
    Returns:
        TimingResult with timing measurements
    """
    import tensorflow as tf
    
    # Determine device
    gpus = tf.config.list_physical_devices('GPU')
    device = "cuda" if gpus else "cpu"
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(input_data, training=False)
    
    times_ms = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(input_data, training=False)
        # TensorFlow operations are synchronous by default when eager execution is enabled
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    return TimingResult(
        times_ms=times_ms,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device,
        framework="tensorflow",
    )


def time_generic(
    inference_fn: Callable[[], Any],
    num_warmup: int = 100,
    num_runs: int = 1000,
    device: str = "cpu",
    framework: str = "generic",
    sync_fn: Optional[Callable[[], None]] = None,
) -> TimingResult:
    """
    Generic timing function for any inference callable.
    
    Args:
        inference_fn: Callable that runs inference (no arguments)
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        device: Device string for metadata
        framework: Framework string for metadata
        sync_fn: Optional synchronization function to call after inference
    
    Returns:
        TimingResult with timing measurements
    """
    # Warmup
    for _ in range(num_warmup):
        _ = inference_fn()
        if sync_fn:
            sync_fn()
    
    times_ms = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = inference_fn()
        if sync_fn:
            sync_fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    return TimingResult(
        times_ms=times_ms,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device,
        framework=framework,
    )
