"""
Memory profiling utilities for GPU and CPU.

Measures peak memory usage, model size, and inference memory footprint.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class MemoryResult:
    """Memory measurement results."""
    peak_allocated_gb: float
    peak_reserved_gb: Optional[float] = None
    model_size_gb: float = 0.0
    inference_delta_gb: Optional[float] = None
    device: str = "unknown"
    framework: str = "unknown"
    
    def to_memory_stats(self):
        """Convert to MemoryStats from core module."""
        from .core import MemoryStats
        return MemoryStats(
            peak_gpu_gb=self.peak_allocated_gb if self.device != "cpu" else None,
            model_size_gb=self.model_size_gb,
            inference_peak_gb=self.peak_allocated_gb,
            peak_cpu_rss_gb=self.peak_allocated_gb if self.device == "cpu" else None,
        )


def measure_memory_pytorch(
    model: Any,
    input_data: Any,
    device: str = "cuda",
) -> MemoryResult:
    """
    Measure memory usage for PyTorch model inference.
    
    Args:
        model: PyTorch model (on device, eval mode)
        input_data: Input tensor(s)
        device: "cuda" or "cpu"
    
    Returns:
        MemoryResult with peak memory measurements
    """
    import torch
    
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    
    # Calculate model size
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_gb = (param_bytes + buffer_bytes) / (1024**3)
    
    if is_cuda:
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            if isinstance(input_data, dict):
                _ = model(**input_data)
            elif isinstance(input_data, (tuple, list)):
                _ = model(*input_data)
            else:
                _ = model(input_data)
        
        torch.cuda.synchronize()
        
        peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        
        return MemoryResult(
            peak_allocated_gb=peak_allocated,
            peak_reserved_gb=peak_reserved,
            model_size_gb=model_size_gb,
            device=device,
            framework="pytorch",
        )
    else:
        # CPU memory measurement using psutil
        return measure_cpu_memory(
            lambda: _run_pytorch_inference(model, input_data),
            model_size_gb=model_size_gb,
            framework="pytorch",
        )


def _run_pytorch_inference(model: Any, input_data: Any) -> Any:
    """Helper to run PyTorch inference."""
    import torch
    model.eval()
    with torch.no_grad():
        if isinstance(input_data, dict):
            return model(**input_data)
        elif isinstance(input_data, (tuple, list)):
            return model(*input_data)
        else:
            return model(input_data)


def measure_memory_jax(
    apply_fn: Callable,
    params: Any,
    input_data: Any,
    rng_key: Optional[Any] = None,
) -> MemoryResult:
    """
    Measure memory usage for JAX model inference.
    
    Note: JAX memory measurement is less straightforward than PyTorch.
    We use device memory stats if available, otherwise fall back to nvidia-smi.
    
    Args:
        apply_fn: Model apply function
        params: Model parameters
        input_data: Input array
        rng_key: Optional PRNG key
    
    Returns:
        MemoryResult with memory measurements
    """
    import jax
    import jax.numpy as jnp
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    # Determine device
    devices = jax.devices()
    is_gpu = any(d.platform == "gpu" for d in devices)
    device = "cuda" if is_gpu else "cpu"
    
    # Calculate model size (approximate)
    def count_params(pytree):
        leaves = jax.tree_util.tree_leaves(pytree)
        return sum(x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size'))
    
    model_size_bytes = count_params(params)
    model_size_gb = model_size_bytes / (1024**3)
    
    if is_gpu:
        # Try to get memory stats from JAX device
        try:
            device_obj = jax.devices('gpu')[0]
            if hasattr(device_obj, 'memory_stats'):
                stats_before = device_obj.memory_stats()
                
                # Run inference
                if isinstance(input_data, dict):
                    result = apply_fn(params, rng_key, **input_data)
                else:
                    result = apply_fn(params, rng_key, input_data)
                
                if hasattr(result, 'block_until_ready'):
                    result.block_until_ready()
                
                stats_after = device_obj.memory_stats()
                
                peak_bytes = stats_after.get('peak_bytes_in_use', 0)
                peak_gb = peak_bytes / (1024**3)
                
                return MemoryResult(
                    peak_allocated_gb=peak_gb,
                    model_size_gb=model_size_gb,
                    device=device,
                    framework="jax",
                )
        except Exception:
            pass
        
        # Fallback to nvidia-smi
        peak_gb = _get_nvidia_smi_memory()
        if peak_gb is not None:
            return MemoryResult(
                peak_allocated_gb=peak_gb,
                model_size_gb=model_size_gb,
                device=device,
                framework="jax",
            )
    
    # CPU fallback
    return measure_cpu_memory(
        lambda: apply_fn(params, rng_key, input_data),
        model_size_gb=model_size_gb,
        framework="jax",
    )


def measure_memory_onnx(
    session: Any,
    input_dict: dict[str, Any],
) -> MemoryResult:
    """
    Measure memory usage for ONNX Runtime inference.
    
    Args:
        session: ONNX Runtime InferenceSession
        input_dict: Input dictionary
    
    Returns:
        MemoryResult with memory measurements
    """
    # Determine device
    providers = session.get_providers()
    is_gpu = 'CUDAExecutionProvider' in providers
    device = "cuda" if is_gpu else "cpu"
    
    # Model size from ONNX graph (approximate)
    model_size_gb = 0.0  # Would need to load the model file to calculate
    
    if is_gpu:
        peak_gb = _get_nvidia_smi_memory()
        
        # Run inference
        _ = session.run(None, input_dict)
        
        peak_after = _get_nvidia_smi_memory()
        
        return MemoryResult(
            peak_allocated_gb=peak_after if peak_after else 0.0,
            model_size_gb=model_size_gb,
            device=device,
            framework="onnx",
        )
    else:
        return measure_cpu_memory(
            lambda: session.run(None, input_dict),
            model_size_gb=model_size_gb,
            framework="onnx",
        )


def measure_cpu_memory(
    inference_fn: Callable[[], Any],
    model_size_gb: float = 0.0,
    framework: str = "unknown",
) -> MemoryResult:
    """
    Measure CPU memory usage using psutil.
    
    Args:
        inference_fn: Callable that runs inference
        model_size_gb: Pre-calculated model size
        framework: Framework name for metadata
    
    Returns:
        MemoryResult with CPU memory measurements
    """
    try:
        import psutil
        
        process = psutil.Process(os.getpid())
        
        # Get memory before
        mem_before = process.memory_info().rss
        
        # Run inference
        _ = inference_fn()
        
        # Get memory after
        mem_after = process.memory_info().rss
        peak_rss = process.memory_info().rss
        
        return MemoryResult(
            peak_allocated_gb=peak_rss / (1024**3),
            inference_delta_gb=(mem_after - mem_before) / (1024**3),
            model_size_gb=model_size_gb,
            device="cpu",
            framework=framework,
        )
    except ImportError:
        # psutil not available
        return MemoryResult(
            peak_allocated_gb=0.0,
            model_size_gb=model_size_gb,
            device="cpu",
            framework=framework,
        )


def _get_nvidia_smi_memory() -> Optional[float]:
    """
    Get current GPU memory usage via nvidia-smi.
    
    Returns:
        Memory usage in GB, or None if nvidia-smi is not available
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Get first GPU's memory (in MiB)
            memory_mib = float(result.stdout.strip().split('\n')[0])
            return memory_mib / 1024  # Convert to GB
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return None


def get_model_size_pytorch(model: Any) -> dict:
    """
    Get detailed model size information for PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict with parameter counts and memory sizes
    """
    import torch
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    buffer_params = sum(b.numel() for b in model.buffers())
    
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "buffer_parameters": buffer_params,
        "total_parameters": trainable_params + non_trainable_params,
        "parameter_memory_mb": param_bytes / (1024**2),
        "buffer_memory_mb": buffer_bytes / (1024**2),
        "total_memory_mb": (param_bytes + buffer_bytes) / (1024**2),
    }


def get_model_size_jax(params: Any) -> dict:
    """
    Get model size information for JAX parameters.
    
    Args:
        params: JAX parameter pytree
    
    Returns:
        Dict with parameter counts and memory sizes
    """
    import jax
    
    leaves = jax.tree_util.tree_leaves(params)
    
    total_params = sum(x.size for x in leaves if hasattr(x, 'size'))
    total_bytes = sum(x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size'))
    
    return {
        "trainable_parameters": total_params,
        "non_trainable_parameters": 0,
        "total_parameters": total_params,
        "total_memory_mb": total_bytes / (1024**2),
    }


def get_onnx_model_size(model_path: str) -> dict:
    """
    Get ONNX model file size.
    
    Args:
        model_path: Path to ONNX model file
    
    Returns:
        Dict with model size information
    """
    import os
    
    file_size_bytes = os.path.getsize(model_path)
    
    # Try to count parameters from ONNX graph
    total_params = 0
    try:
        import onnx
        model = onnx.load(model_path)
        for initializer in model.graph.initializer:
            total_params += int(np.prod(initializer.dims))
    except Exception:
        pass
    
    return {
        "file_size_mb": file_size_bytes / (1024**2),
        "total_parameters": total_params,
    }


# Import numpy for get_onnx_model_size
import numpy as np
