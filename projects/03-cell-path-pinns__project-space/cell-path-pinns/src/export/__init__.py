"""
ONNX Export for Geodesic Trajectory Models.

Integrates with shared/export_pipeline for deployment to:
- Desktop/server inference
- Browser (WebGL/WebNN)
- Mobile devices

Performance targets:
- Robot controllers: <10ms inference
- Game AI: <1ms inference
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from pathlib import Path

# Import from shared export pipeline
try:
    from shared.export_pipeline import (
        ExportConfig,
        TensorSpec,
        run_pipeline,
        PipelineResult,
    )
    HAS_EXPORT_PIPELINE = True
except ImportError:
    HAS_EXPORT_PIPELINE = False
    ExportConfig = None
    TensorSpec = None


def create_trajectory_export_config(
    spatial_dim: int = 2,
    context_dim: int = 4,
    n_time_steps: int = 100,
    project_name: str = "geodesic_trajectory",
    model_version: str = "1.0.0",
    targets: list = None,
    domain: str = "generic",
) -> 'ExportConfig':
    """
    Create export configuration for trajectory models.
    
    Args:
        spatial_dim: Output spatial dimension (2D or 3D)
        context_dim: Context vector dimension (typically start+end = 2*spatial_dim)
        n_time_steps: Number of time points in trajectory
        project_name: Project name for export metadata
        model_version: Semantic version string
        targets: Export targets ["desktop", "browser", "mobile"]
        domain: Domain name for metadata
        
    Returns:
        ExportConfig instance
    """
    if not HAS_EXPORT_PIPELINE:
        raise ImportError(
            "Export pipeline not available. Install with: pip install -e ../shared"
        )
    
    if targets is None:
        targets = ["desktop", "browser"]
    
    return ExportConfig(
        project_name=project_name,
        model_version=model_version,
        input_specs=[
            TensorSpec("time", [n_time_steps, 1], "float32"),
            TensorSpec("context", [n_time_steps, context_dim], "float32"),
        ],
        output_specs=[
            TensorSpec("trajectory", [n_time_steps, spatial_dim], "float32"),
        ],
        targets=targets,
        metadata={
            "spatial_dim": spatial_dim,
            "context_dim": context_dim,
            "n_time_steps": n_time_steps,
            "domain": domain,
            "framework": "geodesic_trajectory",
        }
    )


class TrajectoryExportWrapper(nn.Module):
    """
    Wrapper for exporting trajectory models to ONNX.
    
    Handles the specific input format needed for ONNX export:
    - Flattened context expanded to all time steps
    - Clean forward pass without optional arguments
    """
    
    def __init__(
        self,
        trajectory_net: nn.Module,
        spatial_dim: int = 2,
        context_dim: int = 4,
    ):
        """
        Initialize export wrapper.
        
        Args:
            trajectory_net: The trajectory network to export
            spatial_dim: Output dimension
            context_dim: Context dimension
        """
        super().__init__()
        self.trajectory_net = trajectory_net
        self.spatial_dim = spatial_dim
        self.context_dim = context_dim
    
    def forward(self, time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        
        Args:
            time: Time points [n_steps, 1]
            context: Context vector [n_steps, context_dim]
            
        Returns:
            Trajectory positions [n_steps, spatial_dim]
        """
        return self.trajectory_net(time, context)


class BoundaryConditionedExportWrapper(nn.Module):
    """
    Export wrapper for boundary-conditioned trajectory models.
    
    These models take start/end separately and guarantee exact endpoints.
    """
    
    def __init__(
        self,
        trajectory_net: nn.Module,
        spatial_dim: int = 2,
    ):
        super().__init__()
        self.trajectory_net = trajectory_net
        self.spatial_dim = spatial_dim
    
    def forward(
        self,
        time: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with explicit start/end.
        
        Args:
            time: [n_steps, 1]
            start: [spatial_dim] or [1, spatial_dim]
            end: [spatial_dim] or [1, spatial_dim]
        """
        if start.dim() == 1:
            start = start.unsqueeze(0)
        if end.dim() == 1:
            end = end.unsqueeze(0)
        
        return self.trajectory_net(time, start=start, end=end)


def export_trajectory_model(
    model: nn.Module,
    output_dir: str,
    spatial_dim: int = 2,
    context_dim: int = 4,
    n_time_steps: int = 100,
    project_name: str = "geodesic_trajectory",
    domain: str = "generic",
    targets: list = None,
    verbose: bool = True,
) -> 'PipelineResult':
    """
    Export trajectory model to ONNX.
    
    Uses the shared export pipeline for optimization and validation.
    
    Args:
        model: Trajectory model to export (TrajectoryPINN or similar)
        output_dir: Directory for export outputs
        spatial_dim: Spatial dimension
        context_dim: Context dimension
        n_time_steps: Trajectory length for export
        project_name: Project name
        domain: Domain name (robotics, biology, etc.)
        targets: Export targets
        verbose: Print progress
        
    Returns:
        PipelineResult with export paths and benchmarks
    """
    if not HAS_EXPORT_PIPELINE:
        raise ImportError("shared.export_pipeline not available")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Wrap model for export
    if hasattr(model, 'trajectory_net'):
        # High-level model with trajectory_net attribute
        traj_net = model.trajectory_net
    else:
        traj_net = model
    
    # Check if boundary-conditioned
    is_boundary_conditioned = hasattr(traj_net, 'forward') and 'start' in str(traj_net.forward.__code__.co_varnames)
    
    if is_boundary_conditioned:
        wrapper = BoundaryConditionedExportWrapper(traj_net, spatial_dim)
        input_specs = [
            TensorSpec("time", [n_time_steps, 1], "float32"),
            TensorSpec("start", [spatial_dim], "float32"),
            TensorSpec("end", [spatial_dim], "float32"),
        ]
    else:
        wrapper = TrajectoryExportWrapper(traj_net, spatial_dim, context_dim)
        input_specs = [
            TensorSpec("time", [n_time_steps, 1], "float32"),
            TensorSpec("context", [n_time_steps, context_dim], "float32"),
        ]
    
    # Create config
    config = ExportConfig(
        project_name=project_name,
        model_version="1.0.0",
        input_specs=input_specs,
        output_specs=[
            TensorSpec("trajectory", [n_time_steps, spatial_dim], "float32"),
        ],
        targets=targets or ["desktop", "browser"],
        metadata={
            "spatial_dim": spatial_dim,
            "context_dim": context_dim,
            "domain": domain,
            "boundary_conditioned": is_boundary_conditioned,
        }
    )
    
    # Run export pipeline
    result = run_pipeline(wrapper, config, output_dir=str(output_path))
    
    if verbose:
        print(f"\n=== Export Complete ===")
        print(f"ONNX path: {result.onnx_path}")
        print(f"Model size: {result.model_size_mb:.3f} MB")
        print(f"Inference time: {result.inference_time_ms:.3f} ms")
        print(f"Parameters: {sum(p.numel() for p in wrapper.parameters()):,}")
    
    return result


def quick_export_onnx(
    model: nn.Module,
    output_path: str,
    spatial_dim: int = 2,
    context_dim: int = 4,
    n_time_steps: int = 100,
    opset_version: int = 14,
) -> str:
    """
    Quick ONNX export without full pipeline (for testing).
    
    Args:
        model: Model to export
        output_path: Output .onnx file path
        spatial_dim: Spatial dimension
        context_dim: Context dimension
        n_time_steps: Number of time steps
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX file
    """
    import onnx
    
    # Get trajectory network
    if hasattr(model, 'trajectory_net'):
        traj_net = model.trajectory_net
    else:
        traj_net = model
    
    # Check for boundary conditioning
    is_boundary_conditioned = 'start' in str(getattr(traj_net.forward, '__code__', object()).co_varnames if hasattr(traj_net, 'forward') else '')
    
    # Wrap and create dummy inputs
    if is_boundary_conditioned:
        wrapper = BoundaryConditionedExportWrapper(traj_net, spatial_dim)
        dummy_inputs = (
            torch.randn(n_time_steps, 1),
            torch.randn(spatial_dim),
            torch.randn(spatial_dim),
        )
        input_names = ['time', 'start', 'end']
    else:
        wrapper = TrajectoryExportWrapper(traj_net, spatial_dim, context_dim)
        dummy_inputs = (
            torch.randn(n_time_steps, 1),
            torch.randn(n_time_steps, context_dim),
        )
        input_names = ['time', 'context']
    
    wrapper.eval()
    
    # Export
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=['trajectory'],
        opset_version=opset_version,
        dynamic_axes={
            'time': {0: 'n_steps'},
            'trajectory': {0: 'n_steps'},
        } if not is_boundary_conditioned else {
            'time': {0: 'n_steps'},
            'trajectory': {0: 'n_steps'},
        },
    )
    
    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    return output_path


def benchmark_onnx_inference(
    onnx_path: str,
    n_time_steps: int = 100,
    spatial_dim: int = 2,
    context_dim: int = 4,
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference time.
    
    Args:
        onnx_path: Path to ONNX model
        n_time_steps: Number of time steps
        spatial_dim: Spatial dimension
        context_dim: Context dimension
        n_runs: Number of benchmark runs
        warmup_runs: Warmup iterations
        
    Returns:
        Dictionary with timing statistics
    """
    import onnxruntime as ort
    import numpy as np
    import time
    
    # Create session
    sess = ort.InferenceSession(onnx_path)
    
    # Get input names
    input_names = [inp.name for inp in sess.get_inputs()]
    
    # Create dummy inputs based on input names
    if 'start' in input_names:
        # Boundary conditioned
        inputs = {
            'time': np.random.randn(n_time_steps, 1).astype(np.float32),
            'start': np.random.randn(spatial_dim).astype(np.float32),
            'end': np.random.randn(spatial_dim).astype(np.float32),
        }
    else:
        # Context conditioned
        inputs = {
            'time': np.random.randn(n_time_steps, 1).astype(np.float32),
            'context': np.random.randn(n_time_steps, context_dim).astype(np.float32),
        }
    
    # Warmup
    for _ in range(warmup_runs):
        _ = sess.run(None, inputs)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = sess.run(None, inputs)
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }
