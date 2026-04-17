"""
Benchmark a single project.

Usage:
    python run_single.py --project P09 --config config.yaml
    python run_single.py --project P09 --quick
    python run_single.py --project P15-onnx --device cpu
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness.core import BenchmarkResult, BenchmarkConfig
from harness.hardware import get_hardware_info, get_software_info, get_git_hash, check_dependencies
from harness.timing import time_inference_pytorch, time_inference_jax, time_inference_onnx
from harness.memory import measure_memory_pytorch, measure_memory_jax, measure_memory_onnx
from harness.metrics import compute_rmse, compute_nrmse, compute_l2_relative_error
from harness.report import save_json_result, generate_markdown_tables


def load_config(config_path: Optional[Path] = None) -> BenchmarkConfig:
    """Load configuration from YAML file or use defaults."""
    if config_path and config_path.exists():
        return BenchmarkConfig.from_yaml(config_path)
    return BenchmarkConfig()


def get_adapter(project_id: str, config: BenchmarkConfig):
    """
    Get the appropriate adapter for a project.
    
    Dynamically imports the adapter module based on project ID.
    """
    # Normalize project ID
    pid = project_id.upper().replace("-", "_").replace("P", "p")
    
    # Map project IDs to adapter module names
    adapter_map = {
        "p01": "p01_geopinn_manifold",
        "p02": "p02_wavepinn_complex",
        "p03": "p03_cell_path_pinns",
        "p04": "p04_clothgeom_nif",
        "p05": "p05_clothgnn",
        "p06": "p06_coastflow_gnn",
        "p07": "p07_geom_inr_motion",
        "p08": "p08_hgnn_clothdyn",
        "p09": "p09_hgnn_nif_cloth",
        "p09_anim": "p09_hgnn_nif_cloth",  # Same module, different config
        "p10": "p10_maxwell_pinn_nif",
        "p11": "p11_nif_cloth3d",
        "p12": "p12_nif_cloth4d_temporal",
        "p13": "p13_nif_cloth4d",
        "p14": "p14_pegnn_deform",
        "p15": "p15_pinn_lite_foil",
        "p15_onnx": "p15_pinn_lite_foil",  # Same module, ONNX variant
        "p16": "p16_surfpinn",
        "p17": "p17_wavepinn_scalar",
    }
    
    module_name = adapter_map.get(pid.lower())
    if not module_name:
        raise ValueError(f"Unknown project ID: {project_id}. Available: {list(adapter_map.keys())}")
    
    try:
        module = __import__(f"adapters.{module_name}", fromlist=["get_adapter"])
        return module.get_adapter(project_id, config)
    except ImportError as e:
        raise ImportError(f"Adapter module not found for {project_id}: {e}")


def run_benchmark(
    project_id: str,
    config: BenchmarkConfig,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run benchmark for a single project.
    
    Args:
        project_id: Project identifier (e.g., "P09", "P15-onnx")
        config: Benchmark configuration
        verbose: Print progress messages
    
    Returns:
        BenchmarkResult with all metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {project_id}")
        print(f"{'='*60}")
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    
    try:
        # Try to import torch and set seed if available
        import torch
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
    except ImportError:
        pass
    
    try:
        # Load adapter
        if verbose:
            print(f"  Loading adapter...")
        adapter = get_adapter(project_id, config)
        project_info = adapter.get_project_info()
        
        if verbose:
            print(f"  Project: {project_info.project_name}")
            print(f"  Framework: {project_info.framework}")
            print(f"  Path: {project_info.project_path}")
        
        # Determine device
        device = config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        if verbose:
            print(f"  Device: {device}")
        
        # Load model
        if verbose:
            print(f"  Loading model...")
        model = adapter.load_model(device=device)
        
        # Get model info
        model_info = adapter.get_model_info(model)
        if verbose:
            print(f"  Parameters: {model_info.parameter_count:,}")
            print(f"  Model size: {model_info.model_size_mb:.2f} MB")
        
        # Prepare test input
        if verbose:
            print(f"  Preparing test input...")
        input_data = adapter.prepare_test_input(device=device, batch_size=1)
        
        # Run timing benchmark
        if verbose:
            print(f"  Running timing benchmark ({config.effective_warmup} warmup, {config.effective_runs} timed)...")
        
        framework = project_info.framework.lower()
        
        if framework == "pytorch":
            timing_result = time_inference_pytorch(
                model, input_data,
                num_warmup=config.effective_warmup,
                num_runs=config.effective_runs,
                device=device,
            )
        elif framework in ("jax", "jax_haiku"):
            timing_result = time_inference_jax(
                model[0], model[1], input_data,
                num_warmup=config.effective_warmup,
                num_runs=config.effective_runs,
            )
        elif framework == "onnx":
            timing_result = time_inference_onnx(
                model, input_data,
                num_warmup=config.effective_warmup,
                num_runs=config.effective_runs,
            )
        else:
            # Generic timing
            from harness.timing import time_generic
            timing_result = time_generic(
                lambda: adapter.run_inference(model, input_data),
                num_warmup=config.effective_warmup,
                num_runs=config.effective_runs,
                device=device,
                framework=framework,
            )
        
        if verbose:
            print(f"  Inference time: {timing_result.mean_ms:.2f} ± {timing_result.std_ms:.2f} ms")
            print(f"  P95: {timing_result.p95_ms:.2f} ms, P99: {timing_result.p99_ms:.2f} ms")
        
        # Memory measurement
        if verbose:
            print(f"  Measuring memory...")
        
        if framework == "pytorch":
            memory_result = measure_memory_pytorch(model, input_data, device=device)
        elif framework in ("jax", "jax_haiku"):
            memory_result = measure_memory_jax(model[0], model[1], input_data)
        elif framework == "onnx":
            memory_result = measure_memory_onnx(model, input_data)
        else:
            from harness.memory import MemoryResult
            memory_result = MemoryResult(peak_allocated_gb=0, device=device, framework=framework)
        
        if verbose:
            print(f"  Peak memory: {memory_result.peak_allocated_gb:.4f} GB")
        
        # Compute accuracy metrics
        if verbose:
            print(f"  Computing accuracy metrics...")
        
        is_synthetic = adapter.is_synthetic_fallback()
        
        try:
            predictions = adapter.run_inference(model, input_data)
            references = adapter.get_reference_data()
            
            # Standard metrics
            rmse = compute_rmse(predictions, references)
            nrmse = compute_nrmse(predictions, references)
            l2_error = compute_l2_relative_error(predictions, references)
            
            # Domain-specific metrics
            domain_metrics = adapter.compute_domain_metrics(predictions, references)
            
            if verbose:
                print(f"  RMSE: {rmse:.6f}")
                print(f"  NRMSE: {nrmse:.6f}")
                print(f"  L2 relative error: {l2_error:.6f}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not compute accuracy metrics: {e}")
            rmse = None
            nrmse = None
            l2_error = None
            domain_metrics = {}
            is_synthetic = True
        
        # Get training info
        training_info = adapter.get_training_info()
        
        # Get hardware/software info
        hardware_info = get_hardware_info()
        software_info = get_software_info(framework)
        git_hash = get_git_hash(str(project_info.project_path))
        
        # Calculate throughput
        throughput_fps = 1000.0 / timing_result.mean_ms if timing_result.mean_ms > 0 else 0
        
        # Build result
        result = BenchmarkResult(
            project_id=project_info.project_id,
            project_name=project_info.project_name,
            framework=project_info.framework,
            timestamp=datetime.utcnow().isoformat() + "Z",
            git_hash=git_hash,
            hardware=hardware_info,
            software=software_info,
            model=model_info,
            inference_time_gpu=timing_result.to_timing_stats() if device.startswith("cuda") else None,
            inference_time_cpu=timing_result.to_timing_stats() if device == "cpu" else None,
            throughput_fps=throughput_fps,
            memory=memory_result.to_memory_stats(),
            training=training_info,
            test_dataset=adapter.get_test_dataset_description(),
            test_data_synthetic_fallback=is_synthetic,
            status="success",
            notes="",
        )
        
        # Set accuracy metrics
        result.accuracy.rmse = rmse
        result.accuracy.nrmse = nrmse
        result.accuracy.l2_relative_error = l2_error
        result.domain_metrics = domain_metrics
        
        # Cleanup
        adapter.cleanup(model)
        
        if verbose:
            print(f"\n✅ Benchmark complete for {project_id}")
            print(f"   Throughput: {throughput_fps:.0f} FPS")
        
        return result
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        if verbose:
            print(f"\n❌ Benchmark failed for {project_id}")
            print(f"   Error: {e}")
        
        return BenchmarkResult.create_failed(
            project_id=project_id,
            project_name=project_id,
            framework="unknown",
            error_message=error_msg,
        )


def main():
    """Main entry point for single project benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark a single PINN project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_single.py --project P09
    python run_single.py --project P15-onnx --device cpu
    python run_single.py --project P01 --quick --config config.yaml
        """,
    )
    
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project ID to benchmark (e.g., P01, P09, P15-onnx)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Use quick mode (fewer runs for faster iteration)",
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for results (default: from config)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.quick:
        config.quick_mode = True
    if args.device != "auto":
        config.device = args.device
    if args.output:
        config.output_dir = args.output
    
    # Check dependencies
    if not args.quiet:
        deps = check_dependencies()
        print("Available frameworks:")
        for name, available in deps.items():
            if "torch" in name or "jax" in name or "tensorflow" in name or "onnx" in name:
                status = "✅" if available else "❌"
                print(f"  {status} {name}")
    
    # Run benchmark
    result = run_benchmark(args.project, config, verbose=not args.quiet)
    
    # Save result
    output_dir = Path(config.output_dir) / "raw"
    output_path = save_json_result(result, output_dir)
    
    if not args.quiet:
        print(f"\nResult saved to: {output_path}")
    
    # Print summary
    if result.status == "success":
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY: {result.project_name}")
        print(f"{'='*60}")
        print(f"Framework:      {result.framework}")
        print(f"Parameters:     {result.model.parameter_count:,}")
        print(f"Model size:     {result.model.model_size_mb:.2f} MB")
        if result.inference_time_gpu:
            print(f"Inference (GPU): {result.inference_time_gpu.mean_ms:.2f} ± {result.inference_time_gpu.std_ms:.2f} ms")
        if result.inference_time_cpu:
            print(f"Inference (CPU): {result.inference_time_cpu.mean_ms:.2f} ± {result.inference_time_cpu.std_ms:.2f} ms")
        print(f"Throughput:     {result.throughput_fps:.0f} FPS")
        if result.accuracy.l2_relative_error:
            print(f"L2 Rel. Error:  {result.accuracy.l2_relative_error:.6f}")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n❌ Benchmark failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
