"""
Run benchmarks for all projects.

Usage:
    python run_all.py --config config.yaml
    python run_all.py --quick
    python run_all.py --include P01 P09 P15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness.core import BenchmarkResult, BenchmarkConfig
from harness.hardware import get_hardware_info, check_dependencies
from harness.report import (
    save_aggregated_results,
    generate_markdown_tables,
    generate_html_dashboard,
)


def load_config(config_path: Optional[Path] = None) -> BenchmarkConfig:
    """Load configuration from YAML file or use defaults."""
    if config_path and config_path.exists():
        return BenchmarkConfig.from_yaml(config_path)
    return BenchmarkConfig()


def load_project_list(config_path: Path) -> dict:
    """Load project list from config YAML."""
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config.get("projects", {})


def run_single_project_subprocess(
    project_id: str,
    config_path: Path,
    quick: bool = False,
    device: str = "auto",
) -> Optional[BenchmarkResult]:
    """
    Run a single project benchmark in a subprocess.
    
    Used for JAX projects to isolate GPU memory.
    
    Args:
        project_id: Project identifier
        config_path: Path to config file
        quick: Use quick mode
        device: Device string
    
    Returns:
        BenchmarkResult or None if failed to parse
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_single.py"),
        "--project", project_id,
        "--config", str(config_path),
        "--device", device,
    ]
    
    if quick:
        cmd.append("--quick")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        # Try to find and parse the JSON result file
        output_dir = Path("results/raw")
        if output_dir.exists():
            # Find most recent result file for this project
            pattern = f"{project_id}_*.json"
            files = sorted(output_dir.glob(pattern), reverse=True)
            if files:
                return BenchmarkResult.load(files[0])
        
        # If no file found, create failed result from subprocess output
        if result.returncode != 0:
            return BenchmarkResult.create_failed(
                project_id=project_id,
                project_name=project_id,
                framework="unknown",
                error_message=f"Subprocess failed:\n{result.stderr}",
            )
        
        return None
        
    except subprocess.TimeoutExpired:
        return BenchmarkResult.create_failed(
            project_id=project_id,
            project_name=project_id,
            framework="unknown",
            error_message="Benchmark timed out after 10 minutes",
        )
    except Exception as e:
        return BenchmarkResult.create_failed(
            project_id=project_id,
            project_name=project_id,
            framework="unknown",
            error_message=str(e),
        )


def run_all_benchmarks(
    config: BenchmarkConfig,
    config_path: Path,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run benchmarks for all enabled projects.
    
    Args:
        config: Benchmark configuration
        config_path: Path to config YAML (for subprocess calls)
        verbose: Print progress
    
    Returns:
        List of BenchmarkResults
    """
    from run_single import run_benchmark, get_adapter
    
    results = []
    
    # Load project list from config
    projects = load_project_list(config_path)
    
    if not projects:
        # Default project list
        projects = {
            f"P{i:02d}": {"name": f"Project {i:02d}", "enabled": True}
            for i in range(1, 18)
        }
    
    # Filter projects
    if config.include_projects:
        projects = {
            k: v for k, v in projects.items()
            if k in config.include_projects
        }
    
    for pid in config.exclude_projects:
        projects.pop(pid, None)
    
    # Filter by enabled flag
    enabled_projects = {
        k: v for k, v in projects.items()
        if v.get("enabled", True)
    }
    
    total = len(enabled_projects)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PINN BENCHMARK HARNESS")
        print(f"{'='*70}")
        print(f"Projects to benchmark: {total}")
        print(f"Device: {config.device}")
        print(f"Quick mode: {config.quick_mode}")
        print(f"Warmup runs: {config.effective_warmup}")
        print(f"Timed runs: {config.effective_runs}")
        print(f"{'='*70}\n")
    
    # Group by framework for efficient execution
    pytorch_projects = []
    jax_projects = []
    other_projects = []
    
    for pid, pinfo in enabled_projects.items():
        framework = pinfo.get("framework", "pytorch").lower()
        if framework in ("jax", "jax_haiku"):
            jax_projects.append(pid)
        elif framework == "pytorch":
            pytorch_projects.append(pid)
        else:
            other_projects.append(pid)
    
    start_time = time.time()
    completed = 0
    
    # Run PyTorch projects first (in-process)
    for pid in pytorch_projects:
        completed += 1
        if verbose:
            print(f"\n[{completed}/{total}] Benchmarking {pid}...")
        
        try:
            result = run_benchmark(pid, config, verbose=verbose)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed: {e}")
            results.append(BenchmarkResult.create_failed(
                project_id=pid,
                project_name=enabled_projects[pid].get("name", pid),
                framework="pytorch",
                error_message=str(e),
            ))
    
    # Clear GPU memory before JAX projects
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    # Run JAX projects (in subprocess if isolation enabled)
    for pid in jax_projects:
        completed += 1
        if verbose:
            print(f"\n[{completed}/{total}] Benchmarking {pid}...")
        
        if config.isolate_jax:
            if verbose:
                print(f"  (Running in subprocess for JAX isolation)")
            result = run_single_project_subprocess(
                pid, config_path, config.quick_mode, config.device
            )
            if result:
                results.append(result)
            else:
                results.append(BenchmarkResult.create_failed(
                    project_id=pid,
                    project_name=enabled_projects[pid].get("name", pid),
                    framework="jax",
                    error_message="Failed to retrieve subprocess result",
                ))
        else:
            try:
                result = run_benchmark(pid, config, verbose=verbose)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Failed: {e}")
                results.append(BenchmarkResult.create_failed(
                    project_id=pid,
                    project_name=enabled_projects[pid].get("name", pid),
                    framework="jax",
                    error_message=str(e),
                ))
    
    # Run other projects (ONNX, TensorFlow)
    for pid in other_projects:
        completed += 1
        if verbose:
            print(f"\n[{completed}/{total}] Benchmarking {pid}...")
        
        try:
            result = run_benchmark(pid, config, verbose=verbose)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed: {e}")
            results.append(BenchmarkResult.create_failed(
                project_id=pid,
                project_name=enabled_projects[pid].get("name", pid),
                framework="unknown",
                error_message=str(e),
            ))
    
    elapsed = time.time() - start_time
    
    if verbose:
        successful = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.1f} seconds")
        print(f"Successful: {successful}/{total}")
        print(f"Failed: {failed}/{total}")
        print(f"{'='*70}\n")
    
    return results


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a summary table to console."""
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'ID':<8} {'Project':<25} {'Params':<10} {'Inference':<15} {'L2 Error':<12} {'Status':<10}")
    print("-"*100)
    
    for r in sorted(results, key=lambda x: x.project_id):
        pid = r.project_id
        name = r.project_name[:23]
        
        if r.status == "success":
            params = f"{r.model.parameter_count/1000:.0f}K" if r.model.parameter_count < 1e6 else f"{r.model.parameter_count/1e6:.1f}M"
            
            if r.inference_time_gpu:
                inference = f"{r.inference_time_gpu.mean_ms:.2f} ms"
            elif r.inference_time_cpu:
                inference = f"{r.inference_time_cpu.mean_ms:.2f} ms"
            else:
                inference = "N/A"
            
            l2_err = f"{r.accuracy.l2_relative_error:.6f}" if r.accuracy.l2_relative_error else "N/A"
            status = "✅ OK"
        else:
            params = "N/A"
            inference = "N/A"
            l2_err = "N/A"
            status = "❌ FAIL"
        
        print(f"{pid:<8} {name:<25} {params:<10} {inference:<15} {l2_err:<12} {status:<10}")
    
    print("="*100)


def main():
    """Main entry point for running all benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for all PINN projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all.py --config config.yaml
    python run_all.py --quick
    python run_all.py --include P01 P09 P15 --device cuda
        """,
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
        "--include",
        nargs="+",
        metavar="PROJECT",
        help="Only benchmark these projects (e.g., --include P01 P09)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="PROJECT",
        default=[],
        help="Exclude these projects from benchmark",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for results (default: from config)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip HTML dashboard generation",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip markdown table generation",
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
    if args.include:
        config.include_projects = args.include
    if args.exclude:
        config.exclude_projects = args.exclude
    if args.output:
        config.output_dir = args.output
    if args.no_dashboard:
        config.generate_dashboard = False
    if args.no_tables:
        config.generate_tables = False
    
    # Check dependencies
    if not args.quiet:
        print("\nChecking dependencies...")
        deps = check_dependencies()
        for name, available in deps.items():
            if "torch" in name or "jax" in name or "tensorflow" in name or "onnx" in name:
                status = "✅" if available else "❌"
                print(f"  {status} {name}")
    
    # Run benchmarks
    results = run_all_benchmarks(config, args.config, verbose=not args.quiet)
    
    # Save results
    output_dir = Path(config.output_dir)
    
    # Save aggregated JSON
    if config.save_aggregated:
        agg_path = save_aggregated_results(results, output_dir / "aggregated")
        if not args.quiet:
            print(f"\nAggregated results saved to: {agg_path}")
    
    # Generate markdown tables
    if config.generate_tables:
        table_paths = generate_markdown_tables(results, output_dir / "tables")
        if not args.quiet:
            print(f"Markdown tables saved to: {output_dir / 'tables'}")
    
    # Generate HTML dashboard
    if config.generate_dashboard:
        dashboard_path = generate_html_dashboard(results, output_dir / "dashboard")
        if not args.quiet:
            print(f"HTML dashboard saved to: {dashboard_path}")
    
    # Print summary
    if not args.quiet:
        print_summary_table(results)
    
    # Return exit code based on results
    successful = sum(1 for r in results if r.status == "success")
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
