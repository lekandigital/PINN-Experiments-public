"""
Latency Benchmark for PINN-Lite-Foil
Measures inference time across different platforms and configurations.

Target: <1ms inference on edge devices

Usage:
    python latency_benchmark.py --model models/onnx/student.onnx --iterations 1000
"""

import argparse
import json
import sys
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)

try:
    import psutil
except ImportError:
    psutil = None


def get_system_info() -> Dict:
    """Collect system information."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'onnxruntime_version': ort.__version__
    }
    
    if psutil:
        info['cpu_count'] = psutil.cpu_count(logical=False)
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    
    return info


def get_gpu_info() -> Optional[Dict]:
    """Get GPU information if available."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'name': parts[0],
                'memory_total_mb': int(parts[1]),
                'memory_free_mb': int(parts[2]),
                'power_draw_w': float(parts[3])
            }
    except Exception:
        pass
    return None


def benchmark_inference(
    model_path: str,
    n_iterations: int = 1000,
    batch_sizes: List[int] = [1, 4, 8, 16],
    providers: Optional[List[str]] = None,
    num_threads: int = 1,
    warmup_iterations: int = 50
) -> Dict:
    """
    Run comprehensive latency benchmark.
    
    Args:
        model_path: Path to ONNX model
        n_iterations: Number of iterations per configuration
        batch_sizes: Batch sizes to test
        providers: Execution providers to test
        num_threads: Number of threads
        warmup_iterations: Warmup iterations before timing
    
    Returns:
        Benchmark results dictionary
    """
    print("\n" + "="*60)
    print("PINN-Lite-Foil Latency Benchmark")
    print("="*60)
    
    # System info
    system_info = get_system_info()
    gpu_info = get_gpu_info()
    
    print(f"\nSystem: {system_info['platform']}")
    print(f"CPU: {system_info['processor']}")
    if gpu_info:
        print(f"GPU: {gpu_info['name']} ({gpu_info['memory_total_mb']} MB)")
    
    # Default providers
    if providers is None:
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
    
    print(f"\nProviders to test: {providers}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iterations per config: {n_iterations}")
    
    results = {
        'system_info': system_info,
        'gpu_info': gpu_info,
        'model_path': model_path,
        'n_iterations': n_iterations,
        'num_threads': num_threads,
        'benchmarks': []
    }
    
    # Get model size
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    results['model_size_mb'] = model_size_mb
    print(f"\nModel size: {model_size_mb:.2f} MB")
    
    # Benchmark each configuration
    for provider in providers:
        if provider not in ort.get_available_providers():
            print(f"\n  Skipping {provider} (not available)")
            continue
        
        print(f"\n{'='*40}")
        print(f"Testing: {provider}")
        print(f"{'='*40}")
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.intra_op_num_threads = num_threads
        
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[provider]
            )
        except Exception as e:
            print(f"  Error creating session: {e}")
            continue
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            # Generate random input
            inputs = np.random.randn(batch_size, 3).astype(np.float32)
            
            # Warmup
            print(f"    Warming up ({warmup_iterations} iterations)...")
            for _ in range(warmup_iterations):
                session.run([output_name], {input_name: inputs})
            
            # Benchmark
            print(f"    Benchmarking ({n_iterations} iterations)...")
            times = []
            
            # Power monitoring (if GPU)
            power_readings = []
            
            for i in range(n_iterations):
                # Measure inference time
                start = time.perf_counter()
                session.run([output_name], {input_name: inputs})
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
                
                # Sample power every 100 iterations
                if gpu_info and i % 100 == 0:
                    power_info = get_gpu_info()
                    if power_info:
                        power_readings.append(power_info['power_draw_w'])
            
            times = np.array(times)
            
            # Statistics
            stats = {
                'provider': provider,
                'batch_size': batch_size,
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'p50_ms': float(np.percentile(times, 50)),
                'p95_ms': float(np.percentile(times, 95)),
                'p99_ms': float(np.percentile(times, 99)),
                'throughput_per_sec': float(batch_size * 1000 / np.mean(times))
            }
            
            if power_readings:
                stats['avg_power_w'] = float(np.mean(power_readings))
                stats['energy_per_inference_mj'] = float(
                    np.mean(power_readings) * np.mean(times)  # mW * ms = mJ
                )
            
            results['benchmarks'].append(stats)
            
            # Print results
            print(f"    Results:")
            print(f"      Mean:       {stats['mean_ms']:.3f} ms")
            print(f"      Std:        {stats['std_ms']:.3f} ms")
            print(f"      Min:        {stats['min_ms']:.3f} ms")
            print(f"      P50:        {stats['p50_ms']:.3f} ms")
            print(f"      P95:        {stats['p95_ms']:.3f} ms")
            print(f"      P99:        {stats['p99_ms']:.3f} ms")
            print(f"      Throughput: {stats['throughput_per_sec']:.1f} inferences/sec")
            
            if 'avg_power_w' in stats:
                print(f"      Power:      {stats['avg_power_w']:.1f} W")
                print(f"      Energy:     {stats['energy_per_inference_mj']:.3f} mJ/inference")
            
            # Check target
            if stats['mean_ms'] < 1.0 and batch_size == 1:
                print(f"      ✓ Sub-millisecond inference achieved!")
    
    return results


def generate_benchmark_plots(results: Dict, output_dir: Path) -> None:
    """Generate benchmark visualization plots."""
    
    benchmarks = results['benchmarks']
    
    if not benchmarks:
        return
    
    # Group by provider
    providers = list(set(b['provider'] for b in benchmarks))
    batch_sizes = sorted(set(b['batch_size'] for b in benchmarks))
    
    # 1. Latency vs Batch Size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for provider in providers:
        provider_data = [b for b in benchmarks if b['provider'] == provider]
        provider_data.sort(key=lambda x: x['batch_size'])
        
        bs = [b['batch_size'] for b in provider_data]
        means = [b['mean_ms'] for b in provider_data]
        stds = [b['std_ms'] for b in provider_data]
        
        ax.errorbar(bs, means, yerr=stds, marker='o', label=provider, capsize=5)
    
    ax.axhline(y=1.0, color='r', linestyle='--', label='1ms target')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_vs_batch_size.png', dpi=150)
    plt.close()
    
    # 2. Throughput comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(batch_sizes))
    width = 0.8 / len(providers)
    
    for i, provider in enumerate(providers):
        provider_data = [b for b in benchmarks if b['provider'] == provider]
        provider_data.sort(key=lambda x: x['batch_size'])
        
        throughputs = [b['throughput_per_sec'] for b in provider_data]
        
        ax.bar(x + i * width, throughputs, width, label=provider)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (inferences/sec)')
    ax.set_title('Inference Throughput')
    ax.set_xticks(x + width * (len(providers) - 1) / 2)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=150)
    plt.close()
    
    # 3. Latency distribution (batch size = 1)
    batch1_data = [b for b in benchmarks if b['batch_size'] == 1]
    
    if batch1_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = []
        labels = []
        
        for b in batch1_data:
            # We don't have raw times, so create approximate histogram
            # from percentiles
            labels.append(b['provider'].replace('ExecutionProvider', ''))
        
        # Create bar chart of percentiles
        x = np.arange(len(batch1_data))
        width = 0.15
        
        metrics = ['min_ms', 'p50_ms', 'mean_ms', 'p95_ms', 'p99_ms']
        metric_labels = ['Min', 'P50', 'Mean', 'P95', 'P99']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [b[metric] for b in batch1_data]
            ax.bar(x + i * width, values, width, label=label)
        
        ax.axhline(y=1.0, color='r', linestyle='--', label='1ms target')
        ax.set_xlabel('Provider')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Distribution (Batch Size = 1)')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_distribution.png', dpi=150)
        plt.close()
    
    print(f"  Generated plots in {output_dir}")


def run_full_benchmark(
    model_path: str,
    output_dir: str,
    n_iterations: int = 1000,
    batch_sizes: List[int] = [1, 4, 8, 16],
    num_threads: int = 1
) -> Dict:
    """
    Run full benchmark suite and save results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results = benchmark_inference(
        model_path=model_path,
        n_iterations=n_iterations,
        batch_sizes=batch_sizes,
        num_threads=num_threads
    )
    
    # Generate plots
    generate_benchmark_plots(results, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Find best configuration
    batch1_results = [b for b in results['benchmarks'] if b['batch_size'] == 1]
    
    if batch1_results:
        best = min(batch1_results, key=lambda x: x['mean_ms'])
        
        print(f"\nBest single-inference latency:")
        print(f"  Provider:   {best['provider']}")
        print(f"  Mean:       {best['mean_ms']:.3f} ms")
        print(f"  P99:        {best['p99_ms']:.3f} ms")
        
        if best['mean_ms'] < 1.0:
            print(f"\n  ✓ TARGET ACHIEVED: Sub-millisecond inference!")
        else:
            print(f"\n  ✗ Target not met (need <1ms, got {best['mean_ms']:.3f}ms)")
    
    # Save results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Generate CSV summary
    csv_file = output_dir / 'benchmark_summary.csv'
    with open(csv_file, 'w') as f:
        f.write("provider,batch_size,mean_ms,std_ms,min_ms,p50_ms,p95_ms,p99_ms,throughput\n")
        for b in results['benchmarks']:
            f.write(f"{b['provider']},{b['batch_size']},"
                    f"{b['mean_ms']:.4f},{b['std_ms']:.4f},"
                    f"{b['min_ms']:.4f},{b['p50_ms']:.4f},"
                    f"{b['p95_ms']:.4f},{b['p99_ms']:.4f},"
                    f"{b['throughput_per_sec']:.1f}\n")
    
    print(f"✓ CSV saved to {csv_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='PINN-Lite-Foil latency benchmark'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--output', type=str, default='results/benchmark/',
        help='Output directory'
    )
    parser.add_argument(
        '--iterations', type=int, default=1000,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16],
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--threads', type=int, default=1,
        help='Number of threads'
    )
    
    args = parser.parse_args()
    
    run_full_benchmark(
        model_path=args.model,
        output_dir=args.output,
        n_iterations=args.iterations,
        batch_sizes=args.batch_sizes,
        num_threads=args.threads
    )


if __name__ == '__main__':
    main()
