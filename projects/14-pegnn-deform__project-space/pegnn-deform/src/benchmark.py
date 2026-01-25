"""
PEGNN-Deform: Inference Benchmark Harness

Measures:
- Inference latency (with proper GPU synchronization)
- NRMSE (Normalized Root Mean Square Error)
- Memory usage
- Throughput (samples/second)

Author: PEGNN-Deform Team
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda.amp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from pegdeform_model import PEGNNDeform, count_parameters
from train_pegdeform import MeshDataset


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'cached': 0, 'max_allocated': 0}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'cached': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9
    }


def compute_nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Normalized Root Mean Square Error.
    
    NRMSE = RMSE / (max - min) of target
    """
    mse = F.mse_loss(pred, target).item()
    rmse = np.sqrt(mse)
    target_range = (target.max() - target.min()).item()
    return rmse / (target_range + 1e-8)


def benchmark_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 10,
    num_benchmark: int = 100,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: PEGNN-Deform model
        dataloader: Test data loader
        device: Compute device
        num_warmup: Warmup iterations (not timed)
        num_benchmark: Benchmark iterations
        use_amp: Use automatic mixed precision
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    
    # Get a batch for benchmarking
    batch = next(iter(dataloader))
    pos = batch.pos.to(device)
    vel = batch.vel.to(device)
    edge_index = batch.edge_index.to(device)
    edge_attr = batch.edge_attr.to(device)
    pos_gt = batch.pos_next.to(device)
    
    num_nodes = pos.size(0)
    num_edges = edge_index.size(1)
    
    print(f"\nBenchmark configuration:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_benchmark}")
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Warmup
    print("\nRunning warmup...")
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(pos, vel, edge_index, edge_attr)
            else:
                _ = model(pos, vel, edge_index, edge_attr)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark latency
    print("Running benchmark...")
    latencies = []
    nrmses = []
    
    with torch.no_grad():
        for i in tqdm(range(num_benchmark), desc="Benchmarking"):
            # Synchronize before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    pos_pred, vel_pred, _ = model(pos, vel, edge_index, edge_attr)
            else:
                pos_pred, vel_pred, _ = model(pos, vel, edge_index, edge_attr)
            
            # Synchronize after computation
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Compute NRMSE
            nrmse = compute_nrmse(pos_pred, pos_gt)
            nrmses.append(nrmse)
    
    # Get memory usage
    memory = get_gpu_memory_usage()
    
    # Compute statistics
    latencies = np.array(latencies)
    nrmses = np.array(nrmses)
    
    results = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'throughput_hz': float(1000 / np.mean(latencies)),
        'nrmse_mean': float(np.mean(nrmses)),
        'nrmse_std': float(np.std(nrmses)),
        'nrmse_percent': float(np.mean(nrmses) * 100),
        'memory_allocated_gb': memory['allocated'],
        'memory_cached_gb': memory['cached'],
        'memory_peak_gb': memory['max_allocated']
    }
    
    return results


def benchmark_scaling(
    model: torch.nn.Module,
    device: torch.device,
    node_counts: List[int] = [100, 500, 1000, 2000, 5000],
    use_amp: bool = True
) -> List[Dict[str, float]]:
    """
    Benchmark how inference scales with mesh size.
    
    Args:
        model: PEGNN-Deform model
        device: Compute device
        node_counts: List of node counts to test
        use_amp: Use automatic mixed precision
        
    Returns:
        List of benchmark results for each node count
    """
    model.eval()
    results = []
    
    print("\nScaling benchmark:")
    
    for N in node_counts:
        print(f"\n  Testing N={N} nodes...")
        
        # Generate synthetic data
        E = N * 6  # Approximate edges for grid-like mesh
        
        pos = torch.randn(N, 3, device=device)
        vel = torch.randn(N, 3, device=device) * 0.1
        edge_index = torch.randint(0, N, (2, E), device=device)
        edge_attr = torch.rand(E, 2, device=device)
        edge_attr[:, 0] = edge_attr[:, 0] * 10 + 1  # stiffness
        edge_attr[:, 1] = edge_attr[:, 1] * 0.5 + 0.1  # rest length
        
        # Reset memory
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(pos, vel, edge_index, edge_attr)
                else:
                    _ = model(pos, vel, edge_index, edge_attr)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(50):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(pos, vel, edge_index, edge_attr)
                else:
                    _ = model(pos, vel, edge_index, edge_attr)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                latencies.append((time.perf_counter() - start) * 1000)
        
        memory = get_gpu_memory_usage()
        
        results.append({
            'num_nodes': N,
            'num_edges': E,
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'memory_peak_gb': memory['max_allocated'],
            'throughput_hz': float(1000 / np.mean(latencies))
        })
        
        print(f"    Latency: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
        print(f"    Memory: {memory['max_allocated']:.2f} GB")
    
    return results


def print_results(results: Dict[str, float]):
    """Pretty print benchmark results."""
    print("\n" + "="*60)
    print("PEGNN-DEFORM BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nMesh Size:")
    print(f"  Nodes: {results['num_nodes']}")
    print(f"  Edges: {results['num_edges']}")
    
    print(f"\nLatency (ms):")
    print(f"  Mean:   {results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f}")
    print(f"  P50:    {results['latency_p50_ms']:.2f}")
    print(f"  P95:    {results['latency_p95_ms']:.2f}")
    print(f"  P99:    {results['latency_p99_ms']:.2f}")
    print(f"  Min:    {results['latency_min_ms']:.2f}")
    print(f"  Max:    {results['latency_max_ms']:.2f}")
    
    print(f"\nThroughput:")
    print(f"  {results['throughput_hz']:.1f} samples/second")
    
    print(f"\nAccuracy (NRMSE):")
    print(f"  Mean: {results['nrmse_percent']:.2f}%")
    
    print(f"\nGPU Memory:")
    print(f"  Allocated: {results['memory_allocated_gb']:.2f} GB")
    print(f"  Cached:    {results['memory_cached_gb']:.2f} GB")
    print(f"  Peak:      {results['memory_peak_gb']:.2f} GB")
    
    # Check success criteria
    print("\n" + "-"*60)
    print("SUCCESS CRITERIA:")
    
    criteria = [
        ("Latency < 100ms", results['latency_mean_ms'] < 100),
        ("NRMSE < 5%", results['nrmse_percent'] < 5),
        ("Memory < 40GB", results['memory_peak_gb'] < 40),
    ]
    
    all_passed = True
    for name, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("-"*60)
    if all_passed:
        print("ALL CRITERIA PASSED ✓")
    else:
        print("SOME CRITERIA FAILED ✗")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark PEGNN-Deform')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of benchmark samples')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--scaling-test', action='store_true',
                        help='Run scaling benchmark')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Model hidden size')
    parser.add_argument('--num-mp-layers', type=int, default=3,
                        help='Number of message passing layers')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
    
    # Create or load model
    model = PEGNNDeform(
        hidden_size=args.hidden_size,
        num_mp_layers=args.num_mp_layers,
        dt=0.01
    ).to(device)
    
    if args.model and Path(args.model).exists():
        print(f"\nLoading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("\nUsing randomly initialized model")
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Run scaling test if requested
    if args.scaling_test:
        scaling_results = benchmark_scaling(
            model, device,
            node_counts=[100, 500, 1000, 2000, 5000],
            use_amp=not args.no_amp
        )
        
        print("\n" + "="*60)
        print("SCALING RESULTS")
        print("="*60)
        print(f"{'Nodes':>8} {'Edges':>8} {'Latency (ms)':>15} {'Memory (GB)':>12} {'Throughput':>12}")
        print("-"*60)
        for r in scaling_results:
            print(f"{r['num_nodes']:>8} {r['num_edges']:>8} "
                  f"{r['latency_mean_ms']:>7.2f} ± {r['latency_std_ms']:>5.2f} "
                  f"{r['memory_peak_gb']:>12.2f} "
                  f"{r['throughput_hz']:>10.1f} Hz")
        print("="*60)
        return
    
    # Load test data
    data_dir = Path(args.data_dir)
    test_dataset = MeshDataset(data_dir / 'synthetic', split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Run benchmark
    results = benchmark_inference(
        model, test_loader, device,
        num_warmup=10,
        num_benchmark=args.num_samples,
        use_amp=not args.no_amp
    )
    
    # Print results
    print_results(results)
    
    # Save results
    results_path = Path('benchmark_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
