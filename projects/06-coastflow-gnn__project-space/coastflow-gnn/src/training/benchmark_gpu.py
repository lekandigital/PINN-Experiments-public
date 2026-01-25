"""
GPU Benchmark Script for CoastFlow-GNN
Measures inference speed, training throughput, and memory usage on RTX 3090.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.coastflow_gnn import CoastFlowGNN
from src.data.dataset import SyntheticCoastalDataset
from torch_geometric.loader import DataLoader


def benchmark_inference(model, loader, device, num_warmup=10, num_runs=100):
    """Benchmark inference speed."""
    model.eval()

    # Get a batch
    batch = next(iter(loader)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batch.x, batch.edge_index, batch.batch)

    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(batch.x, batch.edge_index, batch.batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'samples_per_batch': int(len(batch.x)),
    }


def benchmark_training_throughput(model, loader, device, num_batches=50):
    """Benchmark training throughput."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = out.mean()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    total_samples = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        total_samples += batch.num_graphs

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        'samples_per_second': float(total_samples / elapsed),
        'batches_per_second': float(num_batches / elapsed),
        'total_samples': int(total_samples),
        'elapsed_seconds': float(elapsed),
    }


def benchmark_memory(model, loader, device):
    """Benchmark GPU memory usage."""
    if device.type != 'cuda':
        return {'peak_memory_mb': 0, 'allocated_mb': 0}

    torch.cuda.reset_peak_memory_stats()

    model.train()
    batch = next(iter(loader)).to(device)

    # Forward + backward
    out = model(batch.x, batch.edge_index, batch.batch)
    loss = out.mean()
    loss.backward()

    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    return {
        'peak_memory_mb': float(peak_memory),
        'allocated_mb': float(allocated),
    }


def run_benchmarks(output_dir: str = './benchmarks'):
    """Run all benchmarks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
        'benchmarks': {}
    }

    # Test different configurations
    configs = [
        {'hidden': 64, 'batch_size': 4, 'num_nodes': 200},
        {'hidden': 128, 'batch_size': 8, 'num_nodes': 500},
        {'hidden': 256, 'batch_size': 4, 'num_nodes': 1000},
    ]

    for config in configs:
        config_name = f"h{config['hidden']}_b{config['batch_size']}_n{config['num_nodes']}"
        print(f"\nBenchmarking config: {config_name}")

        # Create model
        model = CoastFlowGNN(
            in_channels=6,
            hidden_channels=config['hidden'],
            out_channels=4,
        ).to(device)

        # Create dataset
        dataset = SyntheticCoastalDataset(
            num_samples=100,
            num_nodes=config['num_nodes'],
            seed=42,
        )
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # Run benchmarks
        inference_results = benchmark_inference(model, loader, device)
        throughput_results = benchmark_training_throughput(model, loader, device)
        memory_results = benchmark_memory(model, loader, device)

        results['benchmarks'][config_name] = {
            'config': config,
            'inference': inference_results,
            'throughput': throughput_results,
            'memory': memory_results,
            'model_params': model.count_parameters(),
        }

        print(f"  Inference: {inference_results['mean_ms']:.2f} ms/batch")
        print(f"  Throughput: {throughput_results['samples_per_second']:.1f} samples/sec")
        print(f"  Peak Memory: {memory_results['peak_memory_mb']:.1f} MB")

    # Save results
    results_path = output_dir / 'gpu_benchmark.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./benchmarks')
    args = parser.parse_args()

    run_benchmarks(args.output_dir)
