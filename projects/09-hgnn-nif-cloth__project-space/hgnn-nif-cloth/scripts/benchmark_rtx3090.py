#!/usr/bin/env python
"""
RTX 3090 Benchmark Script

Runs comprehensive benchmarks for the HGNN-NIF-Cloth model on RTX 3090.

Usage:
    python scripts/benchmark_rtx3090.py
    python scripts/benchmark_rtx3090.py --config configs/rtx3090_large.yaml
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.hybrid_model import HGNN_NIF_ClothModel, load_model_from_config


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        print("=" * 60)
        print("GPU Information")
        print("=" * 60)
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  PyTorch: {torch.__version__}")
    else:
        print("WARNING: CUDA not available!")


def benchmark_batch_scaling(model, device, max_batch=64, num_warmup=10, num_iters=50):
    """Benchmark model with different batch sizes."""
    print("\n" + "=" * 60)
    print("Batch Scaling Benchmark")
    print("=" * 60)

    model.eval()

    # Fixed graph structure
    fine_edges = torch.randint(0, 400, (2, 300)).to(device)
    coarse_edges = torch.randint(0, 100, (2, 80)).to(device)

    results = []

    for bs in [1, 2, 4, 8, 16, 32, 64]:
        if bs > max_batch:
            break

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create batch data
            fine_pos = torch.randn(bs, 400, 3).to(device)
            coarse_pos = torch.randn(bs, 100, 3).to(device)
            query_points = torch.randn(bs, 1000, 3).to(device)

            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iters):
                    _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000 / num_iters

            # Memory stats
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            fps = 1000 / elapsed_ms * bs

            print(f"Batch {bs:2d}: {elapsed_ms:6.2f} ms, Peak Memory: {peak_mem:.2f} GB, FPS: {fps:,.1f}")
            results.append({
                'batch_size': bs,
                'latency_ms': elapsed_ms,
                'peak_memory_gb': peak_mem,
                'fps': fps
            })

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Batch {bs:2d}: OOM (Out of Memory)")
                break
            raise

    return results


def benchmark_large_model(device, config_path='configs/rtx3090_large.yaml'):
    """Benchmark the large model configuration."""
    print("\n" + "=" * 60)
    print("Large Model Benchmark (RTX 3090 Config)")
    print("=" * 60)

    try:
        model = load_model_from_config(config_path, device)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Large mesh sizes from config
        fine_nodes = 1600  # 40x40
        coarse_nodes = 400  # 20x20
        query_points_count = 4000

        fine_edges = torch.randint(0, fine_nodes, (2, fine_nodes * 2)).to(device)
        coarse_edges = torch.randint(0, coarse_nodes, (2, coarse_nodes * 2)).to(device)

        results = []
        for bs in [1, 2, 4]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                fine_pos = torch.randn(bs, fine_nodes, 3).to(device)
                coarse_pos = torch.randn(bs, coarse_nodes, 3).to(device)
                query = torch.randn(bs, query_points_count, 3).to(device)

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(20):
                        _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000 / 20

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                fps = 1000 / elapsed_ms * bs

                print(f"Batch {bs:2d}: {elapsed_ms:6.2f} ms, Peak Memory: {peak_mem:.2f} GB, FPS: {fps:.1f}")
                results.append({
                    'batch_size': bs,
                    'latency_ms': elapsed_ms,
                    'peak_memory_gb': peak_mem,
                    'fps': fps
                })

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"Batch {bs:2d}: OOM")
                    break
                raise

        return results

    except Exception as e:
        print(f"Error loading large model: {e}")
        return []


def benchmark_training_step(model, device, batch_size=4):
    """Benchmark a single training step with gradient computation."""
    print("\n" + "=" * 60)
    print("Training Step Benchmark")
    print("=" * 60)

    model.train()

    fine_edges = torch.randint(0, 400, (2, 300)).to(device)
    coarse_edges = torch.randint(0, 100, (2, 80)).to(device)

    fine_pos = torch.randn(batch_size, 400, 3, requires_grad=True).to(device)
    coarse_pos = torch.randn(batch_size, 100, 3).to(device)
    query_points = torch.randn(batch_size, 1000, 3).to(device)
    target_sdf = torch.randn(batch_size, 1000).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
            loss = ((output['sdf'] - target_sdf) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    num_iters = 20
    start = time.perf_counter()
    for _ in range(num_iters):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
            loss = ((output['sdf'] - target_sdf) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / num_iters

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    throughput = 1000 / elapsed_ms

    print(f"Batch size: {batch_size}")
    print(f"Time per iteration: {elapsed_ms:.2f} ms")
    print(f"Throughput: {throughput:.1f} it/s")
    print(f"Peak memory: {peak_mem:.2f} GB")

    return {
        'batch_size': batch_size,
        'ms_per_iter': elapsed_ms,
        'it_per_sec': throughput,
        'peak_memory_gb': peak_mem
    }


def main():
    parser = argparse.ArgumentParser(description='RTX 3090 Benchmark')
    parser.add_argument('--config', type=str, default='configs/rtx3090_large.yaml',
                        help='Config file for large model benchmark')
    parser.add_argument('--skip_large', action='store_true',
                        help='Skip large model benchmark')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_gpu_info()

    # Baseline model benchmark
    print("\n" + "=" * 60)
    print("Baseline Model (136K params)")
    print("=" * 60)
    model = HGNN_NIF_ClothModel(
        node_feat_dim=3,
        latent_dim=64,
        hidden_dim=64,
        siren_hidden_dim=128,
        siren_layers=3
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    benchmark_batch_scaling(model, device)
    benchmark_training_step(model, device, batch_size=8)

    # Large model benchmark
    if not args.skip_large and Path(args.config).exists():
        benchmark_large_model(device, args.config)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
