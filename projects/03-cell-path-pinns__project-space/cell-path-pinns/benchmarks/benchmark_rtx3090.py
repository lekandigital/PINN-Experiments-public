#!/usr/bin/env python3
"""
Benchmark Cell-Path PINNs on RTX 3090

Tests training speed, inference speed, and memory usage with various
configurations including AMP and LR scheduling.
"""

import torch
import time
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell_path_pinns import CellPathModel, generate_synthetic_trajectory


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
        print(f"CUDA: {cuda_version}")
        print(f"PyTorch: {torch.__version__}")
    else:
        print("No GPU available - running on CPU")
    print()


def benchmark_training(
    n_points_list: list = [100, 200, 500, 1000, 2000],
    epochs: int = 100,
    use_amp: bool = False,
    use_scheduler: bool = False
) -> list:
    """
    Benchmark training speed vs data size.

    Args:
        n_points_list: List of data sizes to test
        epochs: Number of epochs per training run
        use_amp: Enable AMP
        use_scheduler: Enable LR scheduler

    Returns:
        List of result dictionaries
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    config_str = []
    if use_amp:
        config_str.append("AMP")
    if use_scheduler:
        config_str.append("Scheduler")
    config_name = "+".join(config_str) if config_str else "Baseline"

    print(f"\n{'='*60}")
    print(f"Training Benchmark ({config_name})")
    print(f"{'='*60}")
    print(f"{'N Points':>10} | {'Time (s)':>10} | {'Epochs/s':>10} | {'Mem (MB)':>10}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for n_points in n_points_list:
        # Generate data
        t, x, y = generate_synthetic_trajectory(n_steps=n_points, seed=42)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Create and train model
        model = CellPathModel(
            hidden_dim=64,
            device=device,
            use_amp=use_amp
        )

        start = time.perf_counter()
        model.fit(
            t, x, y,
            epochs=epochs,
            verbose=False,
            use_scheduler=use_scheduler
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # Memory usage
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            mem_mb = 0

        results.append({
            'config': config_name,
            'n_points': n_points,
            'epochs': epochs,
            'time_sec': elapsed,
            'epochs_per_sec': epochs / elapsed,
            'mem_mb': mem_mb
        })

        print(f"{n_points:>10} | {elapsed:>10.2f} | {epochs/elapsed:>10.1f} | {mem_mb:>10.1f}")

    return results


def benchmark_inference(
    n_predictions_list: list = [100, 1000, 10000, 100000],
    use_amp: bool = False
) -> list:
    """
    Benchmark inference speed.

    Args:
        n_predictions_list: List of prediction counts to test
        use_amp: Whether model was trained with AMP

    Returns:
        List of result dictionaries
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    print(f"\n{'='*60}")
    print(f"Inference Benchmark")
    print(f"{'='*60}")

    # Train a model first
    t, x, y = generate_synthetic_trajectory(n_steps=200, seed=42)
    model = CellPathModel(hidden_dim=64, device=device, use_amp=use_amp)
    model.fit(t, x, y, epochs=50, verbose=False)

    print(f"{'N Predictions':>15} | {'Time (ms)':>12} | {'Pred/s':>12}")
    print(f"{'-'*15}-+-{'-'*12}-+-{'-'*12}")

    for n_pred in n_predictions_list:
        t_pred = np.linspace(0, 1, n_pred)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model.predict_paths(t_pred)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        results.append({
            'n_predictions': n_pred,
            'time_ms': elapsed * 1000,
            'predictions_per_sec': n_pred / elapsed
        })

        print(f"{n_pred:>15,} | {elapsed*1000:>12.3f} | {n_pred/elapsed:>12,.0f}")

    return results


def benchmark_amp_comparison(
    n_points: int = 500,
    epochs: int = 200
) -> dict:
    """
    Compare AMP vs non-AMP training.

    Args:
        n_points: Number of data points
        epochs: Training epochs

    Returns:
        Comparison results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("\nAMP comparison requires CUDA - skipping")
        return {}

    print(f"\n{'='*60}")
    print(f"AMP Comparison (N={n_points}, epochs={epochs})")
    print(f"{'='*60}")

    # Generate data
    t, x, y = generate_synthetic_trajectory(n_steps=n_points, seed=42)

    results = {}

    for use_amp in [False, True]:
        label = "AMP" if use_amp else "FP32"

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = CellPathModel(hidden_dim=64, device=device, use_amp=use_amp)

        torch.cuda.synchronize()
        start = time.perf_counter()

        model.fit(t, x, y, epochs=epochs, verbose=False)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        final_loss = model.history['loss'][-1]

        results[label] = {
            'time_sec': elapsed,
            'mem_mb': mem_mb,
            'final_loss': final_loss,
            'epochs_per_sec': epochs / elapsed
        }

        print(f"{label:>6}: {elapsed:.2f}s, {mem_mb:.1f} MB, loss={final_loss:.6f}")

    # Compute speedup
    if 'FP32' in results and 'AMP' in results:
        speedup = results['FP32']['time_sec'] / results['AMP']['time_sec']
        mem_reduction = (results['FP32']['mem_mb'] - results['AMP']['mem_mb']) / results['FP32']['mem_mb'] * 100
        print(f"\nAMP Speedup: {speedup:.2f}x")
        print(f"Memory Reduction: {mem_reduction:.1f}%")

    return results


def benchmark_scheduler_comparison(
    n_points: int = 500,
    epochs: int = 200
) -> dict:
    """
    Compare training with and without LR scheduler.

    Args:
        n_points: Number of data points
        epochs: Training epochs

    Returns:
        Comparison results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"Scheduler Comparison (N={n_points}, epochs={epochs})")
    print(f"{'='*60}")

    # Generate data
    t, x, y = generate_synthetic_trajectory(n_steps=n_points, seed=42)

    results = {}

    for use_scheduler in [False, True]:
        label = "With Scheduler" if use_scheduler else "No Scheduler"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = CellPathModel(hidden_dim=64, device=device)

        start = time.perf_counter()
        model.fit(
            t, x, y,
            epochs=epochs,
            verbose=False,
            use_scheduler=use_scheduler
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        final_loss = model.history['loss'][-1]
        mse = model.compute_mse(t, x, y)

        results[label] = {
            'time_sec': elapsed,
            'final_loss': final_loss,
            'mse': mse
        }

        print(f"{label:>15}: {elapsed:.2f}s, loss={final_loss:.6f}, MSE={mse:.6f}")

    # Compute improvement
    if 'No Scheduler' in results and 'With Scheduler' in results:
        loss_improvement = (
            (results['No Scheduler']['final_loss'] - results['With Scheduler']['final_loss'])
            / results['No Scheduler']['final_loss'] * 100
        )
        print(f"\nLoss Improvement: {loss_improvement:.1f}%")

    return results


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("=" * 60)
    print("Cell-Path PINNs Benchmark Suite")
    print("=" * 60)
    print()

    print_gpu_info()

    all_results = {}

    # Training benchmarks
    all_results['training_baseline'] = benchmark_training(
        n_points_list=[100, 200, 500, 1000],
        epochs=100,
        use_amp=False,
        use_scheduler=False
    )

    if torch.cuda.is_available():
        all_results['training_amp'] = benchmark_training(
            n_points_list=[100, 200, 500, 1000],
            epochs=100,
            use_amp=True,
            use_scheduler=False
        )

        all_results['training_amp_scheduler'] = benchmark_training(
            n_points_list=[100, 200, 500, 1000],
            epochs=100,
            use_amp=True,
            use_scheduler=True
        )

    # Inference benchmarks
    all_results['inference'] = benchmark_inference()

    # Comparisons
    if torch.cuda.is_available():
        all_results['amp_comparison'] = benchmark_amp_comparison()

    all_results['scheduler_comparison'] = benchmark_scheduler_comparison()

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        # Find best configs
        training_results = (
            all_results.get('training_baseline', []) +
            all_results.get('training_amp', []) +
            all_results.get('training_amp_scheduler', [])
        )

        if training_results:
            best = max(training_results, key=lambda x: x['epochs_per_sec'])
            print(f"Fastest Training: {best['config']} @ {best['epochs_per_sec']:.1f} epochs/s")

        inference_results = all_results.get('inference', [])
        if inference_results:
            # 1000 predictions reference
            ref = next((r for r in inference_results if r['n_predictions'] == 1000), None)
            if ref:
                print(f"Inference (1K points): {ref['time_ms']:.2f} ms")

    print(f"\nBenchmark complete!")

    return all_results


if __name__ == '__main__':
    run_full_benchmark()
