"""
HGNN-ClothDyn Benchmarking Suite

Performance benchmarks including:
- Rollout stability (RMSE over time)
- Inference speed (FPS)
- Edge length preservation error
- Visualization generation

Author: HGNN-ClothDyn
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import h5py
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from model import HGNNClothDyn
from mesh_to_graph import mesh_to_graph, build_graph_pyramid
from train import ClothDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Benchmarker:
    """
    Benchmarking suite for HGNN-ClothDyn model.
    """
    
    def __init__(
        self,
        model: HGNNClothDyn,
        dataset: ClothDataset,
        device: torch.device
    ):
        """
        Args:
            model: Trained HGNNClothDyn model
            dataset: Test dataset
            device: Compute device
        """
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device
        
        # Build graph structures
        self._build_graph_structures()
        
        logger.info(f"Benchmarker initialized on {device}")
    
    def _build_graph_structures(self):
        """Pre-build graph structures for benchmarking."""
        initial_data = self.dataset.get_frame(0).to(self.device)
        graphs, cluster_maps = build_graph_pyramid(initial_data, num_levels=2)
        
        self.fine_template = graphs[0]
        self.coarse_template = graphs[1] if len(graphs) > 1 else None
        self.cluster_map = cluster_maps[0] if cluster_maps else None
        
        if self.coarse_template is not None:
            self.coarse_template = self.coarse_template.to(self.device)
        if self.cluster_map is not None:
            self.cluster_map = self.cluster_map.to(self.device)
    
    def _pool_positions(
        self,
        fine_pos: torch.Tensor,
        cluster_map: torch.Tensor,
        num_coarse: int
    ) -> torch.Tensor:
        """Pool fine positions to coarse level."""
        coarse_pos = torch.zeros(num_coarse, 3, device=fine_pos.device, dtype=fine_pos.dtype)
        counts = torch.zeros(num_coarse, device=fine_pos.device, dtype=fine_pos.dtype)
        
        coarse_pos.scatter_add_(0, cluster_map.unsqueeze(1).expand(-1, 3), fine_pos)
        counts.scatter_add_(0, cluster_map, torch.ones(fine_pos.size(0), device=fine_pos.device))
        
        return coarse_pos / counts.unsqueeze(1).clamp(min=1)
    
    @torch.no_grad()
    def benchmark_rollout(
        self,
        num_frames: int = 50,
        start_frame: int = 0
    ) -> Dict[str, Any]:
        """
        Benchmark rollout stability.
        
        Measures prediction quality over extended autoregressive rollout.
        
        Args:
            num_frames: Number of frames to roll out
            start_frame: Starting frame index
            
        Returns:
            Dictionary with rollout metrics
        """
        logger.info(f"Running rollout benchmark ({num_frames} frames)...")
        
        # Get ground truth
        _, gt_positions, gt_velocities = self.dataset.get_sequence(
            start_frame=start_frame,
            end_frame=min(start_frame + num_frames + 1, self.dataset.num_frames)
        )
        gt_positions = gt_positions.to(self.device)
        gt_velocities = gt_velocities.to(self.device)
        
        edges = self.dataset.edges.to(self.device)
        rest_lengths = self.dataset.rest_lengths.to(self.device)
        
        # Initialize from ground truth
        current_pos = gt_positions[0].clone()
        current_vel = gt_velocities[0].clone()
        
        # Storage for predictions
        pred_positions = [current_pos.cpu().numpy()]
        position_errors = []
        velocity_errors = []
        edge_length_errors = []
        
        # Rollout
        actual_frames = min(num_frames, len(gt_positions) - 1)
        
        for t in range(1, actual_frames + 1):
            # Prepare input
            x = torch.cat([current_pos, current_vel], dim=-1)
            
            data = Data(
                x=x,
                edge_index=self.fine_template.edge_index,
                edge_attr=self.fine_template.edge_attr,
                pos=current_pos,
                num_nodes=self.dataset.num_nodes
            ).to(self.device)
            
            # Prepare coarse data
            if self.coarse_template is not None:
                coarse_pos = self._pool_positions(
                    current_pos, self.cluster_map, self.coarse_template.num_nodes
                )
                coarse_x = torch.cat([coarse_pos, torch.zeros_like(coarse_pos)], dim=-1)
                coarse_data = Data(
                    x=coarse_x,
                    edge_index=self.coarse_template.edge_index,
                    edge_attr=self.coarse_template.edge_attr,
                    pos=coarse_pos,
                    num_nodes=self.coarse_template.num_nodes
                ).to(self.device)
            else:
                coarse_data = None
            
            # Forward pass
            delta_vel = self.model(data, coarse_data, self.cluster_map)
            
            # Update state
            pred_vel = (current_vel + delta_vel) * 0.99  # Damping
            pred_pos = current_pos + pred_vel * 0.01  # dt
            
            # Simple ground collision
            collision_mask = pred_pos[:, 1] < -1.0
            pred_pos[collision_mask, 1] = -1.0
            pred_vel[collision_mask, 1] = torch.abs(pred_vel[collision_mask, 1]) * 0.5
            
            # Store prediction
            pred_positions.append(pred_pos.cpu().numpy())
            
            # Compute errors against ground truth
            pos_error = torch.sqrt(torch.mean((pred_pos - gt_positions[t]) ** 2)).item()
            vel_error = torch.sqrt(torch.mean((pred_vel - gt_velocities[t]) ** 2)).item()
            
            # Edge length error
            pred_lengths = torch.norm(
                pred_pos[edges[:, 0]] - pred_pos[edges[:, 1]], dim=-1
            )
            edge_error = torch.mean(torch.abs(pred_lengths - rest_lengths)).item()
            
            position_errors.append(pos_error)
            velocity_errors.append(vel_error)
            edge_length_errors.append(edge_error)
            
            # Update for next step
            current_pos = pred_pos
            current_vel = pred_vel
        
        # Compile results
        results = {
            'num_frames': actual_frames,
            'position_rmse': {
                'mean': float(np.mean(position_errors)),
                'max': float(np.max(position_errors)),
                'final': float(position_errors[-1]) if position_errors else 0.0,
                'per_frame': position_errors
            },
            'velocity_rmse': {
                'mean': float(np.mean(velocity_errors)),
                'max': float(np.max(velocity_errors)),
                'per_frame': velocity_errors
            },
            'edge_length_error': {
                'mean': float(np.mean(edge_length_errors)),
                'max': float(np.max(edge_length_errors)),
                'per_frame': edge_length_errors
            },
            'predictions': pred_positions
        }
        
        logger.info(f"Rollout complete: Position RMSE = {results['position_rmse']['mean']:.6f}, "
                    f"Edge Error = {results['edge_length_error']['mean']:.6f}")
        
        return results
    
    @torch.no_grad()
    def benchmark_fps(
        self,
        num_warmup: int = 10,
        num_iters: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed (frames per second).
        
        Args:
            num_warmup: Number of warmup iterations
            num_iters: Number of timed iterations
            
        Returns:
            Dictionary with FPS metrics
        """
        logger.info(f"Running FPS benchmark ({num_iters} iterations)...")
        
        # Prepare test data
        data = self.dataset.get_frame(0).to(self.device)
        
        if self.coarse_template is not None:
            coarse_pos = self._pool_positions(
                data.pos, self.cluster_map, self.coarse_template.num_nodes
            )
            coarse_x = torch.cat([coarse_pos, torch.zeros_like(coarse_pos)], dim=-1)
            coarse_data = Data(
                x=coarse_x,
                edge_index=self.coarse_template.edge_index,
                edge_attr=self.coarse_template.edge_attr,
                pos=coarse_pos,
                num_nodes=self.coarse_template.num_nodes
            ).to(self.device)
        else:
            coarse_data = None
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.model(data, coarse_data, self.cluster_map)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed iterations
        start_time = time.perf_counter()
        
        for _ in range(num_iters):
            _ = self.model(data, coarse_data, self.cluster_map)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        
        fps = num_iters / elapsed
        ms_per_frame = elapsed * 1000 / num_iters
        
        results = {
            'fps': fps,
            'ms_per_frame': ms_per_frame,
            'num_iterations': num_iters,
            'total_time_seconds': elapsed
        }
        
        logger.info(f"FPS: {fps:.1f} ({ms_per_frame:.2f} ms/frame)")
        
        return results
    
    @torch.no_grad()
    def benchmark_memory(self) -> Dict[str, float]:
        """
        Benchmark GPU memory usage.
        
        Returns:
            Dictionary with memory metrics
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping memory benchmark")
            return {'gpu_available': False}
        
        logger.info("Running memory benchmark...")
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Measure baseline
        baseline_mem = torch.cuda.memory_allocated() / 1e6
        
        # Run forward pass
        data = self.dataset.get_frame(0).to(self.device)
        
        if self.coarse_template is not None:
            coarse_pos = self._pool_positions(
                data.pos, self.cluster_map, self.coarse_template.num_nodes
            )
            coarse_x = torch.cat([coarse_pos, torch.zeros_like(coarse_pos)], dim=-1)
            coarse_data = Data(
                x=coarse_x,
                edge_index=self.coarse_template.edge_index,
                edge_attr=self.coarse_template.edge_attr,
                pos=coarse_pos,
                num_nodes=self.coarse_template.num_nodes
            ).to(self.device)
        else:
            coarse_data = None
        
        _ = self.model(data, coarse_data, self.cluster_map)
        
        # Measure peak
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        current_mem = torch.cuda.memory_allocated() / 1e6
        
        results = {
            'gpu_available': True,
            'baseline_mb': baseline_mem,
            'peak_mb': peak_mem,
            'current_mb': current_mem,
            'gpu_name': torch.cuda.get_device_name(0),
            'total_vram_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
        
        logger.info(f"Memory: Peak = {peak_mem:.1f} MB, "
                    f"VRAM = {results['total_vram_gb']:.1f} GB")
        
        return results
    
    def run_all_benchmarks(
        self,
        rollout_frames: int = 50,
        fps_iters: int = 100
    ) -> Dict[str, Any]:
        """
        Run all benchmarks.
        
        Args:
            rollout_frames: Number of frames for rollout test
            fps_iters: Number of iterations for FPS test
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("=" * 60)
        logger.info("Running all benchmarks")
        logger.info("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'num_nodes': self.dataset.num_nodes,
            'num_edges': self.dataset.edges.shape[0],
            'benchmarks': {}
        }
        
        # Rollout stability
        results['benchmarks']['rollout'] = self.benchmark_rollout(
            num_frames=rollout_frames
        )
        
        # Remove predictions from saved results (too large)
        rollout_results = results['benchmarks']['rollout'].copy()
        rollout_results.pop('predictions', None)
        results['benchmarks']['rollout'] = rollout_results
        
        # FPS
        results['benchmarks']['fps'] = self.benchmark_fps(num_iters=fps_iters)
        
        # Memory
        results['benchmarks']['memory'] = self.benchmark_memory()
        
        # Summary
        results['summary'] = {
            'position_rmse_mean': results['benchmarks']['rollout']['position_rmse']['mean'],
            'edge_length_error_mean': results['benchmarks']['rollout']['edge_length_error']['mean'],
            'fps': results['benchmarks']['fps']['fps'],
            'peak_memory_mb': results['benchmarks']['memory'].get('peak_mb', 0)
        }
        
        logger.info("=" * 60)
        logger.info("Benchmark Summary:")
        logger.info(f"  Position RMSE: {results['summary']['position_rmse_mean']:.6f}")
        logger.info(f"  Edge Length Error: {results['summary']['edge_length_error_mean']:.6f}")
        logger.info(f"  FPS: {results['summary']['fps']:.1f}")
        logger.info(f"  Peak Memory: {results['summary']['peak_memory_mb']:.1f} MB")
        logger.info("=" * 60)
        
        return results


def generate_plots(results: Dict[str, Any], output_dir: Path):
    """
    Generate benchmark visualization plots.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rollout = results['benchmarks']['rollout']
    
    # Error over time plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    frames = list(range(1, len(rollout['position_rmse']['per_frame']) + 1))
    
    # Position RMSE
    axes[0].plot(frames, rollout['position_rmse']['per_frame'], 'b-', linewidth=2)
    axes[0].axhline(y=0.1, color='r', linestyle='--', label='Target threshold (0.1)')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Position RMSE')
    axes[0].set_title('Position RMSE vs Frame')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Velocity RMSE
    axes[1].plot(frames, rollout['velocity_rmse']['per_frame'], 'g-', linewidth=2)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Velocity RMSE')
    axes[1].set_title('Velocity RMSE vs Frame')
    axes[1].grid(True, alpha=0.3)
    
    # Edge length error
    axes[2].plot(frames, rollout['edge_length_error']['per_frame'], 'orange', linewidth=2)
    axes[2].axhline(y=0.05, color='r', linestyle='--', label='Target threshold (0.05)')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Edge Length MAE')
    axes[2].set_title('Edge Length Error vs Frame')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rollout_error_plot.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved rollout plot to {output_dir / 'rollout_error_plot.png'}")
    
    # Performance summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics = ['Position\nRMSE', 'Edge Length\nError', 'FPS / 100', 'Memory\n(GB)']
    values = [
        results['summary']['position_rmse_mean'],
        results['summary']['edge_length_error_mean'],
        results['summary']['fps'] / 100,
        results['summary']['peak_memory_mb'] / 1000
    ]
    thresholds = [0.1, 0.05, 0.5, 40]  # Normalized targets
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values, width, label='Measured', color='steelblue')
    bars2 = ax.bar(x + width/2, thresholds, width, label='Target', color='coral', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('HGNN-ClothDyn Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, values):
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved performance summary to {output_dir / 'performance_summary.png'}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> HGNNClothDyn:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 128)
    
    model = HGNNClothDyn(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_message_passes=3,
        num_levels=2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Benchmark HGNN-ClothDyn model')
    
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-data', '-d', type=str, required=True,
                        help='Path to test data (HDF5)')
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--rollout-frames', type=int, default=50,
                        help='Number of frames for rollout test')
    parser.add_argument('--fps-iters', type=int, default=100,
                        help='Number of iterations for FPS test')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load dataset
    dataset = ClothDataset(args.test_data, device=device)
    
    # Create benchmarker
    benchmarker = Benchmarker(model, dataset, device)
    
    # Run benchmarks
    results = benchmarker.run_all_benchmarks(
        rollout_frames=args.rollout_frames,
        fps_iters=args.fps_iters
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Generate plots
    if not args.no_plots:
        generate_plots(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Position RMSE (mean):     {results['summary']['position_rmse_mean']:.6f}")
    print(f"Edge Length Error (mean): {results['summary']['edge_length_error_mean']:.6f}")
    print(f"FPS:                      {results['summary']['fps']:.1f}")
    print(f"Peak Memory:              {results['summary']['peak_memory_mb']:.1f} MB")
    print("=" * 60)
    
    # Check pass/fail criteria
    passes = True
    criteria = [
        ('Position RMSE < 0.1', results['summary']['position_rmse_mean'] < 0.1),
        ('Edge Length Error < 0.05', results['summary']['edge_length_error_mean'] < 0.05),
        ('FPS > 50', results['summary']['fps'] > 50),
    ]
    
    print("\nPass/Fail Criteria:")
    for name, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        passes = passes and passed
    
    print("\n" + ("✓ All benchmarks passed!" if passes else "✗ Some benchmarks failed"))
    
    return 0 if passes else 1


if __name__ == '__main__':
    exit(main())
