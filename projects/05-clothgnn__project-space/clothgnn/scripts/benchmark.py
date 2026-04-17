"""
Benchmark Script for ClothGNN.

Benchmarks:
1. Inference speed (FPS) at different mesh sizes
2. Model quality (RMSE, edge length error)
3. Comparison with teacher model (if available)
4. ONNX vs PyTorch performance

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/distilled/best_model.pt
    python scripts/benchmark.py --checkpoint checkpoints/baseline/best_model.pt --teacher-checkpoint path/to/teacher.pt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from models.clothgnn import ClothGNNModel
from register_models import ClothGNNLite
from data.dataloader import create_dataloader
from data.generate_dataset import create_cloth_mesh


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def benchmark_inference_speed(
    model: torch.nn.Module,
    n_nodes_list: list = [400, 625, 900, 1024, 2500],
    n_iterations: int = 100,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """
    Benchmark inference speed at different mesh sizes.
    
    Target: 100+ FPS (< 10ms per step)
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    for n_nodes in n_nodes_list:
        grid_size = int(np.sqrt(n_nodes))
        actual_nodes = grid_size * grid_size
        
        logger.info(f"\nBenchmarking {actual_nodes} nodes ({grid_size}x{grid_size})...")
        
        # Create mesh
        positions, edge_index, fixed_mask = create_cloth_mesh(grid_size)
        positions = positions.to(device)
        edge_index = edge_index.to(device)
        
        # Prepare input
        velocities = torch.zeros_like(positions)
        
        if isinstance(model, ClothGNNLite):
            # Lite model expects [N, 6]
            node_features = torch.cat([positions, velocities], dim=-1)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _, _ = model(node_features, edge_index)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            hidden = None
            
            with torch.no_grad():
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    _, hidden = model(node_features, edge_index, hidden=hidden)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
        else:
            # Full model
            class Data:
                pass
            data = Data()
            data.x = torch.cat([
                positions, velocities,
                torch.zeros(actual_nodes, 10, device=device),
            ], dim=-1)
            data.edge_index = edge_index
            data.pos = positions
            
            hidden = model.init_hidden(actual_nodes, device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _, _ = model(data, hidden)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            
            with torch.no_grad():
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    _, hidden = model(data, hidden)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
        
        times = np.array(times)
        
        results[actual_nodes] = {
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'fps': float(1000 / times.mean()),
            'grid_size': grid_size,
        }
        
        logger.info(f"  {actual_nodes} nodes: {times.mean():.2f} ± {times.std():.2f} ms ({1000/times.mean():.1f} FPS)")
    
    return results


def benchmark_accuracy(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark model accuracy on test set.
    """
    model.eval()
    model.to(device)
    
    total_rmse = 0
    total_edge_error = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            positions = batch['positions']
            velocities = batch['velocities']
            edge_index = batch['edge_index']
            rest_lengths = batch['rest_lengths']
            
            if positions.dim() == 4:
                current_pos = positions[:, 0]
                target_pos = positions[:, 1]
            else:
                current_pos = positions
                target_pos = batch['target_positions']
            
            target_disp = target_pos - current_pos
            
            B = current_pos.shape[0]
            
            for b in range(B):
                n_nodes = batch['node_counts'][b].item()
                n_edges = batch['edge_counts'][b].item()
                
                pos = current_pos[b, :n_nodes]
                vel = velocities[b, :n_nodes] if velocities.dim() == 3 else velocities[b, 0, :n_nodes]
                edges = edge_index[b, :, :n_edges]
                rest = rest_lengths[b, :n_edges]
                gt_disp = target_disp[b, :n_nodes]
                
                if isinstance(model, ClothGNNLite):
                    node_features = torch.cat([pos, vel], dim=-1)
                    pred_disp, _ = model(node_features, edges)
                else:
                    class Data:
                        pass
                    data = Data()
                    data.x = torch.cat([pos, vel, torch.zeros(n_nodes, 10, device=device)], dim=-1)
                    data.edge_index = edges
                    data.pos = pos
                    hidden = model.init_hidden(n_nodes, device)
                    pred_disp, _ = model(data, hidden)
                
                # RMSE
                rmse = torch.sqrt(F.mse_loss(pred_disp, gt_disp))
                total_rmse += rmse.item()
                
                # Edge error
                pred_pos = pos + pred_disp
                src, dst = edges
                pred_lengths = torch.norm(pred_pos[dst] - pred_pos[src], dim=-1)
                edge_error = torch.abs(pred_lengths - rest).mean()
                total_edge_error += edge_error.item()
                
                n_samples += 1
    
    return {
        'rmse': total_rmse / n_samples,
        'edge_error': total_edge_error / n_samples,
        'n_samples': n_samples,
    }


def compare_models(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compare student vs teacher quality.
    """
    student.eval().to(device)
    teacher.eval().to(device)
    
    student_metrics = benchmark_accuracy(student, dataloader, device)
    teacher_metrics = benchmark_accuracy(teacher, dataloader, device)
    
    return {
        'student_rmse': student_metrics['rmse'],
        'teacher_rmse': teacher_metrics['rmse'],
        'student_edge_error': student_metrics['edge_error'],
        'teacher_edge_error': teacher_metrics['edge_error'],
        'rmse_ratio': student_metrics['rmse'] / teacher_metrics['rmse'] if teacher_metrics['rmse'] > 0 else 0,
    }


def run_full_benchmark(
    checkpoint_path: str,
    output_path: str,
    data_path: Optional[str] = None,
    teacher_checkpoint: Optional[str] = None,
    device: str = "cuda",
    is_lite: bool = True,
):
    """Run comprehensive benchmark suite."""
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if is_lite:
        model = ClothGNNLite(
            node_input_dim=6,
            node_hidden_dim=32,
            edge_hidden_dim=16,
            num_message_passes=3,
        )
    else:
        config = checkpoint.get('config', {'node_feat_dim': 16, 'hidden_dim': 64})
        model = ClothGNNModel(**config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    param_count = sum(p.numel() for p in model.parameters())
    
    results = {
        'model_params': param_count,
        'model_type': 'lite' if is_lite else 'full',
        'checkpoint': str(checkpoint_path),
    }
    
    # Speed benchmark
    logger.info("\n=== Speed Benchmark ===")
    results['speed'] = benchmark_inference_speed(
        model,
        n_nodes_list=[400, 625, 900, 1024],
        device=device,
    )
    
    # Accuracy benchmark
    if data_path and Path(data_path).exists():
        logger.info("\n=== Accuracy Benchmark ===")
        splits_path = Path(data_path).parent / "splits.json"
        
        test_loader = create_dataloader(
            data_path,
            batch_size=8,
            split='test' if splits_path.exists() else None,
            splits_path=str(splits_path) if splits_path.exists() else None,
            single_step=True,
            shuffle=False,
        )
        
        results['accuracy'] = benchmark_accuracy(model, test_loader, device)
        logger.info(f"RMSE: {results['accuracy']['rmse']:.4f}")
        logger.info(f"Edge Error: {results['accuracy']['edge_error']:.6f}")
        
        # Teacher comparison
        if teacher_checkpoint and Path(teacher_checkpoint).exists():
            logger.info("\n=== Teacher Comparison ===")
            
            teacher_ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
            teacher_config = teacher_ckpt.get('config', {'node_feat_dim': 16, 'hidden_dim': 64})
            teacher = ClothGNNModel(**teacher_config)
            
            if 'model_state_dict' in teacher_ckpt:
                teacher.load_state_dict(teacher_ckpt['model_state_dict'])
            
            results['comparison'] = compare_models(model, teacher, test_loader, device)
            logger.info(f"Student RMSE: {results['comparison']['student_rmse']:.4f}")
            logger.info(f"Teacher RMSE: {results['comparison']['teacher_rmse']:.4f}")
            logger.info(f"Ratio: {results['comparison']['rmse_ratio']:.2f}")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nBenchmark results saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Parameters: {param_count:,}")
    
    if 1024 in results['speed']:
        fps = results['speed'][1024]['fps']
        logger.info(f"Speed (1024 nodes): {fps:.1f} FPS")
        logger.info(f"Target: 100+ FPS - {'✓ PASS' if fps >= 100 else '✗ FAIL'}")
    
    if 'accuracy' in results:
        rmse = results['accuracy']['rmse']
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"Target: < 0.1 - {'✓ PASS' if rmse < 0.1 else '✗ FAIL'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ClothGNN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/cloth_dynamics.h5",
                        help="Path to test data")
    parser.add_argument("--teacher-checkpoint", type=str,
                        help="Path to teacher checkpoint for comparison")
    parser.add_argument("--output", type=str, default="benchmarks/results.json",
                        help="Output path for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--full-model", action="store_true",
                        help="Benchmark full ClothGNNModel instead of lite")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    run_full_benchmark(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        data_path=args.data,
        teacher_checkpoint=args.teacher_checkpoint,
        device=args.device,
        is_lite=not args.full_model,
    )


if __name__ == "__main__":
    main()
