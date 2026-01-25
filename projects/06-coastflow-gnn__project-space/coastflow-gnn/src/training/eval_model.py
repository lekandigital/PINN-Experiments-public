"""
Model Evaluation Script for CoastFlow-GNN

Computes evaluation metrics and generates analysis reports:
- MSE, MAE, R² for velocity and wave height predictions
- Physics residual statistics
- Inference speed benchmarks
- Per-sample error analysis
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.coastflow_gnn import CoastFlowGNN
from src.models.physics_losses import (
    compute_continuity_loss,
    compute_momentum_loss,
    compute_turbulence_loss,
)
from src.data.dataset import SyntheticCoastalDataset, create_data_loaders


class ModelEvaluator:
    """
    Evaluator for CoastFlow-GNN model.
    
    Computes comprehensive metrics on test data.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Results storage
        self.results = {}
    
    @torch.no_grad()
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics on test set.
        
        Returns:
            Dictionary with MSE, MAE, R² for each output channel
        """
        all_preds = []
        all_targets = []
        all_positions = []
        
        for data in self.test_loader:
            data = data.to(self.device)
            
            predictions = self.model(data.x, data.edge_index, data.batch)
            
            all_preds.append(predictions.cpu())
            all_targets.append(data.y.cpu())
            all_positions.append(data.pos.cpu())
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        metrics = {}
        
        # Overall metrics
        mse = F.mse_loss(preds, targets).item()
        mae = F.l1_loss(preds, targets).item()
        
        # R² score
        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot).item()
        
        metrics['mse'] = mse
        metrics['mae'] = mae
        metrics['r2'] = r2
        
        # Per-channel metrics
        channel_names = ['u_x', 'u_y', 'u_z', 'wave_height']
        for i, name in enumerate(channel_names):
            channel_mse = F.mse_loss(preds[:, i], targets[:, i]).item()
            channel_mae = F.l1_loss(preds[:, i], targets[:, i]).item()
            
            ss_res_ch = ((targets[:, i] - preds[:, i]) ** 2).sum()
            ss_tot_ch = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
            channel_r2 = 1 - (ss_res_ch / ss_tot_ch).item() if ss_tot_ch > 0 else 0
            
            metrics[f'{name}_mse'] = channel_mse
            metrics[f'{name}_mae'] = channel_mae
            metrics[f'{name}_r2'] = channel_r2
        
        self.results['metrics'] = metrics
        return metrics
    
    @torch.no_grad()
    def compute_physics_residuals(self) -> Dict[str, float]:
        """
        Compute physics equation residuals on predictions.
        
        Returns:
            Dictionary with continuity, momentum, turbulence residuals
        """
        continuity_residuals = []
        momentum_residuals = []
        turbulence_residuals = []
        
        for data in self.test_loader:
            data = data.to(self.device)
            
            predictions = self.model(data.x, data.edge_index, data.batch)
            
            # Extract velocity
            u_pred = predictions[:, :3]
            pos = data.x[:, :3]
            
            # Compute residuals
            cont_loss = compute_continuity_loss(u_pred, pos, data.edge_index)
            mom_loss = compute_momentum_loss(u_pred, pos, data.edge_index)
            turb_loss = compute_turbulence_loss(u_pred, None, None, pos, data.edge_index)
            
            continuity_residuals.append(cont_loss.item())
            momentum_residuals.append(mom_loss.item())
            turbulence_residuals.append(turb_loss.item())
        
        residuals = {
            'continuity': {
                'mean': np.mean(continuity_residuals),
                'std': np.std(continuity_residuals),
                'max': np.max(continuity_residuals),
            },
            'momentum': {
                'mean': np.mean(momentum_residuals),
                'std': np.std(momentum_residuals),
                'max': np.max(momentum_residuals),
            },
            'turbulence': {
                'mean': np.mean(turbulence_residuals),
                'std': np.std(turbulence_residuals),
                'max': np.max(turbulence_residuals),
            },
        }
        
        self.results['physics_residuals'] = residuals
        return residuals
    
    @torch.no_grad()
    def benchmark_inference_speed(
        self,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Returns:
            Dictionary with timing statistics (ms per sample)
        """
        # Get a sample batch
        sample_data = next(iter(self.test_loader)).to(self.device)
        batch_size = sample_data.batch.max().item() + 1
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.model(sample_data.x, sample_data.edge_index, sample_data.batch)
        
        # Sync GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.model(sample_data.x, sample_data.edge_index, sample_data.batch)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000 / batch_size)  # ms per sample
        
        timing = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'samples_per_second': 1000 / np.mean(times),
        }
        
        self.results['inference_timing'] = timing
        return timing
    
    def generate_report(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate evaluation report.
        
        Args:
            output_path: Path to save JSON report
            
        Returns:
            Report as formatted string
        """
        # Run all evaluations if not done
        if 'metrics' not in self.results:
            self.compute_metrics()
        if 'physics_residuals' not in self.results:
            self.compute_physics_residuals()
        if 'inference_timing' not in self.results:
            self.benchmark_inference_speed()
        
        # Add model info
        self.results['model_info'] = {
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
        }
        
        # Save JSON
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        # Generate text report
        report = []
        report.append("=" * 60)
        report.append("CoastFlow-GNN Evaluation Report")
        report.append("=" * 60)
        
        report.append("\n📊 Prediction Metrics:")
        report.append("-" * 40)
        metrics = self.results['metrics']
        report.append(f"  Overall MSE: {metrics['mse']:.6f}")
        report.append(f"  Overall MAE: {metrics['mae']:.6f}")
        report.append(f"  Overall R²:  {metrics['r2']:.4f}")
        
        report.append("\n  Per-channel metrics:")
        for name in ['u_x', 'u_y', 'u_z', 'wave_height']:
            report.append(f"    {name:12s}: MSE={metrics[f'{name}_mse']:.6f}, "
                         f"MAE={metrics[f'{name}_mae']:.6f}, R²={metrics[f'{name}_r2']:.4f}")
        
        report.append("\n🔬 Physics Residuals:")
        report.append("-" * 40)
        residuals = self.results['physics_residuals']
        for name in ['continuity', 'momentum', 'turbulence']:
            r = residuals[name]
            report.append(f"  {name:12s}: mean={r['mean']:.6f}, std={r['std']:.6f}, max={r['max']:.6f}")
        
        report.append("\n⚡ Inference Speed:")
        report.append("-" * 40)
        timing = self.results['inference_timing']
        report.append(f"  Mean time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms/sample")
        report.append(f"  Throughput: {timing['samples_per_second']:.1f} samples/sec")
        
        report.append("\n📦 Model Info:")
        report.append("-" * 40)
        report.append(f"  Parameters: {self.results['model_info']['parameters']:,}")
        report.append(f"  Device: {self.results['model_info']['device']}")
        
        report.append("\n" + "=" * 60)
        
        # Check success criteria
        report.append("\n✓ Success Criteria Check:")
        report.append("-" * 40)
        
        mse_pass = metrics['mse'] < 0.01
        speed_pass = timing['mean_ms'] < 100
        
        report.append(f"  MSE < 0.01: {'✓ PASS' if mse_pass else '✗ FAIL'} ({metrics['mse']:.6f})")
        report.append(f"  Inference < 100ms: {'✓ PASS' if speed_pass else '✗ FAIL'} ({timing['mean_ms']:.2f}ms)")
        
        report_str = '\n'.join(report)
        print(report_str)
        
        return report_str


def evaluate_model(
    checkpoint_path: str,
    num_samples: int = 140,
    num_nodes: int = 200,
    hidden_channels: int = 64,
    batch_size: int = 4,
    output_dir: str = "./outputs",
) -> Dict:
    """
    Evaluate a trained CoastFlow-GNN model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Dataset size
        num_nodes: Nodes per mesh
        hidden_channels: Model hidden dimension
        batch_size: Evaluation batch size
        output_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    print("Loading dataset...")
    dataset = SyntheticCoastalDataset(
        num_samples=num_samples,
        num_nodes=num_nodes,
        seed=42,
    )
    
    _, _, test_loader = create_data_loaders(dataset, batch_size=batch_size)
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Create model
    model = CoastFlowGNN(
        in_channels=6,
        hidden_channels=hidden_channels,
        out_channels=4,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # Generate report
    output_path = Path(output_dir) / 'evaluation_results.json'
    evaluator.generate_report(str(output_path))
    
    return evaluator.results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoastFlow-GNN")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden channels (must match checkpoint)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        hidden_channels=args.hidden,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
