"""
Baseline Training Script for ClothGNN.

Trains ClothGNN without knowledge distillation using physics-based losses.
This establishes a baseline performance before applying distillation.

Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml
    python scripts/train_baseline.py --epochs 100 --batch-size 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from models.clothgnn import ClothGNNModel
from data.dataloader import create_dataloader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Trainer for ClothGNN with physics-based losses (no teacher).
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lambda_position: float = 1.0,
        lambda_velocity: float = 0.1,
        lambda_edge: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        
        self.lambda_position = lambda_position
        self.lambda_velocity = lambda_velocity
        self.lambda_edge = lambda_edge
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = None
    
    def _prepare_input(self, batch: Dict) -> torch.Tensor:
        """Prepare node features from batch."""
        positions = batch['positions']  # [B, N, 3] or [B, T, N, 3]
        velocities = batch['velocities']
        
        # Handle sequence vs single-step
        if positions.dim() == 4:
            # Sequence: take first frame
            pos = positions[:, 0]  # [B, N, 3]
            vel = velocities[:, 0]
        else:
            pos = positions
            vel = velocities
        
        # Combine position + velocity + padding
        node_features = torch.cat([
            pos, vel,
            torch.zeros_like(pos[..., :10]),  # Padding to 16 features
        ], dim=-1)
        
        return node_features
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_pos_loss = 0
        total_edge_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Prepare input features
            node_features = self._prepare_input(batch)
            edge_index = batch['edge_index']
            rest_lengths = batch['rest_lengths']
            node_mask = batch['node_mask']
            
            # Get current positions for computing displacement
            if batch['positions'].dim() == 4:
                current_pos = batch['positions'][:, 0]
                target_pos = batch['positions'][:, 1]
            else:
                current_pos = batch['positions']
                target_pos = batch['target_positions']
            
            target_displacement = target_pos - current_pos
            
            self.optimizer.zero_grad()
            
            # Forward pass
            B, N = node_features.shape[:2]
            hidden = self.model.init_hidden(N, self.device)
            hidden = hidden.unsqueeze(0).expand(B, -1, -1)
            
            # Handle batched edge_index - we need to process each sample
            total_batch_loss = 0
            for b in range(B):
                n_nodes = batch['node_counts'][b].item()
                n_edges = batch['edge_counts'][b].item()
                
                # Create PyG-like data object
                class Data:
                    pass
                
                data = Data()
                data.x = node_features[b, :n_nodes]
                data.edge_index = edge_index[b, :, :n_edges]
                data.pos = current_pos[b, :n_nodes]
                
                h = hidden[b, :n_nodes]
                
                pred_displacement, _ = self.model(data, h)
                
                # Position loss
                gt_disp = target_displacement[b, :n_nodes]
                pos_loss = F.mse_loss(pred_displacement, gt_disp)
                
                # Edge length preservation
                pred_pos = current_pos[b, :n_nodes] + pred_displacement
                src, dst = edge_index[b, :, :n_edges]
                pred_lengths = torch.norm(pred_pos[dst] - pred_pos[src], dim=-1)
                rest_len = rest_lengths[b, :n_edges]
                edge_loss = F.mse_loss(pred_lengths, rest_len)
                
                sample_loss = (
                    self.lambda_position * pos_loss +
                    self.lambda_edge * edge_loss
                )
                
                total_batch_loss = total_batch_loss + sample_loss
                total_pos_loss += pos_loss.item()
                total_edge_loss += edge_loss.item()
            
            loss = total_batch_loss / B
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss / n_batches:.4f}",
                'pos': f"{total_pos_loss / (n_batches * B):.4f}",
                'edge': f"{total_edge_loss / (n_batches * B):.6f}",
            })
        
        return {
            'loss': total_loss / n_batches,
            'position_loss': total_pos_loss / (n_batches * B),
            'edge_loss': total_edge_loss / (n_batches * B),
        }
    
    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_rmse = 0
        total_edge_error = 0
        n_samples = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            node_features = self._prepare_input(batch)
            edge_index = batch['edge_index']
            rest_lengths = batch['rest_lengths']
            
            if batch['positions'].dim() == 4:
                current_pos = batch['positions'][:, 0]
                target_pos = batch['positions'][:, 1]
            else:
                current_pos = batch['positions']
                target_pos = batch['target_positions']
            
            B, N = node_features.shape[:2]
            hidden = self.model.init_hidden(N, self.device)
            hidden = hidden.unsqueeze(0).expand(B, -1, -1)
            
            for b in range(B):
                n_nodes = batch['node_counts'][b].item()
                n_edges = batch['edge_counts'][b].item()
                
                class Data:
                    pass
                
                data = Data()
                data.x = node_features[b, :n_nodes]
                data.edge_index = edge_index[b, :, :n_edges]
                data.pos = current_pos[b, :n_nodes]
                
                h = hidden[b, :n_nodes]
                
                pred_displacement, _ = self.model(data, h)
                
                # RMSE
                gt_disp = target_pos[b, :n_nodes] - current_pos[b, :n_nodes]
                rmse = torch.sqrt(F.mse_loss(pred_displacement, gt_disp))
                total_rmse += rmse.item()
                
                # Edge error
                pred_pos = current_pos[b, :n_nodes] + pred_displacement
                src, dst = edge_index[b, :, :n_edges]
                pred_lengths = torch.norm(pred_pos[dst] - pred_pos[src], dim=-1)
                edge_error = torch.abs(pred_lengths - rest_lengths[b, :n_edges]).mean()
                total_edge_error += edge_error.item()
                
                n_samples += 1
        
        return {
            'val_rmse': total_rmse / n_samples,
            'val_edge_error': total_edge_error / n_samples,
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict,
    ):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'node_feat_dim': self.model.encoder.lin_node.in_features,
                'hidden_dim': self.model.hidden_dim,
            },
        }, path)
    
    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 100,
        save_dir: str = "checkpoints/baseline/",
        eval_every: int = 5,
    ) -> Dict:
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        
        best_rmse = float('inf')
        history = []
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate(val_loader)
                
                logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_rmse={val_metrics['val_rmse']:.4f}, "
                    f"val_edge_error={val_metrics['val_edge_error']:.6f}"
                )
                
                # Save best
                if val_metrics['val_rmse'] < best_rmse:
                    best_rmse = val_metrics['val_rmse']
                    self.save_checkpoint(
                        save_dir / "best_model.pt",
                        epoch,
                        {**train_metrics, **val_metrics},
                    )
                    logger.info(f"  -> New best RMSE: {best_rmse:.4f}")
            else:
                val_metrics = {}
            
            # Save periodic checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(
                    save_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    epoch,
                    train_metrics,
                )
            
            history.append({**train_metrics, **val_metrics, 'epoch': epoch})
            
            if self.scheduler:
                self.scheduler.step()
        
        # Save final checkpoint
        self.save_checkpoint(
            save_dir / "final_model.pt",
            n_epochs - 1,
            train_metrics,
        )
        
        # Save history
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\nTraining complete. Best RMSE: {best_rmse:.4f}")
        logger.info(f"Target was < 0.1")
        
        return {
            'best_rmse': best_rmse,
            'history': history,
        }


def main():
    parser = argparse.ArgumentParser(description="Train ClothGNN baseline")
    parser.add_argument("--data", type=str, default="data/cloth_dynamics.h5",
                        help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension")
    parser.add_argument("--save-dir", type=str, default="checkpoints/baseline/",
                        help="Checkpoint save directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Create model
    model = ClothGNNModel(
        node_feat_dim=16,
        hidden_dim=args.hidden_dim,
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    
    # Create data loaders
    data_path = Path(args.data)
    splits_path = data_path.parent / "splits.json"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Generate data first with: python data/generate_dataset.py")
        return
    
    train_loader = create_dataloader(
        str(data_path),
        batch_size=args.batch_size,
        split='train' if splits_path.exists() else None,
        splits_path=str(splits_path) if splits_path.exists() else None,
        single_step=True,
    )
    
    val_loader = create_dataloader(
        str(data_path),
        batch_size=args.batch_size,
        split='val' if splits_path.exists() else None,
        splits_path=str(splits_path) if splits_path.exists() else None,
        single_step=True,
        shuffle=False,
    )
    
    # Create trainer
    trainer = BaselineTrainer(
        model,
        device=args.device,
        lr=args.lr,
    )
    
    # Train
    result = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        save_dir=args.save_dir,
    )
    
    logger.info(f"Training complete!")
    logger.info(f"Best RMSE: {result['best_rmse']:.4f}")


if __name__ == "__main__":
    main()
