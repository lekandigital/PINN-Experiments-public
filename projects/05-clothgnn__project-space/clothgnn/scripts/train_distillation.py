"""
Knowledge Distillation Training Script for ClothGNN.

Trains ClothGNN (student) to match HGNN-NIF-Cloth (teacher) outputs.

Two modes:
1. With teacher checkpoint: Full distillation with soft targets
2. Without teacher: Uses larger baseline ClothGNN as teacher (self-distillation)

Usage:
    # Full distillation (requires teacher checkpoint)
    python scripts/train_distillation.py \
        --teacher-checkpoint path/to/teacher.pt \
        --epochs 200

    # Self-distillation (baseline -> lite)
    python scripts/train_distillation.py \
        --self-distill \
        --teacher-checkpoint checkpoints/baseline/best_model.pt \
        --epochs 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parents[2]))  # PINN-Experiments root

from models.clothgnn import ClothGNNModel
from data.dataloader import create_dataloader
from register_models import ClothGNNLite


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Components:
    1. Soft target loss: Match teacher outputs (temperature-scaled)
    2. Hard target loss: Match ground truth (if available)
    3. Physics loss: Edge length preservation
    """
    
    def __init__(
        self,
        alpha_soft: float = 0.6,
        alpha_hard: float = 0.3,
        alpha_physics: float = 0.1,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.alpha_soft = alpha_soft
        self.alpha_hard = alpha_hard
        self.alpha_physics = alpha_physics
        self.temperature = temperature
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        ground_truth: Optional[torch.Tensor],
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Returns:
            loss: Combined loss
            metrics: Dict of individual components
        """
        # Soft target loss (distillation)
        student_scaled = student_output / self.temperature
        teacher_scaled = teacher_output / self.temperature
        soft_loss = F.mse_loss(student_scaled, teacher_scaled) * (self.temperature ** 2)
        
        # Hard target loss (if ground truth available)
        if ground_truth is not None:
            hard_loss = F.mse_loss(student_output, ground_truth)
        else:
            hard_loss = torch.tensor(0.0, device=student_output.device)
        
        # Physics loss: edge length preservation
        new_positions = positions + student_output
        src, dst = edge_index
        current_lengths = torch.norm(new_positions[dst] - new_positions[src], dim=-1)
        physics_loss = F.mse_loss(current_lengths, rest_lengths)
        
        # Combined
        loss = (
            self.alpha_soft * soft_loss +
            self.alpha_hard * hard_loss +
            self.alpha_physics * physics_loss
        )
        
        metrics = {
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': loss.item(),
        }
        
        return loss, metrics


class DistillationTrainer:
    """
    Trainer for knowledge distillation.
    """
    
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str = "cuda",
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        temperature: float = 3.0,
    ):
        self.device = device
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.criterion = DistillationLoss(temperature=temperature)
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = None
        
        # Count parameters
        student_params = sum(p.numel() for p in self.student.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        logger.info(f"Student parameters: {student_params:,}")
        logger.info(f"Teacher parameters: {teacher_params:,}")
        logger.info(f"Compression ratio: {teacher_params / student_params:.2f}x")
    
    def _prepare_student_input(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input for student model."""
        if batch['positions'].dim() == 4:
            pos = batch['positions'][:, 0]
            vel = batch['velocities'][:, 0]
        else:
            pos = batch['positions']
            vel = batch['velocities']
        
        # Student expects [B, N, 6] (pos + vel)
        node_features = torch.cat([pos, vel], dim=-1)
        
        return node_features, pos
    
    def _prepare_teacher_input(self, batch: Dict) -> torch.Tensor:
        """Prepare input for teacher model."""
        if batch['positions'].dim() == 4:
            pos = batch['positions'][:, 0]
            vel = batch['velocities'][:, 0]
        else:
            pos = batch['positions']
            vel = batch['velocities']
        
        # Teacher (ClothGNNModel) expects 16-dim features
        padding = torch.zeros(*pos.shape[:-1], 10, device=pos.device)
        node_features = torch.cat([pos, vel, padding], dim=-1)
        
        return node_features
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        
        total_metrics = {
            'soft_loss': 0,
            'hard_loss': 0,
            'physics_loss': 0,
            'total_loss': 0,
        }
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Prepare inputs
            student_input, positions = self._prepare_student_input(batch)
            teacher_input = self._prepare_teacher_input(batch)
            
            edge_index = batch['edge_index']
            rest_lengths = batch['rest_lengths']
            node_mask = batch['node_mask']
            
            # Get ground truth if available
            if batch['positions'].dim() == 4:
                target_pos = batch['positions'][:, 1]
                ground_truth = target_pos - batch['positions'][:, 0]
            elif 'target_displacement' in batch:
                ground_truth = batch['target_displacement']
            else:
                ground_truth = None
            
            B = student_input.shape[0]
            batch_loss = 0
            
            for b in range(B):
                n_nodes = batch['node_counts'][b].item()
                n_edges = batch['edge_counts'][b].item()
                
                # Student forward
                s_input = student_input[b, :n_nodes]
                s_edge = edge_index[b, :, :n_edges]
                s_output, _ = self.student(s_input, s_edge)
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    t_input = teacher_input[b, :n_nodes]
                    
                    # Create data object for teacher
                    class Data:
                        pass
                    data = Data()
                    data.x = t_input
                    data.edge_index = s_edge
                    data.pos = positions[b, :n_nodes]
                    
                    hidden = self.teacher.init_hidden(n_nodes, self.device)
                    t_output, _ = self.teacher(data, hidden)
                
                # Ground truth for this sample
                gt = ground_truth[b, :n_nodes] if ground_truth is not None else None
                
                # Compute loss
                loss, metrics = self.criterion(
                    s_output,
                    t_output,
                    gt,
                    positions[b, :n_nodes],
                    s_edge,
                    rest_lengths[b, :n_edges],
                )
                
                batch_loss = batch_loss + loss
                
                for k in total_metrics:
                    total_metrics[k] += metrics[k]
            
            (batch_loss / B).backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_metrics['total_loss'] / (n_batches * B):.4f}",
                'soft': f"{total_metrics['soft_loss'] / (n_batches * B):.4f}",
            })
        
        return {k: v / (n_batches * B) for k, v in total_metrics.items()}
    
    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """Validate model."""
        self.student.eval()
        
        total_rmse = 0
        total_distill_error = 0
        total_edge_error = 0
        n_samples = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            student_input, positions = self._prepare_student_input(batch)
            teacher_input = self._prepare_teacher_input(batch)
            edge_index = batch['edge_index']
            rest_lengths = batch['rest_lengths']
            
            if batch['positions'].dim() == 4:
                target_pos = batch['positions'][:, 1]
                ground_truth = target_pos - batch['positions'][:, 0]
            elif 'target_displacement' in batch:
                ground_truth = batch['target_displacement']
            else:
                ground_truth = None
            
            B = student_input.shape[0]
            
            for b in range(B):
                n_nodes = batch['node_counts'][b].item()
                n_edges = batch['edge_counts'][b].item()
                
                s_input = student_input[b, :n_nodes]
                s_edge = edge_index[b, :, :n_edges]
                s_output, _ = self.student(s_input, s_edge)
                
                # Teacher
                t_input = teacher_input[b, :n_nodes]
                class Data:
                    pass
                data = Data()
                data.x = t_input
                data.edge_index = s_edge
                data.pos = positions[b, :n_nodes]
                hidden = self.teacher.init_hidden(n_nodes, self.device)
                t_output, _ = self.teacher(data, hidden)
                
                # RMSE against ground truth
                if ground_truth is not None:
                    rmse = torch.sqrt(F.mse_loss(s_output, ground_truth[b, :n_nodes]))
                    total_rmse += rmse.item()
                
                # Distillation error
                distill_error = torch.sqrt(F.mse_loss(s_output, t_output))
                total_distill_error += distill_error.item()
                
                # Edge error
                new_pos = positions[b, :n_nodes] + s_output
                src, dst = s_edge
                lengths = torch.norm(new_pos[dst] - new_pos[src], dim=-1)
                edge_error = torch.abs(lengths - rest_lengths[b, :n_edges]).mean()
                total_edge_error += edge_error.item()
                
                n_samples += 1
        
        return {
            'val_rmse': total_rmse / n_samples if ground_truth is not None else 0,
            'val_distill_error': total_distill_error / n_samples,
            'val_edge_error': total_edge_error / n_samples,
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
    
    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 200,
        save_dir: str = "checkpoints/distilled/",
        eval_every: int = 5,
    ) -> Dict:
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        
        best_rmse = float('inf')
        history = []
        
        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate(val_loader)
                
                logger.info(
                    f"Epoch {epoch}: loss={train_metrics['total_loss']:.4f}, "
                    f"rmse={val_metrics['val_rmse']:.4f}, "
                    f"distill_err={val_metrics['val_distill_error']:.4f}"
                )
                
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
            
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(
                    save_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    epoch,
                    train_metrics,
                )
            
            history.append({**train_metrics, **val_metrics, 'epoch': epoch})
            
            if self.scheduler:
                self.scheduler.step()
        
        self.save_checkpoint(save_dir / "final_model.pt", n_epochs - 1, train_metrics)
        
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\nDistillation complete. Best RMSE: {best_rmse:.4f}")
        
        return {'best_rmse': best_rmse, 'history': history}


def load_teacher_model(
    checkpoint_path: str,
    device: str,
    is_self_distill: bool = False,
) -> nn.Module:
    """Load teacher model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if is_self_distill:
        # Load ClothGNNModel as teacher
        config = checkpoint.get('config', {'node_feat_dim': 16, 'hidden_dim': 64})
        model = ClothGNNModel(**config)
    else:
        # Try to load HGNN teacher
        try:
            sys.path.insert(0, str(project_root.parents[2] / 'projects' / 
                                   '09-hgnn-nif-cloth__project-space' / 'hgnn-nif-cloth' / 'src'))
            from register_models import HGNNTeacherWrapper
            
            model = HGNNTeacherWrapper(
                in_dim=6,
                hidden_dim=64,
            )
        except ImportError:
            logger.warning("Could not load HGNN, using ClothGNNModel as teacher")
            config = checkpoint.get('config', {'node_feat_dim': 16, 'hidden_dim': 64})
            model = ClothGNNModel(**config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ClothGNN with distillation")
    parser.add_argument("--data", type=str, default="data/cloth_dynamics.h5")
    parser.add_argument("--teacher-checkpoint", type=str, required=True,
                        help="Path to teacher checkpoint")
    parser.add_argument("--self-distill", action="store_true",
                        help="Use baseline ClothGNN as teacher")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--save-dir", type=str, default="checkpoints/distilled/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load teacher
    logger.info(f"Loading teacher from {args.teacher_checkpoint}")
    teacher = load_teacher_model(
        args.teacher_checkpoint,
        args.device,
        is_self_distill=args.self_distill,
    )
    
    # Create student (lightweight version)
    student = ClothGNNLite(
        node_input_dim=6,
        edge_input_dim=3,
        node_hidden_dim=32,
        edge_hidden_dim=16,
        num_message_passes=3,
        output_dim=3,
    )
    
    # Create data loaders
    data_path = Path(args.data)
    splits_path = data_path.parent / "splits.json"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
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
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        device=args.device,
        lr=args.lr,
        temperature=args.temperature,
    )
    
    # Train
    result = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        save_dir=args.save_dir,
    )
    
    logger.info(f"Training complete! Best RMSE: {result['best_rmse']:.4f}")


if __name__ == "__main__":
    main()
