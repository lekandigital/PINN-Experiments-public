"""
HGNN-ClothDyn Training Script

Training loop with:
- Teacher forcing (ground truth inputs during early training)
- Scheduled sampling (gradual transition to autoregressive predictions)
- Mixed precision training (AMP) for efficiency
- Checkpointing and loss tracking

Author: HGNN-ClothDyn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
import numpy as np
import h5py
import argparse
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import random

from model import HGNNClothDyn, ClothSimulator
from mesh_to_graph import mesh_to_graph, build_graph_pyramid, compute_rest_lengths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ClothDataset:
    """
    Dataset loader for cloth simulation sequences.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: Optional[int] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            data_path: Path to HDF5 data file
            sequence_length: Truncate sequences to this length (optional)
            device: Target device
        """
        self.device = device
        
        # Load data
        with h5py.File(data_path, 'r') as f:
            self.positions = torch.tensor(f['positions'][:], dtype=torch.float32)
            self.velocities = torch.tensor(f['velocities'][:], dtype=torch.float32)
            self.edges = torch.tensor(f['edges'][:], dtype=torch.long)
            self.rest_lengths = torch.tensor(f['rest_lengths'][:], dtype=torch.float32)
            
            if 'collisions' in f:
                self.collisions = torch.tensor(f['collisions'][:], dtype=torch.bool)
            else:
                self.collisions = None
        
        # Truncate if needed
        if sequence_length is not None:
            self.positions = self.positions[:sequence_length]
            self.velocities = self.velocities[:sequence_length]
            if self.collisions is not None:
                self.collisions = self.collisions[:sequence_length]
        
        self.num_frames = len(self.positions)
        self.num_nodes = self.positions.shape[1]
        
        logger.info(f"Loaded dataset: {self.num_frames} frames, {self.num_nodes} nodes")
    
    def get_frame(self, frame_idx: int) -> Data:
        """Get a single frame as a PyG Data object."""
        pos = self.positions[frame_idx]
        vel = self.velocities[frame_idx]
        
        # Node features: [pos, vel]
        x = torch.cat([pos, vel], dim=-1)
        
        # Edge index (undirected)
        edge_index = torch.cat([
            self.edges.T,
            self.edges.T.flip(0)
        ], dim=1)
        
        # Edge attributes (rest lengths, duplicated for undirected)
        edge_attr = torch.cat([
            self.rest_lengths,
            self.rest_lengths
        ]).unsqueeze(-1)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            num_nodes=self.num_nodes
        )
        
        return data
    
    def get_sequence(
        self, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None
    ) -> Tuple[List[Data], torch.Tensor, torch.Tensor]:
        """
        Get a sequence of frames.
        
        Returns:
            graphs: List of PyG Data objects
            gt_positions: Ground truth positions (T, N, 3)
            gt_velocities: Ground truth velocities (T, N, 3)
        """
        if end_frame is None:
            end_frame = self.num_frames
        
        graphs = [self.get_frame(i) for i in range(start_frame, end_frame)]
        gt_positions = self.positions[start_frame:end_frame]
        gt_velocities = self.velocities[start_frame:end_frame]
        
        return graphs, gt_positions, gt_velocities


class Trainer:
    """
    Training manager for HGNN-ClothDyn.
    """
    
    def __init__(
        self,
        model: HGNNClothDyn,
        dataset: ClothDataset,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Args:
            model: HGNNClothDyn model
            dataset: Training dataset
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Mixed precision - DISABLED by default for numerical stability on consumer GPUs
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Scheduled sampling
        self.use_scheduled_sampling = config.get('use_scheduled_sampling', True)
        self.sampling_start_epoch = config.get('sampling_start_epoch', 5)
        self.sampling_initial_prob = config.get('sampling_initial_prob', 0.1)
        self.sampling_increment = config.get('sampling_increment', 0.05)
        
        # Loss weights
        self.pos_weight = config.get('position_weight', 1.0)
        self.vel_weight = config.get('velocity_weight', 0.1)
        self.edge_weight = config.get('edge_length_weight', 0.01)
        
        # Tracking
        self.loss_history = []
        self.best_loss = float('inf')
        
        # Build graph pyramid for dataset
        self._build_graph_structures()
        
        logger.info(f"Trainer initialized: AMP={self.use_amp}, "
                    f"scheduled_sampling={self.use_scheduled_sampling}")
    
    def _build_graph_structures(self):
        """Pre-build graph pyramid structures."""
        # Get initial graph
        initial_data = self.dataset.get_frame(0).to(self.device)
        
        # Build pyramid
        graphs, cluster_maps = build_graph_pyramid(initial_data, num_levels=2)
        
        self.fine_template = graphs[0]
        self.coarse_template = graphs[1] if len(graphs) > 1 else None
        self.cluster_map = cluster_maps[0] if cluster_maps else None
        
        if self.coarse_template is not None:
            self.coarse_template = self.coarse_template.to(self.device)
        if self.cluster_map is not None:
            self.cluster_map = self.cluster_map.to(self.device)
        
        logger.info(f"Graph pyramid: fine={self.fine_template.num_nodes} nodes, "
                    f"coarse={self.coarse_template.num_nodes if self.coarse_template else 'N/A'} nodes")
    
    def compute_loss(
        self,
        pred_pos: torch.Tensor,
        pred_vel: torch.Tensor,
        gt_pos: torch.Tensor,
        gt_vel: torch.Tensor,
        edges: torch.Tensor,
        rest_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            pred_pos: Predicted positions (N, 3)
            pred_vel: Predicted velocities (N, 3)
            gt_pos: Ground truth positions (N, 3)
            gt_vel: Ground truth velocities (N, 3)
            edges: Edge indices (E, 2)
            rest_lengths: Rest lengths (E,)
            
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary of individual losses
        """
        # Check for NaN inputs
        if torch.isnan(pred_pos).any() or torch.isnan(pred_vel).any():
            logger.warning("NaN detected in predictions!")
            # Return zero loss to skip this step
            zero_loss = torch.tensor(0.0, device=pred_pos.device, requires_grad=True)
            return zero_loss, {'position': 0, 'velocity': 0, 'edge_length': 0, 'total': 0}
        
        # Position MSE
        pos_loss = nn.functional.mse_loss(pred_pos, gt_pos)
        
        # Velocity MSE
        vel_loss = nn.functional.mse_loss(pred_vel, gt_vel)
        
        # Edge length preservation loss
        pred_lengths = torch.norm(
            pred_pos[edges[:, 0]] - pred_pos[edges[:, 1]], 
            dim=-1
        )
        edge_loss = nn.functional.mse_loss(pred_lengths, rest_lengths)
        
        # Combined loss with clipping to prevent explosion
        total_loss = (
            self.pos_weight * pos_loss +
            self.vel_weight * vel_loss +
            self.edge_weight * edge_loss
        )
        
        # Clip loss to prevent explosion
        total_loss = torch.clamp(total_loss, max=1000.0)
        
        loss_dict = {
            'position': pos_loss.item() if not torch.isnan(pos_loss) else 0,
            'velocity': vel_loss.item() if not torch.isnan(vel_loss) else 0,
            'edge_length': edge_loss.item() if not torch.isnan(edge_loss) else 0,
            'total': total_loss.item() if not torch.isnan(total_loss) else 0
        }
        
        return total_loss, loss_dict
    
    def get_sampling_prob(self, epoch: int) -> float:
        """Get scheduled sampling probability for current epoch."""
        if not self.use_scheduled_sampling:
            return 0.0
        if epoch < self.sampling_start_epoch:
            return 0.0
        
        prob = self.sampling_initial_prob + self.sampling_increment * (epoch - self.sampling_start_epoch)
        return min(prob, 1.0)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        sampling_prob = self.get_sampling_prob(epoch)
        logger.info(f"Epoch {epoch}: sampling_prob={sampling_prob:.2f}")
        
        # Get sequence
        _, gt_positions, gt_velocities = self.dataset.get_sequence()
        gt_positions = gt_positions.to(self.device)
        gt_velocities = gt_velocities.to(self.device)

        # Normalize inputs for numerical stability
        pos_mean = gt_positions.mean(dim=(0, 1), keepdim=True)
        pos_std = gt_positions.std() + 1e-8
        gt_positions = (gt_positions - pos_mean) / pos_std
        gt_velocities = gt_velocities / pos_std  # Scale velocities consistently

        edges = self.dataset.edges.to(self.device)
        rest_lengths = self.dataset.rest_lengths.to(self.device)
        rest_lengths = rest_lengths / pos_std  # Scale rest lengths consistently
        
        epoch_losses = {'position': 0, 'velocity': 0, 'edge_length': 0, 'total': 0}
        num_steps = 0
        
        # Current state (start with ground truth)
        current_pos = gt_positions[0]
        current_vel = gt_velocities[0]
        
        # Iterate through sequence
        for t in range(1, len(gt_positions)):
            self.optimizer.zero_grad()
            
            # Prepare input data
            x = torch.cat([current_pos, current_vel], dim=-1)
            
            data = Data(
                x=x,
                edge_index=self.fine_template.edge_index,
                edge_attr=self.fine_template.edge_attr,
                pos=current_pos,
                num_nodes=self.dataset.num_nodes
            ).to(self.device)
            
            # Update coarse positions
            if self.coarse_template is not None:
                coarse_pos = self._pool_positions(current_pos, self.cluster_map, 
                                                   self.coarse_template.num_nodes)
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
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                delta_vel = self.model(data, coarse_data, self.cluster_map)
                
                # Predict next state
                pred_vel = (current_vel + delta_vel) * 0.99  # Damping
                pred_pos = current_pos + pred_vel * 0.01  # dt
                
                # Compute loss against ground truth
                loss, loss_dict = self.compute_loss(
                    pred_pos, pred_vel,
                    gt_positions[t], gt_velocities[t],
                    edges, rest_lengths
                )
            
            # Check for NaN loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping step {t} due to NaN/Inf loss")
                continue

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Check for NaN gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                            self.config.get('gradient_clip', 1.0))
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or self.scaler.get_scale() < 1e-4:
                    logger.warning(f"Skipping step {t} due to numerical instability (grad_norm={grad_norm}, scale={self.scaler.get_scale()})")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Check for NaN gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                            self.config.get('gradient_clip', 1.0))
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Skipping step {t} due to NaN/Inf gradients (grad_norm={grad_norm})")
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.step()

            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            num_steps += 1
            
            # Scheduled sampling: decide whether to use prediction or ground truth
            if random.random() < sampling_prob:
                # Use model prediction for next step
                current_pos = pred_pos.detach()
                current_vel = pred_vel.detach()
            else:
                # Use ground truth (teacher forcing)
                current_pos = gt_positions[t]
                current_vel = gt_velocities[t]
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_steps
        
        return epoch_losses
    
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
    
    def validate(self) -> Dict[str, float]:
        """Run validation on held-out frames."""
        self.model.eval()
        
        # Use last 20% of sequence for validation
        val_start = int(self.dataset.num_frames * 0.8)
        _, gt_positions, gt_velocities = self.dataset.get_sequence(
            start_frame=val_start
        )
        gt_positions = gt_positions.to(self.device)
        gt_velocities = gt_velocities.to(self.device)
        
        edges = self.dataset.edges.to(self.device)
        rest_lengths = self.dataset.rest_lengths.to(self.device)
        
        val_losses = {'position': 0, 'velocity': 0, 'edge_length': 0, 'total': 0}
        num_steps = 0
        
        current_pos = gt_positions[0]
        current_vel = gt_velocities[0]
        
        with torch.no_grad():
            for t in range(1, len(gt_positions)):
                x = torch.cat([current_pos, current_vel], dim=-1)
                
                data = Data(
                    x=x,
                    edge_index=self.fine_template.edge_index,
                    edge_attr=self.fine_template.edge_attr,
                    pos=current_pos,
                    num_nodes=self.dataset.num_nodes
                ).to(self.device)
                
                if self.coarse_template is not None:
                    coarse_pos = self._pool_positions(current_pos, self.cluster_map,
                                                       self.coarse_template.num_nodes)
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
                
                delta_vel = self.model(data, coarse_data, self.cluster_map)
                pred_vel = (current_vel + delta_vel) * 0.99
                pred_pos = current_pos + pred_vel * 0.01
                
                _, loss_dict = self.compute_loss(
                    pred_pos, pred_vel,
                    gt_positions[t], gt_velocities[t],
                    edges, rest_lengths
                )
                
                for key in val_losses:
                    val_losses[key] += loss_dict[key]
                num_steps += 1
                
                # Autoregressive: always use predictions in validation
                current_pos = pred_pos
                current_vel = pred_vel
        
        for key in val_losses:
            val_losses[key] /= num_steps
        
        return val_losses
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory for saving checkpoints
            checkpoint_interval: Epochs between checkpoints
            
        Returns:
            Dictionary of loss histories
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = datetime.now()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Record history
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_losses['total']:.6f} "
                f"(pos: {train_losses['position']:.6f}, "
                f"vel: {train_losses['velocity']:.6f}, "
                f"edge: {train_losses['edge_length']:.6f}) - "
                f"Val Loss: {val_losses['total']:.6f}"
            )
            
            # Save best model
            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self.save_checkpoint(checkpoint_dir / 'best_model.pt', epoch, val_losses)
                logger.info(f"  → New best model saved!")
            
            # Periodic checkpoint
            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'checkpoint_epoch{epoch}.pt',
                    epoch, train_losses
                )
        
        elapsed = datetime.now() - start_time
        logger.info(f"Training complete in {elapsed}")
        
        # Save final model
        self.save_checkpoint(checkpoint_dir / 'final_model.pt', num_epochs, train_losses)
        
        # Save loss history
        with open(checkpoint_dir / 'loss_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        losses: Dict[str, float]
    ):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description='Train HGNN-ClothDyn model')
    
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to training data (HDF5)')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Epochs between checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling mode (1 epoch only)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with CLI args
    config['learning_rate'] = args.lr
    config['use_amp'] = not args.no_amp
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    dataset = ClothDataset(args.data, device=device)
    
    # Create model
    model = HGNNClothDyn(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        num_message_passes=3,
        num_levels=2
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, dataset, config, device)
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Training
    num_epochs = 1 if args.profile else args.epochs
    
    if args.profile:
        logger.info("Profiling mode: running 1 epoch")
        import torch.profiler as profiler
        
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            trainer.train(
                num_epochs=1,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_interval=1
            )
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace("trace.json")
    else:
        history = trainer.train(
            num_epochs=num_epochs,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval
        )
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
