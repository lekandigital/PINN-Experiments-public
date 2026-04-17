"""
Integration Example: Project 05 (ClothGNN)

Demonstrates how to integrate the shared training library with
Project 05's GNN-based cloth simulation model.

This example shows:
1. Setting up the adapter for ClothGNN
2. Configuring scheduled sampling and curriculum
3. Running training with the ScheduledSamplingTrainer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Shared training imports
import sys
sys.path.insert(0, '../..')
from shared_training import (
    ScheduledSamplingTrainer,
    CurriculumRolloutScheduler,
    ExponentialDecaySchedule,
    RolloutMetrics,
)
from shared_training.adapters import GNNAdapter
from shared_training.configs import Project05Config

# Project 05 imports (adjust path as needed)
# from clothgnn.models import ClothGNN
# from clothgnn.data import ClothDataset


class MockClothGNN(nn.Module):
    """Mock ClothGNN for demonstration purposes.
    
    Replace this with actual import:
        from projects.05-clothgnn__project-space.clothgnn.models import ClothGNN
    """
    def __init__(self, node_dim=6, hidden_dim=128, output_dim=3, num_layers=4):
        super().__init__()
        self.encoder = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, data, hidden=None):
        """Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, pos
            hidden: Optional GRU hidden state
            
        Returns:
            displacement: (N, 3) predicted displacement
            new_hidden: Updated hidden state
        """
        x = self.encoder(data.x)
        
        for layer in self.layers:
            x = torch.relu(layer(x))
        
        # GRU for temporal consistency
        if hidden is not None:
            x = self.gru(x, hidden)
        
        displacement = self.decoder(x)
        return displacement, x  # x is the new hidden state
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        # For simplicity, return None (adapter will handle)
        return None


def create_mock_batch(batch_size=4, num_nodes=256, seq_len=64):
    """Create mock batch for demonstration."""
    return {
        'positions': torch.randn(batch_size, seq_len, num_nodes, 3),
        'velocities': torch.randn(batch_size, seq_len, num_nodes, 3),
        'edge_index': torch.randint(0, num_nodes, (2, num_nodes * 6)),
        'rest_lengths': torch.rand(num_nodes * 6) * 0.1,
    }


def train_clothgnn():
    """Main training function for ClothGNN with shared training infrastructure."""
    
    # Configuration
    config = Project05Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = MockClothGNN(
        node_dim=6,  # pos + vel
        hidden_dim=config.hidden_dim,
        output_dim=3,
        num_layers=config.gnn_layers,
    ).to(device)
    
    # Loss function
    def position_loss(pred, target):
        return nn.functional.mse_loss(pred, target)
    
    # Create adapter
    adapter = GNNAdapter(
        model=model,
        loss_fn=position_loss,
        position_key='positions',
        velocity_key='velocities',
        edge_key='edge_index',
        rest_lengths_key='rest_lengths',
        dt=config.dt,
        damping=config.physics.damping,
        physics_losses={
            'edge_strain': lambda **kw: torch.tensor(0.0),  # Simplified
        },
        physics_loss_weights={
            'edge_strain': config.physics.edge_strain_weight,
        },
    )
    
    # Schedule and curriculum
    epsilon_schedule = ExponentialDecaySchedule(
        eps_start=config.schedule.eps_start,
        eps_min=config.schedule.eps_min,
        decay_rate=config.schedule.decay_rate,
    )
    
    curriculum = CurriculumRolloutScheduler(
        total_epochs=config.num_epochs,
        stages=config.curriculum.stages,
    )
    
    # Trainer
    trainer = ScheduledSamplingTrainer(
        adapter=adapter,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        ),
        epsilon_schedule=epsilon_schedule,
        max_rollout_length=config.sequence_length,
        use_amp=config.optimizer.use_amp,
        gradient_clip=config.optimizer.gradient_clip,
    )
    
    # Metrics tracker
    metrics = RolloutMetrics(threshold_multiplier=2.0)
    
    # Training loop
    print("=" * 60)
    print("Project 05: ClothGNN Training with Shared Training Library")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        # Update curriculum
        rollout_length = curriculum.get_rollout_length(epoch)
        
        # Create mock batch (replace with real DataLoader)
        batch = create_mock_batch(
            batch_size=config.batch_size,
            num_nodes=256,
            seq_len=rollout_length,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Training step
        loss, info = trainer.train_step(batch, epoch, rollout_length)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | "
                  f"Rollout: {rollout_length:3d} | "
                  f"ε: {info['epsilon']:.3f} | "
                  f"TF ratio: {info['teacher_forced_ratio']:.2%}")
        
        # Evaluation
        if epoch % config.eval_every == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                # Generate full rollout
                eval_batch = create_mock_batch(
                    batch_size=2, 
                    num_nodes=256, 
                    seq_len=256
                )
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                
                predictions, targets = trainer.evaluate_rollout(
                    eval_batch, 
                    max_steps=256
                )
                
                result = metrics.evaluate(predictions, targets)
                print(f"  → Stability horizon: {result.stability_horizon} | "
                      f"Drift rate: {result.drift_rate:.4f}")
            
            model.train()
    
    print("\nTraining complete!")
    summary = metrics.get_summary()
    print(f"Best stability horizon: {summary['best_stability_horizon']}")
    print(f"Mean drift rate: {summary['mean_drift_rate']:.4f}")


if __name__ == '__main__':
    train_clothgnn()
