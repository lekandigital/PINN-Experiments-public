"""
Integration Example: Project 09 (HGNN-NIF-Cloth)

Demonstrates how to integrate the shared training library with
Project 09's hybrid GNN + Neural Implicit Field cloth model.

This project combines:
- GRU-based velocity encoder for temporal dynamics
- SIREN decoder for implicit surface representation
- Frame history for stable predictions
"""

import torch
import torch.nn as nn
import math

# Shared training imports
import sys
sys.path.insert(0, '../..')
from shared_training import (
    ScheduledSamplingTrainer,
    CurriculumRolloutScheduler,
    CosineAnnealSchedule,
    RolloutMetrics,
)
from shared_training.adapters import GRUAdapter
from shared_training.configs import Project09Config


class SirenLayer(nn.Module):
    """SIREN layer with sinusoidal activation."""
    
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / in_features) / omega_0,
                    math.sqrt(6 / in_features) / omega_0
                )
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class MockHGNNNIFCloth(nn.Module):
    """Mock HGNN-NIF-Cloth model for demonstration.
    
    Replace with actual import:
        from projects.09-hgnn-nif-cloth__project-space.hgnn-nif-cloth.src.models import HGNNNIFCloth
    
    Architecture:
    1. GRU encoder: Processes frame history to produce latent state
    2. SIREN decoder: Maps (xyz, latent) -> SDF value
    """
    
    def __init__(
        self,
        input_dim=6,  # pos + vel
        hidden_dim=128,
        latent_dim=64,
        siren_hidden=128,
        siren_layers=3,
        omega_0=30.0,
        frame_history=4,
    ):
        super().__init__()
        self.frame_history = frame_history
        self.latent_dim = latent_dim
        
        # GRU encoder for temporal dynamics
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        
        # SIREN decoder for implicit surface
        siren_input = 3 + latent_dim  # xyz + latent
        self.siren = nn.ModuleList([
            SirenLayer(siren_input, siren_hidden, omega_0, is_first=True)
        ])
        for _ in range(siren_layers - 1):
            self.siren.append(SirenLayer(siren_hidden, siren_hidden, omega_0))
        self.sdf_head = nn.Linear(siren_hidden, 1)
        
        # Velocity predictor
        self.vel_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def encode(self, frame_history, hidden=None):
        """Encode frame history into latent representation.
        
        Args:
            frame_history: (B, T, N, 6) history of pos+vel
            hidden: Optional previous GRU hidden state
            
        Returns:
            latent: (B, latent_dim) latent code
            hidden: Updated GRU hidden state
        """
        B, T, N, D = frame_history.shape
        
        # Flatten spatial dimension, process temporally
        x = frame_history.view(B, T, -1)  # (B, T, N*D)
        
        # Simple aggregation (replace with proper GNN in real model)
        x = self.encoder(frame_history.mean(dim=2))  # (B, T, hidden)
        
        # GRU
        output, hidden = self.gru(x, hidden)
        
        # Take last output for latent
        latent = self.latent_proj(output[:, -1])  # (B, latent_dim)
        
        return latent, hidden
    
    def decode_sdf(self, points, latent):
        """Decode SDF values at query points.
        
        Args:
            points: (B, P, 3) query points
            latent: (B, latent_dim) latent code
            
        Returns:
            sdf: (B, P) SDF values
        """
        B, P, _ = points.shape
        
        # Expand latent to match points
        latent_expanded = latent.unsqueeze(1).expand(-1, P, -1)  # (B, P, latent)
        
        # Concatenate
        x = torch.cat([points, latent_expanded], dim=-1)  # (B, P, 3+latent)
        
        # SIREN forward
        for layer in self.siren:
            x = layer(x)
        
        sdf = self.sdf_head(x).squeeze(-1)  # (B, P)
        return sdf
    
    def forward(self, frame_history, query_points=None, hidden=None):
        """Full forward pass.
        
        Args:
            frame_history: (B, T, N, 6) history frames
            query_points: Optional (B, P, 3) points to query SDF
            hidden: Optional GRU hidden state
            
        Returns:
            dict with 'sdf', 'velocity', 'latent', 'hidden'
        """
        # Encode
        latent, new_hidden = self.encode(frame_history, hidden)
        
        # Decode velocity (for next frame prediction)
        velocity = self.vel_decoder(new_hidden.squeeze(0))  # (B, 3) global vel
        
        # Decode SDF if query points provided
        sdf = None
        if query_points is not None:
            sdf = self.decode_sdf(query_points, latent)
        
        return {
            'sdf': sdf,
            'velocity': velocity,
            'latent': latent,
            'hidden': new_hidden,
        }
    
    def init_hidden(self, batch_size, device):
        """Initialize GRU hidden state."""
        return torch.zeros(1, batch_size, 128, device=device)


class Project09Adapter(GRUAdapter):
    """Specialized adapter for Project 09.
    
    Handles:
    - Frame history windowing
    - SDF query point generation
    - Combined position + SDF loss
    """
    
    def __init__(self, model, loss_fn, frame_history=4, **kwargs):
        super().__init__(model, loss_fn, **kwargs)
        self.frame_history_size = frame_history
        self._history_buffer = None
    
    def init_hidden(self, batch):
        """Initialize with frame history buffer."""
        hidden = super().init_hidden(batch)
        
        # Initialize history buffer with first frames
        positions = batch[self.position_key]
        velocities = batch.get(self.velocity_key, torch.zeros_like(positions))
        
        B, T, N, _ = positions.shape
        device = positions.device
        
        # Combine pos + vel
        frames = torch.cat([positions, velocities], dim=-1)  # (B, T, N, 6)
        
        # Take first frame_history frames
        self._history_buffer = frames[:, :self.frame_history_size].clone()
        
        return hidden
    
    def step(self, state, hidden, t, batch):
        """Step with frame history."""
        # Get velocity from batch if available
        if self.velocity_key in batch:
            vel = batch[self.velocity_key][:, t] if batch[self.velocity_key].dim() == 4 else batch[self.velocity_key]
        else:
            vel = torch.zeros_like(state)
        
        # Update history buffer
        new_frame = torch.cat([state, vel], dim=-1).unsqueeze(1)  # (B, 1, N, 6)
        self._history_buffer = torch.cat([
            self._history_buffer[:, 1:],
            new_frame
        ], dim=1)
        
        # Forward through model
        output = self._model(self._history_buffer, hidden=hidden)
        
        # Predict next position using velocity
        predicted_vel = output['velocity'].unsqueeze(1).expand(-1, state.size(1), -1)
        prediction = state + predicted_vel * self.dt
        
        return prediction, output['hidden']


def create_mock_hgnn_nif_batch(batch_size=4, num_nodes=512, seq_len=64):
    """Create mock batch for HGNN-NIF-Cloth."""
    return {
        'positions': torch.randn(batch_size, seq_len, num_nodes, 3),
        'velocities': torch.randn(batch_size, seq_len, num_nodes, 3),
        'query_points': torch.randn(batch_size, 1024, 3),  # SDF query points
        'sdf_values': torch.randn(batch_size, 1024),  # GT SDF
        'edge_index': torch.randint(0, num_nodes, (2, num_nodes * 6)),
    }


def train_hgnn_nif_cloth():
    """Main training function for HGNN-NIF-Cloth."""
    
    # Configuration
    config = Project09Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = MockHGNNNIFCloth(
        input_dim=6,
        hidden_dim=128,
        latent_dim=64,
        siren_hidden=config.siren_hidden_dim,
        siren_layers=config.siren_layers,
        omega_0=config.omega_0,
        frame_history=config.frame_history,
    ).to(device)
    
    # Loss function
    def combined_loss(pred, target):
        return nn.functional.mse_loss(pred, target)
    
    # Adapter
    adapter = Project09Adapter(
        model=model,
        loss_fn=combined_loss,
        frame_history=config.frame_history,
        position_key='positions',
        velocity_key='velocities',
        dt=config.dt,
    )
    
    # Schedule and curriculum
    epsilon_schedule = CosineAnnealSchedule(
        eps_start=config.schedule.eps_start,
        eps_min=config.schedule.eps_min,
        total_steps=config.num_epochs,
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
    
    # Metrics
    metrics = RolloutMetrics(threshold_multiplier=2.0)
    
    # Training loop
    print("=" * 60)
    print("Project 09: HGNN-NIF-Cloth Training")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        rollout_length = curriculum.get_rollout_length(epoch)
        
        # Ensure minimum length for frame history
        rollout_length = max(rollout_length, config.frame_history + 1)
        
        batch = create_mock_hgnn_nif_batch(
            batch_size=config.batch_size,
            num_nodes=512,
            seq_len=rollout_length,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Training step
        loss, info = trainer.train_step(batch, epoch, rollout_length)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | "
                  f"Rollout: {rollout_length:3d} | "
                  f"ε: {info['epsilon']:.3f}")
        
        # Evaluation
        if epoch % config.eval_every == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                eval_batch = create_mock_hgnn_nif_batch(
                    batch_size=2,
                    num_nodes=512,
                    seq_len=256,
                )
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                
                predictions, targets = trainer.evaluate_rollout(
                    eval_batch,
                    max_steps=256,
                )
                
                result = metrics.evaluate(predictions, targets)
                print(f"  → Stability: {result.stability_horizon} | "
                      f"Drift: {result.drift_rate:.4f}")
            
            model.train()
    
    print("\nTraining complete!")
    summary = metrics.get_summary()
    print(f"Best stability horizon: {summary['best_stability_horizon']}")


if __name__ == '__main__':
    train_hgnn_nif_cloth()
