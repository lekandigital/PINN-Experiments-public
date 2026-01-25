"""
Training configuration for NIF-Cloth4D-Temporal.

Centralized hyperparameter management using dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    in_dim: int = 4                    # Input dimension (x, y, z, t)
    hidden_dim: int = 256              # Hidden layer width
    out_dim: int = 1                   # Output dimension (SDF)
    num_layers: int = 5                # Number of SIREN layers
    
    # Fourier features
    num_freqs: int = 16                # Number of frequency components
    fourier_scale: float = 10.0        # Scale for random projection
    
    # SIREN
    omega_0: float = 30.0              # SIREN frequency parameter
    
    # Temporal conditioning
    use_gru: bool = True               # Enable GRU conditioning
    gru_hidden: int = 128              # GRU hidden dimension
    
    # Regularization
    dropout: float = 0.0               # Dropout probability


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    # Physics loss weights
    lambda_stretch: float = 1.0        # Edge length preservation
    lambda_bend: float = 0.1           # Dihedral angle consistency
    lambda_momentum: float = 0.1       # Newton's 2nd law enforcement
    lambda_collision: float = 10.0     # Penetration penalty
    lambda_self_collision: float = 1.0 # Self-intersection penalty
    
    # Collision parameters
    ground_height: float = 0.0         # Ground plane height
    collision_margin: float = 0.01     # Safety margin for collision
    
    # Scheduled sampling
    eps_start: float = 1.0             # Initial teacher forcing probability
    eps_min: float = 0.2               # Minimum teacher forcing probability
    eps_decay: float = 0.95            # Decay factor per epoch
    schedule_type: str = 'exponential' # 'exponential', 'linear', 'inverse_sigmoid'


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Loss
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = 'adam'            # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'          # 'cosine', 'step', 'plateau', 'none'
    
    # Training loop
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    
    # Sequence handling
    sequence_length: int = 120         # Frames per sequence (4s @ 30fps)
    rollout_length: int = 30           # Max rollout during training
    dt: float = 1.0 / 30.0             # Time step (seconds)
    
    # Mixed precision
    use_amp: bool = True               # Automatic Mixed Precision
    grad_clip_norm: float = 1.0        # Gradient clipping norm
    
    # Curriculum learning
    use_curriculum: bool = True        # Enable curriculum sequence length
    curriculum_min_length: int = 10    # Starting sequence length
    curriculum_warmup_epochs: int = 5  # Epochs before curriculum starts
    curriculum_growth_rate: float = 1.2 # Length growth per epoch
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_epochs: int = 10
    keep_last_n: int = 5
    
    # Logging
    log_every_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = 'nif-cloth4d-temporal'
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Hardware
    device: str = 'cuda'               # 'cuda', 'cpu', 'mps'
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    def get_device(self) -> torch.device:
        """Get torch device from config."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.device(f'cuda:{self.gpu_ids[0]}')
        elif self.device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_fast_dev_config() -> TrainingConfig:
    """Get fast development configuration for testing."""
    config = TrainingConfig()
    
    # Smaller model
    config.model.hidden_dim = 64
    config.model.num_layers = 3
    config.model.num_freqs = 8
    
    # Faster training
    config.epochs = 10
    config.batch_size = 4
    config.rollout_length = 10
    
    # Disable extras
    config.use_wandb = False
    config.use_curriculum = False
    
    return config


def get_l40s_config() -> TrainingConfig:
    """Get configuration optimized for NVIDIA L40S (48GB)."""
    config = TrainingConfig()
    
    # Larger model
    config.model.hidden_dim = 512
    config.model.num_layers = 7
    config.model.num_freqs = 32
    config.model.use_gru = True
    config.model.gru_hidden = 256
    
    # Larger batches (more VRAM)
    config.batch_size = 16
    config.rollout_length = 60
    
    # Full precision for debugging, AMP for production
    config.use_amp = True
    
    return config
