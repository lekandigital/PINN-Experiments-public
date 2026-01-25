"""
Configuration Management for Cell-Path PINNs

Provides dataclass-based configuration with YAML serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import os


@dataclass
class TrainingConfig:
    """
    Training configuration for CellPathModel.

    All parameters can be loaded from/saved to YAML files.
    """
    # Model architecture
    hidden_dim: int = 64

    # Training parameters
    epochs: int = 200
    lr: float = 1e-3
    grad_clip: float = 1.0

    # Loss weights
    lambda_data: float = 1.0
    lambda_geo: float = 0.1
    lambda_chem: float = 1.0
    target_speed: float = 1.0

    # Performance options
    use_amp: bool = True
    use_scheduler: bool = True
    scheduler_T_max: Optional[int] = None

    # Checkpointing
    save_checkpoints: bool = False
    checkpoint_dir: str = 'results/checkpoints'
    checkpoint_every: int = 50

    # Device
    device: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Filter to only include valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save config files. Install with: pip install pyyaml")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def create_results_directory(base_dir: str = 'results') -> Dict[str, str]:
    """
    Create standard results directory structure.

    Args:
        base_dir: Base directory for results

    Returns:
        Dictionary with paths to subdirectories
    """
    dirs = {
        'base': base_dir,
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'figures': os.path.join(base_dir, 'figures'),
        'logs': os.path.join(base_dir, 'logs'),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs
