"""
Cell-Path PINNs: Physics-Informed Neural Networks for Microbe Trajectory Prediction

This package provides tools for modeling microbe movement as geodesics on an
evolving nutrient manifold using physics-informed neural networks.
"""

from .models import PathNet, PotentialNet
from .losses import compute_losses, compute_data_loss, compute_geodesic_loss, compute_chemotactic_loss
from .data_utils import generate_synthetic_trajectory, generate_circular_trajectory
from .api import CellPathModel
from .config import TrainingConfig, create_results_directory
from .data_loader import (
    load_trajectory_csv,
    load_trackmate_xml,
    TrajectoryDataset,
    save_trajectory_csv,
)

__version__ = '0.2.0'
__all__ = [
    # Models
    'PathNet',
    'PotentialNet',
    # API
    'CellPathModel',
    # Losses
    'compute_losses',
    'compute_data_loss',
    'compute_geodesic_loss',
    'compute_chemotactic_loss',
    # Data generation
    'generate_synthetic_trajectory',
    'generate_circular_trajectory',
    # Configuration
    'TrainingConfig',
    'create_results_directory',
    # Data loading
    'load_trajectory_csv',
    'load_trackmate_xml',
    'TrajectoryDataset',
    'save_trajectory_csv',
]
