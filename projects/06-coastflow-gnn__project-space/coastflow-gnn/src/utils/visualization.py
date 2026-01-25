"""
Visualization Utilities for CoastFlow-GNN

Provides plotting functions for:
- Training curves (loss, learning rate)
- Prediction visualizations (velocity fields, wave height)
- Error distributions and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'lr' lists
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Learning rate
    ax = axes[0, 1]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Physics residuals
    ax = axes[1, 0]
    if 'physics_residuals' in history:
        residuals = history['physics_residuals']
        ax.plot(epochs, residuals.get('continuity', []), 'b-', 
                label='Continuity', linewidth=2)
        ax.plot(epochs, residuals.get('momentum', []), 'r-', 
                label='Momentum', linewidth=2)
        ax.plot(epochs, residuals.get('turbulence', []), 'g-', 
                label='Turbulence', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Residual')
        ax.set_title('Physics Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No physics residuals data', 
                ha='center', va='center', transform=ax.transAxes)
    
    # Epoch time
    ax = axes[1, 1]
    if 'epoch_time' in history:
        ax.bar(epochs, history['epoch_time'], color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(history['epoch_time']), color='r', 
                   linestyle='--', label=f'Mean: {np.mean(history["epoch_time"]):.1f}s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Epoch Duration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No timing data', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {output_path}")
    
    return fig


def plot_predictions(
    positions: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Plot prediction vs target comparison.
    
    Args:
        positions: Node positions [N, 3]
        predictions: Predicted values [N, 4]
        targets: Ground truth values [N, 4]
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    channel_names = ['u_x (m/s)', 'u_y (m/s)', 'u_z (m/s)', 'Wave Height (m)']
    
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    
    x, y = positions[:, 0], positions[:, 1]
    
    for i, name in enumerate(channel_names):
        pred = predictions[:, i]
        targ = targets[:, i]
        error = pred - targ
        
        vmin = min(pred.min(), targ.min())
        vmax = max(pred.max(), targ.max())
        
        # Prediction
        ax = axes[i, 0]
        sc = ax.scatter(x, y, c=pred, cmap='viridis', s=10, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} - Prediction')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(sc, ax=ax)
        
        # Target
        ax = axes[i, 1]
        sc = ax.scatter(x, y, c=targ, cmap='viridis', s=10, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} - Ground Truth')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(sc, ax=ax)
        
        # Error
        ax = axes[i, 2]
        err_max = max(abs(error.min()), abs(error.max()))
        sc = ax.scatter(x, y, c=error, cmap='RdBu_r', s=10, 
                        vmin=-err_max, vmax=err_max)
        ax.set_title(f'{name} - Error (MAE={np.abs(error).mean():.4f})')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions plot to: {output_path}")
    
    return fig


def plot_wave_field(
    positions: np.ndarray,
    wave_height: np.ndarray,
    velocity_u: Optional[np.ndarray] = None,
    velocity_v: Optional[np.ndarray] = None,
    title: str = "Wave Height Field",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    quiver_scale: float = 1.0,
    quiver_skip: int = 5,
) -> plt.Figure:
    """
    Plot wave height field with optional velocity vectors.
    
    Args:
        positions: Node positions [N, 3] or [N, 2]
        wave_height: Wave height values [N]
        velocity_u: X-component of velocity [N] (optional)
        velocity_v: Y-component of velocity [N] (optional)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        quiver_scale: Scale for velocity arrows
        quiver_skip: Skip factor for quiver plot (every Nth point)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Plot wave height as filled contour
    sc = ax.scatter(x, y, c=wave_height, cmap='Blues', s=15, 
                    edgecolors='none', alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax, label='Wave Height (m)')
    
    # Overlay velocity vectors
    if velocity_u is not None and velocity_v is not None:
        # Subsample for clarity
        idx = np.arange(0, len(x), quiver_skip)
        ax.quiver(x[idx], y[idx], velocity_u[idx], velocity_v[idx],
                  color='red', alpha=0.6, scale=quiver_scale,
                  width=0.003, headwidth=3)
        ax.plot([], [], 'r-', label='Wind/Current Vectors')
        ax.legend(loc='upper right')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved wave field plot to: {output_path}")
    
    return fig


def plot_velocity_field(
    positions: np.ndarray,
    velocity: np.ndarray,
    title: str = "Velocity Field",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    stream: bool = False,
) -> plt.Figure:
    """
    Plot velocity vector field.
    
    Args:
        positions: Node positions [N, 2] or [N, 3]
        velocity: Velocity vectors [N, 2] or [N, 3]
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        stream: Use streamplot instead of quiver
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = positions[:, 0]
    y = positions[:, 1]
    u = velocity[:, 0]
    v = velocity[:, 1]
    
    # Velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    
    # Color by magnitude
    sc = ax.scatter(x, y, c=speed, cmap='plasma', s=10, alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax, label='Speed (m/s)')
    
    # Quiver plot
    skip = max(1, len(x) // 500)  # Limit number of arrows
    ax.quiver(x[::skip], y[::skip], u[::skip], v[::skip],
              color='black', alpha=0.7, scale=50, width=0.002)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved velocity field plot to: {output_path}")
    
    return fig


def plot_error_distribution(
    errors: np.ndarray,
    channel_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot error distribution histograms.
    
    Args:
        errors: Prediction errors [N, C]
        channel_names: Names for each channel
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if channel_names is None:
        channel_names = ['u_x', 'u_y', 'u_z', 'wave_height']
    
    n_channels = errors.shape[1]
    fig, axes = plt.subplots(1, n_channels, figsize=figsize)
    
    if n_channels == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        err = errors[:, i]
        
        ax.hist(err, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=err.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {err.mean():.4f}')
        ax.axvline(x=err.mean() + err.std(), color='orange', linestyle='--', 
                   linewidth=1, label=f'Std: {err.std():.4f}')
        ax.axvline(x=err.mean() - err.std(), color='orange', linestyle='--', 
                   linewidth=1)
        
        ax.set_xlabel('Error')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Error Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution to: {output_path}")
    
    return fig


def load_and_plot_history(
    history_path: str,
    output_dir: Optional[str] = None,
) -> None:
    """
    Load training history and generate all plots.
    
    Args:
        history_path: Path to training_history.json
        output_dir: Directory to save plots
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if output_dir is None:
        output_dir = Path(history_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    plot_training_curves(
        history,
        output_path=str(output_dir / 'training_curves.png')
    )
    
    print(f"Generated plots in: {output_dir}")


if __name__ == "__main__":
    # Demo with synthetic data
    print("Generating demo visualizations...")
    
    # Synthetic history
    epochs = 50
    history = {
        'train_loss': [1.0 * np.exp(-0.05 * i) + 0.1 + 0.02 * np.random.randn() 
                       for i in range(epochs)],
        'val_loss': [1.2 * np.exp(-0.04 * i) + 0.12 + 0.03 * np.random.randn() 
                     for i in range(epochs)],
        'lr': [1e-3 * (0.9 ** (i // 10)) for i in range(epochs)],
        'physics_residuals': {
            'continuity': [0.5 * np.exp(-0.03 * i) + 0.05 for i in range(epochs)],
            'momentum': [0.3 * np.exp(-0.02 * i) + 0.03 for i in range(epochs)],
            'turbulence': [0.2 * np.exp(-0.025 * i) + 0.02 for i in range(epochs)],
        },
        'epoch_time': [5 + np.random.randn() for _ in range(epochs)],
    }
    
    fig = plot_training_curves(history)
    plt.savefig('./demo_training_curves.png', dpi=100)
    plt.close()
    
    # Synthetic field data
    n = 500
    x = np.random.rand(n) * 1000
    y = np.random.rand(n) * 500
    positions = np.column_stack([x, y, np.zeros(n)])
    wave_height = 0.5 + 0.3 * np.sin(2 * np.pi * x / 1000) * np.cos(2 * np.pi * y / 500)
    velocity_u = 5 * np.ones(n) + np.random.randn(n)
    velocity_v = 2 * np.sin(2 * np.pi * y / 500)
    
    fig = plot_wave_field(
        positions, wave_height, velocity_u, velocity_v,
        title="Demo: Wave Height with Wind Vectors"
    )
    plt.savefig('./demo_wave_field.png', dpi=100)
    plt.close()
    
    print("✓ Demo visualizations saved")
