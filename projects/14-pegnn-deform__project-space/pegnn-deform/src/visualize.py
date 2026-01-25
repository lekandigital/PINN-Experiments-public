"""
PEGNN-Deform: Visualization Utilities

Provides visualization tools for:
- 3D mesh trajectories
- Training loss curves
- Prediction comparisons

Author: PEGNN-Deform Team
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_trajectory(
    pos_trajectory: Union[torch.Tensor, np.ndarray],
    edges: Optional[torch.Tensor] = None,
    frames: Optional[List[int]] = None,
    title: str = "Mesh Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot 3D mesh trajectory as animation frames.

    Args:
        pos_trajectory: Positions over time [T, N, 3] or list of [N, 3]
        edges: Edge indices [2, E] for drawing mesh edges
        frames: Which frames to display (default: [0, T//2, T-1])
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if isinstance(pos_trajectory, torch.Tensor):
        pos_trajectory = pos_trajectory.detach().cpu().numpy()

    T = len(pos_trajectory)

    if frames is None:
        if T <= 3:
            frames = list(range(T))
        else:
            frames = [0, T // 2, T - 1]

    num_frames = len(frames)
    fig = plt.figure(figsize=figsize)

    # Compute global bounds for consistent axes
    all_pos = np.concatenate([pos_trajectory[f] for f in frames])
    bounds = {
        'x': (all_pos[:, 0].min() - 0.1, all_pos[:, 0].max() + 0.1),
        'y': (all_pos[:, 1].min() - 0.1, all_pos[:, 1].max() + 0.1),
        'z': (all_pos[:, 2].min() - 0.1, all_pos[:, 2].max() + 0.1)
    }

    for idx, frame in enumerate(frames):
        ax = fig.add_subplot(1, num_frames, idx + 1, projection='3d')
        pos = pos_trajectory[frame]

        # Plot vertices
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c='blue', s=10, alpha=0.6)

        # Plot edges if provided
        if edges is not None:
            if isinstance(edges, torch.Tensor):
                edges_np = edges.detach().cpu().numpy()
            else:
                edges_np = edges

            for i in range(edges_np.shape[1]):
                src, dst = edges_np[0, i], edges_np[1, i]
                if src < dst:  # Avoid drawing edges twice
                    ax.plot3D(
                        [pos[src, 0], pos[dst, 0]],
                        [pos[src, 1], pos[dst, 1]],
                        [pos[src, 2], pos[dst, 2]],
                        'k-', alpha=0.3, linewidth=0.5
                    )

        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f't={frame}')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    return fig


def plot_loss_curves(
    train_losses: Union[List[float], np.ndarray],
    val_losses: Optional[Union[List[float], np.ndarray]] = None,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        log_scale: Use log scale for y-axis

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)

    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")

    return fig


def plot_prediction_comparison(
    pos_gt: Union[torch.Tensor, np.ndarray],
    pos_pred: Union[torch.Tensor, np.ndarray],
    edges: Optional[torch.Tensor] = None,
    title: str = "Ground Truth vs Prediction",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot side-by-side comparison of ground truth and predicted mesh.

    Args:
        pos_gt: Ground truth positions [N, 3]
        pos_pred: Predicted positions [N, 3]
        edges: Edge indices [2, E]
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if isinstance(pos_gt, torch.Tensor):
        pos_gt = pos_gt.detach().cpu().numpy()
    if isinstance(pos_pred, torch.Tensor):
        pos_pred = pos_pred.detach().cpu().numpy()

    fig = plt.figure(figsize=figsize)

    # Compute global bounds
    all_pos = np.concatenate([pos_gt, pos_pred])
    bounds = {
        'x': (all_pos[:, 0].min() - 0.1, all_pos[:, 0].max() + 0.1),
        'y': (all_pos[:, 1].min() - 0.1, all_pos[:, 1].max() + 0.1),
        'z': (all_pos[:, 2].min() - 0.1, all_pos[:, 2].max() + 0.1)
    }

    # Compute error for coloring
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)

    for idx, (pos, name) in enumerate([(pos_gt, 'Ground Truth'), (pos_pred, 'Prediction')]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        # Color by error for prediction
        if idx == 1:
            scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                                c=errors, cmap='coolwarm', s=15, alpha=0.8)
            plt.colorbar(scatter, ax=ax, label='Error', shrink=0.6)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c='blue', s=15, alpha=0.8)

        # Draw edges
        if edges is not None:
            if isinstance(edges, torch.Tensor):
                edges_np = edges.detach().cpu().numpy()
            else:
                edges_np = edges

            for i in range(edges_np.shape[1]):
                src, dst = edges_np[0, i], edges_np[1, i]
                if src < dst:
                    ax.plot3D(
                        [pos[src, 0], pos[dst, 0]],
                        [pos[src, 1], pos[dst, 1]],
                        [pos[src, 2], pos[dst, 2]],
                        'k-', alpha=0.2, linewidth=0.3
                    )

        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(name)

    fig.suptitle(f"{title}\nMean Error: {errors.mean():.4f}, Max Error: {errors.max():.4f}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_energy_conservation(
    energies: Union[List[float], np.ndarray],
    title: str = "Energy Conservation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Plot energy over time to verify conservation.

    Args:
        energies: Energy values over time
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    energies = np.array(energies)
    timesteps = np.arange(len(energies))

    # Absolute energy
    ax1.plot(timesteps, energies, 'b-', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Total Energy')
    ax1.set_title('Total Energy Over Time')
    ax1.grid(True, alpha=0.3)

    # Relative energy change
    if len(energies) > 1:
        relative_change = (energies - energies[0]) / (energies[0] + 1e-8) * 100
        ax2.plot(timesteps, relative_change, 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Energy Change (%)')
        ax2.set_title('Relative Energy Change')
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved energy plot to {save_path}")

    return fig


def create_animation_frames(
    pos_trajectory: Union[torch.Tensor, np.ndarray],
    edges: Optional[torch.Tensor] = None,
    output_dir: str = "frames",
    prefix: str = "frame"
) -> None:
    """
    Create individual frames for animation.

    Args:
        pos_trajectory: Positions over time [T, N, 3]
        edges: Edge indices [2, E]
        output_dir: Directory to save frames
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(pos_trajectory, torch.Tensor):
        pos_trajectory = pos_trajectory.detach().cpu().numpy()

    T = len(pos_trajectory)

    # Compute global bounds
    all_pos = pos_trajectory.reshape(-1, 3)
    bounds = {
        'x': (all_pos[:, 0].min() - 0.1, all_pos[:, 0].max() + 0.1),
        'y': (all_pos[:, 1].min() - 0.1, all_pos[:, 1].max() + 0.1),
        'z': (all_pos[:, 2].min() - 0.1, all_pos[:, 2].max() + 0.1)
    }

    for t in range(T):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        pos = pos_trajectory[t]

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=10, alpha=0.6)

        if edges is not None:
            if isinstance(edges, torch.Tensor):
                edges_np = edges.detach().cpu().numpy()
            else:
                edges_np = edges

            for i in range(edges_np.shape[1]):
                src, dst = edges_np[0, i], edges_np[1, i]
                if src < dst:
                    ax.plot3D(
                        [pos[src, 0], pos[dst, 0]],
                        [pos[src, 1], pos[dst, 1]],
                        [pos[src, 2], pos[dst, 2]],
                        'k-', alpha=0.3, linewidth=0.5
                    )

        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {t}/{T-1}')

        plt.savefig(output_path / f"{prefix}_{t:04d}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {T} frames to {output_dir}/")
    print(f"Create animation with: ffmpeg -framerate 30 -i {output_dir}/{prefix}_%04d.png -c:v libx264 animation.mp4")


if __name__ == "__main__":
    # Demo visualization
    print("PEGNN-Deform Visualization Demo")

    # Generate synthetic trajectory
    T, N = 10, 100
    pos_trajectory = np.zeros((T, N, 3))

    # Grid positions
    grid_size = 10
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    pos_trajectory[0, :, 0] = xx.flatten()
    pos_trajectory[0, :, 1] = yy.flatten()

    # Simulate falling
    for t in range(1, T):
        pos_trajectory[t] = pos_trajectory[t-1].copy()
        pos_trajectory[t, :, 2] -= 0.05 * t

    # Plot trajectory
    fig = plot_trajectory(pos_trajectory, title="Demo Trajectory")
    plt.show()

    # Plot loss curves
    train_losses = np.exp(-np.linspace(0, 3, 50)) + np.random.randn(50) * 0.01
    val_losses = np.exp(-np.linspace(0, 2.5, 50)) + np.random.randn(50) * 0.02
    fig = plot_loss_curves(train_losses, val_losses, title="Demo Loss Curves")
    plt.show()
