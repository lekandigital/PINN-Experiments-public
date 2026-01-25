"""
Visualization utilities for NIF-Cloth4D-Temporal.

Provides plotting functions for:
    - SDF visualization (slices, isosurfaces)
    - Training curves
    - Benchmark metrics
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path


def visualize_sdf(
    sdf_volume: np.ndarray,
    slice_axis: int = 1,
    slice_index: Optional[int] = None,
    output_path: Optional[str] = None,
    title: str = "SDF Slice",
    cmap: str = 'RdBu',
    vmin: float = -0.5,
    vmax: float = 0.5,
) -> None:
    """
    Visualize a slice of the SDF volume.
    
    Args:
        sdf_volume: 3D SDF array (D, H, W)
        slice_axis: Axis to slice along (0=x, 1=y, 2=z)
        slice_index: Index along slice axis (default: middle)
        output_path: Path to save figure (displays if None)
        title: Plot title
        cmap: Colormap name
        vmin, vmax: Color range limits
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available for visualization")
        return
    
    # Get slice index
    if slice_index is None:
        slice_index = sdf_volume.shape[slice_axis] // 2
    
    # Extract slice
    if slice_axis == 0:
        sdf_slice = sdf_volume[slice_index, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif slice_axis == 1:
        sdf_slice = sdf_volume[:, slice_index, :]
        xlabel, ylabel = 'X', 'Z'
    else:
        sdf_slice = sdf_volume[:, :, slice_index]
        xlabel, ylabel = 'X', 'Y'
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        sdf_slice.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal',
    )
    
    # Add zero contour
    ax.contour(sdf_slice.T, levels=[0], colors='black', linewidths=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='SDF Value')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_sdf_3d(
    sdf_volume: np.ndarray,
    level: float = 0.0,
    output_path: Optional[str] = None,
    title: str = "SDF Isosurface",
) -> None:
    """
    Visualize 3D isosurface of SDF using marching cubes.
    
    Args:
        sdf_volume: 3D SDF array
        level: Isosurface level (0 for surface)
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from skimage import measure
    except ImportError:
        print("Warning: Required libraries not available for 3D visualization")
        return
    
    # Extract isosurface
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_volume, level=level)
    except ValueError:
        print("Warning: Could not extract isosurface at level", level)
        return
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=faces,
        cmap='viridis',
        alpha=0.8,
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    title: str = "Training Progress",
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[0, 1]
    if 'learning_rate' in history:
        ax.plot(history['learning_rate'], color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Epsilon (scheduled sampling)
    ax = axes[1, 0]
    if 'epsilon' in history:
        ax.plot(history['epsilon'], color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Epsilon')
        ax.set_title('Scheduled Sampling Rate')
        ax.grid(True, alpha=0.3)
    
    # Per-loss components (if available)
    ax = axes[1, 1]
    loss_keys = ['stretch', 'bend', 'momentum', 'collision']
    has_components = any(k in history for k in loss_keys)
    if has_components:
        for key in loss_keys:
            if key in history:
                ax.plot(history[key], label=key.capitalize())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Physics Loss Components')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No component losses recorded',
                ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(
    metrics: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    title: str = "Benchmark Metrics",
) -> None:
    """
    Plot benchmark metrics (Chamfer distance, energy drift, smoothness).
    
    Args:
        metrics: Dictionary with metric arrays over time
            - 'chamfer': Per-frame Chamfer distance
            - 'energy': Total energy per frame
            - 'smoothness': Vertex displacement per frame
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available")
        return
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        frames = np.arange(len(values))
        
        ax.plot(frames, values, color=colors[i % len(colors)], linewidth=2)
        ax.fill_between(frames, 0, values, alpha=0.2, color=colors[i % len(colors)])
        
        ax.set_xlabel('Frame')
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.set_title(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(mean_val, color='gray', linestyle='--', alpha=0.5)
        ax.text(
            0.02, 0.98,
            f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mesh_comparison(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Mesh Comparison",
) -> None:
    """
    Plot side-by-side comparison of predicted and ground truth meshes.
    
    Args:
        pred_vertices: Predicted vertex positions (N, 3)
        gt_vertices: Ground truth vertex positions (N, 3)
        faces: Triangle indices (F, 3)
        output_path: Path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("Warning: matplotlib not available")
        return
    
    fig = plt.figure(figsize=(14, 6))
    
    for i, (vertices, subtitle) in enumerate([
        (pred_vertices, 'Predicted'),
        (gt_vertices, 'Ground Truth'),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        
        # Create polygon collection
        triangles = vertices[faces]
        poly = Poly3DCollection(
            triangles,
            alpha=0.7,
            facecolor='cyan',
            edgecolor='navy',
            linewidth=0.5,
        )
        ax.add_collection3d(poly)
        
        # Set axis limits
        max_range = np.max(np.abs(vertices)) * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(subtitle)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
