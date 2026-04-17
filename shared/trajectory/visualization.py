"""
Trajectory Visualization Utilities.

Provides consistent, publication-quality visualizations for trajectory data:
- 2D trajectory plots with potential field backgrounds
- 3D interactive trajectory plots (Plotly)
- Animated trajectory evolution (GIF/MP4)

All functions work with the standard [batch, time, dim] format.
"""

from typing import Optional, Callable, Tuple, List, Union
import torch
import numpy as np

# Import plotting libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def plot_trajectory_2d(
    positions: Union[torch.Tensor, np.ndarray],
    potential_field: Optional[Callable] = None,
    ax: Optional['plt.Axes'] = None,
    color_by: str = "time",
    cmap: str = "plasma",
    show_potential: bool = True,
    potential_cmap: str = "viridis",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 50,
    title: Optional[str] = None,
    show_markers: bool = True,
    linewidth: float = 2.0,
    alpha: float = 0.8,
    figsize: Tuple[int, int] = (8, 8),
) -> 'plt.Figure':
    """
    Plot 2D trajectory with optional potential field background.
    
    Args:
        positions: Trajectory [batch, time, 2] or [time, 2]. Uses first trajectory if batched.
        potential_field: Callable (positions) -> potential for background
        ax: Matplotlib axes (created if None)
        color_by: "time", "speed", or "constant"
        cmap: Colormap for trajectory
        show_potential: Whether to show potential field background
        potential_cmap: Colormap for potential field
        xlim: X-axis limits (auto-detected if None)
        ylim: Y-axis limits (auto-detected if None)
        resolution: Grid resolution for potential field
        title: Plot title
        show_markers: Show start/end markers
        linewidth: Trajectory line width
        alpha: Trajectory alpha
        figsize: Figure size if creating new figure
        
    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for 2D plots: pip install matplotlib")
    
    positions = _to_numpy(positions)
    
    # Handle batch dimension
    if positions.ndim == 3:
        positions = positions[0]  # Use first trajectory
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Auto-detect limits
    if xlim is None:
        margin = 0.1 * (positions[:, 0].max() - positions[:, 0].min() + 1e-6)
        xlim = (positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    if ylim is None:
        margin = 0.1 * (positions[:, 1].max() - positions[:, 1].min() + 1e-6)
        ylim = (positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    
    # Plot potential field background
    if show_potential and potential_field is not None:
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        points = torch.tensor(
            np.stack([X.ravel(), Y.ravel()], axis=-1),
            dtype=torch.float32
        )
        
        with torch.no_grad():
            Z = potential_field(points)
            if isinstance(Z, torch.Tensor):
                Z = Z.numpy()
            Z = Z.reshape(resolution, resolution)
        
        contour = ax.contourf(X, Y, Z, levels=20, cmap=potential_cmap, alpha=0.6)
        plt.colorbar(contour, ax=ax, label='Potential')
    
    # Compute colors
    n_points = len(positions)
    if color_by == "time":
        colors = np.linspace(0, 1, n_points)
    elif color_by == "speed":
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        speeds = np.concatenate([[speeds[0]], speeds])
        colors = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-8)
    else:  # constant
        colors = np.ones(n_points) * 0.5
    
    # Create line segments for colored trajectory
    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=alpha)
    lc.set_array(colors[:-1])
    ax.add_collection(lc)
    
    # Start and end markers
    if show_markers:
        ax.scatter(
            positions[0, 0], positions[0, 1],
            c='green', s=100, marker='o', label='Start', zorder=5, edgecolors='white'
        )
        ax.scatter(
            positions[-1, 0], positions[-1, 1],
            c='red', s=150, marker='*', label='End', zorder=5, edgecolors='white'
        )
        ax.legend(loc='upper right')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if title:
        ax.set_title(title)
    
    return fig


def plot_multiple_trajectories_2d(
    trajectories: Union[torch.Tensor, np.ndarray, List],
    potential_field: Optional[Callable] = None,
    ax: Optional['plt.Axes'] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    show_potential: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
) -> 'plt.Figure':
    """
    Plot multiple 2D trajectories on the same axes.
    
    Args:
        trajectories: [n_traj, time, 2] or list of [time, 2] arrays
        potential_field: Optional potential field for background
        ax: Matplotlib axes
        colors: List of colors for each trajectory
        labels: List of labels for legend
        show_potential: Show potential field background
        xlim, ylim: Axis limits
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required: pip install matplotlib")
    
    # Handle input formats
    if isinstance(trajectories, (torch.Tensor, np.ndarray)):
        trajectories = _to_numpy(trajectories)
        if trajectories.ndim == 2:
            trajectories = [trajectories]
        else:
            trajectories = [trajectories[i] for i in range(len(trajectories))]
    else:
        trajectories = [_to_numpy(t) for t in trajectories]
    
    n_traj = len(trajectories)
    
    # Default colors
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(n_traj)]
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Auto-detect limits
    if xlim is None or ylim is None:
        all_x = np.concatenate([t[:, 0] for t in trajectories])
        all_y = np.concatenate([t[:, 1] for t in trajectories])
        margin_x = 0.1 * (all_x.max() - all_x.min() + 1e-6)
        margin_y = 0.1 * (all_y.max() - all_y.min() + 1e-6)
        if xlim is None:
            xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
        if ylim is None:
            ylim = (all_y.min() - margin_y, all_y.max() + margin_y)
    
    # Plot potential field
    if show_potential and potential_field is not None:
        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        X, Y = np.meshgrid(x, y)
        points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
        
        with torch.no_grad():
            Z = potential_field(points)
            if isinstance(Z, torch.Tensor):
                Z = Z.numpy()
            Z = Z.reshape(50, 50)
        
        ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
    
    # Plot trajectories
    for i, traj in enumerate(trajectories):
        label = labels[i] if labels else f"Trajectory {i+1}"
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, label=label, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=80, marker='o', edgecolors='white', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=120, marker='*', edgecolors='white', zorder=5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    if title:
        ax.set_title(title)
    
    return fig


def plot_trajectory_3d(
    positions: Union[torch.Tensor, np.ndarray],
    color_by: str = "time",
    cmap: str = "Plasma",
    title: Optional[str] = None,
    show_markers: bool = True,
    line_width: float = 3,
    marker_size: int = 8,
) -> 'go.Figure':
    """
    Create interactive 3D trajectory plot using Plotly.
    
    Args:
        positions: Trajectory [batch, time, 3] or [time, 3]
        color_by: "time" or "speed"
        cmap: Plotly colorscale name
        title: Plot title
        show_markers: Show trajectory points
        line_width: Line width
        marker_size: Marker size
        
    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required for 3D plots: pip install plotly")
    
    positions = _to_numpy(positions)
    
    if positions.ndim == 3:
        positions = positions[0]
    
    # Handle 2D positions (add z=0)
    if positions.shape[1] == 2:
        positions = np.concatenate([positions, np.zeros((len(positions), 1))], axis=-1)
    
    n_points = len(positions)
    
    # Compute colors
    if color_by == "time":
        colors = np.linspace(0, 1, n_points)
    else:  # speed
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        speeds = np.concatenate([[speeds[0]], speeds])
        colors = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-8)
    
    fig = go.Figure()
    
    # Trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers' if show_markers else 'lines',
        marker=dict(
            size=marker_size,
            color=colors,
            colorscale=cmap,
            showscale=True,
            colorbar=dict(title=color_by.capitalize())
        ),
        line=dict(color='blue', width=line_width),
        name='Trajectory'
    ))
    
    # Start marker
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[0, 1]],
        z=[positions[0, 2]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        name='Start'
    ))
    
    # End marker
    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]],
        y=[positions[-1, 1]],
        z=[positions[-1, 2]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='End'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True,
    )
    
    return fig


def plot_potential_field_2d(
    potential_field: Callable,
    xlim: Tuple[float, float] = (-1, 1),
    ylim: Tuple[float, float] = (-1, 1),
    resolution: int = 100,
    show_gradient: bool = False,
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> 'plt.Figure':
    """
    Visualize a 2D potential field.
    
    Args:
        potential_field: Callable [batch, 2] -> [batch, 1]
        xlim, ylim: Domain bounds
        resolution: Grid resolution
        show_gradient: Overlay gradient arrows
        cmap: Colormap
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required: pip install matplotlib")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    points = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=-1),
        dtype=torch.float32
    )
    
    # Compute potential
    with torch.no_grad():
        Z = potential_field(points)
        if isinstance(Z, torch.Tensor):
            Z = Z.numpy()
        Z = Z.reshape(resolution, resolution)
    
    # Contour plot
    contour = ax.contourf(X, Y, Z, levels=30, cmap=cmap)
    plt.colorbar(contour, ax=ax, label='Potential φ(x)')
    
    # Add contour lines
    ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    
    # Gradient field overlay
    if show_gradient:
        # Lower resolution for arrows
        arrow_res = min(20, resolution // 5)
        x_arrow = np.linspace(xlim[0], xlim[1], arrow_res)
        y_arrow = np.linspace(ylim[0], ylim[1], arrow_res)
        X_arrow, Y_arrow = np.meshgrid(x_arrow, y_arrow)
        
        points_arrow = torch.tensor(
            np.stack([X_arrow.ravel(), Y_arrow.ravel()], axis=-1),
            dtype=torch.float32,
            requires_grad=True
        )
        
        # Compute gradient
        phi = potential_field(points_arrow)
        if hasattr(potential_field, 'gradient'):
            grad = potential_field.gradient(points_arrow)
        else:
            grad = torch.autograd.grad(phi.sum(), points_arrow)[0]
        
        grad = grad.detach().numpy().reshape(arrow_res, arrow_res, 2)
        U = -grad[:, :, 0]  # Negative gradient (descent direction)
        V = -grad[:, :, 1]
        
        # Normalize arrows
        mag = np.sqrt(U**2 + V**2) + 1e-8
        U = U / mag
        V = V / mag
        
        ax.quiver(X_arrow, Y_arrow, U, V, color='white', alpha=0.6, scale=25)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if title:
        ax.set_title(title)
    
    return fig


def animate_trajectory(
    positions: Union[torch.Tensor, np.ndarray],
    potential_field: Optional[Callable] = None,
    fps: int = 30,
    save_path: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> 'animation.FuncAnimation':
    """
    Create animated visualization of trajectory evolution.
    
    Args:
        positions: Trajectory [time, dim] or [batch, time, dim]
        potential_field: Optional background potential
        fps: Frames per second
        save_path: Path to save animation (.gif or .mp4)
        xlim, ylim: Axis limits
        title: Animation title
        figsize: Figure size
        
    Returns:
        Matplotlib animation object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required: pip install matplotlib")
    
    positions = _to_numpy(positions)
    
    if positions.ndim == 3:
        positions = positions[0]
    
    n_points = len(positions)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Auto-detect limits
    if xlim is None:
        margin = 0.1 * (positions[:, 0].max() - positions[:, 0].min() + 1e-6)
        xlim = (positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    if ylim is None:
        margin = 0.1 * (positions[:, 1].max() - positions[:, 1].min() + 1e-6)
        ylim = (positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    
    # Background potential
    if potential_field is not None:
        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        X, Y = np.meshgrid(x, y)
        points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
        
        with torch.no_grad():
            Z = potential_field(points)
            if isinstance(Z, torch.Tensor):
                Z = Z.numpy()
            Z = Z.reshape(50, 50)
        
        ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    
    # Initialize plot elements
    line, = ax.plot([], [], 'b-', linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=10)
    start, = ax.plot([positions[0, 0]], [positions[0, 1]], 'go', markersize=12, label='Start')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    if title:
        ax.set_title(title)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def animate_frame(i):
        line.set_data(positions[:i+1, 0], positions[:i+1, 1])
        point.set_data([positions[i, 0]], [positions[i, 1]])
        return line, point
    
    anim = animation.FuncAnimation(
        fig, animate_frame, init_func=init,
        frames=n_points, interval=1000//fps, blit=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    return anim
