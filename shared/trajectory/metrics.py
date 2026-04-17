"""
Trajectory Quality Metrics.

Comprehensive metrics for evaluating trajectory quality:
- Path length: Total Euclidean distance traveled
- Path energy: Integral of squared velocity
- Path smoothness: Curvature-based smoothness measure
- Fréchet distance: Shape similarity between trajectories
- Goal reaching: Success rate for reaching target positions

These metrics are domain-agnostic and work with any trajectory data
in the standard [batch, time, dim] format.
"""

from typing import Optional, Union, Callable
import torch
import numpy as np


def path_length(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute total Euclidean path length.
    
    L = Σ ||x(t_{i+1}) - x(t_i)||
    
    Args:
        positions: [batch, time, dim] or [time, dim]
        
    Returns:
        Path length [batch] or scalar
    """
    squeeze_output = False
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
        squeeze_output = True
    
    # Segment lengths
    diffs = positions[:, 1:] - positions[:, :-1]  # [batch, time-1, dim]
    segment_lengths = torch.norm(diffs, dim=-1)  # [batch, time-1]
    total_length = segment_lengths.sum(dim=-1)  # [batch]
    
    if squeeze_output:
        return total_length.squeeze(0)
    return total_length


def path_energy(
    positions: torch.Tensor,
    times: Optional[torch.Tensor] = None,
    metric: str = "euclidean",
    potential_field: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Compute path energy integral.
    
    Euclidean: E = ∫ |dx/dt|² dt ≈ Σ |Δx/Δt|² Δt
    Riemannian: E = ∫ g_ij (dx^i/dt)(dx^j/dt) dt
    
    Args:
        positions: [batch, time, dim] or [time, dim]
        times: [batch, time] or [time]. If None, assumes uniform times in [0, 1]
        metric: "euclidean" or "riemannian"
        potential_field: Required for Riemannian metric (provides metric tensor)
        
    Returns:
        Path energy [batch] or scalar
    """
    squeeze_output = False
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
        squeeze_output = True
    
    batch_size, n_times, dim = positions.shape
    
    # Default times
    if times is None:
        times = torch.linspace(0, 1, n_times, device=positions.device)
        times = times.unsqueeze(0).expand(batch_size, -1)
    elif times.dim() == 1:
        times = times.unsqueeze(0).expand(batch_size, -1)
    
    # Time differences
    dt = times[:, 1:] - times[:, :-1]  # [batch, time-1]
    
    # Position differences (velocity * dt)
    dx = positions[:, 1:] - positions[:, :-1]  # [batch, time-1, dim]
    
    # Velocity
    velocity = dx / (dt.unsqueeze(-1) + 1e-8)  # [batch, time-1, dim]
    
    if metric == "euclidean" or potential_field is None:
        # E = Σ |v|² dt
        speed_squared = (velocity ** 2).sum(dim=-1)  # [batch, time-1]
        energy = (speed_squared * dt).sum(dim=-1)  # [batch]
    else:
        # Riemannian energy with metric from potential field
        midpoints = (positions[:, 1:] + positions[:, :-1]) / 2  # [batch, time-1, dim]
        
        energies = []
        for b in range(batch_size):
            # Get metric tensor at midpoints
            g = potential_field.metric_tensor(midpoints[b])  # [time-1, dim, dim]
            v = velocity[b]  # [time-1, dim]
            
            # Energy: e = v^T g v for each segment
            # e_i = Σ_jk v_j g_jk v_k
            e = torch.einsum('ti,tij,tj->t', v, g, v)  # [time-1]
            energy_b = (e * dt[b]).sum()
            energies.append(energy_b)
        
        energy = torch.stack(energies)
    
    if squeeze_output:
        return energy.squeeze(0)
    return energy


def curvature(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute curvature at each point along trajectory.
    
    κ = |v × a| / |v|³  (3D)
    κ = |v_x a_y - v_y a_x| / |v|³  (2D)
    
    Args:
        positions: [batch, time, dim] or [time, dim]
        
    Returns:
        Curvature [batch, time-2] or [time-2]
    """
    squeeze_output = False
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
        squeeze_output = True
    
    # Velocity (finite difference)
    velocity = positions[:, 1:] - positions[:, :-1]  # [batch, time-1, dim]
    
    # Acceleration
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [batch, time-2, dim]
    
    # Use velocity at acceleration points
    v = velocity[:, :-1]  # [batch, time-2, dim]
    a = acceleration
    
    # Speed
    speed = torch.norm(v, dim=-1, keepdim=True) + 1e-8  # [batch, time-2, 1]
    
    dim = positions.shape[-1]
    
    if dim == 2:
        # 2D: κ = |v_x*a_y - v_y*a_x| / |v|³
        cross = v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]
        kappa = torch.abs(cross) / (speed.squeeze(-1) ** 3)
    else:
        # 3D: κ = |v × a| / |v|³
        cross = torch.cross(v, a, dim=-1)
        kappa = torch.norm(cross, dim=-1) / (speed.squeeze(-1) ** 3)
    
    if squeeze_output:
        return kappa.squeeze(0)
    return kappa


def path_smoothness(positions: torch.Tensor, method: str = "curvature") -> torch.Tensor:
    """
    Compute path smoothness measure.
    
    Methods:
    - "curvature": Mean squared curvature (lower = smoother)
    - "jerk": Mean squared jerk (third derivative)
    - "acceleration": Mean squared acceleration
    
    Args:
        positions: [batch, time, dim] or [time, dim]
        method: Smoothness metric to use
        
    Returns:
        Smoothness measure [batch] or scalar (lower = smoother)
    """
    squeeze_output = False
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
        squeeze_output = True
    
    if method == "curvature":
        kappa = curvature(positions)
        smoothness = (kappa ** 2).mean(dim=-1)
    
    elif method == "jerk":
        # Jerk = d³x/dt³
        v = positions[:, 1:] - positions[:, :-1]
        a = v[:, 1:] - v[:, :-1]
        j = a[:, 1:] - a[:, :-1]
        smoothness = (j ** 2).sum(dim=-1).mean(dim=-1)
    
    elif method == "acceleration":
        v = positions[:, 1:] - positions[:, :-1]
        a = v[:, 1:] - v[:, :-1]
        smoothness = (a ** 2).sum(dim=-1).mean(dim=-1)
    
    else:
        raise ValueError(f"Unknown smoothness method: {method}")
    
    if squeeze_output:
        return smoothness.squeeze(0)
    return smoothness


def frechet_distance(path1: torch.Tensor, path2: torch.Tensor) -> torch.Tensor:
    """
    Compute discrete Fréchet distance between two paths.
    
    The Fréchet distance measures similarity between curves, taking
    spatial ordering into account (unlike Hausdorff distance).
    
    Args:
        path1: First path [n_points, dim]
        path2: Second path [m_points, dim]
        
    Returns:
        Fréchet distance (scalar)
    """
    n = path1.shape[0]
    m = path2.shape[0]
    
    # Pairwise Euclidean distances
    dist = torch.cdist(path1, path2)  # [n, m]
    
    # Dynamic programming table
    dp = torch.full((n, m), float('inf'), device=path1.device, dtype=path1.dtype)
    dp[0, 0] = dist[0, 0]
    
    # Fill first row and column
    for i in range(1, n):
        dp[i, 0] = torch.max(dp[i-1, 0], dist[i, 0])
    for j in range(1, m):
        dp[0, j] = torch.max(dp[0, j-1], dist[0, j])
    
    # Fill rest of table
    for i in range(1, n):
        for j in range(1, m):
            min_prev = torch.min(torch.min(dp[i-1, j], dp[i, j-1]), dp[i-1, j-1])
            dp[i, j] = torch.max(min_prev, dist[i, j])
    
    return dp[n-1, m-1]


def frechet_distance_batch(paths1: torch.Tensor, paths2: torch.Tensor) -> torch.Tensor:
    """
    Compute Fréchet distance for batch of path pairs.
    
    Args:
        paths1: [batch, n_points, dim]
        paths2: [batch, m_points, dim]
        
    Returns:
        Fréchet distances [batch]
    """
    batch_size = paths1.shape[0]
    distances = []
    
    for i in range(batch_size):
        d = frechet_distance(paths1[i], paths2[i])
        distances.append(d)
    
    return torch.stack(distances)


def goal_reaching_accuracy(
    positions: torch.Tensor,
    goals: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Compute fraction of trajectories reaching their goals.
    
    Args:
        positions: [batch, time, dim]
        goals: [batch, dim] target positions
        threshold: Distance threshold for "reaching" goal
        
    Returns:
        Accuracy (fraction of successful reaches)
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    if goals.dim() == 1:
        goals = goals.unsqueeze(0)
    
    final_positions = positions[:, -1]  # [batch, dim]
    distances = torch.norm(final_positions - goals, dim=-1)  # [batch]
    
    success = (distances < threshold).float()
    return success.mean()


def trajectory_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE between predicted and target trajectories.
    
    Args:
        predicted: [batch, time, dim] or [time, dim]
        target: [batch, time, dim] or [time, dim]
        reduction: "mean", "sum", or "none"
        
    Returns:
        MSE loss
    """
    if predicted.dim() == 2:
        predicted = predicted.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    
    mse = (predicted - target) ** 2
    
    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    else:  # none
        return mse


def trajectory_statistics(positions: torch.Tensor) -> dict:
    """
    Compute comprehensive trajectory statistics.
    
    Args:
        positions: [batch, time, dim] or [time, dim]
        
    Returns:
        Dictionary of statistics
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    
    length = path_length(positions)
    energy = path_energy(positions)
    smoothness = path_smoothness(positions)
    
    # Bounding box
    bbox_min = positions.min(dim=1).values  # [batch, dim]
    bbox_max = positions.max(dim=1).values  # [batch, dim]
    bbox_size = bbox_max - bbox_min
    
    # Displacement (start to end)
    displacement = torch.norm(positions[:, -1] - positions[:, 0], dim=-1)
    
    # Efficiency (displacement / length)
    efficiency = displacement / (length + 1e-8)
    
    return {
        'path_length': length,
        'path_energy': energy,
        'smoothness': smoothness,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'bbox_size': bbox_size,
        'displacement': displacement,
        'efficiency': efficiency,
    }
