"""
Rollout-Specific Metrics for Long-Horizon Stability Assessment

Standard per-frame loss doesn't capture rollout quality. This module
provides metrics specifically designed to measure how well a model
maintains stability over autoregressive rollouts.

Key metrics:
- Rollout MSE Curve: MSE at each step of free-running rollout
- Drift Rate: How fast does the model diverge? (slope of log-MSE)
- Stability Horizon: How many frames until MSE exceeds threshold?
- Energy Conservation: Does the model conserve physical energy?
- Spectral Divergence: Does the model lose high-frequency detail?
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RolloutMetricsResult:
    """Container for rollout stability metrics."""
    mse_per_step: Union[Tensor, "np.ndarray"]
    stability_horizon: int
    drift_rate: float
    initial_mse: float
    final_mse: float
    energy_ratio: Optional[float] = None
    spectral_divergence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging."""
        result = {
            "stability_horizon": self.stability_horizon,
            "drift_rate": self.drift_rate,
            "initial_mse": self.initial_mse,
            "final_mse": self.final_mse,
            "mse_ratio": self.final_mse / (self.initial_mse + 1e-8),
        }
        if self.energy_ratio is not None:
            result["energy_ratio"] = self.energy_ratio
        if self.spectral_divergence is not None:
            result["spectral_divergence"] = self.spectral_divergence
        return result


class RolloutMetrics:
    """Compute and track rollout stability metrics over training.
    
    Usage:
        metrics = RolloutMetrics(log_every=10)
        
        for epoch in range(epochs):
            # ... training ...
            
            if epoch % eval_every == 0:
                result = metrics.evaluate(predictions, targets)
                print(f"Stability horizon: {result.stability_horizon}")
                print(f"Drift rate: {result.drift_rate:.4f}")
    """
    
    def __init__(
        self,
        threshold_multiplier: float = 2.0,
        log_every: int = 10,
        compute_energy: bool = True,
        compute_spectral: bool = False,
    ):
        """
        Args:
            threshold_multiplier: MSE threshold = initial_mse * this value
            log_every: Log detailed metrics every N evaluations
            compute_energy: Whether to compute energy conservation metric
            compute_spectral: Whether to compute spectral divergence (slower)
        """
        self.threshold_multiplier = threshold_multiplier
        self.log_every = log_every
        self.compute_energy = compute_energy
        self.compute_spectral = compute_spectral
        self.history: List[RolloutMetricsResult] = []
        self.eval_count = 0
    
    def evaluate(
        self,
        predictions: Tensor,
        targets: Tensor,
        velocities: Optional[Tensor] = None,
        masses: Optional[Tensor] = None,
    ) -> RolloutMetricsResult:
        """Compute rollout metrics from prediction and target sequences.
        
        Args:
            predictions: (B, T, ...) predicted sequence
            targets: (B, T, ...) ground truth sequence
            velocities: Optional (B, T, ..., 3) for energy computation
            masses: Optional (B, N) or (N,) for energy computation
            
        Returns:
            RolloutMetricsResult with computed metrics
        """
        mse_curve = compute_rollout_mse_curve(predictions, targets)
        initial_mse = mse_curve[0].item()
        threshold = initial_mse * self.threshold_multiplier
        stability_horizon = compute_stability_horizon(mse_curve, threshold)
        drift_rate = compute_drift_rate(mse_curve)
        
        energy_ratio = None
        if self.compute_energy and velocities is not None:
            energy_ratio = compute_energy_conservation(predictions, velocities, masses)
        
        spectral_div = None
        if self.compute_spectral:
            spectral_div = compute_spectral_divergence(predictions, targets)
        
        result = RolloutMetricsResult(
            mse_per_step=mse_curve.cpu().numpy() if HAS_NUMPY else mse_curve,
            stability_horizon=stability_horizon,
            drift_rate=drift_rate,
            initial_mse=initial_mse,
            final_mse=mse_curve[-1].item(),
            energy_ratio=energy_ratio,
            spectral_divergence=spectral_div,
        )
        self.history.append(result)
        self.eval_count += 1
        return result
    
    def get_trend(self, metric: str = "stability_horizon", window: int = 5) -> float:
        """Get trend of a metric over recent evaluations.
        
        Positive trend = metric improving (for stability_horizon)
        Negative trend = metric degrading
        """
        if len(self.history) < 2:
            return 0.0
        recent = self.history[-window:]
        values = [getattr(r, metric) for r in recent]
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / (denominator + 1e-8)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics over all evaluations."""
        if not self.history:
            return {}
        horizons = [r.stability_horizon for r in self.history]
        drifts = [r.drift_rate for r in self.history]
        return {
            "best_stability_horizon": max(horizons),
            "latest_stability_horizon": horizons[-1],
            "mean_drift_rate": sum(drifts) / len(drifts),
            "trend_stability": self.get_trend("stability_horizon"),
            "trend_drift": self.get_trend("drift_rate"),
        }


def compute_rollout_mse_curve(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute MSE at each timestep of a rollout.
    
    Args:
        predictions: (B, T, ...) predicted sequence
        targets: (B, T, ...) ground truth sequence
        
    Returns:
        (T,) tensor of per-step MSE values
    """
    sq_error = (predictions - targets) ** 2
    # Average over all dimensions except time
    while sq_error.dim() > 2:
        sq_error = sq_error.mean(dim=-1)
    mse_per_step = sq_error.mean(dim=0)
    return mse_per_step


def compute_drift_rate(mse_curve: Tensor, eps: float = 1e-8) -> float:
    """Compute drift rate as slope of log-MSE curve.
    
    A low drift rate means stable rollouts.
    A high drift rate means exponential error growth.
    
    Args:
        mse_curve: (T,) tensor of per-step MSE values
        
    Returns:
        Drift rate (slope of linear fit to log-MSE)
    """
    log_mse = torch.log(mse_curve + eps)
    T = len(log_mse)
    steps = torch.arange(T, dtype=torch.float32, device=log_mse.device)
    x_mean = steps.mean()
    y_mean = log_mse.mean()
    cov_xy = ((steps - x_mean) * (log_mse - y_mean)).mean()
    var_x = ((steps - x_mean) ** 2).mean()
    slope = cov_xy / (var_x + eps)
    return slope.item()


def compute_stability_horizon(mse_curve: Tensor, threshold: float) -> int:
    """Compute number of steps until MSE exceeds threshold.
    
    Args:
        mse_curve: (T,) tensor of per-step MSE values
        threshold: MSE threshold (typically 2x initial MSE)
        
    Returns:
        Number of stable steps
    """
    T = len(mse_curve)
    for t, mse in enumerate(mse_curve):
        if mse.item() > threshold:
            return t
    return T  # Never exceeded threshold


def compute_energy_conservation(
    positions: Tensor,
    velocities: Tensor,
    masses: Optional[Tensor] = None,
    gravity: float = 9.81,
) -> float:
    """Compute energy conservation ratio over a sequence.
    
    For physics simulations, energy should be approximately conserved.
    A ratio close to 1.0 indicates good energy conservation.
    A ratio > 1.0 indicates energy gain (model "exploding")
    A ratio < 1.0 indicates energy loss (over-damping)
    
    Args:
        positions: (B, T, N, 3) position sequence
        velocities: (B, T, N, 3) velocity sequence
        masses: (B, N) or (N,) node masses. Default: unit mass
        gravity: Gravitational acceleration
        
    Returns:
        Ratio of final energy to initial energy
    """
    B, T, N, _ = positions.shape
    device = positions.device
    if masses is None:
        masses = torch.ones(N, device=device)
    if masses.dim() == 1:
        masses = masses.unsqueeze(0).expand(B, -1)
    
    def compute_energy(pos, vel):
        # Kinetic energy: 0.5 * m * v^2
        v_sq = (vel ** 2).sum(dim=-1)
        KE = 0.5 * (masses * v_sq).sum(dim=-1)
        # Potential energy: m * g * h (using y as height)
        h = pos[..., 1]
        PE = (masses * gravity * h).sum(dim=-1)
        return KE + PE
    
    E_initial = compute_energy(positions[:, 0], velocities[:, 0])
    E_final = compute_energy(positions[:, -1], velocities[:, -1])
    ratio = (E_final / (E_initial + 1e-8)).mean().item()
    return ratio


def compute_spectral_divergence(
    predictions: Tensor,
    targets: Tensor,
    n_frequencies: int = 32,
) -> float:
    """Compute spectral divergence between prediction and target.
    
    Compares the frequency content of predictions vs targets.
    Unstable models often:
    - Lose high-frequency detail (over-smoothing)
    - Develop spurious high-frequency artifacts
    
    Uses FFT along the time dimension.
    
    Args:
        predictions: (B, T, ...) predicted sequence
        targets: (B, T, ...) ground truth sequence
        n_frequencies: Number of frequency bins to compare
        
    Returns:
        Spectral divergence (lower = more similar spectra)
    """
    B, T = predictions.shape[:2]
    pred_flat = predictions.reshape(B, T, -1)
    targ_flat = targets.reshape(B, T, -1)
    
    pred_fft = torch.fft.rfft(pred_flat, dim=1)
    targ_fft = torch.fft.rfft(targ_flat, dim=1)
    
    pred_power = (pred_fft.abs() ** 2).mean(dim=(0, 2))
    targ_power = (targ_fft.abs() ** 2).mean(dim=(0, 2))
    
    pred_power = pred_power / (pred_power.sum() + 1e-8)
    targ_power = targ_power / (targ_power.sum() + 1e-8)
    
    n_freq = min(n_frequencies, len(pred_power))
    pred_power = pred_power[:n_freq]
    targ_power = targ_power[:n_freq]
    
    # Symmetric KL divergence
    kl_forward = F.kl_div(pred_power.log(), targ_power, reduction="sum")
    kl_backward = F.kl_div(targ_power.log(), pred_power, reduction="sum")
    divergence = 0.5 * (kl_forward + kl_backward)
    return divergence.item()


def compute_smoothness(trajectory: Tensor) -> float:
    """Compute smoothness of a trajectory (inverse of jerk magnitude).
    
    Smoother trajectories have lower jerk (derivative of acceleration).
    
    Args:
        trajectory: (B, T, N, 3) position sequence
        
    Returns:
        Smoothness score (higher = smoother)
    """
    velocity = trajectory[:, 1:] - trajectory[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]
    jerk_mag = (jerk ** 2).sum(dim=-1).sqrt().mean()
    smoothness = 1.0 / (jerk_mag.item() + 1e-8)
    return smoothness


def compute_trajectory_similarity(
    pred_trajectory: Tensor,
    target_trajectory: Tensor,
    method: str = "dtw",
) -> float:
    """Compute similarity between predicted and target trajectories.
    
    Args:
        pred_trajectory: (T, 3) or (T, N, 3) predicted trajectory
        target_trajectory: (T, 3) or (T, N, 3) target trajectory
        method: "dtw" for Dynamic Time Warping, "frechet" for Frechet distance
        
    Returns:
        Similarity score (higher = more similar)
    """
    if pred_trajectory.dim() > 2:
        pred_trajectory = pred_trajectory.reshape(pred_trajectory.size(0), -1)
        target_trajectory = target_trajectory.reshape(target_trajectory.size(0), -1)
    
    if method == "dtw":
        T1, T2 = len(pred_trajectory), len(target_trajectory)
        dist_matrix = torch.cdist(pred_trajectory, target_trajectory)
        total_dist = 0.0
        i, j = 0, 0
        while i < T1 - 1 or j < T2 - 1:
            total_dist += dist_matrix[i, j].item()
            if i == T1 - 1:
                j += 1
            elif j == T2 - 1:
                i += 1
            else:
                candidates = [
                    (i + 1, j, dist_matrix[i + 1, j]),
                    (i, j + 1, dist_matrix[i, j + 1]),
                    (i + 1, j + 1, dist_matrix[i + 1, j + 1]),
                ]
                i, j, _ = min(candidates, key=lambda x: x[2])
        total_dist += dist_matrix[-1, -1].item()
        path_length = max(T1, T2)
        similarity = 1.0 / (total_dist / path_length + 1e-8)
    elif method == "frechet":
        T = min(len(pred_trajectory), len(target_trajectory))
        max_dist = 0.0
        for t in range(T):
            dist = torch.norm(pred_trajectory[t] - target_trajectory[t]).item()
            max_dist = max(max_dist, dist)
        similarity = 1.0 / (max_dist + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    return similarity
