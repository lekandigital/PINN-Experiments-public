"""
Standard accuracy metrics for benchmarking.

Provides RMSE, NRMSE, L2 relative error, and other common metrics.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Convert tensor to numpy array regardless of framework."""
    if isinstance(x, np.ndarray):
        return x
    
    # PyTorch
    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        return x.detach().cpu().numpy()
    
    # JAX
    if hasattr(x, 'block_until_ready'):
        x.block_until_ready()
    if hasattr(x, '__array__'):
        return np.asarray(x)
    
    # TensorFlow
    if hasattr(x, 'numpy') and callable(x.numpy):
        return x.numpy()
    
    # Last resort
    return np.asarray(x)


def compute_rmse(
    predictions: Any,
    targets: Any,
    axis: Optional[Union[int, tuple]] = None,
) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(mean((predictions - targets)^2))
    
    Args:
        predictions: Predicted values (any tensor type)
        targets: Ground truth values (any tensor type)
        axis: Axis or axes along which to compute (None = global)
    
    Returns:
        RMSE value as float
    """
    pred = to_numpy(predictions).astype(np.float64)
    tgt = to_numpy(targets).astype(np.float64)
    
    mse = np.mean((pred - tgt) ** 2, axis=axis)
    return float(np.sqrt(mse))


def compute_mse(
    predictions: Any,
    targets: Any,
    axis: Optional[Union[int, tuple]] = None,
) -> float:
    """
    Compute Mean Squared Error.
    
    MSE = mean((predictions - targets)^2)
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        axis: Axis or axes along which to compute
    
    Returns:
        MSE value as float
    """
    pred = to_numpy(predictions).astype(np.float64)
    tgt = to_numpy(targets).astype(np.float64)
    
    return float(np.mean((pred - tgt) ** 2, axis=axis))


def compute_mae(
    predictions: Any,
    targets: Any,
    axis: Optional[Union[int, tuple]] = None,
) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = mean(|predictions - targets|)
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        axis: Axis or axes along which to compute
    
    Returns:
        MAE value as float
    """
    pred = to_numpy(predictions).astype(np.float64)
    tgt = to_numpy(targets).astype(np.float64)
    
    return float(np.mean(np.abs(pred - tgt), axis=axis))


def compute_nrmse(
    predictions: Any,
    targets: Any,
    normalization: str = "range",
) -> float:
    """
    Compute Normalized Root Mean Squared Error.
    
    NRMSE allows comparison across different scales.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        normalization: How to normalize:
            - "range": RMSE / (max - min) of targets
            - "std": RMSE / std of targets
            - "mean": RMSE / mean of targets (relative RMSE)
            - "iqr": RMSE / interquartile range
    
    Returns:
        NRMSE value as float (dimensionless)
    """
    pred = to_numpy(predictions).astype(np.float64)
    tgt = to_numpy(targets).astype(np.float64)
    
    rmse = compute_rmse(pred, tgt)
    
    if normalization == "range":
        norm_factor = np.max(tgt) - np.min(tgt)
    elif normalization == "std":
        norm_factor = np.std(tgt)
    elif normalization == "mean":
        norm_factor = np.abs(np.mean(tgt))
    elif normalization == "iqr":
        q75, q25 = np.percentile(tgt, [75, 25])
        norm_factor = q75 - q25
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    if norm_factor == 0 or np.isnan(norm_factor):
        return float('inf')
    
    return float(rmse / norm_factor)


def compute_l2_relative_error(
    predictions: Any,
    targets: Any,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute L2 relative error.
    
    L2_rel = ||predictions - targets||_2 / ||targets||_2
    
    Standard metric in PDE solver benchmarking.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to prevent division by zero
    
    Returns:
        L2 relative error as float (dimensionless)
    """
    pred = to_numpy(predictions).astype(np.float64).flatten()
    tgt = to_numpy(targets).astype(np.float64).flatten()
    
    diff_norm = np.linalg.norm(pred - tgt)
    tgt_norm = np.linalg.norm(tgt)
    
    if tgt_norm < epsilon:
        return float('inf') if diff_norm > epsilon else 0.0
    
    return float(diff_norm / tgt_norm)


def compute_l1_relative_error(
    predictions: Any,
    targets: Any,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute L1 relative error.
    
    L1_rel = ||predictions - targets||_1 / ||targets||_1
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to prevent division by zero
    
    Returns:
        L1 relative error as float
    """
    pred = to_numpy(predictions).astype(np.float64).flatten()
    tgt = to_numpy(targets).astype(np.float64).flatten()
    
    diff_norm = np.sum(np.abs(pred - tgt))
    tgt_norm = np.sum(np.abs(tgt))
    
    if tgt_norm < epsilon:
        return float('inf') if diff_norm > epsilon else 0.0
    
    return float(diff_norm / tgt_norm)


def compute_max_error(
    predictions: Any,
    targets: Any,
) -> float:
    """
    Compute maximum absolute error (L-infinity norm).
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Maximum absolute error as float
    """
    pred = to_numpy(predictions).astype(np.float64)
    tgt = to_numpy(targets).astype(np.float64)
    
    return float(np.max(np.abs(pred - tgt)))


def compute_r2_score(
    predictions: Any,
    targets: Any,
) -> float:
    """
    Compute R-squared (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        R² score (1.0 = perfect, 0.0 = baseline, negative = worse than mean)
    """
    pred = to_numpy(predictions).astype(np.float64).flatten()
    tgt = to_numpy(targets).astype(np.float64).flatten()
    
    ss_res = np.sum((tgt - pred) ** 2)
    ss_tot = np.sum((tgt - np.mean(tgt)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float('-inf')
    
    return float(1 - ss_res / ss_tot)


def compute_psnr(
    predictions: Any,
    targets: Any,
    max_value: Optional[float] = None,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(max_value^2 / MSE)
    
    Common in image reconstruction tasks.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        max_value: Maximum possible value (default: max of targets)
    
    Returns:
        PSNR in dB
    """
    pred = to_numpy(predictions).astype(np.float64)
    tgt = to_numpy(targets).astype(np.float64)
    
    if max_value is None:
        max_value = np.max(tgt)
    
    mse = compute_mse(pred, tgt)
    
    if mse == 0:
        return float('inf')
    
    return float(10 * np.log10(max_value ** 2 / mse))


# =============================================================================
# Domain-specific metrics
# =============================================================================

def compute_chamfer_distance(
    points_pred: Any,
    points_target: Any,
) -> float:
    """
    Compute Chamfer distance between two point clouds.
    
    CD = mean(min_dist(pred->target)) + mean(min_dist(target->pred))
    
    Used for mesh/surface reconstruction evaluation.
    
    Args:
        points_pred: Predicted points (N, 3)
        points_target: Target points (M, 3)
    
    Returns:
        Chamfer distance as float
    """
    pred = to_numpy(points_pred).astype(np.float64)
    tgt = to_numpy(points_target).astype(np.float64)
    
    # Compute pairwise distances using broadcasting
    # pred: (N, 1, 3), tgt: (1, M, 3) -> (N, M)
    diff = pred[:, np.newaxis, :] - tgt[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # pred -> target: for each pred point, find nearest target
    min_pred_to_tgt = np.min(dists, axis=1)
    
    # target -> pred: for each target point, find nearest pred
    min_tgt_to_pred = np.min(dists, axis=0)
    
    return float(np.mean(min_pred_to_tgt) + np.mean(min_tgt_to_pred))


def compute_eikonal_violation(
    sdf_gradients: Any,
) -> float:
    """
    Compute eikonal equation violation for SDF.
    
    Eikonal: |∇SDF| = 1
    Violation: mean(||∇SDF| - 1|)
    
    Args:
        sdf_gradients: Gradient vectors of SDF (N, 3)
    
    Returns:
        Mean eikonal violation
    """
    grads = to_numpy(sdf_gradients).astype(np.float64)
    
    # Compute gradient magnitudes
    grad_norms = np.linalg.norm(grads, axis=-1)
    
    # Violation is deviation from 1
    violation = np.abs(grad_norms - 1.0)
    
    return float(np.mean(violation))


def compute_edge_preservation_error(
    positions_pred: Any,
    positions_target: Any,
    edges: Any,
) -> float:
    """
    Compute edge length preservation error for mesh deformation.
    
    Used in cloth simulation to measure structural integrity.
    
    Args:
        positions_pred: Predicted vertex positions (N, 3)
        positions_target: Target vertex positions (N, 3)
        edges: Edge indices (E, 2)
    
    Returns:
        Mean relative edge length error
    """
    pred = to_numpy(positions_pred).astype(np.float64)
    tgt = to_numpy(positions_target).astype(np.float64)
    edges = to_numpy(edges).astype(np.int64)
    
    # Compute edge lengths
    def edge_lengths(positions, edge_idx):
        v0 = positions[edge_idx[:, 0]]
        v1 = positions[edge_idx[:, 1]]
        return np.linalg.norm(v1 - v0, axis=-1)
    
    pred_lengths = edge_lengths(pred, edges)
    tgt_lengths = edge_lengths(tgt, edges)
    
    # Relative error
    rel_error = np.abs(pred_lengths - tgt_lengths) / (tgt_lengths + 1e-10)
    
    return float(np.mean(rel_error))


def compute_divergence_residual(
    vector_field: Any,
    dx: float = 1.0,
) -> float:
    """
    Compute divergence residual for vector field (e.g., E or B field).
    
    For Maxwell's equations: ∇·E = ρ/ε₀, ∇·B = 0
    
    Args:
        vector_field: Vector field (Nx, Ny, Nz, 3) or (N, 3)
        dx: Grid spacing
    
    Returns:
        Mean absolute divergence
    """
    field = to_numpy(vector_field).astype(np.float64)
    
    if field.ndim == 4:
        # Structured grid (Nx, Ny, Nz, 3)
        dvx_dx = np.gradient(field[..., 0], dx, axis=0)
        dvy_dy = np.gradient(field[..., 1], dx, axis=1)
        dvz_dz = np.gradient(field[..., 2], dx, axis=2)
        div = dvx_dx + dvy_dy + dvz_dz
    else:
        # Unstructured - would need mesh connectivity
        # For now, return 0 (not computable without mesh)
        return 0.0
    
    return float(np.mean(np.abs(div)))


def compute_physics_residual(
    residuals: Any,
) -> dict[str, float]:
    """
    Compute statistics for physics residuals.
    
    Args:
        residuals: Residual values (can be dict of named residuals or array)
    
    Returns:
        Dict with mean, max, std of residuals
    """
    if isinstance(residuals, dict):
        result = {}
        for name, res in residuals.items():
            res_np = to_numpy(res).astype(np.float64).flatten()
            result[f"{name}_mean"] = float(np.mean(np.abs(res_np)))
            result[f"{name}_max"] = float(np.max(np.abs(res_np)))
            result[f"{name}_std"] = float(np.std(res_np))
        return result
    else:
        res_np = to_numpy(residuals).astype(np.float64).flatten()
        return {
            "residual_mean": float(np.mean(np.abs(res_np))),
            "residual_max": float(np.max(np.abs(res_np))),
            "residual_std": float(np.std(res_np)),
        }


def compute_all_standard_metrics(
    predictions: Any,
    targets: Any,
) -> dict[str, float]:
    """
    Compute all standard metrics at once.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Dict with all standard metrics
    """
    return {
        "rmse": compute_rmse(predictions, targets),
        "mse": compute_mse(predictions, targets),
        "mae": compute_mae(predictions, targets),
        "nrmse_range": compute_nrmse(predictions, targets, "range"),
        "nrmse_std": compute_nrmse(predictions, targets, "std"),
        "l2_relative_error": compute_l2_relative_error(predictions, targets),
        "l1_relative_error": compute_l1_relative_error(predictions, targets),
        "max_error": compute_max_error(predictions, targets),
        "r2_score": compute_r2_score(predictions, targets),
    }
