"""
Utility functions for WavePINN-NIF-Scalar.

This module provides helper functions for:
- JAX tree operations
- Mixed-precision casting
- Coordinate normalization
- Fourier feature encoding
- Checkpoint saving/loading
- Logging utilities
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import h5py
import pickle
from pathlib import Path


# =============================================================================
# Type Aliases
# =============================================================================
PRNGKey = jax.Array
PyTree = Any
Params = PyTree
Array = jax.Array


# =============================================================================
# Mixed Precision Utilities
# =============================================================================

def cast_to_bfloat16(tree: PyTree) -> PyTree:
    """
    Cast float32 arrays in a PyTree to bfloat16 for mixed-precision training.
    
    Args:
        tree: A JAX PyTree (e.g., model parameters).
        
    Returns:
        PyTree with float32 arrays cast to bfloat16.
    """
    def _cast(x):
        if hasattr(x, 'dtype') and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x
    return jax.tree_util.tree_map(_cast, tree)


def cast_to_float32(tree: PyTree) -> PyTree:
    """
    Cast bfloat16 arrays back to float32 (e.g., for loss computation).
    
    Args:
        tree: A JAX PyTree.
        
    Returns:
        PyTree with bfloat16 arrays cast to float32.
    """
    def _cast(x):
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x
    return jax.tree_util.tree_map(_cast, tree)


# =============================================================================
# Coordinate Normalization
# =============================================================================

def normalize_coords(
    coords: Array,
    bounds: Dict[str, Tuple[float, float]]
) -> Array:
    """
    Normalize coordinates to [-1, 1] range for better network conditioning.
    
    Args:
        coords: Array of shape (N, D) where D is dimensionality (e.g., 3 for x,z,t).
        bounds: Dictionary with keys like 'x', 'z', 't' and (min, max) tuples.
        
    Returns:
        Normalized coordinates in [-1, 1].
    """
    # Assume coords columns are [x, z, t] for 2D+time
    coord_names = list(bounds.keys())
    normalized = []
    
    for i, name in enumerate(coord_names):
        lo, hi = bounds[name]
        col = coords[:, i]
        # Map [lo, hi] -> [-1, 1]
        normalized_col = 2.0 * (col - lo) / (hi - lo + 1e-8) - 1.0
        normalized.append(normalized_col)
    
    return jnp.stack(normalized, axis=-1)


def denormalize_coords(
    coords: Array,
    bounds: Dict[str, Tuple[float, float]]
) -> Array:
    """
    Denormalize coordinates from [-1, 1] back to original range.
    
    Args:
        coords: Normalized array of shape (N, D).
        bounds: Dictionary with coordinate bounds.
        
    Returns:
        Denormalized coordinates.
    """
    coord_names = list(bounds.keys())
    denormalized = []
    
    for i, name in enumerate(coord_names):
        lo, hi = bounds[name]
        col = coords[:, i]
        # Map [-1, 1] -> [lo, hi]
        denorm_col = (col + 1.0) / 2.0 * (hi - lo) + lo
        denormalized.append(denorm_col)
    
    return jnp.stack(denormalized, axis=-1)


# =============================================================================
# Fourier Feature Encoding
# =============================================================================

def make_fourier_features(
    B: Array,
    coords: Array,
    include_input: bool = True
) -> Array:
    """
    Apply random Fourier feature encoding to mitigate spectral bias.
    
    Maps input x to [sin(2πBx), cos(2πBx), x] (if include_input=True).
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn 
    High Frequency Functions in Low Dimensional Domains", NeurIPS 2020.
    
    Args:
        B: Random Fourier basis matrix of shape (D, num_features).
        coords: Input coordinates of shape (N, D).
        include_input: Whether to append original coords.
        
    Returns:
        Encoded features of shape (N, 2*num_features + D) or (N, 2*num_features).
    """
    # Project coords onto random basis: (N, D) @ (D, F) -> (N, F)
    projected = 2.0 * jnp.pi * coords @ B
    
    # Apply sin and cos
    features = jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)
    
    if include_input:
        features = jnp.concatenate([features, coords], axis=-1)
    
    return features


def init_fourier_basis(
    key: PRNGKey,
    input_dim: int,
    num_features: int,
    scale: float = 10.0
) -> Array:
    """
    Initialize random Fourier basis matrix.
    
    Args:
        key: JAX random key.
        input_dim: Dimensionality of input coordinates.
        num_features: Number of Fourier features.
        scale: Standard deviation of Gaussian (controls frequency range).
        
    Returns:
        Random basis matrix B of shape (input_dim, num_features).
    """
    return jax.random.normal(key, (input_dim, num_features)) * scale


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(
    filepath: Union[str, Path],
    params: Params,
    opt_state: Any,
    step: int,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None
) -> None:
    """
    Save training checkpoint to disk.
    
    Args:
        filepath: Path to save checkpoint.
        params: Model parameters.
        opt_state: Optimizer state.
        step: Current training step.
        config: Optional configuration dictionary.
        metrics: Optional metrics dictionary.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'params': jax.device_get(params),
        'opt_state': jax.device_get(opt_state),
        'step': step,
        'config': config or {},
        'metrics': metrics or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: Union[str, Path]) -> Dict:
    """
    Load training checkpoint from disk.
    
    Args:
        filepath: Path to checkpoint file.
        
    Returns:
        Dictionary with params, opt_state, step, config, metrics.
    """
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


# =============================================================================
# Data I/O
# =============================================================================

def save_to_hdf5(
    filepath: Union[str, Path],
    data: Dict[str, np.ndarray],
    attrs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save arrays to HDF5 file.
    
    Args:
        filepath: Output file path.
        data: Dictionary of arrays to save.
        attrs: Optional metadata attributes.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        for key, arr in data.items():
            f.create_dataset(key, data=np.asarray(arr), compression='gzip')
        
        if attrs:
            for key, val in attrs.items():
                f.attrs[key] = val
    
    print(f"Data saved to {filepath}")


def load_from_hdf5(filepath: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load arrays from HDF5 file.
    
    Args:
        filepath: Input file path.
        
    Returns:
        Tuple of (data dict, attributes dict).
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])
        attrs = dict(f.attrs)
    
    return data, attrs


# =============================================================================
# Logging Utilities
# =============================================================================

class MetricsLogger:
    """Simple metrics logger for training progress."""
    
    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize logger.
        
        Args:
            log_dir: Optional directory to save logs.
        """
        self.history: Dict[str, list] = {}
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, step: int, **metrics) -> None:
        """
        Log metrics for a given step.
        
        Args:
            step: Training step.
            **metrics: Metric name-value pairs.
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append((step, float(value)))
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if name in self.history and self.history[name]:
            return self.history[name][-1][1]
        return None
    
    def save(self, filename: str = "metrics.pkl") -> None:
        """Save metrics to file."""
        if self.log_dir:
            with open(self.log_dir / filename, 'wb') as f:
                pickle.dump(self.history, f)
    
    def print_status(self, step: int, metrics: Dict[str, float]) -> None:
        """Print formatted status line."""
        parts = [f"Step {step:6d}"]
        for name, value in metrics.items():
            if 'loss' in name.lower():
                parts.append(f"{name}: {value:.4e}")
            else:
                parts.append(f"{name}: {value:.4f}")
        print(" | ".join(parts))


# =============================================================================
# Gradient Utilities
# =============================================================================

def compute_grad_norm(grads: PyTree) -> float:
    """
    Compute global L2 norm of gradients.
    
    Args:
        grads: PyTree of gradient arrays.
        
    Returns:
        Global gradient norm.
    """
    leaves = jax.tree_util.tree_leaves(grads)
    squared_norms = [jnp.sum(g ** 2) for g in leaves]
    return jnp.sqrt(sum(squared_norms))


def count_params(params: Params) -> int:
    """
    Count total number of parameters.
    
    Args:
        params: Model parameters PyTree.
        
    Returns:
        Total parameter count.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


# =============================================================================
# Domain Utilities
# =============================================================================

def create_grid_2d(
    nx: int,
    nz: int,
    x_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[Array, Array, Array]:
    """
    Create 2D spatial grid.
    
    Args:
        nx, nz: Grid resolution.
        x_range, z_range: Coordinate ranges.
        
    Returns:
        Tuple of (xx, zz, flat_coords) where flat_coords is (nx*nz, 2).
    """
    x = jnp.linspace(x_range[0], x_range[1], nx)
    z = jnp.linspace(z_range[0], z_range[1], nz)
    xx, zz = jnp.meshgrid(x, z, indexing='ij')
    flat_coords = jnp.stack([xx.ravel(), zz.ravel()], axis=-1)
    return xx, zz, flat_coords


def create_grid_3d(
    nx: int,
    ny: int,
    nz: int,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[Array, Array, Array, Array]:
    """
    Create 3D spatial grid.
    
    Args:
        nx, ny, nz: Grid resolution.
        x_range, y_range, z_range: Coordinate ranges.
        
    Returns:
        Tuple of (xx, yy, zz, flat_coords) where flat_coords is (nx*ny*nz, 3).
    """
    x = jnp.linspace(x_range[0], x_range[1], nx)
    y = jnp.linspace(y_range[0], y_range[1], ny)
    z = jnp.linspace(z_range[0], z_range[1], nz)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
    flat_coords = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    return xx, yy, zz, flat_coords
