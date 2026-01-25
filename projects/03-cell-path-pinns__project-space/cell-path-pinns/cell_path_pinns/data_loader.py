"""
Data Loading Utilities for Cell-Path PINNs

Supports loading trajectory data from:
- CSV files (standard format)
- TrackMate XML exports
- Multi-trajectory files
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union


def load_trajectory_csv(
    filepath: str,
    t_col: str = 'time',
    x_col: str = 'x',
    y_col: str = 'y',
    trajectory_col: Optional[str] = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Load trajectory data from CSV file.

    Args:
        filepath: Path to CSV file
        t_col: Column name for time
        x_col: Column name for x position
        y_col: Column name for y position
        trajectory_col: Optional column for trajectory ID (for multi-trajectory files)

    Returns:
        If trajectory_col is None: Tuple of (t, x, y) arrays
        If trajectory_col is set: Dict mapping trajectory IDs to (t, x, y) tuples
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load CSV files. Install with: pip install pandas")

    df = pd.read_csv(filepath)

    # Validate columns exist
    required_cols = [t_col, x_col, y_col]
    if trajectory_col:
        required_cols.append(trajectory_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

    if trajectory_col:
        # Return dict of trajectories
        trajectories = {}
        for traj_id in df[trajectory_col].unique():
            mask = df[trajectory_col] == traj_id
            trajectories[traj_id] = (
                df.loc[mask, t_col].values.astype(np.float32),
                df.loc[mask, x_col].values.astype(np.float32),
                df.loc[mask, y_col].values.astype(np.float32)
            )
        return trajectories

    return (
        df[t_col].values.astype(np.float32),
        df[x_col].values.astype(np.float32),
        df[y_col].values.astype(np.float32)
    )


def load_trackmate_xml(
    filepath: str,
    min_track_length: int = 10
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load trajectories from TrackMate XML export.

    TrackMate is a popular ImageJ/Fiji plugin for particle tracking.
    This function parses its XML export format.

    Args:
        filepath: Path to TrackMate XML file
        min_track_length: Minimum number of spots per track (filters short tracks)

    Returns:
        List of (t, x, y) tuples, one per track
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()

    trajectories = []

    # TrackMate XML structure can vary, try common patterns
    # Pattern 1: Model/AllTracks/Track
    tracks = root.findall('.//Track')
    if not tracks:
        # Pattern 2: Model/FilteredTracks/TrackID with AllSpots
        tracks = root.findall('.//AllTracks/Track')

    for track in tracks:
        t_list, x_list, y_list = [], [], []

        # Get spots for this track
        # Try different spot access patterns
        spots = track.findall('.//Spot')
        if not spots:
            spots = track.findall('Spot')

        for spot in spots:
            # TrackMate uses FRAME for time, POSITION_X/Y for coordinates
            frame = spot.get('FRAME', spot.get('t', spot.get('frame')))
            pos_x = spot.get('POSITION_X', spot.get('x', spot.get('X')))
            pos_y = spot.get('POSITION_Y', spot.get('y', spot.get('Y')))

            if frame is not None and pos_x is not None and pos_y is not None:
                t_list.append(float(frame))
                x_list.append(float(pos_x))
                y_list.append(float(pos_y))

        # Filter by minimum length
        if len(t_list) >= min_track_length:
            # Sort by time
            order = np.argsort(t_list)
            trajectories.append((
                np.array(t_list, dtype=np.float32)[order],
                np.array(x_list, dtype=np.float32)[order],
                np.array(y_list, dtype=np.float32)[order]
            ))

    return trajectories


class TrajectoryDataset:
    """
    PyTorch-compatible dataset for trajectory data.

    Handles normalization and provides consistent interface for training.
    """

    def __init__(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        normalize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            trajectories: List of (t, x, y) tuples
            normalize: Whether to normalize data to [0, 1] range
        """
        self.trajectories = trajectories
        self.normalize = normalize
        self.norm_params: Dict[str, float] = {}

        if normalize and trajectories:
            self._compute_normalization()

    def _compute_normalization(self):
        """Compute global normalization parameters across all trajectories."""
        all_t = np.concatenate([t for t, x, y in self.trajectories])
        all_x = np.concatenate([x for t, x, y in self.trajectories])
        all_y = np.concatenate([y for t, x, y in self.trajectories])

        self.norm_params = {
            't_min': float(all_t.min()),
            't_max': float(all_t.max()),
            't_range': float(all_t.max() - all_t.min()) or 1.0,
            'x_min': float(all_x.min()),
            'x_max': float(all_x.max()),
            'x_range': float(all_x.max() - all_x.min()) or 1.0,
            'y_min': float(all_y.min()),
            'y_max': float(all_y.max()),
            'y_range': float(all_y.max() - all_y.min()) or 1.0,
        }

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a trajectory by index.

        Returns normalized data if normalization is enabled.
        """
        t, x, y = self.trajectories[idx]

        if self.normalize and self.norm_params:
            t = (t - self.norm_params['t_min']) / self.norm_params['t_range']
            x = (x - self.norm_params['x_min']) / self.norm_params['x_range']
            y = (y - self.norm_params['y_min']) / self.norm_params['y_range']

        return t, x, y

    def get_normalization_params(self) -> Dict[str, float]:
        """Get normalization parameters for denormalization."""
        return self.norm_params.copy()

    def denormalize(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reverse normalization to get original scale.

        Args:
            t, x, y: Normalized arrays

        Returns:
            Original-scale (t, x, y) arrays
        """
        if not self.norm_params:
            return t, x, y

        t_orig = t * self.norm_params['t_range'] + self.norm_params['t_min']
        x_orig = x * self.norm_params['x_range'] + self.norm_params['x_min']
        y_orig = y * self.norm_params['y_range'] + self.norm_params['y_min']

        return t_orig, x_orig, y_orig

    def get_combined(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all trajectories combined into single arrays.

        Useful for training on all data at once.

        Returns:
            Combined (t, x, y) arrays
        """
        all_t, all_x, all_y = [], [], []

        for idx in range(len(self)):
            t, x, y = self[idx]
            all_t.append(t)
            all_x.append(x)
            all_y.append(y)

        return (
            np.concatenate(all_t),
            np.concatenate(all_x),
            np.concatenate(all_y)
        )


def save_trajectory_csv(
    filepath: str,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    trajectory_id: Optional[str] = None
) -> None:
    """
    Save trajectory data to CSV file.

    Args:
        filepath: Output file path
        t: Time array
        x: X position array
        y: Y position array
        trajectory_id: Optional trajectory identifier (added as column)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to save CSV files. Install with: pip install pandas")

    data = {'time': t, 'x': x, 'y': y}
    if trajectory_id is not None:
        data['trajectory_id'] = [trajectory_id] * len(t)

    df = pd.DataFrame(data)

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
