"""
Standardized Trajectory Data Structures.

Provides consistent data formats for trajectory data across all PINN projects.
Supports loading from HDF5, CSV, NumPy arrays, and torch tensors.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import torch
from torch.utils.data import Dataset
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class TrajectoryBatch:
    """
    Standardized trajectory batch format.
    
    Unified representation for trajectory data across different domains:
    - Microbe paths in 2D petri dishes
    - Robot trajectories on terrain
    - Human motion sequences (joint positions)
    - Financial agent state trajectories
    
    Attributes:
        positions: [batch, time, dim] - spatial positions
        times: [batch, time] - timestamps (normalized to [0,1] or absolute)
        velocities: [batch, time-1, dim] - computed if not provided
        context: [batch, context_dim] - optional conditioning info
        metadata: dict - domain-specific additional information
    """
    positions: torch.Tensor
    times: torch.Tensor
    velocities: Optional[torch.Tensor] = None
    accelerations: Optional[torch.Tensor] = None
    context: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute derived quantities if not provided."""
        # Ensure proper shapes
        if self.positions.dim() == 2:
            self.positions = self.positions.unsqueeze(0)
        if self.times.dim() == 1:
            self.times = self.times.unsqueeze(0)
        
        # Compute velocities if not provided
        if self.velocities is None and self.positions.shape[1] > 1:
            dt = self.times[:, 1:] - self.times[:, :-1]  # [batch, time-1]
            dx = self.positions[:, 1:] - self.positions[:, :-1]  # [batch, time-1, dim]
            self.velocities = dx / (dt.unsqueeze(-1) + 1e-8)
        
        # Compute accelerations if velocities available
        if self.accelerations is None and self.velocities is not None and self.velocities.shape[1] > 1:
            dt = self.times[:, 1:-1] - self.times[:, :-2]  # [batch, time-2]
            dv = self.velocities[:, 1:] - self.velocities[:, :-1]  # [batch, time-2, dim]
            self.accelerations = dv / (dt.unsqueeze(-1) + 1e-8)
    
    @property
    def batch_size(self) -> int:
        """Number of trajectories in batch."""
        return self.positions.shape[0]
    
    @property
    def n_timesteps(self) -> int:
        """Number of time steps per trajectory."""
        return self.positions.shape[1]
    
    @property
    def spatial_dim(self) -> int:
        """Spatial dimension of trajectories."""
        return self.positions.shape[2]
    
    @property
    def start_positions(self) -> torch.Tensor:
        """Starting positions [batch, dim]."""
        return self.positions[:, 0]
    
    @property
    def end_positions(self) -> torch.Tensor:
        """Ending positions [batch, dim]."""
        return self.positions[:, -1]
    
    @property
    def duration(self) -> torch.Tensor:
        """Total duration of each trajectory [batch]."""
        return self.times[:, -1] - self.times[:, 0]
    
    def to(self, device: Union[str, torch.device]) -> 'TrajectoryBatch':
        """Move batch to device."""
        return TrajectoryBatch(
            positions=self.positions.to(device),
            times=self.times.to(device),
            velocities=self.velocities.to(device) if self.velocities is not None else None,
            accelerations=self.accelerations.to(device) if self.accelerations is not None else None,
            context=self.context.to(device) if self.context is not None else None,
            metadata=self.metadata,
        )
    
    def normalize_times(self) -> 'TrajectoryBatch':
        """Normalize times to [0, 1] range."""
        t_min = self.times.min(dim=1, keepdim=True).values
        t_max = self.times.max(dim=1, keepdim=True).values
        normalized_times = (self.times - t_min) / (t_max - t_min + 1e-8)
        
        return TrajectoryBatch(
            positions=self.positions,
            times=normalized_times,
            velocities=None,  # Will be recomputed
            accelerations=None,
            context=self.context,
            metadata={**self.metadata, 't_min': t_min, 't_max': t_max},
        )
    
    def normalize_positions(self, method: str = "unit_box") -> 'TrajectoryBatch':
        """
        Normalize positions.
        
        Args:
            method: "unit_box" (scale to [0,1]^d) or "standardize" (zero mean, unit var)
        """
        if method == "unit_box":
            p_min = self.positions.min(dim=(0, 1), keepdim=True).values
            p_max = self.positions.max(dim=(0, 1), keepdim=True).values
            normalized = (self.positions - p_min) / (p_max - p_min + 1e-8)
            norm_params = {'p_min': p_min, 'p_max': p_max, 'method': method}
        else:  # standardize
            p_mean = self.positions.mean(dim=(0, 1), keepdim=True)
            p_std = self.positions.std(dim=(0, 1), keepdim=True)
            normalized = (self.positions - p_mean) / (p_std + 1e-8)
            norm_params = {'p_mean': p_mean, 'p_std': p_std, 'method': method}
        
        return TrajectoryBatch(
            positions=normalized,
            times=self.times,
            velocities=None,
            accelerations=None,
            context=self.context,
            metadata={**self.metadata, 'position_norm': norm_params},
        )
    
    def __getitem__(self, idx) -> 'TrajectoryBatch':
        """Index into batch."""
        if isinstance(idx, int):
            idx = [idx]
        
        return TrajectoryBatch(
            positions=self.positions[idx],
            times=self.times[idx],
            velocities=self.velocities[idx] if self.velocities is not None else None,
            accelerations=self.accelerations[idx] if self.accelerations is not None else None,
            context=self.context[idx] if self.context is not None else None,
            metadata=self.metadata,
        )


def create_trajectory_batch(
    positions: Union[torch.Tensor, np.ndarray, List],
    times: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    context: Optional[Union[torch.Tensor, np.ndarray]] = None,
    **metadata,
) -> TrajectoryBatch:
    """
    Create TrajectoryBatch from various input formats.
    
    Args:
        positions: Positions as tensor, array, or list. Shape [batch, time, dim] or [time, dim]
        times: Optional time points. If None, creates uniform times in [0, 1]
        context: Optional context tensor
        **metadata: Additional metadata fields
        
    Returns:
        TrajectoryBatch instance
    """
    # Convert to tensor
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).float()
    elif isinstance(positions, list):
        positions = torch.tensor(positions, dtype=torch.float32)
    
    # Ensure 3D
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    
    # Create times if not provided
    if times is None:
        n_times = positions.shape[1]
        times = torch.linspace(0, 1, n_times).unsqueeze(0).expand(positions.shape[0], -1)
    elif isinstance(times, np.ndarray):
        times = torch.from_numpy(times).float()
    elif isinstance(times, list):
        times = torch.tensor(times, dtype=torch.float32)
    
    if times.dim() == 1:
        times = times.unsqueeze(0).expand(positions.shape[0], -1)
    
    # Context
    if context is not None and isinstance(context, np.ndarray):
        context = torch.from_numpy(context).float()
    
    return TrajectoryBatch(
        positions=positions,
        times=times,
        context=context,
        metadata=metadata or {},
    )


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory data.
    
    Supports loading from:
    - HDF5 files (recommended for large datasets)
    - CSV files
    - NumPy arrays
    - In-memory tensors
    """
    
    def __init__(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize trajectory dataset.
        
        Args:
            positions: [n_trajectories, n_timesteps, spatial_dim]
            times: [n_trajectories, n_timesteps]
            context: [n_trajectories, context_dim] optional
            transform: Optional transform to apply to each sample
        """
        self.positions = positions
        self.times = times
        self.context = context
        self.transform = transform
    
    def __len__(self) -> int:
        return self.positions.shape[0]
    
    def __getitem__(self, idx) -> TrajectoryBatch:
        batch = TrajectoryBatch(
            positions=self.positions[idx:idx+1],
            times=self.times[idx:idx+1],
            context=self.context[idx:idx+1] if self.context is not None else None,
        )
        
        if self.transform is not None:
            batch = self.transform(batch)
        
        return batch
    
    @classmethod
    def from_hdf5(cls, path: str, position_key: str = 'positions', time_key: str = 'times') -> 'TrajectoryDataset':
        """
        Load dataset from HDF5 file.
        
        Expected schema:
            positions: [n_trajectories, n_timesteps, spatial_dim]
            times: [n_trajectories, n_timesteps]
            context: [n_trajectories, context_dim] (optional)
        """
        if not HAS_H5PY:
            raise ImportError("h5py required to load HDF5 files: pip install h5py")
        
        with h5py.File(path, 'r') as f:
            positions = torch.tensor(f[position_key][:], dtype=torch.float32)
            times = torch.tensor(f[time_key][:], dtype=torch.float32)
            context = None
            if 'context' in f:
                context = torch.tensor(f['context'][:], dtype=torch.float32)
        
        return cls(positions, times, context)
    
    @classmethod
    def from_csv(
        cls,
        path: str,
        position_cols: List[str],
        time_col: str,
        trajectory_col: Optional[str] = None,
        context_cols: Optional[List[str]] = None,
    ) -> 'TrajectoryDataset':
        """
        Load dataset from CSV file.
        
        Args:
            path: Path to CSV file
            position_cols: Column names for position (e.g., ['x', 'y'])
            time_col: Column name for time
            trajectory_col: Column to group by (e.g., 'trajectory_id'). If None, single trajectory.
            context_cols: Additional context columns
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required to load CSV files: pip install pandas")
        
        df = pd.read_csv(path)
        
        if trajectory_col is None:
            # Single trajectory
            positions = torch.tensor(df[position_cols].values, dtype=torch.float32).unsqueeze(0)
            times = torch.tensor(df[time_col].values, dtype=torch.float32).unsqueeze(0)
            context = None
            if context_cols:
                context = torch.tensor(df[context_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0)
        else:
            # Multiple trajectories
            grouped = df.groupby(trajectory_col)
            positions_list = []
            times_list = []
            context_list = []
            
            for _, group in grouped:
                positions_list.append(group[position_cols].values)
                times_list.append(group[time_col].values)
                if context_cols:
                    context_list.append(group[context_cols].iloc[0].values)
            
            # Pad to same length
            max_len = max(p.shape[0] for p in positions_list)
            dim = positions_list[0].shape[1]
            
            positions = torch.zeros(len(positions_list), max_len, dim)
            times = torch.zeros(len(times_list), max_len)
            
            for i, (p, t) in enumerate(zip(positions_list, times_list)):
                positions[i, :len(p)] = torch.tensor(p, dtype=torch.float32)
                times[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
            
            context = None
            if context_cols:
                context = torch.tensor(np.array(context_list), dtype=torch.float32)
        
        return cls(positions, times, context)
    
    @classmethod
    def from_numpy(
        cls,
        positions: np.ndarray,
        times: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> 'TrajectoryDataset':
        """Create dataset from NumPy arrays."""
        positions_t = torch.tensor(positions, dtype=torch.float32)
        
        if positions_t.dim() == 2:
            positions_t = positions_t.unsqueeze(0)
        
        if times is None:
            n_times = positions_t.shape[1]
            times_t = torch.linspace(0, 1, n_times).unsqueeze(0).expand(positions_t.shape[0], -1)
        else:
            times_t = torch.tensor(times, dtype=torch.float32)
            if times_t.dim() == 1:
                times_t = times_t.unsqueeze(0).expand(positions_t.shape[0], -1)
        
        context_t = None
        if context is not None:
            context_t = torch.tensor(context, dtype=torch.float32)
        
        return cls(positions_t, times_t, context_t)
    
    def save_hdf5(self, path: str):
        """Save dataset to HDF5 file."""
        if not HAS_H5PY:
            raise ImportError("h5py required to save HDF5 files: pip install h5py")
        
        with h5py.File(path, 'w') as f:
            f.create_dataset('positions', data=self.positions.numpy())
            f.create_dataset('times', data=self.times.numpy())
            if self.context is not None:
                f.create_dataset('context', data=self.context.numpy())
