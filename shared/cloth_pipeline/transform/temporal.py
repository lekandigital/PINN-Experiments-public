"""
Temporal feature computation for cloth mesh sequences.

Computes velocities, accelerations, and creates training windows
for temporal models (Projects 05, 08, 09, 12).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Generator

from ..types import ClothSequence


@dataclass
class TemporalData:
    """
    Temporal features for a sequence.
    
    Attributes:
        velocities: Vertex velocities. Shape: (F, N, 3)
        accelerations: Vertex accelerations. Shape: (F, N, 3)
        windows: List of temporal windows for training
    """
    velocities: np.ndarray
    accelerations: Optional[np.ndarray] = None
    windows: List['TemporalWindow'] = field(default_factory=list)


@dataclass
class TemporalWindow:
    """
    A single temporal training window.
    
    Attributes:
        start_frame: Starting frame index in the original sequence
        end_frame: Ending frame index (exclusive)
        positions: Vertex positions in this window. Shape: (window_length, N, 3)
        velocities: Vertex velocities in this window. Shape: (window_length, N, 3)
        accelerations: Vertex accelerations (optional)
        times: Timestamps for each frame in the window
    """
    start_frame: int
    end_frame: int
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    
    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def compute_temporal_features(
    sequence: ClothSequence,
    compute_accelerations: bool = True,
    velocity_method: str = 'central',
) -> TemporalData:
    """
    Compute temporal features (velocities and accelerations) for a sequence.
    
    Args:
        sequence: Input ClothSequence
        compute_accelerations: Whether to compute accelerations
        velocity_method: 'forward', 'backward', or 'central' differencing
        
    Returns:
        TemporalData with computed features
    """
    # Compute velocities
    velocities = sequence.compute_velocities(method=velocity_method)
    
    # Compute accelerations if requested
    accelerations = None
    if compute_accelerations:
        accelerations = sequence.compute_accelerations()
    
    return TemporalData(
        velocities=velocities,
        accelerations=accelerations,
    )


def create_temporal_windows(
    sequence: ClothSequence,
    temporal_data: TemporalData,
    window_length: int = 30,
    window_stride: int = 10,
    include_times: bool = True,
) -> TemporalData:
    """
    Slice a sequence into overlapping training windows.
    
    Args:
        sequence: Input ClothSequence
        temporal_data: Precomputed temporal features
        window_length: Number of frames per window
        window_stride: Frames between window starts
        include_times: Whether to include timestamps
        
    Returns:
        TemporalData with windows populated
    """
    num_frames = sequence.num_frames
    windows = []
    
    # Generate window start positions
    start_positions = list(range(0, num_frames - window_length + 1, window_stride))
    
    # If last window doesn't reach the end, add one more
    if start_positions and start_positions[-1] + window_length < num_frames:
        start_positions.append(num_frames - window_length)
    
    for start in start_positions:
        end = start + window_length
        
        # Extract window data
        positions = sequence.vertices[start:end].copy()
        velocities = temporal_data.velocities[start:end].copy()
        
        accelerations = None
        if temporal_data.accelerations is not None:
            accelerations = temporal_data.accelerations[start:end].copy()
        
        times = None
        if include_times:
            times = np.arange(start, end) * sequence.dt
        
        windows.append(TemporalWindow(
            start_frame=start,
            end_frame=end,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            times=times,
        ))
    
    temporal_data.windows = windows
    return temporal_data


def iterate_windows(
    sequence: ClothSequence,
    window_length: int = 30,
    window_stride: int = 10,
    compute_velocities: bool = True,
) -> Generator[TemporalWindow, None, None]:
    """
    Generator that yields temporal windows one at a time.
    
    More memory-efficient than create_temporal_windows for large sequences.
    
    Args:
        sequence: Input ClothSequence
        window_length: Frames per window
        window_stride: Stride between windows
        compute_velocities: Whether to compute velocities
        
    Yields:
        TemporalWindow objects
    """
    num_frames = sequence.num_frames
    dt = sequence.dt
    
    for start in range(0, num_frames - window_length + 1, window_stride):
        end = start + window_length
        
        positions = sequence.vertices[start:end]
        
        if compute_velocities:
            # Compute velocities for this window
            velocities = np.zeros_like(positions)
            velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
            velocities[0] = (positions[1] - positions[0]) / dt
            velocities[-1] = (positions[-1] - positions[-2]) / dt
        else:
            velocities = np.zeros_like(positions)
        
        times = np.arange(start, end) * dt
        
        yield TemporalWindow(
            start_frame=start,
            end_frame=end,
            positions=positions,
            velocities=velocities,
            times=times,
        )


def compute_window_statistics(windows: List[TemporalWindow]) -> dict:
    """
    Compute statistics across all temporal windows.
    
    Useful for normalization.
    """
    all_positions = np.concatenate([w.positions for w in windows], axis=0)
    all_velocities = np.concatenate([w.velocities for w in windows], axis=0)
    
    stats = {
        'num_windows': len(windows),
        'window_length': windows[0].length if windows else 0,
        'position_mean': all_positions.mean(axis=(0, 1)).tolist(),
        'position_std': all_positions.std(axis=(0, 1)).tolist(),
        'velocity_mean': all_velocities.mean(axis=(0, 1)).tolist(),
        'velocity_std': all_velocities.std(axis=(0, 1)).tolist(),
        'velocity_max': np.abs(all_velocities).max(),
    }
    
    if windows and windows[0].accelerations is not None:
        all_accels = np.concatenate([w.accelerations for w in windows], axis=0)
        stats['acceleration_mean'] = all_accels.mean(axis=(0, 1)).tolist()
        stats['acceleration_std'] = all_accels.std(axis=(0, 1)).tolist()
    
    return stats


def interpolate_temporal(
    sequence: ClothSequence,
    num_substeps: int = 2,
    method: str = 'linear',
) -> ClothSequence:
    """
    Interpolate additional frames between existing frames.
    
    Useful for creating continuous-time training data.
    
    Args:
        sequence: Input sequence
        num_substeps: Number of interpolated frames between each original frame
        method: 'linear' or 'cubic' interpolation
        
    Returns:
        New ClothSequence with interpolated frames
    """
    from scipy.interpolate import interp1d, CubicSpline
    
    original_frames = sequence.num_frames
    new_frames = (original_frames - 1) * (num_substeps + 1) + 1
    
    # Original timestamps
    original_times = np.arange(original_frames)
    
    # New timestamps
    new_times = np.linspace(0, original_frames - 1, new_frames)
    
    # Interpolate vertices
    new_vertices = np.zeros((new_frames, sequence.num_vertices, 3), dtype=np.float32)
    
    for v in range(sequence.num_vertices):
        for c in range(3):
            if method == 'linear':
                f = interp1d(original_times, sequence.vertices[:, v, c])
            else:
                f = CubicSpline(original_times, sequence.vertices[:, v, c])
            new_vertices[:, v, c] = f(new_times)
    
    # Interpolate normals if present
    new_normals = None
    if sequence.normals is not None:
        new_normals = np.zeros((new_frames, sequence.num_vertices, 3), dtype=np.float32)
        for v in range(sequence.num_vertices):
            for c in range(3):
                if method == 'linear':
                    f = interp1d(original_times, sequence.normals[:, v, c])
                else:
                    f = CubicSpline(original_times, sequence.normals[:, v, c])
                new_normals[:, v, c] = f(new_times)
        
        # Re-normalize
        norms = np.linalg.norm(new_normals, axis=2, keepdims=True)
        new_normals = new_normals / np.maximum(norms, 1e-10)
    
    new_dt = sequence.dt / (num_substeps + 1)
    new_fps = 1.0 / new_dt
    
    return ClothSequence(
        vertices=new_vertices,
        faces=sequence.faces.copy(),
        normals=new_normals,
        uvs=sequence.uvs.copy() if sequence.uvs is not None else None,
        fps=new_fps,
        dt=new_dt,
        frame_range=(0, new_frames - 1),
        metadata={
            **sequence.metadata,
            'interpolated': True,
            'original_frames': original_frames,
            'num_substeps': num_substeps,
            'interpolation_method': method,
        }
    )
