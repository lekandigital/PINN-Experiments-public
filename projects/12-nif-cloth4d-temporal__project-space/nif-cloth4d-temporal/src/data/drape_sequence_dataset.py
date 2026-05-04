"""
Dataset + point-sampling utilities for the drape-over-sphere scene.

Modeled on Project 13's `SDFDataset` (one frame = one dataset item) + its mixed
near-surface/uniform sampler. Kept inside Project 12's src/ so training does
not need a runtime dependency on P13.

Both near-surface and uniform-volume samples carry **analytic SDF labels**
(``sdf_at_points``) rather than voxel-corner lookups. The HDF5-stored SDF
grid is used only to detect the near-surface band for the sampling mask;
labels never go through voxel quantization. This removes a ~half-pitch
(≈ 8e-3) label-noise floor that previously capped achievable val MSE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .drape_sdf_generator import load_sdf_from_hdf5, sdf_at_points


def sample_points_from_sdf(
    sdf: np.ndarray,
    coords: np.ndarray,
    time: float,
    n_samples: int = 4096,
    near_surface_ratio: float = 0.7,
    surface_band: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (x, y, z, t) points and their SDF values.

    Strategy:
      * ``near_surface_ratio`` points are drawn from voxels with
        ``|sdf| < surface_band`` (jittered sub-voxel for sub-voxel coverage).
      * Remaining points are uniform within the volume.
      * **Labels are computed analytically at the jittered / uniform
        coordinates via ``sdf_at_points(xyz, time)``** — not looked up from the
        voxel grid. The grid only defines the near-surface mask.

    Returns:
        points: (n_samples, 4) float32, columns (x, y, z, t).
        sdf_values: (n_samples, 1) float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_size = int(sdf.shape[0])
    bounds = (float(coords[0]), float(coords[-1]))
    span = bounds[1] - bounds[0]
    voxel = span / (grid_size - 1)

    n_near = int(n_samples * near_surface_ratio)
    n_uniform = n_samples - n_near

    # Near-surface samples (jittered within voxel)
    near_mask = np.abs(sdf) < surface_band
    near_idx = np.argwhere(near_mask)
    if n_near > 0 and len(near_idx) > 0:
        pick = rng.integers(0, len(near_idx), size=n_near)
        voxels = near_idx[pick]
        jitter = rng.uniform(-0.5, 0.5, size=voxels.shape) * voxel
        xyz_near = coords[voxels] + jitter
    else:
        xyz_near = np.zeros((0, 3), dtype=np.float32)

    # Uniform volume samples
    xyz_uniform = rng.uniform(
        bounds[0], bounds[1], size=(n_uniform, 3)
    ).astype(np.float32)

    xyz = np.concatenate([xyz_near, xyz_uniform], axis=0).astype(np.float32)
    # Analytic labels at the exact sampled coordinates — no voxel quantization.
    sdf_vals = sdf_at_points(xyz, time).astype(np.float32)

    t_col = np.full((xyz.shape[0], 1), float(time), dtype=np.float32)
    points = np.concatenate([xyz, t_col], axis=1)

    perm = rng.permutation(points.shape[0])
    points = points[perm]
    sdf_vals = sdf_vals[perm].reshape(-1, 1)
    return points, sdf_vals


class DrapeSDFDataset(Dataset):
    """
    One dataset item = one frame. ``__getitem__`` returns a fresh random
    subsample of that frame's SDF volume each epoch.
    """

    def __init__(
        self,
        data_dir: str,
        samples_per_frame: int = 4096,
        near_surface_ratio: float = 0.7,
        surface_band: float = 0.1,
        frame_indices: Optional[list] = None,
        preload: bool = True,
    ) -> None:
        self.sdf_dir = Path(data_dir) / "sdf"
        self.samples_per_frame = samples_per_frame
        self.near_surface_ratio = near_surface_ratio
        self.surface_band = surface_band
        self.preload = preload

        files = sorted(self.sdf_dir.glob("*.h5"))
        if frame_indices is not None:
            files = [files[i] for i in frame_indices if 0 <= i < len(files)]
        if not files:
            raise ValueError(f"No HDF5 frames found in {self.sdf_dir}")
        self.files = files

        if preload:
            self._cache = []
            for f in files:
                sdf, coords, attrs = load_sdf_from_hdf5(str(f))
                self._cache.append(
                    {"sdf": sdf, "coords": coords, "time": float(attrs["time"])}
                )
        else:
            self._cache = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            entry = self._cache[idx]
        else:
            sdf, coords, attrs = load_sdf_from_hdf5(str(self.files[idx]))
            entry = {"sdf": sdf, "coords": coords, "time": float(attrs["time"])}

        points, sdf_vals = sample_points_from_sdf(
            sdf=entry["sdf"],
            coords=entry["coords"],
            time=entry["time"],
            n_samples=self.samples_per_frame,
            near_surface_ratio=self.near_surface_ratio,
            surface_band=self.surface_band,
        )
        return torch.from_numpy(points), torch.from_numpy(sdf_vals)


def collate_concat(batch):
    """Concatenate per-frame samples into a single (N, 4) / (N, 1) batch."""
    coords_list, sdf_list = zip(*batch)
    coords = torch.cat(coords_list, dim=0)
    sdf = torch.cat(sdf_list, dim=0)
    return coords, sdf
