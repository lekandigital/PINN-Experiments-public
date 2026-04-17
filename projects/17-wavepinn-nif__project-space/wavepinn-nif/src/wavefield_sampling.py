#!/usr/bin/env python3
"""
Wavefield sampling helpers for visualization.

This module reuses the existing PyTorch checkpoint loader and produces
precomputed frame bundles that are easy to consume from Taichi GGUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from export_blender import load_wavepinn_checkpoint, sample_wavefield_sequence


DEFAULT_CHECKPOINT_CANDIDATES = [
    "outputs/supervised_fd_long/checkpoints/wavefield_nif_supervised.pt",
    "outputs/supervised_fd/checkpoints/wavefield_nif_supervised.pt",
    "outputs/checkpoints/wavepinn_train_v2.pt",
    "artifacts/supervised_fd_long_bundle/checkpoints/wavefield_nif_supervised.pt",
    "artifacts/supervised_fd_bundle/checkpoints/wavefield_nif_supervised.pt",
    "artifacts/train_v2_bundle/checkpoints/wavepinn_train_v2.pt",
    "artifacts/remote_existing/checkpoints/wavepinn_trained.pt",
    "wavepinn_trained.pt",
]


@dataclass
class WavefieldFrameBundle:
    checkpoint_path: Path
    torch_device: str
    resolution: int
    n_frames: int
    t_start: float
    t_end: float
    times: np.ndarray
    raw_fields: np.ndarray
    centered_fields: np.ndarray
    normalized_fields: np.ndarray
    display_fields: np.ndarray
    base_positions: np.ndarray
    triangle_indices: np.ndarray
    line_indices: np.ndarray
    extent: float
    height_scale: float
    amplitude_reference: float
    scale_factor: float
    color_reference: float
    display_transfer: str
    display_gain: float
    center_mode: str
    spatial_smooth_passes: int
    temporal_smooth_passes: int
    model_config: Optional[dict]
    training_args: Optional[dict]
    final_metrics: Optional[dict]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.resolve()

    missing = []
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        path = project_root() / candidate
        if path.exists():
            return path.resolve()
        missing.append(str(path))

    raise FileNotFoundError(
        "No checkpoint was found. Pass --checkpoint explicitly or create one in one of:\n"
        + "\n".join(missing)
    )


def infer_sampling_defaults(checkpoint: dict) -> tuple[float, float, int]:
    training_args = checkpoint.get("training_args") or {}
    script_name = checkpoint.get("script", "")

    n_frames = int(training_args.get("frames", 96))
    t_end = float(training_args.get("total_time", 0.55 if "supervised_fd" in script_name else 0.50))
    t_start = 0.0 if "supervised_fd" in script_name else 0.05
    return t_start, t_end, n_frames


def center_fields(fields: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return fields.copy()
    if mode == "global":
        return fields - fields.mean()
    if mode == "per_frame":
        frame_means = fields.mean(axis=(1, 2), keepdims=True)
        return fields - frame_means
    raise ValueError(f"Unknown center mode: {mode}")


def smooth_fields(fields: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0:
        return fields

    smoothed = fields.copy()
    for _ in range(passes):
        padded = np.pad(smoothed, ((0, 0), (1, 1), (1, 1)), mode="edge")
        smoothed = (
            padded[:, 1:-1, 1:-1] * 4.0
            + padded[:, :-2, 1:-1]
            + padded[:, 2:, 1:-1]
            + padded[:, 1:-1, :-2]
            + padded[:, 1:-1, 2:]
        ) / 8.0
    return smoothed


def smooth_fields_temporal(fields: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0:
        return fields

    smoothed = fields.copy()
    for _ in range(passes):
        padded = np.pad(smoothed, ((1, 1), (0, 0), (0, 0)), mode="edge")
        smoothed = (padded[1:-1] * 4.0 + padded[:-2] + padded[2:]) / 6.0
    return smoothed


def build_heightfield_geometry(resolution: int, extent: float = 2.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-extent * 0.5, extent * 0.5, resolution, dtype=np.float32)
    xx, zz = np.meshgrid(coords, coords, indexing="ij")
    base_positions = np.stack(
        [xx.reshape(-1), np.zeros(resolution * resolution, dtype=np.float32), zz.reshape(-1)],
        axis=-1,
    )

    triangles = np.zeros((resolution - 1) * (resolution - 1) * 2 * 3, dtype=np.int32)
    lines = np.zeros(((resolution - 1) * resolution * 2) * 2, dtype=np.int32)

    tri_cursor = 0
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = a + 1
            c = a + resolution
            d = c + 1
            triangles[tri_cursor:tri_cursor + 6] = np.array([a, c, b, b, c, d], dtype=np.int32)
            tri_cursor += 6

    line_cursor = 0
    for i in range(resolution):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = a + 1
            lines[line_cursor:line_cursor + 2] = np.array([a, b], dtype=np.int32)
            line_cursor += 2
    for i in range(resolution - 1):
        for j in range(resolution):
            a = i * resolution + j
            b = a + resolution
            lines[line_cursor:line_cursor + 2] = np.array([a, b], dtype=np.int32)
            line_cursor += 2

    return base_positions, triangles, lines


def amplitude_color_field(
    frame: np.ndarray,
    color_reference: float,
    neutral: tuple[float, float, float] = (0.80, 0.84, 0.88),
    warm: tuple[float, float, float] = (0.95, 0.58, 0.40),
    cool: tuple[float, float, float] = (0.27, 0.49, 0.80),
    strength_power: float = 0.8,
) -> np.ndarray:
    signed = np.clip(frame / max(color_reference, 1e-6), -1.0, 1.0)
    strength = np.abs(signed) ** strength_power

    neutral = np.array(neutral, dtype=np.float32)
    warm = np.array(warm, dtype=np.float32)
    cool = np.array(cool, dtype=np.float32)

    target = np.where(signed[..., None] >= 0.0, warm, cool)
    colors = neutral + (target - neutral) * strength[..., None]
    return np.clip(colors.astype(np.float32), 0.0, 1.0)


def sample_wavefield_bundle(
    checkpoint_path: str | Path | None = None,
    resolution: Optional[int] = None,
    n_frames: Optional[int] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    height_scale: float = 0.10,
    amplitude_percentile: float = 99.5,
    center_mode: str = "global",
    spatial_smooth_passes: int = 2,
    temporal_smooth_passes: int = 1,
    display_transfer: str = "tanh",
    display_gain: float = 1.25,
    extent: float = 2.1,
    torch_device: str | None = None,
) -> WavefieldFrameBundle:
    checkpoint_file = resolve_checkpoint_path(checkpoint_path)
    model, checkpoint, device = load_wavepinn_checkpoint(str(checkpoint_file), device=torch_device)

    default_t_start, default_t_end, default_frames = infer_sampling_defaults(checkpoint)
    t_start = default_t_start if t_start is None else float(t_start)
    t_end = default_t_end if t_end is None else float(t_end)
    n_frames = default_frames if n_frames is None else int(n_frames)
    resolution = 160 if resolution is None else int(resolution)

    raw_fields, times = sample_wavefield_sequence(
        model=model,
        resolution=resolution,
        n_frames=n_frames,
        t_start=t_start,
        t_end=t_end,
        device=device,
    )

    centered_fields = center_fields(raw_fields, center_mode)
    centered_fields = smooth_fields(centered_fields, spatial_smooth_passes)
    centered_fields = smooth_fields_temporal(centered_fields, temporal_smooth_passes)

    amplitude_reference = float(np.percentile(np.abs(centered_fields), amplitude_percentile))
    amplitude_reference = max(amplitude_reference, 1e-6)
    scale_factor = float(height_scale / amplitude_reference)
    normalized_fields = centered_fields / amplitude_reference

    if display_transfer == "linear":
        display_fields = normalized_fields * height_scale
        color_reference = float(np.percentile(np.abs(display_fields), amplitude_percentile))
        color_reference = max(color_reference, 1e-6)
    elif display_transfer == "tanh":
        display_fields = np.tanh(normalized_fields * display_gain) * height_scale
        color_reference = float(height_scale)
    else:
        raise ValueError(f"Unknown display transfer: {display_transfer}")

    base_positions, triangle_indices, line_indices = build_heightfield_geometry(resolution, extent=extent)

    return WavefieldFrameBundle(
        checkpoint_path=checkpoint_file,
        torch_device=device,
        resolution=resolution,
        n_frames=n_frames,
        t_start=t_start,
        t_end=t_end,
        times=times.astype(np.float32),
        raw_fields=raw_fields.astype(np.float32),
        centered_fields=centered_fields.astype(np.float32),
        normalized_fields=normalized_fields.astype(np.float32),
        display_fields=display_fields.astype(np.float32),
        base_positions=base_positions,
        triangle_indices=triangle_indices,
        line_indices=line_indices,
        extent=float(extent),
        height_scale=float(height_scale),
        amplitude_reference=amplitude_reference,
        scale_factor=scale_factor,
        color_reference=color_reference,
        display_transfer=str(display_transfer),
        display_gain=float(display_gain),
        center_mode=center_mode,
        spatial_smooth_passes=int(spatial_smooth_passes),
        temporal_smooth_passes=int(temporal_smooth_passes),
        model_config=checkpoint.get("model_config"),
        training_args=checkpoint.get("training_args"),
        final_metrics=checkpoint.get("final_metrics"),
    )
