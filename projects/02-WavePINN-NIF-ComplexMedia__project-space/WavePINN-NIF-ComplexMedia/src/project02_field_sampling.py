"""Field loading and fixed-grid sampling helpers for the Project 02 viewer."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STUDIO = {
    "background": (0.02, 0.026, 0.04),
    "ground_color": (0.09, 0.10, 0.13),
    "ambient": (0.22, 0.23, 0.25),
    "lights": [
        {"pos": (2.2, 2.5, 1.2), "color": (1.10, 1.00, 0.92)},
        {"pos": (-1.8, 1.0, -1.3), "color": (0.20, 0.32, 0.52)},
        {"pos": (0.0, 0.55, 2.4), "color": (0.09, 0.10, 0.12)},
    ],
    "camera_fov": 24.0,
    "display_transfer": "tanh",
    "display_gain": 1.3,
    "playback_speed": 0.72,
    "palette": {
        "neutral": (0.76, 0.80, 0.86),
        "positive": (0.99, 0.70, 0.40),
        "negative": (0.18, 0.49, 0.88),
        "strength_power": 0.72,
    },
    "spatial_smooth_passes": 3,
    "temporal_smooth_passes": 1,
}


PRESETS = {
    "research": {
        "background": (0.055, 0.06, 0.072),
        "ground_color": (0.11, 0.12, 0.14),
        "ambient": (0.34, 0.35, 0.36),
        "camera": "diagnostic",
        "camera_fov": 31.0,
        "display_gain": 1.0,
        "height_scale": 0.35,
        "spatial_smooth_passes": 1,
        "temporal_smooth_passes": 0,
        "show_wire": True,
        "palette": {
            "neutral": (0.72, 0.76, 0.82),
            "positive": (0.96, 0.56, 0.32),
            "negative": (0.22, 0.48, 0.82),
            "strength_power": 0.86,
        },
    },
    "pitch": {
        "studio": True,
        "camera": "hero",
        "height_scale": 0.45,
        "medium_blend": 0.54,
        "source_marker_radius": 0.0075,
        "show_wire": False,
    },
    "dramatic": {
        "studio": True,
        "camera": "tight",
        "camera_fov": 21.0,
        "display_gain": 1.75,
        "height_scale": 0.65,
        "medium_blend": 0.46,
        "source_marker_radius": 0.0085,
        "show_wire": False,
    },
}


CAMERAS = {
    "hero": {
        "position": (0.34, 0.55, 1.42),
        "lookat": (0.01, 0.015, 0.02),
        "up": (0.0, 1.0, 0.0),
    },
    "diagnostic": {
        "position": (0.20, 0.92, 1.62),
        "lookat": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
    },
    "tight": {
        "position": (0.12, 0.44, 1.02),
        "lookat": (-0.08, 0.02, 0.02),
        "up": (0.0, 1.0, 0.0),
    },
    "camera_1": {
        "position": (0.34, 0.55, 1.42),
        "lookat": (0.01, 0.015, 0.02),
        "up": (0.0, 1.0, 0.0),
    },
    "camera_2": {
        "position": (-1.10, 0.58, 0.84),
        "lookat": (-0.04, 0.015, 0.02),
        "up": (0.0, 1.0, 0.0),
    },
    "camera_3": {
        "position": (0.02, 1.10, 0.42),
        "lookat": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
    },
    "compare": {
        "position": (0.0, 0.86, 2.05),
        "lookat": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
    },
}


@dataclass(frozen=True)
class Project02Fields:
    reference_wavefield: np.ndarray
    learned_wavefield: np.ndarray
    reference_medium: np.ndarray
    learned_medium: np.ndarray
    times: np.ndarray
    metrics: dict[str, Any]
    reference_metadata: dict[str, Any]
    checkpoint_path: Path
    checkpoint_format: str
    checkpoint_model_type: str
    medium_supervision_status: str

    @property
    def n_frames(self) -> int:
        return int(self.learned_wavefield.shape[0])

    @property
    def ny(self) -> int:
        return int(self.learned_wavefield.shape[1])

    @property
    def nx(self) -> int:
        return int(self.learned_wavefield.shape[2])

    @property
    def vertices_per_surface(self) -> int:
        return self.nx * self.ny

    @property
    def faces_per_surface(self) -> int:
        return 2 * (self.nx - 1) * (self.ny - 1)


def load_project02_fields(project_root: Path = PROJECT_ROOT) -> Project02Fields:
    reference_dir = project_root / "outputs" / "project02_reference"
    learned_dir = project_root / "outputs" / "project02_validation_model"
    checkpoint_path = project_root / "checkpoints" / "project02_wavepinn_nif_best.pkl"

    reference_wavefield = np.load(reference_dir / "reference_wavefield.npy").astype(np.float32)
    learned_wavefield = np.load(learned_dir / "wavefield_sequence.npy").astype(np.float32)
    reference_medium = np.load(reference_dir / "reference_medium.npy").astype(np.float32)
    learned_medium = np.load(learned_dir / "medium_pred.npy").astype(np.float32)
    times = np.load(learned_dir / "times.npy").astype(np.float32)
    with (learned_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with (reference_dir / "reference_metadata.json").open("r", encoding="utf-8") as f:
        reference_metadata = json.load(f)
    checkpoint = {}
    checkpoint_format = "pickle"
    checkpoint_model_type = "JAX/Haiku WavePINN-NIF"
    medium_supervision_status = "directly supervised against reference_medium.npy with real MediaNIF network parameters"
    try:
        with checkpoint_path.open("rb") as f:
            checkpoint = pickle.load(f)
        checkpoint_format = str(checkpoint.get("format", "pickle"))
        checkpoint_model_type = str(checkpoint.get("framework", checkpoint.get("model_type", checkpoint_model_type)))
        if "parameter_count" in checkpoint:
            checkpoint_model_type = f"{checkpoint_model_type} ({int(checkpoint['parameter_count']):,} params)"
    except ModuleNotFoundError:
        # Viewer/rendering does not need optimizer classes from the training venv.
        # The checkpoint remains loadable in the project JAX/Haiku environment.
        pass

    if reference_wavefield.shape != learned_wavefield.shape:
        raise ValueError(f"Reference/learned shape mismatch: {reference_wavefield.shape} vs {learned_wavefield.shape}")
    if reference_medium.shape != learned_medium.shape:
        raise ValueError(f"Medium shape mismatch: {reference_medium.shape} vs {learned_medium.shape}")
    if reference_medium.shape != reference_wavefield.shape[1:]:
        raise ValueError(f"Medium/wave grid mismatch: {reference_medium.shape} vs {reference_wavefield.shape[1:]}")
    if reference_wavefield.shape[0] != times.shape[0]:
        raise ValueError(f"Time/frame mismatch: {times.shape[0]} vs {reference_wavefield.shape[0]}")

    return Project02Fields(
        reference_wavefield=reference_wavefield,
        learned_wavefield=learned_wavefield,
        reference_medium=reference_medium,
        learned_medium=learned_medium,
        times=times,
        metrics=metrics,
        reference_metadata=reference_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_format=checkpoint_format,
        checkpoint_model_type=checkpoint_model_type,
        medium_supervision_status=medium_supervision_status,
    )


def resolved_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    preset = dict(PRESETS[name])
    if preset.get("studio"):
        merged = dict(STUDIO)
        merged.update({k: v for k, v in preset.items() if k != "studio"})
        return merged
    return preset


def smooth_spatial(frame: np.ndarray, passes: int) -> np.ndarray:
    out = frame.astype(np.float32, copy=True)
    for _ in range(max(0, passes)):
        padded = np.pad(out, 1, mode="edge")
        out = (
            4.0 * padded[1:-1, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
        ) / 8.0
    return out.astype(np.float32)


def smooth_temporal(sequence: np.ndarray, passes: int) -> np.ndarray:
    out = sequence.astype(np.float32, copy=True)
    for _ in range(max(0, passes)):
        padded = np.pad(out, ((1, 1), (0, 0), (0, 0)), mode="edge")
        out = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    return out.astype(np.float32)


def prepared_wavefield(sequence: np.ndarray, preset: dict[str, Any]) -> np.ndarray:
    out = smooth_temporal(sequence, int(preset.get("temporal_smooth_passes", 0)))
    spatial_passes = int(preset.get("spatial_smooth_passes", 0))
    if spatial_passes:
        out = np.stack([smooth_spatial(frame, spatial_passes) for frame in out], axis=0)
    return out.astype(np.float32)


def build_grid_positions(ny: int, nx: int, x_offset: float = 0.0, x_scale: float = 1.0, z_scale: float = 1.0) -> np.ndarray:
    xs = (np.linspace(-0.5, 0.5, nx, dtype=np.float32) * x_scale) + x_offset
    zs = np.linspace(-0.5, 0.5, ny, dtype=np.float32) * z_scale
    xx, zz = np.meshgrid(xs, zs)
    positions = np.zeros((ny, nx, 3), dtype=np.float32)
    positions[..., 0] = xx
    positions[..., 2] = zz
    return positions.reshape(-1, 3)


def build_triangle_indices(ny: int, nx: int) -> np.ndarray:
    indices: list[int] = []
    for iy in range(ny - 1):
        row = iy * nx
        next_row = (iy + 1) * nx
        for ix in range(nx - 1):
            a = row + ix
            b = row + ix + 1
            c = next_row + ix
            d = next_row + ix + 1
            indices.extend([a, c, b, b, c, d])
    return np.asarray(indices, dtype=np.int32)


def source_ring_positions(metadata: dict[str, Any], x_offset: float = 0.0, x_scale: float = 1.0) -> np.ndarray:
    sx, sy = metadata["source"]["position"]
    center_x = (float(sx) - 0.5) * x_scale + x_offset
    center_z = float(sy) - 0.5
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    radius = 0.038 * x_scale
    ring = np.zeros((theta.size, 3), dtype=np.float32)
    ring[:, 0] = center_x + radius * np.cos(theta)
    ring[:, 1] = -0.045
    ring[:, 2] = center_z + radius * np.sin(theta)
    return ring


def wave_vertices(base_positions: np.ndarray, frame: np.ndarray, height_scale: float) -> np.ndarray:
    vertices = base_positions.copy()
    vertices[:, 1] = frame.reshape(-1).astype(np.float32) * float(height_scale)
    return vertices.astype(np.float32)


def medium_vertices(base_positions: np.ndarray, y: float = -0.078) -> np.ndarray:
    vertices = base_positions.copy()
    vertices[:, 1] = float(y)
    return vertices.astype(np.float32)


def wave_colors(frame: np.ndarray, amplitude_scale: float, preset: dict[str, Any]) -> np.ndarray:
    palette = preset["palette"]
    neutral = np.asarray(palette["neutral"], dtype=np.float32)
    positive = np.asarray(palette["positive"], dtype=np.float32)
    negative = np.asarray(palette["negative"], dtype=np.float32)
    power = float(palette["strength_power"])
    gain = float(preset.get("display_gain", 1.0))
    scaled = np.tanh(gain * frame.astype(np.float32) / max(amplitude_scale, 1.0e-12))
    strength = np.abs(scaled) ** power
    target = np.where(scaled[..., None] >= 0.0, positive, negative)
    colors = neutral * (1.0 - strength[..., None]) + target * strength[..., None]
    return np.clip(colors.reshape(-1, 3), 0.0, 1.0).astype(np.float32)


def medium_colors(medium: np.ndarray, preset: dict[str, Any]) -> np.ndarray:
    low = np.asarray((0.08, 0.16, 0.22), dtype=np.float32)
    mid = np.asarray((0.12, 0.34, 0.36), dtype=np.float32)
    high = np.asarray((0.46, 0.60, 0.38), dtype=np.float32)
    ground = np.asarray(preset.get("ground_color", STUDIO["ground_color"]), dtype=np.float32)
    blend = float(preset.get("medium_blend", 0.50))
    norm = (medium.astype(np.float32) - float(np.min(medium))) / max(float(np.max(medium) - np.min(medium)), 1.0e-12)
    colors = np.empty((*medium.shape, 3), dtype=np.float32)
    lower = norm < 0.55
    t_low = np.clip(norm / 0.55, 0.0, 1.0)
    t_high = np.clip((norm - 0.55) / 0.45, 0.0, 1.0)
    colors[lower] = low * (1.0 - t_low[lower, None]) + mid * t_low[lower, None]
    colors[~lower] = mid * (1.0 - t_high[~lower, None]) + high * t_high[~lower, None]
    colors = ground * (1.0 - blend) + colors * blend
    return np.clip(colors.reshape(-1, 3), 0.0, 1.0).astype(np.float32)


def global_amplitude_scale(reference: np.ndarray, learned: np.ndarray) -> float:
    both = np.concatenate([np.abs(reference).reshape(-1), np.abs(learned).reshape(-1)])
    return float(max(np.percentile(both, 99.5), 1.0e-9))
