#!/usr/bin/env python3
"""Generate the Project 02 finite-difference acoustic reference field.

This script is intentionally limited to the reference side of the rebuilt
contract. It does not train a network and does not write validation/model
outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "outputs" / "project02_reference"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def default_font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def diverging_rgb(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if vmax is None:
        vmax = float(np.percentile(np.abs(data), 99.7))
    vmax = max(vmax, 1.0e-12)
    n = np.clip(data / vmax, -1.0, 1.0)
    rgb = np.ones((*data.shape, 3), dtype=np.float64)
    pos = n >= 0.0
    rgb[pos, 1] = 1.0 - n[pos]
    rgb[pos, 2] = 1.0 - n[pos]
    neg = ~pos
    rgb[neg, 0] = 1.0 + n[neg]
    rgb[neg, 1] = 1.0 + n[neg]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def sequential_rgb(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))
    span = max(float(vmax - vmin), 1.0e-12)
    n = np.clip((data - vmin) / span, 0.0, 1.0)
    stops = np.array(
        [
            [27, 42, 82],
            [28, 120, 132],
            [68, 170, 116],
            [236, 215, 93],
        ],
        dtype=np.float64,
    )
    scaled = n * (len(stops) - 1)
    left = np.floor(scaled).astype(np.int64)
    right = np.clip(left + 1, 0, len(stops) - 1)
    frac = scaled - left
    rgb = stops[left] * (1.0 - frac[..., None]) + stops[right] * frac[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def resized_image(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize(size, resampling)


def save_panel_grid(
    panels: list[Image.Image],
    labels: list[str],
    title: str,
    path: Path,
    columns: int = 3,
) -> None:
    font = default_font(14)
    title_font = default_font(21)
    label_h = 28
    margin = 16
    gap = 18
    rows = int(math.ceil(len(panels) / columns))
    panel_w = max(p.width for p in panels)
    panel_h = max(p.height for p in panels)
    width = margin * 2 + columns * panel_w + (columns - 1) * gap
    height = margin * 2 + 42 + rows * (panel_h + label_h) + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    tw, th = text_size(draw, title, title_font)
    draw.text(((width - tw) // 2, margin), title, fill=(15, 15, 15), font=title_font)

    y0 = margin + 42
    for idx, panel in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = margin + col * (panel_w + gap)
        y = y0 + row * (panel_h + label_h + gap)
        canvas.paste(panel, (x, y + label_h))
        lw, lh = text_size(draw, labels[idx], font)
        draw.text((x + (panel_w - lw) // 2, y + (label_h - lh) // 2), labels[idx], fill=(25, 25, 25), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_snapshots(wavefield: np.ndarray, times: np.ndarray, path: Path, title: str = "Project 02 FD Reference") -> None:
    indices = np.linspace(0, wavefield.shape[0] - 1, 9).round().astype(int)
    vmax = float(np.percentile(np.abs(wavefield), 99.7))
    panels = [resized_image(diverging_rgb(wavefield[i], vmax=vmax), (232, 232)) for i in indices]
    labels = [f"frame {i} | t={times[i]:.3f}" for i in indices]
    save_panel_grid(panels, labels, f"{title} | vmax={vmax:.4f}", path, columns=3)


def save_slice_xt(wavefield: np.ndarray, times: np.ndarray, y_index: int, path: Path) -> None:
    del times
    rgb = diverging_rgb(wavefield[:, y_index, :])
    image = resized_image(rgb, (960, 420))
    title_h = 42
    canvas = Image.new("RGB", (image.width, image.height + title_h), "white")
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    font = default_font(18)
    title = f"Project 02 FD Reference | x-t slice at y-index {y_index}"
    tw, th = text_size(draw, title, font)
    draw.text(((canvas.width - tw) // 2, (title_h - th) // 2), title, fill=(15, 15, 15), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_medium(medium: np.ndarray, path: Path) -> None:
    image = resized_image(sequential_rgb(medium), (620, 620))
    title_h = 44
    canvas = Image.new("RGB", (image.width, image.height + title_h), "white")
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    font = default_font(20)
    title = f"Project 02 Smooth Ocean Medium | c in [{medium.min():.3f}, {medium.max():.3f}]"
    tw, th = text_size(draw, title, font)
    draw.text(((canvas.width - tw) // 2, (title_h - th) // 2), title, fill=(15, 15, 15), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_ocean_medium(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    thermocline = 0.12 * np.tanh((yy - 0.55) / 0.08)
    lateral_lens = 0.05 * np.sin(2.0 * np.pi * (1.2 * xx + 0.15)) * np.exp(-((yy - 0.55) / 0.35) ** 2)
    medium = 1.05 + 0.45 * yy + thermocline + lateral_lens
    return x.astype(np.float32), y.astype(np.float32), np.clip(medium, 0.95, 1.75).astype(np.float32)


def make_sponge(nx: int, ny: int, width: int, strength: float) -> np.ndarray:
    ix = np.minimum(np.arange(nx), np.arange(nx)[::-1])
    iy = np.minimum(np.arange(ny), np.arange(ny)[::-1])
    dist_x, dist_y = np.meshgrid(ix, iy)
    dist = np.minimum(dist_x, dist_y)
    profile = np.clip((width - dist) / width, 0.0, 1.0)
    return (strength * profile * profile).astype(np.float32)


def ricker(t: float | np.ndarray, f0: float, t0: float) -> np.ndarray:
    tau = np.asarray(t, dtype=np.float64) - t0
    arg = (np.pi * f0 * tau) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def laplacian_edge(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    padded = np.pad(u, 1, mode="edge")
    return (
        (padded[1:-1, 2:] - 2.0 * u + padded[1:-1, :-2]) / (dx * dx)
        + (padded[2:, 1:-1] - 2.0 * u + padded[:-2, 1:-1]) / (dy * dy)
    )


def simulate_reference(
    nx: int,
    ny: int,
    n_frames: int,
    t_max: float,
    source_amplitude: float,
    sponge_width: int,
    sponge_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x, y, medium = make_ocean_medium(nx, ny)
    xx, yy = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    c_max = float(np.max(medium))
    dt_stable = 0.42 * min(dx, dy) / (math.sqrt(2.0) * c_max)
    n_steps = int(math.ceil(t_max / dt_stable))
    dt = t_max / n_steps
    cfl = c_max * dt * math.sqrt(2.0) / min(dx, dy)

    source_position = (0.18, 0.42)
    source_sigma = 0.018
    source_frequency = 9.0
    source_t0 = 0.12

    source_spatial = np.exp(
        -((xx - source_position[0]) ** 2 + (yy - source_position[1]) ** 2) / (2.0 * source_sigma * source_sigma)
    )
    source_spatial /= float(np.sum(source_spatial) * dx * dy)
    source_spatial = source_spatial.astype(np.float64)
    damping = make_sponge(nx, ny, width=sponge_width, strength=sponge_strength).astype(np.float64)

    u = np.zeros((ny, nx), dtype=np.float64)
    velocity = np.zeros_like(u)
    save_steps = np.linspace(0, n_steps, n_frames).round().astype(int)
    save_set = set(int(s) for s in save_steps)
    frames: list[np.ndarray] = []
    times: list[float] = []
    residual_sum = 0.0
    residual_scale_sum = 0.0
    residual_count = 0

    medium_sq = medium.astype(np.float64) ** 2
    for step in range(n_steps + 1):
        t = step * dt
        if step in save_set:
            frames.append(u.astype(np.float32).copy())
            times.append(t)

        lap = laplacian_edge(u, dx, dy)
        force = source_amplitude * float(ricker(t, source_frequency, source_t0)) * source_spatial
        rhs = medium_sq * lap + force - damping * velocity
        velocity_next = velocity + dt * rhs

        update_residual = (velocity_next - velocity) / dt - rhs
        held_out = update_residual[24:-24:7, 24:-24:7]
        held_out_scale = rhs[24:-24:7, 24:-24:7]
        residual_sum += float(np.sum(held_out * held_out))
        residual_scale_sum += float(np.sum(held_out_scale * held_out_scale))
        residual_count += int(held_out.size)

        velocity = velocity_next
        u += dt * velocity

    wavefield = np.stack(frames, axis=0).astype(np.float32)
    times_array = np.asarray(times, dtype=np.float32)
    frame_energies = np.mean(wavefield * wavefield, axis=(1, 2))
    stats = {
        "max_abs": float(np.max(np.abs(wavefield))),
        "temporal_motion_variance": float(np.mean(np.diff(wavefield, axis=0) ** 2)),
        "energy": float(np.mean(wavefield * wavefield)),
        "substantive_frame_count": int(np.sum(frame_energies > 1.0e-5)),
        "frame_energy_first": float(frame_energies[0]),
        "frame_energy_middle": float(frame_energies[len(frame_energies) // 2]),
        "frame_energy_last": float(frame_energies[-1]),
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "representation_family": "Family A continuous field, direct fixed-grid sampling",
        "grid": {"nx": nx, "ny": ny, "n_frames": int(wavefield.shape[0]), "vertex_count": nx * ny},
        "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "t_min": 0.0, "t_max": t_max},
        "finite_difference": {
            "integrator": "velocity-form second-order finite difference",
            "dx": dx,
            "dy": dy,
            "dt_internal": dt,
            "dt_stable_target": dt_stable,
            "cfl": cfl,
            "n_internal_steps": n_steps,
            "boundary": "sponge absorbing layer with edge-padded Laplacian",
            "sponge_width_cells": sponge_width,
            "sponge_strength": sponge_strength,
            "heldout_internal_velocity_form_rms": math.sqrt(residual_sum / max(residual_count, 1)),
            "heldout_internal_velocity_form_normalized_rms": math.sqrt(residual_sum / max(residual_scale_sum, 1.0e-30)),
            "heldout_internal_stride": 7,
            "heldout_internal_boundary_crop": 24,
        },
        "medium": {
            "type": "smooth ocean-like layered thermocline with weak lateral lens",
            "min": float(np.min(medium)),
            "max": float(np.max(medium)),
            "mean": float(np.mean(medium)),
        },
        "source": {
            "type": "Ricker",
            "frequency": source_frequency,
            "t0": source_t0,
            "position": list(source_position),
            "sigma": source_sigma,
            "amplitude": source_amplitude,
            "dominant_wavelength_at_c_min": float(float(np.min(medium)) / source_frequency),
            "dominant_wavelength_grid_cells_at_c_min": float((float(np.min(medium)) / source_frequency) / min(dx, dy)),
        },
        "nontriviality": stats,
    }
    return wavefield, medium, times_array, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=192)
    parser.add_argument("--ny", type=int, default=192)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--t-max", type=float, default=0.65)
    parser.add_argument("--source-amplitude", type=float, default=5.0)
    parser.add_argument("--sponge-width", type=int, default=22)
    parser.add_argument("--sponge-strength", type=float, default=12.0)
    args = parser.parse_args()

    wavefield, medium, times, metadata = simulate_reference(
        nx=args.nx,
        ny=args.ny,
        n_frames=args.frames,
        t_max=args.t_max,
        source_amplitude=args.source_amplitude,
        sponge_width=args.sponge_width,
        sponge_strength=args.sponge_strength,
    )
    stats = metadata["nontriviality"]
    if stats["max_abs"] <= 0.01:
        raise SystemExit(f"reference field is essentially zero: max_abs={stats['max_abs']}")
    if stats["temporal_motion_variance"] <= 1.0e-4:
        raise SystemExit(f"reference has negligible temporal motion: {stats['temporal_motion_variance']}")
    if stats["substantive_frame_count"] < 8:
        raise SystemExit(f"fewer than 8 substantive frames: {stats['substantive_frame_count']}")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REFERENCE_DIR / "reference_wavefield.npy", wavefield)
    np.save(REFERENCE_DIR / "reference_medium.npy", medium)
    np.save(REFERENCE_DIR / "times.npy", times)
    write_json(REFERENCE_DIR / "reference_metadata.json", metadata)
    save_snapshots(wavefield, times, REFERENCE_DIR / "reference_snapshots.png")
    source_y = int(round(metadata["source"]["position"][1] * (args.ny - 1)))
    save_slice_xt(wavefield, times, source_y, REFERENCE_DIR / "reference_slice_xt.png")
    save_medium(medium, REFERENCE_DIR / "reference_medium.png")

    print(json.dumps({"status": "ok", "nontriviality": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
