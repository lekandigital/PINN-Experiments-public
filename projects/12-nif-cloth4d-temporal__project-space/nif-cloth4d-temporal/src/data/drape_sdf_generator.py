"""
Analytic drape-over-sphere SDF generator for Project 12.

Replaces the buggy `SyntheticClothDataset` sign heuristic in pyflex_simulator.py
with a true analytic surface SDF. The scene is a cloth height field h(x, y, t)
that relaxes from flat (t=0) toward draping over a sphere (t=1), with
sinusoidal wrinkles whose amplitude ramps with t.

The SDF is the **signed perpendicular distance** to the cloth surface z = h(x,y,t):

    sdf(x, y, z, t) = (z - h(x, y, t)) / sqrt(1 + h_x^2 + h_y^2)

This is a first-order-correct signed distance that has ||∇sdf|| ≈ 1 exactly on
the surface (and very close to 1 near it), which is the structural invariant
Family A (SDF) supervision relies on. The earlier form ``|z - h| - thickness``
had ||∇|| = sqrt(1 + |∇h|^2) — at the sphere rim that reaches 3+, which made
the analytic reference itself eikonal-inconsistent and inflated NMSE. Cloth
thickness is retained as metadata only; the cloth is modeled as the zero
level set of the field.

The HDF5 serialization format matches Project 13's synthetic_data.py contract
(``sdf``, ``coords``, attrs ``time``, ``grid_size``, ``bounds_min``,
``bounds_max``) so the Dataset can consume frames without cross-project
imports at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


# Scene parameters. "Drape-over-form" where the form is a smooth Gaussian bump.
# The earlier sphere-cap form had two fatal representational defects for an
# eikonal-regularized SDF: (1) a height discontinuity at the footprint rim
# (inside h=0 at r=R, outside h=R — a 0.3 jump), and (2) vertical slope at
# the rim which sent ||∇h||→∞. Both prevented the model from fitting late-t
# frames cleanly and were the dominant source of heightfield Hausdorff /
# NMSE failure under the true normalized SDF. A Gaussian bump is the minimal
# change that preserves the "cloth draped over a rounded form" visual and
# removes both singularities.
BUMP_HEIGHT = 0.30              # peak height of the drape form (≈ old sphere R)
BUMP_SIGMA = 0.22               # Gaussian width (≈ old sphere footprint)
SPHERE_RADIUS = BUMP_HEIGHT     # kept as legacy alias for metadata consumers
SPHERE_CENTER = (0.0, 0.0, 0.0)
CLOTH_THICKNESS = 0.01
FLAT_HEIGHT = 0.55
# Wrinkles were previously K = 6π rad/unit. With fourier_scale=1 (required
# for training stability per the D4 deviation), that frequency sits well
# outside the encoder's representable band and becomes pure unfittable noise.
# Set to zero for the hero scene; reintroduce as an ablation once a higher-
# scale-compatible training config is established.
WRINKLE_K = 6.0 * np.pi
WRINKLE_MAX_AMP = 0.0


def _sphere_cap_and_grad(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian-bump drape form and its (d/dx, d/dy) gradients.

    Smooth C^∞ alternative to the sphere cap. Peak is ``BUMP_HEIGHT`` at
    the scene center, falling off on scale ``BUMP_SIGMA``. Gradients are
    analytically bounded (max |∇| ≈ BUMP_HEIGHT / (BUMP_SIGMA * sqrt(e)) ≈
    0.83 for the defaults here) so the normalized SDF
    (z - h) / sqrt(1 + h_x² + h_y²) stays eikonal-consistent everywhere.

    Name retained for call-site compatibility with the earlier sphere-cap
    implementation; the geometry is a rounded bump rather than a hemisphere.
    """
    cx, cy, _ = SPHERE_CENTER
    dx = (x - cx).astype(np.float32)
    dy = (y - cy).astype(np.float32)
    r2 = dx * dx + dy * dy
    inv_s2 = np.float32(1.0 / (BUMP_SIGMA ** 2))
    val = (BUMP_HEIGHT * np.exp(-0.5 * r2 * inv_s2)).astype(np.float32)
    dtdx = (-dx * inv_s2 * val).astype(np.float32)
    dtdy = (-dy * inv_s2 * val).astype(np.float32)
    return val, dtdx, dtdy


def _draped_height_and_grad(
    x: np.ndarray, y: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draped cloth = drape form + clearance + (optional) wrinkles that grow
    with t. Returns (h_draped, dh/dx, dh/dy).
    """
    clearance = CLOTH_THICKNESS * 2.0
    base, dbx, dby = _sphere_cap_and_grad(x, y)
    base = base + clearance
    amp = WRINKLE_MAX_AMP * float(t)
    if amp == 0.0:
        return (base.astype(np.float32),
                dbx.astype(np.float32),
                dby.astype(np.float32))
    sx = np.sin(WRINKLE_K * x)
    sy = np.sin(WRINKLE_K * y)
    cx = np.cos(WRINKLE_K * x)
    cy = np.cos(WRINKLE_K * y)
    w = amp * sx * sy
    dwx = amp * WRINKLE_K * cx * sy
    dwy = amp * WRINKLE_K * sx * cy
    return (
        (base + w).astype(np.float32),
        (dbx + dwx).astype(np.float32),
        (dby + dwy).astype(np.float32),
    )


def _alpha(t: float) -> float:
    """Smooth drape ramp (smoothstep) from flat at t=0 to draped at t=1."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def height_and_grad(
    x: np.ndarray, y: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cloth height h(x, y, t) and its spatial gradients.

    At t=0 the cloth is flat at FLAT_HEIGHT. At t=1 it drapes over the sphere
    with full-amplitude wrinkles. Intermediate t blends via smoothstep; the
    flat branch has zero spatial gradient.
    """
    a = _alpha(t)
    draped, dhx, dhy = _draped_height_and_grad(x, y, t)
    flat = np.full_like(draped, FLAT_HEIGHT, dtype=np.float32)
    h = ((1.0 - a) * flat + a * draped).astype(np.float32)
    dhdx = (a * dhx).astype(np.float32)
    dhdy = (a * dhy).astype(np.float32)
    return h, dhdx, dhdy


def height_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Height only (no gradients). Kept for callers that just need h."""
    h, _, _ = height_and_grad(x, y, t)
    return h


def sdf_at_points(
    xyz: np.ndarray, t: float
) -> np.ndarray:
    """
    Analytic signed perpendicular distance to the cloth surface at arbitrary
    3D points.

    Uses the first-order normalized form
        sdf = (z - h(x,y,t)) / sqrt(1 + h_x^2 + h_y^2)
    which satisfies ||∇sdf|| ≈ 1 near the surface. ``xyz`` may be any shape
    ending in 3; the output has the leading shape.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    h, dhx, dhy = height_and_grad(x, y, t)
    norm = np.sqrt(1.0 + dhx * dhx + dhy * dhy)
    return ((z - h) / norm).astype(np.float32)


def create_drape_over_sphere_sdf(
    grid_size: int = 128,
    t: float = 0.0,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic signed perpendicular SDF for the drape-over-sphere scene at
    time ``t``, sampled on a regular (grid_size)^3 voxel grid.

    Args:
        grid_size: Voxels per axis.
        t: Time in [0, 1]. 0 → flat, 1 → fully draped.
        bounds: (min, max) coordinate bounds applied symmetrically to x, y, z.

    Returns:
        sdf: (grid_size, grid_size, grid_size) float32.
        coords: (grid_size,) float32 axis coordinates.
    """
    coords = np.linspace(bounds[0], bounds[1], grid_size, dtype=np.float32)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    # Vectorized pointwise analytic SDF.
    xyz = np.stack([X, Y, Z], axis=-1)
    sdf = sdf_at_points(xyz, t)
    return sdf.astype(np.float32), coords


def save_sdf_to_hdf5(
    sdf: np.ndarray,
    coords: np.ndarray,
    filepath: str,
    time: float,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    metadata: Optional[dict] = None,
) -> None:
    """Write one frame in the P13-compatible HDF5 contract."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset("sdf", data=sdf, compression="gzip")
        f.create_dataset("coords", data=coords)
        f.attrs["time"] = float(time)
        f.attrs["bounds_min"] = float(bounds[0])
        f.attrs["bounds_max"] = float(bounds[1])
        f.attrs["grid_size"] = int(sdf.shape[0])
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def load_sdf_from_hdf5(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read one frame. Returns (sdf, coords, attrs_dict)."""
    with h5py.File(filepath, "r") as f:
        sdf = f["sdf"][:]
        coords = f["coords"][:]
        attrs = dict(f.attrs)
    return sdf, coords, attrs


def generate_dataset(
    output_dir: str,
    num_frames: int = 120,
    grid_size: int = 128,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    fps: float = 30.0,
) -> list:
    """
    Generate a full drape-over-sphere dataset.

    Saves ``num_frames`` HDF5 files to ``<output_dir>/sdf/frame_####.h5``.
    Time values span [0, 1] linearly (mapped to ``num_frames/fps`` seconds via
    metadata).

    Returns the list of generated file paths.
    """
    sdf_dir = Path(output_dir) / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    files = []
    ts = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    dt = 1.0 / fps

    print(
        f"[drape_sdf] Generating {num_frames} frames at {grid_size}^3, "
        f"bounds {bounds}, fps {fps:.1f} (normalized analytic SDF)"
    )
    for i, t in enumerate(ts):
        sdf, coords = create_drape_over_sphere_sdf(
            grid_size=grid_size, t=float(t), bounds=bounds
        )
        filepath = sdf_dir / f"frame_{i:04d}.h5"
        save_sdf_to_hdf5(
            sdf=sdf,
            coords=coords,
            filepath=str(filepath),
            time=float(t),
            bounds=bounds,
            metadata={
                "frame_index": i,
                "frame_time_seconds": float(i * dt),
                "scene": "drape_over_bump",
                "bump_height": float(BUMP_HEIGHT),
                "bump_sigma": float(BUMP_SIGMA),
                "sphere_center_x": float(SPHERE_CENTER[0]),
                "sphere_center_y": float(SPHERE_CENTER[1]),
                "sphere_center_z": float(SPHERE_CENTER[2]),
                "cloth_thickness": float(CLOTH_THICKNESS),
                "flat_height": float(FLAT_HEIGHT),
                "wrinkle_k": float(WRINKLE_K),
                "wrinkle_max_amp": float(WRINKLE_MAX_AMP),
                "sdf_form": "perpendicular_normalized",
            },
        )
        files.append(str(filepath))
        if (i + 1) % 20 == 0 or i == num_frames - 1:
            print(
                f"  frame {i + 1:3d}/{num_frames}: t={t:.3f} "
                f"sdf_range=[{sdf.min():+.3f}, {sdf.max():+.3f}]"
            )
    return files


def extract_heightfield(
    sdf: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Backwards-compatible alias for the canonical extractor, which now lives
    in ``src.sampling.heightfield_extractor``. Both the Prompt-1 export and
    the Prompt-2 Taichi viewer share that single implementation so extraction
    can never drift between training-time reference and viewer-time display.
    """
    from ..sampling.heightfield_extractor import extract_heightfield_from_volume
    return extract_heightfield_from_volume(sdf, coords)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/drape")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--grid_size", type=int, default=128)
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        grid_size=args.grid_size,
    )
