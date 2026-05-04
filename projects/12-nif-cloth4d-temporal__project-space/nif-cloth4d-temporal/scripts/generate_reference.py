#!/usr/bin/env python3
"""
Generate the Project 12 reference data pack.

Produces:
    data/drape/sdf/frame_####.h5            — 120 frames × 128^3 analytic SDF
    outputs/reference/reference_sdf_sequence.npy
        (120, 128, 128, 128) float16 memmap
    outputs/reference/heightfield_sequence.npy
        (120, 128, 128)      float32 — zero-crossing heightfield per frame
    outputs/reference/times.npy             (120,) float32
    outputs/reference/snapshots.png         5 thumbnails at t=0, .25, .5, .75, 1
    outputs/reference/reference_metadata.json

Runs entirely on CPU; takes ~1-2 min for the default settings. No GPU needed
for reference generation (the drape SDF is analytic).
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.drape_sdf_generator import (  # noqa: E402
    BUMP_HEIGHT,
    BUMP_SIGMA,
    CLOTH_THICKNESS,
    FLAT_HEIGHT,
    SPHERE_CENTER,
    SPHERE_RADIUS,
    WRINKLE_K,
    WRINKLE_MAX_AMP,
    extract_heightfield,
    generate_dataset,
    load_sdf_from_hdf5,
)


def _render_snapshot(hf: np.ndarray, coords: np.ndarray, t: float, ax) -> None:
    """Shaded-relief style plot of a heightfield."""
    ax.imshow(
        hf.T,
        origin="lower",
        extent=(coords[0], coords[-1], coords[0], coords[-1]),
        cmap="viridis",
        vmin=0.0,
        vmax=max(FLAT_HEIGHT + 0.1, SPHERE_RADIUS + 0.1),
    )
    ax.set_title(f"t = {t:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(PROJECT_ROOT / "data" / "drape"),
        help="Where the per-frame HDF5 files are written.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "reference"),
        help="Where the stacked .npy arrays + snapshots go.",
    )
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time_mod.perf_counter()

    print("[step 1/3] generate HDF5 frames")
    files = generate_dataset(
        output_dir=str(data_dir),
        num_frames=args.num_frames,
        grid_size=args.grid_size,
        fps=args.fps,
    )

    print("[step 2/3] stack into .npy + heightfields")
    # The full SDF volume for 120 frames at 128^3 in float16 is ~500 MB —
    # stream frame-by-frame into a memmap to avoid peak RAM.
    field_path = output_dir / "reference_sdf_sequence.npy"
    field = np.lib.format.open_memmap(
        field_path,
        mode="w+",
        dtype=np.float16,
        shape=(args.num_frames, args.grid_size, args.grid_size, args.grid_size),
    )

    heightfields = np.zeros(
        (args.num_frames, args.grid_size, args.grid_size), dtype=np.float32
    )
    times = np.zeros(args.num_frames, dtype=np.float32)
    coords_ref = None
    for i, f in enumerate(files):
        sdf, coords, attrs = load_sdf_from_hdf5(f)
        field[i] = sdf.astype(np.float16)
        heightfields[i] = extract_heightfield(sdf, coords)
        times[i] = float(attrs["time"])
        if coords_ref is None:
            coords_ref = coords
    field.flush()
    del field  # close memmap

    np.save(output_dir / "heightfield_sequence.npy", heightfields)
    np.save(output_dir / "times.npy", times)
    np.save(output_dir / "coords.npy", coords_ref)

    print("[step 3/3] render snapshots + write metadata")
    snap_ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    fig, axs = plt.subplots(1, len(snap_ts), figsize=(4 * len(snap_ts), 4))
    for ax, t_target in zip(axs, snap_ts):
        idx = int(np.argmin(np.abs(times - t_target)))
        _render_snapshot(heightfields[idx], coords_ref, float(times[idx]), ax)
    fig.tight_layout()
    fig.savefig(output_dir / "snapshots.png", dpi=100)
    plt.close(fig)

    metadata = {
        "scene": "drape_over_bump",
        "num_frames": int(args.num_frames),
        "grid_size": int(args.grid_size),
        "fps": float(args.fps),
        "bounds": [-1.0, 1.0],
        "bump_height": float(BUMP_HEIGHT),
        "bump_sigma": float(BUMP_SIGMA),
        "sphere_radius": float(SPHERE_RADIUS),
        "sphere_center": list(map(float, SPHERE_CENTER)),
        "cloth_thickness": float(CLOTH_THICKNESS),
        "flat_height": float(FLAT_HEIGHT),
        "wrinkle_k": float(WRINKLE_K),
        "wrinkle_max_amp": float(WRINKLE_MAX_AMP),
        "hdf5_dir": str(data_dir / "sdf"),
        "sdf_sequence": str(field_path),
        "heightfield_sequence": str(output_dir / "heightfield_sequence.npy"),
        "times": str(output_dir / "times.npy"),
        "snapshots": str(output_dir / "snapshots.png"),
        "wall_clock_seconds": round(time_mod.perf_counter() - t0, 2),
    }
    with open(output_dir / "reference_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    dt = time_mod.perf_counter() - t0
    print(f"[done] reference pack written to {output_dir} in {dt:.1f}s")


if __name__ == "__main__":
    main()
