#!/usr/bin/env python3
"""
Generate validation artifacts for a trained WavePINN checkpoint.

Outputs:
- wavefield_sequence.npy: sampled wavefield over time
- snapshots.png: small time-slice montage
- slice_xt.png: x-t plot at a chosen depth to visualize propagation
- metrics.json: simple amplitude / energy summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_blender import load_wavepinn_checkpoint, sample_wavefield_sequence


def save_snapshots_png(fields: np.ndarray, times: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    n_cols = min(4, len(times))
    n_rows = int(np.ceil(len(times) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    vmin = float(fields.min())
    vmax = float(fields.max())

    for idx, ax in enumerate(axes):
        if idx >= len(times):
            ax.axis("off")
            continue
        im = ax.imshow(fields[idx], origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {times[idx]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("WavePINN validation snapshots")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_slice_xt_png(fields: np.ndarray, times: np.ndarray, output_path: Path, z_index: int) -> None:
    import matplotlib.pyplot as plt

    slice_xt = fields[:, :, z_index]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        slice_xt,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        extent=[0.0, 1.0, float(times[0]), float(times[-1])],
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"x-t slice at z index {z_index}")
    fig.colorbar(im, ax=ax, label="u(x, z, t)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def maybe_save_gif(fields: np.ndarray, output_path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    vmin = float(fields.min())
    vmax = float(fields.max())
    frames = []
    for field in fields:
        norm = (field - vmin) / (vmax - vmin + 1e-8)
        rgb = np.uint8(np.clip(norm, 0.0, 1.0) * 255.0)
        frames.append(Image.fromarray(rgb))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a trained WavePINN checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="artifacts/validation", help="Output artifact directory")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--t_start", type=float, default=0.05)
    parser.add_argument("--t_end", type=float, default=0.50)
    parser.add_argument("--slice_z", type=float, default=0.15, help="Normalized z location for x-t slice")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, device = load_wavepinn_checkpoint(args.checkpoint)
    fields, times = sample_wavefield_sequence(
        model=model,
        resolution=args.resolution,
        n_frames=args.frames,
        t_start=args.t_start,
        t_end=args.t_end,
        device=device,
    )

    np.save(output_dir / "wavefield_sequence.npy", fields)
    np.save(output_dir / "times.npy", times)

    snapshot_indices = np.linspace(0, len(times) - 1, min(8, len(times)), dtype=int)
    save_snapshots_png(fields[snapshot_indices], times[snapshot_indices], output_dir / "snapshots.png")

    z_index = int(np.clip(round(args.slice_z * (args.resolution - 1)), 0, args.resolution - 1))
    save_slice_xt_png(fields, times, output_dir / "slice_xt.png", z_index=z_index)
    gif_saved = maybe_save_gif(fields, output_dir / "wavefield.gif")

    energy = np.mean(fields ** 2, axis=(1, 2))
    peak_frame = int(np.argmax(energy))
    metrics = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "device": device,
        "resolution": args.resolution,
        "frames": args.frames,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "field_min": float(fields.min()),
        "field_max": float(fields.max()),
        "field_std": float(fields.std()),
        "frame_energy_mean": float(energy.mean()),
        "frame_energy_std": float(energy.std()),
        "peak_energy_frame": peak_frame,
        "peak_energy_time": float(times[peak_frame]),
        "gif_saved": gif_saved,
        "model_config": checkpoint.get("model_config"),
        "training_args": checkpoint.get("training_args"),
        "final_metrics": checkpoint.get("final_metrics"),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved validation artifacts to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
