#!/usr/bin/env python3
"""
Export the final Taichi cloth artifacts for the HGNN-ClothDyn demo.

This script wraps the viewer's deterministic off-screen export mode so the
project can regenerate the same blog/media assets with a single command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VIEWER_SCRIPT = PROJECT_ROOT / "taichi_cloth_demo.py"
DEFAULT_ARCH = "auto"
DEFAULT_PRESET = "pitch"
DEFAULT_CLIP_START = 2
DEFAULT_CLIP_END = 30
DEFAULT_EXPORT_FRAMES = 132
DEFAULT_EXPORT_FPS = 30
DEFAULT_STILL_FRAME = 24


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


def viewer_export(
    python_bin: str,
    *,
    mode: str,
    preset: str,
    export_dir: Path,
    metadata_out: Path,
    arch: str,
    export_frames: int,
    export_fps: int,
    clip_start: int,
    clip_end: int,
    status_every: int = 24,
) -> None:
    cmd = [
        python_bin,
        str(VIEWER_SCRIPT),
        "--preset",
        preset,
        "--mode",
        mode,
        "--clip-start",
        str(clip_start),
        "--clip-end",
        str(clip_end),
        "--export-only",
        "--export-frames",
        str(export_frames),
        "--export-fps",
        str(export_fps),
        "--export-dir",
        str(export_dir),
        "--metadata-out",
        str(metadata_out),
        "--no-gui",
        "--arch",
        arch,
        "--status-every",
        str(status_every),
    ]
    run(cmd)


def ffmpeg_mp4(frames_dir: Path, fps: int, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        str(output_path),
    ]
    run(cmd)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Export final Taichi artifacts for HGNN-ClothDyn.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable with Taichi installed")
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=["auto", "cuda", "vulkan", "metal", "cpu"])
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--clip-start", type=int, default=DEFAULT_CLIP_START)
    parser.add_argument("--clip-end", type=int, default=DEFAULT_CLIP_END)
    parser.add_argument("--export-frames", type=int, default=DEFAULT_EXPORT_FRAMES)
    parser.add_argument("--export-fps", type=int, default=DEFAULT_EXPORT_FPS)
    parser.add_argument("--still-frame", type=int, default=DEFAULT_STILL_FRAME)
    args = parser.parse_args()

    final_dir = PROJECT_ROOT / "artifacts" / "taichi_final"
    presets_dir = PROJECT_ROOT / "artifacts" / "taichi_presets"
    final_dir.mkdir(parents=True, exist_ok=True)
    presets_dir.mkdir(parents=True, exist_ok=True)

    export_specs = [
        ("physics_demo", "physics"),
        ("hgnn_demo", "hgnn"),
        ("comparison", "compare"),
    ]

    generated: dict[str, str] = {}
    render_pass_metadata: dict[str, dict] = {}
    for stem, mode in export_specs:
        frames_dir = final_dir / f"{stem}_frames"
        metadata_path = final_dir / f"{stem}_frames_metadata.json"
        viewer_export(
            args.python_bin,
            mode=mode,
            preset=args.preset,
            export_dir=frames_dir,
            metadata_out=metadata_path,
            arch=args.arch,
            export_frames=args.export_frames,
            export_fps=args.export_fps,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
        )
        mp4_path = final_dir / f"{stem}.mp4"
        ffmpeg_mp4(frames_dir, args.export_fps, mp4_path)
        generated[stem] = str(mp4_path.relative_to(PROJECT_ROOT))
        render_pass_metadata[stem] = load_json(metadata_path)

    poster_frames_dir = final_dir / "poster_frame"
    poster_metadata_path = final_dir / "poster_frame_metadata.json"
    viewer_export(
        args.python_bin,
        mode="compare",
        preset=args.preset,
        export_dir=poster_frames_dir,
        metadata_out=poster_metadata_path,
        arch=args.arch,
        export_frames=1,
        export_fps=args.export_fps,
        clip_start=args.still_frame,
        clip_end=args.still_frame,
        status_every=1,
    )
    poster_path = final_dir / "poster.png"
    shutil.copy2(poster_frames_dir / "frame_0000.png", poster_path)
    generated["poster"] = str(poster_path.relative_to(PROJECT_ROOT))

    for preset in ("research", "pitch", "dramatic"):
        preset_frames_dir = presets_dir / f"{preset}_frame"
        preset_metadata_path = presets_dir / f"{preset}_frame_metadata.json"
        viewer_export(
            args.python_bin,
            mode="compare",
            preset=preset,
            export_dir=preset_frames_dir,
            metadata_out=preset_metadata_path,
            arch=args.arch,
            export_frames=1,
            export_fps=args.export_fps,
            clip_start=args.still_frame,
            clip_end=args.still_frame,
            status_every=1,
        )
        preset_png = presets_dir / f"{preset}.png"
        shutil.copy2(preset_frames_dir / "frame_0000.png", preset_png)
        generated[f"preset_{preset}"] = str(preset_png.relative_to(PROJECT_ROOT))

    rollout_metrics = load_json(PROJECT_ROOT / "outputs" / "validation" / "rollout_metrics.json")
    benchmark_metrics = load_json(PROJECT_ROOT / "results_recheck" / "benchmark_results.json")

    metadata = {
        "selected_default_preset": args.preset,
        "selected_default_mode": "compare",
        "selected_surface_mode": render_pass_metadata["comparison"]["surface_mode"],
        "clip": {
            "start_frame": args.clip_start,
            "end_frame": args.clip_end,
            "still_frame": args.still_frame,
            "export_frames": args.export_frames,
            "export_fps": args.export_fps,
        },
        "renderer": {
            "viewer_script": str(VIEWER_SCRIPT.relative_to(PROJECT_ROOT)),
            "python_bin": args.python_bin,
            "arch": args.arch,
            "source_mesh_size": render_pass_metadata["comparison"]["source_mesh_size"],
            "display_mesh_size": render_pass_metadata["comparison"]["display_mesh_size"],
        },
        "artifacts": generated,
        "render_passes": {
            key: {
                "mode": value["mode"],
                "surface_mode": value["surface_mode"],
                "display_mesh_size": value["display_mesh_size"],
            }
            for key, value in render_pass_metadata.items()
        },
        "validation": {
            "short_rollout_recheck": benchmark_metrics["summary"],
            "full_rollout_metrics": {
                "position_rmse": rollout_metrics["position_rmse"],
                "edge_length_rest_mae": rollout_metrics["edge_length_rest_mae"],
                "fps_benchmark": rollout_metrics["fps_benchmark"],
            },
        },
        "notes": [
            "The polished demos intentionally use the early rollout window where the recovered checkpoint remains close to the physics baseline.",
            "Pitch and dramatic presets use a denser Catmull-Rom display surface for presentation, while research mode remains on the raw simulation mesh.",
            "Long-horizon drift still exists and is documented in outputs/validation/rollout_metrics.json.",
        ],
    }
    metadata_path = final_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
