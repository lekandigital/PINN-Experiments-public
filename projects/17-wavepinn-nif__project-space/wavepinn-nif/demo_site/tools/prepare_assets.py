#!/usr/bin/env python3
"""
Prepare self-contained static-site assets for the WavePINN demo microsite.

This script copies the real project artifacts into demo_site/assets and
generates motion assets for the reference field and learned WavePINN field.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


SITE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SITE_ROOT.parent
DATA_DIR = SITE_ROOT / "assets" / "data"
MEDIA_DIR = SITE_ROOT / "assets" / "media"
MPLCONFIGDIR = SITE_ROOT / ".mpl-cache"


def copy_file(src: Path, dest_name: str) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dest = MEDIA_DIR / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.name


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def render_sequence_frames(
    fields: np.ndarray,
    poster_path: Path,
    temp_dir: Path,
    output_size: int = 640,
) -> None:
    from matplotlib import colormaps

    cmap = colormaps["coolwarm"]
    vmax = float(np.max(np.abs(fields)))
    vmax = max(vmax, 1e-6)

    for idx, field in enumerate(fields):
        normalized = np.clip((field / vmax + 1.0) * 0.5, 0.0, 1.0)
        rgba = cmap(normalized)
        rgb = np.uint8(np.clip(rgba[..., :3], 0.0, 1.0) * 255.0)
        image = Image.fromarray(rgb, mode="RGB").resize((output_size, output_size), Image.Resampling.BICUBIC)
        frame_path = temp_dir / f"frame_{idx:04d}.png"
        image.save(frame_path)
        if idx == len(fields) // 2:
            image.save(poster_path)


def make_mp4_from_npy(
    npy_path: Path,
    output_name: str,
    poster_name: str,
    fps: int = 24,
    output_size: int = 640,
) -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    fields = np.load(npy_path)
    output_path = MEDIA_DIR / output_name
    poster_path = MEDIA_DIR / poster_name

    with tempfile.TemporaryDirectory(prefix="wavepinn_site_frames_") as temp_root:
        temp_dir = Path(temp_root)
        render_sequence_frames(fields=fields, poster_path=poster_path, temp_dir=temp_dir, output_size=output_size)
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%04d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_path.name, poster_path.name


def maybe_make_mp4_from_npy(
    npy_path: Path,
    output_name: str,
    poster_name: str,
    fps: int = 24,
    output_size: int = 640,
) -> tuple[str, str]:
    if not npy_path.exists():
        return "", ""
    return make_mp4_from_npy(
        npy_path=npy_path,
        output_name=output_name,
        poster_name=poster_name,
        fps=fps,
        output_size=output_size,
    )


def build_data() -> dict:
    taichi_meta = load_json(PROJECT_ROOT / "artifacts/taichi_final/metadata.json")
    learned_metrics = load_json(PROJECT_ROOT / "outputs/supervised_fd_long/validation_model/metrics.json")
    reference_meta = load_json(PROJECT_ROOT / "outputs/supervised_fd_long/reference/reference_metadata.json")

    assets: dict[str, dict] = {
        "hero_video": {
            "src": copy_file(PROJECT_ROOT / "artifacts/taichi_final/wave_demo.mp4", "taichi-final.mp4"),
            "poster": copy_file(PROJECT_ROOT / "artifacts/taichi_final/poster.png", "taichi-final-poster.png"),
        },
        "taichi_presets": {
            "research": copy_file(
                first_existing(
                    PROJECT_ROOT / "artifacts/taichi_presets/research_v2.png",
                    PROJECT_ROOT / "artifacts/taichi_presets/research.png",
                ),
                "preset-research.png",
            ),
            "pitch": copy_file(
                first_existing(
                    PROJECT_ROOT / "artifacts/taichi_presets/pitch_v2.png",
                    PROJECT_ROOT / "artifacts/taichi_presets/pitch.png",
                ),
                "preset-pitch.png",
            ),
            "dramatic": copy_file(PROJECT_ROOT / "artifacts/taichi_presets/dramatic.png", "preset-dramatic.png"),
        },
        "reference": {
            "snapshots": copy_file(
                PROJECT_ROOT / "outputs/supervised_fd_long/reference/reference_snapshots.png",
                "reference-snapshots.png",
            ),
            "slice": copy_file(
                PROJECT_ROOT / "outputs/supervised_fd_long/reference/reference_slice_xt.png",
                "reference-slice.png",
            ),
        },
        "learned": {
            "snapshots": copy_file(
                PROJECT_ROOT / "outputs/supervised_fd_long/validation_model/snapshots.png",
                "learned-snapshots.png",
            ),
            "slice": copy_file(
                PROJECT_ROOT / "outputs/supervised_fd_long/validation_model/slice_xt.png",
                "learned-slice.png",
            ),
        },
    }

    reference_motion, reference_poster = maybe_make_mp4_from_npy(
        PROJECT_ROOT / "outputs/supervised_fd_long/reference/reference_wavefield.npy",
        output_name="reference-field.mp4",
        poster_name="reference-field-poster.png",
        fps=24,
    )
    learned_motion, learned_poster = maybe_make_mp4_from_npy(
        PROJECT_ROOT / "outputs/supervised_fd_long/validation_model/wavefield_sequence.npy",
        output_name="learned-field.mp4",
        poster_name="learned-field-poster.png",
        fps=24,
    )

    assets["reference"]["motion"] = reference_motion
    assets["reference"]["poster"] = reference_poster
    assets["learned"]["motion"] = learned_motion
    assets["learned"]["poster"] = learned_poster

    data = {
        "title": "WavePINN Wave Demo",
        "subtitle": "Reference wave dynamics without the PINN, learned WavePINN output with it, and a final Taichi GGUI surface presentation.",
        "taichi": taichi_meta,
        "learned_metrics": learned_metrics,
        "reference_metadata": reference_meta,
        "assets": assets,
        "paths": {
            "taichi_video": "artifacts/taichi_final/wave_demo.mp4",
            "taichi_metadata": "artifacts/taichi_final/metadata.json",
            "checkpoint": "outputs/supervised_fd_long/checkpoints/wavefield_nif_supervised.pt",
            "learned_snapshots": "outputs/supervised_fd_long/validation_model/snapshots.png",
            "learned_slice": "outputs/supervised_fd_long/validation_model/slice_xt.png",
            "learned_metrics": "outputs/supervised_fd_long/validation_model/metrics.json",
            "learned_sequence": "outputs/supervised_fd_long/validation_model/wavefield_sequence.npy",
            "reference_snapshots": "outputs/supervised_fd_long/reference/reference_snapshots.png",
            "reference_slice": "outputs/supervised_fd_long/reference/reference_slice_xt.png",
            "reference_metadata": "outputs/supervised_fd_long/reference/reference_metadata.json",
            "reference_sequence": "outputs/supervised_fd_long/reference/reference_wavefield.npy",
        },
    }
    return data


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    data = build_data()
    output_path = DATA_DIR / "site-data.json"
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
