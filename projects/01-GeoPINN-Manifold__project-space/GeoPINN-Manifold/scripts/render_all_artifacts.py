#!/usr/bin/env python3
"""
Render all final artifacts for the GeoPINN demo:

  artifacts/variant_test/{research,pitch,dramatic}.mp4
  artifacts/taichi_final/geopinn_sphere.mp4
  artifacts/taichi_final/analytical_sphere.mp4
  artifacts/taichi_final/comparison.mp4
  artifacts/taichi_final/poster.png
  artifacts/taichi_final/metadata.json
  artifacts/taichi_presets/{research,pitch,dramatic}.png

Requires ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from taichi_sphere_demo import SphereViewer, PRESETS  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
VARIANT_DIR = os.path.join(ARTIFACT_DIR, "variant_test")
FINAL_DIR = os.path.join(ARTIFACT_DIR, "taichi_final")
PRESETS_DIR = os.path.join(ARTIFACT_DIR, "taichi_presets")


def ffmpeg_png_to_mp4(png_dir: str, out_path: str, fps: int = 30) -> None:
    """Encode a directory of PNG frames (f_0000.png ...) to MP4 (H.264)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(png_dir, "f_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"  encoded: {out_path}")


def ffmpeg_side_by_side(
    left_mp4: str, right_mp4: str, out_path: str,
    left_label: str = "ANALYTICAL",
    right_label: str = "GeoPINN",
) -> None:
    """Create a side-by-side comparison video (no ffmpeg drawtext — labels
    come from the website layer)."""
    filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
    cmd = [
        "ffmpeg", "-y",
        "-i", left_mp4,
        "-i", right_mp4,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"  comparison: {out_path}")


def render_variant_tests(viewer: SphereViewer, n_frames: int = 180, fps: int = 30):
    """Render one clip per preset to verify they look meaningfully different."""
    os.makedirs(VARIANT_DIR, exist_ok=True)
    for preset_name in ["research", "pitch", "dramatic"]:
        print(f"\n=== Variant test: {preset_name} ===")
        with tempfile.TemporaryDirectory() as tmp:
            viewer.render_video_sequence(
                preset_name=preset_name,
                out_dir=tmp,
                n_frames=n_frames,
                fps=fps,
                mode="pinn",
                resolution=(960, 720),
            )
            out_mp4 = os.path.join(VARIANT_DIR, f"{preset_name}.mp4")
            ffmpeg_png_to_mp4(tmp, out_mp4, fps=fps)


def render_final_videos(viewer: SphereViewer, n_frames: int = 300, fps: int = 30):
    """Render the final pitch-preset GeoPINN and analytical clips + comparison."""
    os.makedirs(FINAL_DIR, exist_ok=True)

    # GeoPINN version
    print("\n=== Final: GeoPINN sphere (pitch) ===")
    with tempfile.TemporaryDirectory() as tmp_pinn:
        viewer.render_video_sequence(
            preset_name="pitch", out_dir=tmp_pinn,
            n_frames=n_frames, fps=fps, mode="pinn",
            resolution=(1280, 960),
        )
        pinn_mp4 = os.path.join(FINAL_DIR, "geopinn_sphere.mp4")
        ffmpeg_png_to_mp4(tmp_pinn, pinn_mp4, fps=fps)

    # Analytical version (same camera + preset)
    print("\n=== Final: Analytical sphere (pitch) ===")
    with tempfile.TemporaryDirectory() as tmp_anal:
        viewer.render_video_sequence(
            preset_name="pitch", out_dir=tmp_anal,
            n_frames=n_frames, fps=fps, mode="analytical",
            resolution=(1280, 960),
        )
        anal_mp4 = os.path.join(FINAL_DIR, "analytical_sphere.mp4")
        ffmpeg_png_to_mp4(tmp_anal, anal_mp4, fps=fps)

    # Side-by-side comparison
    print("\n=== Final: Side-by-side comparison ===")
    cmp_mp4 = os.path.join(FINAL_DIR, "comparison.mp4")
    ffmpeg_side_by_side(anal_mp4, pinn_mp4, cmp_mp4,
                         left_label="ANALYTICAL",
                         right_label="GeoPINN")


def render_poster_and_stills(viewer: SphereViewer):
    print("\n=== Poster ===")
    viewer.render_poster(os.path.join(FINAL_DIR, "poster.png"))

    print("\n=== Preset stills ===")
    viewer.render_preset_stills(PRESETS_DIR)


def write_metadata(viewer: SphereViewer):
    """Collect training + render metadata into a single JSON."""
    # Load training metrics and checkpoint config
    with open(os.path.join(PROJECT_ROOT, "outputs", "metrics.json")) as f:
        metrics = json.load(f)

    import torch
    ckpt = torch.load(
        os.path.join(PROJECT_ROOT, "checkpoints", "geopinn_sphere_trained.pt"),
        map_location="cpu",
        weights_only=False,
    )
    cfg = ckpt.get("config", {})
    history = ckpt.get("history", {})

    # Model size (param count)
    from taichi_sphere_demo import PRESETS as _PRESETS
    pitch_vis = _PRESETS["pitch"].visual

    # Count params in the saved state dict
    state = ckpt["model_state_dict"]
    n_params = sum(v.numel() for v in state.values())

    meta = {
        "project": "GeoPINN-Manifold",
        "pde": "du/dt = Delta_S u   (heat equation on unit sphere)",
        "initial_condition": "u(x,0) = x*y + 0.5*y*z   (l=2 spherical harmonics)",
        "analytical": "u(x,t) = exp(-6t) * (x*y + 0.5*y*z)",
        "model": {
            "architecture": "MLP (x,y,z,t) -> u",
            "hidden_dim": cfg.get("hidden_dim"),
            "num_layers": cfg.get("num_layers"),
            "activation": cfg.get("activation"),
            "n_params": n_params,
        },
        "training": {
            "epochs": cfg.get("n_epochs"),
            "learning_rate": cfg.get("lr"),
            "n_collocation": cfg.get("n_collocation"),
            "n_ic": cfg.get("n_ic"),
            "n_supervision": cfg.get("n_supervision"),
            "final_loss": history["loss"][-1] if history.get("loss") else None,
            "final_pde_residual": history["pde_loss"][-1] if history.get("pde_loss") else None,
            "final_ic_loss": history["ic_loss"][-1] if history.get("ic_loss") else None,
            "final_l2_error": history["l2_error"][-1] if history.get("l2_error") else None,
            "device": "NVIDIA GeForce RTX 3090 Ti",
        },
        "evaluation": {
            "mean_l2": metrics["mean_l2"],
            "mean_psnr_db": metrics["mean_psnr"],
            "n_dense_points": metrics["n_dense_points"],
            "n_timesteps": metrics["n_timesteps"],
            "timesteps": metrics["timesteps"],
            "l2_per_timestep": metrics["l2_error_per_timestep"],
            "relative_l2_per_timestep": metrics["relative_l2_per_timestep"],
            "psnr_per_timestep": metrics["psnr_per_timestep"],
        },
        "render": {
            "sphere_resolution": {
                "vertices": viewer.n_verts,
                "triangles": viewer.n_faces,
                "edges": viewer.n_edges,
                "icosphere_subdivisions": 5,
            },
            "deformation_amplitude": pitch_vis.deformation_amplitude,
            "colormap": "diverging (cool blue <-> warm orange, tanh transfer)",
            "color_gain": pitch_vis.color_gain,
            "color_percentile": pitch_vis.color_percentile,
            "rotation_speed_rad_s": pitch_vis.rotation_speed,
            "video_fps": 30,
            "video_frames": 300,
            "video_resolution": [1280, 960],
            "hero_preset": "pitch",
        },
    }

    path = os.path.join(FINAL_DIR, "metadata.json")
    os.makedirs(FINAL_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-variant", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--skip-stills", action="store_true")
    ap.add_argument("--variant-frames", type=int, default=180)
    ap.add_argument("--final-frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    print("Initializing viewer ...")
    viewer = SphereViewer(preset_name="pitch")

    t_start = time.time()

    if not args.skip_variant:
        render_variant_tests(viewer, n_frames=args.variant_frames, fps=args.fps)
    if not args.skip_final:
        render_final_videos(viewer, n_frames=args.final_frames, fps=args.fps)
    if not args.skip_stills:
        render_poster_and_stills(viewer)

    write_metadata(viewer)

    elapsed = time.time() - t_start
    print(f"\n=== All artifacts rendered in {elapsed:.1f}s ===")
    print(f"Variant: {VARIANT_DIR}/")
    print(f"Final:   {FINAL_DIR}/")
    print(f"Presets: {PRESETS_DIR}/")


if __name__ == "__main__":
    main()
