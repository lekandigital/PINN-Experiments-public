#!/usr/bin/env python3
"""Render per-timestep snapshot PNGs for the website's field-views section."""

import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from taichi_sphere_demo import SphereViewer, PRESETS  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "timesteps"
)


def main():
    viewer = SphereViewer(preset_name="pitch")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pick 6 interesting timesteps (evenly spaced)
    T = viewer.n_timesteps
    idxs = [0, 2, 4, 6, 8, 11]

    # Render in a fresh window per chunk to avoid swap chain limits
    CHUNK = 4

    for mode in ["pinn", "analytical"]:
        viewer.preset = PRESETS["pitch"]
        viewer.show_wireframe = False
        viewer.show_analytical = (mode == "analytical")
        viewer._preprocess_all()

        mode_dir = os.path.join(OUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)

        i = 0
        while i < len(idxs):
            chunk_end = min(i + CHUNK, len(idxs))
            window = ti.ui.Window(
                f"ts_{mode}_{i}", (800, 600), show_window=False,
            )
            canvas = window.get_canvas()
            scene = window.get_scene()
            camera = ti.ui.Camera()
            camera.position(0.2, 0.8, 2.3)
            camera.lookat(0.0, 0.0, 0.0)
            camera.up(0.0, 1.0, 0.0)
            camera.fov(45.0)

            for j in range(i, chunk_end):
                t_idx = idxs[j]
                viewer.frame_idx = t_idx
                viewer.rotation_angle = 0.5 + 0.2 * j
                viewer._update_frame()
                canvas.set_background_color(viewer.preset.bg_color)
                viewer._render_scene(scene, camera)
                canvas.scene(scene)
                path = os.path.join(mode_dir, f"t{t_idx:02d}.png")
                window.save_image(path)
                print(f"  saved {path}")

            window.destroy()
            i = chunk_end

    print(f"\nDone. See {OUT_DIR}/")


if __name__ == "__main__":
    main()
