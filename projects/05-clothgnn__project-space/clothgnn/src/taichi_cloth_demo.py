#!/usr/bin/env python3
"""
Taichi GGUI cloth demo for ClothGNN.

Loads precomputed physics-baseline and GNN-predicted cloth sequences,
renders them as a triangulated mesh with height-based coloring,
warm/cool lighting, and dark background.

Supports three presets (research / pitch / dramatic), live playback,
mode toggle (physics / GNN), wireframe overlay, and deterministic
offline export for reproducible video generation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import taichi as ti
except ImportError:
    ti = None


# ════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CameraPose:
    position: tuple
    lookat: tuple
    fov: float


@dataclass(frozen=True)
class LightConfig:
    key_pos: tuple
    key_color: tuple
    key_intensity: float
    fill_pos: tuple
    fill_color: tuple
    fill_intensity: float
    ambient: tuple


@dataclass(frozen=True)
class DemoPreset:
    name: str
    camera: CameraPose
    light: LightConfig
    playback_fps: float
    height_scale: float
    color_mode: str          # "height" | "displacement" | "solid"
    show_wireframe: bool
    display_upsample: int    # 1 = raw mesh
    window_res: tuple
    background: tuple
    ground_color: tuple
    ground_margin: float
    wireframe_color: tuple
    wireframe_width: float
    temporal_smooth: int     # 0 = none, 3 = 3-frame moving avg


# ════════════════════════════════════════════════════════════════
# Presets — tuned for dark-background cloth rendering
# ════════════════════════════════════════════════════════════════

PRESETS: dict[str, DemoPreset] = {
    "research": DemoPreset(
        name="research",
        camera=CameraPose(
            position=(0.0, 2.4, 2.6),
            lookat=(0.0, -0.3, 0.0),
            fov=32.0,
        ),
        light=LightConfig(
            key_pos=(2.0, 3.0, 2.0),
            key_color=(0.92, 0.92, 0.90),
            key_intensity=0.75,
            fill_pos=(-1.8, 1.5, -1.5),
            fill_color=(0.40, 0.45, 0.55),
            fill_intensity=0.40,
            ambient=(0.22, 0.22, 0.24),
        ),
        playback_fps=20.0,
        height_scale=1.0,
        color_mode="height",
        show_wireframe=True,
        display_upsample=1,
        window_res=(1440, 900),
        background=(0.065, 0.070, 0.080),
        ground_color=(0.04, 0.04, 0.05),
        ground_margin=0.08,
        wireframe_color=(0.12, 0.13, 0.16),
        wireframe_width=0.4,
        temporal_smooth=0,
    ),
    "pitch": DemoPreset(
        name="pitch",
        camera=CameraPose(
            position=(-0.6, 2.0, 2.4),
            lookat=(0.0, -0.35, 0.0),
            fov=28.0,
        ),
        light=LightConfig(
            key_pos=(2.2, 3.5, 1.8),
            key_color=(1.00, 0.95, 0.88),
            key_intensity=0.85,
            fill_pos=(-2.0, 1.2, -1.2),
            fill_color=(0.35, 0.42, 0.55),
            fill_intensity=0.45,
            ambient=(0.18, 0.19, 0.22),
        ),
        playback_fps=15.0,
        height_scale=1.2,
        color_mode="height",
        show_wireframe=False,
        display_upsample=1,
        window_res=(1600, 900),
        background=(0.035, 0.040, 0.060),
        ground_color=(0.025, 0.028, 0.035),
        ground_margin=0.06,
        wireframe_color=(0.06, 0.07, 0.09),
        wireframe_width=0.5,
        temporal_smooth=1,
    ),
    "dramatic": DemoPreset(
        name="dramatic",
        camera=CameraPose(
            position=(-1.0, 1.5, 1.8),
            lookat=(0.05, -0.4, 0.05),
            fov=26.0,
        ),
        light=LightConfig(
            key_pos=(2.5, 4.0, 1.2),
            key_color=(1.05, 0.96, 0.85),
            key_intensity=0.90,
            fill_pos=(-2.2, 0.8, -1.5),
            fill_color=(0.25, 0.35, 0.55),
            fill_intensity=0.35,
            ambient=(0.15, 0.16, 0.18),
        ),
        playback_fps=24.0,
        height_scale=1.4,
        color_mode="height",
        show_wireframe=False,
        display_upsample=1,
        window_res=(1920, 1080),
        background=(0.020, 0.025, 0.035),
        ground_color=(0.015, 0.018, 0.025),
        ground_margin=0.05,
        wireframe_color=(0.04, 0.05, 0.06),
        wireframe_width=0.6,
        temporal_smooth=1,
    ),
}

# ════════════════════════════════════════════════════════════════
# Height-based diverging colormap
# ════════════════════════════════════════════════════════════════

def height_colormap(y_values: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """
    Warm (amber/orange) where cloth sags low, cool (blue/teal) where high.
    Neutral silver-grey in the middle.
    """
    # Normalize to [0, 1]
    span = max(y_max - y_min, 1e-8)
    t = np.clip((y_values - y_min) / span, 0.0, 1.0)

    # Colors: low (warm) -> mid (neutral) -> high (cool)
    warm = np.array([0.95, 0.60, 0.32])    # amber-orange
    neutral = np.array([0.82, 0.84, 0.88]) # light silver
    cool = np.array([0.30, 0.55, 0.85])    # steel blue

    colors = np.zeros((len(t), 3), dtype=np.float32)
    low_mask = t < 0.5
    high_mask = ~low_mask

    # Low half: warm -> neutral
    s = t[low_mask] * 2.0  # 0..1
    colors[low_mask] = (1 - s[:, None]) * warm + s[:, None] * neutral

    # High half: neutral -> cool
    s = (t[high_mask] - 0.5) * 2.0  # 0..1
    colors[high_mask] = (1 - s[:, None]) * neutral + s[:, None] * cool

    return colors


def displacement_colormap(positions: np.ndarray, rest_positions: np.ndarray) -> np.ndarray:
    """Single-hue intensity based on displacement magnitude from rest."""
    disp = np.linalg.norm(positions - rest_positions, axis=1)
    d_max = max(disp.max(), 1e-8)
    t = np.clip(disp / d_max, 0.0, 1.0)
    base = np.array([0.25, 0.50, 0.80])
    bright = np.array([0.95, 0.70, 0.40])
    return ((1 - t[:, None]) * base + t[:, None] * bright).astype(np.float32)


# ════════════════════════════════════════════════════════════════
# Sequence loader and preprocessor
# ════════════════════════════════════════════════════════════════

class ClothSequences:
    """Loads and preprocesses physics + GNN cloth sequences."""

    def __init__(self, outputs_dir: Path, trustworthy_frames: int | None = None,
                 height_scale: float = 1.0, temporal_smooth: int = 0):
        mesh_info_path = outputs_dir / "mesh_info.json"
        with open(mesh_info_path) as f:
            self.mesh_info = json.load(f)

        self.physics = np.load(outputs_dir / "physics_baseline" / "cloth_sequence.npy")
        self.gnn = np.load(outputs_dir / "gnn_prediction" / "cloth_sequence.npy")

        self.grid_dims = self.mesh_info["grid_dimensions"]
        self.n_verts = self.mesh_info["vertex_count"]
        self.faces = np.array(self.mesh_info["face_indices"], dtype=np.int32)

        # Determine trustworthy window
        if trustworthy_frames is not None:
            tw = trustworthy_frames
        else:
            tw = self.mesh_info.get("trustworthy_rollout_frames_10pct",
                                    self.mesh_info.get("trustworthy_rollout_frames_5pct", len(self.gnn)))
        tw = min(tw, len(self.physics), len(self.gnn))
        tw = max(tw, 2)

        self.physics = self.physics[:tw].copy()
        self.gnn = self.gnn[:tw].copy()
        self.n_frames = tw

        # Center and scale
        all_pos = np.concatenate([self.physics, self.gnn], axis=0)
        self.center = all_pos.mean(axis=(0, 1))
        self.physics -= self.center
        self.gnn -= self.center

        span = np.ptp(all_pos.reshape(-1, 3), axis=0).max()
        self.scale_factor = 2.0 / max(span, 1e-8)
        self.physics *= self.scale_factor
        self.gnn *= self.scale_factor

        # Rest positions (frame 0 of physics, before height scale)
        self.rest_positions = self.physics[0].copy()

        # Height scale: amplify Y displacement
        if height_scale != 1.0:
            rest_y = self.physics[0, :, 1].copy()
            for seq in [self.physics, self.gnn]:
                for t in range(len(seq)):
                    seq[t, :, 1] = rest_y + (seq[t, :, 1] - rest_y) * height_scale

        # Temporal smoothing
        if temporal_smooth > 0:
            kernel = np.ones(temporal_smooth * 2 + 1) / (temporal_smooth * 2 + 1)
            for seq in [self.physics, self.gnn]:
                for dim in range(3):
                    for v in range(self.n_verts):
                        seq[:, v, dim] = np.convolve(seq[:, v, dim], kernel, mode='same')

        # Global Y range for consistent coloring
        self.y_min = min(self.physics[:, :, 1].min(), self.gnn[:, :, 1].min())
        self.y_max = max(self.physics[:, :, 1].max(), self.gnn[:, :, 1].max())

        # Build wireframe edges from faces
        edge_set = set()
        for face in self.faces:
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                edge_set.add((min(a, b), max(a, b)))
        edges = np.array(sorted(edge_set), dtype=np.int32)
        self.line_indices = edges.flatten()

        # Triangle indices (flat)
        self.triangle_indices = self.faces.flatten().astype(np.int32)

        print(f"Loaded sequences: {self.n_frames} frames, {self.n_verts} verts, "
              f"{len(self.faces)} faces")
        print(f"  Y range: [{self.y_min:.3f}, {self.y_max:.3f}]")
        print(f"  Scale factor: {self.scale_factor:.4f}")


# ════════════════════════════════════════════════════════════════
# Ground plane
# ════════════════════════════════════════════════════════════════

def build_ground_plane(y: float, extent: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    h = extent * 0.5
    verts = np.array([[-h, y, -h], [h, y, -h], [-h, y, h], [h, y, h]], dtype=np.float32)
    idx = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
    return verts, idx


# ════════════════════════════════════════════════════════════════
# Main viewer
# ════════════════════════════════════════════════════════════════

class TaichiClothDemo:
    def __init__(
        self,
        sequences: ClothSequences,
        preset: DemoPreset,
        export_dir: Path | None,
        export_only: bool,
        show_gui: bool,
    ):
        self.seq = sequences
        self.preset = preset
        self.export_dir = export_dir
        self.export_only = export_only
        self.show_gui = show_gui and not export_only

        # State
        self.frame_cursor = 0.0
        self.frame_idx = 0
        self.playing = True
        self.should_close = False
        self.show_wireframe = preset.show_wireframe
        self.show_gnn = False  # start with physics
        self.exported_frames = 0
        self.last_tick = time.perf_counter()
        self.preset_names = list(PRESETS.keys())
        self.preset_idx = self.preset_names.index(preset.name)

        # Taichi fields
        nv = sequences.n_verts
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=nv)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=nv)
        self.tri_idx = ti.field(dtype=ti.i32, shape=len(sequences.triangle_indices))
        self.line_idx = ti.field(dtype=ti.i32, shape=len(sequences.line_indices))
        self.tri_idx.from_numpy(sequences.triangle_indices)
        self.line_idx.from_numpy(sequences.line_indices)

        # Ground plane
        ground_y = float(sequences.y_min - preset.ground_margin)
        gv, gi = build_ground_plane(ground_y, extent=3.5)
        self.ground_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.ground_idx = ti.field(dtype=ti.i32, shape=6)
        self.ground_verts.from_numpy(gv)
        self.ground_idx.from_numpy(gi)

        # Window
        self.window = ti.ui.Window(
            name="ClothGNN Demo",
            res=preset.window_res,
            vsync=not export_only,
            show_window=not export_only,
            fps_limit=60,
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._configure_camera()
        self._push_frame(0)

    def _configure_camera(self):
        p = self.preset.camera
        self.camera.position(*p.position)
        self.camera.lookat(*p.lookat)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.fov(p.fov)

    def _get_frame_data(self, idx: int) -> np.ndarray:
        idx = max(0, min(idx, self.seq.n_frames - 1))
        return self.seq.gnn[idx] if self.show_gnn else self.seq.physics[idx]

    def _push_frame(self, idx: int):
        frame = self._get_frame_data(idx)
        self.vertices.from_numpy(frame.astype(np.float32))

        if self.preset.color_mode == "height":
            colors = height_colormap(frame[:, 1], self.seq.y_min, self.seq.y_max)
        elif self.preset.color_mode == "displacement":
            colors = displacement_colormap(frame, self.seq.rest_positions)
        else:
            colors = np.full((len(frame), 3), 0.7, dtype=np.float32)

        self.colors.from_numpy(colors)
        self.frame_idx = idx

    def _render_scene(self):
        p = self.preset
        self.canvas.set_background_color(p.background)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(p.light.ambient)

        # Key light
        kc = tuple(c * p.light.key_intensity for c in p.light.key_color)
        self.scene.point_light(pos=p.light.key_pos, color=kc)

        # Fill light
        fc = tuple(c * p.light.fill_intensity for c in p.light.fill_color)
        self.scene.point_light(pos=p.light.fill_pos, color=fc)

        # Ground
        self.scene.mesh(self.ground_verts, indices=self.ground_idx,
                        color=p.ground_color, two_sided=True)

        # Cloth mesh
        self.scene.mesh(self.vertices, indices=self.tri_idx,
                        per_vertex_color=self.colors, two_sided=True)

        # Wireframe
        if self.show_wireframe:
            self.scene.lines(self.vertices, width=p.wireframe_width,
                             indices=self.line_idx, color=p.wireframe_color)

        self.canvas.scene(self.scene)
        self._draw_gui()

    def _draw_gui(self):
        if not self.show_gui:
            return
        gui = self.window.get_gui()
        gui.begin("ClothGNN", 0.015, 0.015, 0.28, 0.18)
        mode_str = "GNN prediction" if self.show_gnn else "Physics baseline"
        gui.text(f"Mode: {mode_str}")
        gui.text(f"Preset: {self.preset.name}")
        gui.text(f"Frame: {self.frame_idx + 1}/{self.seq.n_frames}")
        gui.text(f"Wireframe: {'on' if self.show_wireframe else 'off'}")
        gui.text("[space] play  [r] reset  [m] mode")
        gui.text("[[ ]] preset  [w] wire")
        gui.end()

    def _handle_key(self, key):
        if key in (ti.ui.ESCAPE, ti.ui.EXIT):
            self.should_close = True
        elif key == ti.ui.SPACE:
            self.playing = not self.playing
            self.last_tick = time.perf_counter()
            print("play" if self.playing else "pause")
        elif key in ("r", "R"):
            self.frame_cursor = 0.0
            self._push_frame(0)
            self.last_tick = time.perf_counter()
            print("reset")
        elif key in ("m", "M"):
            self.show_gnn = not self.show_gnn
            self._push_frame(self.frame_idx)
            mode = "GNN" if self.show_gnn else "Physics"
            print(f"mode: {mode}")
        elif key in ("w", "W"):
            self.show_wireframe = not self.show_wireframe
            print(f"wireframe: {self.show_wireframe}")
        elif key in ("[", "-"):
            self.preset_idx = (self.preset_idx - 1) % len(self.preset_names)
            self._switch_preset(self.preset_names[self.preset_idx])
        elif key in ("]", "="):
            self.preset_idx = (self.preset_idx + 1) % len(self.preset_names)
            self._switch_preset(self.preset_names[self.preset_idx])

    def _switch_preset(self, name: str):
        self.preset = PRESETS[name]
        self.show_wireframe = self.preset.show_wireframe
        self._configure_camera()

        # Update ground plane position
        ground_y = float(self.seq.y_min - self.preset.ground_margin)
        gv, _ = build_ground_plane(ground_y, extent=3.5)
        self.ground_verts.from_numpy(gv)

        self._push_frame(self.frame_idx)
        print(f"preset: {name}")

    def _advance(self):
        if not self.playing or self.seq.n_frames <= 1:
            return
        now = time.perf_counter()
        dt = min(now - self.last_tick, 0.25)
        self.last_tick = now
        self.frame_cursor += self.preset.playback_fps * dt
        self.frame_cursor %= self.seq.n_frames
        idx = int(self.frame_cursor)
        if idx != self.frame_idx:
            self._push_frame(idx)

    def _export_frame(self):
        if self.export_dir is None:
            return
        path = self.export_dir / f"frame_{self.exported_frames:04d}.png"
        self.window.save_image(str(path))
        self.exported_frames += 1

    def run(self):
        if self.export_only:
            self.playing = False
            cursors = np.linspace(0, self.seq.n_frames - 1, self.seq.n_frames, dtype=np.float32)
            for cursor in cursors:
                self._push_frame(int(cursor))
                self._render_scene()
                self._export_frame()
                if self.exported_frames % 5 == 0:
                    print(f"  Exported frame {self.exported_frames}/{len(cursors)}")
            return

        while self.window.running and not self.should_close:
            while self.window.get_event(ti.ui.PRESS):
                self._handle_key(self.window.event.key)
            self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._advance()
            self._render_scene()
            self._export_frame()
            self.window.show()

        self.window.destroy()


# ════════════════════════════════════════════════════════════════
# Smoke test helper — renders single frames for QA
# ════════════════════════════════════════════════════════════════

def _render_and_save(seq: ClothSequences, preset: DemoPreset, frame_idx: int,
                     save_path: Path, show_wireframe: bool | None = None,
                     camera_override: CameraPose | None = None):
    """Create a fresh window, render one frame, save, destroy.

    Each call gets its own window to guarantee the Vulkan swapchain
    is flushed cleanly — reusing a window across camera changes produces
    stale images on headless backends.
    """
    cam_pose = camera_override or preset.camera
    wire = show_wireframe if show_wireframe is not None else preset.show_wireframe

    window = ti.ui.Window("smoke", res=preset.window_res, vsync=False,
                          show_window=False, fps_limit=60)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(*cam_pose.position)
    camera.lookat(*cam_pose.lookat)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(cam_pose.fov)

    nv = seq.n_verts
    verts = ti.Vector.field(3, dtype=ti.f32, shape=nv)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=nv)
    tri_idx = ti.field(dtype=ti.i32, shape=len(seq.triangle_indices))
    line_idx = ti.field(dtype=ti.i32, shape=len(seq.line_indices))
    tri_idx.from_numpy(seq.triangle_indices)
    line_idx.from_numpy(seq.line_indices)

    frame = seq.physics[min(frame_idx, seq.n_frames - 1)]
    verts.from_numpy(frame.astype(np.float32))
    col = height_colormap(frame[:, 1], seq.y_min, seq.y_max)
    colors.from_numpy(col)

    ground_y = float(seq.y_min - preset.ground_margin)
    gv, gi = build_ground_plane(ground_y, extent=3.5)
    gvf = ti.Vector.field(3, dtype=ti.f32, shape=4)
    gif = ti.field(dtype=ti.i32, shape=6)
    gvf.from_numpy(gv); gif.from_numpy(gi)

    canvas.set_background_color(preset.background)
    scene.set_camera(camera)
    scene.ambient_light(preset.light.ambient)
    kc = tuple(c * preset.light.key_intensity for c in preset.light.key_color)
    scene.point_light(pos=preset.light.key_pos, color=kc)
    fc = tuple(c * preset.light.fill_intensity for c in preset.light.fill_color)
    scene.point_light(pos=preset.light.fill_pos, color=fc)
    scene.mesh(gvf, indices=gif, color=preset.ground_color, two_sided=True)
    scene.mesh(verts, indices=tri_idx, per_vertex_color=colors, two_sided=True)
    if wire:
        scene.lines(verts, width=preset.wireframe_width,
                     indices=line_idx, color=preset.wireframe_color)
    canvas.scene(scene)

    window.save_image(str(save_path))
    window.destroy()
    print(f"Saved: {save_path}")


def run_smoke_test(outputs_dir: Path, artifacts_dir: Path, preset_name: str = "pitch",
                   frame_idx: int = -1, trustworthy_frames: int | None = None):
    """Render one frame from the given preset and save as smoke_test.png."""
    preset = PRESETS[preset_name]
    seq = ClothSequences(
        outputs_dir,
        trustworthy_frames=trustworthy_frames,
        height_scale=preset.height_scale,
        temporal_smooth=preset.temporal_smooth,
    )

    if frame_idx < 0:
        y_range = seq.physics[:, :, 1].max(axis=1) - seq.physics[:, :, 1].min(axis=1)
        frame_idx = int(np.argmax(y_range))
    frame_idx = min(frame_idx, seq.n_frames - 1)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Main smoke test
    _render_and_save(seq, preset, frame_idx, artifacts_dir / "smoke_test.png")

    # Camera angle tests
    camera_angles = [
        CameraPose(position=(0.0, 3.0, 1.5), lookat=(0.0, -0.3, 0.0), fov=30.0),
        CameraPose(position=(-1.5, 1.5, 2.0), lookat=(0.0, -0.3, 0.0), fov=28.0),
        CameraPose(position=(1.2, 1.0, 2.5), lookat=(0.0, -0.4, 0.0), fov=26.0),
    ]
    for i, cam in enumerate(camera_angles):
        _render_and_save(seq, preset, frame_idx,
                         artifacts_dir / f"camera_test_{i + 1}.png",
                         camera_override=cam)

    # Per-preset stills
    for pname, prst in PRESETS.items():
        _render_and_save(seq, prst, frame_idx,
                         artifacts_dir / f"preset_{pname}.png")

    print(f"\nAll smoke test images saved to {artifacts_dir}")
    return frame_idx


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ClothGNN Taichi GGUI Demo")
    p.add_argument("--outputs-dir", type=str, default="outputs",
                    help="Directory with precomputed sequences")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="pitch",
                    help="Render preset")
    p.add_argument("--mode", choices=["physics", "gnn"], default="physics",
                    help="Initial display mode")
    p.add_argument("--trustworthy-frames", type=int, default=None,
                    help="Override trustworthy rollout frame count")
    p.add_argument("--export", action="store_true",
                    help="Deterministic headless export")
    p.add_argument("--export-dir", type=str, default=None,
                    help="Export directory (default: artifacts/export_<preset>)")
    p.add_argument("--smoke-test", action="store_true",
                    help="Run smoke test QA (single frame renders)")
    p.add_argument("--smoke-frame", type=int, default=-1,
                    help="Frame index for smoke test (-1 = auto-pick)")
    p.add_argument("--arch", choices=["auto", "cuda", "vulkan", "cpu"], default="auto")
    p.add_argument("--no-gui", action="store_true",
                    help="Disable on-screen info panel")
    return p


def try_init_taichi(preferred: str) -> str:
    if ti is None:
        raise RuntimeError("Taichi is not installed.")
    order = {
        "auto": ["cuda", "vulkan", "cpu"],
        "cuda": ["cuda", "vulkan", "cpu"],
        "vulkan": ["vulkan", "cpu"],
        "cpu": ["cpu"],
    }[preferred]
    for arch_name in order:
        try:
            ti.reset()
            ti.init(arch=getattr(ti, arch_name), default_fp=ti.f32, offline_cache=True)
            return arch_name
        except Exception:
            continue
    raise RuntimeError("Failed to initialize Taichi.")


def main() -> int:
    args = build_parser().parse_args()
    outputs_dir = Path(args.outputs_dir)
    preset = PRESETS[args.preset]

    arch = try_init_taichi(args.arch)
    print(f"Taichi arch: {arch}")

    if args.smoke_test:
        artifacts_dir = Path("artifacts")
        run_smoke_test(outputs_dir, artifacts_dir, args.preset, args.smoke_frame,
                       args.trustworthy_frames)
        return 0

    seq = ClothSequences(
        outputs_dir,
        trustworthy_frames=args.trustworthy_frames,
        height_scale=preset.height_scale,
        temporal_smooth=preset.temporal_smooth,
    )

    export_dir = None
    if args.export:
        export_dir = Path(args.export_dir or f"artifacts/export_{args.preset}")
        export_dir.mkdir(parents=True, exist_ok=True)

    demo = TaichiClothDemo(
        sequences=seq,
        preset=preset,
        export_dir=export_dir,
        export_only=args.export,
        show_gui=not args.no_gui,
    )
    demo.show_gnn = (args.mode == "gnn")
    demo._push_frame(0)

    t0 = time.time()
    demo.run()
    elapsed = time.time() - t0

    if export_dir is not None:
        print(f"\nExported {demo.exported_frames} frames to {export_dir}")
        print(f"Elapsed: {elapsed:.1f}s")
        fps = int(preset.playback_fps)
        print(f"\nTo create MP4:")
        print(f"  ffmpeg -y -framerate {fps} -i {export_dir}/frame_%04d.png "
              f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p' "
              f"{export_dir}/cloth_demo.mp4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
