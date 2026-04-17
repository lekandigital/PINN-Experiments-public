#!/usr/bin/env python3
"""HGNN-NIF Cloth Viewer — Taichi GGUI

Renders the physics (PBS) baseline alongside the trained HGNN-NIF prediction
on a 40x40 cloth grid. Designed for pitch-quality visuals: dark studio
background, warm key + cool fill lighting, height-based diverging colormap.

Usage
-----
    python3.12 src/taichi_cloth_demo.py                 # default (pitch preset)
    python3.12 src/taichi_cloth_demo.py --preset research
    python3.12 src/taichi_cloth_demo.py --preset dramatic
    python3.12 src/taichi_cloth_demo.py --smoke         # one frame -> artifacts/
    python3.12 src/taichi_cloth_demo.py --camera-test   # 3 angles  -> artifacts/
    python3.12 src/taichi_cloth_demo.py --export        # all presets/modes

Controls
--------
    space       play / pause
    r           reset to first frame
    m           toggle PBS physics / HGNN-NIF prediction
    w           toggle wireframe
    [ / ]       cycle presets (research / pitch / dramatic)
    RMB drag    orbit camera
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import taichi as ti
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS_DIR = os.path.join(PROJECT_ROOT, "outputs", "physics_baseline")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "hgnn_nif_prediction")
MESH_INFO = os.path.join(PROJECT_ROOT, "outputs", "mesh_info.json")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraPose:
    position: Tuple[float, float, float]
    lookat: Tuple[float, float, float]
    fov: float


@dataclass
class LightConfig:
    key_pos: Tuple[float, float, float]
    key_color: Tuple[float, float, float]
    key_intensity: float
    fill_pos: Tuple[float, float, float]
    fill_color: Tuple[float, float, float]
    fill_intensity: float
    ambient: Tuple[float, float, float]
    rim_pos: Optional[Tuple[float, float, float]] = None
    rim_color: Optional[Tuple[float, float, float]] = None
    rim_intensity: float = 0.0


@dataclass
class DemoPreset:
    name: str
    camera: CameraPose
    light: LightConfig
    playback_fps: float
    height_scale: float
    color_mode: str            # "height_diverging" | "displacement" | "muted_stripes"
    show_wireframe: bool
    display_upsample: int      # 1=raw 40x40 mesh; >1 = Catmull-Rom presentation surface
    bg_color: Tuple[float, float, float]
    ground_color: Tuple[float, float, float]
    wire_color: Tuple[float, float, float]
    color_warm: Tuple[float, float, float]   # diverging warm (sag)
    color_cool: Tuple[float, float, float]   # diverging cool (high)
    color_neutral: Tuple[float, float, float]
    color_strength_power: float
    color_gain: float
    smooth_window: int = 1     # temporal smoothing (1 = off)


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────
PRESETS: Dict[str, DemoPreset] = {
    "research": DemoPreset(
        name="research",
        camera=CameraPose(position=(0.15, 0.9, 1.8), lookat=(0.0, -0.08, 0.0), fov=44.0),
        light=LightConfig(
            key_pos=(2.5, 3.5, 2.5),  key_color=(0.95, 0.92, 0.88),  key_intensity=0.85,
            fill_pos=(-2.5, 1.0, -1.5), fill_color=(0.55, 0.65, 0.85), fill_intensity=0.40,
            ambient=(0.20, 0.20, 0.22),
        ),
        playback_fps=12.0,
        height_scale=1.0,
        color_mode="height_diverging",
        show_wireframe=True,
        display_upsample=1,            # honest: raw 40x40 simulation mesh
        bg_color=(0.045, 0.050, 0.060),
        ground_color=(0.025, 0.028, 0.034),
        wire_color=(0.25, 0.30, 0.38),
        color_warm=(0.95, 0.50, 0.22),
        color_cool=(0.28, 0.58, 0.88),
        color_neutral=(0.55, 0.56, 0.60),
        color_strength_power=0.55,
        color_gain=1.8,
    ),
    "pitch": DemoPreset(
        name="pitch",
        camera=CameraPose(position=(0.0, 1.1, 1.7), lookat=(0.0, -0.10, 0.0), fov=44.0),
        light=LightConfig(
            key_pos=(2.2, 3.0, 2.3),   key_color=(1.05, 0.95, 0.82),  key_intensity=0.90,
            fill_pos=(-2.0, 0.6, -1.0), fill_color=(0.42, 0.58, 0.85), fill_intensity=0.45,
            ambient=(0.16, 0.17, 0.20),
            rim_pos=(0.0, 1.5, -2.5),  rim_color=(0.35, 0.40, 0.55),  rim_intensity=0.25,
        ),
        playback_fps=8.0,
        height_scale=1.15,
        color_mode="height_diverging",
        show_wireframe=False,
        display_upsample=4,            # Catmull-Rom presentation surface
        bg_color=(0.030, 0.035, 0.048),
        ground_color=(0.018, 0.020, 0.028),
        wire_color=(0.22, 0.25, 0.32),
        color_warm=(1.00, 0.50, 0.22),
        color_cool=(0.22, 0.55, 0.92),
        color_neutral=(0.92, 0.91, 0.92),
        color_strength_power=0.70,
        color_gain=1.9,
        smooth_window=3,
    ),
    "dramatic": DemoPreset(
        name="dramatic",
        camera=CameraPose(position=(1.55, 0.30, 1.35), lookat=(0.0, -0.05, 0.0), fov=44.0),
        light=LightConfig(
            key_pos=(2.6, 3.2, 1.8),   key_color=(1.20, 0.96, 0.74),  key_intensity=1.05,
            fill_pos=(-2.4, 0.4, -1.2), fill_color=(0.20, 0.34, 0.65), fill_intensity=0.38,
            ambient=(0.10, 0.10, 0.13),
            rim_pos=(0.0, 1.2, -2.2),  rim_color=(0.55, 0.30, 0.55),  rim_intensity=0.35,
        ),
        playback_fps=6.0,
        height_scale=1.30,
        color_mode="height_diverging",
        show_wireframe=False,
        display_upsample=5,            # Catmull-Rom presentation surface
        bg_color=(0.020, 0.020, 0.038),
        ground_color=(0.012, 0.012, 0.022),
        wire_color=(0.30, 0.20, 0.30),
        color_warm=(1.00, 0.45, 0.12),
        color_cool=(0.18, 0.50, 1.00),
        color_neutral=(0.96, 0.93, 0.94),
        color_strength_power=0.80,
        color_gain=2.4,
        smooth_window=3,
    ),
}
PRESET_ORDER = ["research", "pitch", "dramatic"]


# ─────────────────────────────────────────────────────────────────────────────
# Mesh helpers
# ─────────────────────────────────────────────────────────────────────────────
def grid_triangle_indices(rows: int, cols: int) -> np.ndarray:
    """Two triangles per quad. Returns (F, 3) int32."""
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v00 = i * cols + j
            v01 = i * cols + j + 1
            v10 = (i + 1) * cols + j
            v11 = (i + 1) * cols + j + 1
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))
    return np.asarray(faces, dtype=np.int32)


def grid_edge_indices(rows: int, cols: int) -> np.ndarray:
    edges = set()
    for i in range(rows):
        for j in range(cols):
            v = i * cols + j
            if j + 1 < cols:
                edges.add((v, i * cols + j + 1))
            if i + 1 < rows:
                edges.add((v, (i + 1) * cols + j))
    return np.asarray(sorted(edges), dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def shared_normalize(*sequences: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Physics sim is Z-up (gravity along -Z). Viewer uses Y-up. Swap Y<->Z so
    cloth hangs along -Y, then center on the physics mean and scale to a ~1.0
    span. The same transform is applied to every sequence for apples-to-apples
    rendering."""
    def swap_yz(seq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        out[..., [1, 2]] = out[..., [2, 1]]
        return out
    swapped = [swap_yz(seq) for seq in sequences]
    ref = swapped[0]                                  # (T, N, 3)
    flat = ref.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    # Geometric bbox center, so the cloth sits symmetric in the frame
    center = (lo + hi) * 0.5
    spans = hi - lo
    scale = 1.0 / max(spans.max(), 1e-6)
    out = [(seq - center) * scale for seq in swapped]
    return out, center, scale


def temporal_smooth(seq: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return seq
    T = seq.shape[0]
    half = window // 2
    sm = np.empty_like(seq)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        sm[t] = seq[lo:hi].mean(axis=0)
    return sm


def _gaussian_smooth_2d(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian on the spatial (row, col) axes of a (R, C, 3) grid.
    Small sigma (~0.8 cells) just removes the mesh-quantization staircase."""
    if sigma <= 0.0:
        return grid
    radius = max(1, int(np.ceil(3.0 * sigma)))
    ks = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (ks / sigma) ** 2)
    kernel /= kernel.sum()
    out = grid
    # Row convolution
    pad_r = np.pad(out, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    stacked = np.stack([pad_r[i : i + out.shape[0]] for i in range(len(kernel))], axis=0)
    out = (kernel[:, None, None, None] * stacked).sum(axis=0)
    # Col convolution
    pad_c = np.pad(out, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    stacked = np.stack([pad_c[:, i : i + out.shape[1]] for i in range(len(kernel))], axis=0)
    out = (kernel[:, None, None, None] * stacked).sum(axis=0)
    return out.astype(np.float32)


def catmull_rom_upsample(grid_pos: np.ndarray, rows: int, cols: int, factor: int,
                         method: str = "linear", post_blur: float = 0.8) -> np.ndarray:
    """Upsample a (rows, cols, 3) grid by `factor`, then apply a mild spatial
    Gaussian blur to remove triangulation staircases. `method` is linear by
    default (cubic overshoots on high-curvature folds, producing 'brick' artifacts).
    Returns (rows*factor-factor+1, cols*factor-factor+1, 3).
    """
    if factor <= 1:
        return grid_pos.reshape(rows, cols, 3)
    rr = np.arange(rows)
    cc = np.arange(cols)
    grid = grid_pos.reshape(rows, cols, 3)
    out_r = rows * factor - factor + 1
    out_c = cols * factor - factor + 1
    target_r = np.linspace(0, rows - 1, out_r)
    target_c = np.linspace(0, cols - 1, out_c)
    coords = np.stack(np.meshgrid(target_r, target_c, indexing="ij"), axis=-1)  # (R,C,2)
    out = np.empty((out_r, out_c, 3), dtype=np.float32)
    for k in range(3):
        fn = RegularGridInterpolator(
            (rr, cc), grid[..., k], method=method, bounds_error=False, fill_value=None,
        )
        out[..., k] = fn(coords).astype(np.float32)
    if post_blur > 0.0:
        out = _gaussian_smooth_2d(out, sigma=post_blur)
    return out


def upsample_sequence(seq: np.ndarray, rows: int, cols: int, factor: int):
    """Apply Catmull-Rom-ish cubic upsample to every frame. Returns
    (upsampled_positions, new_rows, new_cols).
    """
    T = seq.shape[0]
    up0 = catmull_rom_upsample(seq[0], rows, cols, factor)
    nr, nc = up0.shape[:2]
    up_seq = np.empty((T, nr * nc, 3), dtype=np.float32)
    up_seq[0] = up0.reshape(-1, 3)
    for t in range(1, T):
        up_seq[t] = catmull_rom_upsample(seq[t], rows, cols, factor).reshape(-1, 3)
    return up_seq, nr, nc


def diverging_colors(signed: np.ndarray, warm, cool, neutral, strength_power: float) -> np.ndarray:
    signed = np.clip(signed, -1.0, 1.0)
    strength = np.abs(signed) ** strength_power
    warm_a = np.asarray(warm, dtype=np.float32)
    cool_a = np.asarray(cool, dtype=np.float32)
    neutral_a = np.asarray(neutral, dtype=np.float32)
    target = np.where(signed[..., None] >= 0.0, warm_a, cool_a)
    colors = neutral_a + (target - neutral_a) * strength[..., None]
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def colorize_sequence(positions: np.ndarray, preset: DemoPreset) -> np.ndarray:
    """Per-vertex color using the preset's color_mode."""
    T, N, _ = positions.shape
    if preset.color_mode == "height_diverging":
        y = positions[..., 1]                                  # (T, N)
        y_med = np.median(y)
        y_dev = y - y_med
        ref = max(np.abs(y_dev).max(), 1e-6)
        signed = np.tanh(preset.color_gain * y_dev / ref)      # high -> +1 (cool), low -> -1 (warm)
        # Lower cloth should be warm: invert sign
        signed = -signed
        colors = diverging_colors(
            signed,
            preset.color_warm, preset.color_cool, preset.color_neutral,
            preset.color_strength_power,
        )
    elif preset.color_mode == "displacement":
        disp = np.linalg.norm(positions - positions[0:1], axis=-1)  # (T, N)
        ref = max(np.percentile(disp, 98), 1e-6)
        t = np.clip(disp / ref, 0.0, 1.0)
        warm = np.asarray(preset.color_warm, dtype=np.float32)
        neutral = np.asarray(preset.color_neutral, dtype=np.float32)
        colors = neutral + (warm - neutral) * (t ** preset.color_strength_power)[..., None]
        colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    else:  # muted_stripes
        # Row-striped muted alternation; kept honest (desaturated)
        rows_cols = int(np.sqrt(N))
        idx = (np.arange(N) // rows_cols) % 2
        base = np.where(idx[:, None] == 0, np.asarray(preset.color_warm), np.asarray(preset.color_cool))
        colors = np.broadcast_to(base.astype(np.float32), (T, N, 3)).copy()
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Viewer
# ─────────────────────────────────────────────────────────────────────────────
class ClothViewer:
    def __init__(
        self,
        preset_name: str = "pitch",
        resolution: Tuple[int, int] = (1280, 800),
        trust_only: bool = False,
    ):
        ti.init(arch=ti.metal)
        self.resolution = resolution
        self.preset_name = preset_name
        self.preset = PRESETS[preset_name]
        self.preset_idx = PRESET_ORDER.index(preset_name)

        # Load data & topology
        self.phys_raw = np.load(os.path.join(PHYS_DIR, "cloth_sequence.npy"))
        self.pred_raw = np.load(os.path.join(PRED_DIR, "cloth_sequence.npy"))
        with open(MESH_INFO) as f:
            self.info = json.load(f)
        self.rows = self.info["fine"]["resolution"]
        self.cols = self.rows

        # Truncate to trustworthy window if requested
        if trust_only:
            t_end = int(self.info["rollout"]["trustworthy_frame"])
            t_end = max(t_end, 30)  # floor so we always have something to play
            self.phys_raw = self.phys_raw[:t_end]
            self.pred_raw = self.pred_raw[:t_end]
            print(f"Trust-window playback: {t_end} frames")

        # Shared normalization (same transform for both sequences)
        seqs, self._center, self._scale = shared_normalize(self.phys_raw, self.pred_raw)
        self.phys_seq_raw = seqs[0].astype(np.float32)
        self.pred_seq_raw = seqs[1].astype(np.float32)
        self.n_timesteps = int(self.phys_seq_raw.shape[0])

        # Precompute per-preset display positions + colors + indices
        self._cache_for_preset()

        # State
        self.frame_idx = 0
        self.playing = True
        self.show_wireframe = self.preset.show_wireframe
        self.show_prediction = True    # True = HGNN-NIF, False = physics
        self.last_time = time.time()
        self.last_frame_time = time.time()

        # Ground plane: just below the lowest y the cloth reaches
        y_min = float(min(self.phys_pos[..., 1].min(), self.pred_pos[..., 1].min()))
        ground_y = y_min - 0.08
        self.ground_verts_np = self._make_ground(size=2.4, y=ground_y)
        self.ground_tris_np = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Taichi fields will be created after we know n_verts for current preset
        self._alloc_ti_fields()

    # ── preset-dependent caching ──────────────────────────────────────────
    def _cache_for_preset(self):
        preset = self.preset
        # Smoothing
        phys = temporal_smooth(self.phys_seq_raw, preset.smooth_window)
        pred = temporal_smooth(self.pred_seq_raw, preset.smooth_window)

        # Height scale (along world Y axis — cloth sags in negative Y)
        def scale_y(seq):
            out = seq.copy()
            out[..., 1] = out[..., 1] * preset.height_scale
            return out
        phys = scale_y(phys)
        pred = scale_y(pred)

        # Display upsample for pitch/dramatic
        if preset.display_upsample > 1:
            phys_disp, nr, nc = upsample_sequence(phys, self.rows, self.cols, preset.display_upsample)
            pred_disp, _, _ = upsample_sequence(pred, self.rows, self.cols, preset.display_upsample)
            self.disp_rows, self.disp_cols = nr, nc
        else:
            phys_disp, pred_disp = phys, pred
            self.disp_rows, self.disp_cols = self.rows, self.cols

        self.phys_pos = phys_disp.astype(np.float32)
        self.pred_pos = pred_disp.astype(np.float32)
        self.phys_col = colorize_sequence(self.phys_pos, preset)
        self.pred_col = colorize_sequence(self.pred_pos, preset)

        # Index buffers
        tri = grid_triangle_indices(self.disp_rows, self.disp_cols)
        edge = grid_edge_indices(self.disp_rows, self.disp_cols)
        self.tri_np = tri.astype(np.int32)
        self.edge_np = edge.astype(np.int32)
        self.n_verts = int(self.disp_rows * self.disp_cols)

    def _alloc_ti_fields(self):
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.tri_indices = ti.field(dtype=ti.i32, shape=self.tri_np.size)
        self.tri_indices.from_numpy(self.tri_np.flatten())
        self.edge_indices = ti.field(dtype=ti.i32, shape=self.edge_np.size)
        self.edge_indices.from_numpy(self.edge_np.flatten())
        self.ground_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.ground_verts.from_numpy(self.ground_verts_np)
        self.ground_tris = ti.field(dtype=ti.i32, shape=6)
        self.ground_tris.from_numpy(self.ground_tris_np.flatten())

    def _realloc_for_preset_switch(self):
        # Taichi 1.7 doesn't support resizing fields — recreate by resetting
        # the scope. We rely on fresh ti.init; cheapest is to skip realloc if
        # vertex count matches. If it differs, the caller must recreate the
        # viewer. For preset cycling we instead keep 1 field per preset size.
        pass

    def _make_ground(self, size: float, y: float):
        s = size
        return np.array([
            [-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s],
        ], dtype=np.float32)

    # ── frame update ──────────────────────────────────────────────────────
    def _update_frame(self):
        if self.show_prediction:
            pos = self.pred_pos[self.frame_idx]
            col = self.pred_col[self.frame_idx]
        else:
            pos = self.phys_pos[self.frame_idx]
            col = self.phys_col[self.frame_idx]
        self.vertices.from_numpy(pos)
        self.colors.from_numpy(col)

    # ── scene setup ───────────────────────────────────────────────────────
    def _configure_camera(self, camera):
        pose = self.preset.camera
        camera.position(*pose.position)
        camera.lookat(*pose.lookat)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(pose.fov)

    def _render_scene(self, scene, camera):
        scene.set_camera(camera)
        light = self.preset.light
        scene.ambient_light(light.ambient)
        scene.point_light(
            pos=light.key_pos,
            color=tuple(c * light.key_intensity for c in light.key_color),
        )
        scene.point_light(
            pos=light.fill_pos,
            color=tuple(c * light.fill_intensity for c in light.fill_color),
        )
        if light.rim_pos is not None:
            scene.point_light(
                pos=light.rim_pos,
                color=tuple(c * light.rim_intensity for c in light.rim_color),
            )

        # Ground
        scene.mesh(self.ground_verts, indices=self.ground_tris,
                   color=self.preset.ground_color, two_sided=True)

        # Cloth
        scene.mesh(self.vertices, indices=self.tri_indices,
                   per_vertex_color=self.colors, two_sided=True)

        if self.show_wireframe:
            scene.lines(self.vertices, width=1.0,
                        indices=self.edge_indices, color=self.preset.wire_color)

    # ── interactive run ───────────────────────────────────────────────────
    def run(self):
        window = ti.ui.Window(
            "HGNN-NIF Cloth Viewer", self.resolution, vsync=True,
        )
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        self._configure_camera(camera)

        self._print_mode_banner()

        while window.running:
            now = time.time()
            self.last_time = now

            if window.get_event(ti.ui.PRESS):
                key = window.event.key
                if key == ti.ui.SPACE:
                    self.playing = not self.playing
                    print(f"[{'PLAY' if self.playing else 'PAUSE'}] frame={self.frame_idx}")
                elif key == "r":
                    self.frame_idx = 0
                    print("[reset]")
                elif key == "m":
                    self.show_prediction = not self.show_prediction
                    self._print_mode_banner()
                elif key == "w":
                    self.show_wireframe = not self.show_wireframe
                elif key in (']', '['):
                    direction = 1 if key == ']' else -1
                    self._try_switch_preset(direction)
                    self._configure_camera(camera)

            if self.playing:
                frame_dt = now - self.last_frame_time
                if frame_dt >= 1.0 / self.preset.playback_fps:
                    self.frame_idx = (self.frame_idx + 1) % self.n_timesteps
                    self.last_frame_time = now
                    mode = "HGNN-NIF" if self.show_prediction else "PBS"
                    print(f"\r[{mode}] frame {self.frame_idx:4d}/{self.n_timesteps}  "
                          f"preset={self.preset.name}", end="", flush=True)

            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._update_frame()
            canvas.set_background_color(self.preset.bg_color)
            self._render_scene(scene, camera)
            canvas.scene(scene)
            window.show()

    def _print_mode_banner(self):
        mode = "HGNN-NIF prediction" if self.show_prediction else "PBS physics (baseline)"
        note = ""
        if self.preset.display_upsample > 1:
            note = f"  [display surface: Catmull-Rom x{self.preset.display_upsample}]"
        print(f"\n=== {mode}  | preset={self.preset.name}{note} ===")
        print(f"    space=play/pause  r=reset  m=toggle PBS/HGNN  w=wireframe  [/]=preset")

    def _try_switch_preset(self, direction: int):
        new_idx = (self.preset_idx + direction) % len(PRESET_ORDER)
        new_name = PRESET_ORDER[new_idx]
        new_preset = PRESETS[new_name]
        # If vertex count changes (different upsample factor) we must fully
        # recreate the viewer — Taichi fields can't be resized mid-run.
        def n_for(p):
            f = p.display_upsample
            nr = self.rows * f - f + 1 if f > 1 else self.rows
            nc = self.cols * f - f + 1 if f > 1 else self.cols
            return nr * nc
        if n_for(new_preset) != self.n_verts:
            print(f"\n[preset] {new_name} requires a different mesh size "
                  f"({n_for(new_preset)} vs {self.n_verts}). "
                  f"Restart with --preset {new_name}.")
            return
        self.preset_idx = new_idx
        self.preset_name = new_name
        self.preset = new_preset
        self.show_wireframe = new_preset.show_wireframe
        self._cache_for_preset()
        self._print_mode_banner()

    # ── headless renderings ────────────────────────────────────────────────
    def render_single(self, out_path: str, frame: int = 30,
                      show_prediction: bool = True,
                      camera_override: Optional[CameraPose] = None):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.frame_idx = min(max(frame, 0), self.n_timesteps - 1)
        self.show_prediction = show_prediction

        window = ti.ui.Window("headless", self.resolution, show_window=False)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        if camera_override is not None:
            camera.position(*camera_override.position)
            camera.lookat(*camera_override.lookat)
            camera.up(0.0, 1.0, 0.0)
            camera.fov(camera_override.fov)
        else:
            self._configure_camera(camera)

        self._update_frame()
        canvas.set_background_color(self.preset.bg_color)
        self._render_scene(scene, camera)
        canvas.scene(scene)
        window.save_image(out_path)
        window.destroy()
        print(f"saved: {out_path}")

    def render_camera_test(self, out_dir: str = ARTIFACT_DIR, frame: int = 45):
        os.makedirs(out_dir, exist_ok=True)
        angles = {
            1: CameraPose(position=(0.35, 0.55, 2.1),  lookat=(0.0, -0.03, 0.0), fov=40.0),   # frontal-right
            2: CameraPose(position=(1.6, 0.35, 1.4),   lookat=(0.0, -0.05, 0.0), fov=42.0),   # 3/4 low
            3: CameraPose(position=(0.0, 1.1, 1.7),    lookat=(0.0, -0.1, 0.0),  fov=44.0),   # high hero
        }
        for k, cam in angles.items():
            self.render_single(
                os.path.join(out_dir, f"camera_test_{k}.png"),
                frame=frame, show_prediction=True, camera_override=cam,
            )

    def render_preset_stills(self, out_dir: str = ARTIFACT_DIR, frame: int = 45):
        os.makedirs(out_dir, exist_ok=True)
        for name in PRESET_ORDER:
            v = ClothViewer(preset_name=name, resolution=self.resolution)
            v.render_single(os.path.join(out_dir, f"{name}_hgnn.png"),
                            frame=frame, show_prediction=True)
            v.render_single(os.path.join(out_dir, f"{name}_physics.png"),
                            frame=frame, show_prediction=False)

    def render_animation(self, out_dir: str, show_prediction: bool = True,
                         n_frames: Optional[int] = None, chunk: int = 40,
                         start_frame: int = 0):
        """Export PNGs (chunked to avoid Metal swapchain ceiling)."""
        os.makedirs(out_dir, exist_ok=True)
        last = self.n_timesteps if n_frames is None else min(start_frame + n_frames, self.n_timesteps)
        self.show_prediction = show_prediction
        i = start_frame
        while i < last:
            end = min(i + chunk, last)
            window = ti.ui.Window(
                f"anim_{os.path.basename(out_dir)}_{i}", self.resolution, show_window=False,
            )
            canvas = window.get_canvas()
            scene = window.get_scene()
            camera = ti.ui.Camera()
            self._configure_camera(camera)
            for f in range(i, end):
                self.frame_idx = f
                self._update_frame()
                canvas.set_background_color(self.preset.bg_color)
                self._render_scene(scene, camera)
                canvas.scene(scene)
                window.save_image(os.path.join(out_dir, f"f_{f - start_frame:04d}.png"))
            window.destroy()
            i = end
        print(f"rendered {last - start_frame} frames -> {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESET_ORDER, default="pitch")
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 800))
    parser.add_argument("--trust-only", action="store_true",
                        help="truncate sequences at the trustworthy frame")

    parser.add_argument("--smoke", action="store_true",
                        help="render artifacts/smoke_test.png and exit")
    parser.add_argument("--camera-test", action="store_true",
                        help="render 3 camera angles to artifacts/")
    parser.add_argument("--preset-stills", action="store_true",
                        help="render a still per preset / mode")
    parser.add_argument("--export", action="store_true",
                        help="headless PNG export of all presets")
    parser.add_argument("--export-preset", choices=PRESET_ORDER,
                        help="render one preset to artifacts/{preset}_{mode}/")
    parser.add_argument("--frame", type=int, default=45,
                        help="frame index for single-frame renders")
    parser.add_argument("--mode", choices=("hgnn", "physics"), default="hgnn",
                        help="hgnn (prediction) or physics (baseline) for --smoke")
    parser.add_argument("--out", type=str, default=None,
                        help="output path for --smoke (defaults to artifacts/smoke_test.png)")
    parser.add_argument("--anim-frames", type=int, default=120,
                        help="frame count for --export / --export-preset (default 120)")
    parser.add_argument("--anim-start", type=int, default=0,
                        help="starting frame for animation export (default 0)")
    parser.add_argument("--anim-out", type=str, default=None,
                        help="override output directory for animation export; use with --anim-mode")
    parser.add_argument("--anim-mode", choices=("hgnn", "physics", "both"), default="both",
                        help="which rollout to export when --anim-out is set")
    args = parser.parse_args()

    res = tuple(args.resolution)

    if args.smoke:
        v = ClothViewer(preset_name=args.preset, resolution=res, trust_only=args.trust_only)
        out = args.out or os.path.join(ARTIFACT_DIR, "smoke_test.png")
        v.render_single(out, frame=args.frame, show_prediction=(args.mode == "hgnn"))
        return
    if args.camera_test:
        v = ClothViewer(preset_name=args.preset, resolution=res, trust_only=args.trust_only)
        v.render_camera_test(frame=args.frame)
        return
    if args.preset_stills:
        v = ClothViewer(preset_name=args.preset, resolution=res, trust_only=args.trust_only)
        v.render_preset_stills(frame=args.frame)
        return
    if args.export_preset:
        v = ClothViewer(preset_name=args.export_preset, resolution=res,
                        trust_only=args.trust_only)
        if args.anim_out is not None:
            # Override output dir: render one or both modes into it
            if args.anim_mode in ("hgnn", "both"):
                v.render_animation(args.anim_out if args.anim_mode == "hgnn"
                                   else os.path.join(args.anim_out, "hgnn"),
                                   show_prediction=True, n_frames=args.anim_frames,
                                   start_frame=args.anim_start)
            if args.anim_mode in ("physics", "both"):
                v.render_animation(args.anim_out if args.anim_mode == "physics"
                                   else os.path.join(args.anim_out, "physics"),
                                   show_prediction=False, n_frames=args.anim_frames,
                                   start_frame=args.anim_start)
        else:
            out_pred = os.path.join(ARTIFACT_DIR, f"{args.export_preset}_hgnn")
            out_phys = os.path.join(ARTIFACT_DIR, f"{args.export_preset}_physics")
            v.render_animation(out_pred, show_prediction=True, n_frames=args.anim_frames,
                               start_frame=args.anim_start)
            v.render_animation(out_phys, show_prediction=False, n_frames=args.anim_frames,
                               start_frame=args.anim_start)
        return
    if args.export:
        for name in PRESET_ORDER:
            v = ClothViewer(preset_name=name, resolution=res, trust_only=args.trust_only)
            v.render_animation(os.path.join(ARTIFACT_DIR, f"{name}_hgnn"),
                               show_prediction=True, n_frames=args.anim_frames)
            v.render_animation(os.path.join(ARTIFACT_DIR, f"{name}_physics"),
                               show_prediction=False, n_frames=args.anim_frames)
        return

    v = ClothViewer(preset_name=args.preset, resolution=res, trust_only=args.trust_only)
    v.run()


if __name__ == "__main__":
    main()
