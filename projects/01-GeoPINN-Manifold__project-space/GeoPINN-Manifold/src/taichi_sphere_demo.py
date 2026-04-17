#!/usr/bin/env python3
"""
GeoPINN Sphere Viewer — Taichi GGUI

Renders the learned scalar field (heat equation on unit sphere) as a
rotating, radially-deformed, colormap-shaded globe.

Usage:
    python3.12 src/taichi_sphere_demo.py                    # default (pitch)
    python3.12 src/taichi_sphere_demo.py --preset research
    python3.12 src/taichi_sphere_demo.py --preset dramatic
    python3.12 src/taichi_sphere_demo.py --export           # headless PNG export

Controls:
    Space   play / pause animation
    r       reset to t=0
    m       toggle GeoPINN / analytical
    w       toggle wireframe
    [  ]    cycle presets
    RMB     orbit camera
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import taichi as ti
from scipy.spatial import KDTree


# ── project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "geopinn_prediction")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "outputs", "analytical_baseline")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CameraPose:
    position: Tuple[float, float, float]
    lookat: Tuple[float, float, float]
    fov: float


@dataclass
class Light:
    pos: Tuple[float, float, float]
    color: Tuple[float, float, float]


@dataclass
class LightConfig:
    ambient: Tuple[float, float, float]
    lights: List[Light] = field(default_factory=list)


@dataclass
class SphereVisualConfig:
    deformation_amplitude: float
    color_gain: float
    color_percentile: float
    rotation_speed: float
    show_wireframe: bool
    # Colormap palette
    color_positive: Tuple[float, float, float]  # warm / red
    color_negative: Tuple[float, float, float]  # cool / blue
    color_neutral: Tuple[float, float, float]   # mid-point
    strength_power: float = 0.75


@dataclass
class DemoPreset:
    name: str
    camera: CameraPose
    lighting: LightConfig
    visual: SphereVisualConfig
    bg_color: Tuple[float, float, float]
    wireframe_color: Tuple[float, float, float] = (0.25, 0.30, 0.35)
    wireframe_width: float = 1.0
    playback_fps: float = 6.0


# ── Presets ────────────────────────────────────────────────────────────────
PRESETS = {
    "research": DemoPreset(
        name="research",
        camera=CameraPose(
            position=(0.0, 0.0, 2.8),
            lookat=(0.0, 0.0, 0.0),
            fov=50.0,
        ),
        lighting=LightConfig(
            ambient=(0.28, 0.28, 0.30),
            lights=[
                Light(pos=(2.5, 2.0, 2.0), color=(0.95, 0.92, 0.88)),
                Light(pos=(-2.0, -1.0, 1.5), color=(0.30, 0.38, 0.50)),
            ],
        ),
        visual=SphereVisualConfig(
            deformation_amplitude=0.05,
            color_gain=2.2,
            color_percentile=99.0,
            rotation_speed=0.22,
            show_wireframe=True,
            color_positive=(0.90, 0.28, 0.18),
            color_negative=(0.15, 0.42, 0.88),
            color_neutral=(0.88, 0.90, 0.93),
            strength_power=0.65,
        ),
        bg_color=(0.05, 0.06, 0.08),
        wireframe_color=(0.18, 0.22, 0.28),
        wireframe_width=1.0,
        playback_fps=4.0,
    ),
    "pitch": DemoPreset(
        name="pitch",
        camera=CameraPose(
            position=(0.0, 0.5, 2.5),
            lookat=(0.0, 0.0, 0.0),
            fov=45.0,
        ),
        lighting=LightConfig(
            ambient=(0.18, 0.19, 0.22),
            lights=[
                Light(pos=(2.5, 2.8, 2.0), color=(1.05, 0.98, 0.88)),   # warm key
                Light(pos=(-2.0, -0.5, 1.5), color=(0.22, 0.34, 0.55)), # cool fill
                Light(pos=(0.0, 1.0, -2.5), color=(0.08, 0.10, 0.14)),  # rim
            ],
        ),
        visual=SphereVisualConfig(
            deformation_amplitude=0.06,
            color_gain=2.4,
            color_percentile=99.0,
            rotation_speed=0.32,
            show_wireframe=False,
            color_positive=(1.00, 0.48, 0.20),
            color_negative=(0.18, 0.48, 0.95),
            color_neutral=(0.94, 0.93, 0.92),
            strength_power=0.70,
        ),
        bg_color=(0.02, 0.025, 0.05),
        playback_fps=6.0,
    ),
    "dramatic": DemoPreset(
        name="dramatic",
        camera=CameraPose(
            position=(0.3, 0.6, 2.0),
            lookat=(0.0, 0.0, 0.0),
            fov=55.0,
        ),
        lighting=LightConfig(
            ambient=(0.10, 0.10, 0.14),
            lights=[
                Light(pos=(3.0, 3.0, 1.5), color=(1.20, 0.98, 0.78)),   # strong warm key
                Light(pos=(-2.5, -1.5, 2.0), color=(0.14, 0.24, 0.48)), # dim cool fill
            ],
        ),
        visual=SphereVisualConfig(
            deformation_amplitude=0.09,
            color_gain=3.0,
            color_percentile=98.5,
            rotation_speed=0.50,
            show_wireframe=False,
            color_positive=(1.00, 0.42, 0.10),
            color_negative=(0.12, 0.38, 1.00),
            color_neutral=(0.90, 0.90, 0.95),
            strength_power=0.80,
        ),
        bg_color=(0.015, 0.015, 0.04),
        playback_fps=8.0,
    ),
}

PRESET_ORDER = ["research", "pitch", "dramatic"]


# ═══════════════════════════════════════════════════════════════════════════
# Icosphere mesh generation
# ═══════════════════════════════════════════════════════════════════════════
def _make_icosphere(subdivisions: int = 4):
    """
    Build an icosphere by subdividing an icosahedron.

    Returns (vertices [V,3], faces [F,3]).
    Subdivisions: 3 → 642 verts / 1280 tris
                  4 → 2562 verts / 5120 tris
                  5 → 10242 verts / 20480 tris
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # 12 vertices of icosahedron
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)

    # Subdivide
    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        verts_list = list(verts)

        def _get_midpoint(i, j, vl, em):
            key = (min(i, j), max(i, j))
            if key in em:
                return em[key]
            mid = (vl[i] + vl[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(vl)
            vl.append(mid)
            em[key] = idx
            return idx

        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = _get_midpoint(a, b, verts_list, edge_midpoints)
            bc = _get_midpoint(b, c, verts_list, edge_midpoints)
            ca = _get_midpoint(c, a, verts_list, edge_midpoints)
            new_faces.extend([
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            ])
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int32)

    # Normalize all to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = (verts / norms).astype(np.float32)

    return verts, faces


def _build_line_indices(faces: np.ndarray) -> np.ndarray:
    """Extract unique edge pairs from triangle faces for wireframe."""
    edge_set = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# Field preprocessing (colormap + deformation)
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_fields(
    raw_fields: np.ndarray,        # [T, N_data]
    data_coords: np.ndarray,       # [N_data, 3]
    mesh_verts: np.ndarray,        # [V, 3]
    visual: SphereVisualConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate, normalise, apply tanh transfer, generate colors and
    deformed vertex positions for every timestep.

    Returns:
        all_positions: [T, V, 3]  deformed vertex xyz
        all_colors:    [T, V, 3]  per-vertex RGB
    """
    T, N_data = raw_fields.shape
    V = mesh_verts.shape[0]

    # KDTree interpolation (nearest-neighbor on sphere surface)
    tree = KDTree(data_coords)
    _, nn_idx = tree.query(mesh_verts, k=3)  # 3 nearest neighbors
    nn_dist = np.linalg.norm(
        data_coords[nn_idx] - mesh_verts[:, None, :], axis=2
    )
    # Inverse-distance weights
    weights = 1.0 / (nn_dist + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)

    # Interpolate all timesteps
    interp_fields = np.zeros((T, V), dtype=np.float32)
    for t in range(T):
        vals = raw_fields[t][nn_idx]  # [V, 3]
        interp_fields[t] = (vals * weights).sum(axis=1)

    # Global normalization reference (percentile across ALL timesteps)
    abs_vals = np.abs(interp_fields)
    pct = np.percentile(abs_vals, visual.color_percentile)
    pct = max(pct, 1e-6)

    # Per-frame: center, normalize, tanh transfer
    all_positions = np.zeros((T, V, 3), dtype=np.float32)
    all_colors = np.zeros((T, V, 3), dtype=np.float32)

    for t in range(T):
        fld = interp_fields[t]
        centered = fld - fld.mean()
        normalized = centered / pct
        display = np.tanh(normalized * visual.color_gain)  # in [-1, 1]

        # Colormap: diverging
        colors = _diverging_colormap(
            display,
            visual.color_positive,
            visual.color_negative,
            visual.color_neutral,
            visual.strength_power,
        )
        all_colors[t] = colors

        # Radial deformation
        r = 1.0 + visual.deformation_amplitude * display
        all_positions[t] = mesh_verts * r[:, None]

    return all_positions, all_colors


def _diverging_colormap(
    signed: np.ndarray,
    warm: Tuple[float, float, float],
    cool: Tuple[float, float, float],
    neutral: Tuple[float, float, float],
    strength_power: float,
) -> np.ndarray:
    """Map signed values in [-1, 1] to RGB via diverging colormap."""
    signed = np.clip(signed, -1.0, 1.0)
    strength = np.abs(signed) ** strength_power

    warm_a = np.array(warm, dtype=np.float32)
    cool_a = np.array(cool, dtype=np.float32)
    neutral_a = np.array(neutral, dtype=np.float32)

    target = np.where(signed[..., None] >= 0.0, warm_a, cool_a)
    colors = neutral_a + (target - neutral_a) * strength[..., None]
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Viewer
# ═══════════════════════════════════════════════════════════════════════════
class SphereViewer:
    """Interactive Taichi GGUI viewer for the GeoPINN sphere field."""

    def __init__(
        self,
        preset_name: str = "pitch",
        export_mode: bool = False,
        resolution: Tuple[int, int] = (1280, 960),
    ):
        ti.init(arch=ti.metal)

        self.export_mode = export_mode
        self.resolution = resolution

        # Load field data
        self._load_data()

        # Build icosphere mesh (subdivision 5 → ~10K verts)
        self.base_verts, self.faces = _make_icosphere(subdivisions=5)
        self.line_indices_np = _build_line_indices(self.faces)
        self.n_verts = self.base_verts.shape[0]
        self.n_faces = self.faces.shape[0]
        self.n_edges = self.line_indices_np.shape[0]
        print(f"Icosphere: {self.n_verts} vertices, {self.n_faces} triangles, "
              f"{self.n_edges} edges")

        # Preprocess both datasets
        self.preset_idx = PRESET_ORDER.index(preset_name)
        self.preset = PRESETS[preset_name]
        self._preprocess_all()

        # Taichi fields
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.tri_indices = ti.field(dtype=ti.i32, shape=self.n_faces * 3)
        self.tri_indices.from_numpy(self.faces.flatten())
        self.edge_indices = ti.field(dtype=ti.i32, shape=self.n_edges * 2)
        self.edge_indices.from_numpy(self.line_indices_np.flatten())

        # State
        self.frame_idx = 0
        self.playing = True
        self.show_wireframe = self.preset.visual.show_wireframe
        self.show_analytical = False  # False = PINN, True = analytical
        self.rotation_angle = 0.0
        self.last_time = time.time()
        self.last_frame_time = time.time()

    def _load_data(self):
        """Load PINN predictions and analytical baseline."""
        self.pred_fields = np.load(os.path.join(PRED_DIR, "field_values.npy"))
        self.data_coords = np.load(os.path.join(PRED_DIR, "sphere_coords.npy"))
        self.timesteps = np.load(os.path.join(PRED_DIR, "timesteps.npy"))
        self.anal_fields = np.load(os.path.join(BASELINE_DIR, "field_values.npy"))
        self.n_timesteps = len(self.timesteps)
        print(f"Loaded {self.n_timesteps} timesteps, "
              f"{self.pred_fields.shape[1]} data points")

    def _preprocess_all(self):
        """Preprocess fields for current preset."""
        vis = self.preset.visual
        self.pinn_positions, self.pinn_colors = preprocess_fields(
            self.pred_fields, self.data_coords, self.base_verts, vis,
        )
        self.anal_positions, self.anal_colors = preprocess_fields(
            self.anal_fields, self.data_coords, self.base_verts, vis,
        )
        print(f"Preprocessed frames: PINN {self.pinn_positions.shape}, "
              f"analytical {self.anal_positions.shape}")

    def _apply_rotation(self, positions: np.ndarray) -> np.ndarray:
        """Rotate positions around Y axis by self.rotation_angle."""
        c = np.cos(self.rotation_angle)
        s = np.sin(self.rotation_angle)
        rot = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ], dtype=np.float32)
        return positions @ rot.T

    def _update_frame(self):
        """Push current frame data into Taichi fields."""
        if self.show_analytical:
            pos = self.anal_positions[self.frame_idx]
            col = self.anal_colors[self.frame_idx]
        else:
            pos = self.pinn_positions[self.frame_idx]
            col = self.pinn_colors[self.frame_idx]

        rotated = self._apply_rotation(pos)
        self.vertices.from_numpy(rotated)
        self.colors.from_numpy(col)

    def _configure_camera(self, camera):
        pose = self.preset.camera
        camera.position(*pose.position)
        camera.lookat(*pose.lookat)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(pose.fov)

    def _switch_preset(self, direction: int):
        self.preset_idx = (self.preset_idx + direction) % len(PRESET_ORDER)
        name = PRESET_ORDER[self.preset_idx]
        self.preset = PRESETS[name]
        self.show_wireframe = self.preset.visual.show_wireframe
        self._preprocess_all()
        print(f"Preset: {name}")

    def _render_scene(self, scene, camera):
        scene.set_camera(camera)
        scene.ambient_light(self.preset.lighting.ambient)
        for light in self.preset.lighting.lights:
            scene.point_light(pos=light.pos, color=light.color)

        scene.mesh(
            self.vertices,
            indices=self.tri_indices,
            per_vertex_color=self.colors,
            two_sided=True,
        )

        if self.show_wireframe:
            scene.lines(
                self.vertices,
                width=self.preset.wireframe_width,
                indices=self.edge_indices,
                color=self.preset.wireframe_color,
            )

    # ── public entry points ────────────────────────────────────────────────
    def run(self):
        """Launch interactive GGUI window."""
        window = ti.ui.Window(
            "GeoPINN — Sphere Field Viewer",
            self.resolution,
            vsync=True,
        )
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        self._configure_camera(camera)

        while window.running:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            # Input
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.SPACE:
                    self.playing = not self.playing
                elif window.event.key == "r":
                    self.frame_idx = 0
                    self.rotation_angle = 0.0
                elif window.event.key == "m":
                    self.show_analytical = not self.show_analytical
                    label = "analytical" if self.show_analytical else "GeoPINN"
                    print(f"Mode: {label}")
                elif window.event.key == "w":
                    self.show_wireframe = not self.show_wireframe
                elif window.event.key == "]":
                    self._switch_preset(1)
                    self._configure_camera(camera)
                elif window.event.key == "[":
                    self._switch_preset(-1)
                    self._configure_camera(camera)

            # Rotation
            self.rotation_angle += self.preset.visual.rotation_speed * dt

            # Advance frame
            if self.playing:
                frame_dt = now - self.last_frame_time
                if frame_dt >= 1.0 / self.preset.playback_fps:
                    self.frame_idx = (self.frame_idx + 1) % self.n_timesteps
                    self.last_frame_time = now

            camera.track_user_inputs(
                window, movement_speed=0.03, hold_key=ti.ui.RMB
            )

            self._update_frame()
            canvas.set_background_color(self.preset.bg_color)
            self._render_scene(scene, camera)
            canvas.scene(scene)
            window.show()

    def render_video_sequence(
        self,
        preset_name: str,
        out_dir: str,
        n_frames: int = 240,
        fps: int = 30,
        mode: str = "pinn",  # 'pinn' or 'analytical'
        resolution: Tuple[int, int] | None = None,
        chunk_size: int = 40,
    ) -> str:
        """
        Render a continuous rotation + field evolution sequence as PNGs.

        One full rotation (2pi) across `n_frames`. Field evolution loops
        through the 12 timesteps (ping-pong forward/backward for smoothness).

        The Taichi Metal swap chain caps at 50 images per window session, so
        we recreate the window every `chunk_size` frames.

        Returns the output directory with PNG frames.
        """
        res = resolution if resolution is not None else self.resolution

        self.preset = PRESETS[preset_name]
        self.preset_idx = PRESET_ORDER.index(preset_name)
        self.show_wireframe = self.preset.visual.show_wireframe
        self.show_analytical = (mode == "analytical")
        self._preprocess_all()

        os.makedirs(out_dir, exist_ok=True)

        T = self.n_timesteps
        ping_pong_len = 2 * (T - 1)  # 0,1,..,T-1,T-2,..,1
        cycle_frames = max(1, n_frames // 2)

        i = 0
        while i < n_frames:
            chunk_end = min(i + chunk_size, n_frames)

            window = ti.ui.Window(
                f"vid_{preset_name}_{mode}_{i}", res, show_window=False,
            )
            canvas = window.get_canvas()
            scene = window.get_scene()
            camera = ti.ui.Camera()
            self._configure_camera(camera)

            for j in range(i, chunk_end):
                self.rotation_angle = 2.0 * np.pi * j / n_frames

                cycle_pos = (j % cycle_frames) / cycle_frames  # [0, 1)
                pp_idx = int(cycle_pos * ping_pong_len)
                if pp_idx < T:
                    self.frame_idx = pp_idx
                else:
                    self.frame_idx = 2 * (T - 1) - pp_idx

                self._update_frame()
                canvas.set_background_color(self.preset.bg_color)
                self._render_scene(scene, camera)
                canvas.scene(scene)

                path = os.path.join(out_dir, f"f_{j:04d}.png")
                window.save_image(path)

            window.destroy()
            i = chunk_end

        print(f"Rendered {n_frames} frames @ {res} → {out_dir}/")
        return out_dir

    def render_poster(self, out_path: str, resolution: Tuple[int, int] = (1920, 1440)):
        """Render a single hero frame suitable for a poster image."""
        window = ti.ui.Window("poster", resolution, show_window=False)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()

        self.preset = PRESETS["pitch"]
        self.show_wireframe = False
        self.show_analytical = False
        self._preprocess_all()

        # Nice oblique angle with interesting field state (t=0 has max amplitude)
        camera.position(0.3, 0.9, 2.3)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(42.0)

        self.frame_idx = 0
        self.rotation_angle = 0.7

        self._update_frame()
        canvas.set_background_color(self.preset.bg_color)
        self._render_scene(scene, camera)
        canvas.scene(scene)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        window.save_image(out_path)
        window.destroy()
        print(f"Poster saved: {out_path}")

    def render_preset_stills(self, out_dir: str, resolution: Tuple[int, int] = (1280, 960)):
        """Render one hero still per preset."""
        os.makedirs(out_dir, exist_ok=True)
        window = ti.ui.Window("stills", resolution, show_window=False)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()

        for pname in PRESET_ORDER:
            self.preset = PRESETS[pname]
            self.show_wireframe = self.preset.visual.show_wireframe
            self._preprocess_all()
            self._configure_camera(camera)
            self.frame_idx = 0
            # Mid-rotation for each so they look different
            rot_map = {"research": 0.0, "pitch": 0.7, "dramatic": 1.2}
            self.rotation_angle = rot_map[pname]
            self._update_frame()
            canvas.set_background_color(self.preset.bg_color)
            self._render_scene(scene, camera)
            canvas.scene(scene)
            path = os.path.join(out_dir, f"{pname}.png")
            window.save_image(path)
            print(f"Saved preset still: {path}")
        window.destroy()

    def export_frames(self, out_dir: str | None = None):
        """Headless export of all frames as PNGs for each preset."""
        if out_dir is None:
            out_dir = ARTIFACT_DIR
        os.makedirs(out_dir, exist_ok=True)

        window = ti.ui.Window(
            "export", self.resolution, show_window=False,
        )
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()

        for preset_name in PRESET_ORDER:
            self.preset = PRESETS[preset_name]
            self.preset_idx = PRESET_ORDER.index(preset_name)
            self.show_wireframe = self.preset.visual.show_wireframe
            self._preprocess_all()
            self._configure_camera(camera)

            preset_dir = os.path.join(out_dir, preset_name)
            os.makedirs(preset_dir, exist_ok=True)

            for t_idx in range(self.n_timesteps):
                self.frame_idx = t_idx
                # Render a few rotation angles for each frame
                for rot_step in range(4):
                    self.rotation_angle = rot_step * (np.pi / 2)
                    self._update_frame()
                    canvas.set_background_color(self.preset.bg_color)
                    self._render_scene(scene, camera)
                    canvas.scene(scene)

                    fname = f"frame_{t_idx:03d}_rot{rot_step}.png"
                    window.save_image(os.path.join(preset_dir, fname))

            print(f"Exported {self.n_timesteps * 4} frames → {preset_dir}/")

        window.destroy()

    def smoke_test(self):
        """Render single frame for QA, save to artifacts/."""
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        window = ti.ui.Window(
            "smoke_test", self.resolution, show_window=False,
        )
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()

        # Smoke test: one frame at t=0 with pitch preset
        self.preset = PRESETS["pitch"]
        self.show_wireframe = False
        self._preprocess_all()
        self._configure_camera(camera)
        self.frame_idx = 0
        self.rotation_angle = 0.4  # slight rotation to see 3D

        self._update_frame()
        canvas.set_background_color(self.preset.bg_color)
        self._render_scene(scene, camera)
        canvas.scene(scene)
        window.save_image(os.path.join(ARTIFACT_DIR, "smoke_test.png"))
        print(f"Saved: {ARTIFACT_DIR}/smoke_test.png")

        # Camera test: 3 positions
        cam_tests = [
            ("equatorial", (0.0, 0.0, 2.8), (0.0, 0.0, 0.0), 50.0, 0.0),
            ("above",      (0.0, 1.2, 2.2), (0.0, 0.0, 0.0), 45.0, 0.3),
            ("polar",      (0.0, 2.8, 0.3), (0.0, 0.0, 0.0), 50.0, 0.0),
        ]
        for name, pos, look, fov, rot in cam_tests:
            camera.position(*pos)
            camera.lookat(*look)
            camera.fov(fov)
            self.rotation_angle = rot
            self._update_frame()
            canvas.set_background_color(self.preset.bg_color)
            self._render_scene(scene, camera)
            canvas.scene(scene)
            path = os.path.join(ARTIFACT_DIR, f"camera_test_{name}.png")
            window.save_image(path)
            print(f"Saved: {path}")

        # Per-preset test
        for pname in PRESET_ORDER:
            self.preset = PRESETS[pname]
            self.show_wireframe = self.preset.visual.show_wireframe
            self._preprocess_all()
            self._configure_camera(camera)
            self.frame_idx = 0
            self.rotation_angle = 0.5

            self._update_frame()
            canvas.set_background_color(self.preset.bg_color)
            self._render_scene(scene, camera)
            canvas.scene(scene)
            path = os.path.join(ARTIFACT_DIR, f"preset_{pname}.png")
            window.save_image(path)
            print(f"Saved: {path}")

        window.destroy()
        print(f"\nSmoke test complete — check {ARTIFACT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="GeoPINN Sphere Viewer")
    parser.add_argument(
        "--preset", choices=PRESET_ORDER, default="pitch",
        help="Starting preset",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Headless export of all frames as PNGs",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Render smoke test images for QA",
    )
    parser.add_argument(
        "--res", type=int, nargs=2, default=[1280, 960],
        help="Window resolution",
    )
    args = parser.parse_args()

    viewer = SphereViewer(
        preset_name=args.preset,
        export_mode=args.export,
        resolution=tuple(args.res),
    )

    if args.smoke_test:
        viewer.smoke_test()
    elif args.export:
        viewer.export_frames()
    else:
        viewer.run()


if __name__ == "__main__":
    main()
