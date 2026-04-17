#!/usr/bin/env python3
"""
Taichi GGUI cloth demo for HGNN-ClothDyn.

Primary workflow:
- load precomputed physics and HGNN cloth rollouts
- render the fixed-topology cloth mesh in Taichi GGUI
- support preset-driven live playback and deterministic offline export
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import taichi as ti
except ImportError:  # pragma: no cover - runtime dependency
    ti = None


DEFAULT_PHYSICS_PATH = "outputs/physics_baseline/cloth_sequence.npy"
DEFAULT_HGNN_PATH = "outputs/hgnn_prediction/cloth_sequence.npy"
DEFAULT_MESH_INFO_PATH = "outputs/mesh_info.json"


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float


@dataclass(frozen=True)
class LightConfig:
    pos: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class ClothStyle:
    light_color: tuple[float, float, float]
    dark_color: tuple[float, float, float]
    shadow_tint: tuple[float, float, float]
    highlight_tint: tuple[float, float, float]
    displacement_tint: tuple[float, float, float]
    stripe_repeat: int
    stripe_strength: float
    tone_strength: float
    displacement_strength: float


@dataclass(frozen=True)
class DemoPreset:
    name: str
    description: str
    window_res: tuple[int, int]
    export_frames: int
    export_fps: int
    clip_start: int
    clip_end: int
    playback_speed: float
    mesh_scale: float
    comparison_gap: float
    surface_mode: str
    display_subdivisions: int
    camera: CameraPose
    background: tuple[float, float, float]
    ambient: tuple[float, float, float]
    lights: tuple[LightConfig, ...]
    floor_color: tuple[float, float, float]
    floor_margin: float
    wireframe_default: bool
    wireframe_color: tuple[float, float, float]
    wireframe_width: float
    physics_style: ClothStyle
    hgnn_style: ClothStyle


@dataclass
class DisplaySurfaceTemplate:
    name: str
    source_size: int
    display_size: int
    subdivisions: int
    face_indices: np.ndarray
    line_indices: np.ndarray
    rest_positions: np.ndarray
    row_ids: np.ndarray
    col_ids: np.ndarray
    sample_indices: np.ndarray
    sample_weights: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.rest_positions.shape[0])


@dataclass
class SurfaceFields:
    primary_vertices: Any
    primary_colors: Any
    secondary_vertices: Any
    secondary_colors: Any
    face_indices: Any
    line_indices: Any


PRESETS: dict[str, DemoPreset] = {
    "research": DemoPreset(
        name="research",
        description="Clear analytical framing with a cooler technical palette and wire overlay.",
        window_res=(1500, 940),
        export_frames=120,
        export_fps=30,
        clip_start=0,
        clip_end=28,
        playback_speed=0.82,
        mesh_scale=1.35,
        comparison_gap=0.88,
        surface_mode="raw",
        display_subdivisions=1,
        camera=CameraPose(
            position=(-0.06, 1.88, 4.45),
            lookat=(0.0, 0.56, 0.14),
            fov=30.0,
        ),
        background=(0.082, 0.094, 0.118),
        ambient=(0.34, 0.35, 0.36),
        lights=(
            LightConfig(pos=(2.4, 2.3, 2.1), color=(1.00, 0.99, 0.97)),
            LightConfig(pos=(-2.0, 1.2, -1.7), color=(0.34, 0.42, 0.52)),
        ),
        floor_color=(0.17, 0.19, 0.22),
        floor_margin=0.075,
        wireframe_default=True,
        wireframe_color=(0.13, 0.15, 0.18),
        wireframe_width=0.42,
        physics_style=ClothStyle(
            light_color=(0.82, 0.86, 0.92),
            dark_color=(0.60, 0.68, 0.80),
            shadow_tint=(0.40, 0.51, 0.69),
            highlight_tint=(0.94, 0.97, 0.99),
            displacement_tint=(0.54, 0.64, 0.86),
            stripe_repeat=4,
            stripe_strength=0.13,
            tone_strength=0.35,
            displacement_strength=0.16,
        ),
        hgnn_style=ClothStyle(
            light_color=(0.88, 0.89, 0.92),
            dark_color=(0.74, 0.75, 0.80),
            shadow_tint=(0.54, 0.57, 0.66),
            highlight_tint=(0.98, 0.98, 0.99),
            displacement_tint=(0.81, 0.72, 0.58),
            stripe_repeat=5,
            stripe_strength=0.10,
            tone_strength=0.30,
            displacement_strength=0.12,
        ),
    ),
    "pitch": DemoPreset(
        name="pitch",
        description="Balanced hero preset for laptop playback and export, with clean contrast and restrained color.",
        window_res=(1600, 960),
        export_frames=132,
        export_fps=30,
        clip_start=2,
        clip_end=30,
        playback_speed=0.74,
        mesh_scale=1.45,
        comparison_gap=0.92,
        surface_mode="smooth",
        display_subdivisions=3,
        camera=CameraPose(
            position=(-0.16, 1.78, 4.10),
            lookat=(0.0, 0.54, 0.18),
            fov=28.0,
        ),
        background=(0.040, 0.050, 0.070),
        ambient=(0.30, 0.31, 0.33),
        lights=(
            LightConfig(pos=(2.5, 2.6, 2.0), color=(1.05, 1.00, 0.95)),
            LightConfig(pos=(-2.2, 1.1, -1.4), color=(0.24, 0.38, 0.54)),
            LightConfig(pos=(0.0, 0.9, 2.7), color=(0.10, 0.12, 0.16)),
        ),
        floor_color=(0.13, 0.15, 0.18),
        floor_margin=0.085,
        wireframe_default=False,
        wireframe_color=(0.07, 0.08, 0.10),
        wireframe_width=0.58,
        physics_style=ClothStyle(
            light_color=(0.79, 0.84, 0.89),
            dark_color=(0.58, 0.67, 0.78),
            shadow_tint=(0.32, 0.48, 0.72),
            highlight_tint=(0.95, 0.97, 0.98),
            displacement_tint=(0.44, 0.62, 0.90),
            stripe_repeat=5,
            stripe_strength=0.08,
            tone_strength=0.34,
            displacement_strength=0.10,
        ),
        hgnn_style=ClothStyle(
            light_color=(0.90, 0.87, 0.80),
            dark_color=(0.72, 0.70, 0.64),
            shadow_tint=(0.59, 0.60, 0.69),
            highlight_tint=(0.99, 0.97, 0.92),
            displacement_tint=(0.93, 0.70, 0.45),
            stripe_repeat=6,
            stripe_strength=0.06,
            tone_strength=0.30,
            displacement_strength=0.12,
        ),
    ),
    "dramatic": DemoPreset(
        name="dramatic",
        description="Darker hero preset with stronger contrast, but still grounded in a physically readable camera angle.",
        window_res=(1760, 990),
        export_frames=144,
        export_fps=30,
        clip_start=4,
        clip_end=28,
        playback_speed=0.62,
        mesh_scale=1.42,
        comparison_gap=0.92,
        surface_mode="smooth",
        display_subdivisions=3,
        camera=CameraPose(
            position=(-0.30, 1.72, 4.18),
            lookat=(0.0, 0.54, 0.18),
            fov=27.5,
        ),
        background=(0.024, 0.028, 0.040),
        ambient=(0.26, 0.27, 0.29),
        lights=(
            LightConfig(pos=(2.6, 2.45, 1.7), color=(1.10, 1.00, 0.92)),
            LightConfig(pos=(-2.0, 1.05, -1.5), color=(0.20, 0.33, 0.54)),
            LightConfig(pos=(0.3, 0.75, 2.6), color=(0.10, 0.11, 0.15)),
        ),
        floor_color=(0.09, 0.10, 0.13),
        floor_margin=0.090,
        wireframe_default=False,
        wireframe_color=(0.05, 0.06, 0.08),
        wireframe_width=0.64,
        physics_style=ClothStyle(
            light_color=(0.83, 0.88, 0.95),
            dark_color=(0.54, 0.64, 0.78),
            shadow_tint=(0.24, 0.42, 0.72),
            highlight_tint=(0.96, 0.98, 1.00),
            displacement_tint=(0.50, 0.70, 0.96),
            stripe_repeat=6,
            stripe_strength=0.04,
            tone_strength=0.38,
            displacement_strength=0.12,
        ),
        hgnn_style=ClothStyle(
            light_color=(0.94, 0.88, 0.76),
            dark_color=(0.66, 0.62, 0.56),
            shadow_tint=(0.48, 0.51, 0.62),
            highlight_tint=(1.00, 0.96, 0.88),
            displacement_tint=(0.98, 0.72, 0.38),
            stripe_repeat=7,
            stripe_strength=0.03,
            tone_strength=0.34,
            displacement_strength=0.14,
        ),
    ),
}

PRESET_ORDER = tuple(PRESETS.keys())
MODE_ORDER = ("physics", "hgnn", "compare")
SURFACE_ORDER = ("raw", "smooth")


def choose(value, fallback):
    return fallback if value is None else value


def serialize_preset(preset: DemoPreset) -> dict:
    return asdict(preset)


def print_presets() -> None:
    print("Available presets:")
    for preset in PRESETS.values():
        print(f"- {preset.name}: {preset.description}")


def try_init_taichi(preferred: str) -> str:
    if ti is None:
        raise RuntimeError("Taichi is not installed. Install it with `pip install taichi`.")

    order = {
        "auto": ["cuda", "vulkan", "metal", "cpu"],
        "cuda": ["cuda", "vulkan", "cpu"],
        "vulkan": ["vulkan", "cpu"],
        "metal": ["metal", "cpu"],
        "cpu": ["cpu"],
    }[preferred]

    errors = []
    for arch_name in order:
        arch = getattr(ti, arch_name)
        try:
            ti.reset()
            ti.init(arch=arch, default_fp=ti.f32, offline_cache=True)
            return arch_name
        except Exception as exc:  # pragma: no cover - backend dependent
            errors.append(f"{arch_name}: {exc}")

    raise RuntimeError("Failed to initialize Taichi GGUI backend:\n" + "\n".join(errors))


def ffmpeg_commands(export_dir: Path, fps: int) -> dict[str, str]:
    pattern = export_dir / "frame_%04d.png"
    mp4_path = export_dir / "cloth_demo.mp4"
    gif_path = export_dir / "cloth_demo.gif"
    mp4 = (
        f"ffmpeg -y -framerate {fps} -i {pattern} "
        f"-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p\" "
        f"{mp4_path}"
    )
    gif = (
        f"ffmpeg -y -framerate {fps} -i {pattern} "
        f"-vf \"fps={fps},scale=960:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen[p];[s1][p]paletteuse\" "
        f"{gif_path}"
    )
    return {"mp4": mp4, "gif": gif}


def build_ground_plane(height: float, extent_x: float, extent_z: float) -> tuple[np.ndarray, np.ndarray]:
    half_x = extent_x * 0.5
    half_z = extent_z * 0.5
    vertices = np.array(
        [
            [-half_x, height, -half_z],
            [half_x, height, -half_z],
            [-half_x, height, half_z],
            [half_x, height, half_z],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
    return vertices, indices


def build_surface_topology(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build triangle and line indices for a regular grid surface."""
    faces = []
    edge_set: set[tuple[int, int]] = set()
    for row in range(size - 1):
        for col in range(size - 1):
            v0 = row * size + col
            v1 = v0 + 1
            v2 = v0 + size
            v3 = v2 + 1
            faces.append((v0, v1, v2))
            faces.append((v1, v3, v2))
            for edge in ((v0, v1), (v1, v3), (v3, v2), (v2, v0), (v1, v2)):
                edge_set.add(tuple(sorted(edge)))

    face_indices = np.asarray(faces, dtype=np.int32).reshape(-1)
    line_indices = np.asarray(sorted(edge_set), dtype=np.int32).reshape(-1)
    return face_indices, line_indices


def build_catmull_rom_samples(source_size: int, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute separable Catmull-Rom sampling tables for display upsampling."""
    subdiv = max(int(subdivisions), 1)
    display_size = (source_size - 1) * subdiv + 1
    coords = np.arange(display_size, dtype=np.float32) / float(subdiv)
    base = np.floor(coords).astype(np.int32)
    t = coords - base.astype(np.float32)

    indices = np.stack(
        [
            np.clip(base - 1, 0, source_size - 1),
            np.clip(base, 0, source_size - 1),
            np.clip(base + 1, 0, source_size - 1),
            np.clip(base + 2, 0, source_size - 1),
        ],
        axis=1,
    ).astype(np.int32)

    weights = np.stack(
        [
            0.5 * (-t + 2.0 * t * t - t * t * t),
            0.5 * (2.0 - 5.0 * t * t + 3.0 * t * t * t),
            0.5 * (t + 4.0 * t * t - 3.0 * t * t * t),
            0.5 * (-t * t + t * t * t),
        ],
        axis=1,
    ).astype(np.float32)

    return indices, weights


def resample_grid_catmull_rom(
    grid: np.ndarray,
    sample_indices: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Upsample a regular cloth grid with separable Catmull-Rom interpolation."""
    sampled_rows = np.take(grid, sample_indices.reshape(-1), axis=0)
    sampled_rows = sampled_rows.reshape(sample_indices.shape[0], 4, grid.shape[1], grid.shape[2])
    blended_rows = (sampled_rows * sample_weights[:, :, None, None]).sum(axis=1)

    sampled_cols = np.take(blended_rows, sample_indices.reshape(-1), axis=1)
    sampled_cols = sampled_cols.reshape(blended_rows.shape[0], sample_indices.shape[0], 4, grid.shape[2])
    blended = (sampled_cols * sample_weights[None, :, :, None]).sum(axis=2)
    return blended.astype(np.float32)


def build_display_surface_template(
    name: str,
    rest_positions: np.ndarray,
    mesh_size: int,
    subdivisions: int,
) -> DisplaySurfaceTemplate:
    """Build a raw or smoothed display surface from the fixed cloth grid."""
    sample_indices, sample_weights = build_catmull_rom_samples(mesh_size, subdivisions)
    display_size = int(sample_indices.shape[0])
    face_indices, line_indices = build_surface_topology(display_size)

    rest_grid = rest_positions.reshape(mesh_size, mesh_size, 3).astype(np.float32)
    display_rest_positions = resample_grid_catmull_rom(rest_grid, sample_indices, sample_weights).reshape(-1, 3)
    row_ids = (np.arange(display_rest_positions.shape[0]) // display_size).astype(np.int32)
    col_ids = (np.arange(display_rest_positions.shape[0]) % display_size).astype(np.int32)

    return DisplaySurfaceTemplate(
        name=name,
        source_size=mesh_size,
        display_size=display_size,
        subdivisions=max(int(subdivisions), 1),
        face_indices=face_indices,
        line_indices=line_indices,
        rest_positions=display_rest_positions,
        row_ids=row_ids,
        col_ids=col_ids,
        sample_indices=sample_indices,
        sample_weights=sample_weights,
    )


@dataclass
class ClothDemoBundle:
    physics_positions: np.ndarray
    hgnn_positions: np.ndarray
    face_indices: np.ndarray
    line_indices: np.ndarray
    rest_positions: np.ndarray
    fixed_indices: np.ndarray
    mesh_size: int
    row_ids: np.ndarray
    col_ids: np.ndarray
    frame_offset: int
    timestep: float
    floor_y: float
    extent_x: float
    extent_z: float

    @property
    def n_frames(self) -> int:
        return int(self.physics_positions.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.physics_positions.shape[1])

    @property
    def clip_span(self) -> float:
        return float(max(self.n_frames - 1, 0))


def load_cloth_bundle(
    physics_path: str,
    hgnn_path: str,
    mesh_info_path: str,
    clip_start: int,
    clip_end: int | None,
    mesh_scale: float,
) -> ClothDemoBundle:
    physics = np.load(physics_path).astype(np.float32)
    hgnn = np.load(hgnn_path).astype(np.float32)

    if physics.shape != hgnn.shape:
        raise ValueError(f"Sequence shape mismatch: physics={physics.shape}, hgnn={hgnn.shape}")

    mesh_info = json.loads(Path(mesh_info_path).read_text())
    faces = np.asarray(mesh_info["faces"], dtype=np.int32)
    fixed_indices = np.asarray(mesh_info.get("fixed_vertex_indices", []), dtype=np.int32)
    mesh_size = int(mesh_info["mesh_size"])
    timestep = float(mesh_info.get("timestep", 0.01))

    start = max(int(clip_start), 0)
    last_frame = physics.shape[0] - 1
    end = last_frame if clip_end is None else min(int(clip_end), last_frame)
    if end < start:
        raise ValueError(f"Invalid clip range: start={start}, end={end}")

    physics = physics[start : end + 1]
    hgnn = hgnn[start : end + 1]

    _, line_indices = build_surface_topology(mesh_size)
    rest_positions = physics[0].copy()

    all_positions = np.concatenate([physics, hgnn], axis=0) * mesh_scale
    mins = all_positions.reshape(-1, 3).min(axis=0)
    maxs = all_positions.reshape(-1, 3).max(axis=0)

    row_ids = (np.arange(rest_positions.shape[0]) // mesh_size).astype(np.int32)
    col_ids = (np.arange(rest_positions.shape[0]) % mesh_size).astype(np.int32)

    return ClothDemoBundle(
        physics_positions=physics,
        hgnn_positions=hgnn,
        face_indices=faces.reshape(-1).astype(np.int32),
        line_indices=line_indices,
        rest_positions=rest_positions,
        fixed_indices=fixed_indices,
        mesh_size=mesh_size,
        row_ids=row_ids,
        col_ids=col_ids,
        frame_offset=start,
        timestep=timestep,
        floor_y=float(mins[1]),
        extent_x=float(maxs[0] - mins[0]),
        extent_z=float(maxs[2] - mins[2]),
    )


def mix_colors(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return a * (1.0 - weight) + b * weight


def compute_vertex_colors(
    positions: np.ndarray,
    rest_positions: np.ndarray,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    style: ClothStyle,
) -> np.ndarray:
    base_a = np.asarray(style.dark_color, dtype=np.float32)
    base_b = np.asarray(style.light_color, dtype=np.float32)
    shadow = np.asarray(style.shadow_tint, dtype=np.float32)
    highlight = np.asarray(style.highlight_tint, dtype=np.float32)
    displacement_tint = np.asarray(style.displacement_tint, dtype=np.float32)

    stripe_repeat = max(int(style.stripe_repeat), 1)
    stripe_mask = ((row_ids // stripe_repeat + col_ids // stripe_repeat) % 2).astype(np.float32)
    weave = 0.5 + 0.5 * np.sin((col_ids.astype(np.float32) + 0.6 * row_ids.astype(np.float32)) * 0.55)
    stripe_weight = np.clip(0.35 * weave + style.stripe_strength * stripe_mask, 0.0, 1.0).astype(np.float32)

    base = mix_colors(base_a[None, :], base_b[None, :], stripe_weight[:, None])

    y = positions[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    y_span = max(y_max - y_min, 1e-6)
    height_weight = ((y - y_min) / y_span).astype(np.float32)
    tone = mix_colors(shadow[None, :], highlight[None, :], height_weight[:, None])

    disp = np.linalg.norm(positions - rest_positions, axis=1).astype(np.float32)
    disp_ref = max(np.percentile(disp, 95), 1e-6)
    disp_weight = np.clip(disp / disp_ref, 0.0, 1.0)

    colors = base * (1.0 - style.tone_strength) + tone * style.tone_strength
    colors = colors * (1.0 - style.displacement_strength * disp_weight[:, None])
    colors = colors + displacement_tint[None, :] * (style.displacement_strength * disp_weight[:, None])

    return np.clip(colors, 0.0, 1.0).astype(np.float32)


class TaichiClothDemo:
    def __init__(
        self,
        bundle: ClothDemoBundle,
        preset: DemoPreset,
        window_res: tuple[int, int],
        export_dir: Path | None,
        export_only: bool,
        export_frames: int,
        export_fps: int,
        fps: int,
        playback_speed: float,
        start_wireframe: bool,
        hide_window: bool,
        show_gui: bool,
        status_every: int,
        title: str,
        max_steps: int | None,
        start_mode: str,
        start_surface_mode: str,
        mesh_scale: float,
        comparison_gap: float,
    ):
        self.bundle = bundle
        self.preset = preset
        self.window_res = window_res
        self.export_dir = export_dir
        self.export_only = export_only
        self.export_frames = max(int(export_frames), 1)
        self.export_fps = int(export_fps)
        self.fps = int(fps)
        self.playback_speed = float(playback_speed)
        self.show_wireframe = start_wireframe
        self.show_window = not export_only and not hide_window
        self.show_gui = show_gui and self.show_window
        self.status_every = max(int(status_every), 1)
        self.title = title
        self.max_steps = max_steps
        self.mode = start_mode
        self.mesh_scale = float(mesh_scale)
        self.comparison_gap = float(comparison_gap)
        self.surface_mode = start_surface_mode

        self.clip_span = self.bundle.clip_span
        self.source_frame_rate = float(max(1.0 / max(self.bundle.timestep, 1e-6), 1.0))
        self.frame_cursor = 0.0
        self.frame_idx = 0
        self.global_frame_idx = self.bundle.frame_offset
        self.current_time = self.global_frame_idx * self.bundle.timestep
        self.playing = True
        self.should_close = False
        self.exported_frames = 0
        self.last_status_marker = -1
        self.last_tick = time.perf_counter()
        self.preset_index = PRESET_ORDER.index(self.preset.name)
        smooth_subdivisions = max(
            1,
            max(p.display_subdivisions for p in PRESETS.values()),
        )
        self.surface_templates = {
            "raw": build_display_surface_template("raw", bundle.rest_positions, bundle.mesh_size, 1),
            "smooth": build_display_surface_template(
                "smooth",
                bundle.rest_positions,
                bundle.mesh_size,
                smooth_subdivisions,
            ),
        }
        self.surface_fields = {
            name: self._allocate_surface_fields(template)
            for name, template in self.surface_templates.items()
        }
        self.support_primary_vertices = ti.Vector.field(3, dtype=ti.f32, shape=max(bundle.fixed_indices.size * 2, 1))
        self.support_primary_indices = ti.field(dtype=ti.i32, shape=max(bundle.fixed_indices.size * 2, 1))
        self.support_secondary_vertices = ti.Vector.field(3, dtype=ti.f32, shape=max(bundle.fixed_indices.size * 2, 1))
        self.support_secondary_indices = ti.field(dtype=ti.i32, shape=max(bundle.fixed_indices.size * 2, 1))

        self.ground_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.ground_indices = ti.field(dtype=ti.i32, shape=6)
        self._update_ground_plane()

        self.window = ti.ui.Window(
            name=self.title,
            res=self.window_res,
            vsync=not export_only,
            show_window=self.show_window,
            fps_limit=self.fps,
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._configure_camera()
        self._push_cursor(0.0)

    def _configure_camera(self) -> None:
        pose = self.preset.camera
        self.camera.position(*pose.position)
        self.camera.lookat(*pose.lookat)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.fov(pose.fov)

    def _update_ground_plane(self) -> None:
        floor_extent_x = max(self.bundle.extent_x * self.mesh_scale + 2.2 * self.comparison_gap + 0.8, 4.2)
        floor_extent_z = max(self.bundle.extent_z * self.mesh_scale + 1.8, 4.2)
        ground_y = self.bundle.floor_y * self.mesh_scale - self.preset.floor_margin
        ground_vertices, ground_indices = build_ground_plane(ground_y, floor_extent_x, floor_extent_z)
        self.ground_vertices.from_numpy(ground_vertices)
        self.ground_indices.from_numpy(ground_indices)

    def _allocate_surface_fields(self, template: DisplaySurfaceTemplate) -> SurfaceFields:
        """Allocate Taichi fields for one render surface."""
        fields = SurfaceFields(
            primary_vertices=ti.Vector.field(3, dtype=ti.f32, shape=template.vertex_count),
            primary_colors=ti.Vector.field(3, dtype=ti.f32, shape=template.vertex_count),
            secondary_vertices=ti.Vector.field(3, dtype=ti.f32, shape=template.vertex_count),
            secondary_colors=ti.Vector.field(3, dtype=ti.f32, shape=template.vertex_count),
            face_indices=ti.field(dtype=ti.i32, shape=template.face_indices.shape[0]),
            line_indices=ti.field(dtype=ti.i32, shape=template.line_indices.shape[0]),
        )
        fields.face_indices.from_numpy(template.face_indices)
        fields.line_indices.from_numpy(template.line_indices)
        return fields

    def _active_surface(self) -> tuple[DisplaySurfaceTemplate, SurfaceFields]:
        template = self.surface_templates[self.surface_mode]
        fields = self.surface_fields[self.surface_mode]
        return template, fields

    def _current_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bundle.physics_positions, self.bundle.hgnn_positions

    def _interpolate_positions(self, cursor: float) -> tuple[np.ndarray, np.ndarray, int]:
        physics_seq, hgnn_seq = self._current_sequences()
        if self.bundle.n_frames == 1 or self.clip_span <= 0.0:
            return physics_seq[0], hgnn_seq[0], 0

        cursor = float(np.clip(cursor, 0.0, self.clip_span))
        lower = int(np.floor(cursor))
        upper = min(lower + 1, self.bundle.n_frames - 1)
        alpha = np.float32(cursor - lower)

        physics = (1.0 - alpha) * physics_seq[lower] + alpha * physics_seq[upper]
        hgnn = (1.0 - alpha) * hgnn_seq[lower] + alpha * hgnn_seq[upper]
        return physics, hgnn, lower

    def _transform_positions(self, positions: np.ndarray, x_offset: float) -> np.ndarray:
        transformed = positions.astype(np.float32) * self.mesh_scale
        transformed[:, 0] += x_offset
        return transformed.astype(np.float32)

    def _transform_surface_rest(self, template: DisplaySurfaceTemplate, x_offset: float) -> np.ndarray:
        rest = template.rest_positions.astype(np.float32).copy() * self.mesh_scale
        rest[:, 0] += x_offset
        return rest

    def _resample_surface(self, positions: np.ndarray, template: DisplaySurfaceTemplate) -> np.ndarray:
        if template.subdivisions <= 1:
            return positions.astype(np.float32)
        grid = positions.reshape(template.source_size, template.source_size, 3)
        dense = resample_grid_catmull_rom(grid, template.sample_indices, template.sample_weights)
        return dense.reshape(-1, 3).astype(np.float32)

    def _prepare_surface(
        self,
        positions: np.ndarray,
        x_offset: float,
        style: ClothStyle,
    ) -> tuple[np.ndarray, np.ndarray]:
        template, _ = self._active_surface()
        base_positions = self._transform_positions(positions, x_offset)
        surface_positions = self._resample_surface(base_positions, template)
        surface_rest = self._transform_surface_rest(template, x_offset)
        colors = compute_vertex_colors(
            surface_positions,
            surface_rest,
            template.row_ids,
            template.col_ids,
            style,
        )
        return surface_positions, colors

    def _make_support_segments(self, positions: np.ndarray, x_offset: float) -> tuple[np.ndarray, np.ndarray]:
        if self.bundle.fixed_indices.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        pins = self._transform_positions(positions[self.bundle.fixed_indices], x_offset)
        anchors = pins.copy()
        anchors[:, 1] += 0.16 * self.mesh_scale
        support_positions = np.empty((pins.shape[0] * 2, 3), dtype=np.float32)
        support_positions[0::2] = pins
        support_positions[1::2] = anchors
        support_indices = np.arange(support_positions.shape[0], dtype=np.int32)
        return support_positions, support_indices

    def _push_cursor(self, cursor: float) -> None:
        physics, hgnn, frame_idx = self._interpolate_positions(cursor)
        global_frame = self.bundle.frame_offset + frame_idx
        template, fields = self._active_surface()

        if self.mode == "physics":
            primary_positions, primary_colors = self._prepare_surface(
                physics,
                0.0,
                self.preset.physics_style,
            )
            secondary_positions = np.zeros((template.vertex_count, 3), dtype=np.float32)
            secondary_colors = np.zeros_like(primary_colors)
        elif self.mode == "hgnn":
            primary_positions, primary_colors = self._prepare_surface(
                hgnn,
                0.0,
                self.preset.hgnn_style,
            )
            secondary_positions = np.zeros((template.vertex_count, 3), dtype=np.float32)
            secondary_colors = np.zeros_like(primary_colors)
        else:
            left_offset = -self.comparison_gap
            right_offset = self.comparison_gap
            primary_positions, primary_colors = self._prepare_surface(
                physics,
                left_offset,
                self.preset.physics_style,
            )
            secondary_positions, secondary_colors = self._prepare_surface(
                hgnn,
                right_offset,
                self.preset.hgnn_style,
            )

        fields.primary_vertices.from_numpy(primary_positions)
        fields.primary_colors.from_numpy(primary_colors)
        fields.secondary_vertices.from_numpy(secondary_positions)
        fields.secondary_colors.from_numpy(secondary_colors)
        self._update_support_fields(physics, hgnn)

        self.frame_cursor = float(cursor)
        self.frame_idx = int(frame_idx)
        self.global_frame_idx = int(global_frame)
        self.current_time = float(self.global_frame_idx * self.bundle.timestep)

    def _print_status(self, force: bool = False) -> None:
        status_marker = int(round(self.frame_cursor * 1000.0))
        if not force and status_marker == self.last_status_marker:
            return
        should_print = force or (self.exported_frames % self.status_every == 0 if self.export_only else self.frame_idx % self.status_every == 0)
        if should_print:
            print(
                f"preset={self.preset.name} mode={self.mode} "
                f"surface={self.surface_mode} "
                f"frame={self.global_frame_idx:03d} "
                f"cursor={self.frame_cursor:06.3f}/{self.clip_span:06.3f} "
                f"time={self.current_time:.3f}s "
                f"speed={self.playback_speed:.2f} "
                f"wire={self.show_wireframe}"
            )
            self.last_status_marker = status_marker

    def _cycle_mode(self, step: int = 1) -> None:
        idx = MODE_ORDER.index(self.mode)
        self.mode = MODE_ORDER[(idx + step) % len(MODE_ORDER)]
        self._push_cursor(self.frame_cursor)
        print(f"mode={self.mode}")

    def _cycle_surface(self, step: int = 1) -> None:
        idx = SURFACE_ORDER.index(self.surface_mode)
        self.surface_mode = SURFACE_ORDER[(idx + step) % len(SURFACE_ORDER)]
        self._push_cursor(self.frame_cursor)
        print(f"surface={self.surface_mode}")

    def _set_preset(self, preset_name: str) -> None:
        self.preset = PRESETS[preset_name]
        self.preset_index = PRESET_ORDER.index(preset_name)
        self.playback_speed = self.preset.playback_speed
        self.show_wireframe = self.preset.wireframe_default
        self.mesh_scale = self.preset.mesh_scale
        self.comparison_gap = self.preset.comparison_gap
        self.surface_mode = self.preset.surface_mode
        self._configure_camera()
        self._update_ground_plane()
        self._push_cursor(self.frame_cursor)
        print(f"preset={preset_name}")

    def _handle_press(self, key) -> None:
        if key in (ti.ui.ESCAPE, ti.ui.EXIT):
            self.should_close = True
        elif key == ti.ui.SPACE:
            self.playing = not self.playing
            self.last_tick = time.perf_counter()
            print("play" if self.playing else "pause")
        elif key in ("r", "R"):
            self._push_cursor(0.0)
            self.last_tick = time.perf_counter()
            print("reset")
        elif key in ("[", "-"):
            self.playback_speed = max(0.1, self.playback_speed * 0.8)
            print(f"speed={self.playback_speed:.2f}")
        elif key in ("]", "="):
            self.playback_speed = min(8.0, self.playback_speed * 1.25)
            print(f"speed={self.playback_speed:.2f}")
        elif key in ("w", "W"):
            self.show_wireframe = not self.show_wireframe
            print(f"wireframe={self.show_wireframe}")
        elif key in ("u", "U"):
            self._cycle_surface(1)
        elif key == "1":
            self.mode = "physics"
            self._push_cursor(self.frame_cursor)
            print("mode=physics")
        elif key == "2":
            self.mode = "hgnn"
            self._push_cursor(self.frame_cursor)
            print("mode=hgnn")
        elif key == "3":
            self.mode = "compare"
            self._push_cursor(self.frame_cursor)
            print("mode=compare")
        elif key in ("p", "P"):
            next_name = PRESET_ORDER[(self.preset_index + 1) % len(PRESET_ORDER)]
            self._set_preset(next_name)
        elif key in ("m", "M"):
            self._cycle_mode(1)

    def _poll_events(self) -> None:
        if not self.show_window:
            return
        while self.window.get_event(ti.ui.PRESS):
            self._handle_press(self.window.event.key)

    def _advance(self) -> None:
        if not self.playing or self.clip_span <= 0.0:
            return

        now = time.perf_counter()
        dt = max(0.0, min(now - self.last_tick, 0.25))
        self.last_tick = now
        next_cursor = self.frame_cursor + self.playback_speed * self.source_frame_rate * dt
        next_cursor = next_cursor % self.clip_span
        self._push_cursor(next_cursor)

    def _draw_gui(self) -> None:
        if not self.show_gui:
            return
        gui = self.window.get_gui()
        gui.begin("HGNN Cloth Demo", 0.015, 0.015, 0.32, 0.19)
        gui.text(f"preset: {self.preset.name}")
        gui.text(f"mode: {self.mode}")
        gui.text(f"surface: {self.surface_mode}")
        gui.text(f"frame: {self.global_frame_idx}")
        gui.text(f"time: {self.current_time:.3f}s")
        gui.text(f"speed: {self.playback_speed:.2f}x")
        if self.mode == "compare":
            gui.text("left = physics baseline, right = HGNN")
        gui.text("[space] play  [r] reset  [1/2/3] mode")
        gui.text("[p] preset  [u] surface  [w] wire  [[ ]] speed")
        gui.end()

    def _update_support_fields(self, physics: np.ndarray, hgnn: np.ndarray) -> None:
        support_size = max(self.bundle.fixed_indices.size * 2, 1)
        primary_positions = np.zeros((support_size, 3), dtype=np.float32)
        secondary_positions = np.zeros((support_size, 3), dtype=np.float32)
        support_indices = np.arange(support_size, dtype=np.int32)

        if self.bundle.fixed_indices.size > 0:
            if self.mode == "physics":
                primary, _ = self._make_support_segments(physics, 0.0)
                primary_positions[: primary.shape[0]] = primary
            elif self.mode == "hgnn":
                primary, _ = self._make_support_segments(hgnn, 0.0)
                primary_positions[: primary.shape[0]] = primary
            else:
                primary, _ = self._make_support_segments(physics, -self.comparison_gap)
                secondary, _ = self._make_support_segments(hgnn, self.comparison_gap)
                primary_positions[: primary.shape[0]] = primary
                secondary_positions[: secondary.shape[0]] = secondary

        self.support_primary_vertices.from_numpy(primary_positions)
        self.support_primary_indices.from_numpy(support_indices)
        self.support_secondary_vertices.from_numpy(secondary_positions)
        self.support_secondary_indices.from_numpy(support_indices)

    def _render_supports(self, vertices_field, indices_field) -> None:
        if self.bundle.fixed_indices.size == 0:
            return
        self.scene.lines(
            vertices_field,
            indices=indices_field,
            width=1.6,
            color=(0.90, 0.90, 0.92),
        )

    def _render_scene(self) -> None:
        _, fields = self._active_surface()
        self.canvas.set_background_color(self.preset.background)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.preset.ambient)
        for light in self.preset.lights:
            self.scene.point_light(pos=light.pos, color=light.color)

        self.scene.mesh(
            self.ground_vertices,
            indices=self.ground_indices,
            color=self.preset.floor_color,
            two_sided=True,
        )

        self.scene.mesh(
            fields.primary_vertices,
            indices=fields.face_indices,
            per_vertex_color=fields.primary_colors,
            two_sided=True,
        )

        if self.mode == "compare":
            self.scene.mesh(
                fields.secondary_vertices,
                indices=fields.face_indices,
                per_vertex_color=fields.secondary_colors,
                two_sided=True,
            )

        if self.show_wireframe:
            self.scene.lines(
                fields.primary_vertices,
                width=self.preset.wireframe_width,
                indices=fields.line_indices,
                color=self.preset.wireframe_color,
            )
            if self.mode == "compare":
                self.scene.lines(
                    fields.secondary_vertices,
                    width=self.preset.wireframe_width,
                    indices=fields.line_indices,
                    color=self.preset.wireframe_color,
                )

        if self.mode == "physics":
            self._render_supports(self.support_primary_vertices, self.support_primary_indices)
        elif self.mode == "hgnn":
            self._render_supports(self.support_primary_vertices, self.support_primary_indices)
        else:
            self._render_supports(self.support_primary_vertices, self.support_primary_indices)
            self._render_supports(self.support_secondary_vertices, self.support_secondary_indices)

        self.canvas.scene(self.scene)
        self._draw_gui()

    def _export_frame(self) -> None:
        if self.export_dir is None:
            return
        frame_path = self.export_dir / f"frame_{self.exported_frames:04d}.png"
        self.window.save_image(str(frame_path))
        self.exported_frames += 1

    def _export_cursors(self) -> np.ndarray:
        if self.export_frames <= 1 or self.clip_span <= 0.0:
            return np.array([0.0], dtype=np.float32)
        return np.linspace(0.0, self.clip_span, self.export_frames, dtype=np.float32)

    def run(self) -> None:
        if self.export_only:
            self.playing = False
            for cursor in self._export_cursors():
                self._push_cursor(float(cursor))
                self._render_scene()
                self._export_frame()
                self._print_status(force=False)
            self.window.destroy()
            return

        steps = 0
        while (self.window.running or not self.show_window) and not self.should_close:
            self._poll_events()
            if self.show_window:
                self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._advance()
            self._render_scene()
            self._print_status(force=False)
            if self.show_window:
                self.window.show()
            else:
                time.sleep(1.0 / max(self.fps, 1))
            steps += 1
            if self.max_steps is not None and steps >= self.max_steps:
                break

        self.window.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taichi GGUI cloth demo for HGNN-ClothDyn")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="pitch", help="Render preset")
    parser.add_argument("--list-presets", action="store_true", help="Print preset names and exit")
    parser.add_argument("--mode", choices=MODE_ORDER, default="compare", help="Playback mode")
    parser.add_argument("--surface-mode", choices=SURFACE_ORDER, default=None, help="Render the raw simulation mesh or the smoothed display surface")
    parser.add_argument("--physics-path", type=str, default=DEFAULT_PHYSICS_PATH, help="Physics baseline .npy")
    parser.add_argument("--hgnn-path", type=str, default=DEFAULT_HGNN_PATH, help="HGNN rollout .npy")
    parser.add_argument("--mesh-info", type=str, default=DEFAULT_MESH_INFO_PATH, help="Mesh metadata JSON")
    parser.add_argument("--clip-start", type=int, default=None, help="First source frame to include")
    parser.add_argument("--clip-end", type=int, default=None, help="Last source frame to include")
    parser.add_argument("--window-width", type=int, default=None)
    parser.add_argument("--window-height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--playback-speed", type=float, default=None)
    parser.add_argument("--mesh-scale", type=float, default=None, help="Render-only scale factor")
    parser.add_argument("--comparison-gap", type=float, default=None, help="Horizontal separation in compare mode")
    parser.add_argument(
        "--wireframe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the wireframe overlay",
    )
    parser.add_argument("--arch", choices=["auto", "cuda", "vulkan", "metal", "cpu"], default="auto")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for rendered PNG frames")
    parser.add_argument("--export-only", action="store_true", help="Render a deterministic off-screen frame sequence and exit")
    parser.add_argument("--export-frames", type=int, default=None, help="Number of rendered frames to export")
    parser.add_argument("--export-fps", type=int, default=None, help="Target playback fps for exported frames")
    parser.add_argument("--hide-window", action="store_true", help="Run the playback loop without showing a window")
    parser.add_argument("--max-steps", type=int, default=None, help="Auto-exit after N loop iterations in live mode")
    parser.add_argument("--status-every", type=int, default=8, help="Terminal status interval")
    parser.add_argument("--title", type=str, default="HGNN ClothDyn Taichi Demo")
    parser.add_argument("--no-gui", action="store_true", help="Disable the small on-screen info panel")
    parser.add_argument("--metadata-out", type=str, default=None, help="Optional JSON file to save render metadata")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_presets:
        print_presets()
        return 0

    preset = PRESETS[args.preset]
    window_res = (
        int(choose(args.window_width, preset.window_res[0])),
        int(choose(args.window_height, preset.window_res[1])),
    )
    export_frames = int(choose(args.export_frames, preset.export_frames))
    export_fps = int(choose(args.export_fps, preset.export_fps))
    playback_speed = float(choose(args.playback_speed, preset.playback_speed))
    mesh_scale = float(choose(args.mesh_scale, preset.mesh_scale))
    comparison_gap = float(choose(args.comparison_gap, preset.comparison_gap))
    surface_mode = str(choose(args.surface_mode, preset.surface_mode))
    wireframe = bool(choose(args.wireframe, preset.wireframe_default))
    clip_start = int(choose(args.clip_start, preset.clip_start))
    clip_end = choose(args.clip_end, preset.clip_end)

    arch_name = try_init_taichi(args.arch)
    print(f"Taichi arch: {arch_name}")

    bundle = load_cloth_bundle(
        physics_path=args.physics_path,
        hgnn_path=args.hgnn_path,
        mesh_info_path=args.mesh_info,
        clip_start=clip_start,
        clip_end=clip_end,
        mesh_scale=1.0,
    )

    export_dir = None
    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        if args.export_only:
            for stale_frame in export_dir.glob("frame_*.png"):
                stale_frame.unlink()
        print(f"Export dir: {export_dir}")
        for name, command in ffmpeg_commands(export_dir, export_fps).items():
            print(f"{name}: {command}")

    demo = TaichiClothDemo(
        bundle=bundle,
        preset=preset,
        window_res=window_res,
        export_dir=export_dir,
        export_only=args.export_only,
        export_frames=export_frames,
        export_fps=export_fps,
        fps=args.fps,
        playback_speed=playback_speed,
        start_wireframe=wireframe,
        hide_window=args.hide_window,
        show_gui=not args.no_gui,
        status_every=args.status_every,
        title=args.title,
        max_steps=args.max_steps,
        start_mode=args.mode,
        start_surface_mode=surface_mode,
        mesh_scale=mesh_scale,
        comparison_gap=comparison_gap,
    )
    demo.run()

    metadata = {
        "preset": preset.name,
        "mode": args.mode,
        "arch": arch_name,
        "physics_path": str(args.physics_path),
        "hgnn_path": str(args.hgnn_path),
        "mesh_info": str(args.mesh_info),
        "clip_start": clip_start,
        "clip_end": clip_end,
        "window_res": list(window_res),
        "export_only": bool(args.export_only),
        "export_frames": export_frames,
        "export_fps": export_fps,
        "fps": int(args.fps),
        "playback_speed": playback_speed,
        "mesh_scale": mesh_scale,
        "comparison_gap": comparison_gap,
        "surface_mode": surface_mode,
        "wireframe": wireframe,
        "show_gui": not args.no_gui,
        "preset_config": serialize_preset(preset),
        "timestep": bundle.timestep,
        "bundle_frames": bundle.n_frames,
        "bundle_frame_offset": bundle.frame_offset,
        "source_mesh_size": bundle.mesh_size,
        "display_mesh_size": demo.surface_templates[surface_mode].display_size,
        "ffmpeg": ffmpeg_commands(export_dir, export_fps) if export_dir else None,
    }

    if args.metadata_out:
        metadata_path = Path(args.metadata_out)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"Metadata: {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
