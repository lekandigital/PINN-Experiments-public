#!/usr/bin/env python3
"""
Taichi GGUI demo for NIF-Cloth4D.

Loads the fixed-topology heightfield cloth sequences extracted from the
trained neural implicit field (NIF) and the analytic physics baseline,
then renders them in a dark-themed GGUI scene with warm-key / cool-fill
lighting.

Sequences live under outputs/{nif_prediction,physics_baseline}/*.npy.
Both sequences share the same fixed topology (32,258 triangles), so the
viewer can update only vertices per frame. Toggle between NIF and the
physics baseline with `m`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    import taichi as ti
except ImportError:  # pragma: no cover
    ti = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    rim_pos: tuple
    rim_color: tuple
    rim_intensity: float
    ambient: tuple


@dataclass(frozen=True)
class PaletteConfig:
    # Height-based diverging colormap: low (sag) warm, high (taut) cool.
    low: tuple    # warm cream/amber
    mid: tuple    # neutral
    high: tuple   # cool blue
    gamma: float  # non-linear shaping


@dataclass(frozen=True)
class DemoPreset:
    name: str
    description: str
    window_res: tuple
    background: tuple
    camera: CameraPose
    light: LightConfig
    palette: PaletteConfig
    playback_fps: float
    temporal_subdivisions: int  # how many sub-frames to interpolate between source frames
    height_scale: float
    color_mode: str             # "height" | "sdf" | "curvature"
    show_wireframe: bool
    wireframe_color: tuple
    wireframe_width: float
    ground_color: tuple
    ground_margin: float
    default_mode: str           # "nif" | "physics"
    spatial_smooth_passes: int  # per-frame grid smoothing passes to kill heightfield staircasing


PRESETS: dict[str, DemoPreset] = {
    "research": DemoPreset(
        name="research",
        description="Honest heightfield + wireframe, wide framing. Shows extracted topology.",
        window_res=(1440, 900),
        background=(0.045, 0.055, 0.072),
        camera=CameraPose(
            position=(1.20, 1.55, 1.65),
            lookat=(0.0, 0.05, 0.05),
            fov=34.0,
        ),
        light=LightConfig(
            key_pos=(1.9, 2.2, 1.4),    key_color=(1.00, 0.93, 0.82),  key_intensity=0.80,
            fill_pos=(-1.6, 1.1, -1.3), fill_color=(0.56, 0.68, 0.85), fill_intensity=0.45,
            rim_pos=(0.0, 2.3, -1.8),   rim_color=(0.70, 0.78, 0.92),  rim_intensity=0.25,
            ambient=(0.20, 0.21, 0.23),
        ),
        palette=PaletteConfig(
            low=(0.96, 0.76, 0.52),
            mid=(0.88, 0.87, 0.83),
            high=(0.38, 0.55, 0.82),
            gamma=1.0,
        ),
        playback_fps=24.0,
        temporal_subdivisions=6,
        height_scale=1.05,
        color_mode="height",
        show_wireframe=True,
        wireframe_color=(0.14, 0.16, 0.20),
        wireframe_width=0.6,
        ground_color=(0.14, 0.15, 0.18),
        ground_margin=0.08,
        default_mode="nif",
        spatial_smooth_passes=1,
    ),
    "pitch": DemoPreset(
        name="pitch",
        description="Hero preset. Smooth playback, clean lighting, balanced framing.",
        window_res=(1600, 900),
        background=(0.028, 0.036, 0.054),
        camera=CameraPose(
            position=(0.95, 1.35, 1.70),
            lookat=(0.0, 0.05, 0.05),
            fov=32.0,
        ),
        light=LightConfig(
            key_pos=(2.0, 2.5, 1.5),    key_color=(1.05, 0.95, 0.80),  key_intensity=0.85,
            fill_pos=(-1.8, 1.2, -1.2), fill_color=(0.50, 0.66, 0.90), fill_intensity=0.45,
            rim_pos=(0.1, 2.4, -2.0),   rim_color=(0.80, 0.85, 0.95),  rim_intensity=0.28,
            ambient=(0.18, 0.19, 0.22),
        ),
        palette=PaletteConfig(
            low=(1.00, 0.78, 0.48),
            mid=(0.86, 0.86, 0.84),
            high=(0.32, 0.52, 0.85),
            gamma=0.88,
        ),
        playback_fps=30.0,
        temporal_subdivisions=10,
        height_scale=1.10,
        color_mode="height",
        show_wireframe=False,
        wireframe_color=(0.08, 0.09, 0.12),
        wireframe_width=0.5,
        ground_color=(0.11, 0.12, 0.15),
        ground_margin=0.09,
        default_mode="nif",
        spatial_smooth_passes=3,
    ),
    "dramatic": DemoPreset(
        name="dramatic",
        description="Close camera on the drape, strong key/fill contrast.",
        window_res=(1920, 1080),
        background=(0.018, 0.022, 0.036),
        camera=CameraPose(
            position=(0.90, 1.15, 1.55),
            lookat=(0.00, 0.05, 0.05),
            fov=32.0,
        ),
        light=LightConfig(
            key_pos=(1.8, 1.9, 0.9),    key_color=(1.10, 0.92, 0.72),  key_intensity=0.95,
            fill_pos=(-1.5, 0.6, -1.2), fill_color=(0.40, 0.58, 0.92), fill_intensity=0.35,
            rim_pos=(0.0, 1.2, -1.8),   rim_color=(0.55, 0.65, 0.88),  rim_intensity=0.22,
            ambient=(0.15, 0.16, 0.20),
        ),
        palette=PaletteConfig(
            low=(1.00, 0.72, 0.40),
            mid=(0.80, 0.80, 0.80),
            high=(0.24, 0.46, 0.90),
            gamma=0.75,
        ),
        playback_fps=30.0,
        temporal_subdivisions=12,
        height_scale=1.20,
        color_mode="height",
        show_wireframe=False,
        wireframe_color=(0.05, 0.06, 0.09),
        wireframe_width=0.6,
        ground_color=(0.08, 0.09, 0.12),
        ground_margin=0.10,
        default_mode="nif",
        spatial_smooth_passes=4,
    ),
}


@dataclass
class ClothSequences:
    nif_vertices: np.ndarray        # (T, N, 3), float32
    phys_vertices: np.ndarray       # (T, N, 3), float32
    faces: np.ndarray               # (F, 3), int32
    line_indices: np.ndarray        # (2F, ) int32 wireframe edges
    center: np.ndarray              # (3,)
    scale: float
    y_range: tuple[float, float]    # world y extent across both sequences (after normalization)


def _compute_line_indices(faces: np.ndarray) -> np.ndarray:
    """Build unique undirected-edge index list for wireframe rendering."""
    e0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    e1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    e2 = np.stack([faces[:, 2], faces[:, 0]], axis=1)
    edges = np.concatenate([e0, e1, e2], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.flatten().astype(np.int32)


def load_sequences(outputs_dir: Path) -> ClothSequences:
    """Load both NIF and physics heightfield sequences, normalise to a common frame.

    Source data uses x,y as grid, z as vertical. Taichi GGUI uses +Y as up, so
    we remap (x, y_src, z_src) -> (x, z_src, y_src): z becomes world y.
    """
    nif_raw = np.load(outputs_dir / "nif_prediction" / "vertices.npy").astype(np.float32)
    phys_raw = np.load(outputs_dir / "physics_baseline" / "vertices.npy").astype(np.float32)
    faces_nif = np.load(outputs_dir / "nif_prediction" / "faces.npy").astype(np.int32)
    faces_phys = np.load(outputs_dir / "physics_baseline" / "faces.npy").astype(np.int32)

    if not np.array_equal(faces_nif, faces_phys):
        raise ValueError("NIF and physics face indices differ; expected shared topology.")
    faces = faces_nif

    # Swap axes: source (x, y, z) -> world (x, z, y) so z is vertical.
    def remap(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr)
        out[..., 0] = arr[..., 0]
        out[..., 1] = arr[..., 2]
        out[..., 2] = arr[..., 1]
        return out

    nif = remap(nif_raw)
    phys = remap(phys_raw)

    # Center on xz (ground plane) with the union of both sequences; y center
    # is taken from the mean of start/end so the full drape stays visible.
    combined = np.concatenate([nif, phys], axis=0)
    xz_min = combined[..., [0, 2]].min(axis=(0, 1))
    xz_max = combined[..., [0, 2]].max(axis=(0, 1))
    xz_center = (xz_min + xz_max) * 0.5
    y_min = combined[..., 1].min()
    y_max = combined[..., 1].max()
    y_center = (y_min + y_max) * 0.5

    center = np.array([xz_center[0], y_center, xz_center[1]], dtype=np.float32)

    # Uniform scale: fit the xz extent to ~1.6 world units.
    span = float(max(xz_max[0] - xz_min[0], xz_max[1] - xz_min[1]))
    scale = 1.6 / max(span, 1e-6)

    nif = (nif - center) * scale
    phys = (phys - center) * scale

    y_range = (float(nif[..., 1].min()), float(nif[..., 1].max()))
    combined_y = np.concatenate([nif[..., 1], phys[..., 1]], axis=0)
    y_range = (float(combined_y.min()), float(combined_y.max()))

    line_indices = _compute_line_indices(faces)

    return ClothSequences(
        nif_vertices=np.ascontiguousarray(nif, dtype=np.float32),
        phys_vertices=np.ascontiguousarray(phys, dtype=np.float32),
        faces=np.ascontiguousarray(faces.flatten(), dtype=np.int32),
        line_indices=line_indices,
        center=center,
        scale=scale,
        y_range=y_range,
    )


def _smooth_heightfield(heights: np.ndarray, grid_res: int, passes: int) -> np.ndarray:
    """Apply a per-frame 3x3 box blur on a (T, grid*grid) height array.

    Used to suppress staircase artefacts from the discrete z-sampling in the
    heightfield extractor. Boundary is clamped (edges copied in) to keep the
    pinned-corner geometry stable.
    """
    if passes <= 0:
        return heights
    T = heights.shape[0]
    out = heights.reshape(T, grid_res, grid_res).copy()
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
        acc = np.zeros_like(out)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                acc += padded[:, 1 + dx : 1 + dx + grid_res, 1 + dy : 1 + dy + grid_res]
        out = acc / 9.0
    return out.reshape(T, grid_res * grid_res)


def diverging_colormap(
    heights: np.ndarray,
    y_lo: float,
    y_hi: float,
    palette: PaletteConfig,
) -> np.ndarray:
    """Map per-vertex height to a three-stop (low / mid / high) diverging colormap."""
    span = max(y_hi - y_lo, 1e-6)
    t = np.clip((heights - y_lo) / span, 0.0, 1.0)
    if palette.gamma != 1.0:
        t = np.power(t, palette.gamma)

    low = np.array(palette.low, dtype=np.float32)
    mid = np.array(palette.mid, dtype=np.float32)
    high = np.array(palette.high, dtype=np.float32)

    lo_mask = t < 0.5
    # [0, 0.5] -> low -> mid ; [0.5, 1] -> mid -> high
    t_lo = (t[lo_mask] / 0.5)[:, None]
    t_hi = ((t[~lo_mask] - 0.5) / 0.5)[:, None]

    colors = np.empty((t.shape[0], 3), dtype=np.float32)
    colors[lo_mask] = (1.0 - t_lo) * low + t_lo * mid
    colors[~lo_mask] = (1.0 - t_hi) * mid + t_hi * high
    return colors


def build_ground_plane(height: float, extent: float) -> tuple[np.ndarray, np.ndarray]:
    half = extent * 0.5
    vertices = np.array(
        [
            [-half, height, -half],
            [half,  height, -half],
            [-half, height,  half],
            [half,  height,  half],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
    return vertices, indices


def try_init_taichi(preferred: str) -> str:
    if ti is None:
        raise RuntimeError("Taichi is not installed. `pip install taichi`.")
    order = {
        "auto":   ["metal", "vulkan", "cuda", "cpu"],
        "metal":  ["metal", "cpu"],
        "vulkan": ["vulkan", "cpu"],
        "cuda":   ["cuda", "vulkan", "cpu"],
        "cpu":    ["cpu"],
    }[preferred]

    errors = []
    for arch_name in order:
        arch = getattr(ti, arch_name)
        try:
            ti.reset()
            ti.init(arch=arch, default_fp=ti.f32, offline_cache=True)
            return arch_name
        except Exception as exc:  # pragma: no cover
            errors.append(f"{arch_name}: {exc}")
    raise RuntimeError("Failed to initialize Taichi:\n" + "\n".join(errors))


class TaichiClothDemo:
    def __init__(
        self,
        sequences: ClothSequences,
        preset: DemoPreset,
        export_dir: Path | None,
        export_only: bool,
        export_fps: int,
        playback_fps: float,
        start_wireframe: bool,
        start_mode: str,
        hide_window: bool,
        show_gui: bool,
        title: str,
        max_steps: int | None,
    ):
        self.seq = sequences
        self.preset = preset
        self.export_dir = export_dir
        self.export_only = export_only
        self.export_fps = int(export_fps)
        self.source_fps = float(max(playback_fps, 1e-3))
        self.show_wireframe = start_wireframe
        self.show_window = not export_only and not hide_window
        self.show_gui = show_gui and self.show_window
        self.title = title
        self.max_steps = max_steps
        self.mode = start_mode  # "nif" or "physics"

        self.source_T = sequences.nif_vertices.shape[0]
        self.n_verts = sequences.nif_vertices.shape[1]
        self.n_faces = sequences.faces.shape[0] // 3

        # Temporal interpolation: expand to (source_T-1)*sub + 1 display frames.
        self.sub = max(int(preset.temporal_subdivisions), 1)
        self.n_display = (self.source_T - 1) * self.sub + 1 if self.source_T > 1 else 1
        self.display_span = float(max(self.n_display - 1, 0))

        self.frame_cursor = 0.0
        self.playing = True
        self.should_close = False
        self.exported_frames = 0
        self.last_tick = time.perf_counter()

        # Apply height scaling (in normalized-world y). Preserves the mid of y range.
        self._apply_height_scale()

        # Taichi fields.
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.triangle_indices = ti.field(dtype=ti.i32, shape=sequences.faces.shape[0])
        self.line_indices = ti.field(dtype=ti.i32, shape=sequences.line_indices.shape[0])
        self.triangle_indices.from_numpy(sequences.faces)
        self.line_indices.from_numpy(sequences.line_indices)

        # Ground plane sits slightly under the lowest drape point.
        ground_y = float(min(self.nif_disp[..., 1].min(), self.phys_disp[..., 1].min()) - preset.ground_margin)
        # Extent chosen so ground fills the field of view.
        ground_extent = 2.4
        gv, gi = build_ground_plane(ground_y, ground_extent)
        self.ground_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.ground_indices = ti.field(dtype=ti.i32, shape=6)
        self.ground_vertices.from_numpy(gv)
        self.ground_indices.from_numpy(gi)

        self.window = ti.ui.Window(
            name=self.title,
            res=self.preset.window_res,
            vsync=not export_only,
            show_window=self.show_window,
            fps_limit=int(round(preset.playback_fps)),
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._configure_camera()
        self._push_cursor(0.0)

    def _apply_height_scale(self) -> None:
        """Smooth heightfield staircasing, then amplify drape depth around the taut height."""
        scale = float(self.preset.height_scale)

        nif_disp = self.seq.nif_vertices.copy()
        phys_disp = self.seq.phys_vertices.copy()

        passes = int(self.preset.spatial_smooth_passes)
        if passes > 0:
            grid_res = int(round(math.sqrt(nif_disp.shape[1])))
            if grid_res * grid_res == nif_disp.shape[1]:
                nif_disp[..., 1] = _smooth_heightfield(nif_disp[..., 1], grid_res, passes)
                phys_disp[..., 1] = _smooth_heightfield(phys_disp[..., 1], grid_res, passes)

        nif_ref = float(nif_disp[0, :, 1].mean())
        phys_ref = float(phys_disp[0, :, 1].mean())
        nif_disp[..., 1] = nif_ref + scale * (nif_disp[..., 1] - nif_ref)
        phys_disp[..., 1] = phys_ref + scale * (phys_disp[..., 1] - phys_ref)

        self.nif_disp = nif_disp
        self.phys_disp = phys_disp

        combined_y = np.concatenate([self.nif_disp[..., 1], self.phys_disp[..., 1]], axis=0)
        self.y_lo = float(np.quantile(combined_y, 0.01))
        self.y_hi = float(np.quantile(combined_y, 0.99))

    def _configure_camera(self) -> None:
        pose = self.preset.camera
        self.camera.position(*pose.position)
        self.camera.lookat(*pose.lookat)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.fov(pose.fov)

    def _active_vertices(self) -> np.ndarray:
        return self.nif_disp if self.mode == "nif" else self.phys_disp

    def _interpolate_frame(self, cursor: float) -> tuple[np.ndarray, float, int]:
        """Interpolate per-vertex positions between two source frames."""
        src = self._active_vertices()
        if self.source_T == 1 or self.display_span <= 0.0:
            return src[0], 0.0, 0

        cursor = float(np.clip(cursor, 0.0, self.display_span))
        source_cursor = cursor / max(self.sub, 1)
        lo = int(math.floor(source_cursor))
        hi = min(lo + 1, self.source_T - 1)
        alpha = source_cursor - lo
        frame = (1.0 - alpha) * src[lo] + alpha * src[hi]
        normalized_time = cursor / self.display_span
        return frame.astype(np.float32), normalized_time, lo

    def _make_colors(self, frame: np.ndarray) -> np.ndarray:
        mode = self.preset.color_mode
        if mode == "height":
            return diverging_colormap(frame[:, 1], self.y_lo, self.y_hi, self.preset.palette)
        elif mode == "sdf":
            # Fallback: use z-normalized distance from the median plane.
            y = frame[:, 1]
            median = float(np.median(y))
            signed = y - median
            span = float(max(abs(signed.min()), abs(signed.max()), 1e-6))
            t = np.clip(signed / span * 0.5 + 0.5, 0.0, 1.0)
            return diverging_colormap(t, 0.0, 1.0, self.preset.palette)
        else:
            # curvature: cheap proxy = distance from plane through mean
            return diverging_colormap(frame[:, 1], self.y_lo, self.y_hi, self.preset.palette)

    def _push_cursor(self, cursor: float) -> None:
        frame, normalized_time, source_idx = self._interpolate_frame(cursor)
        self.vertices.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        self.colors.from_numpy(np.ascontiguousarray(self._make_colors(frame), dtype=np.float32))
        self.frame_cursor = float(cursor)
        self.current_time = float(normalized_time)
        self.source_idx = int(source_idx)

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
        elif key in ("m", "M"):
            self.mode = "physics" if self.mode == "nif" else "nif"
            self._push_cursor(self.frame_cursor)
            print(f"mode={self.mode}")
        elif key in ("w", "W"):
            self.show_wireframe = not self.show_wireframe
            print(f"wireframe={self.show_wireframe}")

    def _poll_events(self) -> None:
        if not self.show_window:
            return
        while self.window.get_event(ti.ui.PRESS):
            self._handle_press(self.window.event.key)

    def _advance(self) -> None:
        if not self.playing or self.display_span <= 0.0:
            return
        now = time.perf_counter()
        dt = max(0.0, min(now - self.last_tick, 0.25))
        self.last_tick = now
        # Display frames per second = source fps * sub
        display_fps = self.source_fps * max(self.sub, 1)
        next_cursor = self.frame_cursor + display_fps * dt
        next_cursor = next_cursor % (self.display_span + 1e-3)
        self._push_cursor(next_cursor)

    def _draw_gui(self) -> None:
        if not self.show_gui:
            return
        gui = self.window.get_gui()
        gui.begin("NIF-Cloth4D", 0.015, 0.015, 0.27, 0.17)
        gui.text(f"preset: {self.preset.name}")
        gui.text(f"mode:   {self.mode}")
        gui.text(f"t:      {self.current_time:.3f}")
        gui.text(f"source: {self.source_idx + 1}/{self.source_T}")
        gui.text("[space] play  [r] reset")
        gui.text("[m] NIF/phys  [w] wire")
        gui.end()

    def _render_scene(self) -> None:
        self.canvas.set_background_color(self.preset.background)
        self.scene.set_camera(self.camera)
        light = self.preset.light
        self.scene.ambient_light(light.ambient)
        self.scene.point_light(
            pos=light.key_pos,
            color=tuple(c * light.key_intensity for c in light.key_color),
        )
        self.scene.point_light(
            pos=light.fill_pos,
            color=tuple(c * light.fill_intensity for c in light.fill_color),
        )
        self.scene.point_light(
            pos=light.rim_pos,
            color=tuple(c * light.rim_intensity for c in light.rim_color),
        )

        self.scene.mesh(
            self.ground_vertices,
            indices=self.ground_indices,
            color=self.preset.ground_color,
            two_sided=True,
        )
        self.scene.mesh(
            self.vertices,
            indices=self.triangle_indices,
            per_vertex_color=self.colors,
            two_sided=True,
        )
        if self.show_wireframe:
            self.scene.lines(
                self.vertices,
                width=self.preset.wireframe_width,
                indices=self.line_indices,
                color=self.preset.wireframe_color,
            )
        self.canvas.scene(self.scene)
        self._draw_gui()

    def _export_frame(self) -> None:
        if self.export_dir is None:
            return
        path = self.export_dir / f"frame_{self.exported_frames:04d}.png"
        self.window.save_image(str(path))
        self.exported_frames += 1

    def _export_cursors(self, n_frames: int) -> np.ndarray:
        if n_frames <= 1 or self.display_span <= 0.0:
            return np.array([0.0], dtype=np.float32)
        return np.linspace(0.0, self.display_span, n_frames, dtype=np.float32)

    def _reopen_window(self) -> None:
        """Destroy and recreate the Taichi window. Used during offline export to
        sidestep the Metal backend's 50-swapchain-image limit for `save_image`.
        """
        try:
            self.window.destroy()
        except Exception:  # pragma: no cover - defensive
            pass
        self.window = ti.ui.Window(
            name=self.title,
            res=self.preset.window_res,
            vsync=not self.export_only,
            show_window=self.show_window,
            fps_limit=int(round(self.preset.playback_fps)),
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._configure_camera()

    def run(self, export_frames: int) -> None:
        if self.export_only:
            self.playing = False
            cursors = self._export_cursors(export_frames)
            for i, cursor in enumerate(cursors):
                # Metal backend caps swapchain images per window at ~50.
                # Recycle the window every 40 frames to stay under it.
                if i > 0 and i % 40 == 0:
                    self._reopen_window()
                self._push_cursor(float(cursor))
                self._render_scene()
                self._export_frame()
            return

        steps = 0
        while (self.window.running or not self.show_window) and not self.should_close:
            self._poll_events()
            if self.show_window:
                self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._advance()
            self._render_scene()
            self._export_frame()
            if self.show_window:
                self.window.show()
            else:
                time.sleep(1.0 / max(self.source_fps * self.sub, 1))
            steps += 1
            if self.max_steps is not None and steps >= self.max_steps:
                break

        self.window.destroy()


def ffmpeg_commands(export_dir: Path, fps: int) -> dict[str, str]:
    pattern = export_dir / "frame_%04d.png"
    mp4 = (
        f"ffmpeg -y -framerate {fps} -i {pattern} "
        f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p' "
        f"{export_dir / 'nif_cloth4d.mp4'}"
    )
    gif = (
        f"ffmpeg -y -framerate {fps} -i {pattern} "
        f"-vf 'fps={fps},scale=960:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen[p];[s1][p]paletteuse' "
        f"{export_dir / 'nif_cloth4d.gif'}"
    )
    return {"mp4": mp4, "gif": gif}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Taichi GGUI NIF-Cloth4D demo")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="pitch")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--outputs-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    p.add_argument("--mode", choices=["nif", "physics"], default=None)
    p.add_argument("--arch", choices=["auto", "metal", "vulkan", "cuda", "cpu"], default="auto")
    p.add_argument("--export-dir", type=str, default=None)
    p.add_argument("--export", action="store_true", help="Alias for --export-only")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--export-frames", type=int, default=180)
    p.add_argument("--export-fps", type=int, default=30)
    p.add_argument("--playback-fps", type=float, default=None)
    p.add_argument("--wireframe", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--hide-window", action="store_true")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--title", type=str, default="NIF-Cloth4D Demo")
    p.add_argument("--window-width", type=int, default=None)
    p.add_argument("--window-height", type=int, default=None)
    p.add_argument("--metadata-out", type=str, default=None)
    return p


def choose(value, fallback):
    return fallback if value is None else value


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.list_presets:
        print("Available presets:")
        for preset in PRESETS.values():
            print(f"- {preset.name}: {preset.description}")
        return 0

    preset = PRESETS[args.preset]
    window_res = (
        int(choose(args.window_width, preset.window_res[0])),
        int(choose(args.window_height, preset.window_res[1])),
    )
    preset = replace(preset, window_res=window_res)
    wireframe = bool(choose(args.wireframe, preset.show_wireframe))
    mode = choose(args.mode, preset.default_mode)
    playback_fps = float(choose(args.playback_fps, preset.playback_fps))
    export_only = args.export_only or args.export

    outputs_dir = Path(args.outputs_dir)
    sequences = load_sequences(outputs_dir)
    print(
        f"Loaded sequences: NIF {sequences.nif_vertices.shape}, "
        f"physics {sequences.phys_vertices.shape}, faces {sequences.faces.shape[0] // 3}, "
        f"center={sequences.center.tolist()}, scale={sequences.scale:.4f}"
    )

    arch_name = try_init_taichi(args.arch)
    print(f"Taichi arch: {arch_name}")

    export_dir = Path(args.export_dir) if args.export_dir else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_out) if args.metadata_out else None
    if metadata_path is None and export_dir is not None:
        metadata_path = export_dir / "metadata.json"
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "preset": preset.name,
            "arch": arch_name,
            "outputs_dir": str(outputs_dir.resolve()),
            "mode": mode,
            "wireframe": wireframe,
            "playback_fps": playback_fps,
            "export_frames": args.export_frames,
            "export_fps": args.export_fps,
            "window_res": list(window_res),
            "source_frames": int(sequences.nif_vertices.shape[0]),
            "vertices_per_frame": int(sequences.nif_vertices.shape[1]),
            "faces": int(sequences.faces.shape[0] // 3),
            "center": sequences.center.tolist(),
            "scale": float(sequences.scale),
            "y_range": list(sequences.y_range),
            "preset_config": asdict(preset),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

    demo = TaichiClothDemo(
        sequences=sequences,
        preset=preset,
        export_dir=export_dir,
        export_only=export_only,
        export_fps=args.export_fps,
        playback_fps=playback_fps,
        start_wireframe=wireframe,
        start_mode=mode,
        hide_window=args.hide_window,
        show_gui=not args.no_gui,
        title=args.title,
        max_steps=args.max_steps,
    )

    start = time.time()
    demo.run(export_frames=args.export_frames)
    elapsed = time.time() - start
    if export_dir is not None:
        commands = ffmpeg_commands(export_dir.resolve(), args.export_fps)
        print(f"Saved {demo.exported_frames} frame(s) to {export_dir.resolve()}")
        print(f"MP4: {commands['mp4']}")
        print(f"GIF: {commands['gif']}")
    print(f"Completed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
