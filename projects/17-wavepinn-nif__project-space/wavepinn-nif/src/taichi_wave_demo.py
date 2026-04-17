#!/usr/bin/env python3
"""
Taichi GGUI wave demo for WavePINN.

Primary workflow:
- load a trained checkpoint
- precompute a short wavefield sequence
- render a fixed-topology heightfield mesh in Taichi GGUI
- support preset-driven live playback and deterministic offline export
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
except ImportError:  # pragma: no cover - runtime dependency
    ti = None

from wavefield_sampling import amplitude_color_field, sample_wavefield_bundle


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
class PaletteConfig:
    neutral: tuple[float, float, float]
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    strength_power: float


@dataclass(frozen=True)
class DemoPreset:
    name: str
    description: str
    resolution: int
    frames: int
    export_frames: int
    export_fps: int
    window_res: tuple[int, int]
    height_scale: float
    amplitude_percentile: float
    center_mode: str
    spatial_smooth: int
    temporal_smooth: int
    height_transfer: str
    height_gain: float
    playback_speed: float
    extent: float
    camera: CameraPose
    background: tuple[float, float, float]
    ambient: tuple[float, float, float]
    lights: tuple[LightConfig, ...]
    ground_color: tuple[float, float, float]
    ground_margin: float
    wireframe_default: bool
    wireframe_color: tuple[float, float, float]
    wireframe_width: float
    palette: PaletteConfig


PRESETS: dict[str, DemoPreset] = {
    "research": DemoPreset(
        name="research",
        description="Conservative surface scale with a cleaner, more analytical presentation.",
        resolution=144,
        frames=96,
        export_frames=120,
        export_fps=30,
        window_res=(1440, 900),
        height_scale=0.075,
        amplitude_percentile=99.6,
        center_mode="global",
        spatial_smooth=1,
        temporal_smooth=0,
        height_transfer="tanh",
        height_gain=1.00,
        playback_speed=0.90,
        extent=2.55,
        camera=CameraPose(
            position=(-0.06, 1.88, 1.92),
            lookat=(0.02, 0.0, 0.02),
            fov=26.0,
        ),
        background=(0.070, 0.085, 0.110),
        ambient=(0.29, 0.30, 0.31),
        lights=(
            LightConfig(pos=(1.8, 2.2, 1.6), color=(0.95, 0.95, 0.93)),
            LightConfig(pos=(-1.4, 1.2, -1.2), color=(0.30, 0.36, 0.42)),
        ),
        ground_color=(0.17, 0.19, 0.22),
        ground_margin=0.055,
        wireframe_default=True,
        wireframe_color=(0.16, 0.18, 0.21),
        wireframe_width=0.34,
        palette=PaletteConfig(
            neutral=(0.80, 0.83, 0.87),
            positive=(0.90, 0.63, 0.46),
            negative=(0.35, 0.52, 0.79),
            strength_power=0.90,
        ),
    ),
    "pitch": DemoPreset(
        name="pitch",
        description="Balanced hero preset for calls, demos, and deck captures.",
        resolution=160,
        frames=120,
        export_frames=180,
        export_fps=30,
        window_res=(1600, 900),
        height_scale=0.10,
        amplitude_percentile=99.7,
        center_mode="global",
        spatial_smooth=2,
        temporal_smooth=1,
        height_transfer="tanh",
        height_gain=1.10,
        playback_speed=0.82,
        extent=2.60,
        camera=CameraPose(
            position=(-0.10, 1.92, 2.18),
            lookat=(0.02, -0.01, 0.02),
            fov=27.0,
        ),
        background=(0.040, 0.055, 0.085),
        ambient=(0.27, 0.28, 0.30),
        lights=(
            LightConfig(pos=(2.0, 2.4, 1.8), color=(1.02, 0.99, 0.94)),
            LightConfig(pos=(-1.6, 1.2, -1.0), color=(0.32, 0.40, 0.50)),
            LightConfig(pos=(0.0, 0.85, 2.4), color=(0.14, 0.16, 0.18)),
        ),
        ground_color=(0.14, 0.16, 0.19),
        ground_margin=0.060,
        wireframe_default=False,
        wireframe_color=(0.07, 0.08, 0.10),
        wireframe_width=0.60,
        palette=PaletteConfig(
            neutral=(0.78, 0.82, 0.87),
            positive=(0.96, 0.67, 0.44),
            negative=(0.26, 0.51, 0.84),
            strength_power=0.78,
        ),
    ),
    "dramatic": DemoPreset(
        name="dramatic",
        description="Higher contrast and stronger displacement for prerecorded hero clips.",
        resolution=192,
        frames=144,
        export_frames=210,
        export_fps=30,
        window_res=(1920, 1080),
        height_scale=0.125,
        amplitude_percentile=99.75,
        center_mode="global",
        spatial_smooth=3,
        temporal_smooth=1,
        height_transfer="tanh",
        height_gain=1.30,
        playback_speed=0.72,
        extent=2.45,
        camera=CameraPose(
            position=(-0.54, 1.52, 1.78),
            lookat=(0.10, -0.02, 0.06),
            fov=24.0,
        ),
        background=(0.020, 0.026, 0.040),
        ambient=(0.22, 0.23, 0.25),
        lights=(
            LightConfig(pos=(2.2, 2.5, 1.2), color=(1.10, 1.00, 0.92)),
            LightConfig(pos=(-1.8, 1.0, -1.3), color=(0.20, 0.32, 0.52)),
            LightConfig(pos=(0.0, 0.55, 2.4), color=(0.09, 0.10, 0.12)),
        ),
        ground_color=(0.09, 0.10, 0.13),
        ground_margin=0.065,
        wireframe_default=False,
        wireframe_color=(0.05, 0.06, 0.08),
        wireframe_width=0.70,
        palette=PaletteConfig(
            neutral=(0.76, 0.80, 0.86),
            positive=(0.99, 0.70, 0.40),
            negative=(0.18, 0.49, 0.88),
            strength_power=0.72,
        ),
    ),
}


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


def build_ground_plane(height: float, extent: float) -> tuple[np.ndarray, np.ndarray]:
    half = extent * 0.5
    vertices = np.array(
        [
            [-half, height, -half],
            [half, height, -half],
            [-half, height, half],
            [half, height, half],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
    return vertices, indices


def ffmpeg_commands(export_dir: Path, fps: int) -> dict[str, str]:
    pattern = export_dir / "frame_%04d.png"
    mp4_path = export_dir / "wave_demo.mp4"
    gif_path = export_dir / "wave_demo.gif"
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


def print_presets() -> None:
    print("Available presets:")
    for preset in PRESETS.values():
        print(f"- {preset.name}: {preset.description}")


def choose(value, fallback):
    return fallback if value is None else value


def serialize_preset(preset: DemoPreset) -> dict:
    return asdict(preset)


class TaichiWaveDemo:
    def __init__(
        self,
        bundle,
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
        self.status_every = max(status_every, 1)
        self.title = title
        self.max_steps = max_steps

        self.clip_span = float(max(self.bundle.n_frames - 1, 0))
        self.source_frame_rate = float(max(self.fps, 1))
        self.frame_cursor = 0.0
        self.frame_idx = 0
        self.current_time = float(self.bundle.times[0])
        self.playing = True
        self.should_close = False
        self.exported_frames = 0
        self.last_status_marker = -1
        self.last_tick = time.perf_counter()

        n_vertices = bundle.base_positions.shape[0]
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=n_vertices)
        self.triangle_indices = ti.field(dtype=ti.i32, shape=bundle.triangle_indices.shape[0])
        self.line_indices = ti.field(dtype=ti.i32, shape=bundle.line_indices.shape[0])

        self.triangle_indices.from_numpy(bundle.triangle_indices)
        self.line_indices.from_numpy(bundle.line_indices)

        ground_y = float(bundle.display_fields.min() - self.preset.ground_margin)
        ground_vertices, ground_indices = build_ground_plane(
            ground_y,
            extent=self.bundle.extent * 1.45,
        )
        self.ground_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.ground_indices = ti.field(dtype=ti.i32, shape=6)
        self.ground_vertices.from_numpy(ground_vertices)
        self.ground_indices.from_numpy(ground_indices)

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

    def _interpolate_field(self, cursor: float) -> tuple[np.ndarray, float, int]:
        if self.bundle.n_frames == 1 or self.clip_span <= 0.0:
            return self.bundle.display_fields[0], float(self.bundle.times[0]), 0

        cursor = float(np.clip(cursor, 0.0, self.clip_span))
        lower = int(np.floor(cursor))
        upper = min(lower + 1, self.bundle.n_frames - 1)
        alpha = cursor - lower
        field = (1.0 - alpha) * self.bundle.display_fields[lower] + alpha * self.bundle.display_fields[upper]
        time_value = (1.0 - alpha) * self.bundle.times[lower] + alpha * self.bundle.times[upper]
        return field.astype(np.float32), float(time_value), lower

    def _make_vertices(self, frame: np.ndarray) -> np.ndarray:
        vertices = self.bundle.base_positions.copy()
        vertices[:, 1] = frame.reshape(-1)
        return vertices.astype(np.float32)

    def _make_colors(self, frame: np.ndarray) -> np.ndarray:
        palette = self.preset.palette
        colors = amplitude_color_field(
            frame,
            self.bundle.color_reference,
            neutral=palette.neutral,
            warm=palette.positive,
            cool=palette.negative,
            strength_power=palette.strength_power,
        )
        return colors.reshape(-1, 3).astype(np.float32)

    def _push_cursor(self, cursor: float) -> None:
        frame, current_time, frame_idx = self._interpolate_field(cursor)
        self.vertices.from_numpy(self._make_vertices(frame))
        self.colors.from_numpy(self._make_colors(frame))
        self.frame_cursor = float(cursor)
        self.frame_idx = int(frame_idx)
        self.current_time = float(current_time)

    def _print_status(self, force: bool = False) -> None:
        status_marker = int(round(self.frame_cursor * 1000.0))
        if not force and status_marker == self.last_status_marker:
            return
        if self.export_only:
            should_print = self.exported_frames % self.status_every == 0
        else:
            should_print = self.frame_idx % self.status_every == 0
        if force or should_print:
            print(
                f"frame={self.frame_cursor:07.3f}/{self.clip_span:07.3f} "
                f"time={self.current_time:.4f} "
                f"speed={self.playback_speed:.2f} "
                f"wire={self.show_wireframe}"
            )
            self.last_status_marker = status_marker

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
        gui.begin("Wave Demo", 0.015, 0.015, 0.26, 0.16)
        gui.text(f"preset: {self.preset.name}")
        gui.text(f"time: {self.current_time:.3f}")
        gui.text(f"speed: {self.playback_speed:.2f}x")
        gui.text(f"sample frame: {self.frame_idx + 1}/{self.bundle.n_frames}")
        gui.text("[space] play  [r] reset  [[ ]] speed  [w] wire")
        gui.end()

    def _render_scene(self) -> None:
        self.canvas.set_background_color(self.preset.background)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.preset.ambient)
        for light in self.preset.lights:
            self.scene.point_light(pos=light.pos, color=light.color)

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
            return

        steps = 0
        while (self.window.running or not self.show_window) and not self.should_close:
            self._poll_events()
            if self.show_window:
                self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._advance()
            self._render_scene()
            self._export_frame()
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
    parser = argparse.ArgumentParser(description="Taichi GGUI wave demo for WavePINN")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="pitch", help="Render preset")
    parser.add_argument("--list-presets", action="store_true", help="Print preset names and exit")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--resolution", type=int, default=None, help="Sampling grid resolution")
    parser.add_argument("--frames", type=int, default=None, help="Number of precomputed source wave frames")
    parser.add_argument("--export-frames", type=int, default=None, help="Number of rendered frames to export")
    parser.add_argument("--export-fps", type=int, default=None, help="Target playback fps for exported frames")
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--extent", type=float, default=None, help="World-space mesh width/depth")
    parser.add_argument("--height-scale", type=float, default=None, help="Max visual surface height after autoscaling")
    parser.add_argument("--amplitude-percentile", type=float, default=None, help="Percentile used for autoscaling")
    parser.add_argument("--center-mode", choices=["none", "global", "per_frame"], default=None)
    parser.add_argument("--spatial-smooth", type=int, default=None, help="Number of spatial smoothing passes")
    parser.add_argument("--temporal-smooth", type=int, default=None, help="Number of temporal smoothing passes")
    parser.add_argument("--height-transfer", choices=["tanh", "linear"], default=None, help="Height mapping for display")
    parser.add_argument("--height-gain", type=float, default=None, help="Gain used by the tanh height transfer")
    parser.add_argument("--window-width", type=int, default=None)
    parser.add_argument("--window-height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--playback-speed", type=float, default=None)
    parser.add_argument(
        "--wireframe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the wireframe overlay",
    )
    parser.add_argument("--arch", choices=["auto", "cuda", "vulkan", "metal", "cpu"], default="auto")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for rendered PNG frames")
    parser.add_argument("--export-only", action="store_true", help="Render a deterministic off-screen frame sequence and exit")
    parser.add_argument("--hide-window", action="store_true", help="Run the normal playback loop without showing a window")
    parser.add_argument("--max-steps", type=int, default=None, help="Auto-exit after N loop iterations in live mode")
    parser.add_argument("--status-every", type=int, default=8, help="Terminal status interval")
    parser.add_argument("--title", type=str, default="WavePINN Taichi Demo")
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
    wireframe = bool(choose(args.wireframe, preset.wireframe_default))

    arch_name = try_init_taichi(args.arch)
    print(f"Taichi arch: {arch_name}")

    bundle = sample_wavefield_bundle(
        checkpoint_path=args.checkpoint,
        resolution=choose(args.resolution, preset.resolution),
        n_frames=choose(args.frames, preset.frames),
        t_start=args.t_start,
        t_end=args.t_end,
        height_scale=choose(args.height_scale, preset.height_scale),
        amplitude_percentile=choose(args.amplitude_percentile, preset.amplitude_percentile),
        center_mode=choose(args.center_mode, preset.center_mode),
        spatial_smooth_passes=choose(args.spatial_smooth, preset.spatial_smooth),
        temporal_smooth_passes=choose(args.temporal_smooth, preset.temporal_smooth),
        display_transfer=choose(args.height_transfer, preset.height_transfer),
        display_gain=choose(args.height_gain, preset.height_gain),
        extent=choose(args.extent, preset.extent),
    )

    export_dir = None
    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_out) if args.metadata_out else None
    if metadata_path is None and export_dir is not None:
        metadata_path = export_dir / "metadata.json"

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "preset": preset.name,
            "arch": arch_name,
            "checkpoint_path": str(bundle.checkpoint_path),
            "torch_device": bundle.torch_device,
            "resolution": bundle.resolution,
            "frames": bundle.n_frames,
            "export_frames": export_frames,
            "export_fps": export_fps,
            "window_res": list(window_res),
            "t_start": bundle.t_start,
            "t_end": bundle.t_end,
            "extent": bundle.extent,
            "height_scale": bundle.height_scale,
            "amplitude_reference": bundle.amplitude_reference,
            "scale_factor": bundle.scale_factor,
            "display_transfer": bundle.display_transfer,
            "display_gain": bundle.display_gain,
            "center_mode": bundle.center_mode,
            "spatial_smooth_passes": bundle.spatial_smooth_passes,
            "temporal_smooth_passes": bundle.temporal_smooth_passes,
            "playback_speed": playback_speed,
            "wireframe": wireframe,
            "preset_config": serialize_preset(preset),
            "model_config": bundle.model_config,
            "training_args": bundle.training_args,
            "final_metrics": bundle.final_metrics,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

    demo = TaichiWaveDemo(
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
    )

    start = time.time()
    demo.run()
    elapsed = time.time() - start
    if export_dir is not None:
        commands = ffmpeg_commands(export_dir.resolve(), export_fps)
        print(f"Saved {demo.exported_frames} rendered frame(s) to {export_dir.resolve()}")
        print(f"MP4: {commands['mp4']}")
        print(f"GIF: {commands['gif']}")
        print(
            "Helper: "
            f"python src/make_demo_video.py --frames-dir {export_dir.resolve()} --fps {export_fps}"
        )
    print(f"Completed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
