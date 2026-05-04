#!/usr/bin/env python3
"""
Taichi GGUI demo for NIF-Cloth4D-Temporal (Project 12, hero path).

The viewer renders the fixed-topology heightfield mesh extracted from the
trained neural implicit field and, optionally, the analytic reference SDF or
a side-by-side comparison. Rendering choices are driven by three presets
(``research``, ``pitch``, ``dramatic``); ``pitch`` and ``dramatic`` apply the
STUDIO rig verbatim from the Prompt 2 spec.

Data source strategy (per the Prompt-2 plan):
- default: read the pre-extracted NIF SDF volume
  ``outputs/validation_model/field_sequence.npy`` and call the shared
  zero-crossing extractor once at startup.
- ``--live_query``: load the checkpoint
  ``checkpoints/hero/nif_cloth4d_temporal.pt`` and forward-pass a dense lattice
  per frame through ``extract_heightfield_from_model``. Slower but
  demonstrates that the learned field can be sampled at arbitrary resolution /
  time.
- reference mode always goes through the shared extractor applied to
  ``outputs/reference/reference_sdf_sequence.npy``.

Offline export writes per-frame PNGs to ``--output_dir/frames/`` at
``playback_speed=0.72`` timing. Assembling MP4 / GIF is a Prompt-3 concern;
this viewer only prints the canonical ``ffmpeg`` invocation for later use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import taichi as ti
except ImportError:  # pragma: no cover
    ti = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling.heightfield_extractor import (  # noqa: E402
    build_heightfield_mesh,
    extract_heightfield_from_model,
    extract_heightfield_from_volume,
)


# ---------------------------------------------------------------------------
# STUDIO rig (verbatim from the Prompt 2 spec — must pass Gate 7).
# ---------------------------------------------------------------------------

STUDIO = {
    "background":   (0.02, 0.026, 0.04),
    "ground_color": (0.09, 0.10, 0.13),
    "ambient":      (0.22, 0.23, 0.25),
    "lights": [
        {"pos": (2.2, 2.5,  1.2),  "color": (1.10, 1.00, 0.92)},  # warm key
        {"pos": (-1.8, 1.0, -1.3), "color": (0.20, 0.32, 0.52)},  # cool fill
        {"pos": (0.0, 0.55, 2.4),  "color": (0.09, 0.10, 0.12)},  # dim back
    ],
    "camera_fov":         24.0,
    "display_transfer":   "tanh",
    "display_gain":       1.3,
    "playback_speed":     0.72,
    "palette": {
        "neutral":        (0.76, 0.80, 0.86),
        "positive":       (0.99, 0.70, 0.40),
        "negative":       (0.18, 0.49, 0.88),
        "strength_power": 0.72,
    },
    "spatial_smooth_passes":  3,
    "temporal_smooth_passes": 1,
}


# ---------------------------------------------------------------------------
# Preset dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPose:
    position: tuple
    lookat: tuple
    fov: float


@dataclass(frozen=True)
class LightRig:
    ambient: tuple
    lights: tuple       # tuple of (pos, color) pairs
    # Taichi's point_light attenuates with inverse-square distance. STUDIO
    # colours are specified at unit magnitude; at the camera distances required
    # by fov=24 the raw colours render nearly black. ``intensity`` scales all
    # point_light colours uniformly at draw time — it does not modify the
    # STUDIO palette stored in metadata.
    intensity: float = 1.0


@dataclass(frozen=True)
class PaletteRig:
    neutral: tuple
    positive: tuple    # warm "positive deviation" colour (low / underside)
    negative: tuple    # cool "negative deviation" colour (high / topside)
    strength_power: float
    display_transfer: str  # "linear" or "tanh"
    display_gain: float


@dataclass(frozen=True)
class DemoPreset:
    name: str
    description: str
    window_res: tuple
    background: tuple
    ground_color: tuple
    camera: CameraPose
    light: LightRig
    palette: PaletteRig
    playback_speed: float
    spatial_smooth_passes: int
    temporal_smooth_passes: int
    height_scale: float
    show_wireframe: bool
    wireframe_color: tuple
    wireframe_width: float
    ground_margin: float
    apply_studio: bool


def _studio_camera(position, lookat, fov=STUDIO["camera_fov"]) -> CameraPose:
    return CameraPose(position=tuple(position), lookat=tuple(lookat), fov=float(fov))


def _studio_light(intensity: float = 1.0) -> LightRig:
    return LightRig(
        ambient=tuple(STUDIO["ambient"]),
        lights=tuple((tuple(L["pos"]), tuple(L["color"])) for L in STUDIO["lights"]),
        intensity=float(intensity),
    )


def _studio_palette(display_gain=None) -> PaletteRig:
    pal = STUDIO["palette"]
    return PaletteRig(
        neutral=tuple(pal["neutral"]),
        positive=tuple(pal["positive"]),
        negative=tuple(pal["negative"]),
        strength_power=float(pal["strength_power"]),
        display_transfer=str(STUDIO["display_transfer"]),
        display_gain=float(display_gain if display_gain is not None else STUDIO["display_gain"]),
    )


# Camera presets used by the Prompt-2 camera test. Positions are in
# mesh-world coordinates (world-y = scene height).
# Camera poses are tuned for the STUDIO fov=24 constraint. The scene is 2 units
# wide (x, y in [-1, 1]) so the camera needs to sit ~5.5 world units from the
# lookat point for the drape bump to fill the frame without overfilling it.
_LOOKAT = (0.0, 0.25, 0.0)
CAMERAS = {
    # 1: high three-quarter. Shows the full drape footprint with a gentle top-down read.
    1: CameraPose(position=(2.80, 3.60, 3.80), lookat=_LOOKAT, fov=STUDIO["camera_fov"]),
    # 2: hero three-quarter. Emphasises the bump silhouette through the drape.
    2: CameraPose(position=(3.30, 2.10, 3.10), lookat=_LOOKAT, fov=STUDIO["camera_fov"]),
    # 3: near-front profile. Strongest silhouette at thumbnail size.
    3: CameraPose(position=(4.20, 1.55, 1.40), lookat=_LOOKAT, fov=STUDIO["camera_fov"]),
}


def _make_preset_pitch(camera: CameraPose) -> DemoPreset:
    return DemoPreset(
        name="pitch",
        description="Hero preset. STUDIO rig + pitch camera.",
        window_res=(1600, 1000),
        background=tuple(STUDIO["background"]),
        ground_color=tuple(STUDIO["ground_color"]),
        camera=camera,
        light=_studio_light(),
        palette=_studio_palette(),
        playback_speed=float(STUDIO["playback_speed"]),
        spatial_smooth_passes=int(STUDIO["spatial_smooth_passes"]),
        temporal_smooth_passes=int(STUDIO["temporal_smooth_passes"]),
        height_scale=1.25,
        show_wireframe=False,
        wireframe_color=(0.08, 0.10, 0.13),
        wireframe_width=0.5,
        ground_margin=0.08,
        apply_studio=True,
    )


def _make_preset_dramatic(camera: CameraPose) -> DemoPreset:
    # dramatic narrows the camera and raises gain above pitch; STUDIO rig otherwise identical.
    closer = CameraPose(
        position=(camera.position[0] * 0.75,
                  camera.position[1] * 0.85,
                  camera.position[2] * 0.75),
        lookat=camera.lookat,
        fov=STUDIO["camera_fov"],
    )
    return DemoPreset(
        name="dramatic",
        description="Tighter framing, stronger tanh gain, same STUDIO rig as pitch.",
        window_res=(1920, 1080),
        background=tuple(STUDIO["background"]),
        ground_color=tuple(STUDIO["ground_color"]),
        camera=closer,
        light=_studio_light(intensity=1.3),
        palette=_studio_palette(display_gain=1.6),  # > STUDIO default 1.3
        playback_speed=float(STUDIO["playback_speed"]),
        spatial_smooth_passes=int(STUDIO["spatial_smooth_passes"]),
        temporal_smooth_passes=int(STUDIO["temporal_smooth_passes"]),
        height_scale=1.35,
        show_wireframe=False,
        wireframe_color=(0.05, 0.06, 0.09),
        wireframe_width=0.5,
        ground_margin=0.08,
        apply_studio=True,
    )


def _make_preset_research() -> DemoPreset:
    # Neutral research rig: flat lighting, wireframe on, wider camera.
    # Explicitly does NOT apply STUDIO so the raw heightfield fidelity is visible.
    return DemoPreset(
        name="research",
        description="Honest heightfield + wireframe; neutral lighting; exposes extraction fidelity.",
        window_res=(1440, 900),
        background=(0.07, 0.08, 0.10),
        ground_color=(0.14, 0.15, 0.18),
        camera=CameraPose(position=(2.60, 2.40, 2.80), lookat=(0.0, 0.28, 0.0), fov=36.0),
        light=LightRig(
            ambient=(0.32, 0.33, 0.34),
            lights=(
                ((2.0, 3.0, 2.0), (0.85, 0.85, 0.85)),
                ((-2.0, 1.5, -2.0), (0.40, 0.45, 0.55)),
            ),
            intensity=1.0,
        ),
        palette=PaletteRig(
            neutral=(0.82, 0.84, 0.86),
            positive=(0.98, 0.80, 0.50),
            negative=(0.50, 0.68, 0.86),
            strength_power=1.0,
            display_transfer="linear",
            display_gain=1.0,
        ),
        playback_speed=1.0,
        spatial_smooth_passes=0,
        temporal_smooth_passes=0,
        height_scale=1.0,
        show_wireframe=True,
        wireframe_color=(0.12, 0.14, 0.18),
        wireframe_width=0.5,
        ground_margin=0.06,
        apply_studio=False,
    )


def build_presets(camera_idx: int = 2) -> dict[str, DemoPreset]:
    cam = CAMERAS[int(camera_idx)]
    return {
        "research": _make_preset_research(),
        "pitch":    _make_preset_pitch(cam),
        "dramatic": _make_preset_dramatic(cam),
    }


# ---------------------------------------------------------------------------
# Height post-processing: smoothing + height scaling.
# ---------------------------------------------------------------------------


def _spatial_smooth(heights: np.ndarray, passes: int) -> np.ndarray:
    """3x3 box blur on a (T, Nx, Ny) heightfield, edge-padded."""
    if passes <= 0:
        return heights
    out = heights.copy()
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
        acc = np.zeros_like(out)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                acc += padded[:, 1 + dx:1 + dx + out.shape[1], 1 + dy:1 + dy + out.shape[2]]
        out = acc / 9.0
    return out.astype(np.float32)


def _temporal_smooth(heights: np.ndarray, passes: int) -> np.ndarray:
    """1x3 temporal box blur along the T axis."""
    if passes <= 0 or heights.shape[0] < 3:
        return heights
    out = heights.copy()
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0), (0, 0)), mode="edge")
        out = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    return out.astype(np.float32)


def _apply_height_scale(heights: np.ndarray, scale: float) -> np.ndarray:
    """Amplify drape depth about the first-frame mean (≈ flat-sheet height)."""
    if abs(scale - 1.0) < 1e-6:
        return heights
    ref = float(heights[0].mean())
    return (ref + scale * (heights - ref)).astype(np.float32)


# ---------------------------------------------------------------------------
# Colour mapping.
# ---------------------------------------------------------------------------


def _tanh_gain(t: np.ndarray, gain: float) -> np.ndarray:
    """Apply tanh display transfer centred at 0.5, gain controls slope."""
    centered = (t - 0.5) * gain
    return 0.5 + 0.5 * np.tanh(centered)


def diverging_height_colormap(
    heights: np.ndarray,   # (N,) world-y per vertex
    y_lo: float,
    y_hi: float,
    palette: PaletteRig,
) -> np.ndarray:
    """
    Three-stop diverging colormap: negative -> neutral -> positive (warm=low, cool=high).

    The mapping is: lower heights -> warm (positive colour), higher heights ->
    cool (negative colour). This matches the STUDIO intent — warm on the
    receded cloth, cool on the taut portions.
    """
    span = max(y_hi - y_lo, 1e-6)
    raw = np.clip((heights - y_lo) / span, 0.0, 1.0)

    if palette.strength_power != 1.0:
        # Reshape strength away from the midpoint.
        signed = raw - 0.5
        mag = np.abs(signed) * 2.0
        mag = np.power(mag, palette.strength_power)
        raw = 0.5 + np.sign(signed) * mag * 0.5

    if palette.display_transfer == "tanh":
        raw = _tanh_gain(raw, palette.display_gain)
    elif palette.display_gain != 1.0:
        centered = (raw - 0.5) * palette.display_gain
        raw = np.clip(0.5 + centered, 0.0, 1.0)

    warm = np.asarray(palette.positive, dtype=np.float32)
    neut = np.asarray(palette.neutral, dtype=np.float32)
    cool = np.asarray(palette.negative, dtype=np.float32)

    lo_mask = raw < 0.5
    out = np.empty((raw.shape[0], 3), dtype=np.float32)
    t_lo = (raw[lo_mask] / 0.5)[:, None]
    t_hi = ((raw[~lo_mask] - 0.5) / 0.5)[:, None]
    out[lo_mask] = (1.0 - t_lo) * warm + t_lo * neut
    out[~lo_mask] = (1.0 - t_hi) * neut + t_hi * cool
    return out


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------


@dataclass
class HeightfieldSequence:
    heights: np.ndarray      # (T, Nx, Ny) float32, world units
    coords_x: np.ndarray     # (Nx,) float32
    coords_y: np.ndarray     # (Ny,) float32
    times: np.ndarray        # (T,) float32 in source normalization (0..1)
    label: str               # "nif" | "reference"


def _load_volume_sequence(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected SDF volume at {path}. Run Prompt-1 export first or "
            f"pass --live_query."
        )
    arr = np.load(path, mmap_mode="r")
    return arr


def extract_sequence_from_volume(
    sdf_volume: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    label: str,
    progress_prefix: str = "",
) -> HeightfieldSequence:
    T = int(sdf_volume.shape[0])
    Nx = int(sdf_volume.shape[1])
    Ny = int(sdf_volume.shape[2])
    heights = np.zeros((T, Nx, Ny), dtype=np.float32)
    for i in range(T):
        heights[i] = extract_heightfield_from_volume(
            sdf_volume[i].astype(np.float32), coords
        )
        if progress_prefix and ((i + 1) % max(T // 5, 1) == 0 or i == T - 1):
            print(f"  {progress_prefix} frame {i + 1}/{T}", flush=True)
    return HeightfieldSequence(
        heights=heights,
        coords_x=coords.astype(np.float32),
        coords_y=coords.astype(np.float32),
        times=times.astype(np.float32),
        label=label,
    )


def extract_sequence_from_model(
    checkpoint_path: Path,
    coords: np.ndarray,
    times: np.ndarray,
    device_pref: str = "auto",
    z_samples: int = 96,
) -> HeightfieldSequence:
    import torch

    from src.models.fourier_mlp import FourierFeatureMLP  # lazy

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")

    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(device_pref)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = FourierFeatureMLP(
        in_dim=4,
        hidden_dim=cfg.get("hidden_dim", 256),
        out_dim=1,
        num_layers=cfg.get("num_layers", 5),
        num_freqs=cfg.get("num_freqs", 32),
        fourier_scale=cfg.get("fourier_scale", 1.0),
        omega_0=cfg.get("omega_0", 5.0),
        use_gru=bool(cfg.get("use_gru", False)) and not bool(cfg.get("no_gru", True)),
        gru_hidden=cfg.get("gru_hidden", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[live_query] loaded {checkpoint_path} on {device}; "
          f"T={len(times)} frames x {z_samples} z-taps per column")

    T = int(len(times))
    Nx = int(coords.shape[0])
    heights = np.zeros((T, Nx, Nx), dtype=np.float32)
    for i, t in enumerate(times):
        heights[i] = extract_heightfield_from_model(
            model, float(t), coords, device=device, z_samples=z_samples
        )
        if (i + 1) % max(T // 5, 1) == 0 or i == T - 1:
            print(f"  [live_query] frame {i + 1}/{T}", flush=True)
    return HeightfieldSequence(
        heights=heights,
        coords_x=coords.astype(np.float32),
        coords_y=coords.astype(np.float32),
        times=np.asarray(times, dtype=np.float32),
        label="nif",
    )


# ---------------------------------------------------------------------------
# Taichi backend init and helpers.
# ---------------------------------------------------------------------------


def init_taichi(preferred: str) -> str:
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
        arch = getattr(ti, arch_name, None)
        if arch is None:
            continue
        try:
            ti.reset()
            ti.init(arch=arch, default_fp=ti.f32, offline_cache=True)
            return arch_name
        except Exception as exc:  # pragma: no cover
            errors.append(f"{arch_name}: {exc}")
    raise RuntimeError("Failed to initialize Taichi:\n" + "\n".join(errors))


def _build_ground(extent: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    half = extent * 0.5
    v = np.array(
        [[-half, y, -half], [half, y, -half], [-half, y, half], [half, y, half]],
        dtype=np.float32,
    )
    i = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
    return v, i


def _remap_xy_z_to_xz_y(vertices_xyz: np.ndarray) -> np.ndarray:
    """Source axis order (x, y, z=height) -> Taichi world order (x, height=z, y)."""
    out = np.empty_like(vertices_xyz)
    out[..., 0] = vertices_xyz[..., 0]
    out[..., 1] = vertices_xyz[..., 2]
    out[..., 2] = vertices_xyz[..., 1]
    return out


# ---------------------------------------------------------------------------
# Viewer.
# ---------------------------------------------------------------------------


class TemporalClothViewer:
    def __init__(
        self,
        sequences: list[HeightfieldSequence],
        preset: DemoPreset,
        mode: str,               # "nif" | "reference" | "compare"
        offline: bool,
        output_dir: Path,
        frames: int,
        width: int,
        height: int,
        show_window: bool,
        title: str = "NIF-Cloth4D-Temporal",
    ) -> None:
        if ti is None:
            raise RuntimeError("Taichi missing; cannot construct viewer.")

        self.preset = preset
        self.mode = mode
        self.offline = offline
        self.output_dir = output_dir
        self.show_window = show_window and not offline
        self.title = title
        self.width = int(width or preset.window_res[0])
        self.height = int(height or preset.window_res[1])

        # --- Prepare heightfields ---
        self.sequences: list[HeightfieldSequence] = []
        for seq in sequences:
            h = seq.heights
            h = _spatial_smooth(h, preset.spatial_smooth_passes)
            h = _temporal_smooth(h, preset.temporal_smooth_passes)
            h = _apply_height_scale(h, preset.height_scale)
            self.sequences.append(HeightfieldSequence(
                heights=h,
                coords_x=seq.coords_x,
                coords_y=seq.coords_y,
                times=seq.times,
                label=seq.label,
            ))

        # Ensure all sequences share grid + time axis.
        self.T = self.sequences[0].heights.shape[0]
        self.Nx, self.Ny = self.sequences[0].heights.shape[1:]
        for s in self.sequences[1:]:
            if s.heights.shape != (self.T, self.Nx, self.Ny):
                raise ValueError("compare mode: sequence shapes differ")

        # How many display frames. Default = source length; scale with --frames.
        self.frames = int(frames if frames > 0 else self.T)
        # Playback: the *output* frame index -> source index, timed by playback_speed.
        self.display_times = np.linspace(
            0.0, float(self.T - 1), self.frames, dtype=np.float32
        )

        # Build fixed-topology mesh once per sequence (vertices update, indices constant).
        self.meshes = []
        for seq in self.sequences:
            verts_xyz, faces, lines = build_heightfield_mesh(
                seq.heights[0], seq.coords_x, seq.coords_y
            )
            # Remap to world axes used by the Taichi scene.
            verts_world = _remap_xy_z_to_xz_y(verts_xyz)
            self.meshes.append({
                "faces": faces.reshape(-1),
                "lines": lines,
                "verts": verts_world,
                "seq": seq,
            })

        # Layout: compare mode offsets sequences left/right in world x.
        if mode == "compare" and len(self.sequences) == 2:
            # Use the x-extent of the grid + margin as offset.
            x_extent = float(self.sequences[0].coords_x[-1] - self.sequences[0].coords_x[0])
            self.offsets = [
                np.array([-0.5 * (x_extent + 0.15), 0.0, 0.0], dtype=np.float32),
                np.array([+0.5 * (x_extent + 0.15), 0.0, 0.0], dtype=np.float32),
            ]
        else:
            self.offsets = [np.zeros(3, dtype=np.float32)] * len(self.sequences)

        # Shared height range (for color mapping + ground placement). Computed
        # across the FULL time-series — sampling only the initial frame would
        # collapse the palette to neutral because frame 0 is a flat sheet.
        all_heights = np.concatenate([s.heights for s in self.sequences], axis=0)
        self.y_lo = float(np.quantile(all_heights, 0.02))
        self.y_hi = float(np.quantile(all_heights, 0.98))
        self.ground_y = float(all_heights.min() - preset.ground_margin)

        # --- Taichi resources ---
        self._ti_resources = []
        for idx, m in enumerate(self.meshes):
            n_v = m["verts"].shape[0]
            verts_f = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
            colors_f = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
            tri_f = ti.field(dtype=ti.i32, shape=m["faces"].shape[0])
            line_f = ti.field(dtype=ti.i32, shape=m["lines"].shape[0])
            tri_f.from_numpy(m["faces"])
            line_f.from_numpy(m["lines"])
            self._ti_resources.append({
                "verts": verts_f,
                "colors": colors_f,
                "tri": tri_f,
                "line": line_f,
                "n_v": n_v,
            })

        # Ground plane.
        extent = max(2.4, float(self.sequences[0].coords_x[-1] - self.sequences[0].coords_x[0]) * 1.6)
        if mode == "compare":
            extent *= 2.2
        gv, gi = _build_ground(extent, self.ground_y)
        self.ground_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.ground_idx = ti.field(dtype=ti.i32, shape=6)
        self.ground_verts.from_numpy(gv)
        self.ground_idx.from_numpy(gi)

        self._init_window()

    # ------------------------------------------------------------------
    # Window / camera lifecycle.
    # ------------------------------------------------------------------

    def _init_window(self) -> None:
        self.window = ti.ui.Window(
            name=self.title,
            res=(self.width, self.height),
            vsync=not self.offline,
            show_window=self.show_window,
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._configure_camera()

    def _configure_camera(self) -> None:
        pose = self.preset.camera
        self.camera.position(*pose.position)
        self.camera.lookat(*pose.lookat)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.fov(pose.fov)

    def _reopen_window(self) -> None:
        """Work around Metal's ~50-swapchain-image cap for long save_image runs."""
        try:
            self.window.destroy()
        except Exception:  # pragma: no cover
            pass
        self._init_window()

    # ------------------------------------------------------------------
    # Per-frame update.
    # ------------------------------------------------------------------

    def _push_frame(self, cursor: float) -> None:
        src = int(round(cursor))
        src = max(0, min(self.T - 1, src))
        for idx, res in enumerate(self._ti_resources):
            seq = self.meshes[idx]["seq"]
            verts_xyz = np.stack(
                [
                    np.broadcast_to(seq.coords_x[:, None], (self.Nx, self.Ny)),
                    np.broadcast_to(seq.coords_y[None, :], (self.Nx, self.Ny)),
                    seq.heights[src],
                ],
                axis=-1,
            ).reshape(-1, 3).astype(np.float32)
            verts_world = _remap_xy_z_to_xz_y(verts_xyz)
            verts_world = verts_world + self.offsets[idx][None, :]
            res["verts"].from_numpy(np.ascontiguousarray(verts_world))
            colors = diverging_height_colormap(
                verts_world[:, 1], self.y_lo, self.y_hi, self.preset.palette
            )
            res["colors"].from_numpy(np.ascontiguousarray(colors))

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------

    def _render(self) -> None:
        self.canvas.set_background_color(self.preset.background)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.preset.light.ambient)
        intensity = float(self.preset.light.intensity)
        for pos, col in self.preset.light.lights:
            scaled = tuple(float(c) * intensity for c in col)
            self.scene.point_light(pos=pos, color=scaled)

        self.scene.mesh(
            self.ground_verts,
            indices=self.ground_idx,
            color=self.preset.ground_color,
            two_sided=True,
        )
        for res in self._ti_resources:
            self.scene.mesh(
                res["verts"],
                indices=res["tri"],
                per_vertex_color=res["colors"],
                two_sided=True,
            )
            if self.preset.show_wireframe:
                self.scene.lines(
                    res["verts"],
                    width=self.preset.wireframe_width,
                    indices=res["line"],
                    color=self.preset.wireframe_color,
                )
        self.canvas.scene(self.scene)

    # ------------------------------------------------------------------
    # Public entry points.
    # ------------------------------------------------------------------

    def render_still(self, path: Path, cursor: float = 0.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._push_frame(cursor)
        self._render()
        self.window.save_image(str(path))
        print(f"[still] wrote {path}")

    def run_offline(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, cursor in enumerate(self.display_times):
            if i > 0 and i % 40 == 0:
                # Metal swapchain cap workaround.
                self._reopen_window()
            self._push_frame(float(cursor))
            self._render()
            self.window.save_image(str(frames_dir / f"frame_{i:04d}.png"))
        print(f"[offline] wrote {self.frames} frames to {frames_dir}")

        # Emit the canonical ffmpeg command for Prompt 3.
        effective_fps = 30.0 * self.preset.playback_speed
        cmd = (
            f"ffmpeg -y -framerate {effective_fps:.3f} "
            f"-i {frames_dir}/frame_%04d.png "
            f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
            f"-c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p "
            f"{output_dir}/{self.mode}_demo.mp4"
        )
        print("[ffmpeg]", cmd)

    def run_live(self, max_steps: Optional[int] = None) -> None:
        last_tick = time.perf_counter()
        cursor = 0.0
        steps = 0
        while self.window.running:
            now = time.perf_counter()
            dt = max(0.0, min(now - last_tick, 0.2))
            last_tick = now
            # playback_speed in source-frame units per real second.
            cursor = (cursor + dt * 30.0 * self.preset.playback_speed) % max(self.T - 1, 1)
            self._push_frame(cursor)
            self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._render()
            self.window.show()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break


# ---------------------------------------------------------------------------
# CLI glue.
# ---------------------------------------------------------------------------


def _default_nif_volume() -> Path:
    return PROJECT_ROOT / "outputs" / "validation_model" / "field_sequence.npy"


def _default_reference_volume() -> Path:
    return PROJECT_ROOT / "outputs" / "reference" / "reference_sdf_sequence.npy"


def _load_axes(outputs_ref_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = np.load(outputs_ref_dir / "coords.npy")
    times = np.load(PROJECT_ROOT / "outputs" / "validation_model" / "times.npy")
    return coords.astype(np.float32), times.astype(np.float32)


def load_sequences_for_mode(
    mode: str,
    checkpoint: Path,
    live_query: bool,
    nif_volume_path: Path,
    reference_volume_path: Path,
) -> list[HeightfieldSequence]:
    """Load one or two HeightfieldSequences per the requested mode."""
    ref_dir = PROJECT_ROOT / "outputs" / "reference"
    coords, times = _load_axes(ref_dir)

    def _load_nif() -> HeightfieldSequence:
        if live_query:
            return extract_sequence_from_model(checkpoint, coords, times)
        if not nif_volume_path.exists():
            raise FileNotFoundError(
                f"NIF SDF volume missing: {nif_volume_path}. Re-run Prompt-1 "
                f"export or pass --live_query."
            )
        sdf = _load_volume_sequence(nif_volume_path)
        return extract_sequence_from_volume(
            sdf, coords, times, label="nif", progress_prefix="[nif]"
        )

    def _load_ref() -> HeightfieldSequence:
        if not reference_volume_path.exists():
            raise FileNotFoundError(
                f"Reference SDF volume missing: {reference_volume_path}."
            )
        sdf = _load_volume_sequence(reference_volume_path)
        return extract_sequence_from_volume(
            sdf, coords, times, label="reference", progress_prefix="[ref]"
        )

    if mode == "nif":
        return [_load_nif()]
    if mode == "reference":
        return [_load_ref()]
    if mode == "compare":
        return [_load_ref(), _load_nif()]
    raise ValueError(f"Unknown mode {mode!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Taichi GGUI NIF-Cloth4D-Temporal demo")
    p.add_argument("--preset", choices=["research", "pitch", "dramatic"], default="pitch")
    p.add_argument("--mode", choices=["nif", "reference", "compare"], default="nif")
    p.add_argument("--offline", action="store_true",
                   help="Run without a live window; dump per-frame PNGs.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--grid", type=int, default=128,
                   help="xy lattice resolution (display mesh side length).")
    p.add_argument("--frames", type=int, default=0,
                   help="Display frame count. 0 = take from times.npy.")
    p.add_argument("--seed", type=int, default=0)
    # Supporting flags.
    p.add_argument("--checkpoint", type=str,
                   default=str(PROJECT_ROOT / "checkpoints" / "hero" / "nif_cloth4d_temporal.pt"))
    p.add_argument("--use_precomputed_volume", type=str, default=None,
                   help="Override path for NIF SDF volume (cached path). "
                        "Ignored when --live_query is set.")
    p.add_argument("--reference_volume", type=str, default=None,
                   help="Override path for reference SDF volume.")
    p.add_argument("--live_query", action="store_true",
                   help="Load the checkpoint and forward-pass the lattice per frame.")
    p.add_argument("--arch", choices=["auto", "metal", "vulkan", "cuda", "cpu"],
                   default="auto")
    p.add_argument("--save_still", type=str, default=None,
                   help="Path to a single-frame PNG (skips --frames loop).")
    p.add_argument("--still_cursor", type=float, default=0.0,
                   help="Source frame index used for --save_still.")
    p.add_argument("--camera_preset", type=int, choices=sorted(CAMERAS.keys()), default=3,
                   help="1|2|3 — which CAMERAS entry to use for pitch/dramatic. "
                        "Default 3 (near-front profile) was chosen in the Prompt-2 "
                        "camera test because its silhouette reads strongest at the "
                        "200 px thumbnail size required by Gate 3.")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap live-playback steps (debug).")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    np.random.seed(int(args.seed))

    if ti is None:
        raise RuntimeError("Taichi is not installed; install via `pip install taichi`.")

    nif_vol = Path(args.use_precomputed_volume) if args.use_precomputed_volume else _default_nif_volume()
    ref_vol = Path(args.reference_volume) if args.reference_volume else _default_reference_volume()

    # Load sequences first (pure numpy + optional torch). Doing this before ti.init
    # means any "missing file" failure is loud and surfaces cleanly.
    sequences = load_sequences_for_mode(
        mode=args.mode,
        checkpoint=Path(args.checkpoint),
        live_query=bool(args.live_query),
        nif_volume_path=nif_vol,
        reference_volume_path=ref_vol,
    )

    arch = init_taichi(args.arch)
    print(f"[taichi] init arch={arch}")

    presets = build_presets(camera_idx=args.camera_preset)
    preset = presets[args.preset]

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "artifacts" / "taichi_final"
    )

    viewer = TemporalClothViewer(
        sequences=sequences,
        preset=preset,
        mode=args.mode,
        offline=bool(args.offline or args.save_still),
        output_dir=output_dir,
        frames=int(args.frames),
        width=int(args.width),
        height=int(args.height),
        show_window=False if (args.save_still or args.offline) else True,
        title=f"NIF-Cloth4D-Temporal ({preset.name}/{args.mode})",
    )

    if args.save_still:
        viewer.render_still(Path(args.save_still), cursor=float(args.still_cursor))
        return

    if args.offline:
        viewer.run_offline(output_dir)
        return

    viewer.run_live(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
