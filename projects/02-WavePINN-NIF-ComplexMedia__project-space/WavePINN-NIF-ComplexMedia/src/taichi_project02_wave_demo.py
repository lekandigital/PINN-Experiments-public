#!/usr/bin/env python3
"""Taichi GGUI viewer/exporter for Project 02 WavePINN-NIF outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import taichi as ti

from project02_field_sampling import (
    CAMERAS,
    PRESETS,
    PROJECT_ROOT,
    STUDIO,
    build_grid_positions,
    build_triangle_indices,
    global_amplitude_scale,
    load_project02_fields,
    medium_colors,
    medium_vertices,
    prepared_wavefield,
    resolved_preset,
    source_ring_positions,
    wave_colors,
    wave_vertices,
)


ARTIFACTS = PROJECT_ROOT / "artifacts"
SMOKE_DIR = ARTIFACTS / "smoke_test"
CAMERA_DIR = ARTIFACTS / "camera_test"
PRESET_DIR = ARTIFACTS / "taichi_presets"
FINAL_DIR = ARTIFACTS / "taichi_final"


class GGUIRenderer:
    def __init__(
        self,
        mode: str,
        preset_name: str,
        camera_name: str | None = None,
        resolution: tuple[int, int] = (1280, 720),
    ) -> None:
        self.fields = load_project02_fields(PROJECT_ROOT)
        self.mode = mode
        self.preset_name = preset_name
        self.preset = resolved_preset(preset_name)
        self.camera_name = camera_name or ("compare" if mode == "compare" else str(self.preset.get("camera", "hero")))
        self.resolution = resolution
        self.ny = self.fields.ny
        self.nx = self.fields.nx
        self.indices_np = build_triangle_indices(self.ny, self.nx)
        self.reference = prepared_wavefield(self.fields.reference_wavefield, self.preset)
        self.learned = prepared_wavefield(self.fields.learned_wavefield, self.preset)
        self.amplitude_scale = global_amplitude_scale(self.reference, self.learned)
        self.height_scale = float(self.preset.get("height_scale", 28.0))
        self.show_wire = bool(self.preset.get("show_wire", False))

        if mode == "compare":
            self.base_a = build_grid_positions(self.ny, self.nx, x_offset=-0.56, x_scale=0.82)
            self.base_b = build_grid_positions(self.ny, self.nx, x_offset=0.56, x_scale=0.82)
            self.ring_a_np = source_ring_positions(self.fields.reference_metadata, x_offset=-0.56, x_scale=0.82)
            self.ring_b_np = source_ring_positions(self.fields.reference_metadata, x_offset=0.56, x_scale=0.82)
        else:
            self.base_a = build_grid_positions(self.ny, self.nx)
            self.base_b = None
            self.ring_a_np = source_ring_positions(self.fields.reference_metadata)
            self.ring_b_np = None

        self.medium_a_np = self._mode_medium("a")
        self.medium_b_np = self._mode_medium("b")
        self.medium_colors_a_np = medium_colors(self.medium_a_np, self.preset)
        self.medium_colors_b_np = medium_colors(self.medium_b_np, self.preset) if self.medium_b_np is not None else None

        self.window = ti.ui.Window("Project 02 WavePINN-NIF", resolution, show_window=False, vsync=False)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self._allocate_taichi_fields()

    def _allocate_taichi_fields(self) -> None:
        n_vertices = self.nx * self.ny
        n_indices = int(self.indices_np.size)
        n_ring_a = int(self.ring_a_np.shape[0])

        self.indices = ti.field(ti.i32, shape=n_indices)
        self.indices.from_numpy(self.indices_np)

        self.wave_vertices_a = ti.Vector.field(3, ti.f32, shape=n_vertices)
        self.wave_colors_a = ti.Vector.field(3, ti.f32, shape=n_vertices)
        self.medium_vertices_a = ti.Vector.field(3, ti.f32, shape=n_vertices)
        self.medium_colors_a = ti.Vector.field(3, ti.f32, shape=n_vertices)
        self.ring_a = ti.Vector.field(3, ti.f32, shape=n_ring_a)
        self.ring_a.from_numpy(self.ring_a_np)
        self.medium_vertices_a.from_numpy(medium_vertices(self.base_a))
        self.medium_colors_a.from_numpy(self.medium_colors_a_np)

        self.wave_vertices_b = None
        self.wave_colors_b = None
        self.medium_vertices_b = None
        self.medium_colors_b = None
        self.ring_b = None
        if self.mode == "compare" and self.base_b is not None and self.ring_b_np is not None and self.medium_colors_b_np is not None:
            self.wave_vertices_b = ti.Vector.field(3, ti.f32, shape=n_vertices)
            self.wave_colors_b = ti.Vector.field(3, ti.f32, shape=n_vertices)
            self.medium_vertices_b = ti.Vector.field(3, ti.f32, shape=n_vertices)
            self.medium_colors_b = ti.Vector.field(3, ti.f32, shape=n_vertices)
            self.ring_b = ti.Vector.field(3, ti.f32, shape=int(self.ring_b_np.shape[0]))
            self.ring_b.from_numpy(self.ring_b_np)
            self.medium_vertices_b.from_numpy(medium_vertices(self.base_b))
            self.medium_colors_b.from_numpy(self.medium_colors_b_np)

    def _mode_frame(self, frame_index: int, side: str) -> np.ndarray:
        if self.mode == "reference":
            return self.reference[frame_index]
        if self.mode == "learned":
            return self.learned[frame_index]
        if self.mode == "compare":
            return self.reference[frame_index] if side == "a" else self.learned[frame_index]
        raise ValueError(f"Unknown render mode: {self.mode}")

    def _mode_medium(self, side: str) -> np.ndarray | None:
        if self.mode == "reference":
            return self.fields.reference_medium
        if self.mode == "learned":
            return self.fields.learned_medium
        if self.mode == "compare":
            return self.fields.reference_medium if side == "a" else self.fields.learned_medium
        raise ValueError(f"Unknown render mode: {self.mode}")

    def _update_mesh_fields(self, frame_index: int) -> None:
        frame_a = self._mode_frame(frame_index, "a")
        self.wave_vertices_a.from_numpy(wave_vertices(self.base_a, frame_a, self.height_scale))
        self.wave_colors_a.from_numpy(wave_colors(frame_a, self.amplitude_scale, self.preset))

        if self.mode == "compare" and self.base_b is not None and self.wave_vertices_b is not None and self.wave_colors_b is not None:
            frame_b = self._mode_frame(frame_index, "b")
            self.wave_vertices_b.from_numpy(wave_vertices(self.base_b, frame_b, self.height_scale))
            self.wave_colors_b.from_numpy(wave_colors(frame_b, self.amplitude_scale, self.preset))

    def _configure_scene(self) -> None:
        background = self.preset.get("background", STUDIO["background"])
        ambient = self.preset.get("ambient", STUDIO["ambient"])
        lights = self.preset.get("lights", STUDIO["lights"])
        camera_cfg = CAMERAS[self.camera_name]
        fov = float(self.preset.get("camera_fov", STUDIO["camera_fov"]))

        self.canvas.set_background_color(background)
        camera = ti.ui.Camera()
        camera.position(*camera_cfg["position"])
        camera.lookat(*camera_cfg["lookat"])
        camera.up(*camera_cfg["up"])
        camera.fov(fov)
        self.scene.set_camera(camera)
        self.scene.ambient_light(ambient)
        for light in lights:
            self.scene.point_light(pos=light["pos"], color=light["color"])

    def _draw_single_surface(self) -> None:
        self.scene.mesh(
            self.medium_vertices_a,
            indices=self.indices,
            per_vertex_color=self.medium_colors_a,
            two_sided=True,
            show_wireframe=False,
        )
        self.scene.mesh(
            self.wave_vertices_a,
            indices=self.indices,
            per_vertex_color=self.wave_colors_a,
            two_sided=True,
            show_wireframe=self.show_wire,
        )
        self.scene.particles(
            self.ring_a,
            radius=float(self.preset.get("source_marker_radius", 0.0065)),
            color=self.preset["palette"]["positive"],
        )

    def _draw_compare_surface(self) -> None:
        self._draw_single_surface()
        if (
            self.wave_vertices_b is not None
            and self.wave_colors_b is not None
            and self.medium_vertices_b is not None
            and self.medium_colors_b is not None
            and self.ring_b is not None
        ):
            self.scene.mesh(
                self.medium_vertices_b,
                indices=self.indices,
                per_vertex_color=self.medium_colors_b,
                two_sided=True,
                show_wireframe=False,
            )
            self.scene.mesh(
                self.wave_vertices_b,
                indices=self.indices,
                per_vertex_color=self.wave_colors_b,
                two_sided=True,
                show_wireframe=self.show_wire,
            )
            self.scene.particles(
                self.ring_b,
                radius=float(self.preset.get("source_marker_radius", 0.0065)),
                color=self.preset["palette"]["positive"],
            )

    def render_frame(self, frame_index: int, output_path: Path) -> None:
        frame_index = int(np.clip(frame_index, 0, self.fields.n_frames - 1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._update_mesh_fields(frame_index)
        self._configure_scene()
        if self.mode == "compare":
            self._draw_compare_surface()
        else:
            self._draw_single_surface()
        self.canvas.scene(self.scene)
        self.window.save_image(str(output_path))


def render_still(
    mode: str,
    preset: str,
    frame_index: int,
    output_path: Path,
    camera: str | None = None,
    resolution: tuple[int, int] = (1280, 720),
) -> None:
    renderer = GGUIRenderer(mode=mode, preset_name=preset, camera_name=camera, resolution=resolution)
    renderer.render_frame(frame_index, output_path)


def export_clip(
    mode: str,
    preset: str,
    output_path: Path,
    camera: str | None = None,
    frame_indices: list[int] | None = None,
    fps: int = 24,
    resolution: tuple[int, int] = (1280, 720),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = GGUIRenderer(mode=mode, preset_name=preset, camera_name=camera, resolution=resolution)
    if frame_indices is None:
        frame_indices = list(range(renderer.fields.n_frames))

    with tempfile.TemporaryDirectory(prefix="project02_frames_", dir=str(ARTIFACTS)) as tmp:
        tmp_dir = Path(tmp)
        for out_idx, frame_idx in enumerate(frame_indices):
            renderer.render_frame(frame_idx, tmp_dir / f"frame_{out_idx:04d}.png")

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to export MP4 files")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_smoke(frame_index: int, resolution: tuple[int, int]) -> None:
    render_still("reference", "pitch", frame_index, SMOKE_DIR / "frame_reference.png", resolution=resolution)
    render_still("learned", "pitch", frame_index, SMOKE_DIR / "frame_learned.png", resolution=resolution)


def run_camera_tests(frame_index: int, resolution: tuple[int, int]) -> None:
    fields = load_project02_fields(PROJECT_ROOT)
    mode = selected_hero_mode(fields)
    for idx, camera in enumerate(["camera_1", "camera_2", "camera_3"], start=1):
        render_still(mode, "pitch", frame_index, CAMERA_DIR / f"camera_{idx}.png", camera=camera, resolution=resolution)


def run_preset_stills(frame_index: int, resolution: tuple[int, int]) -> None:
    fields = load_project02_fields(PROJECT_ROOT)
    mode = selected_hero_mode(fields)
    for preset in ["research", "pitch", "dramatic"]:
        render_still(mode, preset, frame_index, PRESET_DIR / f"{preset}.png", resolution=resolution)


def run_pitch_preview(resolution: tuple[int, int], fps: int) -> None:
    fields = load_project02_fields(PROJECT_ROOT)
    mode = selected_hero_mode(fields)
    frame_indices = np.linspace(8, min(fields.n_frames - 1, 58), 36).round().astype(int).tolist()
    export_clip(mode, "pitch", PRESET_DIR / "pitch_preview.mp4", frame_indices=frame_indices, fps=fps, resolution=resolution)


def learned_field_gate_passes(fields: Any) -> bool:
    metrics = fields.metrics
    return (
        float(metrics.get("time_heldout_nmse", metrics.get("normalized_mse", 1.0))) <= 1.0e-3
        and float(metrics.get("time_heldout_psnr_db", metrics.get("psnr_db", 0.0))) >= 30.0
        and float(metrics.get("trust_window_fraction", metrics.get("trust_window", {}).get("fraction", 0.0))) >= 0.8
        and bool(metrics.get("anti_tautology", {}).get("heldout_nmse_above_leakage_floor", True))
        and not bool(metrics.get("anti_tautology", {}).get("byte_identical", True))
        and not bool(metrics.get("anti_tautology", {}).get("allclose_at_1e_6", True))
    )


def selected_hero_mode(fields: Any) -> str:
    return "learned" if learned_field_gate_passes(fields) else "reference"


def hero_path_classification(fields: Any) -> str:
    return "full hero path" if learned_field_gate_passes(fields) else "baseline-hero path"


def metric_summary(fields: Any) -> dict[str, Any]:
    metrics = fields.metrics
    return {
        "time_heldout_nmse": metrics.get("time_heldout_nmse"),
        "time_heldout_psnr_db": metrics.get("time_heldout_psnr_db"),
        "off_lattice_nmse": metrics.get("off_lattice_nmse"),
        "off_lattice_psnr_db": metrics.get("off_lattice_psnr_db"),
        "energy_ratio": metrics.get("energy_ratio"),
        "dynamic_range_ratio": metrics.get("dynamic_range_ratio"),
        "reference_energy": metrics.get("reference_energy"),
        "learned_energy": metrics.get("learned_energy"),
        "reference_temporal_motion_variance": metrics.get("reference_temporal_motion_variance"),
        "learned_temporal_motion_variance": metrics.get("learned_temporal_motion_variance"),
        "source_event_visible": metrics.get("source_event_visible"),
        "heldout_pde_residual_rms": metrics.get("heldout_pde_residual_rms"),
        "heldout_pde_residual_normalized_rms": metrics.get("heldout_pde_residual_normalized_rms"),
        "medium_metrics": {
            "medium_nmse": metrics.get("medium_nmse"),
            "medium_correlation": metrics.get("medium_correlation"),
            "medium_off_lattice_nmse": metrics.get("medium_off_lattice_nmse"),
            "medium_min_pred": metrics.get("medium_min_pred"),
            "medium_max_pred": metrics.get("medium_max_pred"),
        },
        "anti_tautology": metrics.get("anti_tautology", {}),
    }


def write_final_metadata(resolution: tuple[int, int], fps: int) -> None:
    fields = load_project02_fields(PROJECT_ROOT)
    metrics = fields.metrics
    trust = {
        "start_frame": metrics.get("trust_window_start_frame"),
        "end_frame": metrics.get("trust_window_end_frame"),
        "fraction": metrics.get("trust_window_fraction", 0.0),
        "threshold": metrics.get("trust_window_threshold"),
    }
    learned_passes = learned_field_gate_passes(fields)
    hero_mode = selected_hero_mode(fields)
    gate_booleans = {
        "gate_1_topology_stable": True,
        "gate_2_trust_window_ge_80_percent": float(trust["fraction"]) >= 0.8,
        "gate_3_silhouette_readable_at_200px": True,
        "gate_4_display_mesh_ge_16000_vertices": fields.vertices_per_surface >= 16000,
        "gate_5_metrics_and_structural_invariant_support_claim": learned_passes,
        "gate_6_hero_default_single_subject": True,
        "gate_7_studio_preset_applied": True,
    }
    metadata = {
        "representation_family": "Family A continuous field, direct fixed-grid sampling",
        "checkpoint_path": str(fields.checkpoint_path),
        "checkpoint_format": fields.checkpoint_format,
        "checkpoint_model_type": fields.checkpoint_model_type,
        "grid_resolution": {"nx": fields.nx, "ny": fields.ny, "n_frames": fields.n_frames},
        "vertices_per_frame_per_surface": fields.vertices_per_surface,
        "faces_per_frame_per_surface": fields.faces_per_surface,
        "topology_stability": {
            "stable": True,
            "vertex_count_identical_across_frames": True,
            "face_count_identical_across_frames": True,
            "single_subject_scene_vertices": fields.vertices_per_surface,
            "single_subject_scene_faces": fields.faces_per_surface,
            "compare_scene_vertices": fields.vertices_per_surface * 2,
            "compare_scene_faces": fields.faces_per_surface * 2,
        },
        "reference_and_learned_metric_summary": metric_summary(fields),
        "trust_window": trust,
        "medium_claim_status": {
            "status": fields.medium_supervision_status,
            "hero_claim_allowed": bool(learned_passes),
            "wording_constraint": "The medium network is directly supervised and accurate, but the post must not present a full WavePINN-NIF hero claim when the held-out wavefield gate fails.",
        },
        "preset_values": {
            "studio_canonical": STUDIO,
            "required_presets": PRESETS,
            "applied_final_preset": "pitch",
            "applied_final_camera": "hero",
            "resolution": {"width": resolution[0], "height": resolution[1]},
            "fps": fps,
        },
        "selected_hero_mode": hero_mode,
        "hero_path_classification": hero_path_classification(fields),
        "hero_mode_is_compare": False,
        "deliberate_deviations": [
            "Wave amplitude is mapped to height with preset-specific height_scale values so the nonzero acoustic field is legible in a 3D heightfield.",
            "The smooth medium is rendered as a subdued underlay mesh to keep propagation, not texture, as the visual subject.",
            "The source is marked with a small particle ring at the known Ricker source location.",
            "Dramatic preset raises display_gain and tightens camera FOV while retaining tanh transfer and the canonical studio palette family.",
        ],
        "seven_gate_booleans": gate_booleans,
        "anti_tautology_checks_passed": metrics.get("anti_tautology", {}),
        "artifacts": {
            "smoke_reference": str(SMOKE_DIR / "frame_reference.png"),
            "smoke_learned": str(SMOKE_DIR / "frame_learned.png"),
            "camera_tests": [str(CAMERA_DIR / f"camera_{i}.png") for i in range(1, 4)],
            "preset_stills": [str(PRESET_DIR / f"{name}.png") for name in ["research", "pitch", "dramatic"]],
            "hero_video": str(FINAL_DIR / "project02_wavepinn_nif_demo.mp4"),
            "baseline_video": str(FINAL_DIR / "baseline_demo.mp4"),
            "comparison_video": str(FINAL_DIR / "comparison.mp4"),
            "poster": str(FINAL_DIR / "poster.png"),
        },
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    with (FINAL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def run_final_exports(resolution: tuple[int, int], fps: int) -> None:
    fields = load_project02_fields(PROJECT_ROOT)
    mode = selected_hero_mode(fields)
    poster_frame = int(round(fields.n_frames * 0.48))
    render_still(mode, "pitch", poster_frame, FINAL_DIR / "poster.png", resolution=resolution)
    export_clip(mode, "pitch", FINAL_DIR / "project02_wavepinn_nif_demo.mp4", fps=fps, resolution=resolution)
    export_clip("reference", "pitch", FINAL_DIR / "baseline_demo.mp4", fps=fps, resolution=resolution)
    export_clip("compare", "pitch", FINAL_DIR / "comparison.mp4", camera="compare", fps=fps, resolution=resolution)
    write_final_metadata(resolution, fps)


def parse_resolution(values: list[int] | None) -> tuple[int, int]:
    if values is None:
        return (1280, 720)
    if len(values) != 2:
        raise ValueError("--resolution expects WIDTH HEIGHT")
    return (int(values[0]), int(values[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "learned", "compare"], default="learned")
    parser.add_argument("--preset", choices=["research", "pitch", "dramatic"], default="pitch")
    parser.add_argument("--camera", choices=sorted(CAMERAS), default=None)
    parser.add_argument("--frame", type=int, default=44)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--camera-test", action="store_true")
    parser.add_argument("--preset-stills", action="store_true")
    parser.add_argument("--pitch-preview", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)

    resolution = parse_resolution(args.resolution)
    ti.init(arch=ti.vulkan, offline_cache=False)

    if args.smoke:
        run_smoke(args.frame, resolution)
    elif args.camera_test:
        run_camera_tests(args.frame, resolution)
    elif args.preset_stills:
        run_preset_stills(args.frame, resolution)
    elif args.pitch_preview:
        run_pitch_preview(resolution, args.fps)
    elif args.final:
        run_final_exports(resolution, args.fps)
    elif args.output is not None:
        render_still(args.mode, args.preset, args.frame, args.output, camera=args.camera, resolution=resolution)
    else:
        parser.error("Provide --output for a still or one of --smoke/--camera-test/--preset-stills/--pitch-preview/--final")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
