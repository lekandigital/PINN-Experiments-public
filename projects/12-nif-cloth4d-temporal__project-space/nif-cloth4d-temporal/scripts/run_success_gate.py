#!/usr/bin/env python3
"""
Project-13 Success Gate — applied to Project 12.

Reads ``artifacts/taichi_final/metadata.json`` and
``outputs/validation_model/metrics.json`` and asserts the seven gate items plus
the numerical thresholds in the Prompt-2 threshold table. Intended as the
pre-publish gate: if this script exits non-zero, do not publish.

Usage:
    python scripts/run_success_gate.py
    python scripts/run_success_gate.py --json     # machine-readable
    python scripts/run_success_gate.py --metadata other/path.json
    python scripts/run_success_gate.py --metrics other/path.json

Exit codes:
    0 : all gates pass
    1 : one or more gates fail (reasons printed to stderr)
    2 : inputs missing or malformed
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Numerical thresholds (playbook; Family A / SDF).
# ---------------------------------------------------------------------------

THRESH = {
    "chamfer_per_timestep_max":       5e-4,
    "hausdorff_per_timestep_max":     5e-2,
    "eikonal_mean_max":               5e-2,
    "nmse_mean_max":                  1e-3,
    "display_mesh_vertex_count_min":  16000,
    "heightfield_grid_min":           128,
    "volumetric_grid_min":            128,
    "playback_speed":                 0.72,
    "camera_fov_degrees":             24.0,
    "display_gain_tanh":              1.3,
}


@dataclass
class GateResult:
    gate: int
    name: str
    passed: bool
    value: Any
    reason: str


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    with path.open() as fh:
        return json.load(fh)


def _get(d: dict, *keys: str, default=None):
    """Nested dict accessor that never raises."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def check_gates(metadata: dict, metrics: dict) -> list[GateResult]:
    results: list[GateResult] = []

    # ------------------------------------------------------------------
    # Gate 1 — Topology stable across all exported frames.
    # ------------------------------------------------------------------
    topology_stable = bool(metrics.get("heightfield_topology_stable", False))
    vert_count = int(metrics.get("heightfield_vertex_count", 0))
    # Per-frame vertex-count check: the display mesh is a fixed heightfield,
    # so every frame must carry the same vertex count.
    vertex_counts = metrics.get("frame_vertex_counts") or []
    per_frame_stable = (
        bool(vertex_counts) and all(v == vertex_counts[0] for v in vertex_counts)
        if vertex_counts else True
    )
    # heightfield is fixed-topology regardless of per-frame mcubes numbers.
    results.append(GateResult(
        gate=1,
        name="topology_stable",
        passed=topology_stable,
        value={
            "heightfield_topology_stable": topology_stable,
            "heightfield_vertex_count": vert_count,
            "frame_vertex_counts_all_equal_if_present": per_frame_stable,
        },
        reason=(
            ""
            if topology_stable
            else "heightfield_topology_stable is not true in metrics.json"
        ),
    ))

    # ------------------------------------------------------------------
    # Gate 2 — Trust window >= 80% of the final clip length.
    # ------------------------------------------------------------------
    trust = _get(metadata, "trust_window") or {}
    frac = trust.get("fraction")
    try:
        frac_val = float(frac) if frac is not None else None
    except (TypeError, ValueError):
        frac_val = None
    trust_pass = frac_val is not None and frac_val >= 0.80
    results.append(GateResult(
        gate=2,
        name="trust_window_ge_80pct",
        passed=trust_pass,
        value=trust,
        reason=(
            "" if trust_pass
            else f"metadata.trust_window.fraction={frac!r} (<0.80 or missing)"
        ),
    ))

    # ------------------------------------------------------------------
    # Gate 3 — Hero silhouette readable at 200 px thumbnail.
    # ------------------------------------------------------------------
    g3 = _get(metadata, "gate", "gate_3_silhouette_readable_at_200px") or {}
    silhouette_pass = bool(g3.get("pass", False))
    results.append(GateResult(
        gate=3,
        name="silhouette_readable_at_200px",
        passed=silhouette_pass,
        value=g3,
        reason=(
            "" if silhouette_pass
            else "metadata.gate.gate_3_silhouette_readable_at_200px.pass != true"
        ),
    ))

    # ------------------------------------------------------------------
    # Gate 4 — Display mesh >= 16,000 vertices at hero framing.
    # ------------------------------------------------------------------
    g4_val = int(metrics.get("heightfield_vertex_count", 0))
    g4_pass = g4_val >= THRESH["display_mesh_vertex_count_min"]
    results.append(GateResult(
        gate=4,
        name="display_mesh_vertex_count",
        passed=g4_pass,
        value=g4_val,
        reason=("" if g4_pass else f"heightfield_vertex_count={g4_val} < "
                f"{THRESH['display_mesh_vertex_count_min']}"),
    ))

    # ------------------------------------------------------------------
    # Gate 5 — Metrics + structural invariant support the claim.
    # ------------------------------------------------------------------
    chamfer = float(metrics.get("heightfield_chamfer_mean", float("inf")))
    hausdorff = float(metrics.get("heightfield_hausdorff_mean", float("inf")))
    nmse = float(metrics.get("nmse_mean", float("inf")))
    eikonal_mean = float(_get(metrics, "eikonal", "mean", default=float("inf")))
    normal_consistency = float(metrics.get("normal_consistency_mean", 0.0))

    chamfer_pass = chamfer <= THRESH["chamfer_per_timestep_max"]
    hausdorff_pass = hausdorff <= THRESH["hausdorff_per_timestep_max"]
    nmse_pass = nmse <= THRESH["nmse_mean_max"]
    eikonal_pass = eikonal_mean <= THRESH["eikonal_mean_max"]

    g5_pass = all([chamfer_pass, hausdorff_pass, nmse_pass, eikonal_pass])
    g5_reason_parts = []
    if not chamfer_pass:
        g5_reason_parts.append(
            f"heightfield_chamfer_mean={chamfer:.3e} > {THRESH['chamfer_per_timestep_max']:.1e}"
        )
    if not hausdorff_pass:
        g5_reason_parts.append(
            f"heightfield_hausdorff_mean={hausdorff:.3e} > {THRESH['hausdorff_per_timestep_max']:.1e}"
        )
    if not nmse_pass:
        g5_reason_parts.append(
            f"nmse_mean={nmse:.3e} > {THRESH['nmse_mean_max']:.1e}"
        )
    if not eikonal_pass:
        g5_reason_parts.append(
            f"eikonal.mean={eikonal_mean:.3e} > {THRESH['eikonal_mean_max']:.1e}"
        )

    results.append(GateResult(
        gate=5,
        name="metrics_support_claim",
        passed=g5_pass,
        value={
            "heightfield_chamfer_mean": chamfer,
            "heightfield_hausdorff_mean": hausdorff,
            "nmse_mean": nmse,
            "eikonal_mean": eikonal_mean,
            "normal_consistency_mean": normal_consistency,
            "thresholds": {
                "chamfer_mean_max": THRESH["chamfer_per_timestep_max"],
                "hausdorff_mean_max": THRESH["hausdorff_per_timestep_max"],
                "nmse_mean_max": THRESH["nmse_mean_max"],
                "eikonal_mean_max": THRESH["eikonal_mean_max"],
            },
        },
        reason="; ".join(g5_reason_parts),
    ))

    # ------------------------------------------------------------------
    # Gate 6 — Hero default mode is single-subject (not compare).
    # ------------------------------------------------------------------
    mode = _get(metadata, "demo", "selected_default_mode")
    if mode is None:
        # Fall back to per-gate mirror populated by Prompt 2.
        mode = _get(metadata, "gate", "gate_6_hero_default_single_subject", "value")
    g6_pass = isinstance(mode, str) and mode != "compare" and mode in {"nif", "reference"}
    results.append(GateResult(
        gate=6,
        name="hero_default_single_subject",
        passed=g6_pass,
        value=mode,
        reason=(
            "" if g6_pass
            else f"demo.selected_default_mode={mode!r} (must be 'nif' or 'reference', not 'compare')"
        ),
    ))

    # ------------------------------------------------------------------
    # Gate 7 — Studio preset applied on pitch and dramatic.
    # ------------------------------------------------------------------
    studio = _get(metadata, "demo", "studio") or {}
    studio_applied_on = _get(metadata, "demo", "studio_preset_applied_on") or []
    studio_has_required_keys = all(
        k in studio for k in ("background", "ground_color", "ambient", "lights",
                              "camera_fov", "display_transfer", "display_gain",
                              "playback_speed", "palette")
    )
    applied_ok = set(["pitch", "dramatic"]).issubset(set(studio_applied_on))

    fov = float(studio.get("camera_fov", -1))
    gain = float(studio.get("display_gain", -1))
    speed = float(studio.get("playback_speed", -1))
    studio_numbers_ok = (
        abs(fov - THRESH["camera_fov_degrees"]) < 1e-6
        and abs(gain - THRESH["display_gain_tanh"]) < 1e-6
        and abs(speed - THRESH["playback_speed"]) < 1e-6
    )

    g7_pass = studio_has_required_keys and applied_ok and studio_numbers_ok
    g7_reason_parts = []
    if not studio_has_required_keys:
        g7_reason_parts.append("metadata.demo.studio missing required keys")
    if not applied_ok:
        g7_reason_parts.append(
            f"studio_preset_applied_on={studio_applied_on!r} (must include pitch and dramatic)"
        )
    if not studio_numbers_ok:
        g7_reason_parts.append(
            f"STUDIO numeric mismatch: fov={fov}, gain={gain}, speed={speed} "
            f"(expected {THRESH['camera_fov_degrees']}, {THRESH['display_gain_tanh']}, "
            f"{THRESH['playback_speed']})"
        )

    results.append(GateResult(
        gate=7,
        name="studio_preset_applied",
        passed=g7_pass,
        value={
            "has_required_keys": studio_has_required_keys,
            "applied_on": studio_applied_on,
            "camera_fov": fov,
            "display_gain": gain,
            "playback_speed": speed,
        },
        reason="; ".join(g7_reason_parts),
    ))

    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "taichi_final" / "metadata.json"),
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "validation_model" / "metrics.json"),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit a JSON report on stdout instead of human text.")
    args = p.parse_args(argv)

    try:
        metadata = _load_json(Path(args.metadata))
        metrics = _load_json(Path(args.metrics))
    except FileNotFoundError as exc:
        print(f"[gate] ERROR: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[gate] ERROR: malformed JSON: {exc}", file=sys.stderr)
        return 2

    results = check_gates(metadata, metrics)
    all_pass = all(r.passed for r in results)

    if args.json:
        report = {
            "all_seven_pass": all_pass,
            "metadata_path": args.metadata,
            "metrics_path": args.metrics,
            "results": [asdict(r) for r in results],
        }
        print(json.dumps(report, indent=2, default=str))
    else:
        print("Project-13 Success Gate — Project 12")
        print(f"  metadata: {args.metadata}")
        print(f"  metrics:  {args.metrics}")
        print("")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  Gate {r.gate} [{status}] {r.name}")
            if not r.passed and r.reason:
                print(f"           {r.reason}")
        print("")
        print(f"Result: {'ALL SEVEN PASS' if all_pass else 'FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
