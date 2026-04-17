#!/usr/bin/env python3
"""Generate the physics ground-truth (PBS baseline) cloth sequence used as
both training data and the comparison reference for the HGNN-NIF model.

Mass-spring sim with gravity + time-varying wind for richer dynamics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.animation.exporter import MeshGenerator


def simulate(
    initial_pos, edges, num_frames, dt, gravity, stiffness, damping,
    substeps, fixed_rows, side, wind_strength, wind_freq, seed=0,
):
    rng = np.random.default_rng(seed)
    N = initial_pos.shape[0]
    positions = initial_pos.copy()
    velocities = np.zeros_like(positions)
    rest_lengths = np.linalg.norm(
        positions[edges[0]] - positions[edges[1]], axis=-1
    )

    fixed_mask = np.zeros(N, dtype=bool)
    for row in range(fixed_rows):
        fixed_mask[row * side : (row + 1) * side] = True
    fixed_positions = initial_pos[fixed_mask].copy()

    pos_seq = [positions.copy()]
    vel_seq = [velocities.copy()]

    src, tgt = edges[0], edges[1]
    sub_dt = dt / substeps

    for f in range(num_frames - 1):
        t = f * dt
        wind_dir = np.array([
            np.sin(t * wind_freq * 2 * np.pi),
            np.cos(t * wind_freq * 2 * np.pi) * 0.5,
            0.1,
        ])
        wind_dir /= np.linalg.norm(wind_dir)
        wind = wind_strength * (1.0 + 0.5 * np.sin(t * wind_freq * 4 * np.pi)) * wind_dir

        for _ in range(substeps):
            forces = np.zeros_like(positions)
            forces[:, 2] = gravity
            forces += wind  # broadcast per vertex

            diff = positions[tgt] - positions[src]
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            dist = np.maximum(dist, 1e-8)
            direction = diff / dist
            stretch = dist.squeeze(-1) - rest_lengths
            spring_force = (stiffness * stretch)[:, None] * direction
            np.add.at(forces, src, spring_force)
            np.add.at(forces, tgt, -spring_force)

            velocities = (velocities + forces * sub_dt) * damping
            positions = positions + velocities * sub_dt
            positions[fixed_mask] = fixed_positions
            velocities[fixed_mask] = 0.0

            ground = positions[:, 2] < -1.0
            positions[ground, 2] = -1.0
            velocities[ground, 2] = np.abs(velocities[ground, 2]) * 0.3

        pos_seq.append(positions.copy())
        vel_seq.append(velocities.copy())

        if (f + 1) % 100 == 0:
            print(f"  Frame {f + 1}/{num_frames - 1}")

    return np.stack(pos_seq), np.stack(vel_seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--resolution", type=int, default=40)
    p.add_argument("--size", type=float, default=2.0)
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gravity", type=float, default=-9.8)
    p.add_argument("--stiffness", type=float, default=500.0)
    p.add_argument("--damping", type=float, default=0.995)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--fixed_rows", type=int, default=1)
    p.add_argument("--wind_strength", type=float, default=2.0)
    p.add_argument("--wind_freq", type=float, default=0.4)
    p.add_argument("--output", type=str, default="outputs/physics_baseline")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building {args.resolution}x{args.resolution} cloth mesh...")
    vertices, faces, fine_edges = MeshGenerator.create_grid_mesh(
        rows=args.resolution, cols=args.resolution,
        size=args.size, center=(0, 0, args.height),
    )
    coarse_stride = 2
    coarse_verts, coarse_faces, coarse_edges = MeshGenerator.decimate_grid(
        vertices, args.resolution, args.resolution, stride=coarse_stride,
    )
    coarse_res = args.resolution // coarse_stride
    print(f"  fine: N={len(vertices)} edges={fine_edges.shape[1]} faces={len(faces)}")
    print(f"  coarse({coarse_res}x{coarse_res}): N={len(coarse_verts)} edges={coarse_edges.shape[1]}")

    print(f"Simulating {args.frames} frames @ {args.fps} fps with substeps={args.substeps}...")
    t0 = time.time()
    pos_seq, vel_seq = simulate(
        vertices, fine_edges,
        num_frames=args.frames, dt=1.0 / args.fps,
        gravity=args.gravity, stiffness=args.stiffness,
        damping=args.damping, substeps=args.substeps,
        fixed_rows=args.fixed_rows, side=args.resolution,
        wind_strength=args.wind_strength, wind_freq=args.wind_freq,
    )
    sim_dt = time.time() - t0
    print(f"Simulated in {sim_dt:.2f}s ({args.frames/sim_dt:.1f} sim-fps)")

    np.save(out / "cloth_sequence.npy", pos_seq.astype(np.float32))
    np.save(out / "cloth_velocities.npy", vel_seq.astype(np.float32))
    np.savez_compressed(
        out / "topology.npz",
        fine_vertices=vertices, fine_faces=faces, fine_edges=fine_edges,
        coarse_vertices=coarse_verts, coarse_faces=coarse_faces, coarse_edges=coarse_edges,
        rest_lengths=np.linalg.norm(
            vertices[fine_edges[0]] - vertices[fine_edges[1]], axis=-1,
        ).astype(np.float32),
    )

    info = {
        "resolution": args.resolution,
        "coarse_resolution": coarse_res,
        "num_frames": int(pos_seq.shape[0]),
        "num_vertices_fine": int(len(vertices)),
        "num_vertices_coarse": int(len(coarse_verts)),
        "num_edges_fine": int(fine_edges.shape[1]),
        "num_edges_coarse": int(coarse_edges.shape[1]),
        "num_faces_fine": int(len(faces)),
        "num_faces_coarse": int(len(coarse_faces)),
        "fps": args.fps,
        "dt": 1.0 / args.fps,
        "fixed_rows": args.fixed_rows,
        "physics": {
            "gravity": args.gravity, "stiffness": args.stiffness,
            "damping": args.damping, "substeps": args.substeps,
            "wind_strength": args.wind_strength, "wind_freq": args.wind_freq,
        },
        "size": args.size, "height": args.height,
    }
    with open(out / "physics_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote: {out}/cloth_sequence.npy shape={pos_seq.shape}")
    print(f"Wrote: {out}/topology.npz")
    print(f"Wrote: {out}/physics_info.json")


if __name__ == "__main__":
    main()
