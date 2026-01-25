#!/usr/bin/env python3
"""
Quick Demo: Generate a simple cloth animation with physics.

This script creates a cloth animation using pure physics simulation,
demonstrating the animation export pipeline without requiring a trained model.

Perfect for:
- Testing the export pipeline
- Quick visualization
- Generating training data reference

Usage:
    # Basic demo
    python scripts/demo.py --frames 60 --output demo_output/

    # Higher resolution with video export
    python scripts/demo.py --frames 120 --resolution 30 --video

    # Custom physics parameters
    python scripts/demo.py --frames 90 --gravity -5.0 --stiffness 800
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.animation.exporter import AnimationExporter, MeshGenerator


def mass_spring_simulation(
    initial_pos: np.ndarray,
    edges: np.ndarray,
    num_frames: int = 60,
    dt: float = 1/60,
    gravity: float = -9.8,
    stiffness: float = 500.0,
    damping: float = 0.995,
    substeps: int = 4,
    fixed_rows: int = 1
) -> tuple:
    """
    Mass-spring physics simulation for cloth.

    Implements a simple but effective cloth simulation using:
    - Spring forces between connected vertices
    - Gravity
    - Velocity damping
    - Fixed boundary conditions

    Args:
        initial_pos: (N, 3) initial vertex positions
        edges: (2, E) edge connectivity
        num_frames: Number of frames to simulate
        dt: Time step per frame
        gravity: Gravity acceleration (negative = down)
        stiffness: Spring stiffness coefficient
        damping: Velocity damping factor (0-1)
        substeps: Physics substeps per frame for stability
        fixed_rows: Number of top rows to keep fixed

    Returns:
        positions_sequence: (T, N, 3) position trajectory
        velocities_sequence: (T, N, 3) velocity trajectory
    """
    N = initial_pos.shape[0]
    side = int(np.sqrt(N))

    # Initialize state
    positions = initial_pos.copy()
    velocities = np.zeros_like(positions)

    # Compute rest lengths
    rest_lengths = np.linalg.norm(
        positions[edges[0]] - positions[edges[1]], axis=-1
    )

    # Create fixed vertex mask (top row(s))
    fixed_mask = np.zeros(N, dtype=bool)
    for row in range(fixed_rows):
        start_idx = row * side
        end_idx = start_idx + side
        fixed_mask[start_idx:end_idx] = True

    # Store initial positions for fixed vertices
    fixed_positions = initial_pos[fixed_mask].copy()

    # Storage
    positions_sequence = [positions.copy()]
    velocities_sequence = [velocities.copy()]

    # Simulation parameters per substep
    sub_dt = dt / substeps

    print(f"Simulating {num_frames} frames with {substeps} substeps each...")

    for frame_idx in range(num_frames - 1):
        for _ in range(substeps):
            # Initialize forces with gravity
            forces = np.zeros_like(positions)
            forces[:, 2] = gravity  # Z is up

            # Spring forces
            src, tgt = edges[0], edges[1]
            diff = positions[tgt] - positions[src]  # (E, 3)
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            dist = np.maximum(dist, 1e-8)  # Avoid division by zero
            direction = diff / dist

            stretch = dist.squeeze(-1) - rest_lengths
            force_mag = stiffness * stretch

            # Apply forces to both ends of each spring
            spring_force = (force_mag[:, np.newaxis] * direction)

            # Accumulate forces using np.add.at for efficiency
            np.add.at(forces, src, spring_force)
            np.add.at(forces, tgt, -spring_force)

            # Integration (semi-implicit Euler)
            velocities = velocities + forces * sub_dt
            velocities = velocities * damping
            positions = positions + velocities * sub_dt

            # Fixed vertex constraints
            positions[fixed_mask] = fixed_positions
            velocities[fixed_mask] = 0

            # Ground collision (z = -1 plane)
            ground_mask = positions[:, 2] < -1.0
            positions[ground_mask, 2] = -1.0
            velocities[ground_mask, 2] = np.abs(velocities[ground_mask, 2]) * 0.3  # Bounce

        # Store frame
        positions_sequence.append(positions.copy())
        velocities_sequence.append(velocities.copy())

        # Progress update
        if (frame_idx + 1) % 30 == 0:
            print(f"  Frame {frame_idx + 1}/{num_frames - 1}")

    return np.stack(positions_sequence), np.stack(velocities_sequence)


def wind_perturbation(
    positions: np.ndarray,
    velocities: np.ndarray,
    time: float,
    strength: float = 0.5,
    frequency: float = 0.3
) -> np.ndarray:
    """
    Apply time-varying wind force.

    Args:
        positions: (N, 3) current positions
        velocities: (N, 3) current velocities
        time: Current simulation time
        strength: Wind strength
        frequency: Wind oscillation frequency

    Returns:
        wind_force: (N, 3) wind force per vertex
    """
    # Oscillating wind in X-Y plane
    wind_dir = np.array([
        np.sin(time * frequency * 2 * np.pi),
        np.cos(time * frequency * 2 * np.pi) * 0.5,
        0.1
    ])
    wind_dir = wind_dir / np.linalg.norm(wind_dir)

    # Vary strength with time
    wind_strength = strength * (1 + 0.5 * np.sin(time * frequency * 4 * np.pi))

    return np.ones_like(positions) * wind_dir * wind_strength


def main():
    parser = argparse.ArgumentParser(
        description='HGNN-NIF-Cloth Demo - Physics-Based Cloth Animation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/demo.py --frames 60
    python scripts/demo.py --frames 120 --resolution 30 --video
    python scripts/demo.py --gravity -5 --stiffness 800 --video
        """
    )

    # Animation parameters
    parser.add_argument('--frames', type=int, default=60,
                       help='Number of frames to generate')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video')

    # Mesh parameters
    parser.add_argument('--resolution', type=int, default=20,
                       help='Cloth grid resolution (NxN)')
    parser.add_argument('--size', type=float, default=2.0,
                       help='Cloth size in world units')
    parser.add_argument('--height', type=float, default=1.0,
                       help='Initial height of cloth')

    # Physics parameters
    parser.add_argument('--gravity', type=float, default=-9.8,
                       help='Gravity acceleration')
    parser.add_argument('--stiffness', type=float, default=500.0,
                       help='Spring stiffness')
    parser.add_argument('--damping', type=float, default=0.995,
                       help='Velocity damping (0-1)')
    parser.add_argument('--substeps', type=int, default=4,
                       help='Physics substeps per frame')
    parser.add_argument('--fixed_rows', type=int, default=1,
                       help='Number of top rows to fix')

    # Output parameters
    parser.add_argument('--output', type=str, default='demo_output',
                       help='Output directory')
    parser.add_argument('--video', action='store_true',
                       help='Export MP4 video')
    parser.add_argument('--obj', action='store_true',
                       help='Export OBJ sequence')

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("HGNN-NIF-Cloth Demo - Physics Simulation")
    print("=" * 60)
    print()

    # Create cloth mesh
    print(f"Creating {args.resolution}x{args.resolution} cloth mesh...")
    vertices, faces, edges = MeshGenerator.create_grid_mesh(
        rows=args.resolution,
        cols=args.resolution,
        size=args.size,
        center=(0, 0, args.height)
    )

    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Edges: {edges.shape[1]}")

    # Compute UVs and normals for export
    uvs = MeshGenerator.create_uv_coordinates(args.resolution, args.resolution)

    # Run physics simulation
    print()
    print("Running physics simulation...")
    print(f"  Frames: {args.frames}")
    print(f"  Gravity: {args.gravity}")
    print(f"  Stiffness: {args.stiffness}")
    print(f"  Damping: {args.damping}")
    print(f"  Substeps: {args.substeps}")
    print()

    start_time = time.time()

    positions_seq, velocities_seq = mass_spring_simulation(
        initial_pos=vertices,
        edges=edges,
        num_frames=args.frames,
        dt=1.0 / args.fps,
        gravity=args.gravity,
        stiffness=args.stiffness,
        damping=args.damping,
        substeps=args.substeps,
        fixed_rows=args.fixed_rows
    )

    sim_time = time.time() - start_time
    print(f"\nSimulation complete in {sim_time:.2f}s ({args.frames / sim_time:.1f} fps)")

    # Compute normals for each frame
    print("\nComputing vertex normals...")
    normals_seq = []
    for positions in positions_seq:
        normals = MeshGenerator.compute_vertex_normals(positions, faces)
        normals_seq.append(normals)
    normals_seq = np.stack(normals_seq)

    # Export
    print(f"\nExporting to: {args.output}")
    exporter = AnimationExporter(args.output)

    # Always export numpy data
    exporter.export_numpy(
        positions_seq,
        velocities_seq,
        faces=faces,
        edges=edges
    )

    # Export metadata
    exporter.export_metadata(
        num_frames=args.frames,
        num_vertices=len(vertices),
        num_faces=len(faces),
        fps=args.fps,
        extra_info={
            'simulation_type': 'mass_spring',
            'gravity': args.gravity,
            'stiffness': args.stiffness,
            'damping': args.damping,
            'substeps': args.substeps,
            'resolution': args.resolution,
            'fixed_rows': args.fixed_rows
        }
    )

    # Export OBJ sequence
    if args.obj:
        print("\nExporting OBJ sequence...")
        exporter.export_obj_sequence(
            positions_seq, faces,
            prefix="cloth",
            normals=normals_seq,
            uvs=uvs
        )

    # Export video
    if args.video:
        print("\nRendering video...")
        exporter.export_video(
            positions_seq, faces,
            filename="demo.mp4",
            fps=args.fps,
            camera_angle=(30, 45),
            show_edges=True
        )

    # Summary
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Frames generated: {args.frames}")
    print(f"Duration: {args.frames / args.fps:.1f} seconds @ {args.fps} fps")
    print()
    print("Files created:")
    print(f"  - {args.output}/positions.npy")
    print(f"  - {args.output}/velocities.npy")
    print(f"  - {args.output}/animation_data.npz")
    print(f"  - {args.output}/metadata.json")
    if args.obj:
        print(f"  - {args.output}/obj_sequence/cloth_*.obj")
    if args.video:
        print(f"  - {args.output}/demo.mp4")
    print()
    print("To view the animation:")
    if args.video:
        print(f"  Open {args.output}/demo.mp4 in any video player")
    else:
        print(f"  Re-run with --video flag to generate video")
    print()


if __name__ == '__main__':
    main()
