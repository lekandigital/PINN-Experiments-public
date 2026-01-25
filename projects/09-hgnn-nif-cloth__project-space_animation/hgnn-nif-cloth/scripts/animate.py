#!/usr/bin/env python3
"""
Run cloth animation inference with trained model.

Generates cloth animation sequences using a trained HGNN-NIF model,
with optional physics simulation for realistic motion.

Usage:
    # Basic usage with trained model
    python scripts/animate.py --checkpoint outputs/best.pt --frames 120 --output animation_output/

    # With custom initial mesh
    python scripts/animate.py --checkpoint outputs/best.pt --mesh input/cloth.obj --frames 300

    # Export video
    python scripts/animate.py --checkpoint outputs/best.pt --frames 60 --export_video

    # Disable physics for pure neural prediction
    python scripts/animate.py --checkpoint outputs/best.pt --frames 120 --no_physics
"""

import argparse
import torch
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import HGNN_NIF_ClothModel
from src.models.temporal import TemporalHGNN_NIF
from src.animation.exporter import AnimationExporter, MeshGenerator


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device

    Returns:
        model: Loaded model (base or temporal)
        config: Model configuration dict
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration
    config = {
        'node_feat_dim': checkpoint.get('node_feat_dim', 3),
        'latent_dim': checkpoint.get('latent_dim', 64),
        'hidden_dim': checkpoint.get('hidden_dim', 64),
        'siren_hidden_dim': checkpoint.get('siren_hidden_dim', 128),
        'siren_layers': checkpoint.get('siren_layers', 3),
    }

    # Reconstruct base model
    base_model = HGNN_NIF_ClothModel(
        node_feat_dim=config['node_feat_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        siren_hidden_dim=config['siren_hidden_dim'],
        siren_layers=config['siren_layers']
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['state_dict'])

    base_model.to(device)
    base_model.eval()

    # Wrap in temporal model if available
    if 'temporal_state_dict' in checkpoint and checkpoint['temporal_state_dict'] is not None:
        print("Loading temporal model weights...")
        temporal_model = TemporalHGNN_NIF(base_model)
        temporal_model.load_state_dict(checkpoint['temporal_state_dict'])
        temporal_model.to(device)
        temporal_model.eval()
        return temporal_model, config

    return base_model, config


def apply_physics(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor,
    fixed_mask: torch.Tensor,
    dt: float = 1/30,
    gravity: float = -9.8,
    stiffness: float = 500.0,
    damping: float = 0.98
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply physics simulation step to cloth.

    Combines spring forces with gravity for realistic motion.

    Args:
        positions: Current positions (B, N, 3)
        velocity: Current velocities (B, N, 3)
        edges: Edge indices (2, E)
        rest_lengths: Rest lengths (E,)
        fixed_mask: Boolean mask for fixed vertices (N,)
        dt: Time step
        gravity: Gravity acceleration (negative for down)
        stiffness: Spring stiffness
        damping: Velocity damping factor

    Returns:
        new_positions: Updated positions
        new_velocity: Updated velocities
    """
    B, N, _ = positions.shape
    device = positions.device

    # Compute spring forces
    forces = torch.zeros_like(positions)

    # Gravity
    forces[..., 2] += gravity

    # Spring forces
    src, tgt = edges[0], edges[1]
    for b in range(B):
        pos = positions[b]
        diff = pos[tgt] - pos[src]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        direction = diff / dist
        stretch = dist.squeeze(-1) - rest_lengths
        force_mag = stiffness * stretch
        spring_force = force_mag.unsqueeze(-1) * direction

        # Accumulate forces (scatter_add emulation)
        for i in range(len(src)):
            forces[b, src[i]] += spring_force[i]
            forces[b, tgt[i]] -= spring_force[i]

    # Integration
    velocity = velocity + forces * dt
    velocity = velocity * damping
    new_positions = positions + velocity * dt

    # Apply fixed vertex constraints
    if fixed_mask is not None:
        velocity[:, fixed_mask] = 0
        # Fixed vertices don't move (handled in caller)

    # Ground collision (z=0 plane)
    ground_mask = new_positions[..., 2] < 0
    new_positions[..., 2] = torch.where(
        ground_mask,
        torch.zeros_like(new_positions[..., 2]),
        new_positions[..., 2]
    )
    velocity[..., 2] = torch.where(
        ground_mask,
        -velocity[..., 2] * 0.3,  # Bounce with damping
        velocity[..., 2]
    )

    return new_positions, velocity


def run_animation(
    model: torch.nn.Module,
    initial_positions: torch.Tensor,
    fine_edges: torch.Tensor,
    coarse_edges: torch.Tensor,
    rest_lengths: torch.Tensor,
    num_frames: int = 120,
    device: torch.device = torch.device('cuda'),
    use_physics: bool = True,
    fixed_rows: int = 1,
    gravity: float = -9.8,
    stiffness: float = 500.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run animation inference.

    Args:
        model: Trained model (base or temporal)
        initial_positions: Initial vertex positions (N, 3)
        fine_edges: Fine mesh edges (2, E_fine)
        coarse_edges: Coarse mesh edges (2, E_coarse)
        rest_lengths: Edge rest lengths (E,)
        num_frames: Number of frames to generate
        device: Computation device
        use_physics: Apply physics simulation
        fixed_rows: Number of top rows to keep fixed
        gravity: Gravity acceleration
        stiffness: Spring stiffness

    Returns:
        positions_sequence: (T, N, 3) numpy array
        velocities_sequence: (T, N, 3) numpy array
    """
    model.eval()

    N = initial_positions.shape[0]
    side = int(np.sqrt(N))

    # Create fixed vertex mask (top row(s))
    fixed_mask = torch.zeros(N, dtype=torch.bool, device=device)
    for row in range(fixed_rows):
        start_idx = row * side
        end_idx = start_idx + side
        fixed_mask[start_idx:end_idx] = True

    # Initialize
    current_pos = initial_positions.unsqueeze(0).to(device)  # (1, N, 3)
    velocity = torch.zeros_like(current_pos)
    initial_pos_device = initial_positions.to(device)

    # Build frame history for temporal model
    frame_history = [current_pos.clone() for _ in range(3)]

    positions_sequence = []
    velocities_sequence = []

    is_temporal = isinstance(model, TemporalHGNN_NIF)

    with torch.no_grad():
        for frame_idx in range(num_frames):
            # Stack history for temporal model
            history_tensor = torch.stack(frame_history[-3:], dim=1)  # (1, 3, N, 3)

            # Decimate to coarse
            B, T, N_fine, _ = history_tensor.shape
            fine_grid = history_tensor[:, -1].reshape(B, side, side, 3)
            coarse_pos = fine_grid[:, ::2, ::2, :].reshape(B, -1, 3)

            # Query points for SDF (sparse sampling)
            query_points = torch.randn(1, 100, 3, device=device) * 0.5

            # Model inference
            if is_temporal:
                output = model(history_tensor, fine_edges, coarse_edges, query_points)
                pred_pos = output['positions']
                pred_vel = output['velocity']
            else:
                # Static model inference
                output = model(
                    (current_pos, fine_edges),
                    (coarse_pos, coarse_edges),
                    query_points
                )
                # For static model, use physics-driven velocity
                pred_vel = velocity
                pred_pos = current_pos

            # Apply physics simulation
            if use_physics:
                pred_pos, velocity = apply_physics(
                    pred_pos if is_temporal else current_pos,
                    pred_vel,
                    fine_edges,
                    rest_lengths,
                    fixed_mask,
                    gravity=gravity,
                    stiffness=stiffness
                )
            else:
                velocity = pred_vel

            # Enforce fixed vertices
            pred_pos[:, fixed_mask, :] = initial_pos_device[fixed_mask, :].unsqueeze(0)

            # Store frame
            positions_sequence.append(pred_pos.squeeze(0).cpu().numpy())
            velocities_sequence.append(velocity.squeeze(0).cpu().numpy())

            # Update history
            current_pos = pred_pos
            frame_history.append(current_pos.clone())

            # Progress update
            if (frame_idx + 1) % 30 == 0:
                print(f"Frame {frame_idx + 1}/{num_frames}")

    positions_sequence = np.stack(positions_sequence, axis=0)
    velocities_sequence = np.stack(velocities_sequence, axis=0)

    return positions_sequence, velocities_sequence


def load_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh from OBJ file."""
    from src.data.mesh_loader import load_obj
    return load_obj(mesh_path)


def main():
    parser = argparse.ArgumentParser(
        description='Run HGNN-NIF-Cloth Animation Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/animate.py --checkpoint outputs/best.pt --frames 120
    python scripts/animate.py --checkpoint outputs/best.pt --mesh cloth.obj --frames 300 --export_video
        """
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')

    # Input options
    parser.add_argument('--mesh', type=str, default=None,
                       help='Input mesh OBJ file (optional, generates grid if not provided)')
    parser.add_argument('--resolution', type=int, default=20,
                       help='Cloth grid resolution (default: 20x20)')

    # Animation options
    parser.add_argument('--frames', type=int, default=120,
                       help='Number of frames to generate (default: 120)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video export (default: 30)')
    parser.add_argument('--fixed_rows', type=int, default=1,
                       help='Number of top rows to keep fixed (default: 1)')

    # Physics options
    parser.add_argument('--no_physics', action='store_true',
                       help='Disable external physics simulation')
    parser.add_argument('--gravity', type=float, default=-9.8,
                       help='Gravity acceleration (default: -9.8)')
    parser.add_argument('--stiffness', type=float, default=500.0,
                       help='Spring stiffness (default: 500)')

    # Output options
    parser.add_argument('--output', type=str, default='animation_output',
                       help='Output directory (default: animation_output)')
    parser.add_argument('--export_video', action='store_true',
                       help='Export MP4 video')
    parser.add_argument('--export_obj', action='store_true',
                       help='Export OBJ sequence')
    parser.add_argument('--export_ply', action='store_true',
                       help='Export PLY sequence (with colors)')

    # Hardware options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"Model config: {config}")

    # Create or load mesh
    if args.mesh:
        print(f"Loading mesh: {args.mesh}")
        vertices, faces, edges = load_mesh(args.mesh)
        rows = cols = int(np.sqrt(len(vertices)))
    else:
        print(f"Creating {args.resolution}x{args.resolution} cloth mesh")
        rows = cols = args.resolution
        vertices, faces, edges = MeshGenerator.create_grid_mesh(
            rows=rows, cols=cols, size=2.0
        )

    # Compute rest lengths
    rest_lengths = MeshGenerator.compute_rest_lengths(vertices, edges)

    # Create coarse mesh
    coarse_vertices, coarse_faces, coarse_edges = MeshGenerator.decimate_grid(
        vertices, rows, cols, stride=2
    )

    # Prepare tensors
    initial_pos = torch.from_numpy(vertices).float()
    fine_edges = torch.from_numpy(edges).long().to(device)
    coarse_edges_t = torch.from_numpy(coarse_edges).long().to(device)
    rest_lengths_t = torch.from_numpy(rest_lengths).float().to(device)

    # Run animation
    print(f"\nGenerating {args.frames} frames...")
    positions_sequence, velocities_sequence = run_animation(
        model=model,
        initial_positions=initial_pos,
        fine_edges=fine_edges,
        coarse_edges=coarse_edges_t,
        rest_lengths=rest_lengths_t,
        num_frames=args.frames,
        device=device,
        use_physics=not args.no_physics,
        fixed_rows=args.fixed_rows,
        gravity=args.gravity,
        stiffness=args.stiffness
    )

    # Export results
    print(f"\nExporting to: {args.output}")
    exporter = AnimationExporter(args.output)

    # Always export numpy arrays
    exporter.export_numpy(
        positions_sequence,
        velocities_sequence,
        faces=faces,
        edges=edges
    )

    # Export metadata
    exporter.export_metadata(
        num_frames=args.frames,
        num_vertices=len(vertices),
        num_faces=len(faces),
        fps=args.fps,
        model_config=config,
        extra_info={
            'use_physics': not args.no_physics,
            'gravity': args.gravity,
            'stiffness': args.stiffness,
            'fixed_rows': args.fixed_rows
        }
    )

    # Optional exports
    if args.export_obj:
        print("Exporting OBJ sequence...")
        exporter.export_obj_sequence(positions_sequence, faces, prefix="cloth")

    if args.export_ply:
        print("Exporting PLY sequence...")
        # Color by height
        z_vals = positions_sequence[..., 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
        colors = (np.stack([z_norm * 255, (1 - z_norm) * 128, np.ones_like(z_norm) * 200], axis=-1)).astype(np.uint8)
        exporter.export_ply_sequence(positions_sequence, faces, colors=colors)

    if args.export_video:
        print("Rendering video...")
        exporter.export_video(
            positions_sequence, faces,
            filename="animation.mp4",
            fps=args.fps
        )

    print(f"\nAnimation complete!")
    print(f"  Output directory: {args.output}")
    print(f"  Frames generated: {args.frames}")
    print(f"  Duration: {args.frames / args.fps:.1f} seconds @ {args.fps} fps")


if __name__ == '__main__':
    main()
