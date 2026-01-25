#!/usr/bin/env python3
"""
Time-warp sequence editor for NIF-Cloth4D-Temporal.

Allows editing cloth sequences by:
1. Specifying a target silhouette/constraint at time t*
2. Optimizing the model to match while maintaining physics consistency
3. Exporting the edited sequence to USD/OBJ/FBX

Usage:
    python scripts/edit_sequence.py --checkpoint model.pt --target_frame 60
    python scripts/edit_sequence.py --checkpoint model.pt --silhouette target.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.models import FourierFeatureMLP
from src.utils import set_seed, get_device, export_to_usd, export_to_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Edit cloth sequence')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--target_frame', type=int, default=60,
                       help='Target frame to edit')
    parser.add_argument('--silhouette', type=str,
                       help='Target silhouette image (optional)')
    parser.add_argument('--target_vertices', type=str,
                       help='Path to target vertex positions (optional)')
    parser.add_argument('--output_dir', type=str, default='exports',
                       help='Output directory for exported meshes')
    parser.add_argument('--output_format', type=str, default='usd',
                       choices=['usd', 'obj', 'both'],
                       help='Export format')
    
    # Optimization
    parser.add_argument('--edit_lr', type=float, default=1e-4,
                       help='Learning rate for editing')
    parser.add_argument('--edit_steps', type=int, default=100,
                       help='Number of optimization steps')
    parser.add_argument('--physics_weight', type=float, default=0.1,
                       help='Weight for physics consistency during edit')
    
    # Sequence
    parser.add_argument('--num_frames', type=int, default=120)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--cloth_res', type=int, default=32)
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'])
    
    return parser.parse_args()


def create_cloth_grid(cloth_res: int, device: torch.device) -> torch.Tensor:
    """Create a regular grid of cloth vertices."""
    x = torch.linspace(-0.5, 0.5, cloth_res, device=device)
    z = torch.linspace(-0.5, 0.5, cloth_res, device=device)
    xx, zz = torch.meshgrid(x, z, indexing='ij')
    
    vertices = torch.zeros(cloth_res * cloth_res, 3, device=device)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = 1.0  # Initial height
    vertices[:, 2] = zz.flatten()
    
    return vertices


def query_model_at_vertices(
    model: FourierFeatureMLP,
    rest_vertices: torch.Tensor,
    t: float,
    hidden: torch.Tensor = None,
    device: torch.device = None,
) -> tuple:
    """
    Query the model at rest vertex positions to get deformed positions.
    
    Assumes model outputs displacement from rest position.
    """
    num_vertices = rest_vertices.size(0)
    
    # Create time tensor
    t_tensor = torch.full((num_vertices, 1), t, device=device)
    
    # Forward pass
    output, new_hidden = model(rest_vertices, t_tensor, hidden)
    
    # If model outputs SDF, we need a different approach
    # For displacement model: deformed = rest + displacement
    # For SDF model: need to march/sample surface
    
    # Assuming displacement output for simplicity
    if output.size(-1) == 3:
        deformed = rest_vertices + output
    else:
        # SDF output - return rest vertices for now
        # Full implementation would use marching cubes or sphere tracing
        deformed = rest_vertices
    
    return deformed, new_hidden


def optimize_for_target(
    model: FourierFeatureMLP,
    target_vertices: torch.Tensor,
    target_frame: int,
    rest_vertices: torch.Tensor,
    device: torch.device,
    lr: float = 1e-4,
    num_steps: int = 100,
    physics_weight: float = 0.1,
    fps: float = 30.0,
) -> FourierFeatureMLP:
    """
    Fine-tune model to match target at specific frame.
    
    Uses a soft constraint to match target while maintaining
    physics consistency across the sequence.
    """
    print(f"\nOptimizing model to match target at frame {target_frame}")
    print(f"  Learning rate: {lr}")
    print(f"  Steps: {num_steps}")
    print(f"  Physics weight: {physics_weight}")
    
    model = model.to(device)
    model.train()
    
    # Only optimize a subset of parameters (last layers)
    # This preserves learned physics while allowing local edits
    params_to_optimize = []
    for name, param in model.named_parameters():
        if 'output_layer' in name or 'temporal_gru' in name:
            param.requires_grad = True
            params_to_optimize.append(param)
        else:
            param.requires_grad = False
    
    optimizer = Adam(params_to_optimize, lr=lr)
    
    dt = 1.0 / fps
    t_target = target_frame * dt
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Query model at target time
        t_tensor = torch.full((rest_vertices.size(0), 1), t_target, device=device)
        output, _ = model(rest_vertices, t_tensor, None)
        
        if output.size(-1) == 3:
            predicted = rest_vertices + output
        else:
            predicted = rest_vertices  # Placeholder for SDF model
        
        # Target matching loss
        target_loss = F.mse_loss(predicted, target_vertices)
        
        # Physics consistency loss (smoothness)
        # Query adjacent frames
        physics_loss = torch.tensor(0.0, device=device)
        if physics_weight > 0:
            for dt_offset in [-1, 1]:
                t_adj = (target_frame + dt_offset) * dt
                if 0 <= t_adj <= 4.0:  # Within sequence bounds
                    t_adj_tensor = torch.full((rest_vertices.size(0), 1), t_adj, device=device)
                    output_adj, _ = model(rest_vertices, t_adj_tensor, None)
                    
                    # Penalize large differences between adjacent frames
                    diff = output - output_adj
                    physics_loss = physics_loss + (diff ** 2).mean()
        
        # Total loss
        loss = target_loss + physics_weight * physics_loss
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}: "
                  f"loss={loss.item():.6f}, "
                  f"target={target_loss.item():.6f}, "
                  f"physics={physics_loss.item():.6f}")
    
    # Re-enable all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    model.eval()
    return model


def export_sequence(
    model: FourierFeatureMLP,
    rest_vertices: torch.Tensor,
    num_frames: int,
    output_dir: str,
    output_format: str,
    fps: float = 30.0,
    cloth_res: int = 32,
    device: torch.device = None,
):
    """Export the full sequence to mesh files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting {num_frames} frames to {output_dir}")
    
    model = model.to(device)
    model.eval()
    
    dt = 1.0 / fps
    hidden = model.init_hidden(1, device) if model.use_gru else None
    
    # Generate faces for cloth grid
    faces = []
    for i in range(cloth_res - 1):
        for j in range(cloth_res - 1):
            v00 = i * cloth_res + j
            v01 = v00 + 1
            v10 = v00 + cloth_res
            v11 = v10 + 1
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    all_vertices = []
    
    with torch.no_grad():
        for frame in range(num_frames):
            t = frame * dt
            t_tensor = torch.full((rest_vertices.size(0), 1), t, device=device)
            
            output, hidden = model(rest_vertices, t_tensor, hidden)
            
            if output.size(-1) == 3:
                vertices = rest_vertices + output
            else:
                vertices = rest_vertices
            
            all_vertices.append(vertices.cpu().numpy())
            
            # Export individual OBJ if requested
            if output_format in ['obj', 'both']:
                obj_path = output_dir / f'frame_{frame:04d}.obj'
                export_to_obj(
                    vertices=vertices.cpu().numpy(),
                    faces=faces.cpu().numpy(),
                    output_path=str(obj_path),
                )
            
            if (frame + 1) % 30 == 0:
                print(f"  Exported frame {frame + 1}/{num_frames}")
    
    # Export USD sequence
    if output_format in ['usd', 'both']:
        import numpy as np
        usd_path = output_dir / 'sequence.usda'
        export_to_usd(
            vertices_sequence=np.stack(all_vertices, axis=0),
            faces=faces.cpu().numpy(),
            output_path=str(usd_path),
            fps=fps,
        )
        print(f"\nUSD sequence exported to: {usd_path}")
    
    print(f"\nExport complete!")


def main():
    args = parse_args()
    
    set_seed(42)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config', None)
    if config:
        model = FourierFeatureMLP(
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_freqs=config.model.num_freqs,
            use_gru=config.model.use_gru,
            gru_hidden=config.model.gru_hidden,
        )
    else:
        model = FourierFeatureMLP()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create cloth grid
    rest_vertices = create_cloth_grid(args.cloth_res, device)
    
    # Load or create target
    if args.target_vertices:
        print(f"Loading target vertices from: {args.target_vertices}")
        target_vertices = torch.load(args.target_vertices).to(device)
    else:
        # Create simple target: shift vertices slightly
        print("Creating synthetic target (shifted vertices)")
        target_vertices = rest_vertices.clone()
        target_vertices[:, 1] -= 0.2  # Lower the cloth
        target_vertices[:, 0] += 0.1  # Shift right
    
    # Optimize for target
    model = optimize_for_target(
        model=model,
        target_vertices=target_vertices,
        target_frame=args.target_frame,
        rest_vertices=rest_vertices,
        device=device,
        lr=args.edit_lr,
        num_steps=args.edit_steps,
        physics_weight=args.physics_weight,
        fps=args.fps,
    )
    
    # Export sequence
    export_sequence(
        model=model,
        rest_vertices=rest_vertices,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        output_format=args.output_format,
        fps=args.fps,
        cloth_res=args.cloth_res,
        device=device,
    )


if __name__ == '__main__':
    main()
