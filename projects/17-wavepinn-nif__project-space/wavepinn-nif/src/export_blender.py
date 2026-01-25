#!/usr/bin/env python3
"""
Export WavePINN wavefield for Blender visualization.

Generates displacement map image sequence for Blender's Displace modifier.

Usage:
    python export_blender.py --model wavepinn.pt --output blender_export --frames 60
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse


class FourierFeatures(nn.Module):
    """Fourier feature encoding to mitigate spectral bias."""

    def __init__(self, input_dim: int, num_features: int = 64, scale: float = 1.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, num_features) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([x, torch.cos(proj), torch.sin(proj)], dim=-1)


class WavePINN(nn.Module):
    """Physics-Informed Neural Network for Acoustic Wave Equation."""

    def __init__(
        self,
        hidden_dims: list = [128, 128, 64, 32],
        use_fourier: bool = True,
        num_fourier: int = 32,
        fourier_scale: float = 1.0
    ):
        super().__init__()

        self.use_fourier = use_fourier

        if use_fourier:
            self.fourier = FourierFeatures(3, num_fourier, fourier_scale)
            input_dim = 3 + num_fourier * 2
        else:
            input_dim = 3

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hdim), nn.Tanh()])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            x = self.fourier(coords)
        else:
            x = coords
        return self.mlp(x)


def export_blender_sequence(
    model_path=None,
    output_dir="blender_export",
    resolution=128,
    n_frames=60,
    t_start=0.05,
    t_end=0.5,
    seed=42
):
    """Export wavefield animation as image sequence for Blender."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Set seed for reproducible Fourier features
    torch.manual_seed(seed)
    model = WavePINN().to(device)
    
    # Load weights if checkpoint provided
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Support both checkpoint formats
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from: {model_path}")
    else:
        print("Using randomly initialized model (demo mode)")
    
    model.eval()
    
    print(f"\nExporting {n_frames} frames at {resolution}x{resolution}...")
    
    # Create coordinate grid
    x = torch.linspace(0, 1, resolution, device=device)
    xx, zz = torch.meshgrid(x, x, indexing="ij")
    
    all_fields = []
    times = np.linspace(t_start, t_end, n_frames)
    
    for frame, t in enumerate(times):
        coords = torch.stack([
            xx.flatten(),
            zz.flatten(),
            torch.full((resolution**2,), t, device=device)
        ], dim=-1)
        
        with torch.no_grad():
            u = model(coords).cpu().numpy().reshape(resolution, resolution)
        
        all_fields.append(u)
        
        if frame % 10 == 0:
            print(f"  Frame {frame}/{n_frames} (t={t:.3f})")
    
    # Normalize and save as images
    all_fields = np.array(all_fields)
    global_min, global_max = all_fields.min(), all_fields.max()
    
    print(f"\nWavefield range: [{global_min:.6f}, {global_max:.6f}]")
    
    try:
        from PIL import Image
        for frame, u in enumerate(all_fields):
            u_norm = ((u - global_min) / (global_max - global_min + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(u_norm).save(output_path / "images" / f"wave_{frame:04d}.png")
        print(f"Saved {n_frames} PNG images")
    except ImportError:
        print("Warning: PIL not available. Install with: pip install pillow")
        np.save(output_path / "all_frames.npy", all_fields)
        print("Saved numpy arrays instead")
    
    # Save metadata
    metadata = {
        "resolution": resolution,
        "n_frames": n_frames,
        "t_start": t_start,
        "t_end": t_end,
        "fps": 24,
        "duration_seconds": n_frames / 24,
        "global_min": float(global_min),
        "global_max": float(global_max),
        "displacement_scale": 0.5,
        "model_path": str(model_path) if model_path else None
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path.absolute()}/")
    print(f"  - images/: {n_frames} PNG displacement maps")
    print(f"  - metadata.json")
    print(f"\nBlender Import:")
    print(f"  1. Add Grid mesh ({resolution-1}x{resolution-1} subdivisions)")
    print(f"  2. Add Displace modifier → New Texture → Image Sequence")
    print(f"  3. Set Strength: 0.5, Frames: {n_frames}")


def main():
    parser = argparse.ArgumentParser(description="Export WavePINN for Blender")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--output", type=str, default="blender_export")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--t_start", type=float, default=0.05)
    parser.add_argument("--t_end", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    export_blender_sequence(
        model_path=args.model,
        output_dir=args.output,
        resolution=args.resolution,
        n_frames=args.frames,
        t_start=args.t_start,
        t_end=args.t_end,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
