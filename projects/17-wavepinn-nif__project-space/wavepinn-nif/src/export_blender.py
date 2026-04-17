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


def infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer MLP layout from a PyTorch state dict."""
    linear_weights = []
    for key, value in state_dict.items():
        if (key.startswith("mlp.") or key.startswith("net.")) and key.endswith(".weight"):
            linear_weights.append((key, value))
    linear_weights.sort(key=lambda item: int(item[0].split(".")[1]))

    if not linear_weights:
        raise ValueError("Could not infer model config: no MLP weights found in checkpoint")

    hidden_dims = [int(weight.shape[0]) for _, weight in linear_weights[:-1]]

    fourier_basis = state_dict.get("fourier.B")
    if fourier_basis is None:
        fourier_basis = state_dict.get("ff.B")
    if fourier_basis is None:
        use_fourier = False
        num_fourier = 0
        fourier_scale = 1.0
    else:
        use_fourier = True
        num_fourier = int(fourier_basis.shape[1])
        fourier_scale = float(fourier_basis.std().item()) if fourier_basis.numel() > 0 else 1.0

    return {
        "hidden_dims": hidden_dims,
        "use_fourier": use_fourier,
        "num_fourier": num_fourier,
        "fourier_scale": fourier_scale,
    }


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """Map older training-script module names onto the export model names."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("ff."):
            key = key.replace("ff.", "fourier.", 1)
        elif key.startswith("net."):
            key = key.replace("net.", "mlp.", 1)
        normalized[key] = value
    return normalized


def build_model_from_config(model_config: dict, device: str) -> WavePINN:
    return WavePINN(
        hidden_dims=model_config.get("hidden_dims", [128, 128, 64, 32]),
        use_fourier=model_config.get("use_fourier", True),
        num_fourier=model_config.get("num_fourier", 32),
        fourier_scale=model_config.get("fourier_scale", 1.0),
    ).to(device)


def load_wavepinn_checkpoint(model_path: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state") or checkpoint.get("model") or checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    model_config = checkpoint.get("model_config") or infer_model_config_from_state_dict(state_dict)
    state_dict = normalize_state_dict_keys(state_dict)
    model = build_model_from_config(model_config, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, device


def sample_wavefield_sequence(
    model: WavePINN,
    resolution: int,
    n_frames: int,
    t_start: float,
    t_end: float,
    device: str,
):
    """Sample a trained wavefield model on a regular grid over time."""
    x = torch.linspace(0, 1, resolution, device=device)
    xx, zz = torch.meshgrid(x, x, indexing="ij")
    all_fields = []
    times = np.linspace(t_start, t_end, n_frames)

    for frame, t in enumerate(times):
        coords = torch.stack(
            [
                xx.flatten(),
                zz.flatten(),
                torch.full((resolution**2,), t, device=device),
            ],
            dim=-1,
        )

        with torch.no_grad():
            u = model(coords).cpu().numpy().reshape(resolution, resolution)

        all_fields.append(u)

        if frame % 10 == 0:
            print(f"  Frame {frame}/{n_frames} (t={t:.3f})")

    return np.array(all_fields), times


def export_blender_sequence(
    model_path=None,
    output_dir="blender_export",
    resolution=128,
    n_frames=60,
    t_start=0.05,
    t_end=0.5,
    seed=42,
    allow_random_init=False,
):
    """Export wavefield animation as image sequence for Blender."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    checkpoint = None
    if model_path:
        model, checkpoint, device = load_wavepinn_checkpoint(model_path, device=device)
        print(f"Loaded weights from: {model_path}")
    elif allow_random_init:
        torch.manual_seed(seed)
        model = WavePINN().to(device)
        model.eval()
        print("Using randomly initialized model (explicit demo mode)")
    else:
        raise FileNotFoundError(
            "No checkpoint supplied. Pass --model /path/to/checkpoint.pt "
            "or use --allow-random-init for explicit demo-only exports."
        )
    
    print(f"\nExporting {n_frames} frames at {resolution}x{resolution}...")

    all_fields, times = sample_wavefield_sequence(
        model=model,
        resolution=resolution,
        n_frames=n_frames,
        t_start=t_start,
        t_end=t_end,
        device=device,
    )

    # Normalize and save as images
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
        "model_path": str(model_path) if model_path else None,
        "model_config": checkpoint.get("model_config") if checkpoint else None,
        "random_init": checkpoint is None,
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
    parser.add_argument("--allow-random-init", action="store_true")
    args = parser.parse_args()
    
    export_blender_sequence(
        model_path=args.model,
        output_dir=args.output,
        resolution=args.resolution,
        n_frames=args.frames,
        t_start=args.t_start,
        t_end=args.t_end,
        seed=args.seed,
        allow_random_init=args.allow_random_init,
    )


if __name__ == "__main__":
    main()
