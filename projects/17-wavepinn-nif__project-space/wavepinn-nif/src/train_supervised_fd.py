#!/usr/bin/env python3
"""
Train a WavePINN-style neural implicit field on a finite-difference wave reference.

This is a pragmatic fallback when the pure PINN objectives converge to visually
smooth, low-energy fields instead of a convincing propagating wave.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from export_blender import WavePINN
from train_v2 import generate_velocity, set_seed
from validate_wavefield import maybe_save_gif, save_slice_xt_png, save_snapshots_png


DEFAULT_MODEL_CONFIG = {
    "hidden_dims": [256, 256, 256, 128],
    "use_fourier": True,
    "num_fourier": 64,
    "fourier_scale": 8.0,
}


def ricker_wavelet(t: torch.Tensor, f0: float, t0: float) -> torch.Tensor:
    a = (math.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def make_source_mask(
    nx: int,
    nz: int,
    source_x: float,
    source_z: float,
    sigma: float,
    device: str,
) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, device=device)
    z = torch.linspace(0.0, 1.0, nz, device=device)
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    mask = torch.exp(-((xx - source_x) ** 2 + (zz - source_z) ** 2) / (2.0 * sigma**2))
    mask = mask / mask.max().clamp_min(1e-8)
    return mask.unsqueeze(0).unsqueeze(0)


def make_damping_mask(nx: int, nz: int, width: int, strength: float, device: str) -> torch.Tensor:
    ix = torch.arange(nx, device=device)
    iz = torch.arange(nz, device=device)
    dist_x = torch.minimum(ix, (nx - 1) - ix).float()
    dist_z = torch.minimum(iz, (nz - 1) - iz).float()
    taper_x = torch.clamp((width - dist_x) / max(width, 1), min=0.0)
    taper_z = torch.clamp((width - dist_z) / max(width, 1), min=0.0)
    edge = torch.maximum(taper_x[:, None], taper_z[None, :])
    damping = torch.exp(-strength * edge**2)
    return damping.unsqueeze(0).unsqueeze(0)


def simulate_reference_wavefield(
    velocity: torch.Tensor,
    n_frames: int,
    total_time: float,
    source_x: float,
    source_z: float,
    source_sigma: float,
    source_amplitude: float,
    source_frequency: float,
    damping_width: int,
    damping_strength: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a structured 2D wavefield with a simple finite-difference solver."""
    device = velocity.device
    nx, nz = velocity.shape
    dx = 1.0 / (nx - 1)
    dz = 1.0 / (nz - 1)
    vmax = float(velocity.max().item())
    dt = 0.35 * min(dx, dz) / (vmax * math.sqrt(2.0))
    n_steps = max(int(math.ceil(total_time / dt)), n_frames)

    laplace_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3) / (dx * dz)

    c2 = (velocity**2).unsqueeze(0).unsqueeze(0)
    source_mask = make_source_mask(nx, nz, source_x, source_z, source_sigma, device)
    damping = make_damping_mask(nx, nz, damping_width, damping_strength, device)

    u_prev = torch.zeros((1, 1, nx, nz), device=device)
    u_curr = torch.zeros_like(u_prev)

    sample_steps = np.linspace(0, n_steps - 1, n_frames, dtype=int)
    frames = []
    times = []
    sample_cursor = 0
    source_t0 = 1.5 / source_frequency

    for step in range(n_steps):
        lap = F.conv2d(F.pad(u_curr, (1, 1, 1, 1), mode="replicate"), laplace_kernel)
        t = step * dt
        src = source_amplitude * ricker_wavelet(torch.tensor(t, device=device), source_frequency, source_t0)
        u_next = 2.0 * u_curr - u_prev + (dt**2) * (c2 * lap + src * source_mask)
        u_next = u_next * damping
        u_prev, u_curr = u_curr, u_next

        if sample_cursor < len(sample_steps) and step == sample_steps[sample_cursor]:
            frames.append(u_curr.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32))
            times.append(t)
            sample_cursor += 1

    fields = np.stack(frames, axis=0)
    times = np.asarray(times, dtype=np.float32)
    return fields, times, dt


def build_training_tensors(fields: np.ndarray, times: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n_frames, nx, nz = fields.shape
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    spatial = np.stack([xx.reshape(-1), zz.reshape(-1)], axis=-1)

    coords = []
    targets = []
    for frame_idx, t in enumerate(times):
        coords.append(np.column_stack([spatial, np.full((spatial.shape[0],), t, dtype=np.float32)]))
        targets.append(fields[frame_idx].reshape(-1, 1))

    coords_tensor = torch.from_numpy(np.concatenate(coords, axis=0)).to(device)
    targets_tensor = torch.from_numpy(np.concatenate(targets, axis=0)).to(device)
    return coords_tensor, targets_tensor


def evaluate_model(
    model: WavePINN,
    coords: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 262144,
) -> dict:
    err_sq = 0.0
    target_sq = 0.0
    abs_err = 0.0
    n = 0

    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch_coords = coords[start:start + batch_size]
            batch_targets = targets[start:start + batch_size]
            pred = model(batch_coords)
            diff = pred - batch_targets
            err_sq += torch.sum(diff**2).item()
            target_sq += torch.sum(batch_targets**2).item()
            abs_err += torch.sum(torch.abs(diff)).item()
            n += batch_targets.numel()

    mse = err_sq / max(n, 1)
    mae = abs_err / max(n, 1)
    rel_l2 = math.sqrt(err_sq / max(target_sq, 1e-12))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {
        "mse": mse,
        "mae": mae,
        "relative_l2": rel_l2,
        "psnr_db": psnr,
    }


def save_reference_artifacts(
    fields: np.ndarray,
    times: np.ndarray,
    velocity: torch.Tensor,
    output_dir: Path,
    source_x: float,
    source_z: float,
    reference_scale: float,
    dt: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "reference_wavefield.npy", fields)
    np.save(output_dir / "times.npy", times)
    np.save(output_dir / "velocity.npy", velocity.detach().cpu().numpy())

    snapshot_indices = np.linspace(0, len(times) - 1, min(8, len(times)), dtype=int)
    save_snapshots_png(fields[snapshot_indices], times[snapshot_indices], output_dir / "reference_snapshots.png")
    z_index = int(np.clip(round(source_z * (fields.shape[2] - 1)), 0, fields.shape[2] - 1))
    save_slice_xt_png(fields, times, output_dir / "reference_slice_xt.png", z_index=z_index)
    maybe_save_gif(fields, output_dir / "reference_wavefield.gif")

    metadata = {
        "resolution": int(fields.shape[1]),
        "frames": int(fields.shape[0]),
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
        "dt": float(dt),
        "reference_scale": float(reference_scale),
        "source_x": float(source_x),
        "source_z": float(source_z),
    }
    with open(output_dir / "reference_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a supervised implicit wavefield model from FD reference")
    parser.add_argument("--output", default="outputs/supervised_fd", help="Base output directory")
    parser.add_argument("--grid", type=int, default=128, help="Reference simulation grid resolution")
    parser.add_argument("--frames", type=int, default=96, help="Number of stored frames")
    parser.add_argument("--total_time", type=float, default=0.55, help="Final simulation time")
    parser.add_argument("--steps", type=int, default=5000, help="Training optimization steps")
    parser.add_argument("--batch_size", type=int, default=32768, help="Random batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_amplitude", type=float, default=150000.0)
    parser.add_argument("--source_frequency", type=float, default=12.0)
    parser.add_argument("--source_sigma", type=float, default=0.015)
    parser.add_argument("--source_x", type=float, default=0.5)
    parser.add_argument("--source_z", type=float, default=0.15)
    parser.add_argument("--damping_width", type=int, default=18)
    parser.add_argument("--damping_strength", type=float, default=0.35)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    reference_dir = output_dir / "reference"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    velocity = generate_velocity(args.grid, args.grid, args.seed).to(device)
    print(f"Velocity range: [{velocity.min().item():.3f}, {velocity.max().item():.3f}]")

    reference_fields, times, dt = simulate_reference_wavefield(
        velocity=velocity,
        n_frames=args.frames,
        total_time=args.total_time,
        source_x=args.source_x,
        source_z=args.source_z,
        source_sigma=args.source_sigma,
        source_amplitude=args.source_amplitude,
        source_frequency=args.source_frequency,
        damping_width=args.damping_width,
        damping_strength=args.damping_strength,
    )
    reference_scale = float(np.max(np.abs(reference_fields)))
    reference_fields = reference_fields / max(reference_scale, 1e-8)
    save_reference_artifacts(
        fields=reference_fields,
        times=times,
        velocity=velocity,
        output_dir=reference_dir,
        source_x=args.source_x,
        source_z=args.source_z,
        reference_scale=reference_scale,
        dt=dt,
    )
    print(f"Reference wavefield saved to {reference_dir}")

    coords, targets = build_training_tensors(reference_fields, times, device=device)
    n_samples = coords.shape[0]
    print(f"Training samples: {n_samples:,}")

    model = WavePINN(
        hidden_dims=DEFAULT_MODEL_CONFIG["hidden_dims"],
        use_fourier=DEFAULT_MODEL_CONFIG["use_fourier"],
        num_fourier=DEFAULT_MODEL_CONFIG["num_fourier"],
        fourier_scale=DEFAULT_MODEL_CONFIG["fourier_scale"],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-5)

    history = []
    best_val = float("inf")
    best_state = None
    val_size = min(131072, n_samples)
    val_indices = torch.randperm(n_samples, device=device)[:val_size]
    train_start = time.time()

    for step in range(args.steps):
        batch_indices = torch.randint(0, n_samples, (args.batch_size,), device=device)
        pred = model(coords[batch_indices])
        loss = F.mse_loss(pred, targets[batch_indices])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                val_pred = model(coords[val_indices])
                val_loss = F.mse_loss(val_pred, targets[val_indices]).item()
            elapsed = time.time() - train_start
            lr = optimizer.param_groups[0]["lr"]
            history.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "lr": float(lr),
                "elapsed_sec": float(elapsed),
            })
            print(
                f"Step {step:5d} | Train {loss.item():.4e} | Val {val_loss:.4e} | "
                f"LR {lr:.2e} | {elapsed:.1f}s"
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_metrics = evaluate_model(model, coords, targets)
    print(
        f"Eval MSE {eval_metrics['mse']:.4e} | "
        f"Relative L2 {eval_metrics['relative_l2']:.4e} | "
        f"PSNR {eval_metrics['psnr_db']:.2f} dB"
    )

    checkpoint_path = checkpoint_dir / "wavefield_nif_supervised.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": DEFAULT_MODEL_CONFIG,
        "history": history,
        "velocity": velocity.detach().cpu(),
        "training_args": {
            "grid": args.grid,
            "frames": args.frames,
            "total_time": args.total_time,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "final_metrics": {
            **eval_metrics,
            "best_val_loss": best_val,
            "reference_scale": reference_scale,
        },
        "reference": {
            "dt": dt,
            "source_x": args.source_x,
            "source_z": args.source_z,
            "source_sigma": args.source_sigma,
            "source_amplitude": args.source_amplitude,
            "source_frequency": args.source_frequency,
            "damping_width": args.damping_width,
            "damping_strength": args.damping_strength,
            "reference_dir": str(reference_dir),
        },
        "script": "src/train_supervised_fd.py",
    }
    torch.save(checkpoint, checkpoint_path)
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
