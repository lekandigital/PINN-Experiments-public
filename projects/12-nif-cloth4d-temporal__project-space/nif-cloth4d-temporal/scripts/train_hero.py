#!/usr/bin/env python3
"""
Slim training loop for Project 12 (NIF-Cloth4D Temporal), Family A.

Why this exists instead of using ScheduledSamplingTrainer:

  * The stock trainer's `positions` branch is structurally broken — the model
    outputs a scalar SDF (out_dim=1) but the branch reshapes it to 3-vectors
    and adds it to prev_pos (trainer.py:406-407). It would ValueError at first
    batch.
  * The `sdf_samples` branch works but is overkill for Family A without GRU,
    and it pulls in scheduled-sampling/curriculum machinery that does not
    apply when each batch item is a single frame.
  * PhysicsLossStack's stretch/bend/momentum terms target vertex positions
    and require `set_mesh_topology` which the upstream trainer never calls.

So: MSE on SDF samples + eikonal regularizer on the field gradient. Clean.

Usage:
    # Hero (GRU off)
    python scripts/train_hero.py --no_gru --epochs 100 --output_dir checkpoints/hero

    # Ablation (GRU on, batch-local state)
    python scripts/train_hero.py --use_gru --epochs 100 --output_dir checkpoints/ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.drape_sequence_dataset import DrapeSDFDataset, collate_concat  # noqa: E402
from src.losses.physics_losses import compute_eikonal_loss  # noqa: E402
from src.models.fourier_mlp import FourierFeatureMLP  # noqa: E402
from src.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(PROJECT_ROOT / "data" / "drape"))
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "checkpoints" / "hero"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--samples_per_frame", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_eik", type=float, default=0.1)
    p.add_argument("--eik_points", type=int, default=1024,
                   help="Extra points for eikonal regularizer per batch.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")

    gru = p.add_mutually_exclusive_group()
    gru.add_argument("--use_gru", action="store_true")
    gru.add_argument("--no_gru", action="store_true")

    # Architecture (mirrors ModelConfig defaults, overrideable for ablations)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_freqs", type=int, default=16)
    p.add_argument("--fourier_scale", type=float, default=10.0)
    p.add_argument("--omega_0", type=float, default=30.0)
    p.add_argument("--gru_hidden", type=int, default=128)
    return p.parse_args()


def build_model(args: argparse.Namespace) -> FourierFeatureMLP:
    use_gru = args.use_gru and not args.no_gru
    return FourierFeatureMLP(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
        fourier_scale=args.fourier_scale,
        omega_0=args.omega_0,
        use_gru=use_gru,
        gru_hidden=args.gru_hidden,
    )


def eikonal_term(
    model: FourierFeatureMLP,
    coords: torch.Tensor,
    n_extra: int,
    bounds: float = 1.0,
) -> torch.Tensor:
    """
    Sample n_extra random (x,y,z,t) points inside the bounds, compute the
    spatial gradient of the field, return the eikonal loss.

    Times are drawn uniformly from the batch's own time column so the
    regularizer sees the same temporal distribution as the data loss.
    """
    device = coords.device
    xyz = (torch.rand(n_extra, 3, device=device) * 2.0 - 1.0) * bounds
    t_sample = coords[torch.randint(0, coords.shape[0], (n_extra,), device=device), 3:4]
    # cuDNN RNN does not support double-backward (needed for grad of grad).
    with torch.backends.cudnn.flags(enabled=False):
        _, grad, _ = model.compute_gradient(xyz, t_sample)
    return compute_eikonal_loss(grad)


def train_one_epoch(
    model: FourierFeatureMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    model.train()
    running_mse = 0.0
    running_eik = 0.0
    running_total = 0.0
    n_batches = 0
    use_amp = scaler is not None
    for coords, sdf_target in loader:
        coords = coords.to(device, non_blocking=True)
        sdf_target = sdf_target.to(device, non_blocking=True)
        xyz = coords[..., :3]
        t = coords[..., 3:4]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                pred = model.forward_batch(xyz, t)
                mse = F.mse_loss(pred, sdf_target)
            # Eikonal requires autograd on spatial coords — run in fp32 outside autocast
            eik = eikonal_term(model, coords, args.eik_points)
            total = mse + args.lambda_eik * eik
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model.forward_batch(xyz, t)
            mse = F.mse_loss(pred, sdf_target)
            eik = eikonal_term(model, coords, args.eik_points)
            total = mse + args.lambda_eik * eik
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_mse += float(mse.detach())
        running_eik += float(eik.detach())
        running_total += float(total.detach())
        n_batches += 1

    return running_total / n_batches, running_mse / n_batches, running_eik / n_batches


@torch.no_grad()
def validate(
    model: FourierFeatureMLP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for coords, sdf_target in loader:
        coords = coords.to(device, non_blocking=True)
        sdf_target = sdf_target.to(device, non_blocking=True)
        pred = model.forward_batch(coords[..., :3], coords[..., 3:4])
        total += float(F.mse_loss(pred, sdf_target))
        n += 1
    return total / max(n, 1)


def save_checkpoint(
    path: Path,
    model: FourierFeatureMLP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler | None,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "config": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    use_gru = args.use_gru and not args.no_gru
    print(f"[train] device={device}, amp={use_amp}, use_gru={use_gru}")

    dataset = DrapeSDFDataset(
        data_dir=args.data_dir,
        samples_per_frame=args.samples_per_frame,
        near_surface_ratio=0.7,
        surface_band=0.1,
        preload=True,
    )
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_concat,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_concat,
    )
    print(f"[train] dataset: {n_train} train frames, {n_val} val frames, "
          f"{args.samples_per_frame} samples/frame")

    model = build_model(args).to(device)
    print(model)
    print(f"[train] parameters: {model.count_parameters():,}")

    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.learning_rate * 0.01)
    scaler = GradScaler() if use_amp else None

    best_val = float("inf")
    history = []
    t_start = time_mod.perf_counter()
    for epoch in range(args.epochs):
        t_ep = time_mod.perf_counter()
        train_total, train_mse, train_eik = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        dt_ep = time_mod.perf_counter() - t_ep
        line = (f"epoch {epoch+1:3d}/{args.epochs} | "
                f"train total {train_total:.6f} "
                f"mse {train_mse:.6f} eik {train_eik:.6f} | "
                f"val {val_loss:.6f} | lr {lr:.2e} | "
                f"dt {dt_ep:.1f}s")
        print(line, flush=True)
        history.append({
            "epoch": epoch,
            "train_total": train_total,
            "train_mse": train_mse,
            "train_eik": train_eik,
            "val": val_loss,
            "lr": lr,
            "seconds": dt_ep,
        })

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                model, optimizer, scheduler, scaler,
                epoch, best_val, args,
            )

    # Final checkpoint with the canonical name
    final_name = "nif_cloth4d_temporal_gru.pt" if use_gru else "nif_cloth4d_temporal.pt"
    save_checkpoint(
        output_dir / final_name,
        model, optimizer, scheduler, scaler,
        args.epochs - 1, best_val, args,
    )

    total_dt = time_mod.perf_counter() - t_start
    summary = {
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "final_train_total": history[-1]["train_total"],
        "final_train_mse": history[-1]["train_mse"],
        "final_train_eik": history[-1]["train_eik"],
        "wall_clock_seconds": round(total_dt, 2),
        "wall_clock_minutes": round(total_dt / 60.0, 2),
        "use_gru": use_gru,
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / final_name),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(
            {"summary": summary, "history": history, "args": vars(args)},
            f, indent=2,
        )
    print(f"[train] done in {total_dt/60:.2f} min | best val {best_val:.6f}")


if __name__ == "__main__":
    main()
