#!/usr/bin/env python3
"""Train HGNN-NIF (TemporalHGNN_NIF) to predict next-frame cloth positions
from a 3-frame history. Trains directly on the mass-spring physics sequence.

Loss = MSE(pred_pos, gt_pos) + lambda_spring * spring_loss + lambda_vel * MSE(pred_vel, gt_vel)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import HGNN_NIF_ClothModel
from src.models.temporal import TemporalHGNN_NIF


class PhysicsDataset(Dataset):
    """(history, future_K_frames) windows from a single physics rollout."""

    def __init__(self, positions, history=3, rollout_k=1, indices=None):
        self.pos = positions  # (T, N, 3) float32 tensor
        self.history = history
        self.rollout_k = rollout_k
        T = positions.shape[0]
        # Need history frames before i, and rollout_k frames at and after i
        all_idx = list(range(history, T - rollout_k + 1))
        self.indices = all_idx if indices is None else [i for i in all_idx if i in set(indices)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        hist = self.pos[i - self.history : i]              # (H, N, 3)
        future = self.pos[i : i + self.rollout_k]          # (K, N, 3)
        return hist, future


def spring_loss(pred_pos, edges, rest):
    diff = pred_pos[..., edges[0], :] - pred_pos[..., edges[1], :]
    cur = torch.norm(diff, dim=-1)
    return ((cur - rest) ** 2).mean()


def _step_once(model, hist, fine_edges, coarse_edges, fixed_mask):
    """Single autoregressive step. Returns (pred_pos, pred_vel)."""
    out = model(hist, fine_edges, coarse_edges)
    pred = out["positions"]
    if fixed_mask is not None:
        pred = torch.where(fixed_mask[None, :, None], hist[:, -1], pred)
    return pred, out["velocity"]


def rollout_loss(model, hist, future, fine_edges, coarse_edges, rest, fixed_mask, lambdas):
    """Unroll for K = future.shape[1] steps with BPTT, accumulating losses."""
    B, K, N, _ = future.shape
    cur_hist = hist
    l_pos_acc = 0.0
    l_vel_acc = 0.0
    l_spr_acc = 0.0
    for k in range(K):
        pred, pred_vel = _step_once(model, cur_hist, fine_edges, coarse_edges, fixed_mask)
        gt_pos = future[:, k]
        gt_vel = gt_pos - cur_hist[:, -1]
        l_pos_acc = l_pos_acc + F.mse_loss(pred, gt_pos)
        l_vel_acc = l_vel_acc + F.mse_loss(pred_vel, gt_vel)
        l_spr_acc = l_spr_acc + spring_loss(pred, fine_edges, rest)
        cur_hist = torch.cat([cur_hist[:, 1:], pred[:, None]], dim=1)
    l_pos = l_pos_acc / K
    l_vel = l_vel_acc / K
    l_spr = l_spr_acc / K
    total = l_pos + lambdas["vel"] * l_vel + lambdas["spring"] * l_spr
    return total, l_pos, l_vel, l_spr


def train_step(model, hist, future, fine_edges, coarse_edges, rest, fixed_mask, lambdas, scaler, opt, amp, noise_std):
    opt.zero_grad(set_to_none=True)
    if noise_std > 0:
        # Pushforward trick: noise on the history (esp. last frame) to simulate own-rollout drift
        noise = torch.randn_like(hist[:, -1]) * noise_std
        # Don't perturb pinned vertices
        noise[:, :hist.shape[2]] = noise[:, :hist.shape[2]]
        if fixed_mask is not None:
            noise = torch.where(fixed_mask[None, :, None], torch.zeros_like(noise), noise)
        hist = hist.clone()
        hist[:, -1] = hist[:, -1] + noise
    with torch.cuda.amp.autocast(enabled=amp):
        loss, l_pos, l_vel, l_spr = rollout_loss(
            model, hist, future, fine_edges, coarse_edges, rest, fixed_mask, lambdas,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return loss.item(), l_pos.item(), l_vel.item(), l_spr.item()


@torch.no_grad()
def eval_step(model, hist, future, fine_edges, coarse_edges, rest, fixed_mask, lambdas, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        loss, l_pos, l_vel, l_spr = rollout_loss(
            model, hist, future, fine_edges, coarse_edges, rest, fixed_mask, lambdas,
        )
    return loss.item(), l_pos.item(), l_vel.item(), l_spr.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="outputs/physics_baseline")
    p.add_argument("--checkpoint", type=str, default="checkpoints/hgnn_nif_cloth_trained.pt")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--hgnn_layers", type=int, default=2)
    p.add_argument("--temporal_hidden", type=int, default=64)
    p.add_argument("--lambda_vel", type=float, default=1.0)
    p.add_argument("--lambda_spring", type=float, default=0.05)
    p.add_argument("--rollout_k", type=int, default=4, help="BPTT rollout length during training")
    p.add_argument("--noise_std", type=float, default=0.003,
                   help="pushforward noise on history's last frame (world units)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    pos_np = np.load(data_dir / "cloth_sequence.npy")  # (T, N, 3)
    topo = np.load(data_dir / "topology.npz")
    info = json.loads((data_dir / "physics_info.json").read_text())

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    pos = torch.from_numpy(pos_np).float().to(device)               # (T, N, 3)
    fine_edges = torch.from_numpy(topo["fine_edges"]).long().to(device)
    coarse_edges = torch.from_numpy(topo["coarse_edges"]).long().to(device)
    rest = torch.from_numpy(topo["rest_lengths"]).float().to(device)

    side = info["resolution"]
    fixed_mask = torch.zeros(pos.shape[1], dtype=torch.bool, device=device)
    fixed_mask[: side * info["fixed_rows"]] = True

    T = pos.shape[0]
    all_idx = list(range(args.history, T - args.rollout_k + 1))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(all_idx)
    n_val = max(1, int(len(perm) * args.val_frac))
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())

    train_ds = PhysicsDataset(pos, history=args.history, rollout_k=args.rollout_k, indices=train_idx)
    val_ds = PhysicsDataset(pos, history=args.history, rollout_k=args.rollout_k, indices=val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    base = HGNN_NIF_ClothModel(
        node_feat_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        hgnn_layers=args.hgnn_layers,
        siren_hidden_dim=128,
        siren_layers=3,
        num_heads=4,
    )
    model = TemporalHGNN_NIF(base, history_frames=args.history, temporal_hidden=args.temporal_hidden).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")
    print(f"Device: {device}  AMP: {amp}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    lambdas = {"vel": args.lambda_vel, "spring": args.lambda_spring}
    history = []
    best_val = float("inf")
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = pos_sum = vel_sum = spr_sum = 0.0
        nb = 0
        for hist, future in train_loader:
            hist = hist.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            l, lp, lv, ls = train_step(
                model, hist, future, fine_edges, coarse_edges, rest,
                fixed_mask, lambdas, scaler, opt, amp, args.noise_std,
            )
            loss_sum += l; pos_sum += lp; vel_sum += lv; spr_sum += ls
            nb += 1
        sched.step()

        model.eval()
        v_loss = v_pos = v_vel = v_spr = 0.0
        vb = 0
        for hist, future in val_loader:
            hist = hist.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            l, lp, lv, ls = eval_step(
                model, hist, future, fine_edges, coarse_edges, rest,
                fixed_mask, lambdas, amp,
            )
            v_loss += l; v_pos += lp; v_vel += lv; v_spr += ls
            vb += 1

        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / nb,
            "train_pos": pos_sum / nb,
            "train_vel": vel_sum / nb,
            "train_spring": spr_sum / nb,
            "val_loss": v_loss / vb,
            "val_pos": v_pos / vb,
            "val_vel": v_vel / vb,
            "val_spring": v_spr / vb,
            "lr": opt.param_groups[0]["lr"],
            "time_s": time.time() - t0,
        }
        history.append(rec)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"ep {epoch:4d}  "
                f"train {rec['train_loss']:.5f} (pos {rec['train_pos']:.5f})  "
                f"val {rec['val_loss']:.5f} (pos {rec['val_pos']:.5f})  "
                f"lr {rec['lr']:.2e}  {rec['time_s']:.1f}s"
            )

        if rec["val_loss"] < best_val:
            best_val = rec["val_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_state_dict": base.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "args": vars(args),
                "history": history,
                "info": info,
            }, args.checkpoint)

    final_path = Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "base_state_dict": base.state_dict(),
        "epoch": args.epochs - 1,
        "val_loss": history[-1]["val_loss"],
        "args": vars(args),
        "history": history,
        "info": info,
    }, final_path)

    Path("outputs/training_history.json").parent.mkdir(parents=True, exist_ok=True)
    Path("outputs/training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val_loss = {best_val:.6f}")
    print(f"Saved best -> {args.checkpoint}")
    print(f"Saved final -> {final_path}")


if __name__ == "__main__":
    main()
