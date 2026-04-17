#!/usr/bin/env python3
"""Autoregressive rollout of the trained HGNN-NIF model + comparison metrics
against the physics baseline. Writes:

  outputs/hgnn_nif_prediction/cloth_sequence.npy        (T, N, 3)
  outputs/metrics/rollout_metrics.npz
  outputs/metrics/rollout_error.png
  outputs/mesh_info.json   (mesh topology + trustworthy rollout window)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import HGNN_NIF_ClothModel
from src.models.temporal import TemporalHGNN_NIF


def build_model(args, info, device):
    base = HGNN_NIF_ClothModel(
        node_feat_dim=3,
        latent_dim=args["latent_dim"],
        hidden_dim=args["hidden_dim"],
        hgnn_layers=args["hgnn_layers"],
        siren_hidden_dim=128,
        siren_layers=3,
        num_heads=4,
    )
    model = TemporalHGNN_NIF(
        base,
        history_frames=args["history"],
        temporal_hidden=args["temporal_hidden"],
    ).to(device)
    return model


def edge_lengths(pos, edges):
    diff = pos[..., edges[0], :] - pos[..., edges[1], :]
    return np.linalg.norm(diff, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/hgnn_nif_cloth_trained.pt")
    p.add_argument("--data", type=str, default="outputs/physics_baseline")
    p.add_argument("--out", type=str, default="outputs/hgnn_nif_prediction")
    p.add_argument("--metrics_out", type=str, default="outputs/metrics")
    p.add_argument("--mesh_info", type=str, default="outputs/mesh_info.json")
    p.add_argument("--frames", type=int, default=0, help="rollout frames (0=match physics)")
    p.add_argument("--rmse_threshold", type=float, default=0.05,
                   help="per-frame RMSE in world units that defines the trustworthy window")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    info = ckpt["info"]

    data_dir = Path(args.data)
    gt = np.load(data_dir / "cloth_sequence.npy")  # (T, N, 3)
    topo = np.load(data_dir / "topology.npz")
    fine_edges_np = topo["fine_edges"]
    coarse_edges_np = topo["coarse_edges"]

    fine_edges = torch.from_numpy(fine_edges_np).long().to(device)
    coarse_edges = torch.from_numpy(coarse_edges_np).long().to(device)
    rest_lengths_np = topo["rest_lengths"]

    side = info["resolution"]
    fixed_count = side * info["fixed_rows"]

    model = build_model(train_args, info, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    H = train_args["history"]
    T_gt = gt.shape[0]
    T = args.frames if args.frames > 0 else T_gt

    # Seed the rollout with the first H ground-truth frames
    history = torch.from_numpy(gt[:H]).float().to(device).unsqueeze(0)  # (1, H, N, 3)
    seed_pos = torch.from_numpy(gt[H - 1]).float().to(device)            # initial fixed positions
    fixed_pos = seed_pos[:fixed_count]

    pred_seq = list(gt[:H].copy())  # carry the seed frames in the output for alignment

    print(f"Rolling out {T - H} frames (seed = first {H} GT frames)...")
    with torch.no_grad():
        for t in range(H, T):
            out = model(history, fine_edges, coarse_edges)
            pred = out["positions"][0]  # (N, 3)
            pred[:fixed_count] = fixed_pos  # hard pin
            pred_seq.append(pred.detach().cpu().numpy())
            new_hist = torch.cat([history[:, 1:], pred[None, None]], dim=1)
            history = new_hist
    pred_arr = np.stack(pred_seq).astype(np.float32)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cloth_sequence.npy", pred_arr)

    # Metrics
    Tm = min(pred_arr.shape[0], gt.shape[0])
    diff = pred_arr[:Tm] - gt[:Tm]                                 # (T, N, 3)
    per_frame_rmse = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=-1))  # (T,)
    per_vertex_rmse = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=0)  # (N,)

    pred_edge_len = edge_lengths(pred_arr[:Tm], fine_edges_np)
    gt_edge_len = edge_lengths(gt[:Tm], fine_edges_np)
    edge_strain_pred = np.abs(pred_edge_len - rest_lengths_np[None]).mean(axis=-1)
    edge_strain_gt = np.abs(gt_edge_len - rest_lengths_np[None]).mean(axis=-1)
    edge_len_err = np.abs(pred_edge_len - gt_edge_len).mean(axis=-1)

    # Trustworthy window = last frame whose RMSE stays under threshold
    over = np.where(per_frame_rmse > args.rmse_threshold)[0]
    trustworthy_frame = int(over[0]) if len(over) > 0 else int(Tm - 1)

    metrics_dir = Path(args.metrics_out); metrics_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metrics_dir / "rollout_metrics.npz",
        per_frame_rmse=per_frame_rmse,
        per_vertex_rmse=per_vertex_rmse,
        edge_strain_pred=edge_strain_pred,
        edge_strain_gt=edge_strain_gt,
        edge_len_err=edge_len_err,
        rmse_threshold=args.rmse_threshold,
        trustworthy_frame=trustworthy_frame,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(per_frame_rmse, label="pred vs phys RMSE")
        axes[0].axhline(args.rmse_threshold, color="r", linestyle="--",
                        label=f"threshold={args.rmse_threshold}")
        axes[0].axvline(trustworthy_frame, color="k", linestyle=":",
                        label=f"trust frame={trustworthy_frame}")
        axes[0].set_xlabel("frame"); axes[0].set_ylabel("RMSE (world units)")
        axes[0].set_title("Per-frame position RMSE")
        axes[0].legend()
        axes[1].plot(edge_strain_pred, label="HGNN-NIF")
        axes[1].plot(edge_strain_gt, label="physics")
        axes[1].plot(edge_len_err, label="|pred-phys| edge len", linestyle="--")
        axes[1].set_xlabel("frame"); axes[1].set_ylabel("mean edge deviation")
        axes[1].set_title("Edge length preservation")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(metrics_dir / "rollout_error.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"plot skipped: {e}")

    # mesh_info.json — full topology summary + rollout result
    mesh_info = {
        "fine": {
            "resolution": info["resolution"],
            "num_vertices": int(info["num_vertices_fine"]),
            "num_edges": int(info["num_edges_fine"]),
            "num_faces": int(info["num_faces_fine"]),
        },
        "coarse": {
            "resolution": info["coarse_resolution"],
            "num_vertices": int(info["num_vertices_coarse"]),
            "num_edges": int(info["num_edges_coarse"]),
            "num_faces": int(info["num_faces_coarse"]),
        },
        "fps": info["fps"],
        "fixed_rows": info["fixed_rows"],
        "physics": info["physics"],
        "rollout": {
            "history_frames": H,
            "total_frames": int(pred_arr.shape[0]),
            "compared_frames": int(Tm),
            "rmse_threshold": args.rmse_threshold,
            "trustworthy_frame": trustworthy_frame,
            "trustworthy_seconds": trustworthy_frame / info["fps"],
            "final_frame_rmse": float(per_frame_rmse[-1]),
            "mean_rmse": float(per_frame_rmse.mean()),
            "max_rmse": float(per_frame_rmse.max()),
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    Path(args.mesh_info).parent.mkdir(parents=True, exist_ok=True)
    Path(args.mesh_info).write_text(json.dumps(mesh_info, indent=2))

    print("\n=== rollout summary ===")
    print(f"frames compared: {Tm}")
    print(f"mean RMSE: {per_frame_rmse.mean():.4f}")
    print(f"max  RMSE: {per_frame_rmse.max():.4f}")
    print(f"trustworthy frame (RMSE<{args.rmse_threshold}): {trustworthy_frame} "
          f"({trustworthy_frame/info['fps']:.2f} s)")
    print(f"prediction -> {out_dir/'cloth_sequence.npy'}")
    print(f"metrics    -> {metrics_dir/'rollout_metrics.npz'}")
    print(f"mesh_info  -> {args.mesh_info}")


if __name__ == "__main__":
    main()
