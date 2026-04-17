#!/usr/bin/env python3
"""
NIF-Cloth4D: Full Training & Export Pipeline

Generates synthetic data, trains SIREN model, extracts heightfield meshes
with consistent topology, generates physics baseline, and computes metrics.

Run on GPU machine (3090 Ti target):
    python scripts/run_full_pipeline.py --epochs 500

The key insight: the synthetic cloth is a thin-shell heightfield z = f(x,y,t).
Instead of marching cubes (inconsistent topology), we extract a regular-grid
heightfield by finding the z of minimum SDF for each (x,y) column. This gives
fixed vertex/face counts across all timesteps — ideal for animation.
"""

import os
import sys
import time
import json
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works whether run from project root or scripts/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR))

from synthetic_data import generate_dataset
from nif_cloth4d import create_model
from train_nif_cloth4d import Trainer, save_checkpoint


# ============================================================================
# Heightfield mesh extraction
# ============================================================================

def make_grid_faces(res: int) -> np.ndarray:
    """Create triangle face indices for a res x res regular grid.

    Each quad is split into 2 triangles. Vertex indexing: row-major,
    vertex (i, j) -> index i * res + j.
    """
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            v00 = i * res + j
            v10 = (i + 1) * res + j
            v11 = (i + 1) * res + (j + 1)
            v01 = i * res + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return np.array(faces, dtype=np.int32)


def extract_heightfield_nif(
    model: torch.nn.Module,
    times: np.ndarray,
    grid_res: int = 128,
    z_samples: int = 256,
    xy_range: tuple = (-0.9, 0.9),
    z_range: tuple = (-1.0, 1.0),
    device: str = "cuda",
    chunk_size: int = 65536,
) -> np.ndarray:
    """Extract heightfield mesh from trained NIF model.

    For each (x, y) on a regular grid, query the SDF at many z values and
    pick the z where SDF is most negative (the centre of the thin shell).

    Returns: vertices array of shape [T, grid_res*grid_res, 3].
    """
    model.eval()
    xs = np.linspace(xy_range[0], xy_range[1], grid_res, dtype=np.float32)
    ys = np.linspace(xy_range[0], xy_range[1], grid_res, dtype=np.float32)
    zs = np.linspace(z_range[0], z_range[1], z_samples, dtype=np.float32)

    # (x, y) base grid — [grid_res, grid_res]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    n_xy = grid_res * grid_res

    all_vertices = []

    with torch.no_grad():
        for ti, t in enumerate(times):
            t0 = time.time()
            # Build full coord array: for each z, all (x,y) points
            # Layout: [z_samples * n_xy, 4]
            coords = np.empty((z_samples * n_xy, 4), dtype=np.float32)
            x_flat = X.ravel()
            y_flat = Y.ravel()
            for zi in range(z_samples):
                s = zi * n_xy
                e = s + n_xy
                coords[s:e, 0] = x_flat
                coords[s:e, 1] = y_flat
                coords[s:e, 2] = zs[zi]
                coords[s:e, 3] = t

            # Query model in chunks
            sdf_all = np.empty(len(coords), dtype=np.float32)
            for i in range(0, len(coords), chunk_size):
                batch = torch.from_numpy(coords[i : i + chunk_size]).to(device)
                sdf_all[i : i + chunk_size] = (
                    model(batch).cpu().numpy().ravel()
                )

            # Reshape to [z_samples, n_xy] and find z of min SDF per column
            sdf_grid = sdf_all.reshape(z_samples, n_xy)
            min_idx = np.argmin(sdf_grid, axis=0)  # [n_xy]
            z_surface = zs[min_idx]

            vertices = np.stack([x_flat, y_flat, z_surface], axis=-1)
            all_vertices.append(vertices)
            dt = time.time() - t0
            print(f"  t={t:.3f}  ({ti+1}/{len(times)})  "
                  f"z_range=[{z_surface.min():.3f}, {z_surface.max():.3f}]  "
                  f"{dt:.1f}s")

    return np.stack(all_vertices, axis=0).astype(np.float32)


def extract_heightfield_gt(
    times: np.ndarray,
    grid_res: int = 128,
    xy_range: tuple = (-0.9, 0.9),
) -> np.ndarray:
    """Extract heightfield from analytic ground-truth surface.

    Reproduces the formula from synthetic_data.create_falling_cloth_sdf:
        surface_z = lerp(flat, draped, t)
    """
    xs = np.linspace(xy_range[0], xy_range[1], grid_res, dtype=np.float32)
    ys = np.linspace(xy_range[0], xy_range[1], grid_res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    all_vertices = []
    for t in times:
        tc = float(np.clip(t, 0.0, 1.0))
        flat_z = 0.8
        wrinkle_amp = 0.2 * tc
        drape_depth = 0.6 * tc
        draped_z = (
            flat_z
            - drape_depth
            + wrinkle_amp * np.sin(4 * X) * np.sin(4 * Y)
            - 0.2 * tc * (X ** 2 + Y ** 2)
        )
        surface_z = (1 - tc) * flat_z + tc * draped_z
        vertices = np.stack(
            [X.ravel(), Y.ravel(), surface_z.ravel().astype(np.float32)], axis=-1
        )
        all_vertices.append(vertices)

    return np.stack(all_vertices, axis=0).astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Chamfer distance (mean of squared nearest-neighbour distances)."""
    from scipy.spatial import KDTree

    tree_A = KDTree(A)
    tree_B = KDTree(B)
    d_AB, _ = tree_B.query(A)
    d_BA, _ = tree_A.query(B)
    return float((np.mean(d_AB ** 2) + np.mean(d_BA ** 2)) / 2)


def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Hausdorff distance (max nearest-neighbour distance)."""
    from scipy.spatial import KDTree

    tree_A = KDTree(A)
    tree_B = KDTree(B)
    d_AB, _ = tree_B.query(A)
    d_BA, _ = tree_A.query(B)
    return float(max(np.max(d_AB), np.max(d_BA)))


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NIF-Cloth4D full pipeline: data → train → export → metrics"
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--data_dir", type=str, default="/tmp/nif_cloth4d_data")
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Training SDF frames")
    parser.add_argument("--grid_size", type=int, default=128,
                        help="SDF volume resolution (grid_size^3)")
    parser.add_argument("--mesh_res", type=int, default=128,
                        help="Heightfield grid resolution for export")
    parser.add_argument("--num_timesteps", type=int, default=12,
                        help="Timesteps in exported animation")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    checkpoint_dir = PROJECT_DIR / "checkpoints"
    output_base = PROJECT_DIR / "outputs"

    config = {
        "model": {
            "in_dim": 4,
            "cond_dim": 0,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "w0_initial": 30.0,
        },
        "data": {
            "samples_per_frame": 20000,
            "near_surface_ratio": 0.7,
            "surface_band": 0.05,
            "val_split": 0.1,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.0,
            "log_interval": 10,
            "val_interval": 10,
            "checkpoint_interval": 50,
            "loss_weights": {"sdf": 1.0, "eikonal": 0.01},
            "scheduler_params": {"T_max": args.epochs, "eta_min": 1e-6},
        },
        "hardware": {
            "device": "cuda",
            "num_workers": 0,
            "pin_memory": True,
        },
    }

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Generating Training Data")
    print("=" * 60)
    generate_dataset(
        output_dir=args.data_dir,
        num_frames=args.num_frames,
        grid_size=args.grid_size,
        data_type="falling",
    )

    # ------------------------------------------------------------------
    # Step 2: Train model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Training Model")
    print("=" * 60)
    trainer = Trainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=str(checkpoint_dir),
    )
    model = trainer.train()

    # Copy best model to canonical path
    best_pt = checkpoint_dir / "model_best.pt"
    final_pt = checkpoint_dir / "nif_cloth4d_trained.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, final_pt)
        print(f"Final checkpoint saved: {final_pt}")

    # ------------------------------------------------------------------
    # Step 3: Heightfield mesh extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Extracting Heightfield Meshes")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    times = np.linspace(0.0, 1.0, args.num_timesteps)
    faces = make_grid_faces(args.mesh_res)

    print(f"\nNIF prediction ({args.num_timesteps} timesteps, "
          f"{args.mesh_res}x{args.mesh_res} grid):")
    nif_verts = extract_heightfield_nif(
        model, times, grid_res=args.mesh_res, device=device
    )

    print(f"\nGround-truth baseline:")
    gt_verts = extract_heightfield_gt(times, grid_res=args.mesh_res)
    for ti, t in enumerate(times):
        zr = gt_verts[ti, :, 2]
        print(f"  t={t:.3f}  z_range=[{zr.min():.3f}, {zr.max():.3f}]")

    # Save outputs
    nif_dir = output_base / "nif_prediction"
    gt_dir = output_base / "physics_baseline"
    nif_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    np.save(nif_dir / "vertices.npy", nif_verts)
    np.save(nif_dir / "faces.npy", faces)
    np.save(gt_dir / "vertices.npy", gt_verts)
    np.save(gt_dir / "faces.npy", faces)

    mesh_info = {
        "vertex_count": int(args.mesh_res ** 2),
        "face_count": int(len(faces)),
        "timestep_count": int(args.num_timesteps),
        "timesteps": times.tolist(),
        "extraction_method": "heightfield_min_sdf",
        "grid_resolution": args.mesh_res,
        "topology": "fixed_regular_grid",
        "description": (
            "Heightfield extraction: for each (x,y) on a regular grid, "
            "find z where the NIF SDF is most negative (cloth shell centre). "
            "Gives identical vertex/face counts every timestep."
        ),
    }
    with open(output_base / "mesh_info.json", "w") as f:
        json.dump(mesh_info, f, indent=2)

    print(f"\nNIF vertices shape: {nif_verts.shape}")
    print(f"GT  vertices shape: {gt_verts.shape}")
    print(f"Faces shape:        {faces.shape}")

    # ------------------------------------------------------------------
    # Step 4: Compute metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Computing Metrics")
    print("=" * 60)

    metrics = {}
    chamfers, hausdorffs = [], []
    for ti, t in enumerate(times):
        cd = chamfer_distance(nif_verts[ti], gt_verts[ti])
        hd = hausdorff_distance(nif_verts[ti], gt_verts[ti])
        metrics[f"t={t:.3f}"] = {"chamfer": cd, "hausdorff": hd}
        chamfers.append(cd)
        hausdorffs.append(hd)
        print(f"  t={t:.3f}:  Chamfer={cd:.6f}  Hausdorff={hd:.6f}")

    avg_cd = float(np.mean(chamfers))
    avg_hd = float(np.mean(hausdorffs))
    metrics["average"] = {"chamfer": avg_cd, "hausdorff": avg_hd}
    print(f"\n  Average:  Chamfer={avg_cd:.6f}  Hausdorff={avg_hd:.6f}")

    with open(output_base / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time:       {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Checkpoint:       {final_pt}")
    print(f"NIF meshes:       {nif_dir}/")
    print(f"  vertices.npy    {nif_verts.shape}")
    print(f"  faces.npy       {faces.shape}")
    print(f"GT meshes:        {gt_dir}/")
    print(f"  vertices.npy    {gt_verts.shape}")
    print(f"  faces.npy       {faces.shape}")
    print(f"Mesh info:        {output_base / 'mesh_info.json'}")
    print(f"Metrics:          {output_base / 'metrics.json'}")
    print(f"Avg Chamfer:      {avg_cd:.6f}")
    print(f"Avg Hausdorff:    {avg_hd:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
