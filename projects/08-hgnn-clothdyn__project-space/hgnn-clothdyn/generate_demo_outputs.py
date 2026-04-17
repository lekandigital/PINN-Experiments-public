"""
Generate baseline and HGNN rollout outputs for the cloth demo.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from benchmark import load_model_from_checkpoint
from mesh_to_graph import build_graph_pyramid, create_grid_mesh
from train import ClothDataset, set_seed


FRAME_DT = 0.01
GROUND_HEIGHT = -1.0


def pool_positions(
    fine_pos: torch.Tensor,
    cluster_map: torch.Tensor,
    num_coarse: int,
) -> torch.Tensor:
    """Pool fine positions to the coarse graph with mean aggregation."""
    coarse_pos = torch.zeros(num_coarse, 3, device=fine_pos.device, dtype=fine_pos.dtype)
    counts = torch.zeros(num_coarse, device=fine_pos.device, dtype=fine_pos.dtype)

    coarse_pos.scatter_add_(0, cluster_map.unsqueeze(1).expand(-1, 3), fine_pos)
    counts.scatter_add_(0, cluster_map, torch.ones(fine_pos.size(0), device=fine_pos.device))

    return coarse_pos / counts.unsqueeze(1).clamp(min=1)


def build_graph_structures(
    dataset: ClothDataset,
    device: torch.device,
) -> Tuple[Data, Data | None, torch.Tensor | None]:
    """Build fine/coarse graph templates from the initial frame."""
    initial_data = dataset.get_frame(0, normalized=True).to(device)
    graphs, cluster_maps = build_graph_pyramid(initial_data, num_levels=2)

    fine_template = graphs[0]
    coarse_template = graphs[1].to(device) if len(graphs) > 1 else None
    cluster_map = cluster_maps[0].to(device) if cluster_maps else None

    return fine_template, coarse_template, cluster_map


@torch.no_grad()
def rollout_prediction(
    model: torch.nn.Module,
    dataset: ClothDataset,
    device: torch.device,
) -> np.ndarray:
    """Roll out the recovered model over the full baseline sequence."""
    fine_template, coarse_template, cluster_map = build_graph_structures(dataset, device)

    baseline_positions = dataset.positions.to(device)
    baseline_velocities = dataset.velocities.to(device)

    current_pos = dataset.normalize_positions(baseline_positions[0].clone())
    current_vel = dataset.normalize_velocities(baseline_velocities[0].clone())

    predicted_positions = [baseline_positions[0].cpu().numpy()]

    for _ in range(1, dataset.num_frames):
        fine_data = Data(
            x=torch.cat([current_pos, current_vel], dim=-1),
            edge_index=fine_template.edge_index,
            edge_attr=fine_template.edge_attr,
            pos=current_pos,
            num_nodes=dataset.num_nodes,
        ).to(device)

        if coarse_template is not None and cluster_map is not None:
            coarse_pos = pool_positions(current_pos, cluster_map, coarse_template.num_nodes)
            coarse_data = Data(
                x=torch.cat([coarse_pos, torch.zeros_like(coarse_pos)], dim=-1),
                edge_index=coarse_template.edge_index,
                edge_attr=coarse_template.edge_attr,
                pos=coarse_pos,
                num_nodes=coarse_template.num_nodes,
            ).to(device)
        else:
            coarse_data = None

        delta_vel = model(fine_data, coarse_data, cluster_map)
        next_vel = (current_vel + delta_vel) * 0.99
        next_pos = current_pos + next_vel * FRAME_DT

        next_pos_raw = dataset.denormalize_positions(next_pos)
        next_vel_raw = dataset.denormalize_velocities(next_vel)

        collision_mask = next_pos_raw[:, 1] < GROUND_HEIGHT
        next_pos_raw[collision_mask, 1] = GROUND_HEIGHT
        next_vel_raw[collision_mask, 1] = torch.abs(next_vel_raw[collision_mask, 1]) * 0.5

        predicted_positions.append(next_pos_raw.cpu().numpy())
        current_pos = dataset.normalize_positions(next_pos_raw)
        current_vel = dataset.normalize_velocities(next_vel_raw)

    return np.stack(predicted_positions, axis=0)


def compute_metrics(
    baseline_positions: np.ndarray,
    predicted_positions: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
) -> Dict[str, Any]:
    """Compute rollout and structural preservation metrics."""
    rmse_per_frame = np.sqrt(np.mean((predicted_positions - baseline_positions) ** 2, axis=(1, 2)))

    baseline_edge_lengths = np.linalg.norm(
        baseline_positions[:, edges[:, 0]] - baseline_positions[:, edges[:, 1]],
        axis=-1,
    )
    predicted_edge_lengths = np.linalg.norm(
        predicted_positions[:, edges[:, 0]] - predicted_positions[:, edges[:, 1]],
        axis=-1,
    )

    baseline_rest_mae = np.mean(np.abs(baseline_edge_lengths - rest_lengths[None, :]), axis=1)
    predicted_rest_mae = np.mean(np.abs(predicted_edge_lengths - rest_lengths[None, :]), axis=1)
    edge_delta_mae = np.mean(np.abs(predicted_edge_lengths - baseline_edge_lengths), axis=1)

    max_displacement_per_frame = np.max(
        np.linalg.norm(predicted_positions - baseline_positions, axis=-1),
        axis=1,
    )
    mean_height_baseline = baseline_positions[:, :, 1].mean(axis=1)
    mean_height_prediction = predicted_positions[:, :, 1].mean(axis=1)

    return {
        "frame_count": int(baseline_positions.shape[0]),
        "vertex_count": int(baseline_positions.shape[1]),
        "position_rmse": {
            "mean": float(rmse_per_frame.mean()),
            "max": float(rmse_per_frame.max()),
            "final": float(rmse_per_frame[-1]),
            "per_frame": rmse_per_frame.tolist(),
        },
        "edge_length_rest_mae": {
            "baseline_mean": float(baseline_rest_mae.mean()),
            "prediction_mean": float(predicted_rest_mae.mean()),
            "prediction_final": float(predicted_rest_mae[-1]),
            "baseline_per_frame": baseline_rest_mae.tolist(),
            "prediction_per_frame": predicted_rest_mae.tolist(),
        },
        "edge_length_delta_mae": {
            "mean": float(edge_delta_mae.mean()),
            "max": float(edge_delta_mae.max()),
            "final": float(edge_delta_mae[-1]),
            "per_frame": edge_delta_mae.tolist(),
        },
        "max_vertex_displacement": {
            "mean": float(max_displacement_per_frame.mean()),
            "max": float(max_displacement_per_frame.max()),
            "final": float(max_displacement_per_frame[-1]),
            "per_frame": max_displacement_per_frame.tolist(),
        },
        "mean_height": {
            "baseline_per_frame": mean_height_baseline.tolist(),
            "prediction_per_frame": mean_height_prediction.tolist(),
        },
    }


@torch.no_grad()
def benchmark_fps(
    model: torch.nn.Module,
    dataset: ClothDataset,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> Dict[str, float]:
    """Measure inference speed for the recovered checkpoint."""
    fine_template, coarse_template, cluster_map = build_graph_structures(dataset, device)
    fine_data = dataset.get_frame(0, normalized=True).to(device)

    if coarse_template is not None and cluster_map is not None:
        coarse_pos = pool_positions(fine_data.pos, cluster_map, coarse_template.num_nodes)
        coarse_data = Data(
            x=torch.cat([coarse_pos, torch.zeros_like(coarse_pos)], dim=-1),
            edge_index=coarse_template.edge_index,
            edge_attr=coarse_template.edge_attr,
            pos=coarse_pos,
            num_nodes=coarse_template.num_nodes,
        ).to(device)
    else:
        coarse_data = None

    for _ in range(warmup_iters):
        _ = model(fine_data, coarse_data, cluster_map)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(timing_iters):
        _ = model(fine_data, coarse_data, cluster_map)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    fps = timing_iters / elapsed

    return {
        "fps": float(fps),
        "ms_per_frame": float(elapsed * 1000.0 / timing_iters),
        "num_iterations": int(timing_iters),
        "warmup_iterations": int(warmup_iters),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_mesh_info(
    data_path: Path,
    baseline_positions: np.ndarray,
    output_path: Path,
) -> None:
    """Save topology and sequence metadata needed by the later viewer."""
    with h5py.File(data_path, 'r') as f:
        mesh_size = int(f['mesh_size'][0]) if 'mesh_size' in f else int(round(np.sqrt(baseline_positions.shape[1])))
        edges = f['edges'][:].astype(np.int64)
        rest_lengths = f['rest_lengths'][:].astype(np.float32)
        fixed_mask = f['fixed_mask'][:].astype(bool) if 'fixed_mask' in f else np.zeros(baseline_positions.shape[1], dtype=bool)
        spacing = float(f['spacing'][0]) if 'spacing' in f else None
        mass = float(f['mass'][0]) if 'mass' in f else None
        stiffness = float(f['stiffness'][0]) if 'stiffness' in f else None
        damping = float(f['damping'][0]) if 'damping' in f else None
        gravity = float(f['gravity'][0]) if 'gravity' in f else None
        ground_height = float(f['ground_height'][0]) if 'ground_height' in f else GROUND_HEIGHT
        dt = float(f['dt'][0]) if 'dt' in f else None
        substeps = int(f['substeps'][0]) if 'substeps' in f else None

    _, faces, _ = create_grid_mesh(size=mesh_size, spacing=1.0, center=(0.0, 0.0, 0.0))
    timestep = float(dt * substeps) if dt is not None and substeps is not None else FRAME_DT

    mesh_info = {
        "mesh_size": mesh_size,
        "grid_dimensions": [mesh_size, mesh_size],
        "vertex_count": int(baseline_positions.shape[1]),
        "face_count": int(faces.shape[0]),
        "edge_count": int(edges.shape[0]),
        "frame_count": int(baseline_positions.shape[0]),
        "timestep": timestep,
        "ground_height": ground_height,
        "faces": faces.tolist(),
        "edges": edges.tolist(),
        "rest_lengths": rest_lengths.tolist(),
        "fixed_vertex_indices": np.flatnonzero(fixed_mask).astype(int).tolist(),
        "simulation": {
            "spacing": spacing,
            "mass": mass,
            "stiffness": stiffness,
            "damping": damping,
            "gravity": gravity,
            "dt": dt,
            "substeps": substeps,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mesh_info, indent=2))


def plot_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot rollout quality metrics across frames."""
    frames = np.arange(metrics["frame_count"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(frames, metrics["position_rmse"]["per_frame"], linewidth=2)
    axes[0, 0].set_title("Position RMSE")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(frames, metrics["edge_length_rest_mae"]["baseline_per_frame"], label="baseline", linewidth=2)
    axes[0, 1].plot(frames, metrics["edge_length_rest_mae"]["prediction_per_frame"], label="prediction", linewidth=2)
    axes[0, 1].set_title("Edge Length vs Rest Length MAE")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(frames, metrics["edge_length_delta_mae"]["per_frame"], linewidth=2)
    axes[1, 0].set_title("Prediction vs Baseline Edge Length MAE")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(frames, metrics["mean_height"]["baseline_per_frame"], label="baseline", linewidth=2)
    axes[1, 1].plot(frames, metrics["mean_height"]["prediction_per_frame"], label="prediction", linewidth=2)
    axes[1, 1].set_title("Mean Cloth Height")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Height")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_snapshots(
    baseline_positions: np.ndarray,
    predicted_positions: np.ndarray,
    output_path: Path,
) -> None:
    """Plot baseline and prediction snapshots for a few key frames."""
    frame_indices = [0, baseline_positions.shape[0] // 2, baseline_positions.shape[0] - 1]
    labels = ["start", "mid", "final"]

    fig = plt.figure(figsize=(15, 8))

    all_positions = np.concatenate([baseline_positions, predicted_positions], axis=0)
    x_min, x_max = float(all_positions[:, :, 0].min()), float(all_positions[:, :, 0].max())
    y_min, y_max = float(all_positions[:, :, 1].min()), float(all_positions[:, :, 1].max())
    z_min, z_max = float(all_positions[:, :, 2].min()), float(all_positions[:, :, 2].max())

    for row, sequence in enumerate([baseline_positions, predicted_positions]):
        for col, (frame_idx, label) in enumerate(zip(frame_indices, labels), start=1):
            ax = fig.add_subplot(2, 3, row * 3 + col, projection='3d')
            pos = sequence[frame_idx]
            ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=pos[:, 1], cmap='viridis', s=6)
            ax.set_title(f"{'Baseline' if row == 0 else 'HGNN'} {label} ({frame_idx})")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            ax.set_zlim(y_min, y_max)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")
            ax.view_init(elev=20, azim=-60)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cloth demo sequences and validation artifacts")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to cloth HDF5 data")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps-iters", type=int, default=100, help="Timed iterations for FPS benchmark")
    parser.add_argument("--fps-warmup", type=int, default=10, help="Warmup iterations for FPS benchmark")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "physics_baseline"
    prediction_dir = output_dir / "hgnn_prediction"
    validation_dir = output_dir / "validation"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ClothDataset(args.data, device=device)
    model = load_model_from_checkpoint(args.checkpoint, device).to(device)
    model.eval()

    baseline_positions = dataset.positions.cpu().numpy()
    predicted_positions = rollout_prediction(model, dataset, device)

    baseline_path = baseline_dir / "cloth_sequence.npy"
    prediction_path = prediction_dir / "cloth_sequence.npy"
    np.save(baseline_path, baseline_positions)
    np.save(prediction_path, predicted_positions)

    mesh_info_path = output_dir / "mesh_info.json"
    save_mesh_info(Path(args.data), baseline_positions, mesh_info_path)

    metrics = compute_metrics(
        baseline_positions,
        predicted_positions,
        dataset.edges.cpu().numpy(),
        dataset.rest_lengths.cpu().numpy(),
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["data"] = str(args.data)
    metrics["sequence_paths"] = {
        "baseline": str(baseline_path),
        "prediction": str(prediction_path),
    }

    fps_results = benchmark_fps(model, dataset, device, args.fps_warmup, args.fps_iters)
    metrics["fps_benchmark"] = fps_results

    metrics_path = validation_dir / "rollout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    fps_path = validation_dir / "fps_benchmark.json"
    fps_path.write_text(json.dumps(fps_results, indent=2))

    plot_metrics(metrics, validation_dir / "rollout_metrics_plot.png")
    plot_snapshots(baseline_positions, predicted_positions, validation_dir / "snapshot_comparison.png")

    print(json.dumps({
        "checkpoint": args.checkpoint,
        "baseline_sequence": str(baseline_path),
        "prediction_sequence": str(prediction_path),
        "mesh_info": str(mesh_info_path),
        "rollout_metrics": str(metrics_path),
        "fps_benchmark": str(fps_path),
    }, indent=2))


if __name__ == "__main__":
    main()
