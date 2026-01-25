"""
NIF-Cloth4D: Evaluation Script

This script evaluates a trained NIF-Cloth4D model by:
1. Loading the trained model
2. Extracting meshes at specified timestamps using marching cubes
3. Computing geometric metrics (Chamfer distance, Hausdorff distance)
4. Generating visualization plots

Usage:
    python test_evaluation.py --checkpoint ./checkpoints/model_best.pt --data_dir /tmp/cloth_test_data
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to import mcubes (PyMCubes)
try:
    import mcubes
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False
    print("Warning: mcubes not installed. Install with: pip install PyMCubes")

# Local imports
from nif_cloth4d import FourierFeatureSIREN, create_model
from synthetic_data import load_sdf_from_hdf5


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[FourierFeatureSIREN, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device
    
    Returns:
        model: Loaded model
        config: Training configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.6f}")
    
    return model, config


def query_sdf_grid(
    model: FourierFeatureSIREN,
    frame_time: float,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: int = 64,
    device: torch.device = torch.device('cpu'),
    chunk_size: int = 65536
) -> np.ndarray:
    """
    Query the model's SDF on a 3D grid.
    
    Args:
        model: Trained SDF model
        frame_time: Time value to query
        grid_min: Minimum coordinates (3,)
        grid_max: Maximum coordinates (3,)
        resolution: Grid resolution per axis
        device: Torch device
        chunk_size: Batch size for queries (for memory efficiency)
    
    Returns:
        sdf_grid: SDF values as (resolution, resolution, resolution) array
    """
    model.eval()
    
    # Create coordinate grid
    xs = np.linspace(grid_min[0], grid_max[0], resolution)
    ys = np.linspace(grid_min[1], grid_max[1], resolution)
    zs = np.linspace(grid_min[2], grid_max[2], resolution)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    times = np.full((len(coords), 1), frame_time)
    coords_4d = np.hstack([coords, times]).astype(np.float32)
    
    # Query in chunks
    sdf_values = []
    with torch.no_grad():
        for i in range(0, len(coords_4d), chunk_size):
            batch = torch.from_numpy(coords_4d[i:i+chunk_size]).to(device)
            sdf = model(batch).cpu().numpy()
            sdf_values.append(sdf)
    
    sdf_values = np.concatenate(sdf_values, axis=0)
    sdf_grid = sdf_values.reshape((resolution, resolution, resolution))
    
    return sdf_grid


def extract_mesh(
    sdf_grid: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    iso_level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from SDF using marching cubes.
    
    Args:
        sdf_grid: SDF volume
        grid_min: Minimum coordinates
        grid_max: Maximum coordinates
        iso_level: ISO surface level (0 for SDF)
    
    Returns:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
    """
    if not HAS_MCUBES:
        raise ImportError("mcubes required for mesh extraction. Install with: pip install PyMCubes")
    
    # Run marching cubes
    vertices, faces = mcubes.marching_cubes(sdf_grid, iso_level)
    
    # Scale vertices to world coordinates
    resolution = sdf_grid.shape[0]
    scale = (grid_max - grid_min) / (resolution - 1)
    vertices = vertices * scale + grid_min
    
    return vertices, faces


def chamfer_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        P: First point cloud (N, 3)
        Q: Second point cloud (M, 3)
    
    Returns:
        Chamfer distance (average bidirectional)
    """
    # P to Q
    d_PQ = 0.0
    for p in P:
        dists = np.sum((p - Q) ** 2, axis=1)
        d_PQ += np.min(dists)
    d_PQ /= len(P)
    
    # Q to P
    d_QP = 0.0
    for q in Q:
        dists = np.sum((q - P) ** 2, axis=1)
        d_QP += np.min(dists)
    d_QP /= len(Q)
    
    return (d_PQ + d_QP) / 2


def chamfer_distance_fast(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Fast Chamfer distance using vectorized operations.
    
    Args:
        P: First point cloud (N, 3)
        Q: Second point cloud (M, 3)
    
    Returns:
        Chamfer distance
    """
    # Subsample if too large
    max_points = 5000
    if len(P) > max_points:
        idx = np.random.choice(len(P), max_points, replace=False)
        P = P[idx]
    if len(Q) > max_points:
        idx = np.random.choice(len(Q), max_points, replace=False)
        Q = Q[idx]
    
    # Compute pairwise distances
    # P: (N, 3), Q: (M, 3)
    # dist: (N, M)
    diff = P[:, np.newaxis, :] - Q[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    
    d_PQ = np.mean(np.min(dists, axis=1))
    d_QP = np.mean(np.min(dists, axis=0))
    
    return (d_PQ + d_QP) / 2


def hausdorff_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two point clouds.
    
    Args:
        P: First point cloud (N, 3)
        Q: Second point cloud (M, 3)
    
    Returns:
        Hausdorff distance (max deviation)
    """
    # Subsample if too large
    max_points = 5000
    if len(P) > max_points:
        idx = np.random.choice(len(P), max_points, replace=False)
        P = P[idx]
    if len(Q) > max_points:
        idx = np.random.choice(len(Q), max_points, replace=False)
        Q = Q[idx]
    
    # Compute pairwise distances
    diff = P[:, np.newaxis, :] - Q[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    d_PQ = np.max(np.min(dists, axis=1))
    d_QP = np.max(np.min(dists, axis=0))
    
    return max(d_PQ, d_QP)


def compute_metrics(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred_vertices: Predicted mesh vertices
        gt_vertices: Ground truth mesh vertices
    
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Chamfer distance
    metrics['chamfer'] = chamfer_distance_fast(pred_vertices, gt_vertices)
    
    # Hausdorff distance
    metrics['hausdorff'] = hausdorff_distance(pred_vertices, gt_vertices)
    
    return metrics


def visualize_comparison(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    metrics: Dict[str, float],
    time: float,
    save_path: Optional[str] = None
):
    """
    Create visualization comparing predicted and ground truth.
    
    Args:
        pred_vertices: Predicted mesh vertices
        gt_vertices: Ground truth mesh vertices
        metrics: Computed metrics
        time: Timestamp
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Subsample for visualization
    max_points = 2000
    if len(pred_vertices) > max_points:
        idx = np.random.choice(len(pred_vertices), max_points, replace=False)
        pred_vis = pred_vertices[idx]
    else:
        pred_vis = pred_vertices
    
    if len(gt_vertices) > max_points:
        idx = np.random.choice(len(gt_vertices), max_points, replace=False)
        gt_vis = gt_vertices[idx]
    else:
        gt_vis = gt_vertices
    
    # Plot 1: Prediction
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pred_vis[:, 0], pred_vis[:, 1], pred_vis[:, 2], 
                s=1, c='red', alpha=0.5, label='Predicted')
    ax1.set_title(f'Predicted (t={time:.2f})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Comparison
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(gt_vis[:, 0], gt_vis[:, 1], gt_vis[:, 2], 
                s=1, c='blue', alpha=0.5, label='Ground Truth')
    ax2.scatter(pred_vis[:, 0], pred_vis[:, 1], pred_vis[:, 2], 
                s=1, c='red', alpha=0.5, label='Predicted')
    ax2.set_title(f'Comparison\nChamfer: {metrics["chamfer"]:.4f}, Hausdorff: {metrics["hausdorff"]:.4f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_loss_curves(history_path: str, save_path: Optional[str] = None):
    """
    Plot training loss curves.
    
    Args:
        history_path: Path to training_history.npz
        save_path: Optional path to save figure
    """
    data = np.load(history_path)
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    
    if len(val_losses) > 0:
        # Val losses are recorded less frequently
        val_epochs = np.linspace(1, len(train_losses), len(val_losses))
        ax.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('NIF-Cloth4D Training Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to: {save_path}")
    
    plt.show()


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    resolution: int = 64,
    times: list = [0.0, 0.5, 1.0]
):
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to ground truth data
        output_dir: Output directory for results
        resolution: Marching cubes resolution
        times: List of timestamps to evaluate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Define grid bounds (same as training data)
    grid_min = np.array([-1.0, -1.0, -1.0])
    grid_max = np.array([1.0, 1.0, 1.0])
    
    # Evaluate at each timestamp
    all_metrics = {}
    
    for t in times:
        print(f"\n{'='*50}")
        print(f"Evaluating at t = {t:.2f}")
        print('='*50)
        
        # Query model SDF
        print("Querying model SDF...")
        sdf_pred = query_sdf_grid(
            model, t, grid_min, grid_max, resolution, device
        )
        print(f"Predicted SDF range: [{sdf_pred.min():.4f}, {sdf_pred.max():.4f}]")
        
        # Extract predicted mesh
        if HAS_MCUBES:
            print("Extracting mesh with marching cubes...")
            try:
                pred_verts, pred_faces = extract_mesh(sdf_pred, grid_min, grid_max)
                print(f"Predicted mesh: {len(pred_verts)} vertices, {len(pred_faces)} faces")
            except Exception as e:
                print(f"Mesh extraction failed: {e}")
                pred_verts = None
        else:
            pred_verts = None
        
        # Load ground truth
        sdf_files = sorted(Path(data_dir).glob('sdf/*.h5'))
        if len(sdf_files) > 0:
            # Find closest frame to time t
            frame_idx = int(t * (len(sdf_files) - 1))
            gt_sdf, coords, attrs = load_sdf_from_hdf5(str(sdf_files[frame_idx]))
            print(f"Ground truth SDF range: [{gt_sdf.min():.4f}, {gt_sdf.max():.4f}]")
            
            # Extract GT mesh for comparison
            if HAS_MCUBES and pred_verts is not None:
                try:
                    gt_verts, gt_faces = extract_mesh(gt_sdf, grid_min, grid_max)
                    print(f"Ground truth mesh: {len(gt_verts)} vertices, {len(gt_faces)} faces")
                    
                    # Compute metrics
                    metrics = compute_metrics(pred_verts, gt_verts)
                    all_metrics[t] = metrics
                    
                    print(f"\nMetrics at t={t:.2f}:")
                    print(f"  Chamfer Distance: {metrics['chamfer']:.6f}")
                    print(f"  Hausdorff Distance: {metrics['hausdorff']:.6f}")
                    
                    # Visualize
                    vis_path = output_path / f'comparison_t{t:.2f}.png'
                    visualize_comparison(pred_verts, gt_verts, metrics, t, str(vis_path))
                    
                except Exception as e:
                    print(f"Ground truth mesh extraction failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if len(all_metrics) > 0:
        avg_chamfer = np.mean([m['chamfer'] for m in all_metrics.values()])
        avg_hausdorff = np.mean([m['hausdorff'] for m in all_metrics.values()])
        
        print(f"Average Chamfer Distance: {avg_chamfer:.6f}")
        print(f"Average Hausdorff Distance: {avg_hausdorff:.6f}")
        
        # Save metrics
        metrics_path = output_path / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("NIF-Cloth4D Evaluation Metrics\n")
            f.write("="*40 + "\n\n")
            for t, m in all_metrics.items():
                f.write(f"t = {t:.2f}:\n")
                f.write(f"  Chamfer: {m['chamfer']:.6f}\n")
                f.write(f"  Hausdorff: {m['hausdorff']:.6f}\n\n")
            f.write(f"Average Chamfer: {avg_chamfer:.6f}\n")
            f.write(f"Average Hausdorff: {avg_hausdorff:.6f}\n")
        
        print(f"\nMetrics saved to: {metrics_path}")
    
    # Plot loss curves if available
    history_path = Path(checkpoint_path).parent / 'training_history.npz'
    if history_path.exists():
        loss_plot_path = output_path / 'loss_curves.png'
        visualize_loss_curves(str(history_path), str(loss_plot_path))
    
    print("\nEvaluation complete!")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate NIF-Cloth4D')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/tmp/cloth_test_data',
                        help='Path to ground truth data')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory for results')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Marching cubes resolution')
    parser.add_argument('--times', type=str, default='0.0,0.5,1.0',
                        help='Comma-separated list of times to evaluate')
    args = parser.parse_args()
    
    times = [float(t) for t in args.times.split(',')]
    
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        times=times
    )


if __name__ == "__main__":
    main()
