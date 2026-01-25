#!/usr/bin/env python3
"""
Evaluation metrics for ClothGeom-NIF: Chamfer Distance + Surface Analysis.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best.pt --data data/cloth_dataset_500.h5 --num_samples 20
"""

import argparse
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def chamfer_distance_scipy(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Compute Chamfer distance between two point clouds using scipy."""
    from scipy.spatial import cKDTree

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    dist1, _ = tree2.query(pts1)
    dist2, _ = tree1.query(pts2)

    return float(np.mean(dist1**2) + np.mean(dist2**2))


def compute_surface_points_from_sdf(sdf_volume: np.ndarray,
                                    threshold: float = 0.05,
                                    max_points: int = 10000) -> np.ndarray:
    """Extract surface points from SDF volume (where |SDF| < threshold)."""
    resolution = sdf_volume.shape[0]

    # Create coordinate grid
    coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, resolution),
        np.linspace(-1, 1, resolution),
        np.linspace(-1, 1, resolution),
        indexing='ij'
    ), axis=-1)

    # Find surface points
    surface_mask = np.abs(sdf_volume) < threshold
    surface_pts = coords[surface_mask]

    # Subsample if too many points
    if len(surface_pts) > max_points:
        idx = np.random.choice(len(surface_pts), max_points, replace=False)
        surface_pts = surface_pts[idx]

    return surface_pts


def evaluate_sdf_directly(model, latent: torch.Tensor, gt_sdf: np.ndarray,
                          resolution: int = 64, device: torch.device = None) -> dict:
    """Evaluate by comparing predicted SDF to ground truth SDF directly."""
    if device is None:
        device = next(model.parameters()).device

    gt_resolution = gt_sdf.shape[0]

    # Create query grid at GT resolution
    coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, gt_resolution),
        np.linspace(-1, 1, gt_resolution),
        np.linspace(-1, 1, gt_resolution),
        indexing='ij'
    ), axis=-1).reshape(-1, 3)

    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    latent_expanded = latent.to(device).expand(len(coords), -1)

    # Predict SDF in batches
    batch_size = 65536
    pred_sdf = []

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords_tensor[i:i+batch_size]
            batch_latent = latent_expanded[i:i+batch_size]
            sdf_pred, _ = model(batch_coords, batch_latent)
            pred_sdf.append(sdf_pred.cpu().numpy())

    pred_sdf = np.concatenate(pred_sdf).reshape(gt_resolution, gt_resolution, gt_resolution)
    gt_sdf_flat = gt_sdf.flatten()
    pred_sdf_flat = pred_sdf.flatten()

    # Compute metrics
    mse = np.mean((pred_sdf_flat - gt_sdf_flat) ** 2)
    mae = np.mean(np.abs(pred_sdf_flat - gt_sdf_flat))

    # Compute surface-weighted error (more important near surface)
    surface_weight = np.exp(-10 * np.abs(gt_sdf_flat))
    weighted_mse = np.sum(surface_weight * (pred_sdf_flat - gt_sdf_flat) ** 2) / np.sum(surface_weight)

    # IoU-like metric (agreement on sign)
    sign_agreement = np.mean((pred_sdf_flat > 0) == (gt_sdf_flat > 0))

    return {
        'success': True,
        'mse': float(mse),
        'mae': float(mae),
        'weighted_mse': float(weighted_mse),
        'sign_agreement': float(sign_agreement),
    }


def evaluate_sample(model, extractor, latent: torch.Tensor,
                    gt_sdf: np.ndarray, resolution: int = 64) -> dict:
    """Evaluate a single sample with both mesh and SDF metrics."""
    device = next(model.parameters()).device

    # First, evaluate SDF directly
    sdf_result = evaluate_sdf_directly(model, latent, gt_sdf, resolution, device)

    if not sdf_result['success']:
        return sdf_result

    # Then extract mesh
    try:
        vertices, faces, normals = extractor.extract_mesh(
            latent.to(device), resolution=resolution
        )

        if vertices is None or len(vertices) == 0:
            sdf_result['mesh_success'] = False
            return sdf_result

        # Compute basic mesh statistics
        edge_lengths = []
        for face in faces[:1000]:  # Sample faces for efficiency
            for i in range(3):
                v1, v2 = vertices[face[i]], vertices[face[(i+1) % 3]]
                edge_lengths.append(np.linalg.norm(v2 - v1))

        sdf_result['mesh_success'] = True
        sdf_result['num_vertices'] = len(vertices)
        sdf_result['num_faces'] = len(faces)
        sdf_result['mean_edge_length'] = float(np.mean(edge_lengths))

        return sdf_result

    except Exception as e:
        sdf_result['mesh_success'] = False
        sdf_result['mesh_error'] = str(e)
        return sdf_result


def evaluate_model(checkpoint_path: str, data_path: str,
                   resolution: int = 64, num_samples: int = 20,
                   seed: int = 42):
    """Evaluate trained model on test samples."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Import model and extractor
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from models import NIFDecoder
    from inference.mesh_extractor import MeshExtractor

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get('model_config', {})
    model = NIFDecoder(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create mesh extractor
    extractor = MeshExtractor(model, device=device)

    # Load dataset
    logger.info(f"Loading dataset from {data_path}...")
    with h5py.File(data_path, 'r') as f:
        total_samples = len(f['latent'])
        num_samples = min(num_samples, total_samples)

        # Random sample indices (sorted for h5py access)
        indices = np.sort(np.random.choice(total_samples, num_samples, replace=False))

        latents = torch.tensor(f['latent'][indices.tolist()], dtype=torch.float32)
        gt_sdfs = np.array(f['sdf'][indices.tolist()])

    # Evaluate samples
    results = []
    mse_values = []
    mae_values = []
    weighted_mse_values = []
    sign_agreements = []
    vertex_counts = []

    logger.info(f"\nEvaluating {num_samples} samples at {resolution}³ resolution...")

    for i in tqdm(range(num_samples), desc="Evaluating"):
        latent = latents[i:i+1]
        gt_sdf = gt_sdfs[i]

        result = evaluate_sample(model, extractor, latent, gt_sdf, resolution)
        results.append(result)

        if result['success']:
            mse_values.append(result['mse'])
            mae_values.append(result['mae'])
            weighted_mse_values.append(result['weighted_mse'])
            sign_agreements.append(result['sign_agreement'])
            if result.get('mesh_success', False):
                vertex_counts.append(result['num_vertices'])
        else:
            logger.warning(f"  Sample {indices[i]}: {result.get('error', 'Unknown error')}")

    # Summary statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_path}")
    print(f"Resolution: {resolution}³")
    print(f"Samples evaluated: {len(mse_values)}/{num_samples}")
    print("-"*60)

    if mse_values:
        print(f"\nSDF RECONSTRUCTION QUALITY:")
        print(f"  MSE (mean):         {np.mean(mse_values):.6f}")
        print(f"  MSE (std):          {np.std(mse_values):.6f}")
        print(f"  MAE (mean):         {np.mean(mae_values):.6f}")
        print(f"  Weighted MSE:       {np.mean(weighted_mse_values):.6f}")
        print(f"  Sign Agreement:     {np.mean(sign_agreements):.2%}")

        # Quality assessment
        mean_mse = np.mean(mse_values)
        if mean_mse < 0.01:
            print(f"\n✅ MSE < 0.01: EXCELLENT reconstruction quality!")
        elif mean_mse < 0.05:
            print(f"\n✅ MSE < 0.05: GOOD reconstruction quality")
        else:
            print(f"\n⚠️  MSE = {mean_mse:.4f}: Room for improvement")

    if vertex_counts:
        print(f"\nMESH STATISTICS:")
        print(f"  Mean vertices: {np.mean(vertex_counts):.0f}")
        print(f"  Min vertices:  {np.min(vertex_counts):.0f}")
        print(f"  Max vertices:  {np.max(vertex_counts):.0f}")
        print(f"  Mesh success rate: {len(vertex_counts)}/{len(mse_values)}")

    print("="*60)

    # Return results for programmatic use
    return {
        'mse_values': mse_values,
        'mae_values': mae_values,
        'weighted_mse_values': weighted_mse_values,
        'sign_agreements': sign_agreements,
        'vertex_counts': vertex_counts,
        'results': results,
        'summary': {
            'mse_mean': np.mean(mse_values) if mse_values else None,
            'mae_mean': np.mean(mae_values) if mae_values else None,
            'sign_agreement_mean': np.mean(sign_agreements) if sign_agreements else None,
            'success_rate': len(mse_values) / num_samples
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ClothGeom-NIF model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--resolution', '-r', type=int, default=64,
                        help='Marching cubes resolution (default: 64)')
    parser.add_argument('--num_samples', '-n', type=int, default=20,
                        help='Number of samples to evaluate (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    evaluate_model(
        args.checkpoint,
        args.data,
        args.resolution,
        args.num_samples,
        args.seed
    )
