#!/usr/bin/env python3
"""
Inverse Design for ClothGeom-NIF.

Given a target SDF volume, optimize a latent code to reproduce that geometry.
This enables finding latent codes that correspond to desired cloth configurations.

Usage:
    python inverse_design.py --checkpoint outputs/checkpoints/best.pt \
        --target_idx 10 --data data/cloth_dataset_500.h5 --steps 500
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from pathlib import Path
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class InverseDesignOptimizer:
    """Optimize latent codes to match target SDF volumes."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def create_query_points(self, resolution: int, num_points: int = 10000,
                            surface_ratio: float = 0.5,
                            target_sdf: np.ndarray = None) -> torch.Tensor:
        """Create query points for SDF evaluation.

        Args:
            resolution: Grid resolution
            num_points: Total number of query points
            surface_ratio: Fraction of points sampled near surface
            target_sdf: Ground truth SDF for importance sampling
        """
        if target_sdf is not None and surface_ratio > 0:
            # Importance sampling near surface
            num_surface = int(num_points * surface_ratio)
            num_uniform = num_points - num_surface

            # Find near-surface voxels
            surface_mask = np.abs(target_sdf) < 0.1
            surface_indices = np.argwhere(surface_mask)

            if len(surface_indices) > num_surface:
                idx = np.random.choice(len(surface_indices), num_surface, replace=False)
                surface_indices = surface_indices[idx]

            # Convert to coordinates in [-1, 1]
            coords = []

            # Surface points with jitter
            for voxel_idx in surface_indices:
                coord = (voxel_idx / (resolution - 1)) * 2 - 1
                coord += np.random.normal(0, 0.02, 3)  # Small jitter
                coord = np.clip(coord, -1, 1)
                coords.append(coord)

            # Uniform points
            uniform_pts = np.random.uniform(-1, 1, (num_uniform, 3))
            coords.extend(uniform_pts.tolist())

            return torch.tensor(np.array(coords), dtype=torch.float32, device=self.device)
        else:
            # Pure uniform sampling
            return torch.rand(num_points, 3, device=self.device) * 2 - 1

    def sample_target_sdf(self, target_sdf: np.ndarray,
                          coords: torch.Tensor) -> torch.Tensor:
        """Sample SDF values at given coordinates using trilinear interpolation."""
        resolution = target_sdf.shape[0]

        # Convert coordinates from [-1, 1] to [0, resolution-1]
        grid_coords = (coords + 1) / 2 * (resolution - 1)
        grid_coords = grid_coords.clamp(0, resolution - 1)

        # Use grid_sample for trilinear interpolation
        target_tensor = torch.tensor(target_sdf, dtype=torch.float32, device=self.device)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

        # grid_sample expects coordinates in [-1, 1] and shape [1, D_out, H_out, W_out, 3]
        sample_coords = coords.view(1, 1, 1, -1, 3)  # [1, 1, 1, N, 3]

        sampled = F.grid_sample(
            target_tensor,
            sample_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return sampled.view(-1)  # [N]

    def optimize(self, target_sdf: np.ndarray,
                 init_latent: torch.Tensor = None,
                 num_steps: int = 500,
                 lr: float = 0.01,
                 num_points: int = 8192,
                 surface_ratio: float = 0.5,
                 log_every: int = 50) -> tuple:
        """Optimize latent code to match target SDF.

        Args:
            target_sdf: Target SDF volume [D, H, W]
            init_latent: Initial latent code (random if None)
            num_steps: Number of optimization steps
            lr: Learning rate
            num_points: Number of query points per iteration
            surface_ratio: Fraction of points near surface
            log_every: Log progress every N steps

        Returns:
            optimized_latent: Final optimized latent code
            losses: List of loss values during optimization
        """
        latent_dim = self.model.latent_dim
        resolution = target_sdf.shape[0]

        # Initialize latent code
        if init_latent is None:
            latent = torch.randn(1, latent_dim, device=self.device) * 0.1
        else:
            latent = init_latent.clone().to(self.device)

        latent.requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=lr * 0.01)

        losses = []
        best_loss = float('inf')
        best_latent = latent.clone()

        logger.info(f"Starting inverse design optimization ({num_steps} steps)...")

        for step in tqdm(range(num_steps), desc="Optimizing"):
            optimizer.zero_grad()

            # Sample query points (resample each step for diversity)
            coords = self.create_query_points(
                resolution, num_points, surface_ratio, target_sdf
            )

            # Get target SDF values at query points
            target_values = self.sample_target_sdf(target_sdf, coords)

            # Predict SDF values
            latent_expanded = latent.expand(len(coords), -1)
            pred_sdf, pred_var = self.model(coords, latent_expanded)
            pred_sdf = pred_sdf.squeeze()

            # Compute loss
            # Main SDF loss
            sdf_loss = F.mse_loss(pred_sdf, target_values)

            # Surface weighting (emphasize near-surface accuracy)
            surface_weight = torch.exp(-10 * torch.abs(target_values))
            weighted_loss = (surface_weight * (pred_sdf - target_values) ** 2).mean()

            # Latent regularization (prefer smaller latent codes)
            latent_reg = 0.001 * (latent ** 2).mean()

            # Total loss
            loss = sdf_loss + 0.5 * weighted_loss + latent_reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_latent = latent.clone().detach()

            if (step + 1) % log_every == 0:
                logger.info(f"  Step {step+1}/{num_steps}: Loss = {loss.item():.6f}, "
                            f"SDF = {sdf_loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        logger.info(f"Optimization complete. Best loss: {best_loss:.6f}")

        return best_latent, losses


def run_inverse_design(checkpoint_path: str, data_path: str,
                       target_idx: int = 0, num_steps: int = 500,
                       output_dir: str = 'outputs/inverse_design'):
    """Run inverse design experiment."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Import model and extractor
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

    # Load target SDF
    logger.info(f"Loading target SDF (index {target_idx}) from {data_path}...")
    with h5py.File(data_path, 'r') as f:
        target_sdf = np.array(f['sdf'][target_idx])
        original_latent = torch.tensor(f['latent'][target_idx], dtype=torch.float32).unsqueeze(0)

    logger.info(f"Target SDF shape: {target_sdf.shape}")

    # Create optimizer
    optimizer = InverseDesignOptimizer(model, device)

    # Run optimization
    optimized_latent, losses = optimizer.optimize(
        target_sdf,
        init_latent=None,  # Start from random
        num_steps=num_steps,
        lr=0.02,
        num_points=8192,
        surface_ratio=0.6
    )

    # Save results
    logger.info("Saving results...")

    # Save optimized latent
    torch.save({
        'optimized_latent': optimized_latent,
        'original_latent': original_latent,
        'losses': losses,
        'target_idx': target_idx,
    }, output_dir / f'inverse_design_{target_idx}.pt')

    # Save loss curve
    np.save(output_dir / f'losses_{target_idx}.npy', np.array(losses))

    # Extract mesh from optimized latent
    logger.info("Extracting mesh from optimized latent...")
    extractor = MeshExtractor(model, device=device)
    vertices, faces, normals = extractor.extract_mesh(optimized_latent, resolution=64)

    if vertices is not None:
        # Save mesh using trimesh
        import trimesh
        mesh_path = output_dir / f'optimized_mesh_{target_idx}.obj'
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.export(str(mesh_path))
        logger.info(f"Saved optimized mesh: {mesh_path} ({len(vertices)} vertices)")

    # Also extract mesh from original latent for comparison
    vertices_orig, faces_orig, normals_orig = extractor.extract_mesh(original_latent.to(device), resolution=64)
    if vertices_orig is not None:
        mesh_path_orig = output_dir / f'original_mesh_{target_idx}.obj'
        mesh_orig = trimesh.Trimesh(vertices=vertices_orig, faces=faces_orig, vertex_normals=normals_orig)
        mesh_orig.export(str(mesh_path_orig))
        logger.info(f"Saved original mesh: {mesh_path_orig} ({len(vertices_orig)} vertices)")

    # Print summary
    print("\n" + "="*60)
    print("INVERSE DESIGN RESULTS")
    print("="*60)
    print(f"Target index: {target_idx}")
    print(f"Optimization steps: {num_steps}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")
    print(f"Loss reduction: {(losses[0] - min(losses)) / losses[0] * 100:.1f}%")

    # Compute latent similarity
    cosine_sim = F.cosine_similarity(optimized_latent, original_latent.to(device)).item()
    l2_dist = torch.norm(optimized_latent - original_latent.to(device)).item()
    print(f"\nLatent comparison:")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  L2 distance: {l2_dist:.4f}")
    print(f"\nOutputs saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inverse design for ClothGeom-NIF')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--target_idx', '-t', type=int, default=0,
                        help='Index of target sample in dataset')
    parser.add_argument('--steps', '-s', type=int, default=500,
                        help='Number of optimization steps')
    parser.add_argument('--output', '-o', type=str, default='outputs/inverse_design',
                        help='Output directory')

    args = parser.parse_args()

    run_inverse_design(
        args.checkpoint,
        args.data,
        args.target_idx,
        args.steps,
        args.output
    )
