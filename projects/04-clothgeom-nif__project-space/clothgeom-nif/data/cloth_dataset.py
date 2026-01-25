"""
PyTorch Dataset for ClothGeom-NIF Training

Provides efficient loading and sampling of SDF training data from HDF5 files.
Supports:
- Random point sampling from SDF volumes
- Importance sampling near surface (SDF ≈ 0)
- Data augmentation (rotation, scaling)
- Batched coordinate-latent-SDF triplets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, List
import h5py


class ClothSDFDataset(Dataset):
    """
    Dataset for training NIF decoder on cloth SDF data.
    
    Each sample provides:
    - coords: 3D query coordinates [num_points, 3]
    - latent: Latent cloth state [latent_dim]
    - sdf: Ground truth SDF values [num_points, 1]
    
    Args:
        data_path: Path to HDF5 dataset file
        num_points_per_sample: Points to sample per cloth state
        surface_ratio: Fraction of points sampled near surface (0-1)
        surface_std: Std dev for near-surface sampling (in grid units)
        augment: Whether to apply data augmentation
        cache_in_memory: Load entire dataset into memory (faster, more RAM)
    """
    
    def __init__(
        self,
        data_path: str,
        num_points_per_sample: int = 10000,
        surface_ratio: float = 0.5,
        surface_std: float = 0.05,
        augment: bool = True,
        cache_in_memory: bool = True
    ):
        self.data_path = data_path
        self.num_points = num_points_per_sample
        self.surface_ratio = surface_ratio
        self.surface_std = surface_std
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        
        # Load metadata and optionally cache data
        self._load_data()
    
    def _load_data(self):
        """Load dataset from HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            # Read metadata
            self.num_samples = f.attrs['num_samples']
            self.sdf_resolution = f.attrs['sdf_resolution']
            self.latent_dim = f.attrs['latent_dim']
            self.sdf_bounds = f.attrs.get('sdf_bounds', 1.5)
            
            if self.cache_in_memory:
                # Load all data into memory
                self.latents = torch.from_numpy(f['latent'][:])
                self.sdfs = torch.from_numpy(f['sdf'][:])
                self.nodes = torch.from_numpy(f['nodes'][:]) if 'nodes' in f else None
            else:
                # Store file handle for lazy loading
                self.latents = None
                self.sdfs = None
                self.nodes = None
        
        # Pre-compute grid coordinates
        self._setup_grid()
    
    def _setup_grid(self):
        """Pre-compute normalized grid coordinates."""
        r = self.sdf_resolution
        b = self.sdf_bounds
        
        # Create normalized coordinates in [-1, 1]
        coords_1d = torch.linspace(-1, 1, r)
        xx, yy, zz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        
        # Flatten grid coordinates
        self.grid_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Coordinate scale factor (for converting grid indices to world coords)
        self.coord_scale = b
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample with sampled points.
        
        Returns:
            dict with keys:
                - 'coords': [num_points, 3] query coordinates
                - 'latent': [latent_dim] latent code
                - 'sdf': [num_points, 1] ground truth SDF values
        """
        # Get latent and SDF volume
        if self.cache_in_memory:
            latent = self.latents[idx]
            sdf_volume = self.sdfs[idx]
        else:
            with h5py.File(self.data_path, 'r') as f:
                latent = torch.from_numpy(f['latent'][idx])
                sdf_volume = torch.from_numpy(f['sdf'][idx])
        
        # Sample points
        coords, sdf_values = self._sample_points(sdf_volume)
        
        # Apply augmentation
        if self.augment:
            coords, sdf_values = self._augment(coords, sdf_values)
        
        return {
            'coords': coords,  # [N, 3]
            'latent': latent,  # [D]
            'sdf': sdf_values  # [N, 1]
        }
    
    def _sample_points(
        self,
        sdf_volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points with importance sampling near surface.
        
        Args:
            sdf_volume: [R, R, R] SDF volume
        
        Returns:
            coords: [N, 3] sampled coordinates
            sdf: [N, 1] SDF values at sampled points
        """
        r = self.sdf_resolution
        n_surface = int(self.num_points * self.surface_ratio)
        n_uniform = self.num_points - n_surface
        
        coords_list = []
        sdf_list = []
        
        # 1. Uniform random sampling
        if n_uniform > 0:
            # Random indices in volume
            indices = torch.randint(0, r**3, (n_uniform,))
            
            # Convert to coordinates
            iz = indices // (r * r)
            iy = (indices % (r * r)) // r
            ix = indices % r
            
            # Normalized coordinates
            coords_uniform = torch.stack([
                2 * ix.float() / (r - 1) - 1,
                2 * iy.float() / (r - 1) - 1,
                2 * iz.float() / (r - 1) - 1
            ], dim=1)
            
            # Get SDF values
            sdf_uniform = sdf_volume[iz, iy, ix].unsqueeze(1)
            
            coords_list.append(coords_uniform)
            sdf_list.append(sdf_uniform)
        
        # 2. Near-surface sampling (importance sampling)
        if n_surface > 0:
            # Find surface voxels (where SDF is close to 0)
            sdf_flat = sdf_volume.flatten()
            
            # Weight by inverse distance to surface
            weights = 1.0 / (torch.abs(sdf_flat) + 0.01)
            weights = weights / weights.sum()
            
            # Sample indices weighted by proximity to surface
            surface_indices = torch.multinomial(weights, n_surface, replacement=True)
            
            # Convert to 3D indices
            iz = surface_indices // (r * r)
            iy = (surface_indices % (r * r)) // r
            ix = surface_indices % r
            
            # Coordinates with small random offset
            coords_surface = torch.stack([
                2 * ix.float() / (r - 1) - 1,
                2 * iy.float() / (r - 1) - 1,
                2 * iz.float() / (r - 1) - 1
            ], dim=1)
            
            # Add small noise for continuous coordinates
            coords_surface += self.surface_std * torch.randn_like(coords_surface)
            coords_surface = torch.clamp(coords_surface, -1, 1)
            
            # Interpolate SDF values at offset positions
            # For simplicity, use nearest neighbor (could use trilinear)
            ix_new = ((coords_surface[:, 0] + 1) / 2 * (r - 1)).long().clamp(0, r-1)
            iy_new = ((coords_surface[:, 1] + 1) / 2 * (r - 1)).long().clamp(0, r-1)
            iz_new = ((coords_surface[:, 2] + 1) / 2 * (r - 1)).long().clamp(0, r-1)
            
            sdf_surface = sdf_volume[iz_new, iy_new, ix_new].unsqueeze(1)
            
            coords_list.append(coords_surface)
            sdf_list.append(sdf_surface)
        
        # Concatenate all samples
        coords = torch.cat(coords_list, dim=0)
        sdf = torch.cat(sdf_list, dim=0)
        
        # Shuffle
        perm = torch.randperm(coords.shape[0])
        coords = coords[perm]
        sdf = sdf[perm]
        
        return coords, sdf
    
    def _augment(
        self,
        coords: torch.Tensor,
        sdf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation.
        
        Augmentations:
        - Random rotation around Z axis
        - Random scale (0.9-1.1)
        - Random flip in X or Y
        """
        if torch.rand(1) < 0.5:
            # Random rotation around Z axis
            angle = torch.rand(1) * 2 * np.pi
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            
            rot_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ]).squeeze()
            
            coords = coords @ rot_matrix.T
        
        if torch.rand(1) < 0.3:
            # Random scale (SDF values scale accordingly)
            scale = 0.9 + 0.2 * torch.rand(1)
            coords = coords * scale
            sdf = sdf * scale
        
        if torch.rand(1) < 0.3:
            # Random flip in X
            coords[:, 0] = -coords[:, 0]
        
        if torch.rand(1) < 0.3:
            # Random flip in Y
            coords[:, 1] = -coords[:, 1]
        
        return coords, sdf
    
    def get_full_volume(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get complete SDF volume for a sample (for visualization/evaluation).
        
        Returns:
            dict with:
                - 'coords': [R*R*R, 3] all grid coordinates
                - 'latent': [latent_dim] latent code
                - 'sdf': [R, R, R] full SDF volume
        """
        if self.cache_in_memory:
            latent = self.latents[idx]
            sdf = self.sdfs[idx]
        else:
            with h5py.File(self.data_path, 'r') as f:
                latent = torch.from_numpy(f['latent'][idx])
                sdf = torch.from_numpy(f['sdf'][idx])
        
        return {
            'coords': self.grid_coords,
            'latent': latent,
            'sdf': sdf
        }


class MultiResolutionClothDataset(ClothSDFDataset):
    """
    Dataset with multi-resolution SDF sampling for LOD training.
    
    Samples points at multiple resolutions to train the variance
    field for LOD blending.
    """
    
    def __init__(
        self,
        data_path: str,
        resolutions: List[int] = [32, 64, 128],
        **kwargs
    ):
        self.resolutions = resolutions
        super().__init__(data_path, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get multi-resolution samples."""
        sample = super().__getitem__(idx)
        
        # Add resolution indicator for each point
        # This can be used to weight variance loss
        res_idx = torch.randint(0, len(self.resolutions), (sample['coords'].shape[0],))
        sample['resolution_idx'] = res_idx
        
        return sample


def create_dataloader(
    data_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    num_points: int = 10000,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        data_path: Path to HDF5 dataset
        batch_size: Number of cloth states per batch
        num_workers: Dataloader workers
        shuffle: Shuffle data
        num_points: Points per cloth state
        **dataset_kwargs: Additional dataset arguments
    
    Returns:
        PyTorch DataLoader
    """
    dataset = ClothSDFDataset(
        data_path,
        num_points_per_sample=num_points,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def collate_flatten(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that flattens point samples across batch.
    
    Instead of [B, N, 3] -> [B*N, 3] for efficient processing.
    """
    coords = torch.cat([b['coords'] for b in batch], dim=0)
    sdf = torch.cat([b['sdf'] for b in batch], dim=0)
    
    # Expand latent to match each point
    latents = []
    for b in batch:
        n_points = b['coords'].shape[0]
        latents.append(b['latent'].unsqueeze(0).expand(n_points, -1))
    latent = torch.cat(latents, dim=0)
    
    return {
        'coords': coords,
        'latent': latent,
        'sdf': sdf
    }


if __name__ == '__main__':
    # Test dataset
    import os
    
    test_path = 'data/generated/test_dataset.h5'
    
    if os.path.exists(test_path):
        dataset = ClothSDFDataset(test_path, num_points_per_sample=1000)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Sample shapes:")
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")
        
        # Test dataloader
        loader = create_dataloader(test_path, batch_size=4, num_points=500)
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"Test dataset not found at {test_path}")
        print("Generate with: python generate_dataset.py --num_samples 10 --output data/generated/test_dataset.h5")
