"""
H5ClothDataset: PyTorch Dataset for HDF5 Cloth Data

Loads cloth simulation data from HDF5 files and provides
batched access for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class H5ClothDataset(Dataset):
    """
    PyTorch Dataset for cloth simulation data stored in HDF5.
    
    Provides:
    - Fine and coarse mesh node positions
    - Edge indices for both resolutions
    - SDF volumes as ground truth
    - Optional data augmentation
    
    Args:
        data_path: Path to HDF5 file
        mode: 'train', 'val', or 'test' (determines data split)
        train_ratio: Fraction of data for training
        transform: Optional transform function
        load_to_memory: Whether to load entire dataset to RAM
        
    Example:
        >>> dataset = H5ClothDataset('data/cloth_data.h5')
        >>> sample = dataset[0]
        >>> print(sample['fine_pos'].shape)  # (N_fine, 3)
    """
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transform: Optional[callable] = None,
        load_to_memory: bool = True,
        num_query_points: int = 1000
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.mode = mode
        self.transform = transform
        self.load_to_memory = load_to_memory
        self.num_query_points = num_query_points
        
        assert self.data_path.exists(), f"Data file not found: {data_path}"
        
        # Load data
        with h5py.File(self.data_path, 'r') as f:
            self.num_total = f['fine_positions'].shape[0]
            
            # Compute split indices
            num_train = int(self.num_total * train_ratio)
            num_val = int(self.num_total * val_ratio)
            
            if mode == 'train':
                self.indices = list(range(0, num_train))
            elif mode == 'val':
                self.indices = list(range(num_train, num_train + num_val))
            else:  # test
                self.indices = list(range(num_train + num_val, self.num_total))
                
            # Load edges (same for all samples)
            self.fine_edges = torch.from_numpy(f['fine_edges'][:]).long()
            self.coarse_edges = torch.from_numpy(f['coarse_edges'][:]).long()
            
            # Load metadata
            self.metadata = dict(f.attrs) if hasattr(f, 'attrs') else {}
            
            # Optionally load all data to memory
            if load_to_memory:
                self.fine_positions = torch.from_numpy(f['fine_positions'][:]).float()
                self.coarse_positions = torch.from_numpy(f['coarse_positions'][:]).float()
                self.sdf_volumes = torch.from_numpy(f['sdf_volumes'][:]).float()
            else:
                self.fine_positions = None
                self.coarse_positions = None
                self.sdf_volumes = None
                
        # Compute SDF grid bounds (for query point generation)
        if load_to_memory:
            self._compute_bounds()
            
    def _compute_bounds(self):
        """Compute bounding box for SDF query points."""
        all_pos = self.fine_positions.reshape(-1, 3)
        self.min_bound = all_pos.min(dim=0).values - 0.2
        self.max_bound = all_pos.max(dim=0).values + 0.2
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns dict with:
            - fine_pos: (N_fine, 3) fine mesh positions
            - fine_edges: (2, E_fine) edge indices
            - coarse_pos: (N_coarse, 3) coarse mesh positions  
            - coarse_edges: (2, E_coarse) edge indices
            - sdf_volume: (R, R, R) SDF ground truth
            - query_points: (M, 3) random query points
            - query_sdf: (M,) SDF values at query points
        """
        data_idx = self.indices[idx]
        
        if self.load_to_memory:
            fine_pos = self.fine_positions[data_idx]
            coarse_pos = self.coarse_positions[data_idx]
            sdf_volume = self.sdf_volumes[data_idx]
        else:
            with h5py.File(self.data_path, 'r') as f:
                fine_pos = torch.from_numpy(f['fine_positions'][data_idx]).float()
                coarse_pos = torch.from_numpy(f['coarse_positions'][data_idx]).float()
                sdf_volume = torch.from_numpy(f['sdf_volumes'][data_idx]).float()
                
        # Generate random query points
        query_points, query_sdf = self._sample_query_points(
            fine_pos, sdf_volume, self.num_query_points
        )
        
        sample = {
            'fine_pos': fine_pos,
            'fine_edges': self.fine_edges.T,  # (2, E)
            'coarse_pos': coarse_pos,
            'coarse_edges': self.coarse_edges.T,  # (2, E)
            'sdf_volume': sdf_volume,
            'query_points': query_points,
            'query_sdf': query_sdf,
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def _sample_query_points(
        self,
        fine_pos: torch.Tensor,
        sdf_volume: torch.Tensor,
        num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random query points and interpolate SDF values.
        
        Samples more points near the surface (SDF ≈ 0) for better training.
        """
        R = sdf_volume.shape[0]
        
        # Compute bounds for this sample
        min_bound = fine_pos.min(dim=0).values - 0.2
        max_bound = fine_pos.max(dim=0).values + 0.2
        
        # Sample uniformly in bounding box
        num_uniform = num_points // 2
        uniform_points = torch.rand(num_uniform, 3) * (max_bound - min_bound) + min_bound
        
        # Sample near surface (perturb mesh vertices)
        num_surface = num_points - num_uniform
        surface_indices = torch.randint(0, len(fine_pos), (num_surface,))
        surface_points = fine_pos[surface_indices] + torch.randn(num_surface, 3) * 0.05
        
        query_points = torch.cat([uniform_points, surface_points], dim=0)
        
        # Interpolate SDF values from volume
        # Normalize coordinates to [-1, 1] for grid_sample
        normalized = 2 * (query_points - min_bound) / (max_bound - min_bound) - 1
        normalized = normalized.clamp(-1, 1)
        
        # grid_sample expects (N, D, H, W) input and (N, H_out, W_out, 3) grid
        sdf_volume_5d = sdf_volume.unsqueeze(0).unsqueeze(0)  # (1, 1, R, R, R)
        grid = normalized.view(1, 1, 1, num_points, 3)  # (1, 1, 1, M, 3)
        
        # Note: grid_sample uses (x, y, z) but our grid is (z, y, x) order
        # We need to flip the order
        grid_flipped = torch.stack([grid[..., 2], grid[..., 1], grid[..., 0]], dim=-1)
        
        query_sdf = torch.nn.functional.grid_sample(
            sdf_volume_5d, grid_flipped,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        query_sdf = query_sdf.view(num_points)
        
        return query_points, query_sdf
    
    def get_rest_lengths(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rest lengths of edges from first frame.
        
        Returns:
            fine_rest: (E_fine,) rest lengths for fine edges
            coarse_rest: (E_coarse,) rest lengths for coarse edges
        """
        first_fine = self.fine_positions[0]  # (N_fine, 3)
        first_coarse = self.coarse_positions[0]  # (N_coarse, 3)
        
        # Fine edges
        fine_edges = self.fine_edges  # (E, 2)
        fine_src = first_fine[fine_edges[:, 0]]
        fine_tgt = first_fine[fine_edges[:, 1]]
        fine_rest = torch.norm(fine_src - fine_tgt, dim=-1)
        
        # Coarse edges
        coarse_edges = self.coarse_edges  # (E, 2)
        coarse_src = first_coarse[coarse_edges[:, 0]]
        coarse_tgt = first_coarse[coarse_edges[:, 1]]
        coarse_rest = torch.norm(coarse_src - coarse_tgt, dim=-1)
        
        return fine_rest, coarse_rest


def collate_cloth_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for cloth data.
    
    Handles variable-sized edge lists by keeping them as-is
    (same edges for all samples in batch).
    """
    # Stack tensors that have consistent shape
    fine_pos = torch.stack([b['fine_pos'] for b in batch], dim=0)
    coarse_pos = torch.stack([b['coarse_pos'] for b in batch], dim=0)
    sdf_volume = torch.stack([b['sdf_volume'] for b in batch], dim=0)
    query_points = torch.stack([b['query_points'] for b in batch], dim=0)
    query_sdf = torch.stack([b['query_sdf'] for b in batch], dim=0)
    
    # Edges are shared across batch
    fine_edges = batch[0]['fine_edges']
    coarse_edges = batch[0]['coarse_edges']
    
    return {
        'fine_pos': fine_pos,
        'fine_edges': fine_edges,
        'coarse_pos': coarse_pos,
        'coarse_edges': coarse_edges,
        'sdf_volume': sdf_volume,
        'query_points': query_points,
        'query_sdf': query_sdf,
    }


def create_dataloader(
    data_path: str,
    mode: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for cloth data.
    
    Args:
        data_path: Path to HDF5 file
        mode: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for H5ClothDataset
        
    Returns:
        DataLoader instance
    """
    dataset = H5ClothDataset(data_path, mode=mode, **kwargs)
    
    shuffle = (mode == 'train')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cloth_batch,
        pin_memory=True
    )
    
    return loader
