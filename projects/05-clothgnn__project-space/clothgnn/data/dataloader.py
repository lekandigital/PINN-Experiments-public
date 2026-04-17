"""
Data Loading Utilities for ClothGNN Training.

Provides PyTorch Dataset and DataLoader for cloth dynamics sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class ClothSequenceDataset(Dataset):
    """
    Dataset for cloth dynamics sequences.
    
    Each sample is a window of consecutive frames from a simulation.
    """
    
    def __init__(
        self,
        h5_path: str,
        window_size: int = 10,
        stride: int = 1,
        split: Optional[str] = None,
        splits_path: Optional[str] = None,
        transform=None,
    ):
        """
        Args:
            h5_path: Path to HDF5 dataset
            window_size: Number of frames per training sample
            stride: Stride between windows
            split: 'train', 'val', or 'test' (optional)
            splits_path: Path to splits.json file
            transform: Optional transform to apply
        """
        self.h5_path = h5_path
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Determine which sequences to use
        with h5py.File(h5_path, 'r') as f:
            total_sequences = f.attrs['n_sequences']
            self.n_steps = f.attrs['n_steps']
        
        # Load split indices if provided
        if split is not None and splits_path is not None:
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            self.sequence_indices = splits[split]
        else:
            self.sequence_indices = list(range(total_sequences))
        
        # Index all valid windows
        self.windows = []
        with h5py.File(h5_path, 'r') as f:
            for seq_idx in self.sequence_indices:
                seq_grp = f['sequences'][str(seq_idx)]
                n_frames = seq_grp['positions'].shape[0]
                n_windows = (n_frames - window_size) // stride + 1
                
                for win_idx in range(n_windows):
                    start = win_idx * stride
                    self.windows.append((seq_idx, start))
        
        print(f"Dataset: {len(self.windows)} windows from {len(self.sequence_indices)} sequences")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, start = self.windows[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            seq = f['sequences'][str(seq_idx)]
            
            positions = torch.from_numpy(
                seq['positions'][start:start + self.window_size]
            ).float()
            velocities = torch.from_numpy(
                seq['velocities'][start:start + self.window_size]
            ).float()
            edge_index = torch.from_numpy(seq['edge_index'][:]).long()
            rest_lengths = torch.from_numpy(seq['rest_lengths'][:]).float()
            fixed_mask = torch.from_numpy(seq['fixed_mask'][:]).bool()
        
        sample = {
            'positions': positions,
            'velocities': velocities,
            'edge_index': edge_index,
            'rest_lengths': rest_lengths,
            'fixed_mask': fixed_mask,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class SingleStepDataset(Dataset):
    """
    Dataset providing single-step position prediction samples.
    
    Input: (position_t, velocity_t)
    Target: position_{t+1}
    """
    
    def __init__(
        self,
        h5_path: str,
        split: Optional[str] = None,
        splits_path: Optional[str] = None,
        transform=None,
    ):
        self.h5_path = h5_path
        self.transform = transform
        
        # Load split indices
        with h5py.File(h5_path, 'r') as f:
            total_sequences = f.attrs['n_sequences']
        
        if split is not None and splits_path is not None:
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            self.sequence_indices = splits[split]
        else:
            self.sequence_indices = list(range(total_sequences))
        
        # Index all valid frame pairs
        self.samples = []
        with h5py.File(h5_path, 'r') as f:
            for seq_idx in self.sequence_indices:
                seq_grp = f['sequences'][str(seq_idx)]
                n_frames = seq_grp['positions'].shape[0]
                
                for frame_idx in range(n_frames - 1):
                    self.samples.append((seq_idx, frame_idx))
        
        print(f"SingleStepDataset: {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, frame_idx = self.samples[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            seq = f['sequences'][str(seq_idx)]
            
            pos_t = torch.from_numpy(seq['positions'][frame_idx]).float()
            vel_t = torch.from_numpy(seq['velocities'][frame_idx]).float()
            pos_next = torch.from_numpy(seq['positions'][frame_idx + 1]).float()
            
            edge_index = torch.from_numpy(seq['edge_index'][:]).long()
            rest_lengths = torch.from_numpy(seq['rest_lengths'][:]).float()
            fixed_mask = torch.from_numpy(seq['fixed_mask'][:]).bool()
        
        # Target is displacement
        displacement = pos_next - pos_t
        
        sample = {
            'positions': pos_t,
            'velocities': vel_t,
            'edge_index': edge_index,
            'rest_lengths': rest_lengths,
            'fixed_mask': fixed_mask,
            'target_displacement': displacement,
            'target_positions': pos_next,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_cloth_batch(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate cloth samples into a batch.
    
    Since meshes may have different sizes, we handle this by:
    1. Padding to max size (for batch processing)
    2. Providing masks for valid nodes
    """
    batch_size = len(samples)
    
    # Find max sizes
    max_nodes = max(s['positions'].shape[-2] for s in samples)
    max_edges = max(s['edge_index'].shape[1] for s in samples)
    
    # Check if this is a sequence or single-step dataset
    is_sequence = samples[0]['positions'].dim() == 3
    
    if is_sequence:
        T = samples[0]['positions'].shape[0]
        
        # Initialize padded tensors
        positions = torch.zeros(batch_size, T, max_nodes, 3)
        velocities = torch.zeros(batch_size, T, max_nodes, 3)
    else:
        positions = torch.zeros(batch_size, max_nodes, 3)
        velocities = torch.zeros(batch_size, max_nodes, 3)
        target_displacement = torch.zeros(batch_size, max_nodes, 3)
        target_positions = torch.zeros(batch_size, max_nodes, 3)
    
    edge_index = torch.zeros(batch_size, 2, max_edges, dtype=torch.long)
    rest_lengths = torch.zeros(batch_size, max_edges)
    fixed_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    edge_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    
    node_counts = []
    edge_counts = []
    
    for i, sample in enumerate(samples):
        n_nodes = sample['positions'].shape[-2]
        n_edges = sample['edge_index'].shape[1]
        
        node_counts.append(n_nodes)
        edge_counts.append(n_edges)
        
        if is_sequence:
            positions[i, :, :n_nodes] = sample['positions']
            velocities[i, :, :n_nodes] = sample['velocities']
        else:
            positions[i, :n_nodes] = sample['positions']
            velocities[i, :n_nodes] = sample['velocities']
            target_displacement[i, :n_nodes] = sample['target_displacement']
            target_positions[i, :n_nodes] = sample['target_positions']
        
        edge_index[i, :, :n_edges] = sample['edge_index']
        rest_lengths[i, :n_edges] = sample['rest_lengths']
        fixed_mask[i, :n_nodes] = sample['fixed_mask']
        node_mask[i, :n_nodes] = True
        edge_mask[i, :n_edges] = True
    
    batch = {
        'positions': positions,
        'velocities': velocities,
        'edge_index': edge_index,
        'rest_lengths': rest_lengths,
        'fixed_mask': fixed_mask,
        'node_mask': node_mask,
        'edge_mask': edge_mask,
        'node_counts': torch.tensor(node_counts),
        'edge_counts': torch.tensor(edge_counts),
    }
    
    if not is_sequence:
        batch['target_displacement'] = target_displacement
        batch['target_positions'] = target_positions
    
    return batch


def create_dataloader(
    h5_path: str,
    batch_size: int = 8,
    window_size: int = 10,
    shuffle: bool = True,
    num_workers: int = 4,
    split: Optional[str] = None,
    splits_path: Optional[str] = None,
    single_step: bool = False,
) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        h5_path: Path to HDF5 dataset
        batch_size: Batch size
        window_size: Frames per sample (ignored if single_step=True)
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        split: 'train', 'val', or 'test'
        splits_path: Path to splits.json
        single_step: Use single-step dataset instead of sequences
    """
    if single_step:
        dataset = SingleStepDataset(
            h5_path,
            split=split,
            splits_path=splits_path,
        )
    else:
        dataset = ClothSequenceDataset(
            h5_path,
            window_size=window_size,
            split=split,
            splits_path=splits_path,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cloth_batch,
        pin_memory=True,
    )


# Data augmentation transforms
class RandomRotation:
    """Random rotation around Y axis."""
    
    def __init__(self, max_angle: float = 180.0):
        self.max_angle = max_angle
    
    def __call__(self, sample: Dict) -> Dict:
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        angle_rad = np.radians(angle)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=torch.float32)
        
        sample['positions'] = sample['positions'] @ rotation.T
        sample['velocities'] = sample['velocities'] @ rotation.T
        
        if 'target_displacement' in sample:
            sample['target_displacement'] = sample['target_displacement'] @ rotation.T
        if 'target_positions' in sample:
            sample['target_positions'] = sample['target_positions'] @ rotation.T
        
        return sample


class RandomScale:
    """Random uniform scaling."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, sample: Dict) -> Dict:
        scale = np.random.uniform(*self.scale_range)
        
        sample['positions'] = sample['positions'] * scale
        sample['velocities'] = sample['velocities'] * scale
        sample['rest_lengths'] = sample['rest_lengths'] * scale
        
        if 'target_displacement' in sample:
            sample['target_displacement'] = sample['target_displacement'] * scale
        if 'target_positions' in sample:
            sample['target_positions'] = sample['target_positions'] * scale
        
        return sample


class AddNoise:
    """Add small Gaussian noise to positions."""
    
    def __init__(self, std: float = 0.001):
        self.std = std
    
    def __call__(self, sample: Dict) -> Dict:
        noise = torch.randn_like(sample['positions']) * self.std
        sample['positions'] = sample['positions'] + noise
        return sample


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample
