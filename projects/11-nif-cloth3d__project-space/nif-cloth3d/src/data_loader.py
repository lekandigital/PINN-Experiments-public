"""
Data loading utilities for NIF-Cloth3D-Interactive.
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any


class ClothSimulationDataset(Dataset):
    """
    Dataset for cloth simulation training data.
    
    Expected data format:
        data_dir/
            mesh_0000.npz  # Contains: vertices, rest_vertices, velocity, force, time, material_id
            mesh_0001.npz
            ...
    
    Each .npz file contains:
        - vertices: (N, 3) deformed vertex positions
        - rest_vertices: (N, 3) rest vertex positions  
        - velocity: (N, 3) vertex velocities
        - force: (3,) applied force vector
        - time: scalar time value
        - material_id: integer material identifier
        - edges: (E, 2) edge indices (optional)
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_dir: Path to directory containing .npz files
            transform: Optional transform to apply to samples
            max_samples: Maximum number of samples to load
            cache_in_memory: Whether to cache all data in memory
        """
        self.data_dir = data_dir
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        
        # Find all data files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Optionally cache in memory
        self.cache = {}
        if cache_in_memory:
            print(f"Caching {len(self.files)} samples in memory...")
            for i, f in enumerate(self.files):
                self.cache[i] = self._load_sample(f)
    
    def _load_sample(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load a single sample from disk."""
        data = np.load(filepath)
        
        sample = {
            'vertices': torch.tensor(data['vertices'], dtype=torch.float32),
            'rest_vertices': torch.tensor(data['rest_vertices'], dtype=torch.float32),
            'force': torch.tensor(data['force'], dtype=torch.float32),
            'time': torch.tensor(float(data['time']), dtype=torch.float32),
        }
        
        # Optional fields
        if 'velocity' in data:
            sample['velocity'] = torch.tensor(data['velocity'], dtype=torch.float32)
        if 'material_id' in data:
            sample['material_id'] = torch.tensor(int(data['material_id']), dtype=torch.long)
        if 'edges' in data:
            sample['edges'] = torch.tensor(data['edges'], dtype=torch.long)
        
        return sample
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory:
            sample = self.cache[idx]
        else:
            sample = self._load_sample(self.files[idx])
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class ClothSequenceDataset(Dataset):
    """
    Dataset for cloth simulation sequences (for temporal training).
    
    Expected data format:
        data_dir/
            sequence_0000/
                frame_0000.npz
                frame_0001.npz
                ...
            sequence_0001/
                ...
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 10,
        stride: int = 1,
        transform: Optional[callable] = None
    ):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Find all sequences
        self.sequences = []
        seq_dirs = sorted(glob.glob(os.path.join(data_dir, "sequence_*")))
        
        for seq_dir in seq_dirs:
            frames = sorted(glob.glob(os.path.join(seq_dir, "frame_*.npz")))
            # Create overlapping subsequences
            for start in range(0, len(frames) - sequence_length + 1, stride):
                self.sequences.append(frames[start:start + sequence_length])
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_files = self.sequences[idx]
        
        # Load all frames
        vertices_list = []
        rest_vertices = None
        forces = []
        times = []
        
        for f in frame_files:
            data = np.load(f)
            vertices_list.append(data['vertices'])
            if rest_vertices is None:
                rest_vertices = data['rest_vertices']
            forces.append(data['force'])
            times.append(data['time'])
        
        sample = {
            'vertices_seq': torch.tensor(np.stack(vertices_list), dtype=torch.float32),
            'rest_vertices': torch.tensor(rest_vertices, dtype=torch.float32),
            'forces': torch.tensor(np.stack(forces), dtype=torch.float32),
            'times': torch.tensor(times, dtype=torch.float32)
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class SyntheticClothDataset(Dataset):
    """
    Synthetic dataset for testing (no disk I/O required).
    Generates simple deformations based on analytical functions.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_vertices: int = 2500,
        mesh_size: float = 2.0,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.n_vertices = n_vertices
        self.mesh_size = mesh_size
        
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # Generate base mesh (grid)
        resolution = int(np.sqrt(n_vertices))
        x = np.linspace(-mesh_size/2, mesh_size/2, resolution)
        y = np.linspace(-mesh_size/2, mesh_size/2, resolution)
        xx, yy = np.meshgrid(x, y)
        self.rest_vertices = np.stack([
            xx.flatten(), 
            yy.flatten(), 
            np.zeros(resolution * resolution)
        ], axis=1).astype(np.float32)
        
        # Precompute edges
        edges = []
        for i in range(resolution):
            for j in range(resolution - 1):
                idx = i * resolution + j
                edges.append([idx, idx + 1])
        for i in range(resolution - 1):
            for j in range(resolution):
                idx = i * resolution + j
                edges.append([idx, idx + resolution])
        self.edges = np.array(edges)
    
    def _generate_deformation(
        self, 
        force: np.ndarray, 
        time: float
    ) -> np.ndarray:
        """Generate synthetic deformation based on force and time."""
        vertices = self.rest_vertices.copy()
        
        # Simple wave-based deformation
        force_magnitude = np.linalg.norm(force)
        force_direction = force / (force_magnitude + 1e-8)
        
        # Position-dependent displacement
        x, y = vertices[:, 0], vertices[:, 1]
        
        # Wave propagation effect
        phase = 2 * np.pi * (x * force_direction[0] + y * force_direction[1])
        wave = np.sin(phase + time * 5) * force_magnitude * 0.1
        
        # Apply displacement
        vertices[:, 0] += wave * force_direction[0] * 0.3
        vertices[:, 1] += wave * force_direction[1] * 0.3
        vertices[:, 2] += wave
        
        # Add gravity effect (droop)
        dist_from_top = (y - self.rest_vertices[:, 1].max())
        vertices[:, 2] += dist_from_top * 0.1 * (1 + time)
        
        return vertices
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random force and time
        force = self.rng.randn(3).astype(np.float32) * 5.0
        time = self.rng.rand() * 2.0
        material_id = self.rng.randint(0, 4)
        
        # Generate deformation
        vertices = self._generate_deformation(force, time)
        displacement = vertices - self.rest_vertices
        
        return {
            'vertices': torch.tensor(vertices, dtype=torch.float32),
            'rest_vertices': torch.tensor(self.rest_vertices, dtype=torch.float32),
            'displacement': torch.tensor(displacement, dtype=torch.float32),
            'force': torch.tensor(force, dtype=torch.float32),
            'time': torch.tensor(time, dtype=torch.float32),
            'material_id': torch.tensor(material_id, dtype=torch.long),
            'edges': torch.tensor(self.edges, dtype=torch.long)
        }


def collate_cloth_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for cloth simulation data.
    Handles variable-size meshes by padding.
    """
    # Find max vertices in batch
    max_verts = max(b['vertices'].shape[0] for b in batch)
    batch_size = len(batch)
    
    # Prepare output tensors
    result = {
        'vertices': torch.zeros(batch_size, max_verts, 3),
        'rest_vertices': torch.zeros(batch_size, max_verts, 3),
        'force': torch.stack([b['force'] for b in batch]),
        'time': torch.stack([b['time'] for b in batch]),
        'mask': torch.zeros(batch_size, max_verts, dtype=torch.bool)
    }
    
    # Fill in data
    for i, b in enumerate(batch):
        n_verts = b['vertices'].shape[0]
        result['vertices'][i, :n_verts] = b['vertices']
        result['rest_vertices'][i, :n_verts] = b['rest_vertices']
        result['mask'][i, :n_verts] = True
    
    # Optional fields
    if 'material_id' in batch[0]:
        result['material_id'] = torch.stack([b['material_id'] for b in batch])
    if 'displacement' in batch[0]:
        result['displacement'] = torch.zeros(batch_size, max_verts, 3)
        for i, b in enumerate(batch):
            n_verts = b['displacement'].shape[0]
            result['displacement'][i, :n_verts] = b['displacement']
    
    return result


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_in_memory: bool = False,
    synthetic: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for cloth simulation data.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        cache_in_memory: Cache data in memory
        synthetic: Use synthetic data for testing
        **kwargs: Additional arguments for Dataset
    
    Returns:
        DataLoader instance
    """
    if synthetic:
        dataset = SyntheticClothDataset(**kwargs)
    else:
        dataset = ClothSimulationDataset(
            data_dir, 
            cache_in_memory=cache_in_memory,
            **kwargs
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_cloth_batch
    )


if __name__ == "__main__":
    # Test synthetic dataset
    print("Testing SyntheticClothDataset...")
    dataset = SyntheticClothDataset(n_samples=100, n_vertices=2500)
    sample = dataset[0]
    
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Vertices shape: {sample['vertices'].shape}")
    print(f"Force: {sample['force']}")
    print(f"Time: {sample['time']}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    loader = create_dataloader(
        data_dir=".",  # Not used for synthetic
        batch_size=8,
        num_workers=0,
        synthetic=True,
        n_samples=100,
        n_vertices=2500
    )
    
    batch = next(iter(loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch vertices shape: {batch['vertices'].shape}")
    print(f"Batch force shape: {batch['force'].shape}")
    
    print("\nAll data loading tests passed!")
