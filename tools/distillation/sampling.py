"""
Input sampling strategies for knowledge distillation.

Different domains require different approaches to generating input data
for distillation training. This module provides a common interface and
implementations for various sampling strategies.

Strategies:
- RandomSampling: Sample random points from a defined domain
- DatasetSampling: Replay samples from an existing dataset
- GridQuerySampling: Sample query points for implicit field models (SIREN)
- TeacherGuidedSampling: Focus on regions where teacher behavior is complex
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class SamplingStrategy(ABC):
    """
    Abstract base class for input sampling strategies.
    
    Subclasses must implement `sample()` and `sample_epoch()`.
    """
    
    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
    
    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Generate a single batch of inputs.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Dictionary mapping input names to tensors
        """
        pass
    
    @abstractmethod
    def sample_epoch(
        self, 
        batch_size: int, 
        num_samples: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Generate batches for a full epoch.
        
        Args:
            batch_size: Number of samples per batch
            num_samples: Total samples in the epoch
            
        Yields:
            Dictionaries mapping input names to tensors
        """
        pass
    
    def to(self, device: str | torch.device) -> "SamplingStrategy":
        """Move strategy to a device."""
        self.device = torch.device(device)
        return self


class RandomSampling(SamplingStrategy):
    """
    Sample random points uniformly from a defined domain.
    
    Suitable for simple PINN distillation where the input is coordinate-based
    (e.g., airfoil flow with (x, y, angle_of_attack)).
    
    Args:
        spatial_range: Spatial coordinate ranges, either [min, max] for all dims
                      or [[min1, max1], [min2, max2], ...] per dimension
        parameter_range: Additional parameter ranges (e.g., {"aoa": [-5, 15]})
        input_dim: Total input dimension (inferred if spatial_range is nested)
    """
    
    def __init__(
        self,
        spatial_range: list[float] | list[list[float]] = (-1.0, 1.0),
        parameter_range: dict[str, list[float]] | None = None,
        input_dim: int | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device)
        
        # Normalize spatial_range to list of [min, max] pairs
        if isinstance(spatial_range[0], (int, float)):
            # Single range for all dimensions
            if input_dim is None:
                input_dim = 2  # Default to 2D
            self.spatial_ranges = [spatial_range] * input_dim
        else:
            self.spatial_ranges = spatial_range
            input_dim = len(spatial_range)
        
        self.parameter_range = parameter_range or {}
        self.input_dim = input_dim + len(self.parameter_range)
        
        # Pre-compute ranges as tensors
        self._setup_ranges()
    
    def _setup_ranges(self):
        """Convert ranges to tensors for efficient sampling."""
        ranges = []
        for r in self.spatial_ranges:
            ranges.append(r)
        for name, r in self.parameter_range.items():
            ranges.append(r)
        
        self.range_min = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
        self.range_max = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
        self.range_span = self.range_max - self.range_min
    
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample random points uniformly from the domain."""
        # Uniform random in [0, 1], then scale to ranges
        samples = torch.rand(batch_size, self.input_dim, device=self.device)
        samples = samples * self.range_span.to(self.device) + self.range_min.to(self.device)
        
        return {'input': samples}
    
    def sample_epoch(
        self, 
        batch_size: int, 
        num_samples: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Generate batches for an epoch."""
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            yield self.sample(batch_size)


class GridQuerySampling(SamplingStrategy):
    """
    Sample query points for implicit field distillation (SIREN models).
    
    Used when distilling a graph-based teacher (e.g., HGNN-NIF-Cloth) into
    a coordinate-based student (e.g., NIF-Cloth4D SIREN).
    
    The strategy generates random (x, y, z, t) query points, and the teacher
    is queried at these points to produce ground truth SDF values.
    
    Args:
        spatial_range: [min, max] for spatial coordinates (x, y, z)
        temporal_range: [min, max] for time coordinate
        num_query_points: Number of query points per sample
        include_surface_points: Whether to include points on the surface
    """
    
    def __init__(
        self,
        spatial_range: list[float] = (-1.0, 1.0),
        temporal_range: list[float] = (0.0, 1.0),
        num_query_points: int = 4096,
        include_surface_points: bool = True,
        surface_point_ratio: float = 0.3,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device)
        
        self.spatial_range = spatial_range
        self.temporal_range = temporal_range
        self.num_query_points = num_query_points
        self.include_surface_points = include_surface_points
        self.surface_point_ratio = surface_point_ratio
    
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample query coordinates.
        
        Returns batch_size sets of query points, each with num_query_points points.
        """
        total_points = batch_size * self.num_query_points
        
        # Spatial coordinates (x, y, z)
        spatial = torch.rand(total_points, 3, device=self.device)
        spatial = spatial * (self.spatial_range[1] - self.spatial_range[0])
        spatial = spatial + self.spatial_range[0]
        
        # Temporal coordinate
        temporal = torch.rand(total_points, 1, device=self.device)
        temporal = temporal * (self.temporal_range[1] - self.temporal_range[0])
        temporal = temporal + self.temporal_range[0]
        
        # Concatenate to (x, y, z, t)
        coords = torch.cat([spatial, temporal], dim=-1)
        
        # Reshape to [batch_size, num_query_points, 4]
        coords = coords.view(batch_size, self.num_query_points, 4)
        
        return {
            'coords': coords,
            'input': coords.view(-1, 4),  # Flattened for model input
        }
    
    def sample_epoch(
        self, 
        batch_size: int, 
        num_samples: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Generate batches for an epoch."""
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            yield self.sample(batch_size)


class DatasetSampling(SamplingStrategy):
    """
    Replay samples from an existing dataset.
    
    Used when you have a dataset of (input, output) pairs that the teacher
    was trained on, and you want to distill using the same distribution.
    
    Supports augmentation (noise, jitter) to expand coverage.
    
    Args:
        dataset_path: Path to the dataset (HDF5, numpy, or PyTorch format)
        input_key: Key for input data in the dataset
        augmentation: Augmentation config {"noise_std": 0.001, "temporal_jitter": 0.05}
    """
    
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        dataset: Dataset | None = None,
        input_key: str = "input",
        output_key: str = "output",
        augmentation: dict[str, float] | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device)
        
        self.input_key = input_key
        self.output_key = output_key
        self.augmentation = augmentation or {}
        
        if dataset is not None:
            self.dataset = dataset
        elif dataset_path is not None:
            self.dataset = self._load_dataset(Path(dataset_path))
        else:
            raise ValueError("Must provide either dataset or dataset_path")
        
        self._dataloader: DataLoader | None = None
        self._dataloader_iter: Iterator | None = None
    
    def _load_dataset(self, path: Path) -> Dataset:
        """Load dataset from various formats."""
        if path.suffix == '.h5' or path.suffix == '.hdf5':
            return HDF5Dataset(path, self.input_key, self.output_key)
        elif path.suffix == '.npz':
            return NumpyDataset(path, self.input_key, self.output_key)
        elif path.suffix == '.pt':
            data = torch.load(path)
            return TensorDataset(data[self.input_key], data.get(self.output_key))
        elif path.is_dir():
            # Assume it's a directory of numpy files or similar
            return DirectoryDataset(path, self.input_key, self.output_key)
        else:
            raise ValueError(f"Unknown dataset format: {path.suffix}")
    
    def _augment(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply augmentation to a batch."""
        if not self.augmentation:
            return batch
        
        result = {}
        for key, value in batch.items():
            if key == self.input_key or key == 'input':
                # Add noise
                if 'noise_std' in self.augmentation:
                    noise = torch.randn_like(value) * self.augmentation['noise_std']
                    value = value + noise
                
                # Temporal jitter (for time-dependent models)
                if 'temporal_jitter' in self.augmentation and value.shape[-1] >= 4:
                    jitter = torch.randn(value.shape[0], 1, device=value.device)
                    jitter = jitter * self.augmentation['temporal_jitter']
                    value[:, -1:] = value[:, -1:] + jitter
            
            result[key] = value
        
        return result
    
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch from the dataset."""
        if self._dataloader is None or self._dataloader.batch_size != batch_size:
            self._dataloader = DataLoader(
                self.dataset, 
                batch_size=batch_size, 
                shuffle=True,
                drop_last=True
            )
            self._dataloader_iter = iter(self._dataloader)
        
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            batch = next(self._dataloader_iter)
        
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Apply augmentation
        batch = self._augment(batch)
        
        return batch
    
    def sample_epoch(
        self, 
        batch_size: int, 
        num_samples: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Generate batches for an epoch."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        samples_yielded = 0
        while samples_yielded < num_samples:
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch = self._augment(batch)
                yield batch
                samples_yielded += batch['input'].shape[0]
                if samples_yielded >= num_samples:
                    break


class TeacherGuidedSampling(SamplingStrategy):
    """
    Focus sampling on regions where the teacher's behavior is complex.
    
    Uses the teacher model to identify "interesting" input regions:
    - High output variance
    - Large output gradient norms
    - Near decision boundaries
    
    Then oversamples from these regions during distillation.
    
    Args:
        base_sampler: Underlying sampling strategy
        teacher: The teacher model (used read-only)
        complexity_metric: How to measure complexity
        oversample_ratio: How much to oversample complex regions
        cache_size: Number of samples to precompute complexity for
    """
    
    def __init__(
        self,
        base_sampler: SamplingStrategy,
        teacher: torch.nn.Module | None = None,
        complexity_metric: str = "output_gradient_norm",
        oversample_ratio: float = 3.0,
        cache_size: int = 10000,
        complexity_threshold_percentile: float = 75.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device)
        
        self.base_sampler = base_sampler
        self.teacher = teacher
        self.complexity_metric = complexity_metric
        self.oversample_ratio = oversample_ratio
        self.cache_size = cache_size
        self.complexity_threshold_percentile = complexity_threshold_percentile
        
        # Cache of complex samples
        self._complex_samples: torch.Tensor | None = None
        self._cache_idx = 0
    
    def set_teacher(self, teacher: torch.nn.Module):
        """Set the teacher model."""
        self.teacher = teacher
        self._complex_samples = None  # Invalidate cache
    
    def _compute_complexity(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute complexity scores for a batch of inputs."""
        if self.teacher is None:
            return torch.ones(inputs.shape[0], device=inputs.device)
        
        inputs = inputs.clone().requires_grad_(True)
        
        with torch.enable_grad():
            outputs = self.teacher(inputs)
            
            if self.complexity_metric == "output_gradient_norm":
                # Compute gradient of output w.r.t. input
                grad_outputs = torch.ones_like(outputs)
                grads = torch.autograd.grad(
                    outputs, inputs, grad_outputs,
                    create_graph=False, retain_graph=False
                )[0]
                complexity = torch.norm(grads, dim=-1)
            
            elif self.complexity_metric == "output_variance":
                # Use output magnitude as proxy for complexity
                complexity = torch.var(outputs, dim=-1)
            
            elif self.complexity_metric == "output_magnitude":
                complexity = torch.norm(outputs, dim=-1)
            
            else:
                complexity = torch.ones(inputs.shape[0], device=inputs.device)
        
        return complexity
    
    def _build_cache(self):
        """Build cache of complex samples."""
        if self.teacher is None:
            logger.warning("Teacher not set, using uniform sampling")
            return
        
        self.teacher.eval()
        
        # Sample many points and compute complexity
        all_samples = []
        all_complexities = []
        
        with torch.no_grad():
            remaining = self.cache_size * 3  # Oversample, then filter
            batch_size = min(1000, remaining)
            
            while remaining > 0:
                batch = self.base_sampler.sample(batch_size)
                inputs = batch['input'].to(self.device)
                
                complexity = self._compute_complexity(inputs)
                
                all_samples.append(inputs.cpu())
                all_complexities.append(complexity.cpu())
                
                remaining -= batch_size
        
        all_samples = torch.cat(all_samples, dim=0)
        all_complexities = torch.cat(all_complexities, dim=0)
        
        # Keep samples above complexity threshold
        threshold = torch.quantile(all_complexities, self.complexity_threshold_percentile / 100)
        mask = all_complexities >= threshold
        
        self._complex_samples = all_samples[mask][:self.cache_size].to(self.device)
        self._cache_idx = 0
        
        logger.info(f"Built complexity cache with {len(self._complex_samples)} samples")
    
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample with bias towards complex regions."""
        # Build cache if needed
        if self._complex_samples is None and self.teacher is not None:
            self._build_cache()
        
        # Determine how many samples from complex cache vs base
        if self._complex_samples is not None and len(self._complex_samples) > 0:
            num_complex = int(batch_size * (self.oversample_ratio / (1 + self.oversample_ratio)))
            num_base = batch_size - num_complex
        else:
            num_complex = 0
            num_base = batch_size
        
        samples = []
        
        # Sample from complex cache
        if num_complex > 0:
            # Wrap around cache if needed
            if self._cache_idx + num_complex > len(self._complex_samples):
                self._cache_idx = 0
            
            complex_batch = self._complex_samples[self._cache_idx:self._cache_idx + num_complex]
            self._cache_idx += num_complex
            samples.append(complex_batch)
        
        # Sample from base distribution
        if num_base > 0:
            base_batch = self.base_sampler.sample(num_base)
            samples.append(base_batch['input'].to(self.device))
        
        # Combine and shuffle
        combined = torch.cat(samples, dim=0)
        perm = torch.randperm(combined.shape[0], device=self.device)
        combined = combined[perm]
        
        return {'input': combined}
    
    def sample_epoch(
        self, 
        batch_size: int, 
        num_samples: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Generate batches for an epoch."""
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            yield self.sample(batch_size)


# =============================================================================
# Dataset Implementations
# =============================================================================

class TensorDataset(Dataset):
    """Simple dataset from tensors."""
    
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor | None = None):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        result = {'input': self.inputs[idx]}
        if self.outputs is not None:
            result['output'] = self.outputs[idx]
        return result


class HDF5Dataset(Dataset):
    """Dataset from HDF5 file."""
    
    def __init__(self, path: Path, input_key: str, output_key: str):
        import h5py
        self.file = h5py.File(path, 'r')
        self.input_key = input_key
        self.output_key = output_key
        
        # Find the data (handle nested groups)
        self.inputs = self._find_data(self.file, input_key)
        self.outputs = self._find_data(self.file, output_key)
    
    def _find_data(self, group, key: str):
        """Recursively find data in HDF5 groups."""
        if key in group:
            return group[key]
        for name, item in group.items():
            if isinstance(item, h5py.Group):
                result = self._find_data(item, key)
                if result is not None:
                    return result
        return None
    
    def __len__(self) -> int:
        return len(self.inputs) if self.inputs is not None else 0
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        result = {'input': torch.tensor(self.inputs[idx], dtype=torch.float32)}
        if self.outputs is not None:
            result['output'] = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return result


class NumpyDataset(Dataset):
    """Dataset from numpy .npz file."""
    
    def __init__(self, path: Path, input_key: str, output_key: str):
        data = np.load(path)
        self.inputs = torch.tensor(data[input_key], dtype=torch.float32)
        self.outputs = torch.tensor(data[output_key], dtype=torch.float32) if output_key in data else None
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        result = {'input': self.inputs[idx]}
        if self.outputs is not None:
            result['output'] = self.outputs[idx]
        return result


class DirectoryDataset(Dataset):
    """Dataset from directory of files."""
    
    def __init__(self, path: Path, input_key: str, output_key: str):
        self.path = path
        self.input_key = input_key
        self.output_key = output_key
        self.files = sorted(path.glob('*.npz')) + sorted(path.glob('*.npy'))
        
        if not self.files:
            self.files = sorted(path.glob('*.pt'))
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file = self.files[idx]
        if file.suffix == '.npz':
            data = np.load(file)
            result = {'input': torch.tensor(data[self.input_key], dtype=torch.float32)}
            if self.output_key in data:
                result['output'] = torch.tensor(data[self.output_key], dtype=torch.float32)
        elif file.suffix == '.npy':
            data = np.load(file)
            result = {'input': torch.tensor(data, dtype=torch.float32)}
        elif file.suffix == '.pt':
            data = torch.load(file)
            result = {'input': data[self.input_key] if isinstance(data, dict) else data}
            if isinstance(data, dict) and self.output_key in data:
                result['output'] = data[self.output_key]
        else:
            raise ValueError(f"Unknown file format: {file.suffix}")
        
        return result


# =============================================================================
# Factory Function
# =============================================================================

def create_sampler(
    strategy: str,
    config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> SamplingStrategy:
    """
    Create a sampling strategy from configuration.
    
    Args:
        strategy: "random", "dataset", "grid_query", or "teacher_guided"
        config: Strategy-specific configuration
        device: Device to put tensors on
        
    Returns:
        SamplingStrategy instance
    """
    if strategy == "random":
        return RandomSampling(
            spatial_range=config.get('spatial_range', [-1.0, 1.0]),
            parameter_range=config.get('parameter_range'),
            input_dim=config.get('input_dim'),
            device=device,
        )
    
    elif strategy == "grid_query":
        return GridQuerySampling(
            spatial_range=config.get('spatial_range', [-1.0, 1.0]),
            temporal_range=config.get('temporal_range', [0.0, 1.0]),
            num_query_points=config.get('num_query_points_per_sample', 4096),
            device=device,
        )
    
    elif strategy == "dataset":
        return DatasetSampling(
            dataset_path=config.get('source_dataset'),
            input_key=config.get('input_key', 'input'),
            output_key=config.get('output_key', 'output'),
            augmentation=config.get('augmentation'),
            device=device,
        )
    
    elif strategy == "teacher_guided":
        base_sampler = RandomSampling(
            spatial_range=config.get('spatial_range', [-1.0, 1.0]),
            parameter_range=config.get('parameter_range'),
            device=device,
        )
        return TeacherGuidedSampling(
            base_sampler=base_sampler,
            complexity_metric=config.get('complexity_metric', 'output_gradient_norm'),
            oversample_ratio=config.get('oversample_ratio', 3.0),
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
