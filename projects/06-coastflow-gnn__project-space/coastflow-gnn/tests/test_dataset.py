"""
Unit Tests for Synthetic Coastal Dataset

Tests:
- Dataset creation and loading
- Data sample structure and shapes
- Data loader batching
- Graph construction (edges)
- Boundary identification
"""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import SyntheticCoastalDataset, create_data_loaders
from src.data.mesh_builder import create_coastal_mesh, mesh_to_graph


class TestSyntheticCoastalDataset:
    """Test suite for synthetic dataset."""
    
    @pytest.fixture
    def dataset(self, tmp_path):
        """Create small dataset for testing."""
        return SyntheticCoastalDataset(
            root=str(tmp_path / 'test_data'),
            num_samples=20,
            num_nodes=50,
            seed=42,
        )
    
    def test_dataset_creation(self, dataset):
        """Test dataset creates successfully."""
        assert len(dataset) == 20
        print(f"✓ Dataset created with {len(dataset)} samples")
    
    def test_sample_structure(self, dataset):
        """Test individual sample has correct structure."""
        sample = dataset[0]
        
        assert isinstance(sample, Data)
        assert hasattr(sample, 'x')
        assert hasattr(sample, 'y')
        assert hasattr(sample, 'edge_index')
        assert hasattr(sample, 'pos')
        assert hasattr(sample, 'boundary_mask')
        assert hasattr(sample, 'boundary_type')
        print("✓ Sample has correct attributes")
    
    def test_sample_shapes(self, dataset):
        """Test sample tensor shapes."""
        sample = dataset[0]
        N = sample.x.shape[0]
        
        assert sample.x.shape[1] == 6, "Node features should have 6 channels"
        assert sample.y.shape == (N, 4), "Labels should have 4 channels"
        assert sample.edge_index.shape[0] == 2, "Edge index should be [2, E]"
        assert sample.pos.shape == (N, 3), "Positions should be [N, 3]"
        assert sample.boundary_mask.shape == (N,)
        assert sample.boundary_type.shape == (N,)
        print(f"✓ Sample shapes correct (N={N}, E={sample.edge_index.shape[1]})")
    
    def test_sample_values(self, dataset):
        """Test sample values are valid."""
        sample = dataset[0]
        
        assert not torch.isnan(sample.x).any(), "Features contain NaN"
        assert not torch.isnan(sample.y).any(), "Labels contain NaN"
        assert not torch.isinf(sample.x).any(), "Features contain Inf"
        assert not torch.isinf(sample.y).any(), "Labels contain Inf"
        print("✓ Sample values are valid")
    
    def test_edge_index_valid(self, dataset):
        """Test edge indices are within bounds."""
        sample = dataset[0]
        N = sample.x.shape[0]
        
        assert sample.edge_index.min() >= 0
        assert sample.edge_index.max() < N
        print("✓ Edge indices are valid")
    
    def test_boundary_mask(self, dataset):
        """Test boundary identification."""
        sample = dataset[0]
        
        num_boundary = sample.boundary_mask.sum().item()
        assert num_boundary > 0, "Should have boundary nodes"
        assert num_boundary < len(sample.boundary_mask), "Not all nodes should be boundary"
        print(f"✓ Boundary nodes: {num_boundary}/{len(sample.boundary_mask)}")
    
    def test_multiple_samples(self, dataset):
        """Test loading multiple samples."""
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert sample.x.shape[0] > 0
        print("✓ Multiple samples loaded successfully")
    
    def test_deterministic(self, tmp_path):
        """Test dataset is deterministic with same seed."""
        ds1 = SyntheticCoastalDataset(
            root=str(tmp_path / 'test_data1'),
            num_samples=5,
            num_nodes=30,
            seed=123,
        )
        ds2 = SyntheticCoastalDataset(
            root=str(tmp_path / 'test_data2'),
            num_samples=5,
            num_nodes=30,
            seed=123,
        )
        
        # Same seed should give same first sample
        assert torch.allclose(ds1[0].x, ds2[0].x)
        print("✓ Dataset is deterministic")


class TestDataLoaders:
    """Test suite for data loaders."""
    
    @pytest.fixture
    def dataset(self, tmp_path):
        return SyntheticCoastalDataset(
            root=str(tmp_path / 'loader_test'),
            num_samples=20,
            num_nodes=50,
            seed=42,
        )
    
    def test_create_data_loaders(self, dataset):
        """Test data loader creation."""
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset, batch_size=4
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        print(f"✓ Data loaders created (train: {len(train_loader)}, "
              f"val: {len(val_loader)}, test: {len(test_loader)} batches)")
    
    def test_batch_structure(self, dataset):
        """Test batch has correct structure."""
        train_loader, _, _ = create_data_loaders(dataset, batch_size=4)
        batch = next(iter(train_loader))
        
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'y')
        assert hasattr(batch, 'edge_index')
        assert hasattr(batch, 'batch')
        print(f"✓ Batch structure correct (nodes: {batch.x.shape[0]})")
    
    def test_batch_iteration(self, dataset):
        """Test iterating through all batches."""
        train_loader, _, _ = create_data_loaders(dataset, batch_size=4)
        
        total_nodes = 0
        for batch in train_loader:
            total_nodes += batch.x.shape[0]
        
        print(f"✓ Iterated through all batches (total nodes: {total_nodes})")


class TestMeshBuilder:
    """Test suite for mesh builder utilities."""
    
    def test_create_coastal_mesh(self):
        """Test mesh creation."""
        points, elevation = create_coastal_mesh(
            x_range=(0, 500),
            y_range=(0, 300),
            resolution=50.0,
        )
        
        assert len(points) > 0
        assert points.shape[1] == 3
        assert len(elevation) == len(points)
        print(f"✓ Created mesh with {len(points)} points")
    
    def test_mesh_to_graph(self):
        """Test mesh to graph conversion."""
        points, elevation = create_coastal_mesh(
            x_range=(0, 500),
            y_range=(0, 300),
            resolution=50.0,
        )
        
        data = mesh_to_graph(points, elevation)
        
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'pos')
        assert data.edge_index.shape[0] == 2
        print(f"✓ Converted to graph with {data.edge_index.shape[1]} edges")


def test_dataset_creation():
    """Standalone dataset test."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = SyntheticCoastalDataset(
            root=tmp_dir,
            num_samples=10,
            num_nodes=30,
        )
        
        assert len(dataset) == 10
        sample = dataset[0]
        assert sample.x.shape[1] == 6
        assert sample.y.shape[1] == 4
        
        print("✓ Dataset creation test passed")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running Dataset Tests")
    print("=" * 50 + "\n")
    
    test_dataset_creation()
    
    print("\nRunning pytest suite...")
    pytest.main([__file__, "-v", "--tb=short"])
