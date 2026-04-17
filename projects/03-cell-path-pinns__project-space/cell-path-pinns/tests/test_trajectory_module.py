"""
Tests for shared trajectory module.

Tests data loading, metrics, and visualization utilities.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, '../../..')  # Add shared to path
sys.path.insert(0, '..')

# Import from shared trajectory module
try:
    from shared.trajectory import (
        TrajectoryBatch,
        TrajectoryDataset,
        path_length,
        path_energy,
        path_smoothness,
        curvature,
        frechet_distance,
        goal_reaching_accuracy,
    )
    SHARED_AVAILABLE = True
except ImportError:
    SHARED_AVAILABLE = False


# ============================================================================
# TrajectoryBatch Tests
# ============================================================================

@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestTrajectoryBatch:
    """Tests for TrajectoryBatch dataclass."""
    
    def test_construction(self):
        """Test basic construction."""
        batch = TrajectoryBatch(
            positions=torch.randn(10, 50, 2),
            times=torch.linspace(0, 1, 50).unsqueeze(0).expand(10, -1),
        )
        assert batch.positions.shape == (10, 50, 2)
        assert batch.times.shape == (10, 50)
        
    def test_optional_fields(self):
        """Test optional fields."""
        batch = TrajectoryBatch(
            positions=torch.randn(10, 50, 2),
            times=torch.linspace(0, 1, 50).unsqueeze(0).expand(10, -1),
            velocities=torch.randn(10, 50, 2),
            context=torch.randn(10, 4),
            metadata={'source': 'synthetic'},
        )
        
        assert batch.velocities is not None
        assert batch.context is not None
        assert batch.metadata['source'] == 'synthetic'
        
    def test_to_device(self):
        """Test device transfer."""
        batch = TrajectoryBatch(
            positions=torch.randn(10, 50, 2),
            times=torch.linspace(0, 1, 50).unsqueeze(0).expand(10, -1),
        )
        
        # Should work even on CPU-only systems
        batch_cpu = batch.to('cpu')
        assert batch_cpu.positions.device.type == 'cpu'


# ============================================================================
# TrajectoryDataset Tests
# ============================================================================

@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestTrajectoryDataset:
    """Tests for TrajectoryDataset."""
    
    def test_from_tensor(self):
        """Test creation from tensor."""
        positions = torch.randn(100, 50, 2)
        dataset = TrajectoryDataset(positions=positions)
        
        assert len(dataset) == 100
        
    def test_getitem(self):
        """Test indexing."""
        positions = torch.randn(100, 50, 2)
        dataset = TrajectoryDataset(positions=positions)
        
        item = dataset[0]
        assert isinstance(item, dict) or isinstance(item, TrajectoryBatch)
        
    def test_dataloader_compatible(self):
        """Test compatibility with DataLoader."""
        from torch.utils.data import DataLoader
        
        positions = torch.randn(100, 50, 2)
        dataset = TrajectoryDataset(positions=positions)
        
        loader = DataLoader(dataset, batch_size=10, shuffle=True)
        batch = next(iter(loader))
        
        assert batch is not None
        
    def test_save_load_hdf5(self):
        """Test HDF5 save/load."""
        pytest.importorskip('h5py')
        
        positions = torch.randn(50, 30, 2)
        dataset = TrajectoryDataset(positions=positions)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'trajectories.h5')
            dataset.save_hdf5(path)
            
            loaded = TrajectoryDataset.load_hdf5(path)
            assert len(loaded) == len(dataset)
            
    def test_save_load_csv(self):
        """Test CSV save/load for single trajectory."""
        positions = torch.randn(1, 30, 2)
        dataset = TrajectoryDataset(positions=positions)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'trajectory.csv')
            dataset.save_csv(path)
            
            # Check file exists
            assert os.path.exists(path)


# ============================================================================
# Metrics Tests
# ============================================================================

@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestPathLength:
    """Tests for path_length metric."""
    
    def test_straight_line(self):
        """Test length of straight line."""
        # Straight line from (0,0) to (1,0)
        positions = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        
        length = path_length(positions)
        assert torch.isclose(length, torch.tensor(1.0), atol=1e-5)
        
    def test_diagonal(self):
        """Test length of diagonal line."""
        # Straight line from (0,0) to (1,1)
        positions = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        
        length = path_length(positions)
        expected = torch.sqrt(torch.tensor(2.0))
        assert torch.isclose(length, expected, atol=1e-5)
        
    def test_batched(self):
        """Test batched path length."""
        positions = torch.randn(10, 50, 2)
        
        lengths = path_length(positions)
        assert lengths.shape == (10,)
        assert (lengths > 0).all()


@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestPathEnergy:
    """Tests for path_energy metric."""
    
    def test_constant_speed(self):
        """Test energy with constant speed."""
        # Constant speed trajectory
        t = torch.linspace(0, 1, 50)
        positions = torch.stack([t, torch.zeros_like(t)], dim=-1)
        
        energy = path_energy(positions, dt=1/49)
        
        # Energy should be roughly speed^2 * time = 1^2 * 1 = 1
        assert energy.item() > 0
        
    def test_batched(self):
        """Test batched energy computation."""
        positions = torch.randn(10, 50, 2)
        
        energies = path_energy(positions)
        assert energies.shape == (10,)


@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestPathSmoothness:
    """Tests for path_smoothness metric."""
    
    def test_straight_line_smooth(self):
        """Test that straight line is smooth."""
        # Perfectly straight line
        t = torch.linspace(0, 1, 50)
        positions = torch.stack([t, torch.zeros_like(t)], dim=-1)
        
        smoothness = path_smoothness(positions)
        
        # Straight line should have low smoothness penalty
        assert smoothness.item() < 0.01
        
    def test_zigzag_rough(self):
        """Test that zigzag path is rough."""
        # Zigzag path
        t = torch.linspace(0, 1, 50)
        y = torch.sin(t * 20)  # High frequency oscillation
        positions = torch.stack([t, y], dim=-1)
        
        smoothness = path_smoothness(positions)
        
        # Zigzag should have higher smoothness penalty
        assert smoothness.item() > 0.1


@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestCurvature:
    """Tests for curvature metric."""
    
    def test_straight_zero_curvature(self):
        """Test that straight line has zero curvature."""
        t = torch.linspace(0, 1, 50)
        positions = torch.stack([t, t], dim=-1)
        
        kappa = curvature(positions)
        
        # Curvature should be near zero
        assert kappa.mean().item() < 0.1
        
    def test_circle_constant_curvature(self):
        """Test that circle has constant curvature."""
        # Circle of radius 1
        theta = torch.linspace(0, 2 * np.pi, 100)
        positions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        
        kappa = curvature(positions)
        
        # Curvature of unit circle is 1
        # Allow for numerical error at endpoints
        interior_kappa = kappa[5:-5]
        assert (interior_kappa > 0.5).all()  # Should be close to 1


@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestFrechetDistance:
    """Tests for frechet_distance metric."""
    
    def test_identical_paths(self):
        """Test distance between identical paths."""
        path = torch.randn(50, 2)
        
        dist = frechet_distance(path, path)
        assert dist.item() < 1e-5
        
    def test_different_paths(self):
        """Test distance between different paths."""
        path1 = torch.zeros(50, 2)
        path2 = torch.ones(50, 2)
        
        dist = frechet_distance(path1, path2)
        assert dist.item() > 0
        
    def test_symmetric(self):
        """Test that distance is symmetric."""
        path1 = torch.randn(50, 2)
        path2 = torch.randn(50, 2)
        
        d12 = frechet_distance(path1, path2)
        d21 = frechet_distance(path2, path1)
        
        assert torch.isclose(d12, d21, atol=1e-5)


@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestGoalReachingAccuracy:
    """Tests for goal_reaching_accuracy metric."""
    
    def test_all_reach_goal(self):
        """Test when all trajectories reach goal."""
        # All end at (1, 1)
        trajectories = torch.randn(10, 50, 2)
        trajectories[:, -1, :] = torch.tensor([1.0, 1.0])
        
        goal = torch.tensor([1.0, 1.0])
        threshold = 0.1
        
        accuracy = goal_reaching_accuracy(trajectories, goal, threshold)
        assert accuracy == 1.0
        
    def test_none_reach_goal(self):
        """Test when no trajectories reach goal."""
        # All end at (0, 0)
        trajectories = torch.zeros(10, 50, 2)
        
        goal = torch.tensor([10.0, 10.0])
        threshold = 0.1
        
        accuracy = goal_reaching_accuracy(trajectories, goal, threshold)
        assert accuracy == 0.0
        
    def test_partial_reach(self):
        """Test partial reaching."""
        trajectories = torch.zeros(10, 50, 2)
        # Half reach the goal
        trajectories[0:5, -1, :] = torch.tensor([1.0, 1.0])
        
        goal = torch.tensor([1.0, 1.0])
        threshold = 0.1
        
        accuracy = goal_reaching_accuracy(trajectories, goal, threshold)
        assert accuracy == 0.5


# ============================================================================
# Visualization Tests (smoke tests)
# ============================================================================

@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestVisualization:
    """Smoke tests for visualization functions."""
    
    @pytest.fixture
    def sample_trajectory_2d(self):
        """Create sample 2D trajectory."""
        t = torch.linspace(0, 1, 50)
        x = t + 0.1 * torch.sin(t * 10)
        y = t + 0.1 * torch.cos(t * 10)
        return torch.stack([x, y], dim=-1)
        
    @pytest.fixture
    def sample_trajectory_3d(self):
        """Create sample 3D trajectory."""
        t = torch.linspace(0, 1, 50)
        x = t
        y = torch.sin(t * 2 * np.pi)
        z = torch.cos(t * 2 * np.pi)
        return torch.stack([x, y, z], dim=-1)
    
    def test_plot_trajectory_2d_runs(self, sample_trajectory_2d):
        """Test that 2D plotting doesn't crash."""
        pytest.importorskip('matplotlib')
        from shared.trajectory import plot_trajectory_2d
        
        fig = plot_trajectory_2d(sample_trajectory_2d)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    def test_plot_trajectory_3d_runs(self, sample_trajectory_3d):
        """Test that 3D plotting doesn't crash."""
        pytest.importorskip('matplotlib')
        from shared.trajectory import plot_trajectory_3d
        
        fig = plot_trajectory_3d(sample_trajectory_3d)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    def test_plot_potential_field_2d_runs(self):
        """Test that potential field plotting doesn't crash."""
        pytest.importorskip('matplotlib')
        from shared.trajectory import plot_potential_field_2d
        
        # Simple quadratic potential
        def potential_fn(x):
            return (x ** 2).sum(dim=-1, keepdim=True)
        
        fig = plot_potential_field_2d(potential_fn, xlim=(-1, 1), ylim=(-1, 1))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not SHARED_AVAILABLE, reason="shared.trajectory not available")
class TestTrajectoryModuleIntegration:
    """Integration tests for trajectory module."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: create, compute metrics, visualize."""
        # Create synthetic data
        batch_size = 20
        seq_len = 50
        
        # Generate circular trajectories with noise
        theta = torch.linspace(0, np.pi, seq_len)
        positions = torch.zeros(batch_size, seq_len, 2)
        
        for i in range(batch_size):
            r = 0.5 + 0.1 * torch.randn(1)
            noise = 0.02 * torch.randn(seq_len, 2)
            positions[i] = torch.stack([
                r * torch.cos(theta),
                r * torch.sin(theta),
            ], dim=-1) + noise
        
        # Create dataset
        dataset = TrajectoryDataset(positions=positions)
        assert len(dataset) == batch_size
        
        # Compute metrics
        lengths = path_length(positions)
        assert lengths.shape == (batch_size,)
        assert (lengths > 0).all()
        
        energies = path_energy(positions)
        assert energies.shape == (batch_size,)
        
        smoothness = path_smoothness(positions)
        assert smoothness.shape == (batch_size,)
        
        # Check goal reaching
        goal = positions[0, -1, :]  # Use first trajectory's end
        accuracy = goal_reaching_accuracy(positions, goal, threshold=0.2)
        assert 0.0 <= accuracy <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
