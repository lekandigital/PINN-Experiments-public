"""
Unit tests for metric calculations.
"""

import numpy as np
import pytest

from harness.metrics import (
    compute_rmse,
    compute_mse,
    compute_mae,
    compute_nrmse,
    compute_l2_relative_error,
    compute_l1_relative_error,
    compute_max_error,
    compute_r2_score,
    compute_chamfer_distance,
    compute_eikonal_violation,
    compute_edge_preservation_error,
)


class TestBasicMetrics:
    """Test basic accuracy metrics."""
    
    def test_rmse_perfect(self):
        """RMSE should be 0 for identical arrays."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(pred, target) == pytest.approx(0.0)
    
    def test_rmse_known_value(self):
        """RMSE should match hand-calculated value."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])
        # MSE = (1 + 1 + 1) / 3 = 1, RMSE = 1
        assert compute_rmse(pred, target) == pytest.approx(1.0)
    
    def test_mse_known_value(self):
        """MSE should be mean of squared differences."""
        pred = np.array([0.0, 0.0])
        target = np.array([3.0, 4.0])
        # MSE = (9 + 16) / 2 = 12.5
        assert compute_mse(pred, target) == pytest.approx(12.5)
    
    def test_mae_known_value(self):
        """MAE should be mean of absolute differences."""
        pred = np.array([1.0, 5.0])
        target = np.array([3.0, 2.0])
        # MAE = (2 + 3) / 2 = 2.5
        assert compute_mae(pred, target) == pytest.approx(2.5)
    
    def test_max_error(self):
        """Max error should be largest absolute difference."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.0, 5.0])
        assert compute_max_error(pred, target) == pytest.approx(2.0)


class TestNormalizedMetrics:
    """Test normalized metrics."""
    
    def test_nrmse_range(self):
        """NRMSE with range normalization."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([0.0, 2.0, 4.0])  # range = 4
        # RMSE = sqrt(2/3) ≈ 0.816
        # NRMSE = 0.816 / 4 ≈ 0.204
        nrmse = compute_nrmse(pred, target, normalization="range")
        assert nrmse == pytest.approx(0.204, rel=0.01)
    
    def test_nrmse_std(self):
        """NRMSE with std normalization."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.0, 2.0, 3.0, 4.0])
        nrmse = compute_nrmse(pred, target, normalization="std")
        assert nrmse == pytest.approx(0.0)
    
    def test_l2_relative_error_perfect(self):
        """L2 relative error should be 0 for identical arrays."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        assert compute_l2_relative_error(pred, target) == pytest.approx(0.0)
    
    def test_l2_relative_error_known(self):
        """L2 relative error with known values."""
        pred = np.array([0.0, 0.0])
        target = np.array([3.0, 4.0])
        # ||pred - target|| = 5, ||target|| = 5
        # L2 rel = 5 / 5 = 1.0
        assert compute_l2_relative_error(pred, target) == pytest.approx(1.0)
    
    def test_l1_relative_error(self):
        """L1 relative error calculation."""
        pred = np.array([0.0, 0.0])
        target = np.array([2.0, 3.0])
        # L1 diff = 5, L1 target = 5
        assert compute_l1_relative_error(pred, target) == pytest.approx(1.0)


class TestR2Score:
    """Test R-squared calculation."""
    
    def test_r2_perfect(self):
        """R2 should be 1.0 for perfect prediction."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        assert compute_r2_score(pred, target) == pytest.approx(1.0)
    
    def test_r2_mean_predictor(self):
        """R2 should be 0 for mean predictor."""
        target = np.array([1.0, 2.0, 3.0])
        pred = np.full_like(target, target.mean())
        assert compute_r2_score(pred, target) == pytest.approx(0.0)
    
    def test_r2_negative(self):
        """R2 can be negative for bad predictions."""
        pred = np.array([3.0, 2.0, 1.0])  # Reversed
        target = np.array([1.0, 2.0, 3.0])
        r2 = compute_r2_score(pred, target)
        assert r2 < 0


class TestDomainMetrics:
    """Test domain-specific metrics."""
    
    def test_chamfer_distance_identical(self):
        """Chamfer distance should be 0 for identical point clouds."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        cd = compute_chamfer_distance(points, points)
        assert cd == pytest.approx(0.0)
    
    def test_chamfer_distance_shifted(self):
        """Chamfer distance for shifted point cloud."""
        points_a = np.array([[0, 0, 0], [1, 0, 0]])
        points_b = np.array([[0.1, 0, 0], [1.1, 0, 0]])
        cd = compute_chamfer_distance(points_a, points_b)
        # Each point is 0.1 away from nearest, so CD = 0.1 + 0.1 = 0.2
        assert cd == pytest.approx(0.2)
    
    def test_eikonal_violation_perfect(self):
        """Eikonal violation should be 0 for unit norm gradients."""
        # Gradients with exactly unit norm
        gradients = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
        ])
        violation = compute_eikonal_violation(gradients)
        assert violation == pytest.approx(0.0, abs=1e-10)
    
    def test_eikonal_violation_nonunit(self):
        """Eikonal violation for non-unit gradients."""
        gradients = np.array([
            [2, 0, 0],  # norm = 2, violation = 1
            [0, 0.5, 0],  # norm = 0.5, violation = 0.5
        ])
        violation = compute_eikonal_violation(gradients)
        # Mean violation = (1 + 0.5) / 2 = 0.75
        assert violation == pytest.approx(0.75)
    
    def test_edge_preservation_perfect(self):
        """Edge preservation error should be 0 for identical meshes."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        edges = np.array([[0, 1], [1, 2], [2, 0]])
        
        error = compute_edge_preservation_error(positions, positions, edges)
        assert error == pytest.approx(0.0)
    
    def test_edge_preservation_scaled(self):
        """Edge preservation error for scaled mesh."""
        positions_orig = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ])
        positions_scaled = np.array([
            [0, 0, 0],
            [2, 0, 0],  # Doubled edge length
        ])
        edges = np.array([[0, 1]])
        
        error = compute_edge_preservation_error(positions_scaled, positions_orig, edges)
        # Relative error = |2 - 1| / 1 = 1.0
        assert error == pytest.approx(1.0)


class TestFrameworkConversion:
    """Test conversion from different framework tensors."""
    
    def test_numpy_passthrough(self):
        """NumPy arrays should pass through unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        rmse = compute_rmse(arr, arr)
        assert rmse == pytest.approx(0.0)
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available"
    )
    def test_pytorch_tensor(self):
        """Should handle PyTorch tensors."""
        import torch
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        rmse = compute_rmse(pred, target)
        assert rmse == pytest.approx(0.0)
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available"
    )
    def test_pytorch_cuda_tensor(self):
        """Should handle CUDA tensors."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        pred = torch.tensor([1.0, 2.0, 3.0]).cuda()
        target = torch.tensor([1.0, 2.0, 3.0]).cuda()
        rmse = compute_rmse(pred, target)
        assert rmse == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
