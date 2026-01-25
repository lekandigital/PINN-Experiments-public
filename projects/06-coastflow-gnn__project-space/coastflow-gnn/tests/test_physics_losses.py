"""
Unit Tests for Physics-Informed Loss Functions

Tests:
- Continuity loss (divergence-free constraint)
- Momentum loss (Navier-Stokes residuals)
- Turbulence loss (k-ε model constraints)
- Boundary loss (wall/outlet conditions)
- Combined physics-informed loss
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.physics_losses import (
    GraphDifferentialOperators,
    compute_continuity_loss,
    compute_momentum_loss,
    compute_turbulence_loss,
    compute_boundary_loss,
    physics_informed_loss,
    PhysicsInformedLoss,
)


class TestGraphDifferentialOperators:
    """Test suite for graph-based differential operators."""
    
    @pytest.fixture
    def graph_data(self):
        """Create simple graph for testing."""
        N = 50
        E = 150
        
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        return pos, edge_index
    
    def test_compute_edge_vectors(self, graph_data):
        """Test edge vector computation."""
        pos, edge_index = graph_data
        
        edge_vec, edge_dist = GraphDifferentialOperators.compute_edge_vectors(
            pos, edge_index
        )
        
        assert edge_vec.shape == (edge_index.shape[1], 3)
        assert edge_dist.shape == (edge_index.shape[1], 1)
        assert (edge_dist > 0).all(), "Distances should be positive"
        print("✓ Edge vector computation test passed")
    
    def test_compute_gradient(self, graph_data):
        """Test gradient computation on scalar field."""
        pos, edge_index = graph_data
        N = pos.shape[0]
        
        # Linear scalar field: f = x + 2y + 3z
        f = pos[:, 0] + 2 * pos[:, 1] + 3 * pos[:, 2]
        
        grad_f = GraphDifferentialOperators.compute_gradient(f, pos, edge_index)
        
        assert grad_f.shape == (N, 3)
        assert not torch.isnan(grad_f).any(), "Gradient contains NaN"
        print("✓ Gradient computation test passed")
    
    def test_compute_divergence(self, graph_data):
        """Test divergence computation on vector field."""
        pos, edge_index = graph_data
        N = pos.shape[0]
        
        # Random velocity field
        u = torch.randn(N, 3)
        
        div_u = GraphDifferentialOperators.compute_divergence(u, pos, edge_index)
        
        assert div_u.shape == (N, 1)
        assert not torch.isnan(div_u).any(), "Divergence contains NaN"
        print("✓ Divergence computation test passed")
    
    def test_compute_laplacian(self, graph_data):
        """Test Laplacian computation."""
        pos, edge_index = graph_data
        N = pos.shape[0]
        
        # Scalar field
        f = torch.randn(N, 1)
        
        lap_f = GraphDifferentialOperators.compute_laplacian(f, pos, edge_index)
        
        assert lap_f.shape == (N, 1)
        assert not torch.isnan(lap_f).any(), "Laplacian contains NaN"
        print("✓ Laplacian computation test passed")


class TestContinuityLoss:
    """Test suite for continuity loss."""
    
    def test_continuity_loss_basic(self):
        """Test basic continuity loss computation."""
        N, E = 100, 300
        
        u = torch.randn(N, 3, requires_grad=True)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss = compute_continuity_loss(u, pos, edge_index)
        
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss), "Continuity loss is NaN"
        assert loss >= 0, "Loss should be non-negative"
        print(f"✓ Continuity loss: {loss.item():.6f}")
    
    def test_continuity_loss_gradient(self):
        """Test gradient flow through continuity loss."""
        N, E = 50, 150
        
        u = torch.randn(N, 3, requires_grad=True)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss = compute_continuity_loss(u, pos, edge_index)
        loss.backward()
        
        assert u.grad is not None, "Gradient should exist"
        assert not torch.isnan(u.grad).any(), "Gradient contains NaN"
        print("✓ Continuity loss gradient test passed")
    
    def test_continuity_loss_reduction(self):
        """Test different reduction modes."""
        N, E = 50, 150
        
        u = torch.randn(N, 3)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss_mean = compute_continuity_loss(u, pos, edge_index, reduction='mean')
        loss_sum = compute_continuity_loss(u, pos, edge_index, reduction='sum')
        loss_none = compute_continuity_loss(u, pos, edge_index, reduction='none')
        
        assert loss_mean.dim() == 0, "Mean reduction should give scalar"
        assert loss_sum.dim() == 0, "Sum reduction should give scalar"
        assert loss_none.dim() == 1, "No reduction should give vector"
        print("✓ Continuity loss reduction test passed")


class TestMomentumLoss:
    """Test suite for momentum loss."""
    
    def test_momentum_loss_basic(self):
        """Test basic momentum loss computation."""
        N, E = 100, 300
        
        u = torch.randn(N, 3, requires_grad=True)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss = compute_momentum_loss(u, pos, edge_index)
        
        assert not torch.isnan(loss), "Momentum loss is NaN"
        assert loss >= 0, "Loss should be non-negative"
        print(f"✓ Momentum loss: {loss.item():.6f}")
    
    def test_momentum_loss_gradient(self):
        """Test gradient flow through momentum loss."""
        N, E = 50, 150
        
        u = torch.randn(N, 3, requires_grad=True)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss = compute_momentum_loss(u, pos, edge_index)
        loss.backward()
        
        assert u.grad is not None
        print("✓ Momentum loss gradient test passed")
    
    def test_momentum_loss_with_viscosity(self):
        """Test momentum loss with different viscosity values."""
        N, E = 50, 150
        
        u = torch.randn(N, 3)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        # Water vs air viscosity
        loss_water = compute_momentum_loss(u, pos, edge_index, nu=1e-6)
        loss_air = compute_momentum_loss(u, pos, edge_index, nu=1.5e-5)
        
        assert not torch.isnan(loss_water)
        assert not torch.isnan(loss_air)
        print("✓ Momentum loss viscosity test passed")


class TestTurbulenceLoss:
    """Test suite for turbulence loss."""
    
    def test_turbulence_loss_basic(self):
        """Test basic turbulence loss computation."""
        N, E = 100, 300
        
        u = torch.randn(N, 3)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss = compute_turbulence_loss(u, None, None, pos, edge_index)
        
        assert not torch.isnan(loss), "Turbulence loss is NaN"
        print(f"✓ Turbulence loss: {loss.item():.6f}")
    
    def test_turbulence_loss_with_ke(self):
        """Test turbulence loss with explicit k and epsilon."""
        N, E = 100, 300
        
        u = torch.randn(N, 3)
        k = torch.rand(N) + 0.1  # Positive values
        epsilon = torch.rand(N) + 0.1
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        
        loss = compute_turbulence_loss(u, k, epsilon, pos, edge_index)
        
        assert not torch.isnan(loss)
        print("✓ Turbulence loss with k-ε test passed")


class TestBoundaryLoss:
    """Test suite for boundary condition loss."""
    
    def test_boundary_loss_no_slip(self):
        """Test no-slip wall boundary condition."""
        N = 100
        
        u = torch.randn(N, 3)
        wave_height = torch.rand(N)
        pos = torch.randn(N, 3)
        
        # Mark some nodes as walls
        boundary_mask = torch.zeros(N, dtype=torch.bool)
        boundary_mask[:10] = True
        
        boundary_type = torch.zeros(N, dtype=torch.long)
        boundary_type[:10] = 1  # No-slip wall
        
        loss = compute_boundary_loss(
            u, wave_height, pos, boundary_mask, boundary_type
        )
        
        assert not torch.isnan(loss)
        assert loss >= 0
        print(f"✓ Boundary loss (no-slip): {loss.item():.6f}")
    
    def test_boundary_loss_mixed(self):
        """Test mixed boundary conditions."""
        N = 100
        
        u = torch.randn(N, 3)
        wave_height = torch.rand(N)
        pos = torch.randn(N, 3)
        
        boundary_mask = torch.zeros(N, dtype=torch.bool)
        boundary_mask[:30] = True
        
        boundary_type = torch.zeros(N, dtype=torch.long)
        boundary_type[:10] = 1   # No-slip wall
        boundary_type[10:20] = 2  # Free surface
        boundary_type[20:30] = 4  # Outlet
        
        loss = compute_boundary_loss(
            u, wave_height, pos, boundary_mask, boundary_type
        )
        
        assert not torch.isnan(loss)
        print("✓ Boundary loss (mixed) test passed")


class TestPhysicsInformedLoss:
    """Test suite for combined physics-informed loss."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy PyG Data object."""
        N, E = 100, 300
        
        x = torch.randn(N, 6)  # [x, y, z, elev, wind_u, wind_v]
        y = torch.randn(N, 4)  # [u_x, u_y, u_z, wave_height]
        edge_index = torch.randint(0, N, (2, E))
        pos = x[:, :3]
        
        boundary_mask = torch.zeros(N, dtype=torch.bool)
        boundary_mask[:10] = True
        boundary_type = torch.ones(N, dtype=torch.long)  # All walls
        boundary_type[~boundary_mask] = 0  # Interior
        
        data = Data(
            x=x, y=y, edge_index=edge_index, pos=pos,
            boundary_mask=boundary_mask, boundary_type=boundary_type
        )
        
        return data
    
    def test_physics_informed_loss_basic(self, dummy_data):
        """Test combined physics-informed loss."""
        predictions = torch.randn(100, 4)
        
        losses = physics_informed_loss(dummy_data, predictions)
        
        assert 'total' in losses
        assert 'data' in losses
        assert 'continuity' in losses
        assert 'momentum' in losses
        assert 'turbulence' in losses
        assert 'boundary' in losses
        
        assert not torch.isnan(losses['total'])
        print(f"✓ Physics-informed loss: {losses['total'].item():.6f}")
    
    def test_physics_informed_loss_weights(self, dummy_data):
        """Test loss weighting."""
        predictions = torch.randn(100, 4)
        
        losses_default = physics_informed_loss(dummy_data, predictions)
        losses_high_phys = physics_informed_loss(
            dummy_data, predictions,
            lambda_data=1.0, lambda_cont=10.0, lambda_mom=1.0
        )
        
        # Higher physics weight should give different total
        assert losses_default['total'] != losses_high_phys['total']
        print("✓ Loss weighting test passed")
    
    def test_physics_informed_loss_module(self, dummy_data):
        """Test PhysicsInformedLoss module wrapper."""
        predictions = torch.randn(100, 4)
        
        criterion = PhysicsInformedLoss(
            lambda_data=1.0,
            lambda_cont=1.0,
            lambda_mom=0.1,
        )
        
        losses = criterion(dummy_data, predictions)
        
        assert not torch.isnan(losses['total'])
        print("✓ PhysicsInformedLoss module test passed")


def test_continuity_loss():
    """Standalone test for continuity loss."""
    u = torch.randn(100, 3, requires_grad=True)
    pos = torch.randn(100, 3)
    edge_index = torch.randint(0, 100, (2, 300))
    
    loss = compute_continuity_loss(u, pos, edge_index)
    
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"✓ Continuity loss test passed (loss={loss.item():.6f})")


def test_momentum_loss():
    """Standalone test for momentum loss."""
    u = torch.randn(100, 3, requires_grad=True)
    pos = torch.randn(100, 3)
    edge_index = torch.randint(0, 100, (2, 300))
    
    loss = compute_momentum_loss(u, pos, edge_index)
    
    assert not torch.isnan(loss), "Momentum loss is NaN"
    print(f"✓ Momentum loss test passed (loss={loss.item():.6f})")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running Physics Loss Tests")
    print("=" * 50 + "\n")
    
    test_continuity_loss()
    test_momentum_loss()
    
    print("\nRunning pytest suite...")
    pytest.main([__file__, "-v", "--tb=short"])
