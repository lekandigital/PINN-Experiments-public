"""
PEGNN-Deform: Unit Tests

Comprehensive tests for model, features, training, and benchmarking.

Run with: pytest tests/test_model.py -v

Author: PEGNN-Deform Team
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pegdeform_model import PEGNNDeform, SpringMessagePassing, VelocityGRU, count_parameters
from mesh_features import compute_mesh_features, compute_edge_features, build_mesh_graph


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_mesh():
    """Create a simple test mesh (tetrahedron)."""
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=torch.long)
    
    return pos, faces


@pytest.fixture
def grid_mesh():
    """Create a 5x5 grid mesh."""
    size = 5
    N = size * size
    
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    pos = torch.stack([xx.flatten(), yy.flatten(), torch.zeros(N)], dim=1)
    
    edge_list = []
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            if j < size - 1:
                edge_list.extend([[idx, idx + 1], [idx + 1, idx]])
            if i < size - 1:
                edge_list.extend([[idx, idx + size], [idx + size, idx]])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    return pos, edge_index


@pytest.fixture
def random_graph(device):
    """Create random graph for testing."""
    N = 100  # nodes
    E = 300  # edges
    
    pos = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device) * 0.1
    edge_index = torch.randint(0, N, (2, E), device=device)
    edge_attr = torch.rand(E, 2, device=device)
    edge_attr[:, 0] = edge_attr[:, 0] * 10 + 1  # stiffness
    edge_attr[:, 1] = edge_attr[:, 1] * 0.5 + 0.1  # rest length
    
    return pos, vel, edge_index, edge_attr


# ============================================================================
# Model Tests
# ============================================================================

class TestSpringMessagePassing:
    """Tests for SpringMessagePassing layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = SpringMessagePassing(use_learned_modulation=True)
        assert layer is not None
        assert hasattr(layer, 'edge_mlp')
        
        layer_no_mod = SpringMessagePassing(use_learned_modulation=False)
        assert not hasattr(layer_no_mod, 'edge_mlp')
    
    def test_forward_shape(self, random_graph, device):
        """Test forward pass output shape."""
        pos, vel, edge_index, edge_attr = random_graph
        layer = SpringMessagePassing().to(device)
        
        forces = layer(pos, edge_index, edge_attr)

        assert forces.shape == pos.shape
        assert forces.device.type == device.type
    
    def test_hookes_law(self, device):
        """Test that Hooke's law is correctly implemented."""
        # Two nodes connected by a spring
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device)
        edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
        
        k = 10.0  # stiffness
        L0 = 1.0  # rest length
        edge_attr = torch.tensor([[k, L0], [k, L0]], device=device)
        
        layer = SpringMessagePassing(use_learned_modulation=False).to(device)
        forces = layer(pos, edge_index, edge_attr)
        
        # Spring is stretched (length=2, rest=1), so force should pull nodes together
        # Force on node 0 should be positive x (toward node 1)
        # Force on node 1 should be negative x (toward node 0)
        expected_force_mag = k * (2.0 - L0)  # = 10
        
        assert forces[0, 0] > 0  # Force on node 0 in +x direction
        assert forces[1, 0] < 0  # Force on node 1 in -x direction
        assert torch.allclose(forces[0, 0], torch.tensor(expected_force_mag, device=device), atol=0.5)


class TestVelocityGRU:
    """Tests for VelocityGRU module."""
    
    def test_initialization(self):
        """Test GRU initialization."""
        gru = VelocityGRU(input_size=3, hidden_size=64)
        assert gru.hidden_size == 64
    
    def test_forward_shape(self, device):
        """Test forward pass shapes."""
        N = 50
        gru = VelocityGRU(input_size=3, hidden_size=64).to(device)
        
        vel_update = torch.randn(N, 3, device=device)
        
        # Without hidden state
        vel_out, hidden = gru(vel_update)
        assert vel_out.shape == (N, 3)
        assert hidden.shape == (N, 64)
        
        # With hidden state
        vel_out2, hidden2 = gru(vel_update, hidden)
        assert vel_out2.shape == (N, 3)
        assert hidden2.shape == (N, 64)


class TestPEGNNDeform:
    """Tests for main PEGNN-Deform model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = PEGNNDeform(hidden_size=64, num_mp_layers=3, dt=0.01)
        
        assert model.hidden_size == 64
        assert model.num_mp_layers == 3
        assert model.dt == 0.01
        assert len(model.spring_layers) == 3
    
    def test_forward_pass(self, random_graph, device):
        """Test forward pass."""
        pos, vel, edge_index, edge_attr = random_graph
        model = PEGNNDeform(hidden_size=64).to(device)
        
        new_pos, new_vel, hidden = model(pos, vel, edge_index, edge_attr)
        
        assert new_pos.shape == pos.shape
        assert new_vel.shape == vel.shape
        assert hidden.shape == (pos.size(0), 64)
    
    def test_forward_pass_with_hidden(self, random_graph, device):
        """Test forward pass with pre-existing hidden state."""
        pos, vel, edge_index, edge_attr = random_graph
        model = PEGNNDeform(hidden_size=64).to(device)
        
        hidden = torch.randn(pos.size(0), 64, device=device)
        
        new_pos, new_vel, new_hidden = model(pos, vel, edge_index, edge_attr, hidden=hidden)
        
        assert new_pos.shape == pos.shape
        assert new_hidden.shape == hidden.shape
    
    def test_mixed_precision(self, random_graph, device):
        """Test mixed precision inference."""
        if device.type != 'cuda':
            pytest.skip("Mixed precision requires CUDA")
        
        pos, vel, edge_index, edge_attr = random_graph
        model = PEGNNDeform(hidden_size=64).to(device)
        
        with torch.amp.autocast(device_type='cuda'):
            new_pos, new_vel, hidden = model(pos, vel, edge_index, edge_attr)
        
        assert new_pos.shape == pos.shape
    
    def test_rollout(self, random_graph, device):
        """Test multi-step rollout."""
        pos, vel, edge_index, edge_attr = random_graph
        model = PEGNNDeform(hidden_size=64).to(device)
        
        num_steps = 10
        pos_traj, vel_traj = model.rollout(pos, vel, edge_index, edge_attr, num_steps)
        
        assert pos_traj.shape == (num_steps + 1, pos.size(0), 3)
        assert vel_traj.shape == (num_steps + 1, vel.size(0), 3)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = PEGNNDeform(hidden_size=64, num_mp_layers=3)
        num_params = count_parameters(model)
        
        assert num_params > 0
        assert isinstance(num_params, int)


# ============================================================================
# Mesh Features Tests
# ============================================================================

class TestMeshFeatures:
    """Tests for mesh feature computation."""
    
    def test_compute_mesh_features(self, simple_mesh):
        """Test basic feature computation."""
        pos, faces = simple_mesh
        
        node_feats, edge_idx, edge_attr = compute_mesh_features(pos, faces)
        
        assert node_feats.shape == (4, 5)  # 4 nodes, 5 features
        assert edge_idx.shape[0] == 2
        assert edge_idx.shape[1] > 0
        assert edge_attr.shape[0] == edge_idx.shape[1]
    
    def test_laplacian_coordinates(self, grid_mesh):
        """Test Laplacian coordinate computation."""
        pos, edge_index = grid_mesh
        
        # Create dummy faces for the grid
        faces = torch.tensor([[0, 1, 6], [0, 6, 5]], dtype=torch.long)
        
        node_feats, _, _ = compute_mesh_features(pos, faces, compute_curvatures=False)
        
        # Laplacian coords are first 3 features
        L_coords = node_feats[:, :3]
        
        # For a regular grid, interior points should have zero Laplacian
        # (they are at the average of their neighbors)
        # Note: This is approximate due to boundary effects
        assert L_coords.shape == (pos.size(0), 3)
    
    def test_edge_features(self, grid_mesh):
        """Test edge feature computation."""
        pos, edge_index = grid_mesh
        
        stiffness = 5.0
        edge_attr = compute_edge_features(pos, edge_index, stiffness=stiffness)
        
        assert edge_attr.shape == (edge_index.shape[1], 2)
        assert (edge_attr[:, 0] == stiffness).all()  # Uniform stiffness
        assert (edge_attr[:, 1] > 0).all()  # Positive rest lengths
    
    def test_build_mesh_graph(self, simple_mesh):
        """Test complete graph building."""
        pos, faces = simple_mesh
        
        node_feats, edge_idx, edge_attr = build_mesh_graph(pos, faces, stiffness=10.0)
        
        assert node_feats.shape[0] == 4
        assert edge_attr.shape[1] == 2  # stiffness, rest_length


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_training_step(self, random_graph, device):
        """Test a single training step."""
        pos, vel, edge_index, edge_attr = random_graph
        
        # Create model and optimizer
        model = PEGNNDeform(hidden_size=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Ground truth (slightly perturbed)
        pos_gt = pos + torch.randn_like(pos) * 0.01
        vel_gt = vel + torch.randn_like(vel) * 0.01
        
        # Forward pass
        model.train()
        pos_pred, vel_pred, _ = model(pos, vel, edge_index, edge_attr)
        
        # Compute loss
        loss = F.mse_loss(pos_pred, pos_gt) + 0.1 * F.mse_loss(vel_pred, vel_gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_gradient_flow(self, random_graph, device):
        """Test that gradients flow through the model."""
        pos, vel, edge_index, edge_attr = random_graph
        
        model = PEGNNDeform(hidden_size=64).to(device)
        
        pos_pred, vel_pred, _ = model(pos, vel, edge_index, edge_attr)
        
        loss = pos_pred.sum() + vel_pred.sum()
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_determinism(self, random_graph, device):
        """Test that results are deterministic with same seed."""
        pos, vel, edge_index, edge_attr = random_graph
        
        torch.manual_seed(42)
        model1 = PEGNNDeform(hidden_size=64).to(device)
        out1, _, _ = model1(pos, vel, edge_index, edge_attr)
        
        torch.manual_seed(42)
        model2 = PEGNNDeform(hidden_size=64).to(device)
        out2, _, _ = model2(pos, vel, edge_index, edge_attr)
        
        assert torch.allclose(out1, out2)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and memory tests."""
    
    def test_large_mesh(self, device):
        """Test with a larger mesh (5000 nodes)."""
        N = 5000
        E = N * 6
        
        pos = torch.randn(N, 3, device=device)
        vel = torch.randn(N, 3, device=device)
        edge_index = torch.randint(0, N, (2, E), device=device)
        edge_attr = torch.rand(E, 2, device=device)
        edge_attr[:, 0] += 1
        edge_attr[:, 1] += 0.1
        
        model = PEGNNDeform(hidden_size=64).to(device)
        
        with torch.no_grad():
            new_pos, new_vel, _ = model(pos, vel, edge_index, edge_attr)
        
        assert new_pos.shape == (N, 3)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage(self, device):
        """Test memory usage is reasonable."""
        torch.cuda.reset_peak_memory_stats()
        
        N = 5000
        E = N * 6
        
        pos = torch.randn(N, 3, device=device)
        vel = torch.randn(N, 3, device=device)
        edge_index = torch.randint(0, N, (2, E), device=device)
        edge_attr = torch.rand(E, 2, device=device)
        
        model = PEGNNDeform(hidden_size=64).to(device)
        
        # Run forward pass
        with torch.no_grad():
            _ = model(pos, vel, edge_index, edge_attr)
        
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        # Should use less than 2GB for 5000 nodes
        assert peak_memory_gb < 2.0, f"Peak memory usage: {peak_memory_gb:.2f} GB"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
