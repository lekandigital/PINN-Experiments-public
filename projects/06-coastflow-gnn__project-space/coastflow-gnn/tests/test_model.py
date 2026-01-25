"""
Unit Tests for CoastFlow-GNN Model

Tests:
- Model initialization
- Forward pass with dummy data
- Backward pass (gradient flow)
- Output shape and value ranges
- Model save/load
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coastflow_gnn import CoastFlowGNN, create_model


class TestCoastFlowGNN:
    """Test suite for CoastFlowGNN model."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return CoastFlowGNN(
            in_channels=6,
            hidden_channels=64,
            out_channels=4,
            pool_ratios=(0.8, 0.5),
            dropout=0.1,
        )
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy graph data fixture."""
        num_nodes = 100
        num_edges = 300
        
        x = torch.randn(num_nodes, 6)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        return x, edge_index, batch
    
    @pytest.fixture
    def dummy_batch(self):
        """Create batched graph data fixture."""
        batch_size = 4
        num_nodes_per_graph = 50
        num_edges_per_graph = 150
        
        data_list = []
        for i in range(batch_size):
            x = torch.randn(num_nodes_per_graph, 6)
            edge_index = torch.randint(0, num_nodes_per_graph, (2, num_edges_per_graph))
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        return batch
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert isinstance(model, nn.Module)
        assert model.in_channels == 6
        assert model.hidden_channels == 64
        assert model.out_channels == 4
        print("✓ Model initialization test passed")
    
    def test_model_parameters(self, model):
        """Test model has learnable parameters."""
        num_params = model.count_parameters()
        assert num_params > 0, "Model should have parameters"
        assert num_params < 10_000_000, "Model should be reasonably sized"
        print(f"✓ Model has {num_params:,} parameters")
    
    def test_forward_pass(self, model, dummy_data):
        """Test model forward pass with dummy data."""
        x, edge_index, batch = dummy_data
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        
        assert out.shape == (100, 4), f"Expected (100, 4), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN values"
        assert not torch.isinf(out).any(), "Output contains Inf values"
        print("✓ Forward pass test passed")
    
    def test_forward_pass_batched(self, model, dummy_batch):
        """Test model forward pass with batched data."""
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch.x, dummy_batch.edge_index, dummy_batch.batch)
        
        expected_nodes = 4 * 50  # batch_size * nodes_per_graph
        assert out.shape[0] == expected_nodes, f"Expected {expected_nodes} nodes"
        assert out.shape[1] == 4, "Expected 4 output channels"
        print("✓ Batched forward pass test passed")
    
    def test_backward_pass(self, model, dummy_data):
        """Test backpropagation through model."""
        x, edge_index, batch = dummy_data
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        out = model(x, edge_index, batch)
        loss = out.mean()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "Model should have non-zero gradients"
        
        optimizer.step()
        print("✓ Backward pass test passed")
    
    def test_output_range(self, model, dummy_data):
        """Test output values are in reasonable range."""
        x, edge_index, batch = dummy_data
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        
        # Outputs should be bounded (not exploding)
        assert out.abs().max() < 100, "Output values should be bounded"
        print(f"✓ Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    def test_return_aux_data(self, model, dummy_data):
        """Test model can return auxiliary data."""
        x, edge_index, batch = dummy_data
        
        model.eval()
        with torch.no_grad():
            out, aux = model(x, edge_index, batch, return_aux=True)
        
        assert isinstance(aux, dict), "aux_data should be a dictionary"
        assert 'x0' in aux, "aux_data should contain x0"
        assert 'perm1' in aux, "aux_data should contain pooling indices"
        print("✓ Auxiliary data test passed")
    
    def test_model_repr(self, model):
        """Test model string representation."""
        repr_str = repr(model)
        assert "CoastFlowGNN" in repr_str
        assert "in_channels=6" in repr_str
        print("✓ Model repr test passed")
    
    def test_create_model_factory(self):
        """Test model factory function."""
        config = {
            'in_channels': 6,
            'hidden_channels': 32,
            'out_channels': 4,
            'pool_ratios': [0.8, 0.5],
            'dropout': 0.2,
        }
        
        model = create_model(config)
        assert model.hidden_channels == 32
        assert model.count_parameters() > 0
        print("✓ Factory function test passed")
    
    def test_gpu_forward(self, model, dummy_data):
        """Test forward pass on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        x, edge_index, batch = dummy_data
        
        device = torch.device('cuda')
        model = model.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        
        assert out.device.type == 'cuda'
        assert out.shape == (100, 4)
        print("✓ GPU forward pass test passed")


def test_model_forward():
    """Standalone test for model forward pass."""
    model = CoastFlowGNN(in_channels=6, hidden_channels=64, out_channels=4)
    
    x = torch.randn(100, 6)
    edge_index = torch.randint(0, 100, (2, 300))
    batch = torch.zeros(100, dtype=torch.long)
    
    out = model(x, edge_index, batch)
    
    assert out.shape == (100, 4), f"Expected (100, 4), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"
    print("✓ Model forward pass test passed")


def test_model_backward():
    """Standalone test for backpropagation."""
    model = CoastFlowGNN(in_channels=6, hidden_channels=64, out_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    x = torch.randn(50, 6)
    edge_index = torch.randint(0, 50, (2, 150))
    batch = torch.zeros(50, dtype=torch.long)
    
    out = model(x, edge_index, batch)
    loss = out.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✓ Model backward pass test passed")


if __name__ == "__main__":
    # Run standalone tests
    print("\n" + "=" * 50)
    print("Running CoastFlow-GNN Model Tests")
    print("=" * 50 + "\n")
    
    test_model_forward()
    test_model_backward()
    
    # Run pytest tests
    print("\nRunning pytest suite...")
    pytest.main([__file__, "-v", "--tb=short"])
