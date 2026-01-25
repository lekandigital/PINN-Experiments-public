"""
Unit Tests for HGNN-NIF-Cloth Models

Tests individual model components:
- GraphConv
- CrossLevelAttention
- SIRENDecoder
- AdaptiveHGNN
- HGNN_NIF_ClothModel
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.graph_conv import GraphConv, GraphConvBlock
from src.models.attention import CrossLevelAttention, BidirectionalCrossAttention
from src.models.siren import SirenLayer, SIRENDecoder
from src.models.hgnn import AdaptiveHGNN
from src.models.hybrid_model import HGNN_NIF_ClothModel, HGNNOnlyModel, NIFOnlyModel


class TestGraphConv:
    """Tests for GraphConv layer."""
    
    def test_init(self):
        """Test layer initialization."""
        conv = GraphConv(64, 128)
        assert conv.in_feats == 64
        assert conv.out_feats == 128
        
    def test_forward_unbatched(self):
        """Test forward pass with single graph."""
        conv = GraphConv(64, 128)
        x = torch.randn(100, 64)  # 100 nodes
        edges = torch.randint(0, 100, (2, 300))  # 300 edges
        
        out = conv(x, edges)
        assert out.shape == (100, 128)
        
    def test_forward_batched(self):
        """Test forward pass with batched graphs."""
        conv = GraphConv(64, 128)
        x = torch.randn(4, 100, 64)  # 4 graphs, 100 nodes each
        edges = torch.randint(0, 100, (2, 300))
        
        out = conv(x, edges)
        assert out.shape == (4, 100, 128)
        
    def test_with_edge_weights(self):
        """Test with edge weights."""
        conv = GraphConv(64, 128)
        x = torch.randn(100, 64)
        edges = torch.randint(0, 100, (2, 300))
        weights = torch.rand(300)
        
        out = conv(x, edges, edge_weight=weights)
        assert out.shape == (100, 128)
        
    def test_mean_aggregation(self):
        """Test mean aggregation mode."""
        conv = GraphConv(64, 64, aggr='mean')
        x = torch.ones(10, 64)
        # Complete graph edges
        edges = torch.tensor([[i, j] for i in range(10) for j in range(10) if i != j]).T
        
        out = conv(x, edges)
        assert out.shape == (10, 64)


class TestCrossLevelAttention:
    """Tests for CrossLevelAttention layer."""
    
    def test_init(self):
        """Test layer initialization."""
        attn = CrossLevelAttention(dim=64, num_heads=4)
        assert attn.dim == 64
        assert attn.num_heads == 4
        
    def test_forward_unbatched(self):
        """Test with unbatched input."""
        attn = CrossLevelAttention(dim=64, num_heads=4)
        query = torch.randn(400, 64)
        key_value = torch.randn(100, 64)
        
        out, _ = attn(query, key_value)
        assert out.shape == (400, 64)
        
    def test_forward_batched(self):
        """Test with batched input."""
        attn = CrossLevelAttention(dim=64, num_heads=4)
        query = torch.randn(2, 400, 64)
        key_value = torch.randn(2, 100, 64)
        
        out, _ = attn(query, key_value)
        assert out.shape == (2, 400, 64)
        
    def test_return_attention(self):
        """Test returning attention weights."""
        attn = CrossLevelAttention(dim=64, num_heads=4)
        query = torch.randn(2, 400, 64)
        key_value = torch.randn(2, 100, 64)
        
        out, weights = attn(query, key_value, return_attn=True)
        assert weights.shape == (2, 4, 400, 100)  # (B, H, N_q, N_kv)


class TestBidirectionalAttention:
    """Tests for BidirectionalCrossAttention."""
    
    def test_forward(self):
        """Test bidirectional attention."""
        attn = BidirectionalCrossAttention(dim=64, num_heads=4)
        fine = torch.randn(2, 400, 64)
        coarse = torch.randn(2, 100, 64)
        
        fine_out, coarse_out = attn(fine, coarse)
        assert fine_out.shape == (2, 400, 64)
        assert coarse_out.shape == (2, 100, 64)


class TestSIREN:
    """Tests for SIREN layers and decoder."""
    
    def test_siren_layer(self):
        """Test single SIREN layer."""
        layer = SirenLayer(3, 256, is_first=True, w0=30)
        x = torch.randn(1000, 3)
        out = layer(x)
        assert out.shape == (1000, 256)
        # Output should be bounded by sine
        assert out.min() >= -1 and out.max() <= 1
        
    def test_siren_decoder_unbatched(self):
        """Test SIREN decoder without batch."""
        decoder = SIRENDecoder(coord_dim=3, latent_dim=64)
        coords = torch.randn(1000, 3)
        latent = torch.randn(64)
        
        sdf = decoder(coords, latent)
        assert sdf.shape == (1000, 1)
        
    def test_siren_decoder_batched(self):
        """Test SIREN decoder with batch."""
        decoder = SIRENDecoder(coord_dim=3, latent_dim=64)
        coords = torch.randn(2, 1000, 3)
        latent = torch.randn(2, 64)
        
        sdf = decoder(coords, latent)
        assert sdf.shape == (2, 1000, 1)
        
    def test_siren_with_fourier(self):
        """Test SIREN with Fourier features."""
        decoder = SIRENDecoder(coord_dim=3, latent_dim=64, use_fourier=True)
        coords = torch.randn(100, 3)
        latent = torch.randn(64)
        
        sdf = decoder(coords, latent)
        assert sdf.shape == (100, 1)


class TestAdaptiveHGNN:
    """Tests for AdaptiveHGNN."""
    
    def test_init(self):
        """Test initialization."""
        hgnn = AdaptiveHGNN(in_dim=3, hidden_dim=64)
        assert hgnn.in_dim == 3
        assert hgnn.hidden_dim == 64
        
    def test_forward_unbatched(self):
        """Test forward with single sample."""
        hgnn = AdaptiveHGNN(in_dim=3, hidden_dim=64)
        fine_x = torch.randn(400, 3)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_x = torch.randn(100, 3)
        coarse_edges = torch.randint(0, 100, (2, 80))
        
        fine_out, coarse_out, latent = hgnn(fine_x, fine_edges, coarse_x, coarse_edges)
        
        assert fine_out.shape == (400, 64)
        assert coarse_out.shape == (100, 64)
        assert latent.shape == (64,)
        
    def test_forward_batched(self):
        """Test forward with batch."""
        hgnn = AdaptiveHGNN(in_dim=3, hidden_dim=64)
        fine_x = torch.randn(2, 400, 3)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_x = torch.randn(2, 100, 3)
        coarse_edges = torch.randint(0, 100, (2, 80))
        
        fine_out, coarse_out, latent = hgnn(fine_x, fine_edges, coarse_x, coarse_edges)
        
        assert fine_out.shape == (2, 400, 64)
        assert coarse_out.shape == (2, 100, 64)
        assert latent.shape == (2, 64)
        
    def test_energy_computation(self):
        """Test spring energy computation."""
        hgnn = AdaptiveHGNN(in_dim=3, hidden_dim=64)
        positions = torch.randn(2, 100, 3)
        edges = torch.randint(0, 100, (2, 80))
        
        energy = hgnn.compute_spring_energy(positions, edges)
        assert energy.shape == (2,)
        assert (energy >= 0).all()


class TestHybridModel:
    """Tests for full HGNN_NIF_ClothModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        assert model.node_feat_dim == 3
        assert model.latent_dim == 64
        
    def test_forward_no_query(self):
        """Test forward without query points."""
        model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        
        fine_pos = torch.randn(2, 400, 3)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_pos = torch.randn(2, 100, 3)
        coarse_edges = torch.randint(0, 100, (2, 80))
        
        output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges))
        
        assert 'latent' in output
        assert 'energy' in output
        assert output['latent'].shape == (2, 64)
        assert 'sdf' not in output
        
    def test_forward_with_query(self):
        """Test forward with query points."""
        model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        
        fine_pos = torch.randn(2, 400, 3)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_pos = torch.randn(2, 100, 3)
        coarse_edges = torch.randint(0, 100, (2, 80))
        query_points = torch.randn(2, 1000, 3)
        
        output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
        
        assert 'sdf' in output
        assert output['sdf'].shape == (2, 1000)
        
    def test_encode_decode(self):
        """Test separate encode and decode."""
        model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        
        fine_pos = torch.randn(1, 400, 3)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_pos = torch.randn(1, 100, 3)
        coarse_edges = torch.randint(0, 100, (2, 80))
        
        latent = model.encode((fine_pos, fine_edges), (coarse_pos, coarse_edges))
        assert latent.shape == (1, 64)
        
        query_points = torch.randn(1, 500, 3)
        sdf = model.decode(query_points, latent)
        assert sdf.shape == (1, 500)
        
    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        
        fine_pos = torch.randn(1, 400, 3, requires_grad=True)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_pos = torch.randn(1, 100, 3, requires_grad=True)
        coarse_edges = torch.randint(0, 100, (2, 80))
        query_points = torch.randn(1, 100, 3)
        
        output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
        
        loss = output['sdf'].mean()
        loss.backward()
        
        # Check gradients exist
        assert fine_pos.grad is not None
        assert coarse_pos.grad is not None


class TestAblationModels:
    """Tests for ablation baseline models."""
    
    def test_hgnn_only(self):
        """Test HGNN-only model."""
        model = HGNNOnlyModel(node_feat_dim=3, hidden_dim=64)
        
        fine_pos = torch.randn(2, 400, 3)
        fine_edges = torch.randint(0, 400, (2, 300))
        coarse_pos = torch.randn(2, 100, 3)
        coarse_edges = torch.randint(0, 100, (2, 80))
        
        output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges))
        
        assert 'pred_pos' in output
        assert output['pred_pos'].shape == (2, 400, 3)
        
    def test_nif_only(self):
        """Test NIF-only model."""
        model = NIFOnlyModel(latent_dim=64, hidden_dim=128)
        
        fine_pos = torch.randn(2, 400, 3)
        query_points = torch.randn(2, 1000, 3)
        
        output = model(fine_pos, query_points)
        
        assert 'sdf' in output
        assert output['sdf'].shape == (2, 1000)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
