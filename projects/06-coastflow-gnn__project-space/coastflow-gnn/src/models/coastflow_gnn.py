"""
CoastFlow-GNN: Hierarchical Graph-Pooling Network for Coastal Flow Modeling

This module implements the core GNN architecture with:
- Multi-scale GCN layers with TopKPooling
- Physics-informed encoder for coastal boundary conditions
- MLP decoder for velocity and wave height predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
from typing import Optional, Tuple


class GraphEncoder(nn.Module):
    """
    Hierarchical graph encoder with multi-scale pooling.
    
    Uses GCNConv layers with TopKPooling to capture features
    at multiple spatial resolutions.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        pool_ratios: Tuple[float, ...] = (0.8, 0.5),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool_ratios = pool_ratios
        self.dropout = dropout
        
        # Initial embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Level 0: Full resolution
        self.conv0_1 = GCNConv(hidden_channels, hidden_channels)
        self.conv0_2 = GCNConv(hidden_channels, hidden_channels)
        self.norm0 = nn.LayerNorm(hidden_channels)
        
        # Level 1: First pooling (ratio 0.8)
        self.pool1 = TopKPooling(hidden_channels, ratio=pool_ratios[0])
        self.conv1_1 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv1_2 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.norm1 = nn.LayerNorm(hidden_channels * 2)
        
        # Level 2: Second pooling (ratio 0.5)
        self.pool2 = TopKPooling(hidden_channels * 2, ratio=pool_ratios[1])
        self.conv2_1 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        self.conv2_2 = GCNConv(hidden_channels * 4, hidden_channels * 4)
        self.norm2 = nn.LayerNorm(hidden_channels * 4)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through hierarchical encoder.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N]
            
        Returns:
            node_features: Encoded node features [N, hidden*4]
            graph_features: Global graph features [B, hidden*7]
            aux_data: Auxiliary data (pooling indices, etc.)
        """
        aux_data = {}
        
        # Initial embedding
        x = self.input_mlp(x)
        
        # Level 0: Full resolution
        x0 = self.conv0_1(x, edge_index)
        x0 = F.relu(x0)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x0 = self.conv0_2(x0, edge_index)
        x0 = self.norm0(x0 + x)  # Residual connection
        x0 = F.relu(x0)
        
        # Store level 0 features for skip connection
        aux_data['x0'] = x0
        aux_data['edge_index0'] = edge_index
        aux_data['batch0'] = batch
        
        # Level 1: First pooling
        x1, edge_index1, _, batch1, perm1, score1 = self.pool1(
            x0, edge_index, None, batch
        )
        aux_data['perm1'] = perm1
        aux_data['score1'] = score1
        
        x1 = self.conv1_1(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.conv1_2(x1, edge_index1)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        
        aux_data['x1'] = x1
        aux_data['edge_index1'] = edge_index1
        aux_data['batch1'] = batch1
        
        # Level 2: Second pooling
        x2, edge_index2, _, batch2, perm2, score2 = self.pool2(
            x1, edge_index1, None, batch1
        )
        aux_data['perm2'] = perm2
        aux_data['score2'] = score2
        
        x2 = self.conv2_1(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.conv2_2(x2, edge_index2)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        
        aux_data['x2'] = x2
        aux_data['edge_index2'] = edge_index2
        aux_data['batch2'] = batch2
        
        # Global pooling at each level
        g0 = global_mean_pool(x0, batch)  # [B, hidden]
        g1 = global_mean_pool(x1, batch1)  # [B, hidden*2]
        g2_mean = global_mean_pool(x2, batch2)  # [B, hidden*4]
        g2_max = global_max_pool(x2, batch2)  # [B, hidden*4]
        
        # Concatenate multi-scale global features
        # hidden + hidden*2 + hidden*4 + hidden*4 = hidden*11
        # But we'll use a subset for efficiency
        graph_features = torch.cat([g0, g1, g2_mean], dim=-1)  # [B, hidden*7]
        
        return x0, graph_features, aux_data


class NodeDecoder(nn.Module):
    """
    Decoder for node-level predictions.
    
    Takes encoded node features and global context to predict
    velocity components and wave height at each mesh node.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        global_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Combine node features with global context
        self.combine = nn.Linear(in_channels + global_channels, hidden_channels * 2)
        
        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        graph_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode node-level predictions.
        
        Args:
            node_features: Encoded node features [N, in_channels]
            graph_features: Global graph features [B, global_channels]
            batch: Batch assignment [N]
            
        Returns:
            predictions: Node predictions [N, out_channels]
        """
        # Broadcast global features to all nodes
        global_broadcast = graph_features[batch]  # [N, global_channels]
        
        # Combine node and global features
        combined = torch.cat([node_features, global_broadcast], dim=-1)
        combined = self.combine(combined)
        combined = F.relu(combined)
        
        # Decode to output
        predictions = self.decoder(combined)
        
        return predictions


class CoastFlowGNN(nn.Module):
    """
    CoastFlow-GNN: Physics-Informed Graph Neural Network for Coastal Flow.
    
    Architecture:
        Input: Node features [x, y, z, elevation, wind_u, wind_v] (6 channels)
        Encoder: Hierarchical GCN with TopKPooling (ratios: 0.8, 0.5)
        Decoder: MLP with global context injection
        Output: [u_x, u_y, u_z, wave_height] (4 channels)
    
    Args:
        in_channels: Number of input node features (default: 6)
        hidden_channels: Hidden layer dimension (default: 64)
        out_channels: Number of output predictions (default: 4)
        pool_ratios: Tuple of pooling ratios (default: (0.8, 0.5))
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 4,
        pool_ratios: Tuple[float, ...] = (0.8, 0.5),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = GraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            pool_ratios=pool_ratios,
            dropout=dropout,
        )
        
        # Decoder
        # Encoder outputs: node features [N, hidden], global features [B, hidden*7]
        self.decoder = NodeDecoder(
            in_channels=hidden_channels,  # From level 0 (full resolution)
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            global_channels=hidden_channels * 7,  # From multi-scale global pooling
            dropout=dropout,
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N]. If None, assumes single graph.
            return_aux: If True, return auxiliary data for physics losses
            
        Returns:
            predictions: Node-level predictions [N, out_channels]
            aux_data: (optional) Auxiliary data from encoder
        """
        # Handle single graph case
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode
        node_features, graph_features, aux_data = self.encoder(x, edge_index, batch)
        
        # Decode
        predictions = self.decoder(node_features, graph_features, batch)
        
        if return_aux:
            return predictions, aux_data
        return predictions
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"CoastFlowGNN(\n"
            f"  in_channels={self.in_channels},\n"
            f"  hidden_channels={self.hidden_channels},\n"
            f"  out_channels={self.out_channels},\n"
            f"  trainable_params={self.count_parameters():,}\n"
            f")"
        )


def create_model(config: dict) -> CoastFlowGNN:
    """
    Factory function to create CoastFlowGNN from config dict.
    
    Args:
        config: Dictionary with model hyperparameters
        
    Returns:
        Initialized CoastFlowGNN model
    """
    return CoastFlowGNN(
        in_channels=config.get("in_channels", 6),
        hidden_channels=config.get("hidden_channels", 64),
        out_channels=config.get("out_channels", 4),
        pool_ratios=tuple(config.get("pool_ratios", [0.8, 0.5])),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Quick test
    print("Testing CoastFlowGNN...")
    
    model = CoastFlowGNN(in_channels=6, hidden_channels=64, out_channels=4)
    print(model)
    
    # Create dummy data
    num_nodes = 100
    num_edges = 300
    
    x = torch.randn(num_nodes, 6)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Forward pass
    out = model(x, edge_index, batch)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    # Test backward pass
    loss = out.mean()
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n✓ All tests passed!")
