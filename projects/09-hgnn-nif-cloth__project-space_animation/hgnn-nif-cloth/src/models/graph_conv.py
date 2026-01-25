"""
GraphConv: Vectorized Graph Convolution Layer

A simple but efficient graph convolution that aggregates neighbor features
using scatter operations (no Python loops over edges).

Supports:
- Batched processing with batch indices
- Optional edge features
- Self-loop handling
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GraphConv(nn.Module):
    """
    Vectorized graph convolution layer that aggregates neighbor features.
    
    Uses scatter_add for efficient message passing without loops.
    Combines self-features and aggregated neighbor features.
    
    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension
        aggr: Aggregation method ('add', 'mean')
        bias: Whether to include bias term
        
    Example:
        >>> conv = GraphConv(64, 128)
        >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> edges = torch.randint(0, 100, (2, 300))  # 300 edges
        >>> out = conv(x, edges)  # (100, 128)
    """
    
    def __init__(
        self, 
        in_feats: int, 
        out_feats: int, 
        aggr: str = 'mean',
        bias: bool = True
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggr = aggr
        
        # Linear transformations for self and neighbor features
        self.lin_self = nn.Linear(in_feats, out_feats, bias=False)
        self.lin_neigh = nn.Linear(in_feats, out_feats, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with vectorized message passing.
        
        Args:
            x: Node features of shape (N, in_feats) or (B, N, in_feats)
            edge_index: Edge indices of shape (2, E) with [source, target]
            edge_weight: Optional edge weights of shape (E,)
            batch: Optional batch indices for each node
            
        Returns:
            Updated node features of shape (N, out_feats) or (B, N, out_feats)
        """
        # Handle batched input by flattening
        batched = x.dim() == 3
        if batched:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            # Adjust edge indices for batched graph
            # Each graph in batch has edges shifted by graph_idx * N
            if edge_index.dim() == 2:
                # Same edges for all graphs in batch
                edge_index = self._expand_edges_for_batch(edge_index, B, N)
        
        num_nodes = x.size(0)
        
        # Transform self features: (N, out_feats)
        h_self = self.lin_self(x)
        
        # Get source and target indices
        src, tgt = edge_index[0], edge_index[1]
        
        # Gather source features: (E, in_feats)
        src_features = x[src]
        
        # Apply edge weights if provided
        if edge_weight is not None:
            src_features = src_features * edge_weight.unsqueeze(-1)
        
        # Aggregate neighbor features using scatter_add
        # h_neigh[i] = sum of features from neighbors of node i
        h_neigh = torch.zeros(num_nodes, self.in_feats, device=x.device, dtype=x.dtype)
        h_neigh.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.in_feats), src_features)
        
        # Normalize if using mean aggregation
        if self.aggr == 'mean':
            # Count number of incoming edges per node
            deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            deg.scatter_add_(0, tgt, torch.ones_like(tgt, dtype=x.dtype))
            deg = deg.clamp(min=1)  # Avoid division by zero
            h_neigh = h_neigh / deg.unsqueeze(-1)
        
        # Transform aggregated neighbor features
        h_neigh = self.lin_neigh(h_neigh)
        
        # Combine self and neighbor features
        out = h_self + h_neigh
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        # Reshape back to batched format if needed
        if batched:
            out = out.reshape(B, N, self.out_feats)
            
        return out
    
    def _expand_edges_for_batch(
        self, 
        edge_index: torch.Tensor, 
        batch_size: int, 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Expand edge indices for batched graphs.
        
        For batched processing, we replicate edges for each graph
        and offset indices appropriately.
        """
        edges_list = []
        for b in range(batch_size):
            offset = b * num_nodes
            edges_list.append(edge_index + offset)
        return torch.cat(edges_list, dim=1)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_feats}, {self.out_feats}, aggr={self.aggr})"


class GraphConvBlock(nn.Module):
    """
    Graph convolution block with normalization and activation.
    
    Applies: GraphConv -> LayerNorm -> Activation -> Dropout
    """
    
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        aggr: str = 'mean',
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.conv = GraphConv(in_feats, out_feats, aggr=aggr)
        self.norm = nn.LayerNorm(out_feats)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with residual connection if dimensions match."""
        h = self.conv(x, edge_index, edge_weight)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Residual connection if dimensions match
        if x.shape[-1] == h.shape[-1]:
            h = h + x
            
        return h
