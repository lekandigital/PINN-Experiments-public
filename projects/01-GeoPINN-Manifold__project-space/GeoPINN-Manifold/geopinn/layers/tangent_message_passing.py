"""
Tangent Message Passing Layer

Projects neighbor features into each node's local tangent plane before aggregation.
This ensures geometric equivariance on curved manifolds.

Mathematical formulation:
1. Compute tangent basis from vertex normal using Gram-Schmidt
2. Project edge vector v = p_src - p_tgt onto tangent plane: v_tangent = v - (v·n)n
3. Encode local 2D coordinates: (u, w) = (v_tangent·t1, v_tangent·t2)
4. Concatenate with source features and pass through MLP
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def compute_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute orthonormal tangent basis (t1, t2) from surface normals using Gram-Schmidt.
    
    Args:
        normals: [N, 3] unit surface normals
        
    Returns:
        t1: [N, 3] first tangent vector
        t2: [N, 3] second tangent vector (computed as n × t1)
    """
    # Choose arbitrary reference vector not parallel to normal
    ref = torch.zeros_like(normals)
    ref[:, 0] = 1.0  # x-axis
    
    # Handle case where normal is nearly parallel to reference
    parallel_mask = torch.abs(torch.sum(normals * ref, dim=-1)) > 0.9
    ref[parallel_mask, 0] = 0.0
    ref[parallel_mask, 1] = 1.0  # Use y-axis instead
    
    # Gram-Schmidt: t1 = ref - (ref·n)n, then normalize
    dot = torch.sum(ref * normals, dim=-1, keepdim=True)
    t1 = ref - dot * normals
    t1 = t1 / (torch.norm(t1, dim=-1, keepdim=True) + 1e-8)
    
    # t2 = n × t1
    t2 = torch.cross(normals, t1, dim=-1)
    
    return t1, t2


class TangentMessagePassing(nn.Module):
    """
    Message passing layer that operates in local tangent planes.
    
    For each edge (src -> tgt):
    1. Project the edge vector into the target node's tangent plane
    2. Encode the 2D tangent coordinates
    3. Concatenate with source node features
    4. Pass through MLP and aggregate at target
    
    Args:
        in_features: Input feature dimension per node
        out_features: Output feature dimension per node
        hidden_dim: Hidden dimension for message MLP
        aggregation: Aggregation method ('sum', 'mean', 'max')
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        # Message MLP: takes (features + 2D tangent coords) -> hidden
        self.message_mlp = nn.Sequential(
            nn.Linear(in_features + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Update MLP: combines aggregated messages with node features
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )
        
        self.aggregation = aggregation
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        normals: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with tangent plane message passing.
        
        Args:
            node_feats: [N, in_features] node feature matrix
            pos: [N, 3] node positions in 3D
            normals: [N, 3] unit surface normals at each node
            edge_index: [2, E] edge indices (src, tgt)
            
        Returns:
            out_feats: [N, out_features] updated node features
        """
        N = node_feats.size(0)
        device = node_feats.device
        
        # Compute tangent basis for all nodes
        t1, t2 = compute_tangent_basis(normals)  # [N, 3] each
        
        src, tgt = edge_index[0], edge_index[1]
        
        # Compute edge vectors from source to target
        edge_vec = pos[src] - pos[tgt]  # [E, 3]
        
        # Get target node's normal and tangent basis
        n_tgt = normals[tgt]  # [E, 3]
        t1_tgt = t1[tgt]      # [E, 3]
        t2_tgt = t2[tgt]      # [E, 3]
        
        # Project edge vector onto tangent plane at target
        # v_tangent = v - (v·n)n
        normal_component = torch.sum(edge_vec * n_tgt, dim=-1, keepdim=True)
        edge_tangent = edge_vec - normal_component * n_tgt
        
        # Encode as 2D coordinates in tangent plane
        u = torch.sum(edge_tangent * t1_tgt, dim=-1, keepdim=True)  # [E, 1]
        w = torch.sum(edge_tangent * t2_tgt, dim=-1, keepdim=True)  # [E, 1]
        tangent_coords = torch.cat([u, w], dim=-1)  # [E, 2]
        
        # Get source node features
        src_feats = node_feats[src]  # [E, in_features]
        
        # Concatenate features with tangent coordinates
        message_input = torch.cat([src_feats, tangent_coords], dim=-1)  # [E, in_features + 2]
        
        # Compute messages
        messages = self.message_mlp(message_input)  # [E, hidden_dim]
        
        # Aggregate messages at target nodes
        hidden_dim = messages.size(-1)
        aggregated = torch.zeros(N, hidden_dim, device=device)
        
        if self.aggregation == 'sum':
            aggregated.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, hidden_dim), messages)
        elif self.aggregation == 'mean':
            aggregated.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, hidden_dim), messages)
            # Count edges per node for mean
            count = torch.zeros(N, device=device)
            count.scatter_add_(0, tgt, torch.ones(tgt.size(0), device=device))
            count = count.clamp(min=1).unsqueeze(-1)
            aggregated = aggregated / count
        elif self.aggregation == 'max':
            aggregated.scatter_reduce_(0, tgt.unsqueeze(-1).expand(-1, hidden_dim), messages, reduce='amax')
        
        # Update: combine aggregated messages with original features
        update_input = torch.cat([aggregated, node_feats], dim=-1)
        out_feats = self.update_mlp(update_input)
        
        return out_feats


class TangentMessagePassingStack(nn.Module):
    """
    Stack of Tangent Message Passing layers with residual connections.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 3,
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TangentMessagePassing(
                    hidden_features, hidden_features,
                    hidden_dim=hidden_features,
                    aggregation=aggregation
                )
            )
        
        self.output_proj = nn.Linear(hidden_features, out_features)
        self.layer_norm = nn.LayerNorm(hidden_features)
        
    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        normals: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with residual connections."""
        x = self.input_proj(node_feats)
        
        for layer in self.layers:
            residual = x
            x = layer(x, pos, normals, edge_index)
            x = self.layer_norm(x + residual)
        
        return self.output_proj(x)
