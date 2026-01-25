"""
CrossLevelAttention: Multi-Resolution Graph Attention

Implements cross-attention between graph levels (fine ↔ coarse),
allowing information to flow between different resolutions.

This enables:
- Fine nodes to attend to coarse nodes (get global context)
- Coarse nodes to attend to fine nodes (get local details)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossLevelAttention(nn.Module):
    """
    Cross-attention from one set of node features (query) to another (key/value).
    
    Allows fine-level nodes to attend to coarse-level information (or vice versa).
    Uses scaled dot-product attention with optional multi-head variant.
    
    Args:
        dim: Feature dimension for queries, keys, and values
        num_heads: Number of attention heads (default: 1)
        dropout: Dropout probability for attention weights
        
    Example:
        >>> attn = CrossLevelAttention(dim=64, num_heads=4)
        >>> fine_feat = torch.randn(2, 400, 64)   # (B, N_fine, D)
        >>> coarse_feat = torch.randn(2, 100, 64) # (B, N_coarse, D)
        >>> context = attn(fine_feat, coarse_feat)  # (B, N_fine, D)
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize with Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute cross-attention from query nodes to key/value nodes.
        
        Args:
            query: Query features of shape (B, N_q, D) or (N_q, D)
            key_value: Key/Value features of shape (B, N_kv, D) or (N_kv, D)
            attn_mask: Optional mask of shape (N_q, N_kv) or (B, N_q, N_kv)
            return_attn: Whether to return attention weights
            
        Returns:
            context: Attended features of shape (B, N_q, D)
            attn_weights: Attention weights (optional) of shape (B, H, N_q, N_kv)
        """
        # Handle unbatched input
        unbatched = query.dim() == 2
        if unbatched:
            query = query.unsqueeze(0)
            key_value = key_value.unsqueeze(0)
            
        B, N_q, D = query.shape
        _, N_kv, _ = key_value.shape
        
        # Project to queries, keys, values
        Q = self.q_proj(query)   # (B, N_q, D)
        K = self.k_proj(key_value)  # (B, N_kv, D)
        V = self.v_proj(key_value)  # (B, N_kv, D)
        
        # Reshape for multi-head attention: (B, N, D) -> (B, H, N, head_dim)
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: (B, H, N_q, N_kv)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
            
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (B, H, N_q, head_dim)
        context = torch.matmul(attn_weights, V)
        
        # Reshape back: (B, H, N_q, head_dim) -> (B, N_q, D)
        context = context.transpose(1, 2).contiguous().view(B, N_q, D)
        
        # Final projection
        context = self.out_proj(context)
        
        # Handle unbatched output
        if unbatched:
            context = context.squeeze(0)
            
        if return_attn:
            return context, attn_weights
        return context, None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, num_heads={self.num_heads})"


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between fine and coarse graphs.
    
    Applies:
    1. Fine-to-coarse attention: fine nodes attend to coarse nodes
    2. Coarse-to-fine attention: coarse nodes attend to fine nodes
    
    Returns both updated feature sets.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Fine queries, coarse keys/values
        self.fine_to_coarse = CrossLevelAttention(dim, num_heads, dropout)
        
        # Coarse queries, fine keys/values  
        self.coarse_to_fine = CrossLevelAttention(dim, num_heads, dropout)
        
        # Layer norms
        self.norm_fine = nn.LayerNorm(dim)
        self.norm_coarse = nn.LayerNorm(dim)
        
    def forward(
        self,
        fine_feat: torch.Tensor,
        coarse_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional cross-attention.
        
        Args:
            fine_feat: Fine-level features (B, N_fine, D)
            coarse_feat: Coarse-level features (B, N_coarse, D)
            
        Returns:
            fine_out: Updated fine features with coarse context
            coarse_out: Updated coarse features with fine details
        """
        # Fine attends to coarse (get global context)
        fine_context, _ = self.fine_to_coarse(fine_feat, coarse_feat)
        fine_out = self.norm_fine(fine_feat + fine_context)
        
        # Coarse attends to fine (get local details)
        coarse_context, _ = self.coarse_to_fine(coarse_feat, fine_feat)
        coarse_out = self.norm_coarse(coarse_feat + coarse_context)
        
        return fine_out, coarse_out
