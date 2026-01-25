"""
AdaptiveHGNN: Adaptive Hierarchical Graph Neural Network

A multi-resolution GNN that operates on fine and coarse graph representations
of cloth, with energy-based adaptive resolution control.

Key features:
- Processes both fine and coarse graphs simultaneously
- Cross-level attention for information exchange between resolutions
- Energy-based gating to adaptively use fine vs coarse features
- Global pooling to produce latent code for SIREN decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .graph_conv import GraphConv, GraphConvBlock
from .attention import CrossLevelAttention, BidirectionalCrossAttention


class AdaptiveHGNN(nn.Module):
    """
    Adaptive Hierarchical Graph Neural Network for cloth simulation.
    
    Processes multi-resolution cloth graphs (fine + coarse) with:
    1. Per-level graph convolutions
    2. Cross-level attention for multi-scale information flow
    3. Energy-based resolution adaptation
    4. Global pooling for latent code generation
    
    Args:
        in_dim: Input node feature dimension (typically 3 for XYZ coords)
        hidden_dim: Hidden feature dimension
        num_layers: Number of GNN layers per resolution level
        num_heads: Attention heads for cross-level attention
        energy_threshold: Threshold for adaptive resolution switching
        dropout: Dropout probability
        
    Example:
        >>> hgnn = AdaptiveHGNN(in_dim=3, hidden_dim=64)
        >>> fine_x = torch.randn(2, 400, 3)    # (B, N_fine, 3)
        >>> fine_edges = torch.randint(0, 400, (2, 300))
        >>> coarse_x = torch.randn(2, 100, 3)  # (B, N_coarse, 3)
        >>> coarse_edges = torch.randint(0, 100, (2, 80))
        >>> fine_out, coarse_out, latent = hgnn(fine_x, fine_edges, coarse_x, coarse_edges)
    """
    
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        energy_threshold: float = 0.1,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.energy_threshold = energy_threshold
        self.use_cross_attention = use_cross_attention
        
        # Input embedding layers
        self.fine_embed = nn.Linear(in_dim, hidden_dim)
        self.coarse_embed = nn.Linear(in_dim, hidden_dim)
        
        # Fine-level GNN layers
        self.fine_convs = nn.ModuleList([
            GraphConvBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Coarse-level GNN layers
        self.coarse_convs = nn.ModuleList([
            GraphConvBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-level attention (optional)
        if use_cross_attention:
            self.cross_attention = BidirectionalCrossAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout
            )
        else:
            self.cross_attention = None
            
        # Global pooling for latent generation
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable pooling weights
        self.pool_attention = nn.Linear(hidden_dim, 1)
        
        # Energy estimation network (for adaptive resolution)
        self.energy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_spring_energy(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute normalized spring energy from current positions.
        
        High energy indicates large deformations (need fine resolution).
        Low energy indicates near-rest state (can use coarse resolution).
        
        Args:
            positions: Node positions (B, N, 3) or (N, 3)
            edge_index: Edge indices (2, E)
            rest_lengths: Optional rest lengths per edge (E,)
            
        Returns:
            Normalized energy scalar per sample
        """
        batched = positions.dim() == 3
        if not batched:
            positions = positions.unsqueeze(0)
            
        B, N, _ = positions.shape
        src, tgt = edge_index[0], edge_index[1]
        
        # Current edge lengths
        pos_flat = positions.reshape(B * N, 3)
        
        # Handle batched edge computation
        curr_lengths_list = []
        for b in range(B):
            pos_b = positions[b]  # (N, 3)
            diff = pos_b[src] - pos_b[tgt]  # (E, 3)
            curr_lengths = torch.norm(diff, dim=-1)  # (E,)
            curr_lengths_list.append(curr_lengths)
            
        curr_lengths = torch.stack(curr_lengths_list, dim=0)  # (B, E)
        
        # Compute rest lengths if not provided (from first batch)
        if rest_lengths is None:
            rest_lengths = curr_lengths[0].detach()  # Use current as rest
            
        # Spring energy: sum of squared stretch
        stretch = curr_lengths - rest_lengths.unsqueeze(0)
        energy = (stretch ** 2).mean(dim=-1)  # (B,)
        
        # Normalize to [0, 1] range (heuristic normalization)
        energy = torch.tanh(energy * 10)
        
        if not batched:
            energy = energy.squeeze(0)
            
        return energy
    
    def forward(
        self,
        fine_x: torch.Tensor,
        fine_edges: torch.Tensor,
        coarse_x: torch.Tensor,
        coarse_edges: torch.Tensor,
        return_energy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical GNN.
        
        Args:
            fine_x: Fine node features (B, N_fine, in_dim) or (N_fine, in_dim)
            fine_edges: Fine edge indices (2, E_fine)
            coarse_x: Coarse node features (B, N_coarse, in_dim) or (N_coarse, in_dim)
            coarse_edges: Coarse edge indices (2, E_coarse)
            return_energy: Whether to return energy values
            
        Returns:
            fine_out: Updated fine node features
            coarse_out: Updated coarse node features
            latent: Global latent code for SIREN conditioning
        """
        # Handle unbatched input
        unbatched = fine_x.dim() == 2
        if unbatched:
            fine_x = fine_x.unsqueeze(0)
            coarse_x = coarse_x.unsqueeze(0)
            
        B, N_fine, _ = fine_x.shape
        _, N_coarse, _ = coarse_x.shape
        
        # Store original positions for energy computation
        fine_pos = fine_x[..., :3]  # Assume first 3 dims are XYZ
        
        # Embed inputs
        fine_h = self.fine_embed(fine_x)    # (B, N_fine, hidden_dim)
        coarse_h = self.coarse_embed(coarse_x)  # (B, N_coarse, hidden_dim)
        
        # First layer of GNN
        fine_h = self.fine_convs[0](fine_h.reshape(-1, self.hidden_dim), fine_edges)
        fine_h = fine_h.reshape(B, N_fine, self.hidden_dim)
        
        coarse_h = self.coarse_convs[0](coarse_h.reshape(-1, self.hidden_dim), coarse_edges)
        coarse_h = coarse_h.reshape(B, N_coarse, self.hidden_dim)
        
        # Cross-level attention (after first layer)
        if self.cross_attention is not None:
            fine_h, coarse_h = self.cross_attention(fine_h, coarse_h)
            
        # Compute energy for adaptive resolution
        energy = self.compute_spring_energy(fine_pos, fine_edges)
        
        # Second layer of GNN (conditional on energy)
        fine_h2 = self.fine_convs[1](fine_h.reshape(-1, self.hidden_dim), fine_edges)
        fine_h2 = fine_h2.reshape(B, N_fine, self.hidden_dim)
        
        coarse_h2 = self.coarse_convs[1](coarse_h.reshape(-1, self.hidden_dim), coarse_edges)
        coarse_h2 = coarse_h2.reshape(B, N_coarse, self.hidden_dim)
        
        # Energy-based adaptive resolution
        # If energy is low (cloth at rest), rely more on coarse features
        # If energy is high (cloth deforming), use fine features
        if isinstance(energy, torch.Tensor) and energy.dim() == 0:
            energy = energy.unsqueeze(0)
        
        # Expand energy for broadcasting: (B,) -> (B, 1, 1)
        energy_weight = energy.reshape(B, 1, 1)
        
        # Adaptive blending (smooth gating)
        fine_out = fine_h + energy_weight * (fine_h2 - fine_h)
        coarse_out = coarse_h2  # Always use full coarse update
        
        # Apply remaining layers if any
        for i in range(2, self.num_layers):
            fine_out = self.fine_convs[i](fine_out.reshape(-1, self.hidden_dim), fine_edges)
            fine_out = fine_out.reshape(B, N_fine, self.hidden_dim)
            
            coarse_out = self.coarse_convs[i](coarse_out.reshape(-1, self.hidden_dim), coarse_edges)
            coarse_out = coarse_out.reshape(B, N_coarse, self.hidden_dim)
        
        # Global pooling from coarse graph to get latent
        # Use attention-weighted pooling
        pool_weights = F.softmax(self.pool_attention(coarse_out), dim=1)  # (B, N_coarse, 1)
        pooled = (coarse_out * pool_weights).sum(dim=1)  # (B, hidden_dim)
        latent = self.global_pool(pooled)  # (B, hidden_dim)
        
        # Handle unbatched output
        if unbatched:
            fine_out = fine_out.squeeze(0)
            coarse_out = coarse_out.squeeze(0)
            latent = latent.squeeze(0)
            energy = energy.squeeze(0)
            
        if return_energy:
            return fine_out, coarse_out, latent, energy
        return fine_out, coarse_out, latent
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in_dim={self.in_dim}, "
                f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers})")
