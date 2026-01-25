"""
HGNN_NIF_ClothModel: End-to-End Hybrid Model

Combines the Adaptive Hierarchical GNN with SIREN decoder for
complete cloth simulation with implicit surface reconstruction.

Pipeline:
1. AdaptiveHGNN processes fine + coarse graphs → produces latent
2. SIREN decoder reconstructs SDF from query points + latent
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from .hgnn import AdaptiveHGNN
from .siren import SIRENDecoder


class HGNN_NIF_ClothModel(nn.Module):
    """
    End-to-end hybrid model combining HGNN and Neural Implicit Field.
    
    Takes multi-resolution cloth graphs, processes them with HGNN,
    then uses the learned latent to condition a SIREN decoder for
    SDF reconstruction at arbitrary query points.
    
    Args:
        node_feat_dim: Input node feature dimension (typically 3 for XYZ)
        latent_dim: Dimension of latent code from HGNN
        hidden_dim: Hidden dimension for both HGNN and SIREN
        hgnn_layers: Number of GNN layers in HGNN
        siren_layers: Number of hidden layers in SIREN
        num_heads: Attention heads for cross-level attention
        energy_threshold: Threshold for adaptive resolution
        use_fourier: Whether to use Fourier features in SIREN
        dropout: Dropout probability
        
    Example:
        >>> model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        >>> fine_data = (fine_pos, fine_edges)  # Fine graph
        >>> coarse_data = (coarse_pos, coarse_edges)  # Coarse graph
        >>> query_points = torch.randn(1000, 3)  # Query SDF at these points
        >>> output = model(fine_data, coarse_data, query_points)
        >>> # output['sdf'], output['latent'], output['fine_feat'], etc.
    """
    
    def __init__(
        self,
        node_feat_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        hgnn_layers: int = 2,
        siren_layers: int = 3,
        siren_hidden_dim: int = 128,
        num_heads: int = 4,
        energy_threshold: float = 0.1,
        use_fourier: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Hierarchical GNN for graph processing
        self.hgnn = AdaptiveHGNN(
            in_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=hgnn_layers,
            num_heads=num_heads,
            energy_threshold=energy_threshold,
            dropout=dropout,
            use_cross_attention=True
        )
        
        # Project HGNN output to latent dimension if different
        if hidden_dim != latent_dim:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        else:
            self.latent_proj = nn.Identity()
            
        # SIREN decoder for implicit surface
        self.siren = SIRENDecoder(
            coord_dim=3,
            latent_dim=latent_dim,
            hidden_dim=siren_hidden_dim,
            hidden_layers=siren_layers,
            out_dim=1,
            use_fourier=use_fourier
        )
        
    def forward(
        self,
        fine_graph: Tuple[torch.Tensor, torch.Tensor],
        coarse_graph: Tuple[torch.Tensor, torch.Tensor],
        query_points: Optional[torch.Tensor] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full hybrid model.
        
        Args:
            fine_graph: Tuple of (fine_pos, fine_edges)
                - fine_pos: (B, N_fine, 3) or (N_fine, 3) node positions
                - fine_edges: (2, E_fine) edge indices
            coarse_graph: Tuple of (coarse_pos, coarse_edges)
                - coarse_pos: (B, N_coarse, 3) or (N_coarse, 3) node positions
                - coarse_edges: (2, E_coarse) edge indices
            query_points: Optional (B, M, 3) or (M, 3) points to query SDF
            return_all: If True, return all intermediate outputs
            
        Returns:
            Dict containing:
                - 'fine_feat': Updated fine node features
                - 'coarse_feat': Updated coarse node features
                - 'latent': Global latent code
                - 'energy': Deformation energy estimate
                - 'sdf': SDF values at query points (if provided)
        """
        fine_pos, fine_edges = fine_graph
        coarse_pos, coarse_edges = coarse_graph
        
        # Process through HGNN
        fine_feat, coarse_feat, latent_raw, energy = self.hgnn(
            fine_pos, fine_edges,
            coarse_pos, coarse_edges,
            return_energy=True
        )
        
        # Project to latent space
        latent = self.latent_proj(latent_raw)
        
        # Build output dictionary
        output = {
            'latent': latent,
            'energy': energy
        }
        
        if return_all:
            output['fine_feat'] = fine_feat
            output['coarse_feat'] = coarse_feat
            
        # Query SDF if points are provided
        if query_points is not None:
            sdf = self.siren(query_points, latent)
            output['sdf'] = sdf.squeeze(-1)  # Remove last dim if 1
            
        return output
    
    def encode(
        self,
        fine_graph: Tuple[torch.Tensor, torch.Tensor],
        coarse_graph: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode cloth state to latent code only.
        
        Useful for precomputing latents during inference.
        """
        fine_pos, fine_edges = fine_graph
        coarse_pos, coarse_edges = coarse_graph
        
        _, _, latent_raw = self.hgnn(
            fine_pos, fine_edges,
            coarse_pos, coarse_edges,
            return_energy=False
        )
        
        return self.latent_proj(latent_raw)
    
    def decode(
        self,
        query_points: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode SDF from latent code and query points.
        
        Useful for querying many points after encoding.
        """
        return self.siren(query_points, latent).squeeze(-1)
    
    def compute_sdf_gradient(
        self,
        query_points: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial gradient of SDF (for Eikonal regularization).
        """
        return self.siren.compute_gradient(query_points, latent)
    
    @torch.no_grad()
    def infer(
        self,
        fine_graph: Tuple[torch.Tensor, torch.Tensor],
        coarse_graph: Tuple[torch.Tensor, torch.Tensor],
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference-only forward pass (no gradients).
        """
        self.eval()
        output = self.forward(fine_graph, coarse_graph, query_points, return_all=False)
        return output['sdf']
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"node_feat_dim={self.node_feat_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"hidden_dim={self.hidden_dim})")


class HGNNOnlyModel(nn.Module):
    """
    HGNN-only baseline model (no implicit field decoder).
    
    For ablation study: tests the graph-based component alone.
    Outputs predicted node positions directly from fine graph features.
    """
    
    def __init__(
        self,
        node_feat_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hgnn = AdaptiveHGNN(
            in_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output layer: predict position update
        self.out_layer = nn.Linear(hidden_dim, 3)
        
    def forward(
        self,
        fine_graph: Tuple[torch.Tensor, torch.Tensor],
        coarse_graph: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning predicted positions."""
        fine_pos, fine_edges = fine_graph
        coarse_pos, coarse_edges = coarse_graph
        
        fine_feat, coarse_feat, latent = self.hgnn(
            fine_pos, fine_edges,
            coarse_pos, coarse_edges
        )
        
        # Predict position update
        delta_pos = self.out_layer(fine_feat)
        pred_pos = fine_pos + delta_pos
        
        return {
            'pred_pos': pred_pos,
            'fine_feat': fine_feat,
            'latent': latent
        }


class NIFOnlyModel(nn.Module):
    """
    NIF-only baseline model (no graph structure).
    
    For ablation study: tests the implicit field component alone.
    Uses an auto-decoder approach with learnable per-sample latents.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_fourier: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Direct latent encoder from flattened positions
        self.encoder = nn.Sequential(
            nn.Linear(400 * 3, 512),  # Assuming 400 fine nodes
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        self.siren = SIRENDecoder(
            coord_dim=3,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers,
            use_fourier=use_fourier
        )
        
    def forward(
        self,
        fine_pos: torch.Tensor,
        query_points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using position encoding only (no graph).
        
        Args:
            fine_pos: Fine node positions (B, N, 3)
            query_points: Query points (B, M, 3)
        """
        # Encode positions to latent
        B = fine_pos.shape[0]
        flat_pos = fine_pos.reshape(B, -1)  # (B, N*3)
        latent = self.encoder(flat_pos)      # (B, latent_dim)
        
        # Decode SDF
        sdf = self.siren(query_points, latent)
        
        return {
            'sdf': sdf.squeeze(-1),
            'latent': latent
        }
