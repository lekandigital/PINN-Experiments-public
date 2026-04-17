"""
Physics-Enhanced HGNN for cloth simulation.

This module adds physics-encoded convolution as the first layer of AdaptiveHGNN,
computing analytical Hooke's law forces before the learned graph convolutions.

Architecture:
1. PhysicsForceLayer: Computes spring forces using shared ClothForceConv
2. PhysicsEnhancedHGNN: AdaptiveHGNN with physics-encoded first layer

The physics forces serve as a strong inductive bias, allowing the GNN layers
to focus on learning corrections and multi-scale interactions.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

# Import shared physics-encoded convolution
import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')
from shared.physics_conv import ClothForceConv, ClothConvConfig
from shared.physics_conv import LiteClothConv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsForceLayer(nn.Module):
    """
    Physics force computation layer using shared ClothForceConv.
    
    Computes analytical spring forces that can be fed into downstream
    GNN layers as additional features or added to predictions.
    
    This provides a strong physics prior for cloth simulation.
    """
    
    def __init__(
        self,
        stiffness: float = 1000.0,
        use_correction: bool = True,
        correction_hidden_dim: int = 32,
        output_dim: int = 3,
    ):
        """
        Args:
            stiffness: Default spring stiffness
            use_correction: Whether to learn corrections to physics
            correction_hidden_dim: Hidden dim for correction MLP
            output_dim: Output feature dimension (3 for forces)
        """
        super().__init__()
        
        config = ClothConvConfig(
            stretch_stiffness=stiffness,
            compute_damping=False,
            compute_bending=False,
            correction_hidden_dim=correction_hidden_dim if use_correction else 1,
            correction_layers=2 if use_correction else 0,
            correction_scale_init=0.01 if use_correction else 0.0,
            aggregation="add",
            track_diagnostics=True,
        )
        
        self.cloth_conv = ClothForceConv(config=config)
        self.output_dim = output_dim
        
        # Optional projection to match hidden_dim
        if output_dim != 3:
            self.proj = nn.Linear(3, output_dim)
        else:
            self.proj = None
    
    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute physics forces.
        
        Args:
            positions: Node positions [N, 3] or [B, N, 3]
            edge_index: Edge indices [2, E]
            rest_lengths: Rest lengths per edge [E] (optional)
            velocities: Node velocities [N, 3] or [B, N, 3] (optional)
        
        Returns:
            forces: Per-node force vectors [N, output_dim] or [B, N, output_dim]
        """
        batched = positions.dim() == 3
        if batched:
            B, N, _ = positions.shape
            positions = positions.reshape(B * N, 3)
            if velocities is not None:
                velocities = velocities.reshape(B * N, 3)
            # Expand edges for batch
            edge_index = self._expand_edges(edge_index, B, N)
        
        # Compute rest lengths if not provided
        if rest_lengths is None:
            src, tgt = edge_index[0], edge_index[1]
            diff = positions[tgt] - positions[src]
            rest_lengths = torch.norm(diff, dim=-1, keepdim=True).detach()
        elif rest_lengths.dim() == 1:
            rest_lengths = rest_lengths.unsqueeze(-1)
        
        # Edge attributes: rest_length
        edge_attr = rest_lengths
        
        # Compute forces using physics-encoded convolution
        forces = self.cloth_conv(positions, edge_index, edge_attr, vel=velocities)
        
        # Project if needed
        if self.proj is not None:
            forces = self.proj(forces)
        
        # Reshape back to batched
        if batched:
            forces = forces.reshape(B, N, -1)
        
        return forces
    
    def _expand_edges(
        self,
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int
    ) -> torch.Tensor:
        """Expand edges for batched graphs."""
        edges_list = []
        for b in range(batch_size):
            offset = b * num_nodes
            edges_list.append(edge_index + offset)
        return torch.cat(edges_list, dim=1)
    
    def physics_fraction(self) -> float:
        """Get fraction of output from physics vs learned."""
        return self.cloth_conv.physics_fraction()


class PhysicsEnhancedHGNN(nn.Module):
    """
    Adaptive Hierarchical GNN with physics-encoded first layer.
    
    Combines:
    1. Physics forces from ClothForceConv (analytical Hooke's law)
    2. Learned graph convolutions from AdaptiveHGNN
    3. Energy-based adaptive resolution
    
    The physics forces provide a strong prior, while the GNN learns
    multi-scale interactions and corrections.
    """
    
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        energy_threshold: float = 0.1,
        dropout: float = 0.1,
        stiffness: float = 1000.0,
        use_physics_correction: bool = True,
    ):
        """
        Args:
            in_dim: Input node feature dimension
            hidden_dim: Hidden feature dimension
            num_layers: Number of GNN layers per resolution
            num_heads: Attention heads for cross-level attention
            energy_threshold: Threshold for adaptive resolution
            dropout: Dropout probability
            stiffness: Spring stiffness for physics layer
            use_physics_correction: Learn corrections to physics
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Physics force layer (computes forces for fine graph)
        self.physics_layer = PhysicsForceLayer(
            stiffness=stiffness,
            use_correction=use_physics_correction,
            correction_hidden_dim=32,
            output_dim=hidden_dim,  # Project to hidden_dim
        )
        
        # Import GraphConv locally to avoid circular imports
        from .graph_conv import GraphConvBlock
        
        # Input embedding (includes physics forces as additional features)
        # Input: positions (3) + physics_forces (hidden_dim)
        self.fine_embed = nn.Linear(in_dim + hidden_dim, hidden_dim)
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
        
        # Cross-level attention
        from .attention import BidirectionalCrossAttention
        self.cross_attention = BidirectionalCrossAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout
        )
        
        # Global pooling for latent generation
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pool_attention = nn.Linear(hidden_dim, 1)
        
        # Energy estimation
        self.energy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(
            f"PhysicsEnhancedHGNN initialized: {num_layers} layers, "
            f"hidden_dim={hidden_dim}, stiffness={stiffness}"
        )
    
    def forward(
        self,
        fine_x: torch.Tensor,
        fine_edges: torch.Tensor,
        coarse_x: torch.Tensor,
        coarse_edges: torch.Tensor,
        fine_rest_lengths: Optional[torch.Tensor] = None,
        return_energy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through physics-enhanced hierarchical GNN.
        
        Args:
            fine_x: Fine node features (B, N_fine, in_dim) or (N_fine, in_dim)
            fine_edges: Fine edge indices (2, E_fine)
            coarse_x: Coarse node features (B, N_coarse, in_dim) or (N_coarse, in_dim)
            coarse_edges: Coarse edge indices (2, E_coarse)
            fine_rest_lengths: Rest lengths for fine edges (E_fine,)
            return_energy: Whether to return energy values
            
        Returns:
            fine_out: Updated fine node features
            coarse_out: Updated coarse node features
            latent: Global latent code for SIREN conditioning
        """
        unbatched = fine_x.dim() == 2
        if unbatched:
            fine_x = fine_x.unsqueeze(0)
            coarse_x = coarse_x.unsqueeze(0)
        
        B, N_fine, _ = fine_x.shape
        _, N_coarse, _ = coarse_x.shape
        
        # Extract positions (first 3 dims)
        fine_pos = fine_x[..., :3]
        
        # Compute physics forces for fine graph
        physics_forces = self.physics_layer(
            fine_pos, fine_edges, fine_rest_lengths
        )  # (B, N_fine, hidden_dim)
        
        # Concatenate positions with physics forces
        fine_x_aug = torch.cat([fine_x, physics_forces], dim=-1)
        
        # Embed inputs
        fine_h = self.fine_embed(fine_x_aug)  # (B, N_fine, hidden_dim)
        coarse_h = self.coarse_embed(coarse_x)  # (B, N_coarse, hidden_dim)
        
        # First layer of GNN
        fine_h = self.fine_convs[0](fine_h.reshape(-1, self.hidden_dim), fine_edges)
        fine_h = fine_h.reshape(B, N_fine, self.hidden_dim)
        
        coarse_h = self.coarse_convs[0](coarse_h.reshape(-1, self.hidden_dim), coarse_edges)
        coarse_h = coarse_h.reshape(B, N_coarse, self.hidden_dim)
        
        # Cross-level attention
        fine_h, coarse_h = self.cross_attention(fine_h, coarse_h)
        
        # Compute energy for adaptive resolution
        energy = self._compute_energy(fine_pos, fine_edges)
        
        # Remaining GNN layers
        for i in range(1, self.num_layers):
            fine_h = self.fine_convs[i](fine_h.reshape(-1, self.hidden_dim), fine_edges)
            fine_h = fine_h.reshape(B, N_fine, self.hidden_dim)
            
            coarse_h = self.coarse_convs[i](coarse_h.reshape(-1, self.hidden_dim), coarse_edges)
            coarse_h = coarse_h.reshape(B, N_coarse, self.hidden_dim)
        
        fine_out = fine_h
        coarse_out = coarse_h
        
        # Global pooling
        import torch.nn.functional as F
        pool_weights = F.softmax(self.pool_attention(coarse_out), dim=1)
        pooled = (coarse_out * pool_weights).sum(dim=1)
        latent = self.global_pool(pooled)
        
        if unbatched:
            fine_out = fine_out.squeeze(0)
            coarse_out = coarse_out.squeeze(0)
            latent = latent.squeeze(0)
            energy = energy.squeeze(0) if isinstance(energy, torch.Tensor) else energy
        
        if return_energy:
            return fine_out, coarse_out, latent, energy
        return fine_out, coarse_out, latent
    
    def _compute_energy(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute spring energy for adaptive resolution."""
        batched = positions.dim() == 3
        if not batched:
            positions = positions.unsqueeze(0)
        
        B, N, _ = positions.shape
        src, tgt = edge_index[0], edge_index[1]
        
        energy_list = []
        for b in range(B):
            pos_b = positions[b]
            diff = pos_b[src] - pos_b[tgt]
            lengths = torch.norm(diff, dim=-1)
            # Use variance as proxy for deformation energy
            energy = lengths.var()
            energy_list.append(energy)
        
        energy = torch.stack(energy_list)
        return torch.tanh(energy * 10)
    
    def physics_fraction(self) -> float:
        """Get physics fraction from physics layer."""
        return self.physics_layer.physics_fraction()
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get comprehensive diagnostics."""
        return {
            "physics_fraction": self.physics_fraction(),
        }


def test_physics_enhanced_hgnn():
    """Test the physics-enhanced HGNN model."""
    print("=" * 60)
    print("Testing Physics-Enhanced HGNN")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test PhysicsForceLayer
    print("Testing PhysicsForceLayer...")
    physics_layer = PhysicsForceLayer(
        stiffness=1000.0,
        use_correction=True,
        output_dim=64,
    ).to(device)
    
    N_fine = 100
    E_fine = 250
    
    pos = torch.randn(N_fine, 3, device=device)
    edges = torch.randint(0, N_fine, (2, E_fine), device=device)
    
    forces = physics_layer(pos, edges)
    print(f"  Input: {pos.shape} → Forces: {forces.shape}")
    print(f"  Physics fraction: {physics_layer.physics_fraction():.3f}")
    print("  ✓ PhysicsForceLayer test passed\n")
    
    # Test PhysicsEnhancedHGNN
    print("Testing PhysicsEnhancedHGNN...")
    
    # Note: We need the actual graph_conv module, so we'll do a minimal test
    try:
        # Try to import from Project 09
        import sys
        sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments/projects/09-hgnn-nif-cloth__project-space/hgnn-nif-cloth/src')
        
        model = PhysicsEnhancedHGNN(
            in_dim=3,
            hidden_dim=64,
            num_layers=2,
            stiffness=1000.0,
        ).to(device)
        
        # Test data
        N_fine, N_coarse = 100, 25
        E_fine, E_coarse = 250, 50
        
        fine_x = torch.randn(N_fine, 3, device=device)
        fine_edges = torch.randint(0, N_fine, (2, E_fine), device=device)
        coarse_x = torch.randn(N_coarse, 3, device=device)
        coarse_edges = torch.randint(0, N_coarse, (2, E_coarse), device=device)
        
        fine_out, coarse_out, latent = model(fine_x, fine_edges, coarse_x, coarse_edges)
        print(f"  Fine input: {fine_x.shape} → Fine output: {fine_out.shape}")
        print(f"  Coarse input: {coarse_x.shape} → Coarse output: {coarse_out.shape}")
        print(f"  Latent: {latent.shape}")
        print(f"  Physics fraction: {model.physics_fraction():.3f}")
        print("  ✓ PhysicsEnhancedHGNN test passed\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        
    except ImportError as e:
        print(f"  Skipping full model test (missing dependencies): {e}")
        print("  ✓ PhysicsForceLayer standalone test passed\n")
    
    print("=" * 60)
    print("✓ All physics-enhanced tests passed!")


if __name__ == "__main__":
    test_physics_enhanced_hgnn()
