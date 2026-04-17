"""
ClothGNN with Physics-Encoded Forces.

This module enhances ClothGNN with physics-encoded spring forces using the
shared LiteClothConv layer. The physics forces provide a strong prior while
the GNN learns corrections and complex interactions.

Architecture:
1. LiteClothConv: Computes analytical Hooke's law forces (low parameter count)
2. Original ClothEncoder: Processes node features with learned message passing
3. GRU: Combines physics forces with learned features for temporal dynamics
4. Decoder: Outputs displacement predictions

This keeps the model lightweight (~70K params) while adding physics inductive bias.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Tuple, List, Dict

# Import shared physics-encoded convolution
import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')
from shared.physics_conv import LiteClothConv, PurePhysicsClothConv
from shared.physics_conv.lightweight_conv import LiteClothConvConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsEncoder(nn.Module):
    """
    Physics-based encoder using LiteClothConv.
    
    Computes analytical spring forces and projects them to hidden dimension.
    This has very few learnable parameters (~200) while providing strong physics.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        stiffness: float = 1000.0,
        use_correction: bool = True,
    ):
        super().__init__()
        
        # Lightweight physics convolution
        config = LiteClothConvConfig(
            stretch_stiffness=stiffness,
            correction_hidden_dim=16 if use_correction else 1,
            correction_layers=1 if use_correction else 0,
            correction_scale_init=0.01 if use_correction else 0.0,
            track_diagnostics=True,
        )
        
        self.physics_conv = LiteClothConv(config=config)
        
        # Project forces to hidden dimension
        self.proj = nn.Linear(3, hidden_channels)
        
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute physics forces and project to hidden dimension.
        
        Args:
            pos: Node positions (N, 3)
            edge_index: Edge indices (2, E)
            rest_lengths: Rest lengths per edge (E,) - computed if not provided
        
        Returns:
            forces_hidden: Physics forces projected to hidden dim (N, hidden)
        """
        # Compute rest lengths if not provided
        if rest_lengths is None:
            src, tgt = edge_index[0], edge_index[1]
            diff = pos[tgt] - pos[src]
            rest_lengths = torch.norm(diff, dim=-1, keepdim=True).detach()
        elif rest_lengths.dim() == 1:
            rest_lengths = rest_lengths.unsqueeze(-1)
        
        # Compute physics forces
        forces = self.physics_conv(pos, edge_index, rest_lengths)
        
        # Project to hidden dimension
        return self.proj(forces)
    
    def physics_fraction(self) -> float:
        """Get fraction of output from physics vs learned."""
        return self.physics_conv.physics_fraction()


class ClothEncoderPhysics(MessagePassing):
    """
    GNN encoder that incorporates physics forces into message passing.
    
    Combines learned message passing with physics-encoded forces for
    stronger inductive bias.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stiffness: float = 1000.0,
    ):
        super().__init__(aggr="mean")
        
        # Node feature projection
        self.lin_node = nn.Linear(in_channels, hidden_channels)
        
        # Edge feature projection
        self.lin_edge = nn.Linear(4, hidden_channels)
        
        # Physics encoder (adds physics forces as additional features)
        self.physics_encoder = PhysicsEncoder(
            hidden_channels=hidden_channels,
            stiffness=stiffness,
            use_correction=True,
        )
        
        # Message MLP (now includes physics forces)
        self.lin_msg = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),  # +1 for physics
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with physics-enhanced encoding.
        
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connectivity
            pos: (N, 3) node positions
            rest_lengths: (E,) optional rest lengths
        
        Returns:
            h: (N, hidden_channels) encoded node embeddings
        """
        # Project node features
        h = self.lin_node(x)
        
        # Compute physics forces
        physics_h = self.physics_encoder(pos, edge_index, rest_lengths)
        
        # Compute edge features
        row, col = edge_index
        rel_pos = pos[col] - pos[row]
        dist = rel_pos.norm(dim=1, keepdim=True)
        edge_attr = torch.cat([rel_pos, dist], dim=1)
        
        # Message passing with physics
        out = self.propagate(
            edge_index, x=h, edge_attr=edge_attr, physics=physics_h
        )
        
        # Combine with physics and apply residual connection
        out = self.norm(h + out + physics_h)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        physics_i: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute messages with physics information.
        """
        edge_feat = self.lin_edge(edge_attr)
        msg_input = torch.cat([x_i, x_j, edge_feat, physics_i], dim=1)
        return self.lin_msg(msg_input)
    
    def physics_fraction(self) -> float:
        """Get physics fraction from physics encoder."""
        return self.physics_encoder.physics_fraction()


class ClothGNNPhysics(nn.Module):
    """
    ClothGNN with physics-encoded forces.
    
    Maintains the lightweight architecture while adding physics inductive bias
    through LiteClothConv. Total parameters should be ~70-80K.
    
    Architecture:
    1. PhysicsEncoder: Computes spring forces (LiteClothConv)
    2. ClothEncoder: Learns corrections with message passing
    3. GRU: Temporal dynamics combining physics and learned features
    4. Decoder: Outputs displacement predictions
    """
    
    def __init__(
        self,
        node_feat_dim: int = 16,
        hidden_dim: int = 64,
        stiffness: float = 1000.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.node_feat_dim = node_feat_dim
        
        # Physics-enhanced encoder
        self.encoder = ClothEncoderPhysics(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            stiffness=stiffness,
        )
        
        # GRU for temporal dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        logger.info(
            f"ClothGNNPhysics initialized: hidden_dim={hidden_dim}, "
            f"stiffness={stiffness}"
        )
    
    def forward(
        self,
        data: Data,
        h: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one timestep.
        
        Args:
            data: PyG Data object with x, edge_index, pos
            h: (N, hidden_dim) current hidden state
            rest_lengths: (E,) optional rest lengths
        
        Returns:
            predictions: (N, 3) displacement predictions
            h_next: (N, hidden_dim) updated hidden state
        """
        pos = data.pos if hasattr(data, "pos") else None
        if pos is None:
            raise ValueError("ClothGNNPhysics requires pos in data")
        
        # Encode with physics
        encoded = self.encoder(data.x, data.edge_index, pos, rest_lengths)
        
        # Update hidden state
        h_next = self.gru(encoded, h)
        
        # Decode to displacements
        predictions = self.decoder(h_next)
        
        return predictions, h_next
    
    def init_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for a new sequence."""
        return torch.zeros(num_nodes, self.hidden_dim, device=device)
    
    def rollout(
        self,
        data: Data,
        h: torch.Tensor,
        num_steps: int,
        rest_lengths: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Multi-step prediction rollout.
        """
        predictions = []
        hidden_states = [] if return_hidden else None
        
        current_pos = data.pos.clone()
        
        for step in range(num_steps):
            pred, h = self.forward(data, h, rest_lengths)
            predictions.append(pred)
            
            if return_hidden:
                hidden_states.append(h)
            
            # Update positions
            current_pos = current_pos + pred
            data.pos = current_pos
        
        if return_hidden:
            return predictions, hidden_states
        return predictions, h
    
    def physics_fraction(self) -> float:
        """Get physics fraction from encoder."""
        return self.encoder.physics_fraction()
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get comprehensive diagnostics."""
        return {
            "physics_fraction": self.physics_fraction(),
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_physics_clothgnn():
    """Test the physics-enhanced ClothGNN model."""
    print("=" * 60)
    print("Testing Physics-Enhanced ClothGNN")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test PhysicsEncoder
    print("Testing PhysicsEncoder...")
    physics_enc = PhysicsEncoder(hidden_channels=64, stiffness=1000.0).to(device)
    
    N, E = 100, 250
    pos = torch.randn(N, 3, device=device)
    edges = torch.randint(0, N, (2, E), device=device)
    
    forces = physics_enc(pos, edges)
    print(f"  Input: {pos.shape} → Forces: {forces.shape}")
    print(f"  Physics fraction: {physics_enc.physics_fraction():.3f}")
    print(f"  Parameters: {count_parameters(physics_enc):,}")
    print("  ✓ PhysicsEncoder test passed\n")
    
    # Test ClothGNNPhysics
    print("Testing ClothGNNPhysics...")
    model = ClothGNNPhysics(
        node_feat_dim=16,
        hidden_dim=64,
        stiffness=1000.0,
    ).to(device)
    
    # Create test data
    x = torch.randn(N, 16, device=device)
    data = Data(x=x, edge_index=edges, pos=pos)
    h = model.init_hidden(N, device)
    
    pred, h_next = model(data, h)
    print(f"  Input: x={x.shape}, pos={pos.shape}")
    print(f"  Output: pred={pred.shape}, h={h_next.shape}")
    print(f"  Physics fraction: {model.physics_fraction():.3f}")
    
    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    assert total_params < 100000, f"Model too large: {total_params} > 100K"
    print("  ✓ Parameter budget check passed (< 100K)\n")
    
    # Test rollout
    print("Testing rollout...")
    data.pos = pos.clone()
    h = model.init_hidden(N, device)
    predictions, h_final = model.rollout(data, h, num_steps=5)
    print(f"  Rollout: {len(predictions)} steps")
    print(f"  Final hidden: {h_final.shape}")
    print("  ✓ Rollout test passed\n")
    
    print("=" * 60)
    print("✓ All ClothGNN physics tests passed!")


if __name__ == "__main__":
    test_physics_clothgnn()
