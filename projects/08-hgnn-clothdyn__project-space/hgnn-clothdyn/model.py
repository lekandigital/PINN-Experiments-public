"""
HGNN-ClothDyn: Hierarchical Graph Neural Network for Cloth Dynamics

This module implements the core model architecture with:
- EdgeForceConv: Physics-informed message passing with learned edge stiffness
- HGNNClothDyn: Hierarchical multi-resolution graph neural network

The model learns spring constants approximating Hookean constraints:
    F_spring = -k_learned * (||x_i - x_j|| - L_rest) * (x_i - x_j) / ||x_i - x_j||

Author: HGNN-ClothDyn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeForceConv(MessagePassing):
    """
    Physics-informed message passing layer with learned edge stiffness.
    
    Computes spring-like forces between connected nodes:
        strain = (current_len - rest_len) / rest_len
        stiffness = MLP(strain)
        force = stiffness * edge_direction
    
    This approximates Hookean spring constraints in a differentiable manner.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        aggr: str = 'add'
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden dimension for stiffness MLP
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Stiffness MLP: strain → learned spring constant
        self.stiffness_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softplus()  # Ensure positive stiffness
        )
        
        # Message MLP: combines force with node features
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 4, hidden_channels),  # 4 = force(3) + stiffness(1)
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Update MLP: combines aggregated messages with node features
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        rest_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with physics-informed message passing.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge connectivity (2, E)
            pos: Node positions (N, 3)
            rest_lengths: Rest lengths for edges (E, 1) or None
            
        Returns:
            Updated node features (N, out_channels)
        """
        # Propagate messages
        out = self.propagate(
            edge_index, 
            x=x, 
            pos=pos, 
            rest_lengths=rest_lengths
        )
        
        # Update with residual connection from input
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        rest_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute messages between connected nodes with physics-informed forces.
        
        Args:
            x_i: Features of target nodes (E, in_channels)
            x_j: Features of source nodes (E, in_channels)
            pos_i: Positions of target nodes (E, 3)
            pos_j: Positions of source nodes (E, 3)
            rest_lengths: Rest lengths (E, 1)
            
        Returns:
            Messages (E, out_channels)
        """
        # Compute edge vector and current length
        edge_vec = pos_j - pos_i  # Direction from i to j
        current_length = torch.norm(edge_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_direction = edge_vec / current_length
        
        # Compute strain (clamped for numerical stability)
        if rest_lengths is not None:
            rest_len = rest_lengths.view(-1, 1).clamp(min=1e-8)
            strain = (current_length - rest_len) / rest_len
            strain = torch.clamp(strain, min=-2.0, max=2.0)  # Limit strain magnitude
        else:
            # Fallback: assume rest length is 1
            strain = torch.clamp(current_length - 1.0, min=-2.0, max=2.0)
        
        # Compute learned stiffness (clamped for stability)
        stiffness = self.stiffness_mlp(strain)
        stiffness = torch.clamp(stiffness, max=100.0)  # Limit stiffness
        
        # Compute spring force (Hookean), clamped
        force = stiffness * edge_direction * strain
        force = torch.clamp(force, min=-10.0, max=10.0)  # Limit force magnitude
        
        # Combine features with physics information
        msg_input = torch.cat([x_i, x_j, force, stiffness], dim=-1)
        message = self.message_mlp(msg_input)
        
        return message


class HGNNClothDyn(nn.Module):
    """
    Hierarchical Graph Neural Network for Cloth Dynamics.
    
    Architecture:
        Input → MLP → Fine Conv → Pool → Coarse Conv → Upsample → Fine Conv → Output
    
    This enables multi-resolution message passing:
    - Fine level: Local interactions (wrinkles)
    - Coarse level: Global interactions (drape)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_message_passes: int = 3,
        num_levels: int = 2
    ):
        """
        Args:
            input_dim: Input feature dimension (default: 6 for pos + vel)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default: 3 for velocity delta)
            num_message_passes: Number of message passing iterations per level
            num_levels: Number of hierarchy levels
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_message_passes = num_message_passes
        self.num_levels = num_levels
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fine-level convolutions (before and after coarse)
        self.fine_convs_pre = nn.ModuleList([
            EdgeForceConv(hidden_dim, hidden_dim)
            for _ in range(num_message_passes)
        ])
        
        self.fine_convs_post = nn.ModuleList([
            EdgeForceConv(hidden_dim, hidden_dim)
            for _ in range(num_message_passes)
        ])
        
        # Coarse-level convolutions
        self.coarse_convs = nn.ModuleList([
            EdgeForceConv(hidden_dim, hidden_dim)
            for _ in range(num_message_passes)
        ])
        
        # Upsampling MLP (combines coarse and fine features)
        self.upsample_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Layer norms for residual connections
        self.fine_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_message_passes * 2)
        ])
        self.coarse_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_message_passes)
        ])
        
        self._reset_parameters()
        
        logger.info(f"HGNNClothDyn initialized: input={input_dim}, hidden={hidden_dim}, "
                    f"output={output_dim}, levels={num_levels}, passes={num_message_passes}")
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        data_fine: Data,
        data_coarse: Optional[Data] = None,
        cluster_map: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical graph network.
        
        Args:
            data_fine: Fine-level graph data
            data_coarse: Coarse-level graph data (optional for single-level)
            cluster_map: Cluster assignments for pooling/unpooling
            return_latent: If True, also return latent features
            
        Returns:
            output: Velocity/position delta (N, output_dim)
            latent: (optional) Latent features (N, hidden_dim)
        """
        x = data_fine.x
        edge_index = data_fine.edge_index
        pos = data_fine.pos if hasattr(data_fine, 'pos') else x[:, :3]
        rest_lengths = data_fine.edge_attr
        
        # Encode input
        h = self.input_encoder(x)
        
        # Fine-level message passing (pre-coarse)
        for i, conv in enumerate(self.fine_convs_pre):
            h_new = conv(h, edge_index, pos, rest_lengths)
            h = self.fine_norms[i](h + h_new)  # Residual connection
        
        # Store fine features for skip connection
        h_fine_skip = h
        
        # Hierarchical processing (if coarse level available)
        if data_coarse is not None and cluster_map is not None:
            # Pool to coarse level
            num_coarse = data_coarse.num_nodes
            h_coarse = self._pool_features(h, cluster_map, num_coarse)
            
            # Coarse-level message passing
            coarse_edge_index = data_coarse.edge_index
            coarse_pos = data_coarse.pos if hasattr(data_coarse, 'pos') else None
            coarse_rest = data_coarse.edge_attr
            
            for i, conv in enumerate(self.coarse_convs):
                h_coarse_new = conv(h_coarse, coarse_edge_index, coarse_pos, coarse_rest)
                h_coarse = self.coarse_norms[i](h_coarse + h_coarse_new)
            
            # Upsample back to fine level
            h_upsampled = h_coarse[cluster_map]
            
            # Combine with skip connection
            h = self.upsample_mlp(torch.cat([h_fine_skip, h_upsampled], dim=-1))
        
        # Fine-level message passing (post-coarse)
        for i, conv in enumerate(self.fine_convs_post):
            h_new = conv(h, edge_index, pos, rest_lengths)
            h = self.fine_norms[self.num_message_passes + i](h + h_new)
        
        # Decode output
        output = self.output_decoder(h)
        
        if return_latent:
            return output, h
        return output
    
    def _pool_features(
        self,
        features: torch.Tensor,
        cluster_map: torch.Tensor,
        num_coarse: int
    ) -> torch.Tensor:
        """Pool fine features to coarse level using mean aggregation."""
        device = features.device
        dtype = features.dtype
        feat_dim = features.size(1)
        
        coarse_features = torch.zeros(num_coarse, feat_dim, device=device, dtype=dtype)
        counts = torch.zeros(num_coarse, device=device, dtype=dtype)
        
        # Scatter add
        coarse_features.scatter_add_(
            0, 
            cluster_map.unsqueeze(1).expand(-1, feat_dim), 
            features
        )
        counts.scatter_add_(0, cluster_map, torch.ones(features.size(0), device=device, dtype=dtype))
        
        # Mean
        coarse_features = coarse_features / counts.unsqueeze(1).clamp(min=1)
        
        return coarse_features
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'input_encoder': count_params(self.input_encoder),
            'fine_convs_pre': count_params(self.fine_convs_pre),
            'fine_convs_post': count_params(self.fine_convs_post),
            'coarse_convs': count_params(self.coarse_convs),
            'upsample_mlp': count_params(self.upsample_mlp),
            'output_decoder': count_params(self.output_decoder),
            'total': count_params(self)
        }


class ClothSimulator(nn.Module):
    """
    Complete cloth simulation wrapper that handles time stepping.
    
    Integrates velocity predictions to update positions.
    """
    
    def __init__(
        self,
        model: HGNNClothDyn,
        dt: float = 0.01,
        damping: float = 0.99
    ):
        """
        Args:
            model: HGNNClothDyn model
            dt: Time step for integration
            damping: Velocity damping factor
        """
        super().__init__()
        self.model = model
        self.dt = dt
        self.damping = damping
    
    def forward(
        self,
        data_fine: Data,
        data_coarse: Optional[Data] = None,
        cluster_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state (position, velocity).
        
        Args:
            data_fine: Current state as graph
            data_coarse: Coarse graph (optional)
            cluster_map: Pooling assignments
            
        Returns:
            next_pos: Updated positions (N, 3)
            next_vel: Updated velocities (N, 3)
        """
        # Get current state
        pos = data_fine.pos if hasattr(data_fine, 'pos') else data_fine.x[:, :3]
        vel = data_fine.x[:, 3:6] if data_fine.x.size(1) >= 6 else torch.zeros_like(pos)
        
        # Predict velocity delta
        delta_vel = self.model(data_fine, data_coarse, cluster_map)
        
        # Update velocity with damping
        next_vel = (vel + delta_vel) * self.damping
        
        # Update position (semi-implicit Euler)
        next_pos = pos + next_vel * self.dt
        
        return next_pos, next_vel
    
    def rollout(
        self,
        initial_data: Data,
        num_steps: int,
        coarse_data: Optional[Data] = None,
        cluster_map: Optional[torch.Tensor] = None,
        ground_height: float = -1.0
    ) -> List[torch.Tensor]:
        """
        Rollout simulation for multiple steps.
        
        Args:
            initial_data: Initial state
            num_steps: Number of simulation steps
            coarse_data: Coarse graph (recomputed if None)
            cluster_map: Pooling assignments
            ground_height: Ground plane height for collision
            
        Returns:
            List of position tensors for each timestep
        """
        positions = [initial_data.pos.clone()]
        
        data = initial_data.clone()
        
        for _ in range(num_steps):
            next_pos, next_vel = self.forward(data, coarse_data, cluster_map)
            
            # Simple ground collision
            collision_mask = next_pos[:, 1] < ground_height
            next_pos[collision_mask, 1] = ground_height
            next_vel[collision_mask, 1] = torch.abs(next_vel[collision_mask, 1]) * 0.5
            
            # Update data for next step
            data = Data(
                x=torch.cat([next_pos, next_vel], dim=-1),
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                pos=next_pos,
                num_nodes=data.num_nodes
            )
            
            positions.append(next_pos.clone())
        
        return positions


# ============================================================================
# Testing
# ============================================================================

def test_model():
    """Test model components."""
    print("=" * 60)
    print("Testing model.py")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy data
    num_nodes = 100
    num_edges = 400
    
    x = torch.randn(num_nodes, 6, device=device)  # pos + vel
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    pos = x[:, :3]
    rest_lengths = torch.ones(num_edges, 1, device=device) * 0.1
    
    data_fine = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=rest_lengths,
        pos=pos,
        num_nodes=num_nodes
    ).to(device)
    
    # Test EdgeForceConv
    print("\nTesting EdgeForceConv...")
    conv = EdgeForceConv(6, 128).to(device)
    out = conv(x, edge_index, pos, rest_lengths)
    print(f"  Input: {x.shape} → Output: {out.shape}")
    assert out.shape == (num_nodes, 128), "EdgeForceConv output shape mismatch"
    print("  ✓ EdgeForceConv test passed")
    
    # Test HGNNClothDyn (single level)
    print("\nTesting HGNNClothDyn (single level)...")
    model = HGNNClothDyn(
        input_dim=6,
        hidden_dim=128,
        output_dim=3,
        num_message_passes=2
    ).to(device)
    
    output = model(data_fine)
    print(f"  Input: {x.shape} → Output: {output.shape}")
    assert output.shape == (num_nodes, 3), "HGNNClothDyn output shape mismatch"
    print("  ✓ Single-level test passed")
    
    # Test with latent output
    print("\nTesting latent extraction...")
    output, latent = model(data_fine, return_latent=True)
    print(f"  Output: {output.shape}, Latent: {latent.shape}")
    assert latent.shape == (num_nodes, 128), "Latent shape mismatch"
    print("  ✓ Latent extraction test passed")
    
    # Test with coarse level
    print("\nTesting HGNNClothDyn (hierarchical)...")
    num_coarse = 25
    coarse_x = torch.randn(num_coarse, 6, device=device)
    coarse_edges = torch.randint(0, num_coarse, (2, 50), device=device)
    coarse_pos = coarse_x[:, :3]
    coarse_rest = torch.ones(50, 1, device=device) * 0.2
    
    data_coarse = Data(
        x=coarse_x,
        edge_index=coarse_edges,
        edge_attr=coarse_rest,
        pos=coarse_pos,
        num_nodes=num_coarse
    ).to(device)
    
    cluster_map = torch.randint(0, num_coarse, (num_nodes,), device=device)
    
    output = model(data_fine, data_coarse, cluster_map)
    print(f"  Hierarchical output: {output.shape}")
    assert output.shape == (num_nodes, 3), "Hierarchical output shape mismatch"
    print("  ✓ Hierarchical test passed")
    
    # Test ClothSimulator
    print("\nTesting ClothSimulator...")
    simulator = ClothSimulator(model, dt=0.01, damping=0.99).to(device)
    next_pos, next_vel = simulator(data_fine)
    print(f"  Next pos: {next_pos.shape}, Next vel: {next_vel.shape}")
    assert next_pos.shape == (num_nodes, 3), "Position output shape mismatch"
    print("  ✓ ClothSimulator test passed")
    
    # Test rollout
    print("\nTesting rollout (5 steps)...")
    positions = simulator.rollout(data_fine, num_steps=5)
    print(f"  Generated {len(positions)} frames")
    assert len(positions) == 6, "Rollout length mismatch"  # Initial + 5 steps
    print("  ✓ Rollout test passed")
    
    # Parameter count
    print("\nModel parameters:")
    param_counts = model.get_num_parameters()
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB allocated")
    
    print("\n✓ All model tests passed!")
    return True


if __name__ == "__main__":
    test_model()
