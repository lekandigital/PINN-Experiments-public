"""
CoastFlow-GNN with Physics-Encoded Shallow Water Equations.

This module enhances CoastFlowGNN with physics-encoded message passing
using the shared ShallowWaterConv layer. The physics layer computes
analytical shallow water forces (pressure gradient, bottom friction, etc.)
while the GNN learns corrections and complex interactions.

Architecture:
1. ShallowWaterConv: Physics-encoded shallow water equations
2. Original GraphEncoder: Hierarchical GCN with TopKPooling
3. NodeDecoder: MLP with global context injection
4. Output: Velocity and wave height predictions

The physics forces serve as a strong inductive bias, improving:
- Generalization to unseen bathymetry
- Physical consistency of predictions
- Training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from typing import Optional, Tuple, Dict

# Import shared physics-encoded convolution
import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')
from shared.physics_conv import ShallowWaterConv
from shared.physics_conv.shallow_water_conv import ShallowWaterConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsLayerSW(nn.Module):
    """
    Physics layer using ShallowWaterConv for coastal flow.
    
    Computes:
    - Pressure gradient forces from water surface elevation
    - Bottom friction forces
    - Mass conservation constraints
    """
    
    def __init__(
        self,
        hidden_channels: int,
        gravity: float = 9.81,
        friction_coefficient: float = 0.01,
        use_correction: bool = True,
    ):
        super().__init__()
        
        # Configure physics-encoded convolution
        config = ShallowWaterConfig(
            gravity=gravity,
            drag_coefficient=friction_coefficient,  # Use drag_coefficient
            correction_hidden_dim=32 if use_correction else 1,
            correction_layers=2 if use_correction else 0,
            correction_scale_init=0.1 if use_correction else 0.0,
            track_diagnostics=True,
            enforce_mass_conservation=True,
        )
        
        self.sw_conv = ShallowWaterConv(config=config)
        
        # Project physics outputs to hidden dimension
        # ShallowWaterConv outputs: [dη/dt, du/dt, dv/dt] = 3 dims
        self.proj = nn.Linear(3, hidden_channels)
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        velocity: torch.Tensor,
        eta: torch.Tensor,
        bathymetry: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute physics-based updates.
        
        Args:
            pos: Node positions [N, 3]
            edge_index: Edge connectivity [2, E]
            velocity: Current velocity [N, 3]
            eta: Water surface elevation [N, 1]
            bathymetry: Bottom depth [N, 1] (optional)
            boundary_mask: Land boundary mask [N] (optional)
        
        Returns:
            physics_features: Physics-encoded features [N, hidden]
        """
        N = pos.size(0)
        device = pos.device
        
        # Get bathymetry
        if bathymetry is None:
            # Use z-coordinate as proxy for bathymetry
            bathymetry = pos[:, 2:3] if pos.size(-1) >= 3 else torch.ones(N, 1, device=device)
        elif bathymetry.dim() == 1:
            bathymetry = bathymetry.unsqueeze(-1)
        
        if eta.dim() == 1:
            eta = eta.unsqueeze(-1)
        
        # Pack state for shallow water conv: [η, u, v, h_bathy]
        # Extract u, v from velocity (ignore w for shallow water)
        u = velocity[:, 0:1] if velocity.size(-1) >= 1 else torch.zeros(N, 1, device=device)
        v = velocity[:, 1:2] if velocity.size(-1) >= 2 else torch.zeros(N, 1, device=device)
        
        state = torch.cat([eta, u, v, bathymetry], dim=-1)  # [N, 4]
        
        # Compute edge attributes for shallow water conv
        # Need: [edge_length, normal_x, normal_y, edge_type]
        src, tgt = edge_index[0], edge_index[1]
        edge_vec = pos[tgt, :2] - pos[src, :2]  # [E, 2] (x, y only)
        edge_length = torch.norm(edge_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_normal = edge_vec / edge_length  # Normalized direction
        edge_type = torch.zeros_like(edge_length)  # 0 = interior
        
        edge_attr = torch.cat([edge_length, edge_normal, edge_type], dim=-1)  # [E, 4]
        
        # Compute physics updates using shallow water equations
        updates = self.sw_conv(state, edge_index, edge_attr)
        
        # Project to hidden dimension
        return self.proj(updates)
    
    def physics_fraction(self) -> float:
        """Get fraction of output from physics vs learned."""
        return self.sw_conv.physics_fraction()


class PhysicsEnhancedEncoder(nn.Module):
    """
    Graph encoder with physics-encoded first layer.
    
    Combines:
    1. ShallowWaterConv for physics-based features
    2. GCN layers for learned features
    3. TopKPooling for multi-scale representation
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        pool_ratios: Tuple[float, ...] = (0.8, 0.5),
        dropout: float = 0.1,
        gravity: float = 9.81,
        friction_coefficient: float = 0.01,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.pool_ratios = pool_ratios
        self.dropout = dropout
        
        # Physics layer for shallow water equations
        self.physics_layer = PhysicsLayerSW(
            hidden_channels=hidden_channels,
            gravity=gravity,
            friction_coefficient=friction_coefficient,
            use_correction=True,
        )
        
        # Initial embedding (includes physics features)
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Level 0: Full resolution
        self.conv0_1 = GCNConv(hidden_channels, hidden_channels)
        self.conv0_2 = GCNConv(hidden_channels, hidden_channels)
        self.norm0 = nn.LayerNorm(hidden_channels)
        
        # Level 1: First pooling
        self.pool1 = TopKPooling(hidden_channels, ratio=pool_ratios[0])
        self.conv1_1 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv1_2 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.norm1 = nn.LayerNorm(hidden_channels * 2)
        
        # Level 2: Second pooling
        self.pool2 = TopKPooling(hidden_channels * 2, ratio=pool_ratios[1])
        self.conv2_1 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        self.conv2_2 = GCNConv(hidden_channels * 4, hidden_channels * 4)
        self.norm2 = nn.LayerNorm(hidden_channels * 4)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with physics-enhanced encoding.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N]
            pos: Node positions [N, 3] (optional, extracted from x if not provided)
            velocity: Current velocity [N, 3] (optional)
            eta: Water surface elevation [N, 1] (optional)
        
        Returns:
            node_features: Encoded features [N, hidden]
            graph_features: Global features [B, hidden*7]
            aux_data: Auxiliary data
        """
        aux_data = {}
        
        # Extract position if not provided
        if pos is None:
            pos = x[:, :3] if x.size(-1) >= 3 else x
        
        # Compute physics features
        if velocity is None:
            velocity = torch.zeros(x.size(0), 3, device=x.device)
        if eta is None:
            eta = torch.zeros(x.size(0), 1, device=x.device)
        
        physics_features = self.physics_layer(pos, edge_index, velocity, eta)
        aux_data['physics_features'] = physics_features
        
        # Combine input with physics features
        x_aug = torch.cat([x, physics_features], dim=-1)
        x = self.input_mlp(x_aug)
        
        # Level 0
        x0 = self.conv0_1(x, edge_index)
        x0 = F.relu(x0)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x0 = self.conv0_2(x0, edge_index)
        x0 = self.norm0(x0 + x)
        x0 = F.relu(x0)
        
        aux_data['x0'] = x0
        aux_data['batch0'] = batch
        
        # Level 1
        x1, edge_index1, _, batch1, perm1, score1 = self.pool1(
            x0, edge_index, None, batch
        )
        aux_data['perm1'] = perm1
        
        x1 = self.conv1_1(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.conv1_2(x1, edge_index1)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        
        aux_data['x1'] = x1
        aux_data['batch1'] = batch1
        
        # Level 2
        x2, edge_index2, _, batch2, perm2, score2 = self.pool2(
            x1, edge_index1, None, batch1
        )
        aux_data['perm2'] = perm2
        
        x2 = self.conv2_1(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.conv2_2(x2, edge_index2)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        
        aux_data['x2'] = x2
        aux_data['batch2'] = batch2
        
        # Global pooling
        g0 = global_mean_pool(x0, batch)
        g1 = global_mean_pool(x1, batch1)
        g2 = global_mean_pool(x2, batch2)
        
        graph_features = torch.cat([g0, g1, g2], dim=-1)
        
        return x0, graph_features, aux_data
    
    def physics_fraction(self) -> float:
        """Get physics fraction from physics layer."""
        return self.physics_layer.physics_fraction()


class CoastFlowGNNPhysics(nn.Module):
    """
    CoastFlow-GNN with physics-encoded shallow water equations.
    
    Combines:
    1. ShallowWaterConv for analytical physics
    2. Hierarchical GCN for learned multi-scale features
    3. MLP decoder for velocity and wave height predictions
    
    The physics layer provides strong inductive bias for:
    - Pressure-driven flow
    - Bottom friction effects
    - Mass conservation
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 4,
        pool_ratios: Tuple[float, ...] = (0.8, 0.5),
        dropout: float = 0.1,
        gravity: float = 9.81,
        friction_coefficient: float = 0.01,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Physics-enhanced encoder
        self.encoder = PhysicsEnhancedEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            pool_ratios=pool_ratios,
            dropout=dropout,
            gravity=gravity,
            friction_coefficient=friction_coefficient,
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels * 7, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        
        logger.info(
            f"CoastFlowGNNPhysics initialized: in={in_channels}, "
            f"hidden={hidden_channels}, out={out_channels}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N]
            velocity: Current velocity [N, 3] (optional)
            eta: Water surface elevation [N, 1] (optional)
            return_aux: Return auxiliary data
        
        Returns:
            predictions: [N, out_channels] (u_x, u_y, u_z, wave_height)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode
        node_features, graph_features, aux_data = self.encoder(
            x, edge_index, batch, velocity=velocity, eta=eta
        )
        
        # Broadcast global features
        global_broadcast = graph_features[batch]
        
        # Combine and decode
        combined = torch.cat([node_features, global_broadcast], dim=-1)
        predictions = self.decoder(combined)
        
        if return_aux:
            return predictions, aux_data
        return predictions
    
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


def test_coastflow_physics():
    """Test the physics-enhanced CoastFlow-GNN model."""
    print("=" * 60)
    print("Testing Physics-Enhanced CoastFlow-GNN")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test PhysicsLayerSW
    print("Testing PhysicsLayerSW...")
    physics_layer = PhysicsLayerSW(
        hidden_channels=64,
        gravity=9.81,
        friction_coefficient=0.01,
    ).to(device)
    
    N, E = 100, 300
    pos = torch.randn(N, 3, device=device)
    edges = torch.randint(0, N, (2, E), device=device)
    velocity = torch.randn(N, 3, device=device) * 0.1
    eta = torch.randn(N, 1, device=device) * 0.5
    
    physics_features = physics_layer(pos, edges, velocity, eta)
    print(f"  Input: pos={pos.shape}, vel={velocity.shape}, eta={eta.shape}")
    print(f"  Output: {physics_features.shape}")
    print(f"  Physics fraction: {physics_layer.physics_fraction():.3f}")
    print("  ✓ PhysicsLayerSW test passed\n")
    
    # Test CoastFlowGNNPhysics
    print("Testing CoastFlowGNNPhysics...")
    model = CoastFlowGNNPhysics(
        in_channels=6,
        hidden_channels=64,
        out_channels=4,
    ).to(device)
    
    x = torch.randn(N, 6, device=device)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    
    predictions = model(x, edges, batch, velocity=velocity, eta=eta)
    print(f"  Input: x={x.shape}")
    print(f"  Output: {predictions.shape}")
    print(f"  Physics fraction: {model.physics_fraction():.3f}")
    
    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print("  ✓ CoastFlowGNNPhysics test passed\n")
    
    # Test batched input
    print("Testing batched input...")
    batch_multi = torch.cat([
        torch.zeros(50, dtype=torch.long, device=device),
        torch.ones(50, dtype=torch.long, device=device)
    ])
    x_batch = torch.randn(100, 6, device=device)
    
    pred_batch = model(x_batch, edges, batch_multi)
    print(f"  Batched output: {pred_batch.shape}")
    print("  ✓ Batched input test passed\n")
    
    print("=" * 60)
    print("✓ All CoastFlow-GNN physics tests passed!")


if __name__ == "__main__":
    test_coastflow_physics()
