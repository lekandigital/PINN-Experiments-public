"""
Model Registration for ClothGNN (Project 05).

Registers ClothGNN models with the distillation pipeline for:
1. Use as a student model (distill from HGNN-NIF-Cloth)
2. Use as a teacher model (distill to even smaller models)

The lightweight "cloth_gnn_lite" variant is designed for mobile deployment.
"""

import sys
from pathlib import Path

# Add tools to path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from tools.distillation.registry import registry, InputSpec, OutputSpec


# =============================================================================
# Model Definitions
# =============================================================================

class ClothGNNLite(nn.Module):
    """
    Lightweight ClothGNN variant for mobile deployment.
    
    Key differences from full ClothGNN:
    - Smaller hidden dimensions (32 vs 64)
    - Fewer message passing layers
    - Simplified edge features
    - No optional physics enhancement layers
    
    Target: <50K parameters, 100+ FPS on mobile
    """
    
    def __init__(
        self,
        node_input_dim: int = 6,      # position (3) + velocity (3)
        edge_input_dim: int = 3,       # relative position
        node_hidden_dim: int = 32,     # Smaller for mobile
        edge_hidden_dim: int = 16,
        num_message_passes: int = 3,
        output_dim: int = 3,           # acceleration/displacement
        aggregation: str = "mean",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_message_passes = num_message_passes
        self.output_dim = output_dim
        self.aggregation = aggregation
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        
        # Edge encoder  
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
        )
        
        # Message layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        for _ in range(num_message_passes):
            # Message MLP: node_i + node_j + edge -> hidden
            self.message_layers.append(nn.Sequential(
                nn.Linear(node_hidden_dim * 2 + edge_hidden_dim, node_hidden_dim),
                nn.ReLU(),
            ))
            
            # Update MLP: node + aggregated_messages -> hidden
            self.update_layers.append(nn.Sequential(
                nn.Linear(node_hidden_dim * 2, node_hidden_dim),
                nn.ReLU(),
            ))
            
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(node_hidden_dim))
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(node_hidden_dim, output_dim),
        )
        
        # GRU for temporal dynamics (optional, for multi-step prediction)
        self.gru = nn.GRUCell(node_hidden_dim, node_hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass.
        
        Args:
            node_features: [N, node_input_dim] or [B, N, node_input_dim]
            edge_index: [2, E] edge connectivity
            edge_features: [E, edge_input_dim] optional edge features
            hidden: [N, hidden] or [B, N, hidden] optional GRU hidden state
            
        Returns:
            output: [N, output_dim] or [B, N, output_dim] predicted displacements
            new_hidden: Updated hidden state
        """
        # Handle batched vs unbatched input
        batched = node_features.dim() == 3
        if batched:
            B, N, _ = node_features.shape
            node_features = node_features.reshape(B * N, -1)
        else:
            N = node_features.shape[0]
            B = 1
        
        # Encode nodes
        h = self.node_encoder(node_features)  # [B*N, hidden]
        
        # Compute edge features if not provided
        src, dst = edge_index
        if edge_features is None:
            # Compute relative positions
            if batched:
                pos = node_features[:, :3].reshape(B, N, 3)
                edge_features = (pos[:, dst, :] - pos[:, src, :]).reshape(-1, 3)
            else:
                edge_features = node_features[dst, :3] - node_features[src, :3]
        
        edge_attr = self.edge_encoder(edge_features)  # [E, edge_hidden]
        
        # Message passing layers
        for i in range(self.num_message_passes):
            # Compute messages
            h_i = h[dst] if not batched else h.reshape(B, N, -1)[:, dst, :].reshape(-1, self.node_hidden_dim)
            h_j = h[src] if not batched else h.reshape(B, N, -1)[:, src, :].reshape(-1, self.node_hidden_dim)
            
            msg_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
            messages = self.message_layers[i](msg_input)  # [E, hidden]
            
            # Aggregate messages
            if self.aggregation == "mean":
                # Use scatter_mean or manual implementation
                agg = torch.zeros_like(h)
                count = torch.zeros(h.shape[0], 1, device=h.device)
                agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.node_hidden_dim), messages)
                count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst.unsqueeze(-1).float()))
                agg = agg / count.clamp(min=1)
            else:  # sum
                agg = torch.zeros_like(h)
                agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.node_hidden_dim), messages)
            
            # Update nodes
            update_input = torch.cat([h, agg], dim=-1)
            h_new = self.update_layers[i](update_input)
            
            # Residual + norm
            h = h + h_new
            if self.norms is not None:
                h = self.norms[i](h)
        
        # GRU update (optional temporal consistency)
        if hidden is not None:
            if batched:
                hidden = hidden.reshape(B * N, -1)
            h = self.gru(h, hidden)
        
        new_hidden = h.clone()
        
        # Decode to output
        output = self.decoder(h)  # [B*N, output_dim]
        
        # Reshape back if batched
        if batched:
            output = output.reshape(B, N, self.output_dim)
            new_hidden = new_hidden.reshape(B, N, self.node_hidden_dim)
        
        return output, new_hidden
    
    def init_hidden(self, num_nodes: int, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Initialize hidden state."""
        if batch_size > 1:
            return torch.zeros(batch_size, num_nodes, self.node_hidden_dim, device=device)
        return torch.zeros(num_nodes, self.node_hidden_dim, device=device)


# =============================================================================
# Physics Loss for ClothGNN
# =============================================================================

class ClothPhysicsLoss:
    """
    Physics-based loss terms for cloth simulation.
    
    Includes:
    - Edge length preservation (prevent stretching)
    - Momentum conservation
    - Optional collision penalty
    """
    
    def __init__(
        self,
        edge_weight: float = 1.0,
        momentum_weight: float = 0.1,
        edge_threshold: float = 0.02,
    ):
        self.edge_weight = edge_weight
        self.momentum_weight = momentum_weight
        self.edge_threshold = edge_threshold
    
    def __call__(
        self,
        model_output: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute physics loss.
        
        Args:
            model_output: [N, 3] predicted displacements
            inputs: Dict with 'positions', 'edge_index', 'rest_lengths'
            
        Returns:
            Physics loss scalar
        """
        loss = torch.tensor(0.0, device=model_output.device)
        
        # Edge length preservation
        if 'edge_index' in inputs and 'rest_lengths' in inputs:
            positions = inputs.get('positions', inputs.get('node_features', None))
            if positions is not None:
                if positions.shape[-1] > 3:
                    positions = positions[..., :3]
                
                new_positions = positions + model_output
                
                edge_index = inputs['edge_index']
                rest_lengths = inputs['rest_lengths']
                
                src, dst = edge_index
                current_lengths = torch.norm(
                    new_positions[dst] - new_positions[src], dim=-1
                )
                
                stretch = torch.abs(current_lengths - rest_lengths) / (rest_lengths + 1e-8)
                edge_loss = torch.mean(torch.relu(stretch - self.edge_threshold))
                
                loss = loss + self.edge_weight * edge_loss
        
        # Momentum conservation (total displacement should be bounded)
        if self.momentum_weight > 0:
            total_disp = model_output.sum(dim=0).norm()
            loss = loss + self.momentum_weight * total_disp
        
        return loss


# =============================================================================
# Sampling Strategy
# =============================================================================

class ClothSamplingStrategy:
    """
    Sampling strategy for cloth simulation data.
    
    Generates or loads cloth configurations for distillation training.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        num_nodes: int = 1024,
        spatial_range: tuple = (-1.0, 1.0),
    ):
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.spatial_range = spatial_range
        
        self._dataset = None
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of cloth configurations."""
        # For now, generate synthetic data
        # In production, this would load from HDF5 dataset
        
        grid_size = int(self.num_nodes ** 0.5)
        
        # Create grid positions
        x = torch.linspace(self.spatial_range[0], self.spatial_range[1], grid_size)
        y = torch.linspace(self.spatial_range[0], self.spatial_range[1], grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        positions = torch.stack([
            xx.flatten(),
            torch.zeros(self.num_nodes),
            yy.flatten(),
        ], dim=-1)  # [N, 3]
        
        # Add random perturbation
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)
        positions = positions + 0.05 * torch.randn_like(positions)
        
        # Random velocities
        velocities = 0.1 * torch.randn(batch_size, self.num_nodes, 3)
        
        # Combine into node features
        node_features = torch.cat([positions, velocities], dim=-1)  # [B, N, 6]
        
        # Create edge index (grid connectivity)
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j
                if j < grid_size - 1:
                    edges.append([node, node + 1])
                    edges.append([node + 1, node])
                if i < grid_size - 1:
                    edges.append([node, node + grid_size])
                    edges.append([node + grid_size, node])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Compute rest lengths
        src, dst = edge_index
        rest_lengths = torch.norm(positions[0, dst] - positions[0, src], dim=-1)
        
        return {
            'node_features': node_features,
            'positions': positions[..., :3],
            'velocities': positions[..., 3:6],
            'edge_index': edge_index,
            'rest_lengths': rest_lengths,
        }
    
    def sample_epoch(
        self,
        batch_size: int,
        num_samples: int,
    ) -> list:
        """Generate batches for an epoch."""
        num_batches = num_samples // batch_size
        return [self.sample(batch_size) for _ in range(num_batches)]


# =============================================================================
# Registration
# =============================================================================

def register_clothgnn_models():
    """Register ClothGNN models with the distillation registry."""
    
    # Import the full ClothGNN model
    try:
        from models.clothgnn import ClothGNNModel
        
        # Register full ClothGNN as potential teacher
        registry.register(
            name="cloth_gnn",
            model_class=ClothGNNModel,
            default_config={
                "node_feat_dim": 16,
                "hidden_dim": 64,
            },
            input_spec=InputSpec(
                type="graph",
                node_features=16,
                edge_features=4,
                description="Cloth mesh graph with node/edge features",
            ),
            output_spec=OutputSpec(
                type="tensor",
                shape=[-1, 3],
                description="3D vertex displacements",
            ),
            physics_loss_class=ClothPhysicsLoss,
            sampling_strategy_class=ClothSamplingStrategy,
            description="Full ClothGNN model (~70K params)",
            project="05-ClothGNN",
            param_count=70000,
        )
        print("Registered: cloth_gnn")
    except ImportError as e:
        print(f"Could not register cloth_gnn: {e}")
    
    # Register lightweight variant for mobile
    registry.register(
        name="cloth_gnn_lite",
        model_class=ClothGNNLite,
        default_config={
            "node_input_dim": 6,
            "edge_input_dim": 3,
            "node_hidden_dim": 32,
            "edge_hidden_dim": 16,
            "num_message_passes": 3,
            "output_dim": 3,
            "aggregation": "mean",
            "use_layer_norm": True,
            "dropout": 0.0,
        },
        input_spec=InputSpec(
            type="graph",
            node_features=6,
            edge_features=3,
            description="Cloth mesh with position + velocity node features",
        ),
        output_spec=OutputSpec(
            type="tensor",
            shape=[-1, 3],
            description="3D vertex displacements/accelerations",
        ),
        physics_loss_class=ClothPhysicsLoss,
        sampling_strategy_class=ClothSamplingStrategy,
        description="Lightweight ClothGNN for mobile (~35K params)",
        project="05-ClothGNN",
        param_count=35000,
    )
    print("Registered: cloth_gnn_lite")


# Auto-register when imported
register_clothgnn_models()
