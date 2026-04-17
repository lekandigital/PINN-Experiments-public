"""
HGNN-NIF-Cloth Model Registration (Project 09).

Registers the hierarchical GNN as teacher model for distillation.
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
# Import HGNN Model
# =============================================================================

try:
    from src.models.hgnn import AdaptiveHGNN
    from src.models.graph_conv import GraphConv, GraphConvBlock
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    AdaptiveHGNN = None


# =============================================================================
# Teacher Wrapper for Distillation
# =============================================================================

class HGNNTeacherWrapper(nn.Module):
    """
    Wrapper around AdaptiveHGNN for distillation compatibility.
    
    The distillation pipeline expects a simple forward signature:
    output = model(inputs)
    
    This wrapper handles the multi-resolution inputs internally.
    """
    
    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        coarse_ratio: float = 0.25,  # Coarse graph has 25% of fine nodes
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.coarse_ratio = coarse_ratio
        
        if MODELS_AVAILABLE:
            self.hgnn = AdaptiveHGNN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        else:
            # Fallback: simple GNN for testing
            self.hgnn = None
            self.fallback_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
        
        # Output projection (from HGNN latent to displacement)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def _create_coarse_graph(
        self,
        fine_x: torch.Tensor,
        fine_edges: torch.Tensor,
    ):
        """Create coarse graph by subsampling."""
        if fine_x.dim() == 2:
            N = fine_x.shape[0]
        else:
            N = fine_x.shape[1]
        
        N_coarse = max(int(N * self.coarse_ratio), 4)
        
        # Simple subsampling (every k-th node)
        k = max(1, N // N_coarse)
        coarse_indices = torch.arange(0, N, k)[:N_coarse]
        
        if fine_x.dim() == 2:
            coarse_x = fine_x[coarse_indices]
        else:
            coarse_x = fine_x[:, coarse_indices]
        
        # Create coarse edges (connect nearby coarse nodes)
        coarse_edges = []
        grid_size = int(N_coarse ** 0.5)
        for i in range(N_coarse):
            row = i // grid_size
            col = i % grid_size
            if col < grid_size - 1:
                coarse_edges.append([i, i + 1])
            if row < grid_size - 1 and i + grid_size < N_coarse:
                coarse_edges.append([i, i + grid_size])
        
        if len(coarse_edges) > 0:
            coarse_edges = torch.tensor(coarse_edges, dtype=torch.long).t().to(fine_x.device)
        else:
            coarse_edges = torch.zeros(2, 0, dtype=torch.long, device=fine_x.device)
        
        return coarse_x, coarse_edges
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass compatible with distillation pipeline.
        
        Args:
            node_features: [N, in_dim] or [B, N, in_dim]
            edge_index: [2, E]
            
        Returns:
            displacements: [N, 3] or [B, N, 3]
        """
        if self.hgnn is None:
            # Fallback for testing
            return self.fallback_net(node_features)
        
        # Create coarse graph
        coarse_x, coarse_edges = self._create_coarse_graph(node_features, edge_index)
        
        # Forward through HGNN
        fine_out, coarse_out, latent = self.hgnn(
            node_features, edge_index,
            coarse_x, coarse_edges,
        )
        
        # Project to displacement
        output = self.output_head(fine_out)
        
        return output


# =============================================================================
# Physics Loss
# =============================================================================

class HGNNPhysicsLoss:
    """Physics loss for hierarchical cloth model."""
    
    def __init__(
        self,
        edge_weight: float = 1.0,
        bending_weight: float = 0.5,
    ):
        self.edge_weight = edge_weight
        self.bending_weight = bending_weight
    
    def __call__(
        self,
        model_output: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=model_output.device)
        
        if 'edge_index' in inputs and 'positions' in inputs:
            positions = inputs['positions']
            if positions.shape[-1] > 3:
                positions = positions[..., :3]
            
            new_pos = positions + model_output
            edge_index = inputs['edge_index']
            
            src, dst = edge_index
            lengths = torch.norm(new_pos[dst] - new_pos[src], dim=-1)
            
            if 'rest_lengths' in inputs:
                rest = inputs['rest_lengths']
                stretch = (lengths - rest).abs() / (rest + 1e-8)
                loss = loss + self.edge_weight * stretch.mean()
        
        return loss


# =============================================================================
# Registration
# =============================================================================

def register_hgnn_models():
    """Register HGNN models with the distillation registry."""
    
    # Register teacher wrapper
    registry.register(
        name="hgnn_nif_cloth",
        model_class=HGNNTeacherWrapper,
        default_config={
            "in_dim": 6,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "coarse_ratio": 0.25,
        },
        input_spec=InputSpec(
            type="graph",
            node_features=6,
            edge_features=3,
            description="Hierarchical cloth mesh",
        ),
        output_spec=OutputSpec(
            type="tensor",
            shape=[-1, 3],
            description="3D vertex displacements",
        ),
        physics_loss_class=HGNNPhysicsLoss,
        description="Adaptive HGNN for cloth (~136K params)",
        project="09-HGNN-NIF-Cloth",
        param_count=136000,
    )
    print("Registered: hgnn_nif_cloth")
    
    # Register the raw AdaptiveHGNN if available
    if MODELS_AVAILABLE and AdaptiveHGNN is not None:
        registry.register(
            name="adaptive_hgnn_raw",
            model_class=AdaptiveHGNN,
            default_config={
                "in_dim": 3,
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
            },
            input_spec=InputSpec(
                type="graph",
                description="Raw HGNN (requires fine + coarse graphs)",
            ),
            output_spec=OutputSpec(
                type="tensor",
                description="Fine features + coarse features + latent",
            ),
            description="Raw AdaptiveHGNN module",
            project="09-HGNN-NIF-Cloth",
        )
        print("Registered: adaptive_hgnn_raw")


# Auto-register when imported
register_hgnn_models()
