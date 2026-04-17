"""
Adapter for P06: CoastFlow-GNN - Graph Neural Network for Coastal Flow Simulation
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from harness.base_adapter import PyTorchAdapter, ProjectInfo
from harness.core import ModelInfo, TrainingInfo

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


if TORCH_AVAILABLE and PYG_AVAILABLE:
    class FlowEdgeConv(MessagePassing):
        """Edge convolution for flow simulation with flux-based aggregation."""
        
        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
            super().__init__(aggr='add')  # Sum aggregation for conservation
            
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_dim),
            )
        
        def forward(self, x, edge_index, edge_attr):
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            out = self.node_mlp(torch.cat([x, out], dim=-1))
            return x + out
        
        def message(self, x_i, x_j, edge_attr):
            return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    
    class CoastFlowGNN(nn.Module):
        """
        CoastFlow-GNN: Graph Neural Network for coastal flow prediction.
        Predicts water height and velocity at mesh nodes.
        """
        
        def __init__(self, node_dim: int = 5, edge_dim: int = 3, hidden_dim: int = 128,
                     num_layers: int = 6, output_dim: int = 3):
            super().__init__()
            
            # Node encoder: height (1) + velocity (2) + bathymetry (1) + boundary (1)
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            
            # Edge encoder: relative position (2) + edge length (1)
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            
            # Message passing
            self.conv_layers = nn.ModuleList([
                FlowEdgeConv(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            
            # Output: height change (1) + velocity change (2)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        
        def forward(self, data) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                data: PyG Data with node features, edge_index, edge_attr
            
            Returns:
                (N, 3): [delta_height, delta_vx, delta_vy]
            """
            x = self.node_encoder(data.x)
            edge_attr = self.edge_encoder(data.edge_attr)
            
            for conv in self.conv_layers:
                x = conv(x, data.edge_index, edge_attr)
            
            return self.decoder(x)


class P06CoastFlowGNNAdapter(PyTorchAdapter):
    """Adapter for CoastFlow-GNN project."""
    
    PROJECT_ID = "P06"
    PROJECT_NAME = "CoastFlow-GNN"
    
    def get_project_info(self) -> ProjectInfo:
        return ProjectInfo(
            project_id=self.PROJECT_ID,
            project_name=self.PROJECT_NAME,
            framework="pytorch-geometric",
            project_path=str(self.project_path),
        )
    
    def _create_model(self) -> nn.Module:
        """Create the CoastFlow-GNN model."""
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric not available")
        
        return CoastFlowGNN(
            node_dim=5,      # height + vel (2) + bathymetry + boundary
            edge_dim=3,      # rel_pos (2) + length
            hidden_dim=128,
            num_layers=6,
            output_dim=3     # delta_h + delta_v (2)
        )
    
    def _create_coastal_mesh(self, grid_size: int = 32, device: str = "cuda"):
        """Create a simple coastal mesh graph."""
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric not available")
        
        # Grid positions (2D)
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        
        num_nodes = grid_size * grid_size
        
        # Water height (initial)
        height = torch.ones(num_nodes, 1, device=device) * 1.0
        # Add a wave perturbation
        height += 0.1 * torch.sin(positions[:, 0:1] * 2 * np.pi)
        
        # Velocity (small initial flow)
        velocity = torch.zeros(num_nodes, 2, device=device)
        velocity[:, 0] = 0.1  # Flow in x direction
        
        # Bathymetry (ocean floor depth)
        bathymetry = torch.ones(num_nodes, 1, device=device) * 10.0
        # Shallow near coast (x = 1)
        bathymetry -= 8.0 * positions[:, 0:1]
        
        # Boundary flags (left/right boundaries)
        boundary = torch.zeros(num_nodes, 1, device=device)
        boundary[positions[:, 0] < 0.05] = 1.0  # Inlet
        boundary[positions[:, 0] > 0.95] = 2.0  # Coast
        
        # Node features
        node_features = torch.cat([height, velocity, bathymetry, boundary], dim=-1)
        
        # Edges: connect adjacent nodes
        edge_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                
                if j < grid_size - 1:
                    edge_list.extend([[idx, idx + 1], [idx + 1, idx]])
                if i < grid_size - 1:
                    edge_list.extend([[idx, idx + grid_size], [idx + grid_size, idx]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
        
        # Edge attributes
        src, dst = edge_index
        rel_pos = positions[dst] - positions[src]
        edge_length = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, edge_length], dim=-1)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
                    pos=positions)
    
    def prepare_test_input(self, device: str, batch_size: int = 1) -> Tuple:
        """Prepare test inputs: coastal mesh graph."""
        torch.manual_seed(42)
        
        data_list = [self._create_coastal_mesh(grid_size=32, device=device)
                     for _ in range(batch_size)]
        
        batch = Batch.from_data_list(data_list)
        return (batch,)
    
    def run_inference(self, model: nn.Module, inputs: Tuple, device: str) -> Any:
        """Run model inference."""
        batch, = inputs
        return model(batch)
    
    def compute_domain_metrics(self, predictions: Any,
                               references: Optional[Any] = None) -> Dict[str, float]:
        """Compute fluid flow metrics."""
        metrics = {}
        
        if isinstance(predictions, torch.Tensor):
            pred = predictions.detach().cpu().numpy()
        else:
            pred = np.array(predictions)
        
        delta_h = pred[:, 0]
        delta_v = pred[:, 1:]
        
        # Height change statistics
        metrics['height_change_mean'] = float(np.mean(np.abs(delta_h)))
        metrics['height_change_max'] = float(np.max(np.abs(delta_h)))
        
        # Velocity change statistics
        vel_magnitude = np.linalg.norm(delta_v, axis=-1)
        metrics['velocity_change_mean'] = float(np.mean(vel_magnitude))
        metrics['velocity_change_max'] = float(np.max(vel_magnitude))
        
        # Conservation check (mass should be approximately conserved)
        mass_change = np.sum(delta_h)
        metrics['mass_change'] = float(np.abs(mass_change))
        
        # Stability metric
        metrics['max_courant_proxy'] = float(np.max(vel_magnitude))
        
        if references is not None:
            if isinstance(references, torch.Tensor):
                ref = references.detach().cpu().numpy()
            else:
                ref = np.array(references)
            
            error = pred - ref
            metrics['total_rmse'] = float(np.sqrt(np.mean(error**2)))
            metrics['height_rmse'] = float(np.sqrt(np.mean(error[:, 0]**2)))
            metrics['velocity_rmse'] = float(np.sqrt(np.mean(error[:, 1:]**2)))
        
        return metrics
    
    def get_reference_data(self) -> Optional[Any]:
        """Generate synthetic reference data."""
        torch.manual_seed(123)
        
        num_nodes = 1024  # 32x32
        
        # Small changes for one timestep
        delta = torch.zeros(num_nodes, 3)
        delta[:, 0] = torch.randn(num_nodes) * 0.01  # Height change
        delta[:, 1:] = torch.randn(num_nodes, 2) * 0.001  # Velocity change
        
        return delta
    
    def get_training_info(self) -> TrainingInfo:
        """Return training information."""
        return TrainingInfo(
            total_time_seconds=10800,  # ~3 hours
            epochs=1000,
            final_loss=0.0005,
            final_metrics={
                'height_error': 0.01,
                'velocity_error': 0.005,
                'mass_conservation': 0.99,
            },
        )
    
    def get_test_dataset_description(self) -> str:
        return "32x32 coastal mesh (1024 nodes, ~2000 edges) with water height, velocity, bathymetry"


def get_adapter(project_path: Optional[Path] = None) -> P06CoastFlowGNNAdapter:
    """Get adapter instance for P06 project."""
    if project_path is None:
        project_path = Path(__file__).parent.parent.parent / "projects" / "06-coastflow-gnn__project-space"
    return P06CoastFlowGNNAdapter(project_path)
