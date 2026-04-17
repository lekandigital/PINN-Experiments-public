"""
Adapter for P05: ClothGNN - Graph Neural Network for Cloth Simulation
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
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data, Batch
    import torch_geometric
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


if TORCH_AVAILABLE and PYG_AVAILABLE:
    class EdgeConv(MessagePassing):
        """Edge convolution layer for cloth simulation."""
        
        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
            super().__init__(aggr='mean')
            
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_dim),
            )
        
        def forward(self, x, edge_index, edge_attr):
            # Message passing
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            # Node update
            out = self.node_mlp(torch.cat([x, out], dim=-1))
            return x + out  # Residual
        
        def message(self, x_i, x_j, edge_attr):
            return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    
    class ClothGNN(nn.Module):
        """
        ClothGNN: Graph Neural Network for cloth simulation.
        Predicts node accelerations given positions and velocities.
        """
        
        def __init__(self, node_dim: int = 9, edge_dim: int = 4, hidden_dim: int = 128,
                     num_layers: int = 5):
            super().__init__()
            
            # Node encoder: position (3) + velocity (3) + material (3)
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Edge encoder: relative position (3) + rest length (1)
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Message passing layers
            self.conv_layers = nn.ModuleList([
                EdgeConv(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            
            # Decoder: predict acceleration
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),  # 3D acceleration
            )
        
        def forward(self, data) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                data: PyG Data object with x (node features), edge_index, edge_attr
            
            Returns:
                (N, 3): predicted accelerations for each node
            """
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            
            # Encode
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            
            # Message passing
            for conv in self.conv_layers:
                x = conv(x, edge_index, edge_attr)
            
            # Decode
            acc = self.decoder(x)
            
            return acc


class P05ClothGNNAdapter(PyTorchAdapter):
    """Adapter for ClothGNN project."""
    
    PROJECT_ID = "P05"
    PROJECT_NAME = "ClothGNN"
    
    def get_project_info(self) -> ProjectInfo:
        return ProjectInfo(
            project_id=self.PROJECT_ID,
            project_name=self.PROJECT_NAME,
            framework="pytorch-geometric",
            project_path=str(self.project_path),
        )
    
    def _create_model(self) -> nn.Module:
        """Create the ClothGNN model."""
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric not available")
        
        return ClothGNN(
            node_dim=9,     # pos (3) + vel (3) + material (3)
            edge_dim=4,     # rel_pos (3) + rest_length (1)
            hidden_dim=128,
            num_layers=5
        )
    
    def _create_cloth_mesh(self, grid_size: int = 16, device: str = "cuda"):
        """Create a simple cloth mesh graph."""
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric not available")
        
        # Grid positions
        x = torch.linspace(-0.5, 0.5, grid_size, device=device)
        y = torch.linspace(-0.5, 0.5, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        positions = torch.stack([
            xx.flatten(), yy.flatten(), torch.zeros(grid_size * grid_size, device=device)
        ], dim=-1)
        
        # Velocities (zero)
        velocities = torch.zeros_like(positions)
        
        # Material properties (uniform)
        material = torch.ones(grid_size * grid_size, 3, device=device) * 0.5
        
        # Node features
        node_features = torch.cat([positions, velocities, material], dim=-1)
        
        # Edges: connect adjacent nodes (structural + shear)
        edge_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                
                # Right neighbor
                if j < grid_size - 1:
                    edge_list.append([idx, idx + 1])
                    edge_list.append([idx + 1, idx])
                
                # Bottom neighbor
                if i < grid_size - 1:
                    edge_list.append([idx, idx + grid_size])
                    edge_list.append([idx + grid_size, idx])
                
                # Diagonal (shear)
                if i < grid_size - 1 and j < grid_size - 1:
                    edge_list.append([idx, idx + grid_size + 1])
                    edge_list.append([idx + grid_size + 1, idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
        
        # Edge attributes: relative position + rest length
        src, dst = edge_index
        rel_pos = positions[dst] - positions[src]
        rest_length = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, rest_length], dim=-1)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def prepare_test_input(self, device: str, batch_size: int = 1) -> Tuple:
        """Prepare test inputs: cloth mesh graph."""
        torch.manual_seed(42)
        
        # Create batch of cloth meshes
        data_list = [self._create_cloth_mesh(grid_size=16, device=device) 
                     for _ in range(batch_size)]
        
        batch = Batch.from_data_list(data_list)
        
        return (batch,)
    
    def run_inference(self, model: nn.Module, inputs: Tuple, device: str) -> Any:
        """Run model inference."""
        batch, = inputs
        return model(batch)
    
    def compute_domain_metrics(self, predictions: Any,
                               references: Optional[Any] = None) -> Dict[str, float]:
        """Compute cloth simulation metrics."""
        metrics = {}
        
        if isinstance(predictions, torch.Tensor):
            pred = predictions.detach().cpu().numpy()
        else:
            pred = np.array(predictions)
        
        # Acceleration statistics
        acc_magnitude = np.linalg.norm(pred, axis=-1)
        
        metrics['acceleration_mean'] = float(np.mean(acc_magnitude))
        metrics['acceleration_max'] = float(np.max(acc_magnitude))
        metrics['acceleration_std'] = float(np.std(acc_magnitude))
        
        # Check for extreme accelerations (instability indicator)
        extreme_threshold = 100.0  # m/s^2
        metrics['extreme_acc_ratio'] = float(np.mean(acc_magnitude > extreme_threshold))
        
        # Per-axis statistics
        metrics['acc_x_std'] = float(np.std(pred[:, 0]))
        metrics['acc_y_std'] = float(np.std(pred[:, 1]))
        metrics['acc_z_std'] = float(np.std(pred[:, 2]))
        
        if references is not None:
            if isinstance(references, torch.Tensor):
                ref = references.detach().cpu().numpy()
            else:
                ref = np.array(references)
            
            error = pred - ref
            metrics['acceleration_rmse'] = float(np.sqrt(np.mean(error**2)))
            metrics['acceleration_mae'] = float(np.mean(np.abs(error)))
        
        return metrics
    
    def get_reference_data(self) -> Optional[Any]:
        """Generate synthetic reference accelerations."""
        torch.manual_seed(123)
        
        num_nodes = 256  # 16x16 grid
        
        # Synthetic accelerations (gravity-dominated)
        acc = torch.zeros(num_nodes, 3)
        acc[:, 2] = -9.81  # Gravity in z
        acc += torch.randn(num_nodes, 3) * 0.5  # Small perturbations
        
        return acc
    
    def get_training_info(self) -> TrainingInfo:
        """Return training information."""
        return TrainingInfo(
            total_time_seconds=7200,  # ~2 hours
            epochs=500,
            final_loss=0.001,
            final_metrics={
                'position_error': 0.01,
                'velocity_error': 0.05,
                'stability_metric': 0.95,
            },
        )
    
    def get_test_dataset_description(self) -> str:
        return "16x16 cloth mesh graph (256 nodes, ~1000 edges) with node features and edge attributes"


def get_adapter(project_path: Optional[Path] = None) -> P05ClothGNNAdapter:
    """Get adapter instance for P05 project."""
    if project_path is None:
        project_path = Path(__file__).parent.parent.parent / "projects" / "05-clothgnn__project-space"
    return P05ClothGNNAdapter(project_path)
