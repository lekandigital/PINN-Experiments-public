"""
Adapter for Project 09: HGNN-NIF-Cloth

Domain: Hybrid hierarchical GNN + SIREN implicit field for cloth simulation
Framework: PyTorch
Parameters: ~136K
Key metrics: 465 FPS, SDF RMSE, temporal coherence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
PROJECT_PATH = WORKSPACE_ROOT / "projects" / "09-hgnn-nif-cloth__project-space"

from harness.base_adapter import PyTorchAdapter, ProjectInfo
from harness.core import ModelInfo, TrainingInfo


class P09Adapter(PyTorchAdapter):
    """
    Adapter for HGNN-NIF-Cloth project.
    
    This is the flagship cloth simulation model combining:
    - Hierarchical GNN encoder for mesh processing
    - SIREN decoder for implicit surface representation
    """
    
    def __init__(self, config=None, variant=None):
        self.config = config
        self.variant = variant  # "anim" for animation variant
        super().__init__(PROJECT_PATH)
    
    def _default_project_path(self) -> Path:
        return PROJECT_PATH
    
    def get_project_info(self) -> ProjectInfo:
        name = "HGNN-NIF-Cloth"
        if self.variant == "anim":
            name += "-Animation"
        
        return ProjectInfo(
            project_id="P09" if not self.variant else f"P09-{self.variant}",
            project_name=name,
            framework="pytorch",
            project_path=self.project_path,
            description="Hybrid hierarchical GNN + SIREN for cloth SDF",
            domain="Cloth Simulation",
        )
    
    def _create_model(self) -> Any:
        """Create the HGNN-NIF-Cloth model."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class SIRENLayer(nn.Module):
            """SIREN layer with sinusoidal activation."""
            def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
                super().__init__()
                self.omega_0 = omega_0
                self.is_first = is_first
                self.linear = nn.Linear(in_features, out_features)
                self._init_weights()
            
            def _init_weights(self):
                with torch.no_grad():
                    if self.is_first:
                        self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                                     1 / self.linear.in_features)
                    else:
                        self.linear.weight.uniform_(
                            -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                             np.sqrt(6 / self.linear.in_features) / self.omega_0
                        )
            
            def forward(self, x):
                return torch.sin(self.omega_0 * self.linear(x))
        
        class SIREN(nn.Module):
            """SIREN network for implicit surface representation."""
            def __init__(self, in_features=3, hidden_features=64, hidden_layers=3, out_features=1):
                super().__init__()
                
                layers = [SIRENLayer(in_features, hidden_features, is_first=True)]
                for _ in range(hidden_layers - 1):
                    layers.append(SIRENLayer(hidden_features, hidden_features))
                
                self.net = nn.Sequential(*layers)
                self.final = nn.Linear(hidden_features, out_features)
                
                with torch.no_grad():
                    self.final.weight.uniform_(
                        -np.sqrt(6 / hidden_features) / 30.0,
                         np.sqrt(6 / hidden_features) / 30.0
                    )
            
            def forward(self, x):
                return self.final(self.net(x))
        
        class HGNNEncoder(nn.Module):
            """Simplified hierarchical GNN encoder."""
            def __init__(self, node_dim=3, hidden_dim=64, latent_dim=64, num_layers=3):
                super().__init__()
                
                self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                
                # Message passing layers (simplified)
                self.mp_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(num_layers)
                ])
                
                self.pool = nn.Sequential(
                    nn.Linear(hidden_dim, latent_dim),
                    nn.Tanh(),
                )
            
            def forward(self, node_features, edge_index=None):
                """
                Encode mesh to latent code.
                
                Args:
                    node_features: (batch, num_nodes, 3) node positions
                    edge_index: Optional edge connectivity
                
                Returns:
                    latent: (batch, latent_dim) latent code
                """
                batch_size = node_features.shape[0]
                
                # Encode nodes
                h = self.node_encoder(node_features)  # (batch, N, hidden)
                
                # Simple aggregation (mean pooling) as stand-in for message passing
                for layer in self.mp_layers:
                    # Self-attention style update
                    h_mean = h.mean(dim=1, keepdim=True).expand_as(h)
                    h = layer(torch.cat([h, h_mean], dim=-1))
                
                # Global pooling
                latent = self.pool(h.mean(dim=1))  # (batch, latent_dim)
                
                return latent
        
        class HGNN_NIF_ClothModel(nn.Module):
            """
            Combined HGNN encoder + SIREN decoder model.
            
            Architecture:
            1. HGNN encodes mesh to latent code
            2. SIREN decodes (latent + xyz) to SDF value
            """
            def __init__(
                self,
                node_dim=3,
                hidden_dim=64,
                latent_dim=64,
                siren_hidden=64,
                siren_layers=3,
            ):
                super().__init__()
                
                self.encoder = HGNNEncoder(
                    node_dim=node_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                )
                
                # SIREN takes (xyz + latent) as input
                self.decoder = SIREN(
                    in_features=3 + latent_dim,
                    hidden_features=siren_hidden,
                    hidden_layers=siren_layers,
                    out_features=1,
                )
            
            def forward(self, mesh_nodes, query_points):
                """
                Forward pass.
                
                Args:
                    mesh_nodes: (batch, num_nodes, 3) mesh vertex positions
                    query_points: (batch, num_queries, 3) query coordinates
                
                Returns:
                    sdf: (batch, num_queries, 1) signed distance values
                """
                # Encode mesh
                latent = self.encoder(mesh_nodes)  # (batch, latent_dim)
                
                # Expand latent for each query point
                batch_size, num_queries, _ = query_points.shape
                latent_expanded = latent.unsqueeze(1).expand(-1, num_queries, -1)
                
                # Concatenate query points with latent
                decoder_input = torch.cat([query_points, latent_expanded], dim=-1)
                
                # Decode to SDF
                sdf = self.decoder(decoder_input)
                
                return {
                    'sdf': sdf,
                    'latent': latent,
                }
        
        return HGNN_NIF_ClothModel(
            node_dim=3,
            hidden_dim=64,
            latent_dim=64,
            siren_hidden=64,
            siren_layers=3,
        )
    
    def prepare_test_input(
        self,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> Any:
        """
        Prepare test input for cloth model.
        
        Returns:
            Tuple of (mesh_nodes, query_points)
        """
        import torch
        
        # Cloth mesh: 32x32 grid
        num_nodes = 1024
        num_queries = 1000
        
        # Generate cloth mesh nodes (flat grid with some deformation)
        grid_size = int(np.sqrt(num_nodes))
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Add some z variation (draped cloth)
        zz = 0.1 * np.sin(np.pi * xx) * np.cos(np.pi * yy)
        
        mesh_nodes = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
        mesh_nodes = mesh_nodes.astype(np.float32)
        mesh_nodes = np.tile(mesh_nodes[np.newaxis], (batch_size, 1, 1))
        
        # Generate query points in bounding box
        query_points = np.random.uniform(-1.5, 1.5, (batch_size, num_queries, 3))
        query_points = query_points.astype(np.float32)
        
        return (
            torch.from_numpy(mesh_nodes).to(device),
            torch.from_numpy(query_points).to(device),
        )
    
    def run_inference(self, model: Any, input_data: Any) -> Any:
        """Run inference with the HGNN-NIF model."""
        import torch
        
        model.eval()
        mesh_nodes, query_points = input_data
        
        with torch.no_grad():
            output = model(mesh_nodes, query_points)
        
        return output
    
    def compute_domain_metrics(
        self,
        predictions: Any,
        references: Any,
    ) -> dict[str, Any]:
        """
        Compute HGNN-NIF-Cloth specific metrics.
        
        Key metrics:
        - SDF RMSE
        - Surface reconstruction quality (Chamfer distance approximation)
        - Latent space statistics
        """
        from harness.metrics import compute_rmse, to_numpy
        
        # Extract SDF values
        if isinstance(predictions, dict):
            pred_sdf = to_numpy(predictions['sdf'])
            latent = to_numpy(predictions.get('latent', np.zeros(64)))
        else:
            pred_sdf = to_numpy(predictions)
            latent = np.zeros(64)
        
        ref_sdf = to_numpy(references)
        
        metrics = {
            "sdf_rmse": float(compute_rmse(pred_sdf, ref_sdf)),
            "sdf_max_error": float(np.max(np.abs(pred_sdf - ref_sdf))),
            "sdf_mean_abs_error": float(np.mean(np.abs(pred_sdf - ref_sdf))),
            "latent_norm": float(np.linalg.norm(latent)),
            "latent_std": float(np.std(latent)),
        }
        
        # Surface points (where SDF ≈ 0)
        surface_threshold = 0.05
        pred_surface = np.abs(pred_sdf) < surface_threshold
        ref_surface = np.abs(ref_sdf) < surface_threshold
        
        # IoU of surface regions
        intersection = np.sum(pred_surface & ref_surface)
        union = np.sum(pred_surface | ref_surface)
        if union > 0:
            metrics["surface_iou"] = float(intersection / union)
        
        return metrics
    
    def get_reference_data(self) -> Any:
        """
        Get reference SDF data.
        
        For synthetic benchmarking, generates analytical SDF for a sphere.
        """
        import torch
        
        # Generate reference SDF (sphere centered at origin)
        num_queries = 1000
        query_points = np.random.uniform(-1.5, 1.5, (1, num_queries, 3))
        
        # Sphere SDF: distance to surface
        radius = 0.8
        distances = np.linalg.norm(query_points, axis=-1) - radius
        
        return torch.from_numpy(distances.astype(np.float32)).unsqueeze(-1)
    
    def get_training_info(self) -> TrainingInfo:
        """Extract training info from project documentation."""
        return TrainingInfo(
            total_time_seconds=1800,  # ~30 minutes
            epochs=500,
            time_per_epoch_seconds=3.6,
            hardware="NVIDIA RTX 3090",
            convergence_metric="total_loss",
            convergence_value=0.001,
        )
    
    def get_test_dataset_description(self) -> str:
        desc = "Synthetic cloth mesh (32x32 grid, 1024 vertices), 1000 query points, sphere SDF reference"
        if self.variant == "anim":
            desc += " [Animation variant with temporal GRU]"
        return desc


def get_adapter(project_id: str, config=None):
    """Factory function for adapter."""
    variant = None
    if "anim" in project_id.lower():
        variant = "anim"
    return P09Adapter(config=config, variant=variant)
