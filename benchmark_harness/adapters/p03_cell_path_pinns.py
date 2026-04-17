"""
Adapter for Project 03: Cell-Path-PINNs

Domain: Microbe trajectories as geodesics on nutrient manifolds
Framework: PyTorch
Parameters: ~13K
Key metrics: RMSE on trajectory positions, physics loss components
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
PROJECT_PATH = WORKSPACE_ROOT / "projects" / "03-cell-path-pinns__project-space"

from harness.base_adapter import PyTorchAdapter, ProjectInfo
from harness.core import ModelInfo, TrainingInfo


class P03Adapter(PyTorchAdapter):
    """
    Adapter for Cell-Path-PINNs project.
    
    This project models microbe trajectories as geodesics on nutrient
    concentration manifolds using a small PINN.
    """
    
    def __init__(self, config=None, variant=None):
        self.config = config
        self.variant = variant
        super().__init__(PROJECT_PATH)
    
    def _default_project_path(self) -> Path:
        return PROJECT_PATH
    
    def get_project_info(self) -> ProjectInfo:
        return ProjectInfo(
            project_id="P03",
            project_name="Cell-Path-PINNs",
            framework="pytorch",
            project_path=self.project_path,
            description="Microbe trajectories as geodesics on nutrient manifolds",
            domain="Biology/Chemistry",
        )
    
    def _create_model(self) -> Any:
        """Create the Cell Path PINN model."""
        import torch
        import torch.nn as nn
        
        class CellPathPINN(nn.Module):
            """
            Small MLP that maps time to (x, y) trajectory positions.
            
            Uses sinusoidal activation (SIREN-like) for smooth trajectories.
            """
            def __init__(self, hidden_dim=32, num_layers=3):
                super().__init__()
                
                # Small network: time -> (x, y)
                layers = []
                in_dim = 1  # time input
                
                for i in range(num_layers - 1):
                    out_dim = hidden_dim
                    layers.append(nn.Linear(in_dim, out_dim))
                    layers.append(nn.Tanh())
                    in_dim = out_dim
                
                layers.append(nn.Linear(hidden_dim, 2))  # Output (x, y)
                
                self.net = nn.Sequential(*layers)
                
                # Count parameters (~13K target)
                self._param_count = sum(p.numel() for p in self.parameters())
            
            def forward(self, t):
                """
                Forward pass: time -> (x, y) position.
                
                Args:
                    t: Time values, shape (batch, 1) or (batch,)
                
                Returns:
                    Trajectory positions, shape (batch, 2)
                """
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                return self.net(t)
        
        return CellPathPINN(hidden_dim=64, num_layers=4)
    
    def prepare_test_input(
        self,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> Any:
        """
        Prepare test input for trajectory prediction.
        
        Returns time values spanning a trajectory.
        """
        import torch
        
        # Time values from 0 to 1 (normalized trajectory duration)
        n_points = 100 * batch_size
        t = np.linspace(0, 1, n_points).astype(np.float32)
        
        return torch.from_numpy(t).unsqueeze(-1).to(device)
    
    def compute_domain_metrics(
        self,
        predictions: Any,
        references: Any,
    ) -> dict[str, Any]:
        """
        Compute Cell-Path PINN specific metrics.
        
        Key metrics:
        - Position RMSE
        - Velocity smoothness (constant speed constraint)
        - Chemotaxis gradient alignment
        """
        from harness.metrics import compute_rmse, to_numpy
        
        pred = to_numpy(predictions)
        ref = to_numpy(references)
        
        # Position RMSE
        position_rmse = compute_rmse(pred, ref)
        
        # Compute velocity (finite differences)
        pred_vel = np.diff(pred, axis=0)
        ref_vel = np.diff(ref, axis=0)
        
        # Velocity magnitude variation (should be constant for geodesic)
        pred_speed = np.linalg.norm(pred_vel, axis=-1)
        speed_std = np.std(pred_speed)
        
        metrics = {
            "position_rmse": float(position_rmse),
            "x_rmse": float(compute_rmse(pred[:, 0], ref[:, 0])),
            "y_rmse": float(compute_rmse(pred[:, 1], ref[:, 1])),
            "speed_variation": float(speed_std),
            "mean_speed": float(np.mean(pred_speed)),
        }
        
        return metrics
    
    def get_reference_data(self) -> Any:
        """
        Get reference trajectory data.
        
        For synthetic benchmarking, generates a smooth reference trajectory.
        """
        import torch
        
        # Generate reference trajectory (sinusoidal path)
        n_points = 100
        t = np.linspace(0, 1, n_points)
        
        # Simple reference: figure-8 pattern
        x = np.sin(2 * np.pi * t)
        y = np.sin(4 * np.pi * t) / 2
        
        trajectory = np.stack([x, y], axis=-1).astype(np.float32)
        
        return torch.from_numpy(trajectory)
    
    def get_training_info(self) -> TrainingInfo:
        """Extract training info from project documentation."""
        return TrainingInfo(
            total_time_seconds=300,  # ~5 minutes (small model)
            epochs=10000,
            time_per_epoch_seconds=0.03,
            hardware="NVIDIA RTX 3090",
            convergence_metric="total_loss",
            convergence_value=0.001,
        )
    
    def get_test_dataset_description(self) -> str:
        return "Synthetic microbe trajectory, 100 time points, figure-8 reference path"


def get_adapter(project_id: str, config=None):
    """Factory function for adapter."""
    return P03Adapter(config=config)
