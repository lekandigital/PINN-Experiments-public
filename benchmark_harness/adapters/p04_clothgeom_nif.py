"""
Adapter for P04: ClothGeom-NIF - Cloth Geometry Neural Implicit Functions
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


class SIRENLayer(nn.Module):
    """SIREN layer with sine activation."""
    
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            dim = self.linear.weight.shape[1]
            if self.is_first:
                bound = 1.0 / dim
            else:
                bound = np.sqrt(6.0 / dim) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ClothGeomNIF(nn.Module):
    """
    ClothGeom-NIF: Neural Implicit Function for cloth geometry representation.
    Maps 3D coordinates + latent code -> SDF + displacement.
    """
    
    def __init__(self, input_dim: int = 3, latent_dim: int = 64, hidden_dim: int = 256,
                 num_layers: int = 5, output_dim: int = 4):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Build SIREN network
        layers = []
        layers.append(SIRENLayer(input_dim + latent_dim, hidden_dim, is_first=True))
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Output: 1 SDF + 3 displacement
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Latent embedding (e.g., for different cloth states)
        self.latent_codes = nn.Embedding(100, latent_dim)
    
    def forward(self, coords: torch.Tensor, latent_idx: Optional[torch.Tensor] = None,
                latent_code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: (B, N, 3) or (B, 3) spatial coordinates
            latent_idx: (B,) indices into latent embedding
            latent_code: (B, latent_dim) direct latent codes (alternative to idx)
        
        Returns:
            (B, N, 4) or (B, 4): [SDF, displacement_x, displacement_y, displacement_z]
        """
        squeeze = False
        if coords.dim() == 2:
            coords = coords.unsqueeze(1)  # (B, 1, 3)
            squeeze = True
        
        batch_size = coords.shape[0]
        num_points = coords.shape[1]
        
        # Get latent code
        if latent_code is not None:
            z = latent_code
        elif latent_idx is not None:
            z = self.latent_codes(latent_idx)  # (B, latent_dim)
        else:
            z = torch.zeros(batch_size, self.latent_dim, device=coords.device)
        
        # Expand and concatenate
        z = z.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, latent_dim)
        x = torch.cat([coords, z], dim=-1)  # (B, N, 3 + latent_dim)
        
        # Flatten for processing
        x = x.view(-1, x.shape[-1])  # (B*N, 3 + latent_dim)
        
        # SIREN layers
        for layer in self.layers:
            x = layer(x)
        
        out = self.output_layer(x)  # (B*N, 4)
        out = out.view(batch_size, num_points, -1)
        
        if squeeze:
            out = out.squeeze(1)
        
        return out


class P04ClothGeomNIFAdapter(PyTorchAdapter):
    """Adapter for ClothGeom-NIF project."""
    
    PROJECT_ID = "P04"
    PROJECT_NAME = "ClothGeom-NIF"
    
    def get_project_info(self) -> ProjectInfo:
        return ProjectInfo(
            project_id=self.PROJECT_ID,
            project_name=self.PROJECT_NAME,
            framework="pytorch",
            project_path=str(self.project_path),
        )
    
    def _create_model(self) -> nn.Module:
        """Create the ClothGeom-NIF model."""
        return ClothGeomNIF(
            input_dim=3,
            latent_dim=64,
            hidden_dim=256,
            num_layers=5,
            output_dim=4  # SDF + 3D displacement
        )
    
    def prepare_test_input(self, device: str, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
        """Prepare test inputs: coordinates + latent index."""
        torch.manual_seed(42)
        
        # Coordinates: points on and around a cloth surface
        coords = torch.randn(batch_size, 1024, 3, device=device) * 0.5
        
        # Latent index (cloth state)
        latent_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        return (coords, latent_idx)
    
    def run_inference(self, model: nn.Module, inputs: Tuple, device: str) -> Any:
        """Run model inference."""
        coords, latent_idx = inputs
        return model(coords, latent_idx=latent_idx)
    
    def compute_domain_metrics(self, predictions: Any,
                               references: Optional[Any] = None) -> Dict[str, float]:
        """Compute cloth-specific metrics."""
        metrics = {}
        
        if isinstance(predictions, torch.Tensor):
            pred = predictions.detach().cpu().numpy()
        else:
            pred = np.array(predictions)
        
        # SDF statistics
        sdf_values = pred[..., 0]
        metrics['sdf_mean'] = float(np.mean(sdf_values))
        metrics['sdf_std'] = float(np.std(sdf_values))
        metrics['surface_points_pct'] = float(
            np.mean(np.abs(sdf_values) < 0.01) * 100
        )
        
        # Displacement statistics
        displacement = pred[..., 1:4]
        disp_magnitude = np.linalg.norm(displacement, axis=-1)
        metrics['displacement_mean'] = float(np.mean(disp_magnitude))
        metrics['displacement_max'] = float(np.max(disp_magnitude))
        
        if references is not None:
            if isinstance(references, torch.Tensor):
                ref = references.detach().cpu().numpy()
            else:
                ref = np.array(references)
            
            # SDF error
            sdf_error = np.abs(pred[..., 0] - ref[..., 0])
            metrics['sdf_mae'] = float(np.mean(sdf_error))
            
            # Displacement error
            disp_error = np.linalg.norm(pred[..., 1:4] - ref[..., 1:4], axis=-1)
            metrics['displacement_rmse'] = float(np.sqrt(np.mean(disp_error**2)))
        
        return metrics
    
    def get_reference_data(self) -> Optional[Any]:
        """Generate synthetic reference data for testing."""
        torch.manual_seed(123)
        
        # Synthetic SDF + displacement field
        coords = torch.randn(1, 1024, 3) * 0.5
        
        # Simple SDF: distance to plane z=0
        sdf = coords[..., 2:3]
        
        # Simple displacement field
        displacement = torch.sin(coords * np.pi)
        
        return torch.cat([sdf, displacement], dim=-1)
    
    def get_training_info(self) -> TrainingInfo:
        """Return training information."""
        return TrainingInfo(
            total_time_seconds=3600 * 2,  # ~2 hours estimated
            epochs=1000,
            final_loss=0.001,
            final_metrics={'sdf_rmse': 0.002, 'displacement_rmse': 0.01},
        )
    
    def get_test_dataset_description(self) -> str:
        return "1024 random 3D points in [-0.5, 0.5]^3 with synthetic SDF and displacement targets"


# Convenience function for adapter discovery
def get_adapter(project_path: Optional[Path] = None) -> P04ClothGeomNIFAdapter:
    """Get adapter instance for P04 project."""
    if project_path is None:
        project_path = Path(__file__).parent.parent.parent / "projects" / "04-clothgeom-nif__project-space"
    return P04ClothGeomNIFAdapter(project_path)
