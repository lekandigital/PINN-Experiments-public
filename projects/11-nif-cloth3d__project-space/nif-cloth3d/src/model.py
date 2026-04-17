"""
NIF-Cloth3D-Interactive: Sine-Activated MLP (SIREN) for Cloth Simulation

This module implements a SIREN-style neural network that maps spatial coordinates,
time, force vectors, and material properties to cloth displacement predictions.

Architecture follows "Implicit Neural Representations with Periodic Activation Functions"
(Sitzmann et al., NeurIPS 2020) with modifications for physics-informed cloth simulation.

Uses the shared implicit_fields library for core SIREN components.

Input: 8D tensor (x, y, z, t, force_x, force_y, force_z, material_id)
Output: 3D displacement vector (dx, dy, dz)
"""

import sys
from pathlib import Path

# Add repository root to path for implicit_fields import
repo_root = Path(__file__).parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Import from shared library
from implicit_fields import SirenLayer as _BaseSirenLayer, SirenNetwork


class SineLayer(nn.Module):
    """
    Sine activation layer with SIREN-style initialization.
    
    Wrapper for P11 backward compatibility - uses zero bias init instead of
    uniform bias init (project-specific convention).
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        w0: Frequency scaling factor (default 30.0 for first layer, 1.0 for hidden)
        is_first: Whether this is the first layer (affects initialization)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        w0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.is_first = is_first
        
        # Use shared library layer
        self._layer = _BaseSirenLayer(
            in_features=in_features,
            out_features=out_features,
            omega=w0,
            is_first=is_first,
            bias=True
        )
        # Override bias init to zero (P11 convention)
        with torch.no_grad():
            if self._layer.linear.bias is not None:
                self._layer.linear.bias.zero_()
    
    @property
    def linear(self):
        """Expose linear layer for compatibility."""
        return self._layer.linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation followed by sine activation."""
        return self._layer(x)


class SineMLP(nn.Module):
    """
    SIREN-based MLP for cloth deformation prediction.
    
    This network takes spatial coordinates, temporal information, external forces,
    and material properties as input, and outputs predicted vertex displacements.
    
    The architecture uses sine activations throughout hidden layers to enable
    learning of high-frequency cloth deformations (wrinkles, folds).
    
    Args:
        in_dim: Input dimension (default 8: xyz + t + force_xyz + material_id)
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension (default 3: displacement xyz)
        n_layers: Number of hidden layers
        w0_first: Frequency for first layer (default 30.0)
        w0_hidden: Frequency for hidden layers (default 1.0)
        use_conditioning: Whether to use FiLM-style conditioning (future extension)
    """
    
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 256,
        out_dim: int = 3,
        n_layers: int = 6,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
        use_conditioning: bool = False
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        
        # Build network layers
        layers = []
        
        # First layer with higher frequency
        layers.append(SineLayer(in_dim, hidden_dim, w0=w0_first, is_first=True))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, w0=w0_hidden, is_first=False))
        
        self.net = nn.Sequential(*layers)
        
        # Final linear layer (no activation) for displacement output
        self.final = nn.Linear(hidden_dim, out_dim)
        self._init_final_layer()
    
    def _init_final_layer(self):
        """Initialize final layer with small weights for stable training."""
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.hidden_dim) / 30.0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict displacement from input features.
        
        Args:
            x: Input tensor of shape (batch, in_dim)
               Expected format: [x, y, z, t, fx, fy, fz, material_id]
        
        Returns:
            Displacement tensor of shape (batch, 3)
        """
        features = self.net(x)
        displacement = self.final(features)
        return displacement
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns intermediate features.
        Useful for visualization and debugging.
        """
        features = self.net(x)
        displacement = self.final(features)
        return displacement, features


class ConditionedSineMLP(nn.Module):
    """
    SIREN MLP with FiLM-style conditioning for force and material modulation.
    
    This variant applies learned affine transformations (scale and shift) based on
    conditioning inputs (forces, material properties), enabling the network to
    adapt its behavior based on external conditions.
    
    FiLM (Feature-wise Linear Modulation) from Perez et al., 2018.
    """
    
    def __init__(
        self,
        coord_dim: int = 4,       # x, y, z, t
        cond_dim: int = 4,        # force_xyz + material_id
        hidden_dim: int = 256,
        out_dim: int = 3,
        n_layers: int = 6,
        w0: float = 30.0
    ):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        
        # Coordinate encoder (main SIREN branch)
        self.coord_layers = nn.ModuleList()
        self.coord_layers.append(SineLayer(coord_dim, hidden_dim, w0=w0, is_first=True))
        for _ in range(n_layers - 1):
            self.coord_layers.append(SineLayer(hidden_dim, hidden_dim, w0=1.0))
        
        # Conditioning encoder (produces scale and shift for FiLM)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # gamma and beta
        )
        
        # Output layer
        self.final = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.
        
        Args:
            coords: Spatial-temporal coordinates (batch, 4) - [x, y, z, t]
            cond: Conditioning inputs (batch, 4) - [fx, fy, fz, material_id]
        
        Returns:
            Displacement prediction (batch, 3)
        """
        # Encode conditioning to get FiLM parameters
        film_params = self.cond_encoder(cond)
        gamma = film_params[:, :self.hidden_dim]  # scale
        beta = film_params[:, self.hidden_dim:]    # shift
        
        # Process coordinates through SIREN layers
        h = coords
        for i, layer in enumerate(self.coord_layers):
            h = layer(h)
            # Apply FiLM modulation after each layer
            h = gamma * h + beta
        
        return self.final(h)


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Dictionary with model configuration
            - hidden_dim: Hidden layer size
            - n_layers: Number of layers
            - w0: Sine frequency
            - use_conditioning: Whether to use FiLM conditioning
    
    Returns:
        Configured SineMLP or ConditionedSineMLP model
    """
    use_conditioning = config.get('use_conditioning', False)
    
    if use_conditioning:
        return ConditionedSineMLP(
            coord_dim=config.get('coord_dim', 4),
            cond_dim=config.get('cond_dim', 4),
            hidden_dim=config.get('hidden_dim', 256),
            out_dim=config.get('out_dim', 3),
            n_layers=config.get('n_layers', 6),
            w0=config.get('w0', 30.0)
        )
    else:
        return SineMLP(
            in_dim=config.get('in_dim', 8),
            hidden_dim=config.get('hidden_dim', 256),
            out_dim=config.get('out_dim', 3),
            n_layers=config.get('n_layers', 6),
            w0_first=config.get('w0', 30.0),
            w0_hidden=config.get('w0_hidden', 1.0)
        )


if __name__ == "__main__":
    # Quick test
    print("Testing SineMLP...")
    model = SineMLP(in_dim=8, hidden_dim=256, out_dim=3, n_layers=6)
    
    # Test forward pass
    batch_size = 100
    x = torch.randn(batch_size, 8)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        out = model(x)
        print(f"CUDA test passed. Device: {out.device}")
