"""
Neural Implicit Field (NIF) Decoder for Cloth Geometry.

Ported from Project 04 (ClothGeom-NIF) with enhancements for Project 09.

This module implements a decoder that maps latent cloth states to signed 
distance field (SDF) values and variance estimates for LOD blending.

Architecture:
- Input: 3D coordinates (x, y, z) + latent code + optional time
- Hidden: 8 layers × 256 units with SIREN activations (ω₀=30)
- Output: SDF value (signed distance) + variance (for LOD)

Key Features:
- SIREN-style sine activations for high-frequency detail
- Latent modulation via concatenation
- Dual output heads (SDF + variance)
- Optional temporal conditioning for dynamic cloth
- FP16 mixed precision support
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repository root to path for implicit_fields import
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from implicit_fields import SirenLayer as BaseSirenLayer, SineActivation


class SirenLinear(nn.Module):
    """SIREN linear layer with proper initialization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NIFDecoder(nn.Module):
    """
    Neural Implicit Field decoder for cloth geometry.
    
    Maps 3D query points, latent cloth states, and optionally time
    to SDF values and uncertainty estimates.
    
    Args:
        latent_dim: Dimension of latent code (default: 128)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of hidden layers (default: 8)
        coord_dim: Input coordinate dimension (default: 3 for xyz)
        omega_0: SIREN frequency for first layer (default: 30.0)
        omega_hidden: SIREN frequency for hidden layers (default: 30.0)
        use_skip_connection: Add skip connection at middle layer
        use_positional_encoding: Apply positional encoding to coords
        num_frequencies: Number of frequency bands if using pos encoding
        use_time: Whether to include time as input (for dynamics)
    
    Example:
        >>> decoder = NIFDecoder(latent_dim=128, hidden_dim=256)
        >>> coords = torch.randn(1000, 3)  # Query points
        >>> latent = torch.randn(1000, 128)  # Latent codes
        >>> sdf, var = decoder(coords, latent)
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 8,
        coord_dim: int = 3,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        use_skip_connection: bool = True,
        use_positional_encoding: bool = False,
        num_frequencies: int = 6,
        use_time: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.coord_dim = coord_dim
        self.omega_0 = omega_0
        self.omega_hidden = omega_hidden
        self.use_skip_connection = use_skip_connection
        self.use_positional_encoding = use_positional_encoding
        self.num_frequencies = num_frequencies
        self.use_time = use_time
        
        # Compute input dimension
        if use_positional_encoding:
            self.encoded_coord_dim = coord_dim * (1 + 2 * num_frequencies)
        else:
            self.encoded_coord_dim = coord_dim
        
        # Total input: encoded coordinates + latent code + optional time
        input_dim = self.encoded_coord_dim + latent_dim
        if use_time:
            input_dim += 1
        
        self.input_dim = input_dim
        
        # Skip connection reinjection point
        self.skip_layer = num_layers // 2 if use_skip_connection else -1
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # First layer
        self.layers.append(SirenLinear(input_dim, hidden_dim, omega_0, is_first=True))
        self.activations.append(SineActivation(omega_0))
        
        # Hidden layers
        for i in range(1, num_layers):
            if i == self.skip_layer and use_skip_connection:
                layer_input_dim = hidden_dim + input_dim
            else:
                layer_input_dim = hidden_dim
            
            self.layers.append(SirenLinear(layer_input_dim, hidden_dim, omega_hidden, is_first=False))
            self.activations.append(SineActivation(omega_hidden))
        
        # SDF output head
        self.sdf_head = nn.Sequential(
            SirenLinear(hidden_dim, hidden_dim // 2, omega_hidden, is_first=False),
            SineActivation(omega_hidden),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Variance output head
        self.variance_head = nn.Sequential(
            SirenLinear(hidden_dim, hidden_dim // 2, omega_hidden, is_first=False),
            SineActivation(omega_hidden),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_output_layers()
    
    def _init_output_layers(self):
        """Initialize output layers for stable training."""
        for head in [self.sdf_head, self.variance_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to coordinates."""
        if not self.use_positional_encoding:
            return coords
        
        encoded = [coords]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * math.pi
            encoded.append(torch.sin(freq * coords))
            encoded.append(torch.cos(freq * coords))
        
        return torch.cat(encoded, dim=-1)
    
    def forward(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the NIF decoder.
        
        Args:
            coords: 3D query coordinates [B, 3] in [-1, 1]
            latent: Latent cloth state [B, latent_dim]
            time: Optional time values [B, 1] for temporal models
            return_variance: Whether to compute variance output
        
        Returns:
            sdf: Signed distance values [B, 1]
            variance: Uncertainty estimates [B, 1] or None
        """
        # Apply positional encoding
        encoded_coords = self.positional_encoding(coords)
        
        # Concatenate inputs
        if self.use_time and time is not None:
            if time.dim() == 1:
                time = time.unsqueeze(-1)
            x = torch.cat([encoded_coords, latent, time], dim=-1)
        else:
            x = torch.cat([encoded_coords, latent], dim=-1)
        
        skip_input = x
        
        # Forward through hidden layers
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            if i == self.skip_layer and self.use_skip_connection:
                x = torch.cat([x, skip_input], dim=-1)
            x = activation(layer(x))
        
        # Compute outputs
        sdf = self.sdf_head(x)
        
        if return_variance:
            variance = F.softplus(self.variance_head(x)) + 1e-6
            return sdf, variance
        else:
            return sdf, None
    
    def forward_batched(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        batch_size: int = 65536,
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Memory-efficient batched forward pass."""
        n_points = coords.shape[0]
        device = coords.device
        
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        sdf_list = []
        var_list = [] if return_variance else None
        
        for i in range(0, n_points, batch_size):
            end = min(i + batch_size, n_points)
            batch_coords = coords[i:end]
            batch_latent = latent.expand(batch_coords.shape[0], -1)
            
            if time is not None:
                if time.dim() == 0:
                    batch_time = time.unsqueeze(0).expand(batch_coords.shape[0], 1)
                elif time.dim() == 1:
                    batch_time = time.unsqueeze(1).expand(batch_coords.shape[0], 1)
                else:
                    batch_time = time.expand(batch_coords.shape[0], -1)
            else:
                batch_time = None
            
            sdf, var = self.forward(batch_coords, batch_latent, batch_time, return_variance)
            sdf_list.append(sdf)
            
            if return_variance:
                var_list.append(var)
        
        sdf = torch.cat(sdf_list, dim=0)
        variance = torch.cat(var_list, dim=0) if return_variance else None
        
        return sdf, variance
    
    def compute_gradient(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SDF value and its gradient w.r.t. coordinates.
        
        The gradient gives the surface normal direction.
        """
        coords = coords.requires_grad_(True)
        sdf, _ = self.forward(coords, latent, time, return_variance=False)
        
        gradient = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return sdf, gradient
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving/loading."""
        return {
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'coord_dim': self.coord_dim,
            'omega_0': self.omega_0,
            'omega_hidden': self.omega_hidden,
            'use_skip_connection': self.use_skip_connection,
            'use_positional_encoding': self.use_positional_encoding,
            'num_frequencies': self.num_frequencies,
            'use_time': self.use_time,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NIFDecoder':
        """Create model from configuration dict."""
        return cls(**config)


def create_nif_decoder(
    config: Optional[Dict[str, Any]] = None,
    temporal: bool = False,
) -> NIFDecoder:
    """
    Factory function to create NIF decoder with default or custom config.
    
    Args:
        config: Configuration dict (uses defaults if None)
        temporal: Whether to enable time input for dynamics
    
    Returns:
        Configured NIFDecoder instance
    """
    default_config = {
        'latent_dim': 128,
        'hidden_dim': 256,
        'num_layers': 8,
        'coord_dim': 3,
        'omega_0': 30.0,
        'omega_hidden': 30.0,
        'use_skip_connection': True,
        'use_positional_encoding': False,
        'num_frequencies': 6,
        'use_time': temporal,
    }
    
    if config is not None:
        default_config.update(config)
    
    return NIFDecoder(**default_config)
