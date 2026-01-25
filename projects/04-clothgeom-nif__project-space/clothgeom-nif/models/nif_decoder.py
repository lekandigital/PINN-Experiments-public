"""
Neural Implicit Field (NIF) Decoder for Cloth Geometry

This module implements the main decoder that maps latent cloth states
(node positions + edge strains) to signed distance field (SDF) values
and variance estimates for LOD blending.

Architecture:
- Input: 3D coordinates (x, y, z) + latent code
- Hidden: 8 layers × 256 units with SIREN activations (ω₀=30)
- Output: SDF value (signed distance) + variance (for LOD)

Key Features:
- SIREN-style sine activations for high-frequency detail
- Latent modulation via concatenation
- Dual output heads (SDF + variance)
- FP16 mixed precision support
- Efficient batched inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .siren import SirenLinear, SineActivation


class NIFDecoder(nn.Module):
    """
    Neural Implicit Field decoder for cloth geometry.
    
    Maps 3D query points and latent cloth states to SDF values
    and uncertainty estimates.
    
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
        num_frequencies: int = 6
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
        
        # Compute input dimension
        if use_positional_encoding:
            # sin and cos for each frequency band, for each coordinate
            self.encoded_coord_dim = coord_dim * (1 + 2 * num_frequencies)
        else:
            self.encoded_coord_dim = coord_dim
        
        # Total input: encoded coordinates + latent code
        input_dim = self.encoded_coord_dim + latent_dim
        
        # Skip connection reinjection point (middle of network)
        self.skip_layer = num_layers // 2 if use_skip_connection else -1
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # First layer
        self.layers.append(SirenLinear(
            input_dim, hidden_dim, omega_0=omega_0, is_first=True
        ))
        self.activations.append(SineActivation(omega_0))
        
        # Hidden layers
        for i in range(1, num_layers):
            # Account for skip connection input
            if i == self.skip_layer and use_skip_connection:
                layer_input_dim = hidden_dim + input_dim
            else:
                layer_input_dim = hidden_dim
            
            self.layers.append(SirenLinear(
                layer_input_dim, hidden_dim,
                omega_0=omega_hidden, is_first=False
            ))
            self.activations.append(SineActivation(omega_hidden))
        
        # Output heads
        # SDF head: predicts signed distance
        self.sdf_head = nn.Sequential(
            SirenLinear(hidden_dim, hidden_dim // 2, omega_hidden, is_first=False),
            SineActivation(omega_hidden),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Variance head: predicts uncertainty (positive via softplus)
        self.variance_head = nn.Sequential(
            SirenLinear(hidden_dim, hidden_dim // 2, omega_hidden, is_first=False),
            SineActivation(omega_hidden),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize output layer weights to small values
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
        """
        Apply positional encoding to coordinates.
        
        PE(x) = [x, sin(2^0 π x), cos(2^0 π x), ..., sin(2^L π x), cos(2^L π x)]
        
        Args:
            coords: Input coordinates [B, coord_dim]
        
        Returns:
            Encoded coordinates [B, encoded_coord_dim]
        """
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
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the NIF decoder.
        
        Args:
            coords: 3D query coordinates [B, 3] in [-1, 1]
            latent: Latent cloth state [B, latent_dim]
            return_variance: Whether to compute variance output
        
        Returns:
            sdf: Signed distance values [B, 1]
            variance: Uncertainty estimates [B, 1] or None
        """
        # Apply positional encoding
        encoded_coords = self.positional_encoding(coords)
        
        # Concatenate coordinates and latent code
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
            # Softplus to ensure positive variance
            variance = F.softplus(self.variance_head(x)) + 1e-6
            return sdf, variance
        else:
            return sdf, None
    
    def forward_batched(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor,
        batch_size: int = 65536,
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Memory-efficient batched forward pass.
        
        Useful for inference on large point clouds or high-resolution grids.
        
        Args:
            coords: Query coordinates [N, 3]
            latent: Single latent code [latent_dim] (will be expanded)
            batch_size: Points to process per batch
            return_variance: Whether to compute variance
        
        Returns:
            sdf: [N, 1]
            variance: [N, 1] or None
        """
        n_points = coords.shape[0]
        device = coords.device
        
        # Expand latent to match coord dimension
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        sdf_list = []
        var_list = [] if return_variance else None
        
        for i in range(0, n_points, batch_size):
            end = min(i + batch_size, n_points)
            batch_coords = coords[i:end]
            
            # Expand latent to batch size
            batch_latent = latent.expand(batch_coords.shape[0], -1)
            
            sdf, var = self.forward(batch_coords, batch_latent, return_variance)
            sdf_list.append(sdf)
            
            if return_variance:
                var_list.append(var)
        
        sdf = torch.cat(sdf_list, dim=0)
        variance = torch.cat(var_list, dim=0) if return_variance else None
        
        return sdf, variance
    
    def compute_gradient(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SDF value and its gradient w.r.t. coordinates.
        
        The gradient of SDF gives the surface normal direction,
        and its magnitude should be 1 for valid SDFs (Eikonal constraint).
        
        Args:
            coords: Query coordinates [B, 3] (requires grad)
            latent: Latent code [B, latent_dim]
        
        Returns:
            sdf: SDF values [B, 1]
            gradient: SDF gradient [B, 3]
        """
        coords = coords.requires_grad_(True)
        
        sdf, _ = self.forward(coords, latent, return_variance=False)
        
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
            'num_frequencies': self.num_frequencies
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NIFDecoder':
        """Create model from configuration dict."""
        return cls(**config)


class NIFDecoderWithHash(NIFDecoder):
    """
    NIF Decoder with optional hash encoding for faster inference.
    
    Falls back to standard encoding if tinycudann is not available.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 8,
        use_hash_encoding: bool = True,
        hash_log2_size: int = 19,
        hash_n_levels: int = 16,
        hash_n_features: int = 2,
        **kwargs
    ):
        # Try to import tinycudann
        self.tcnn_available = False
        if use_hash_encoding:
            try:
                import tinycudann as tcnn
                self.tcnn_available = True
            except ImportError:
                print("Warning: tinycudann not available, using standard encoding")
        
        # Modify input dim if using hash encoding
        if self.tcnn_available:
            self.hash_encoding = self._create_hash_encoding(
                hash_log2_size, hash_n_levels, hash_n_features
            )
            kwargs['use_positional_encoding'] = False
            # Hash encoding output dim
            encoded_dim = hash_n_levels * hash_n_features
        else:
            self.hash_encoding = None
            encoded_dim = None
        
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )
        
        # Rebuild first layer if using hash encoding
        if self.tcnn_available:
            input_dim = encoded_dim + latent_dim
            self.layers[0] = SirenLinear(
                input_dim, hidden_dim, omega_0=self.omega_0, is_first=True
            )
            # Update skip connection dimension
            if self.use_skip_connection:
                skip_layer = self.layers[self.skip_layer]
                new_input = hidden_dim + input_dim
                self.layers[self.skip_layer] = SirenLinear(
                    new_input, hidden_dim,
                    omega_0=self.omega_hidden, is_first=False
                )
    
    def _create_hash_encoding(self, log2_size, n_levels, n_features):
        """Create tinycudann hash encoding."""
        import tinycudann as tcnn
        
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features,
            "log2_hashmap_size": log2_size,
            "base_resolution": 16,
            "per_level_scale": 1.5
        }
        
        return tcnn.Encoding(3, encoding_config)
    
    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Use hash encoding if available, else fall back to base."""
        if self.tcnn_available and self.hash_encoding is not None:
            # Normalize coords to [0, 1] for hash encoding
            coords_normalized = (coords + 1) / 2
            return self.hash_encoding(coords_normalized)
        else:
            return super().positional_encoding(coords)


def create_nif_decoder(
    config: Optional[Dict[str, Any]] = None,
    use_hash: bool = False
) -> NIFDecoder:
    """
    Factory function to create NIF decoder with default or custom config.
    
    Args:
        config: Configuration dict (uses defaults if None)
        use_hash: Whether to use hash encoding (requires tinycudann)
    
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
        'num_frequencies': 6
    }
    
    if config is not None:
        default_config.update(config)
    
    if use_hash:
        return NIFDecoderWithHash(**default_config)
    else:
        return NIFDecoder(**default_config)
