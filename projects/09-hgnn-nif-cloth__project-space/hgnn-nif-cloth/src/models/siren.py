"""
SIREN: Sinusoidal Representation Networks for Implicit Fields

Re-exports from the shared implicit_fields library with backward-compatible
interfaces specific to the HGNN-NIF-Cloth project.

Original implementation based on "Implicit Neural Representations 
with Periodic Activation Functions" (Sitzmann et al., 2020).
"""

import sys
from pathlib import Path

# Add repository root to path for implicit_fields import
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

# Import from shared library
from implicit_fields import (
    SirenLayer as _BaseSirenLayer,
    LatentConditionedSiren,
    FourierFeatureEncoding,
)


class SirenLayer(nn.Module):
    """
    A single SIREN layer with sinusoidal activation.
    
    Wrapper for backward compatibility with P09 convention (c=6.0 parameter).
    Uses the shared implicit_fields library underneath.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        is_first: Whether this is the first layer
        w0: Frequency scaling factor for sine
        c: Initialization constant (default 6 for uniform distribution)
        
    Example:
        >>> layer = SirenLayer(3, 256, is_first=True, w0=30)
        >>> x = torch.randn(1000, 3)  # 1000 3D points
        >>> out = layer(x)  # (1000, 256)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        w0: float = 30.0,
        c: float = 6.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.w0 = w0
        self.c = c
        
        # Use shared library SirenLayer
        # Note: shared lib uses omega and sqrt(6/n), this uses c parameter
        self._layer = _BaseSirenLayer(
            in_features=in_features,
            out_features=out_features,
            omega=w0,
            is_first=is_first,
            bias=True,
            c=c  # Pass through the c parameter
        )
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transform followed by scaled sine activation."""
        return self._layer(x)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.in_features}, {self.out_features}, "
                f"is_first={self.is_first}, w0={self.w0})")


class SIRENDecoder(nn.Module):
    """
    Latent-conditioned SIREN decoder for implicit surface reconstruction.
    
    Takes 3D coordinates and a latent code, outputs SDF values.
    The latent code allows representing different cloth configurations
    with the same network.
    
    Architecture:
        Input: concat(xyz, latent) -> SIREN layers -> SDF value
        
    Args:
        coord_dim: Dimension of input coordinates (default: 3 for XYZ)
        latent_dim: Dimension of latent conditioning vector
        hidden_dim: Hidden layer dimension
        hidden_layers: Number of hidden SIREN layers
        out_dim: Output dimension (1 for SDF, 3 for RGB, etc.)
        w0_first: Frequency for first SIREN layer
        w0_hidden: Frequency for hidden SIREN layers
        use_fourier: Whether to apply Fourier feature encoding to coords
        
    Example:
        >>> decoder = SIRENDecoder(coord_dim=3, latent_dim=64)
        >>> points = torch.randn(1000, 3)  # Query points
        >>> latent = torch.randn(64)        # Cloth state encoding
        >>> sdf = decoder(points, latent)   # (1000, 1)
    """
    
    def __init__(
        self,
        coord_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        out_dim: int = 1,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
        use_fourier: bool = False,
        fourier_scale: float = 1.0,
        num_fourier_features: int = 64
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier
        
        # Optional Fourier feature encoding
        if use_fourier:
            self.fourier_B = nn.Parameter(
                torch.randn(coord_dim, num_fourier_features) * fourier_scale,
                requires_grad=False
            )
            input_dim = num_fourier_features * 2 + latent_dim
        else:
            input_dim = coord_dim + latent_dim
            
        # Build SIREN network
        layers = []
        
        # First layer with special initialization
        layers.append(SirenLayer(input_dim, hidden_dim, is_first=True, w0=w0_first))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_dim, hidden_dim, is_first=False, w0=w0_hidden))
            
        self.net = nn.Sequential(*layers)
        
        # Final linear layer (no activation) to output SDF
        self.final = nn.Linear(hidden_dim, out_dim)
        self._init_final()
        
    def _init_final(self):
        """Initialize final layer to output small values initially."""
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.hidden_dim)
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()
                
    def forward(
        self, 
        coords: torch.Tensor, 
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SDF values at given coordinates conditioned on latent.
        
        Args:
            coords: Query coordinates of shape (N, 3) or (B, N, 3)
            latent: Latent code of shape (D,) or (B, D)
            
        Returns:
            SDF values of shape (N, out_dim) or (B, N, out_dim)
        """
        # Handle different input shapes
        batched = coords.dim() == 3
        if not batched:
            coords = coords.unsqueeze(0)  # (1, N, 3)
            
        B, N, C = coords.shape
        
        # Ensure latent has batch dimension
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)  # (1, D)
            
        # Expand latent to match number of query points
        # latent: (B, D) -> (B, N, D)
        latent_expanded = latent.unsqueeze(1).expand(B, N, -1)
        
        # Apply Fourier features if enabled
        if self.use_fourier:
            # coords: (B, N, 3) -> fourier: (B, N, 2*num_fourier)
            proj = torch.matmul(coords, self.fourier_B)  # (B, N, num_fourier)
            fourier_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            x = torch.cat([fourier_feat, latent_expanded], dim=-1)
        else:
            # Simple concatenation of coords and latent
            x = torch.cat([coords, latent_expanded], dim=-1)  # (B, N, 3+D)
        
        # Flatten for SIREN processing
        x = x.reshape(B * N, -1)
        
        # Forward through SIREN
        h = self.net(x)
        sdf = self.final(h)
        
        # Reshape output
        sdf = sdf.reshape(B, N, -1)
        
        if not batched:
            sdf = sdf.squeeze(0)
            
        return sdf
    
    def compute_gradient(
        self,
        coords: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial gradient of SDF (for Eikonal loss).
        
        Args:
            coords: Query coordinates (N, 3), requires_grad=True
            latent: Latent code (D,)
            
        Returns:
            Gradient of SDF w.r.t. coords, shape (N, 3)
        """
        coords = coords.requires_grad_(True)
        sdf = self.forward(coords, latent)
        
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(coord_dim={self.coord_dim}, "
                f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim})")
