"""
FourierFeatureMLP: Core neural implicit field Φθ(x,y,z,t) for cloth simulation.

Combines Fourier feature embedding with SIREN-style architecture and optional
GRU temporal conditioning. Designed for high-frequency cloth wrinkle capture.

Uses the shared implicit_fields library for Fourier features and SIREN layers.

Architecture:
    1. Fourier Feature Embedding: [x,y,z,t] -> [sin(2πBx), cos(2πBx)]
    2. SIREN Layers: Multiple layers with sin activations
    3. Optional GRU: Temporal conditioning via hidden state
    4. Output: Scalar SDF value

Reference:
    - Tancik et al., "Fourier Features Let Networks Learn High Frequency
      Functions in Low Dimensional Domains", NeurIPS 2020.
    - Sitzmann et al., "Implicit Neural Representations with Periodic
      Activation Functions", NeurIPS 2020.
"""

import sys
from pathlib import Path

# Add repository root to path for implicit_fields import
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .siren import SIRENLayer
from .temporal_gru import TemporalGRU

# Import Fourier features from shared library
from implicit_fields import FourierFeatureEncoding as _FourierFeatureEncoding


class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier feature embedding for positional encoding.
    
    Wrapper around shared implicit_fields.FourierFeatureEncoding for P12 compatibility.
    
    Args:
        in_dim: Input dimension (4 for x,y,z,t)
        num_freqs: Number of frequency components
        scale: Standard deviation for random matrix B (controls frequency range)
        learnable: Whether to make B learnable (usually fixed)
    """
    
    def __init__(
        self,
        in_dim: int = 4,
        num_freqs: int = 16,
        scale: float = 10.0,
        learnable: bool = False,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.scale = scale
        self.out_dim = num_freqs * 2  # sin and cos
        
        # Use shared library
        self._encoding = _FourierFeatureEncoding(
            in_features=in_dim,
            num_frequencies=num_freqs,
            scale=scale,
            learnable=learnable,
            include_input=False,  # P12 doesn't include raw input
        )
    
    @property
    def B(self):
        """Expose B matrix for compatibility."""
        return self._encoding.B
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features.
        
        Args:
            x: Input coordinates of shape (batch, in_dim)
            
        Returns:
            Fourier features of shape (batch, num_freqs * 2)
        """
        return self._encoding(x)


class FourierFeatureMLP(nn.Module):
    """
    Main neural implicit field Φθ(x,y,z,t) for continuous cloth representation.
    
    The network predicts the signed distance field (SDF) value for any 
    spatiotemporal point, enabling continuous-time cloth simulation with
    arbitrary temporal resolution.
    
    Args:
        in_dim: Input dimension (4 for x,y,z,t coordinates)
        hidden_dim: Hidden layer width
        out_dim: Output dimension (1 for SDF, 3 for displacement field)
        num_layers: Number of SIREN hidden layers
        num_freqs: Number of Fourier frequency components
        fourier_scale: Scale for random Fourier features
        omega_0: SIREN frequency parameter
        use_gru: Whether to use GRU temporal conditioning
        gru_hidden: GRU hidden state dimension
        dropout: Dropout probability (for regularization)
    
    Example:
        >>> model = FourierFeatureMLP(
        ...     in_dim=4, hidden_dim=256, num_layers=5,
        ...     use_gru=True, gru_hidden=128
        ... )
        >>> xyz = torch.randn(1024, 3)  # Spatial coordinates
        >>> t = torch.full((1024, 1), 0.5)  # Time = 0.5s
        >>> sdf, hidden = model(xyz, t)
        >>> print(sdf.shape)  # (1024, 1)
    """
    
    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 256,
        out_dim: int = 1,
        num_layers: int = 5,
        num_freqs: int = 16,
        fourier_scale: float = 10.0,
        omega_0: float = 30.0,
        use_gru: bool = False,
        gru_hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.gru_hidden = gru_hidden
        
        # Fourier feature embedding
        self.fourier_embedding = FourierFeatureEmbedding(
            in_dim=in_dim,
            num_freqs=num_freqs,
            scale=fourier_scale,
        )
        fourier_out_dim = self.fourier_embedding.out_dim
        
        # Build SIREN layers
        layers = []
        
        # First layer: Fourier features -> hidden
        layers.append(SIRENLayer(
            in_features=fourier_out_dim,
            out_features=hidden_dim,
            omega_0=omega_0,
            is_first=True,
        ))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden SIREN layers
        for i in range(num_layers - 2):
            layers.append(SIRENLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                omega_0=omega_0,
                is_first=False,
            ))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.siren_layers = nn.Sequential(*layers)
        
        # Optional GRU for temporal conditioning
        if use_gru:
            self.temporal_gru = TemporalGRU(
                input_dim=hidden_dim,
                hidden_dim=gru_hidden,
                num_layers=1,
            )
        else:
            self.temporal_gru = None
        
        # Final output layer (linear, no sine activation)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self._init_output_layer(omega_0)
    
    def _init_output_layer(self, omega_0: float) -> None:
        """Initialize output layer with SIREN-compatible scaling."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_dim) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()
    
    def forward(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: compute SDF values for given spatiotemporal coordinates.
        
        Args:
            xyz: Spatial coordinates of shape (batch, 3)
            t: Temporal coordinates of shape (batch, 1)
            hidden_state: Optional GRU hidden state from previous frame
                         Shape: (num_layers, batch, gru_hidden)
                         
        Returns:
            sdf: Signed distance field values of shape (batch, out_dim)
            new_hidden: Updated GRU hidden state (None if use_gru=False)
        """
        # Concatenate spatial and temporal coordinates
        xyzt = torch.cat([xyz, t], dim=-1)  # (batch, 4)
        
        # Fourier feature embedding
        features = self.fourier_embedding(xyzt)  # (batch, num_freqs * 2)
        
        # SIREN hidden layers
        features = self.siren_layers(features)  # (batch, hidden_dim)
        
        # Optional GRU temporal conditioning
        new_hidden = None
        if self.use_gru and self.temporal_gru is not None:
            features, new_hidden = self.temporal_gru(features, hidden_state)
        
        # Output layer
        sdf = self.output_layer(features)  # (batch, out_dim)
        
        return sdf, new_hidden
    
    def forward_batch(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified forward without GRU state management (for evaluation).
        
        Args:
            xyz: Spatial coordinates (batch, 3)
            t: Temporal coordinates (batch, 1)
            
        Returns:
            sdf: SDF values (batch, out_dim)
        """
        sdf, _ = self.forward(xyz, t, hidden_state=None)
        return sdf
    
    def compute_gradient(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SDF and its spatial gradient (surface normal direction).
        
        Useful for:
        - Computing surface normals: n = ∇SDF / ||∇SDF||
        - Eikonal regularization: ||∇SDF|| ≈ 1
        
        Args:
            xyz: Spatial coordinates (batch, 3), requires_grad will be set
            t: Temporal coordinates (batch, 1)
            hidden_state: Optional GRU state
            
        Returns:
            sdf: SDF values (batch, 1)
            gradient: Spatial gradient ∂SDF/∂xyz (batch, 3)
            new_hidden: Updated GRU state
        """
        # Ensure xyz requires gradient for autograd
        xyz = xyz.clone().requires_grad_(True)
        
        # Forward pass
        sdf, new_hidden = self.forward(xyz, t, hidden_state)
        
        # Compute gradient w.r.t. spatial coordinates
        gradient = torch.autograd.grad(
            outputs=sdf,
            inputs=xyz,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,  # For second-order derivatives if needed
            retain_graph=True,
        )[0]
        
        return sdf, gradient, new_hidden
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """
        Initialize GRU hidden state.
        
        Args:
            batch_size: Number of parallel sequences
            device: Target device
            dtype: Data type
            
        Returns:
            Initialized hidden state, or None if GRU is disabled
        """
        if self.use_gru and self.temporal_gru is not None:
            return self.temporal_gru.init_hidden(batch_size, device, dtype)
        return None
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation with architecture summary."""
        return (
            f"FourierFeatureMLP(\n"
            f"  in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim}\n"
            f"  num_layers={self.num_layers}, use_gru={self.use_gru}\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


class DisplacementMLP(FourierFeatureMLP):
    """
    Variant that outputs displacement vectors instead of SDF values.
    
    For each query point, predicts the displacement from rest position
    to deformed position: Δx = Φθ(x_rest, t)
    
    This formulation can be more stable for large deformations.
    """
    
    def __init__(self, **kwargs):
        # Force output dimension to 3 (displacement vector)
        kwargs['out_dim'] = 3
        super().__init__(**kwargs)
    
    def forward(
        self,
        xyz_rest: torch.Tensor,
        t: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict deformed position from rest position.
        
        Args:
            xyz_rest: Rest position coordinates (batch, 3)
            t: Time coordinates (batch, 1)
            hidden_state: Optional GRU state
            
        Returns:
            xyz_deformed: Deformed positions (batch, 3)
            displacement: Displacement vectors (batch, 3)
            new_hidden: Updated GRU state
        """
        displacement, new_hidden = super().forward(xyz_rest, t, hidden_state)
        xyz_deformed = xyz_rest + displacement
        
        return xyz_deformed, displacement, new_hidden
