"""
SIREN (Sinusoidal Representation Networks) implementation.

This module provides configurable SIREN networks following Sitzmann et al. (2020)
"Implicit Neural Representations with Periodic Activation Functions".

The key insight is that sine activation functions with proper initialization
enable neural networks to represent complex signals with fine details.

Example:
    >>> from implicit_fields import SirenNetwork
    >>> model = SirenNetwork(in_features=3, hidden_features=256, hidden_layers=3, out_features=1)
    >>> x = torch.randn(1000, 3)  # 1000 query points in 3D
    >>> sdf = model(x)  # Predicted SDF values
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SineActivation(nn.Module):
    """Sine activation function with frequency scaling.
    
    Computes sin(omega * x) where omega controls the frequency of oscillations.
    
    Args:
        omega: Frequency scaling factor. Higher values = higher frequency output.
               Default is 30.0 as recommended by Sitzmann et al.
    
    Example:
        >>> act = SineActivation(omega=30.0)
        >>> x = torch.linspace(-1, 1, 100)
        >>> y = act(x)  # High-frequency sine wave
    """
    
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sine activation with frequency scaling.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            sin(omega * x) with same shape as input.
        """
        return torch.sin(self.omega * x)


class SirenLayer(nn.Module):
    """A single SIREN layer: linear transformation followed by sine activation.
    
    Implements the layer: y = sin(omega * (Wx + b))
    
    The weight initialization is critical for SIREN to work correctly:
    - First layer: weights uniform in [-1/in_features, 1/in_features]
    - Hidden layers: weights uniform in [-sqrt(6/n)/omega, sqrt(6/n)/omega]
    
    This ensures the pre-activations have the right variance for the sine
    nonlinearity to operate in its useful range.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        omega: Frequency scaling for the sine activation.
        is_first: Whether this is the first layer (affects initialization).
        bias: If True, adds a learnable bias. Default: True.
    
    Example:
        >>> layer = SirenLayer(3, 256, omega=30.0, is_first=True)
        >>> x = torch.randn(100, 3)
        >>> y = layer(x)  # Shape: (100, 256)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
        c: float = 6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.c = c  # Initialization constant
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = SineActivation(omega)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper.
        
        First layer: uniform[-1/in, 1/in]
        Hidden layers: uniform[-sqrt(c/n)/omega, sqrt(c/n)/omega]
        
        The constant c (default 6) comes from the variance of a uniform distribution:
        Var(U[-a,a]) = a²/3, and we want the pre-activation variance to be 1,
        combined with the sine's output variance of 0.5.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in_features, 1/in_features]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(c/n)/omega, sqrt(c/n)/omega]
                bound = math.sqrt(self.c / self.in_features) / self.omega
            
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation then sine activation.
        
        Args:
            x: Input tensor of shape (*, in_features).
            
        Returns:
            Output tensor of shape (*, out_features).
        """
        return self.activation(self.linear(x))


class SirenNetwork(nn.Module):
    """Complete SIREN network for implicit neural representations.
    
    A multi-layer perceptron with sine activations and proper initialization
    for representing continuous signals (SDFs, images, audio, etc.).
    
    Args:
        in_features: Input dimensionality (e.g., 3 for 3D coordinates, 4 for 4D).
        hidden_features: Width of hidden layers.
        hidden_layers: Number of hidden layers.
        out_features: Output dimensionality (e.g., 1 for SDF, 3 for RGB).
        omega_0: Frequency scaling for the first layer. Default: 30.0.
        omega_hidden: Frequency scaling for hidden layers. Default: 30.0.
        bias: Whether to use bias in linear layers. Default: True.
        final_activation: Optional activation for the output layer (e.g., nn.Tanh()).
        dropout: Dropout probability between layers. Default: 0.0.
    
    Example:
        >>> # SDF network: 3D coords -> 1D signed distance
        >>> model = SirenNetwork(
        ...     in_features=3,
        ...     hidden_features=256,
        ...     hidden_layers=5,
        ...     out_features=1,
        ...     omega_0=30.0
        ... )
        >>> coords = torch.randn(1000, 3)
        >>> sdf_values = model(coords)
        
        >>> # 4D spacetime network
        >>> model_4d = SirenNetwork(in_features=4, hidden_features=128, hidden_layers=3, out_features=1)
        >>> xyzt = torch.randn(500, 4)
        >>> sdf = model_4d(xyzt)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        bias: bool = True,
        final_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_0 = omega_0
        self.omega_hidden = omega_hidden
        
        # Build network layers
        layers = []
        
        # First layer
        layers.append(SirenLayer(
            in_features, hidden_features,
            omega=omega_0, is_first=True, bias=bias
        ))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SirenLayer(
                hidden_features, hidden_features,
                omega=omega_hidden, is_first=False, bias=bias
            ))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        
        # Final layer (no sine activation by default)
        self.final_linear = nn.Linear(hidden_features, out_features, bias=bias)
        self._init_final_layer()
        
        # Store final activation - create SineActivation if True, else None
        if final_activation is True:
            self.final_activation = SineActivation(omega_hidden)
        elif final_activation:
            # Allow passing custom activation
            self.final_activation = final_activation
        else:
            self.final_activation = None
    
    def _init_final_layer(self) -> None:
        """Initialize final layer with standard initialization."""
        with torch.no_grad():
            # Xavier uniform initialization for final layer
            bound = math.sqrt(6.0 / (self.hidden_features + self.out_features))
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SIREN network.
        
        Args:
            x: Input coordinates of shape (batch_size, in_features).
            
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        h = self.layers(x)
        out = self.final_linear(h)
        
        if self.final_activation is not None:
            out = self.final_activation(out)
        
        return out
    
    def forward_with_activations(self, x: torch.Tensor) -> tuple:
        """Forward pass returning intermediate activations.
        
        Useful for debugging, visualization, or computing gradients
        at intermediate layers.
        
        Args:
            x: Input coordinates of shape (batch_size, in_features).
            
        Returns:
            Tuple of (output, list of intermediate activations).
        """
        activations = [x]
        h = x
        
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, SirenLayer):
                activations.append(h)
        
        out = self.final_linear(h)
        if self.final_activation is not None:
            out = self.final_activation(out)
        
        activations.append(out)
        return out, activations


class LatentConditionedSiren(SirenNetwork):
    """SIREN network conditioned on a latent code.
    
    Used for generative models where a latent vector (e.g., from an encoder)
    is concatenated with spatial coordinates before feeding to the SIREN.
    
    The latent code can come from:
    - A GNN encoder (Project 09: HGNN-NIF-Cloth)
    - A VAE encoder
    - A learned embedding table
    
    Args:
        in_features: Spatial coordinate dimensionality (e.g., 3 for xyz).
        latent_dim: Dimensionality of the latent conditioning vector.
        hidden_features: Width of hidden layers.
        hidden_layers: Number of hidden layers.
        out_features: Output dimensionality.
        omega_0: Frequency scaling for first layer. Default: 30.0.
        omega_hidden: Frequency scaling for hidden layers. Default: 30.0.
        bias: Whether to use bias. Default: True.
        final_activation: Optional output activation.
        dropout: Dropout probability. Default: 0.0.
    
    Example:
        >>> # Cloth SDF conditioned on garment latent code
        >>> model = LatentConditionedSiren(
        ...     in_features=3,      # xyz coordinates
        ...     latent_dim=64,      # latent code from GNN encoder
        ...     hidden_features=128,
        ...     hidden_layers=3,
        ...     out_features=1      # SDF value
        ... )
        >>> coords = torch.randn(1000, 3)    # Query points
        >>> latent = torch.randn(1, 64)      # Latent code (broadcast to all points)
        >>> latent_expanded = latent.expand(1000, -1)
        >>> sdf = model(coords, latent_expanded)
    """
    
    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        bias: bool = True,
        final_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        # The actual input to the network is coords + latent
        super().__init__(
            in_features=in_features + latent_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            omega_0=omega_0,
            omega_hidden=omega_hidden,
            bias=bias,
            final_activation=final_activation,
            dropout=dropout,
        )
        self.coord_dim = in_features
        self.latent_dim = latent_dim
    
    def forward(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass with coordinate and latent inputs.
        
        Args:
            coords: Spatial coordinates of shape (batch_size, coord_dim).
            latent: Latent conditioning vector of shape (batch_size, latent_dim).
                   Can also be (1, latent_dim) if the same latent applies to all points,
                   but caller should broadcast it first.
            
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Concatenate coordinates and latent code
        x = torch.cat([coords, latent], dim=-1)
        return super().forward(x)


class ModulatedSirenLayer(nn.Module):
    """SIREN layer with FiLM-style modulation.
    
    Implements: y = sin(omega * (gamma * (Wx + b) + beta))
    
    Where gamma (scale) and beta (shift) come from a conditioning network.
    This allows the network to be modulated by external signals like
    force vectors, material parameters, or time.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        omega: Frequency scaling for sine activation. Default: 30.0.
        is_first: Whether this is the first layer.
        bias: Whether to use bias. Default: True.
    
    Example:
        >>> layer = ModulatedSirenLayer(256, 256, omega=30.0)
        >>> x = torch.randn(100, 256)
        >>> gamma = torch.ones(100, 256)   # Scale modulation
        >>> beta = torch.zeros(100, 256)   # Shift modulation
        >>> y = layer(x, gamma, beta)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega
            
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply modulated SIREN layer.
        
        Args:
            x: Input tensor of shape (*, in_features).
            gamma: Scale modulation of shape (*, out_features). Default: 1.
            beta: Shift modulation of shape (*, out_features). Default: 0.
            
        Returns:
            Modulated output of shape (*, out_features).
        """
        h = self.linear(x)
        
        if gamma is not None:
            h = gamma * h
        if beta is not None:
            h = h + beta
        
        return torch.sin(self.omega * h)
