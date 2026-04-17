"""
Fourier Feature Encoding for neural implicit representations.

This module implements random Fourier feature encoding following Tancik et al. (2020)
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains".

Fourier features map low-dimensional inputs to a high-dimensional space using
random sinusoidal projections, enabling standard MLPs to learn high-frequency functions.

Example:
    >>> from implicit_fields import FourierFeatureEncoding, FourierFeatureMLP
    >>> encoder = FourierFeatureEncoding(in_features=3, num_frequencies=128, scale=10.0)
    >>> coords = torch.randn(1000, 3)
    >>> encoded = encoder(coords)  # Shape: (1000, 3 + 256) with include_input=True
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatureEncoding(nn.Module):
    """Random Fourier feature positional encoding.
    
    Maps input coordinates to a higher-dimensional space using random sinusoidal
    projections. This enables MLPs with standard activations (ReLU, GELU) to
    learn high-frequency functions.
    
    The encoding computes:
        γ(x) = [sin(2π * Bx), cos(2π * Bx)]
    
    Where B is a (num_frequencies, in_features) matrix with entries sampled
    from N(0, scale²).
    
    Args:
        in_features: Input dimensionality (e.g., 3 for xyz coordinates).
        num_frequencies: Number of Fourier basis frequencies. More frequencies
                        capture higher-frequency details but increase memory.
        scale: Standard deviation of the Gaussian frequency sampling.
               Higher scale = higher frequency content. Default: 10.0.
        learnable: Whether the frequency matrix B is learnable. Default: False.
        include_input: Whether to concatenate raw input alongside Fourier features.
                      Default: True.
    
    Attributes:
        output_dim: Total output dimensionality after encoding.
    
    Example:
        >>> encoder = FourierFeatureEncoding(
        ...     in_features=3,
        ...     num_frequencies=128,
        ...     scale=10.0,
        ...     include_input=True
        ... )
        >>> x = torch.randn(100, 3)
        >>> encoded = encoder(x)
        >>> print(encoded.shape)  # (100, 3 + 256) = (100, 259)
    """
    
    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 128,
        scale: float = 10.0,
        learnable: bool = False,
        include_input: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.scale = scale
        self.include_input = include_input
        
        # Output dimensionality: 2 * num_frequencies (sin + cos) + optional input
        self.output_dim = 2 * num_frequencies + (in_features if include_input else 0)
        
        # Initialize random frequency matrix B ~ N(0, scale²)
        B = torch.randn(num_frequencies, in_features) * scale
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding.
        
        Args:
            x: Input coordinates of shape (*, in_features).
            
        Returns:
            Encoded features of shape (*, output_dim).
        """
        # Project inputs: (*, in_features) @ (in_features, num_frequencies) -> (*, num_frequencies)
        projected = 2.0 * math.pi * torch.matmul(x, self.B.T)
        
        # Compute sin and cos features
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        
        # Optionally include raw input
        if self.include_input:
            features = torch.cat([x, features], dim=-1)
        
        return features
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, num_frequencies={self.num_frequencies}, '
                f'scale={self.scale}, include_input={self.include_input}, '
                f'output_dim={self.output_dim}')


class FourierFeatureMLP(nn.Module):
    """MLP with Fourier feature input encoding.
    
    Combines Fourier feature encoding with a standard MLP using configurable
    activations (ReLU, GELU, or even sine for hybrid SIREN+Fourier).
    
    Architecture:
        Input -> FourierFeatureEncoding -> MLP layers -> Output
    
    Args:
        in_features: Input dimensionality.
        hidden_features: Width of hidden layers.
        hidden_layers: Number of hidden layers.
        out_features: Output dimensionality.
        num_frequencies: Number of Fourier basis frequencies. Default: 128.
        scale: Frequency scale for Fourier features. Default: 10.0.
        activation: Activation function name ('relu', 'gelu', 'sine'). Default: 'relu'.
        include_input: Include raw input in Fourier features. Default: True.
        dropout: Dropout probability. Default: 0.0.
        learnable_frequencies: Whether Fourier frequencies are learnable. Default: False.
    
    Example:
        >>> model = FourierFeatureMLP(
        ...     in_features=4,           # (x, y, z, t)
        ...     hidden_features=256,
        ...     hidden_layers=4,
        ...     out_features=1,          # SDF
        ...     num_frequencies=128,
        ...     scale=10.0,
        ...     activation='gelu'
        ... )
        >>> coords = torch.randn(1000, 4)
        >>> sdf = model(coords)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        num_frequencies: int = 128,
        scale: float = 10.0,
        activation: str = 'relu',
        include_input: bool = True,
        dropout: float = 0.0,
        learnable_frequencies: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        
        # Fourier feature encoding
        self.encoding = FourierFeatureEncoding(
            in_features=in_features,
            num_frequencies=num_frequencies,
            scale=scale,
            learnable=learnable_frequencies,
            include_input=include_input,
        )
        
        # Get activation function
        self.activation = self._get_activation(activation)
        
        # Build MLP
        layers = []
        
        # First layer: encoded features -> hidden
        layers.append(nn.Linear(self.encoding.output_dim, hidden_features))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_features, out_features)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'sine': SineActivation(omega=30.0),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'elu': nn.ELU(inplace=True),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]
    
    def _init_weights(self) -> None:
        """Initialize MLP weights using Kaiming initialization."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Output layer: smaller initialization for stability
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Fourier feature MLP.
        
        Args:
            x: Input coordinates of shape (batch_size, in_features).
            
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Encode inputs
        encoded = self.encoding(x)
        
        # Pass through MLP
        h = self.mlp(encoded)
        
        # Output projection
        return self.output_layer(h)


class SineActivation(nn.Module):
    """Sine activation for use with Fourier feature networks.
    
    Creates a hybrid architecture: Fourier input encoding + SIREN-like hidden layers.
    """
    
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class MultiScaleFourierFeatures(nn.Module):
    """Multi-scale Fourier feature encoding.
    
    Uses multiple frequency scales to capture both low and high frequency
    content, similar to the coarse-to-fine strategy in NeRF.
    
    Args:
        in_features: Input dimensionality.
        num_frequencies_per_scale: Frequencies per scale. Default: 64.
        scales: List of scale values. Default: [1.0, 2.0, 4.0, 8.0].
        include_input: Include raw input. Default: True.
    
    Example:
        >>> encoder = MultiScaleFourierFeatures(
        ...     in_features=3,
        ...     num_frequencies_per_scale=32,
        ...     scales=[1.0, 4.0, 16.0, 64.0]
        ... )
        >>> x = torch.randn(100, 3)
        >>> encoded = encoder(x)
    """
    
    def __init__(
        self,
        in_features: int,
        num_frequencies_per_scale: int = 64,
        scales: Optional[list] = None,
        include_input: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.include_input = include_input
        
        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0]
        self.scales = scales
        
        # Create frequency matrices for each scale
        B_list = []
        for scale in scales:
            B = torch.randn(num_frequencies_per_scale, in_features) * scale
            B_list.append(B)
        
        # Concatenate all frequency matrices
        self.register_buffer('B', torch.cat(B_list, dim=0))
        self.num_total_frequencies = num_frequencies_per_scale * len(scales)
        
        # Output dimensionality
        self.output_dim = 2 * self.num_total_frequencies + (in_features if include_input else 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale Fourier encoding.
        
        Args:
            x: Input of shape (*, in_features).
            
        Returns:
            Encoded features of shape (*, output_dim).
        """
        projected = 2.0 * math.pi * torch.matmul(x, self.B.T)
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        
        if self.include_input:
            features = torch.cat([x, features], dim=-1)
        
        return features
