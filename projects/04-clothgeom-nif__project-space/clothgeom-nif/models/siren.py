"""
SIREN (Sinusoidal Representation Networks) Activation Layer

Implementation of sine activation with proper initialization for
high-frequency signal representation, based on:
"Implicit Neural Representations with Periodic Activation Functions"
(Sitzmann et al., NeurIPS 2020)

Key features:
- Sine activation with configurable omega_0 (frequency scaling)
- Special initialization for first layer vs hidden layers
- Supports both Linear and modulated variants
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SineActivation(nn.Module):
    """
    Sine activation function with frequency scaling.
    
    f(x) = sin(omega_0 * x)
    
    Args:
        omega_0: Frequency scaling factor (default: 30.0)
                 Higher values = higher frequency details
    """
    
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)
    
    def __repr__(self):
        return f"SineActivation(omega_0={self.omega_0})"


class SirenLinear(nn.Module):
    """
    Linear layer with SIREN-style initialization.
    
    The initialization ensures that the distribution of activations
    is preserved through the network, enabling learning of high-frequency
    functions.
    
    For the first layer:
        w ~ U(-1/fan_in, 1/fan_in)
    
    For hidden layers:
        w ~ U(-sqrt(6 / fan_in) / omega_0, sqrt(6 / fan_in) / omega_0)
    
    Args:
        in_features: Size of input
        out_features: Size of output
        omega_0: Frequency scaling for this layer
        is_first: Whether this is the first layer (uses different init)
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/fan_in, 1/fan_in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def forward_with_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sine activation applied."""
        return torch.sin(self.omega_0 * self.linear(x))


class SirenLayer(nn.Module):
    """
    Complete SIREN layer: Linear + Sine activation.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        omega_0: Frequency scaling (default: 30.0)
        is_first: Whether this is the first layer
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.siren_linear = SirenLinear(
            in_features, out_features, omega_0, is_first, bias
        )
        self.activation = SineActivation(omega_0)
        self.omega_0 = omega_0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.siren_linear(x))


class ModulatedSirenLayer(nn.Module):
    """
    SIREN layer with latent code modulation.
    
    The latent code modulates the hidden features via:
    - Shift modulation: h = sin(omega_0 * (Wx + gamma))
    - Scale modulation: h = sin(omega_0 * Wx) * beta
    - Combined: h = sin(omega_0 * (Wx + gamma)) * beta
    
    where gamma and beta are predicted from the latent code.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        latent_dim: Dimension of latent code
        omega_0: Frequency scaling
        is_first: Whether this is the first layer
        modulation_type: 'shift', 'scale', or 'both'
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        modulation_type: str = 'shift'
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.modulation_type = modulation_type
        
        # Main linear layer
        self.siren_linear = SirenLinear(
            in_features, out_features, omega_0, is_first
        )
        
        # Modulation networks
        if modulation_type in ['shift', 'both']:
            self.gamma_net = nn.Linear(latent_dim, out_features)
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.zeros_(self.gamma_net.bias)
        
        if modulation_type in ['scale', 'both']:
            self.beta_net = nn.Linear(latent_dim, out_features)
            nn.init.zeros_(self.beta_net.weight)
            nn.init.ones_(self.beta_net.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input coordinates [B, in_features]
            latent: Latent code [B, latent_dim] or None
        
        Returns:
            Modulated features [B, out_features]
        """
        h = self.siren_linear(x)
        
        if latent is not None:
            if self.modulation_type in ['shift', 'both']:
                gamma = self.gamma_net(latent)
                h = h + gamma
            
            h = torch.sin(self.omega_0 * h)
            
            if self.modulation_type in ['scale', 'both']:
                beta = self.beta_net(latent)
                h = h * beta
        else:
            h = torch.sin(self.omega_0 * h)
        
        return h


class SirenNetwork(nn.Module):
    """
    Complete SIREN network with configurable depth and width.
    
    Architecture:
        Input -> [SirenLayer x (num_layers-1)] -> Linear -> Output
    
    Args:
        in_features: Input dimension (e.g., 3 for xyz coordinates)
        hidden_features: Hidden layer dimension
        out_features: Output dimension
        num_layers: Total number of layers (including output)
        omega_0: Frequency scaling for first layer
        omega_hidden: Frequency scaling for hidden layers
        final_activation: Whether to apply sine to final output
    """
    
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        out_features: int = 1,
        num_layers: int = 8,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        final_activation: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.final_activation = final_activation
        
        layers = []
        
        # First layer (special initialization)
        layers.append(SirenLayer(
            in_features, hidden_features,
            omega_0=omega_0, is_first=True
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(
                hidden_features, hidden_features,
                omega_0=omega_hidden, is_first=False
            ))
        
        # Output layer (no activation by default)
        self.layers = nn.ModuleList(layers)
        self.output_layer = SirenLinear(
            hidden_features, out_features,
            omega_0=omega_hidden, is_first=False
        )
        
        if final_activation:
            self.final_act = SineActivation(omega_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates [B, in_features]
        
        Returns:
            Network output [B, out_features]
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        if self.final_activation:
            x = self.final_act(x)
        
        return x
    
    def forward_with_intermediates(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning intermediate activations.
        Useful for feature extraction or visualization.
        
        Returns:
            (output, list of intermediate activations)
        """
        intermediates = []
        
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)
        
        x = self.output_layer(x)
        if self.final_activation:
            x = self.final_act(x)
        
        return x, intermediates
