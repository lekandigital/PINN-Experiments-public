"""
SIREN (Sinusoidal Representation Networks) Activation Layer

Re-exports from the shared implicit_fields library for backward compatibility.

Original implementation based on:
"Implicit Neural Representations with Periodic Activation Functions"
(Sitzmann et al., NeurIPS 2020)
"""

import sys
from pathlib import Path

# Add repository root to path for implicit_fields import
repo_root = Path(__file__).parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import from shared library
from implicit_fields import (
    SineActivation,
    SirenLayer,
    ModulatedSirenLayer,
)
from implicit_fields import SirenNetwork as _SirenNetwork

import math
import torch
import torch.nn as nn
from typing import Optional


class SirenLinear(nn.Module):
    """
    Linear layer with SIREN-style initialization.
    
    Thin wrapper for backward compatibility with existing code that uses
    SirenLinear directly (e.g., nif_decoder.py).
    
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


class SirenNetwork(nn.Module):
    """
    Complete SIREN network with configurable depth and width.
    
    Wrapper around shared implicit_fields.SirenNetwork for backward compatibility.
    Accepts `num_layers` (P04 convention) and converts to `hidden_layers`.
    
    Architecture:
        Input -> [SirenLayer x (num_layers-1)] -> Linear -> Output
    
    Args:
        in_features: Input dimension (e.g., 3 for xyz coordinates)
        hidden_features: Hidden layer dimension
        out_features: Output dimension
        num_layers: Total number of layers (including output) - P04 convention
        hidden_layers: Number of hidden layers (alternative to num_layers)
        omega_0: Frequency scaling for first layer
        omega_hidden: Frequency scaling for hidden layers
        final_activation: Whether to apply sine to final output
    """
    
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        out_features: int = 1,
        num_layers: int = None,
        hidden_layers: int = None,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        final_activation: bool = False
    ):
        super().__init__()
        
        # Handle both num_layers (P04) and hidden_layers (shared lib) conventions
        if num_layers is not None:
            # P04 convention: num_layers includes input processing and output
            # hidden_layers = num_layers - 2 (subtract first and output layer)
            actual_hidden_layers = num_layers - 2
        elif hidden_layers is not None:
            actual_hidden_layers = hidden_layers
        else:
            actual_hidden_layers = 6  # Default: 8 total layers - 2
        
        self.num_layers = (actual_hidden_layers + 2) if num_layers is None else num_layers
        self.final_activation = final_activation
        
        # Use shared library implementation
        self._network = _SirenNetwork(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            hidden_layers=actual_hidden_layers,
            omega_0=omega_0,
            omega_hidden=omega_hidden,
            final_activation=final_activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x)
    
    def forward_with_intermediates(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning intermediate activations.
        Useful for feature extraction or visualization.
        
        Returns:
            (output, list of intermediate activations)
        """
        return self._network.forward_with_activations(x)


# Re-export everything for backward compatibility
__all__ = [
    'SineActivation',
    'SirenLinear',
    'SirenLayer',
    'ModulatedSirenLayer',
    'SirenNetwork',
]
