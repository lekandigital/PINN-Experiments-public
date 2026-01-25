"""
SIREN (Sinusoidal Representation Networks) layers for high-frequency details.

SIREN uses sin(ωx) activations instead of ReLU, enabling neural networks to
represent signals with fine spatial and temporal details - critical for
capturing cloth wrinkles and folds.

Reference:
    Sitzmann et al., "Implicit Neural Representations with Periodic Activation
    Functions", NeurIPS 2020.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SIRENLayer(nn.Module):
    """
    A single SIREN layer with sine activation and specialized initialization.
    
    The key insight is that sin activations require careful weight initialization
    to ensure the distribution of activations is preserved across layers.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features  
        omega_0: Frequency multiplier for the sine activation (default: 30.0)
        is_first: Whether this is the first layer (affects initialization)
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # SIREN-specific initialization
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize weights according to SIREN paper.
        
        First layer: Uniform(-1/in, 1/in)
        Other layers: Uniform(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)
        """
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: scaled by omega_0
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            
            self.linear.weight.uniform_(-bound, bound)
            
            if self.linear.bias is not None:
                # Bias initialized to zero or small uniform
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sin(omega_0 * (Wx + b))
        
        Args:
            x: Input tensor of shape (batch, in_features)
            
        Returns:
            Output tensor of shape (batch, out_features)
        """
        return torch.sin(self.omega_0 * self.linear(x))
    
    def forward_with_intermediate(self, x: torch.Tensor) -> tuple:
        """
        Return both pre-activation and activation for gradient analysis.
        
        Useful for computing spatial gradients (normals) via autodiff.
        """
        pre_activation = self.omega_0 * self.linear(x)
        return torch.sin(pre_activation), pre_activation


class SIRENNetwork(nn.Module):
    """
    Complete SIREN network with multiple layers.
    
    Architecture:
        input -> [SIRENLayer] x (num_layers - 1) -> Linear -> output
    
    The final layer is linear (no sine activation) to allow arbitrary
    output range for SDF values.
    
    Args:
        in_features: Input dimension (e.g., 4 for x,y,z,t)
        hidden_features: Hidden layer dimension
        out_features: Output dimension (e.g., 1 for SDF)
        num_layers: Total number of layers including output
        omega_0: Frequency for sine activations
        omega_0_first: Frequency for first layer (often different)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 5,
        omega_0: float = 30.0,
        omega_0_first: Optional[float] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.omega_0 = omega_0
        self.omega_0_first = omega_0_first if omega_0_first is not None else omega_0
        
        # Build network layers
        layers = []
        
        # First layer (input -> hidden)
        layers.append(SIRENLayer(
            in_features=in_features,
            out_features=hidden_features,
            omega_0=self.omega_0_first,
            is_first=True,
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(
                in_features=hidden_features,
                out_features=hidden_features,
                omega_0=omega_0,
                is_first=False,
            ))
        
        self.layers = nn.ModuleList(layers)
        
        # Final linear layer (no sine activation)
        self.final_layer = nn.Linear(hidden_features, out_features)
        self._init_final_layer()
    
    def _init_final_layer(self) -> None:
        """Initialize final layer with SIREN-compatible scaling."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_features) / self.omega_0
            self.final_layer.weight.uniform_(-bound, bound)
            if self.final_layer.bias is not None:
                self.final_layer.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input coordinates of shape (batch, in_features)
            
        Returns:
            Output of shape (batch, out_features)
        """
        for layer in self.layers:
            x = layer(x)
        
        return self.final_layer(x)
    
    def forward_with_gradient(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that also computes spatial gradient (useful for normals).
        
        Args:
            x: Input coordinates of shape (batch, in_features)
            
        Returns:
            output: SDF values of shape (batch, out_features)
            gradient: Spatial gradient of shape (batch, in_features)
        """
        x.requires_grad_(True)
        output = self.forward(x)
        
        gradient = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return output, gradient


class ModulatedSIRENLayer(nn.Module):
    """
    SIREN layer with external modulation for conditioning.
    
    Allows injecting latent codes (e.g., from GRU) to modulate the
    frequency content, enabling temporal conditioning.
    
    output = sin(omega_0 * (Wx + b) * (1 + gamma) + beta)
    
    where gamma, beta are computed from the conditioning vector.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        conditioning_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # Main linear transformation
        self.linear = nn.Linear(in_features, out_features)
        
        # Modulation layers (FiLM-style)
        self.gamma_layer = nn.Linear(conditioning_dim, out_features)
        self.beta_layer = nn.Linear(conditioning_dim, out_features)
        
        # Initialize
        self._init_weights(in_features)
    
    def _init_weights(self, in_features: int) -> None:
        """SIREN initialization plus modulation init."""
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
            
            # Initialize modulation to identity (gamma=0, beta=0)
            self.gamma_layer.weight.zero_()
            self.gamma_layer.bias.zero_()
            self.beta_layer.weight.zero_()
            self.beta_layer.bias.zero_()
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with modulation.
        
        Args:
            x: Input features (batch, in_features)
            conditioning: Conditioning vector (batch, conditioning_dim)
            
        Returns:
            Modulated output (batch, out_features)
        """
        # Compute modulation parameters
        gamma = self.gamma_layer(conditioning)  # (batch, out_features)
        beta = self.beta_layer(conditioning)    # (batch, out_features)
        
        # Apply modulated SIREN
        pre_activation = self.omega_0 * self.linear(x)
        modulated = pre_activation * (1.0 + gamma) + beta
        
        return torch.sin(modulated)
