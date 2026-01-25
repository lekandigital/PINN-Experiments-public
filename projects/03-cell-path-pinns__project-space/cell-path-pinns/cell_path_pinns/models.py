"""
Neural Network Architectures for Cell-Path PINNs

Contains:
- PathNet: Maps time t → (x, y) position
- PotentialNet: Maps (x, y, t) → scalar potential U (nutrient concentration)
"""

import torch
import torch.nn as nn
from typing import Optional


class PathNet(nn.Module):
    """
    Trajectory network that maps time to 2D position.
    
    Architecture: 1 → hidden_dim → hidden_dim → 2
    Activation: Tanh (for smooth derivatives needed in geodesic loss)
    
    Args:
        hidden_dim: Number of neurons in hidden layers (default: 64)
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Initialize weights using Xavier initialization (good for Tanh)
        self._init_weights()
    
    def _init_weights(self):
        """Apply Xavier initialization for Tanh activation."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: time → position.
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            Position tensor of shape (batch_size, 2) containing (x, y)
        """
        return self.net(t)


class PotentialNet(nn.Module):
    """
    Potential field network that maps (x, y, t) to scalar nutrient concentration.
    
    Architecture: 3 → hidden_dim → hidden_dim → 1
    Activation: ReLU (standard for regression)
    
    Args:
        hidden_dim: Number of neurons in hidden layers (default: 64)
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights using He initialization (good for ReLU)
        self._init_weights()
    
    def _init_weights(self):
        """Apply He initialization for ReLU activation."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (x, y, t) → potential U.
        
        Args:
            x: X position tensor of shape (batch_size, 1)
            y: Y position tensor of shape (batch_size, 1)
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            Potential tensor of shape (batch_size, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
