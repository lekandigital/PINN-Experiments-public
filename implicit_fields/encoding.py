"""
Positional encoding utilities for neural implicit representations.

This module provides NeRF-style positional encoding, which maps low-dimensional
coordinates to a higher-dimensional space using fixed sinusoidal functions.

Example:
    >>> from implicit_fields import positional_encoding
    >>> coords = torch.randn(100, 3)  # 3D coordinates
    >>> encoded = positional_encoding(coords, num_frequencies=10)
    >>> print(encoded.shape)  # (100, 63) = 3 + 3*2*10
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def positional_encoding(
    x: torch.Tensor,
    num_frequencies: int = 6,
    include_input: bool = True,
    log_sampling: bool = True,
    max_frequency: Optional[float] = None,
) -> torch.Tensor:
    """Apply NeRF-style positional encoding.
    
    Maps input coordinates to a higher-dimensional space using sinusoids
    at exponentially increasing frequencies:
    
        γ(x) = [x, sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]
    
    This is different from random Fourier features — positional encoding uses
    fixed, exponentially-spaced frequencies rather than randomly sampled ones.
    
    Args:
        x: Input tensor of shape (*, D) where D is the coordinate dimension.
        num_frequencies: Number of frequency bands L. Default: 6.
        include_input: If True, concatenate raw input. Default: True.
        log_sampling: If True, use log-linear frequency spacing (2^k).
                     If False, use linear spacing. Default: True.
        max_frequency: Maximum frequency (for log_sampling=True, this is 2^max_freq).
                      If None, uses num_frequencies - 1. Default: None.
    
    Returns:
        Encoded tensor of shape (*, D * (1 + 2 * num_frequencies)) if include_input,
        else (*, D * 2 * num_frequencies).
    
    Example:
        >>> # Standard NeRF encoding for 3D coordinates
        >>> coords = torch.randn(100, 3)
        >>> encoded = positional_encoding(coords, num_frequencies=10)
        >>> print(encoded.shape)  # (100, 63)
        
        >>> # Without raw input
        >>> encoded = positional_encoding(coords, num_frequencies=6, include_input=False)
        >>> print(encoded.shape)  # (100, 36)
        
        >>> # Linear frequency spacing
        >>> encoded = positional_encoding(coords, num_frequencies=8, log_sampling=False)
    """
    if max_frequency is None:
        max_frequency = num_frequencies - 1
    
    # Generate frequency bands
    if log_sampling:
        # Exponential spacing: [2^0, 2^1, ..., 2^(L-1)] * pi
        freq_bands = 2.0 ** torch.linspace(0, max_frequency, num_frequencies, device=x.device, dtype=x.dtype)
    else:
        # Linear spacing: [1, 2, 3, ..., L] * pi
        freq_bands = torch.linspace(1, num_frequencies, num_frequencies, device=x.device, dtype=x.dtype)
    
    freq_bands = freq_bands * math.pi
    
    # Compute encodings
    # x: (*, D), freq_bands: (L,) -> expanded: (*, D, L)
    # We want: sin(freq * x) for each freq and each dimension
    x_expanded = x.unsqueeze(-1)  # (*, D, 1)
    freq_expanded = freq_bands.view(*([1] * (x.dim() - 1)), 1, -1)  # (1, ..., 1, 1, L)
    
    # (*, D, L)
    scaled = x_expanded * freq_expanded.expand(*x_expanded.shape[:-1], -1)
    
    # Compute sin and cos
    sin_features = torch.sin(scaled)  # (*, D, L)
    cos_features = torch.cos(scaled)  # (*, D, L)
    
    # Interleave sin and cos: [sin(f1*x), cos(f1*x), sin(f2*x), cos(f2*x), ...]
    # Shape: (*, D, 2L)
    features = torch.stack([sin_features, cos_features], dim=-1)  # (*, D, L, 2)
    features = features.reshape(*x.shape[:-1], -1)  # (*, D * 2 * L)
    
    if include_input:
        features = torch.cat([x, features], dim=-1)
    
    return features


class PositionalEncoding(nn.Module):
    """Positional encoding as a nn.Module.
    
    Wraps the positional_encoding function for use in nn.Sequential pipelines.
    
    Args:
        num_frequencies: Number of frequency bands.
        include_input: Include raw input. Default: True.
        log_sampling: Use log-linear frequency spacing. Default: True.
        max_frequency: Maximum frequency exponent. Default: None.
    
    Attributes:
        output_dim: Output dimensionality given input_dim.
    
    Example:
        >>> encoder = PositionalEncoding(num_frequencies=10, include_input=True)
        >>> print(encoder.output_dim(3))  # 63 for 3D input
    """
    
    def __init__(
        self,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
        max_frequency: Optional[float] = None,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.max_frequency = max_frequency
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input coordinates of shape (*, D).
            
        Returns:
            Encoded features.
        """
        return positional_encoding(
            x,
            num_frequencies=self.num_frequencies,
            include_input=self.include_input,
            log_sampling=self.log_sampling,
            max_frequency=self.max_frequency,
        )
    
    def output_dim(self, input_dim: int) -> int:
        """Compute output dimensionality for a given input dimension.
        
        Args:
            input_dim: Input feature dimension.
            
        Returns:
            Output feature dimension.
        """
        base = input_dim * 2 * self.num_frequencies
        if self.include_input:
            base += input_dim
        return base
    
    def extra_repr(self) -> str:
        return (f'num_frequencies={self.num_frequencies}, '
                f'include_input={self.include_input}, log_sampling={self.log_sampling}')


class IntegratedPositionalEncoding(nn.Module):
    """Integrated Positional Encoding from Mip-NeRF.
    
    Instead of encoding point samples, encodes conical frustums by integrating
    the positional encoding over the frustum volume. This reduces aliasing
    artifacts in NeRF-style rendering.
    
    Args:
        num_frequencies: Number of frequency bands.
        include_input: Include raw input.
    
    Note:
        This is a simplified version. Full Mip-NeRF IPE requires ray parameterization.
    """
    
    def __init__(
        self,
        num_frequencies: int = 6,
        include_input: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Precompute frequency bands
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands * math.pi)
    
    def forward(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Apply integrated positional encoding.
        
        For a Gaussian with mean μ and variance σ², the expected value of
        sin(ωx) where x ~ N(μ, σ²) is:
            E[sin(ωx)] = sin(ωμ) * exp(-ω²σ²/2)
        
        Args:
            mean: Mean positions, shape (*, D).
            variance: Variance (isotropic), shape (*, D) or (*, 1).
            
        Returns:
            Integrated positional encoding.
        """
        # Expand variance if scalar per point
        if variance.shape[-1] == 1:
            variance = variance.expand_as(mean)
        
        # (*, D, 1) x (L,) -> (*, D, L)
        mean_exp = mean.unsqueeze(-1)
        var_exp = variance.unsqueeze(-1)
        freq_exp = self.freq_bands.view(*([1] * mean.dim()), -1)
        
        # Scaled mean and variance
        scaled_mean = mean_exp * freq_exp  # (*, D, L)
        scaled_var = var_exp * (freq_exp ** 2)  # (*, D, L)
        
        # Damping factor from variance
        damping = torch.exp(-0.5 * scaled_var)
        
        # Expected sin and cos
        sin_features = torch.sin(scaled_mean) * damping
        cos_features = torch.cos(scaled_mean) * damping
        
        # Combine
        features = torch.cat([
            sin_features.reshape(*mean.shape[:-1], -1),
            cos_features.reshape(*mean.shape[:-1], -1),
        ], dim=-1)
        
        if self.include_input:
            features = torch.cat([mean, features], dim=-1)
        
        return features


def compute_encoding_dim(
    input_dim: int,
    num_frequencies: int,
    include_input: bool = True,
) -> int:
    """Compute the output dimensionality of positional encoding.
    
    Args:
        input_dim: Input coordinate dimension (e.g., 3 for xyz).
        num_frequencies: Number of frequency bands.
        include_input: Whether raw input is included.
    
    Returns:
        Output feature dimension.
    
    Example:
        >>> dim = compute_encoding_dim(3, 10, include_input=True)
        >>> print(dim)  # 63
    """
    encoding_dim = input_dim * 2 * num_frequencies
    if include_input:
        encoding_dim += input_dim
    return encoding_dim
