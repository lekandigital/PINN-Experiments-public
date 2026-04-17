"""
Spectral Graph Convolution Layers
=================================

PyTorch modules for spectral convolution on manifolds.
Uses precomputed Laplacian eigenbasis to perform convolution in frequency domain.

Mathematical formulation:
1. Project: f_hat = Φᵀf where Φ is eigenvector matrix [N, k]
2. Filter: g_hat = h(λ) · f_hat where h(λ) is learnable or fixed
3. Reconstruct: g = Φ g_hat

Migrated from Project 01 (GeoPINN-Manifold) geopinn/layers/spectral_conv.py
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

import numpy as np
from typing import Optional, Callable, Union


if HAS_TORCH:
    
    class SpectralGraphConv(nn.Module):
        """
        Spectral convolution layer using precomputed Laplacian eigenbasis.
        
        Supports:
        - Learnable spectral filters (per-frequency weights)
        - Fixed filters: heat kernel, wave equation, etc.
        - Multi-channel input/output
        
        Args:
            eigenvals: (k,) Laplacian eigenvalues (numpy or torch)
            eigenvecs: (N, k) Laplacian eigenvectors (numpy or torch)
            in_channels: Input feature channels
            out_channels: Output feature channels
            learnable_filter: If True, learn per-frequency weights
            filter_fn: Fixed filter function λ -> h(λ), used if learnable_filter=False
            
        Example:
            >>> eigenvals, eigenvecs = compute_laplacian_eigenpairs(mesh, k=50)
            >>> conv = SpectralGraphConv(eigenvals, eigenvecs, in_channels=3, out_channels=16)
            >>> out = conv(features)  # features: (N, 3) -> out: (N, 16)
        """
        
        def __init__(
            self,
            eigenvals: Union[np.ndarray, torch.Tensor],
            eigenvecs: Union[np.ndarray, torch.Tensor],
            in_channels: int = 1,
            out_channels: int = 1,
            learnable_filter: bool = True,
            filter_fn: Optional[Callable] = None,
        ):
            super().__init__()
            
            # Convert to tensors if needed
            if isinstance(eigenvals, np.ndarray):
                eigenvals = torch.tensor(eigenvals, dtype=torch.float32)
            if isinstance(eigenvecs, np.ndarray):
                eigenvecs = torch.tensor(eigenvecs, dtype=torch.float32)
            
            k = eigenvals.shape[0]
            N = eigenvecs.shape[0]
            
            self.register_buffer('eigenvals', eigenvals)  # (k,)
            self.register_buffer('eigenvecs', eigenvecs)  # (N, k)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.k = k
            self.N = N
            
            self.learnable_filter = learnable_filter
            
            if learnable_filter:
                # Learnable weights for each frequency and channel pair
                self.filter_weights = nn.Parameter(torch.ones(out_channels, in_channels, k))
                nn.init.xavier_uniform_(self.filter_weights)
            else:
                # Fixed filter from function
                if filter_fn is None:
                    # Default: low-pass filter (exponential decay)
                    filter_fn = lambda lam: torch.exp(-0.5 * lam)
                
                with torch.no_grad():
                    fixed_filter = filter_fn(eigenvals)
                self.register_buffer('fixed_filter', fixed_filter)
                
                # Channel mixing layer if dimensions differ
                if in_channels != out_channels:
                    self.channel_mix = nn.Linear(in_channels, out_channels, bias=False)
                else:
                    self.channel_mix = nn.Identity()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Apply spectral convolution.
            
            Args:
                x: (N, in_channels) or (B, N, in_channels) input signals
                
            Returns:
                out: (N, out_channels) or (B, N, out_channels) filtered signals
            """
            # Handle batched input
            if x.dim() == 3:
                B, N, C = x.shape
                batched = True
            else:
                N, C = x.shape
                batched = False
                x = x.unsqueeze(0)  # (1, N, C)
                B = 1
            
            # Project to spectral domain: (B, k, C)
            # x: (B, N, C), eigenvecs.T: (k, N) -> coeffs: (B, k, C)
            coeffs = torch.einsum('kn,bnc->bkc', self.eigenvecs.T, x)
            
            if self.learnable_filter:
                # Apply learnable filter: (out_channels, in_channels, k)
                # coeffs: (B, k, in_channels)
                # Result: (B, k, out_channels)
                filtered = torch.einsum('oik,bki->bko', self.filter_weights, coeffs)
            else:
                # Apply fixed filter: (k,)
                filtered = self.fixed_filter.view(1, -1, 1) * coeffs  # (B, k, C)
            
            # Reconstruct in spatial domain: (B, N, out_channels)
            out = torch.einsum('nk,bko->bno', self.eigenvecs, filtered)
            
            if not self.learnable_filter:
                out = self.channel_mix(out)
            
            if not batched:
                out = out.squeeze(0)
            
            return out
        
        def set_eigenbasis(
            self,
            eigenvals: torch.Tensor,
            eigenvecs: torch.Tensor,
        ):
            """Update eigenbasis (e.g., when mesh changes)."""
            self.eigenvals = eigenvals
            self.eigenvecs = eigenvecs
            self.k = eigenvals.shape[0]
            self.N = eigenvecs.shape[0]


    class HeatKernelConv(SpectralGraphConv):
        """
        Spectral convolution with heat kernel filter: h(λ) = exp(-tλ)
        
        Simulates heat diffusion on the manifold for time t.
        Larger t = more smoothing (low-pass filtering).
        
        Args:
            eigenvals: (k,) Laplacian eigenvalues
            eigenvecs: (N, k) Laplacian eigenvectors
            diffusion_time: Heat diffusion time parameter
            in_channels: Input feature channels
            out_channels: Output feature channels
            learnable_time: If True, make diffusion time learnable
        """
        
        def __init__(
            self,
            eigenvals: Union[np.ndarray, torch.Tensor],
            eigenvecs: Union[np.ndarray, torch.Tensor],
            diffusion_time: float = 1.0,
            in_channels: int = 1,
            out_channels: int = 1,
            learnable_time: bool = False,
        ):
            self.diffusion_time = diffusion_time
            self.learnable_time = learnable_time
            
            if isinstance(eigenvals, np.ndarray):
                eigenvals = torch.tensor(eigenvals, dtype=torch.float32)
            
            if learnable_time:
                # Initialize with heat kernel, but make weights learnable
                super().__init__(
                    eigenvals, eigenvecs, in_channels, out_channels,
                    learnable_filter=True
                )
                # Initialize filter weights as heat kernel
                with torch.no_grad():
                    heat_filter = torch.exp(-diffusion_time * eigenvals)
                    self.filter_weights.data = heat_filter.unsqueeze(0).unsqueeze(0).expand(
                        out_channels, in_channels, -1
                    ).clone()
            else:
                filter_fn = lambda lam: torch.exp(-diffusion_time * lam)
                super().__init__(
                    eigenvals, eigenvecs, in_channels, out_channels,
                    learnable_filter=False, filter_fn=filter_fn
                )


    class WaveKernelConv(SpectralGraphConv):
        """
        Spectral convolution with wave kernel filter: h(λ) = cos(t√λ)
        
        Simulates wave propagation on the manifold.
        
        Args:
            eigenvals: (k,) Laplacian eigenvalues
            eigenvecs: (N, k) Laplacian eigenvectors
            propagation_time: Wave propagation time
            in_channels: Input feature channels
            out_channels: Output feature channels
        """
        
        def __init__(
            self,
            eigenvals: Union[np.ndarray, torch.Tensor],
            eigenvecs: Union[np.ndarray, torch.Tensor],
            propagation_time: float = 1.0,
            in_channels: int = 1,
            out_channels: int = 1,
        ):
            self.propagation_time = propagation_time
            
            if isinstance(eigenvals, np.ndarray):
                eigenvals = torch.tensor(eigenvals, dtype=torch.float32)
            
            filter_fn = lambda lam: torch.cos(propagation_time * torch.sqrt(lam.clamp(min=0)))
            super().__init__(
                eigenvals, eigenvecs, in_channels, out_channels,
                learnable_filter=False, filter_fn=filter_fn
            )


    class BiharmonicKernelConv(SpectralGraphConv):
        """
        Spectral convolution with biharmonic kernel: h(λ) = 1 / (1 + λ²)
        
        Useful for smoothing that preserves sharp features better than heat kernel.
        """
        
        def __init__(
            self,
            eigenvals: Union[np.ndarray, torch.Tensor],
            eigenvecs: Union[np.ndarray, torch.Tensor],
            scale: float = 1.0,
            in_channels: int = 1,
            out_channels: int = 1,
        ):
            if isinstance(eigenvals, np.ndarray):
                eigenvals = torch.tensor(eigenvals, dtype=torch.float32)
            
            filter_fn = lambda lam: 1.0 / (1.0 + scale * lam ** 2)
            super().__init__(
                eigenvals, eigenvecs, in_channels, out_channels,
                learnable_filter=False, filter_fn=filter_fn
            )


    class SpectralConvStack(nn.Module):
        """
        Stack of spectral convolution layers with nonlinearities.
        
        Args:
            eigenvals: (k,) Laplacian eigenvalues
            eigenvecs: (N, k) Laplacian eigenvectors
            channels: List of channel dimensions [in, hidden1, hidden2, ..., out]
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            eigenvals: Union[np.ndarray, torch.Tensor],
            eigenvecs: Union[np.ndarray, torch.Tensor],
            channels: list,
            activation: str = 'relu',
            dropout: float = 0.0,
        ):
            super().__init__()
            
            # Select activation
            if activation == 'relu':
                act_fn = nn.ReLU
            elif activation == 'gelu':
                act_fn = nn.GELU
            elif activation == 'tanh':
                act_fn = nn.Tanh
            else:
                act_fn = nn.ReLU
            
            layers = []
            for i in range(len(channels) - 1):
                layers.append(SpectralGraphConv(
                    eigenvals, eigenvecs,
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    learnable_filter=True,
                ))
                
                # Add activation except for last layer
                if i < len(channels) - 2:
                    layers.append(act_fn())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
            
            self.layers = nn.ModuleList(layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through spectral conv stack."""
            for layer in self.layers:
                x = layer(x)
            return x

else:
    # Stub classes when PyTorch is not available
    class SpectralGraphConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("SpectralGraphConv requires PyTorch")
    
    class HeatKernelConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("HeatKernelConv requires PyTorch")
    
    class WaveKernelConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("WaveKernelConv requires PyTorch")
    
    class BiharmonicKernelConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("BiharmonicKernelConv requires PyTorch")
    
    class SpectralConvStack:
        def __init__(self, *args, **kwargs):
            raise ImportError("SpectralConvStack requires PyTorch")
