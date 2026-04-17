"""
Spectral Convolution Module for Manifold Geometry
=================================================

Spectral methods on manifolds using Laplacian eigenbasis.
Projects signals onto eigenvectors, applies learnable/fixed spectral filters,
then reconstructs in spatial domain.

Migrated from Project 01 (GeoPINN-Manifold) for use across multiple projects.

Key Components:
- compute_laplacian_eigenpairs: Compute k smallest eigenpairs
- SpectralGraphConv: General spectral convolution layer
- HeatKernelConv: Heat diffusion filter h(λ) = exp(-tλ)
- WaveKernelConv: Wave propagation filter h(λ) = cos(t√λ)
- ChebConv: Chebyshev polynomial approximation (faster, no eigendecomp)

Usage:
    >>> from diffgeo.spectral import SpectralGraphConv, compute_laplacian_eigenpairs
    >>>
    >>> # Compute eigenpairs
    >>> eigenvals, eigenvecs = compute_laplacian_eigenpairs(mesh, k=50)
    >>>
    >>> # Create spectral conv layer
    >>> conv = SpectralGraphConv(eigenvals, eigenvecs, in_channels=3, out_channels=16)
    >>> features_out = conv(features_in)

For efficient computation without eigendecomposition:
    >>> from diffgeo.spectral import ChebConv
    >>> conv = ChebConv(mesh.laplacian, in_channels=3, out_channels=16, K=5)
"""

from .eigenpairs import compute_laplacian_eigenpairs, SpectralBasis
from .spectral_conv import SpectralGraphConv, HeatKernelConv, WaveKernelConv
from .chebyshev import ChebConv

__all__ = [
    # Eigenpair computation
    "compute_laplacian_eigenpairs",
    "SpectralBasis",
    # Spectral convolution
    "SpectralGraphConv",
    "HeatKernelConv", 
    "WaveKernelConv",
    # Chebyshev approximation
    "ChebConv",
]
