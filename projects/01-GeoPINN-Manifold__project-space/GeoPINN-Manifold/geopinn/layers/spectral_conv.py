"""
Spectral Graph Convolution Layer

Projects signals onto Laplacian eigenbasis, applies learnable/fixed spectral filter,
then reconstructs. This enables frequency-domain convolution on manifolds.

Mathematical formulation:
1. Project: f_hat = Φᵀf where Φ is eigenvector matrix [N, k]
2. Filter: g_hat = h(λ) · f_hat where h(λ) is learnable or fixed (e.g., heat kernel e^(-tλ))
3. Reconstruct: g = Φ g_hat

Requires precomputed Laplacian eigenpairs from scipy.sparse.linalg.eigsh
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from typing import Optional, Callable, Tuple


def compute_laplacian_eigenpairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: int = 50,
    return_sparse: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k smallest eigenpairs of the cotangent Laplacian.
    
    Args:
        vertices: [N, 3] vertex positions
        faces: [F, 3] triangle indices
        k: number of eigenpairs to compute
        return_sparse: if True, also return sparse Laplacian matrix
        
    Returns:
        eigenvalues: [k] sorted eigenvalues (smallest first)
        eigenvectors: [N, k] corresponding eigenvectors
    """
    N = vertices.shape[0]
    F = faces.shape[0]
    
    # Build cotangent Laplacian using cotan weights
    # L[i,j] = -cot(α_ij) - cot(β_ij) for edge (i,j)
    # L[i,i] = sum of off-diagonal entries in row i
    
    # Initialize sparse matrix components
    rows, cols, data = [], [], []
    
    for f in range(F):
        i, j, k_idx = faces[f]
        
        # Get vertices of triangle
        vi, vj, vk = vertices[i], vertices[j], vertices[k_idx]
        
        # Compute edge vectors
        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk
        
        # Compute cotangent weights
        # cot(angle at i) = (e_ij · e_ki) / |e_ij × e_ki|
        def cotan(e1, e2):
            cross = np.cross(e1, e2)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-10:
                return 0.0
            dot = np.dot(e1, e2)
            return dot / cross_norm
        
        # Cotangent weights for each edge of triangle
        cot_i = cotan(e_ij, -e_ki)  # angle at i
        cot_j = cotan(e_jk, -e_ij)  # angle at j  
        cot_k = cotan(e_ki, -e_jk)  # angle at k
        
        # Add contributions to Laplacian
        # Edge (j, k) opposite to vertex i
        w_jk = 0.5 * cot_i
        rows.extend([j, k, j, k])
        cols.extend([k, j, j, k])
        data.extend([-w_jk, -w_jk, w_jk, w_jk])
        
        # Edge (i, k) opposite to vertex j
        w_ik = 0.5 * cot_j
        rows.extend([i, k_idx, i, k_idx])
        cols.extend([k_idx, i, i, k_idx])
        data.extend([-w_ik, -w_ik, w_ik, w_ik])
        
        # Edge (i, j) opposite to vertex k
        w_ij = 0.5 * cot_k
        rows.extend([i, j, i, j])
        cols.extend([j, i, i, j])
        data.extend([-w_ij, -w_ij, w_ij, w_ij])
    
    # Build sparse Laplacian
    L = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Symmetrize (handle numerical issues)
    L = 0.5 * (L + L.T)
    
    # Compute eigenpairs (smallest eigenvalues)
    # Use shift-invert for better numerical stability
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if return_sparse:
        return eigenvalues, eigenvectors, L
    return eigenvalues, eigenvectors


class SpectralGraphConv(nn.Module):
    """
    Spectral convolution layer using precomputed Laplacian eigenbasis.
    
    Supports:
    - Learnable spectral filters (per-frequency weights)
    - Fixed filters: heat kernel, wave equation, etc.
    - Multi-channel input/output
    
    Args:
        eigenvals: [k] Laplacian eigenvalues
        eigenvecs: [N, k] Laplacian eigenvectors  
        in_channels: Input feature channels
        out_channels: Output feature channels
        learnable_filter: If True, learn per-frequency weights
        filter_fn: Fixed filter function λ -> h(λ), used if learnable_filter=False
    """
    
    def __init__(
        self,
        eigenvals: torch.Tensor,
        eigenvecs: torch.Tensor,
        in_channels: int = 1,
        out_channels: int = 1,
        learnable_filter: bool = True,
        filter_fn: Optional[Callable] = None
    ):
        super().__init__()
        
        k = eigenvals.shape[0]
        N = eigenvecs.shape[0]
        
        self.register_buffer('eigenvals', eigenvals)  # [k]
        self.register_buffer('eigenvecs', eigenvecs)  # [N, k]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        
        self.learnable_filter = learnable_filter
        
        if learnable_filter:
            # Learnable weights for each frequency and channel pair
            self.filter_weights = nn.Parameter(torch.ones(out_channels, in_channels, k))
            nn.init.xavier_uniform_(self.filter_weights)
        else:
            # Fixed filter from function
            if filter_fn is None:
                # Default: low-pass filter
                filter_fn = lambda lam: torch.exp(-0.5 * lam)
            fixed_filter = filter_fn(eigenvals)
            self.register_buffer('fixed_filter', fixed_filter)
            
            # Channel mixing layer
            if in_channels != out_channels:
                self.channel_mix = nn.Linear(in_channels, out_channels)
            else:
                self.channel_mix = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral convolution.
        
        Args:
            x: [N, in_channels] or [B, N, in_channels] input signals
            
        Returns:
            out: [N, out_channels] or [B, N, out_channels] filtered signals
        """
        # Handle batched input
        if x.dim() == 3:
            B, N, C = x.shape
            batched = True
        else:
            N, C = x.shape
            batched = False
            x = x.unsqueeze(0)  # [1, N, C]
            B = 1
        
        # Project to spectral domain: [B, k, C]
        # x: [B, N, C], eigenvecs.T: [k, N] -> coeffs: [B, k, C]
        coeffs = torch.einsum('kn,bnc->bkc', self.eigenvecs.T, x)
        
        if self.learnable_filter:
            # Apply learnable filter: [out_channels, in_channels, k]
            # coeffs: [B, k, in_channels]
            # Result: [B, k, out_channels]
            filtered = torch.einsum('oik,bki->bko', self.filter_weights, coeffs)
        else:
            # Apply fixed filter: [k]
            filtered = self.fixed_filter.view(1, -1, 1) * coeffs  # [B, k, C]
        
        # Reconstruct in spatial domain: [B, N, out_channels]
        out = torch.einsum('nk,bko->bno', self.eigenvecs, filtered)
        
        if not self.learnable_filter:
            out = self.channel_mix(out)
        
        if not batched:
            out = out.squeeze(0)
        
        return out


class HeatKernelConv(SpectralGraphConv):
    """
    Spectral convolution with heat kernel filter: h(λ) = exp(-tλ)
    
    Simulates heat diffusion on the manifold for time t.
    """
    
    def __init__(
        self,
        eigenvals: torch.Tensor,
        eigenvecs: torch.Tensor,
        diffusion_time: float = 1.0,
        in_channels: int = 1,
        out_channels: int = 1,
        learnable_time: bool = False
    ):
        if learnable_time:
            super().__init__(eigenvals, eigenvecs, in_channels, out_channels, learnable_filter=True)
            # Initialize weights as heat kernel
            with torch.no_grad():
                heat_filter = torch.exp(-diffusion_time * eigenvals)
                self.filter_weights.data = heat_filter.unsqueeze(0).unsqueeze(0).expand(out_channels, in_channels, -1).clone()
        else:
            filter_fn = lambda lam: torch.exp(-diffusion_time * lam)
            super().__init__(eigenvals, eigenvecs, in_channels, out_channels, 
                           learnable_filter=False, filter_fn=filter_fn)
        
        self.diffusion_time = diffusion_time


class WaveKernelConv(SpectralGraphConv):
    """
    Spectral convolution with wave kernel filter: h(λ) = cos(t√λ)
    
    Simulates wave propagation on the manifold.
    """
    
    def __init__(
        self,
        eigenvals: torch.Tensor,
        eigenvecs: torch.Tensor,
        propagation_time: float = 1.0,
        in_channels: int = 1,
        out_channels: int = 1
    ):
        filter_fn = lambda lam: torch.cos(propagation_time * torch.sqrt(lam.clamp(min=0)))
        super().__init__(eigenvals, eigenvecs, in_channels, out_channels,
                        learnable_filter=False, filter_fn=filter_fn)
        
        self.propagation_time = propagation_time
