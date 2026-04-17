"""
Chebyshev Polynomial Convolution
================================

Efficient spectral convolution using Chebyshev polynomials of the Laplacian.
Avoids explicit eigendecomposition, making it O(kE) instead of O(V³).

Based on ChebNet (Defferrard et al., 2016):
    g_θ * x ≈ Σ_{k=0}^{K-1} θ_k T_k(L̃) x

where T_k are Chebyshev polynomials and L̃ is the scaled Laplacian.
"""

try:
    import torch
    import torch.nn as nn
    from scipy import sparse
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

import numpy as np
from typing import Union, Optional


if HAS_TORCH:

    def chebyshev_polynomials(
        L: torch.Tensor,
        x: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """
        Compute Chebyshev polynomial basis T_0(L)x, T_1(L)x, ..., T_{K-1}(L)x.
        
        Uses the recurrence relation:
            T_0(L) = I
            T_1(L) = L
            T_k(L) = 2L @ T_{k-1}(L) - T_{k-2}(L)
        
        Args:
            L: (V, V) scaled Laplacian (sparse or dense)
            x: (V, C) input signals
            K: Number of Chebyshev polynomials
            
        Returns:
            Tx: (K, V, C) polynomial evaluations
        """
        V, C = x.shape
        
        Tx = [x]  # T_0(L)x = x
        
        if K > 1:
            # T_1(L)x = Lx
            if L.is_sparse:
                Lx = torch.sparse.mm(L, x)
            else:
                Lx = L @ x
            Tx.append(Lx)
        
        for k in range(2, K):
            # T_k(L)x = 2L @ T_{k-1}(L)x - T_{k-2}(L)x
            if L.is_sparse:
                Lx_prev = torch.sparse.mm(L, Tx[-1])
            else:
                Lx_prev = L @ Tx[-1]
            Tx_k = 2 * Lx_prev - Tx[-2]
            Tx.append(Tx_k)
        
        return torch.stack(Tx, dim=0)  # (K, V, C)


    def scale_laplacian(
        L: Union[np.ndarray, sparse.csr_matrix, torch.Tensor],
        lambda_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Scale Laplacian to [-1, 1] for Chebyshev polynomials.
        
        L̃ = 2L/λ_max - I
        
        Args:
            L: Laplacian matrix
            lambda_max: Largest eigenvalue (estimated if not provided)
            
        Returns:
            L_scaled: Scaled Laplacian as torch sparse tensor
        """
        # Convert to sparse if needed
        if isinstance(L, np.ndarray):
            L = sparse.csr_matrix(L)
        elif isinstance(L, torch.Tensor) and not L.is_sparse:
            L = sparse.csr_matrix(L.numpy())
        elif isinstance(L, torch.Tensor):
            # Already sparse torch tensor
            if lambda_max is None:
                lambda_max = 2.0  # Estimate
            
            V = L.shape[0]
            indices = L.indices()
            values = L.values()
            
            # Scale: 2L/λ_max - I
            scaled_values = 2.0 * values / lambda_max
            
            # Subtract identity (add -1 to diagonal)
            diag_indices = torch.arange(V, device=L.device)
            all_indices = torch.cat([indices, torch.stack([diag_indices, diag_indices])], dim=1)
            all_values = torch.cat([scaled_values, -torch.ones(V, device=L.device)])
            
            return torch.sparse_coo_tensor(all_indices, all_values, (V, V)).coalesce()
        
        # Handle scipy sparse matrix
        if lambda_max is None:
            from scipy.sparse.linalg import eigsh
            try:
                lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
            except:
                lambda_max = 2.0  # Fallback estimate
        
        V = L.shape[0]
        L_scaled = (2.0 / lambda_max) * L - sparse.eye(V)
        
        # Convert to torch sparse
        L_coo = L_scaled.tocoo()
        indices = torch.tensor([L_coo.row, L_coo.col], dtype=torch.long)
        values = torch.tensor(L_coo.data, dtype=torch.float32)
        
        return torch.sparse_coo_tensor(indices, values, (V, V))


    class ChebConv(nn.Module):
        """
        Chebyshev spectral convolution layer.
        
        Efficient O(K|E|) implementation using Chebyshev polynomial recurrence.
        No eigendecomposition required.
        
        Args:
            laplacian: (V, V) Laplacian matrix (numpy, scipy sparse, or torch)
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            K: Number of Chebyshev polynomial terms
            lambda_max: Largest Laplacian eigenvalue (estimated if None)
            bias: Whether to include bias term
            
        Example:
            >>> conv = ChebConv(mesh.laplacian, in_channels=3, out_channels=16, K=5)
            >>> out = conv(features)  # (V, 3) -> (V, 16)
        """
        
        def __init__(
            self,
            laplacian: Union[np.ndarray, sparse.csr_matrix, torch.Tensor],
            in_channels: int,
            out_channels: int,
            K: int = 5,
            lambda_max: Optional[float] = None,
            bias: bool = True,
        ):
            super().__init__()
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.K = K
            
            # Scale Laplacian
            L_scaled = scale_laplacian(laplacian, lambda_max)
            self.register_buffer('L', L_scaled)
            
            # Learnable parameters
            # One filter weight for each (input channel, output channel, polynomial order)
            self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            
            self._init_parameters()
        
        def _init_parameters(self):
            """Initialize parameters."""
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Apply Chebyshev convolution.
            
            Args:
                x: (V, in_channels) or (B, V, in_channels) input features
                
            Returns:
                out: (V, out_channels) or (B, V, out_channels) output features
            """
            # Handle batched input
            if x.dim() == 3:
                B, V, C = x.shape
                batched = True
                # Process each batch separately
                outputs = []
                for b in range(B):
                    outputs.append(self._forward_single(x[b]))
                return torch.stack(outputs, dim=0)
            else:
                batched = False
                return self._forward_single(x)
        
        def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
            """Forward for single (non-batched) input."""
            # Compute Chebyshev basis: (K, V, in_channels)
            Tx = chebyshev_polynomials(self.L, x, self.K)
            
            # Apply filter: (K, in_channels, out_channels) @ (K, V, in_channels)
            # Result: (V, out_channels)
            out = torch.einsum('kio,kvi->vo', self.weight, Tx)
            
            if self.bias is not None:
                out = out + self.bias
            
            return out
        
        def set_laplacian(self, laplacian: Union[np.ndarray, torch.Tensor], lambda_max: Optional[float] = None):
            """Update Laplacian (e.g., when mesh deforms)."""
            self.L = scale_laplacian(laplacian, lambda_max)


    class ChebConvStack(nn.Module):
        """
        Stack of Chebyshev convolution layers with nonlinearities.
        
        Args:
            laplacian: (V, V) Laplacian matrix
            channels: List of channel dimensions
            K: Chebyshev polynomial order for each layer (or single int)
            activation: Activation function
            dropout: Dropout probability
            residual: Whether to use residual connections
        """
        
        def __init__(
            self,
            laplacian: Union[np.ndarray, torch.Tensor],
            channels: list,
            K: Union[int, list] = 5,
            activation: str = 'relu',
            dropout: float = 0.0,
            residual: bool = True,
        ):
            super().__init__()
            
            self.residual = residual
            
            # K per layer
            if isinstance(K, int):
                K = [K] * (len(channels) - 1)
            
            # Activation
            if activation == 'relu':
                act_fn = nn.ReLU
            elif activation == 'gelu':
                act_fn = nn.GELU
            elif activation == 'tanh':
                act_fn = nn.Tanh
            else:
                act_fn = nn.ReLU
            
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.acts = nn.ModuleList()
            
            for i in range(len(channels) - 1):
                self.convs.append(ChebConv(
                    laplacian,
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    K=K[i],
                ))
                self.norms.append(nn.LayerNorm(channels[i+1]))
                
                if i < len(channels) - 2:
                    self.acts.append(act_fn())
                else:
                    self.acts.append(nn.Identity())
            
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            
            # Residual projections
            if residual:
                self.res_projs = nn.ModuleList()
                for i in range(len(channels) - 1):
                    if channels[i] != channels[i+1]:
                        self.res_projs.append(nn.Linear(channels[i], channels[i+1], bias=False))
                    else:
                        self.res_projs.append(nn.Identity())
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through Chebyshev conv stack."""
            for i, (conv, norm, act) in enumerate(zip(self.convs, self.norms, self.acts)):
                residual = x if self.residual else None
                
                x = conv(x)
                x = norm(x)
                
                if self.residual:
                    x = x + self.res_projs[i](residual)
                
                x = act(x)
                x = self.dropout(x)
            
            return x

else:
    # Stub classes
    class ChebConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChebConv requires PyTorch")
    
    class ChebConvStack:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChebConvStack requires PyTorch")
