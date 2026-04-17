"""
Tangent Message Passing Layers
==============================

GNN message passing layers that respect manifold geometry by:
1. Computing tangent bases at each vertex from surface normals
2. Projecting edge vectors onto local tangent planes
3. Encoding relative positions in 2D tangent coordinates
4. Optionally parallel-transporting features between tangent planes

Migrated from Project 01 (GeoPINN-Manifold) geopinn/layers/tangent_message_passing.py
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

from typing import Optional, Tuple

if HAS_TORCH:
    from .basis import compute_tangent_basis


    class TangentMessagePassing(nn.Module):
        """
        Message passing layer that operates in local tangent planes.
        
        For each edge (src -> tgt):
        1. Project the edge vector into the target node's tangent plane
        2. Encode the 2D tangent coordinates
        3. Concatenate with source node features
        4. Pass through MLP and aggregate at target
        
        This ensures geometric equivariance on curved manifolds.
        
        Args:
            in_features: Input feature dimension per node
            out_features: Output feature dimension per node
            hidden_dim: Hidden dimension for message MLP
            aggregation: Aggregation method ('sum', 'mean', 'max')
            use_edge_features: Whether to include edge features
            edge_dim: Dimension of edge features (if used)
            
        Example:
            >>> layer = TangentMessagePassing(16, 32, hidden_dim=64)
            >>> out = layer(node_feats, positions, normals, edge_index)
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_dim: int = 64,
            aggregation: str = 'mean',
            use_edge_features: bool = False,
            edge_dim: int = 0,
        ):
            super().__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            self.aggregation = aggregation
            self.use_edge_features = use_edge_features
            
            # Message MLP: takes (features + 2D tangent coords + optional edge features)
            message_in_dim = in_features + 2
            if use_edge_features:
                message_in_dim += edge_dim
            
            self.message_mlp = nn.Sequential(
                nn.Linear(message_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            # Update MLP: combines aggregated messages with node features
            self.update_mlp = nn.Sequential(
                nn.Linear(hidden_dim + in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_features),
            )
        
        def forward(
            self,
            node_feats: torch.Tensor,
            pos: torch.Tensor,
            normals: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass with tangent plane message passing.
            
            Args:
                node_feats: (N, in_features) node feature matrix
                pos: (N, 3) node positions in 3D
                normals: (N, 3) unit surface normals at each node
                edge_index: (2, E) edge indices (src, tgt)
                edge_attr: (E, edge_dim) optional edge features
                
            Returns:
                out_feats: (N, out_features) updated node features
            """
            N = node_feats.size(0)
            device = node_feats.device
            
            # Compute tangent basis for all nodes
            t1, t2 = compute_tangent_basis(normals)  # (N, 3) each
            
            src, tgt = edge_index[0], edge_index[1]
            
            # Compute edge vectors from source to target
            edge_vec = pos[src] - pos[tgt]  # (E, 3)
            
            # Get target node's normal and tangent basis
            n_tgt = normals[tgt]  # (E, 3)
            t1_tgt = t1[tgt]      # (E, 3)
            t2_tgt = t2[tgt]      # (E, 3)
            
            # Project edge vector onto tangent plane at target
            # v_tangent = v - (v·n)n
            normal_component = torch.sum(edge_vec * n_tgt, dim=-1, keepdim=True)
            edge_tangent = edge_vec - normal_component * n_tgt
            
            # Encode as 2D coordinates in tangent plane
            u = torch.sum(edge_tangent * t1_tgt, dim=-1, keepdim=True)  # (E, 1)
            w = torch.sum(edge_tangent * t2_tgt, dim=-1, keepdim=True)  # (E, 1)
            tangent_coords = torch.cat([u, w], dim=-1)  # (E, 2)
            
            # Get source node features
            src_feats = node_feats[src]  # (E, in_features)
            
            # Build message input
            message_input = torch.cat([src_feats, tangent_coords], dim=-1)
            if self.use_edge_features and edge_attr is not None:
                message_input = torch.cat([message_input, edge_attr], dim=-1)
            
            # Compute messages
            messages = self.message_mlp(message_input)  # (E, hidden_dim)
            
            # Aggregate messages at target nodes
            hidden_dim = messages.size(-1)
            aggregated = torch.zeros(N, hidden_dim, device=device, dtype=messages.dtype)
            
            if self.aggregation == 'sum':
                aggregated.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, hidden_dim), messages)
            elif self.aggregation == 'mean':
                aggregated.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, hidden_dim), messages)
                # Count edges per node for mean
                count = torch.zeros(N, device=device, dtype=messages.dtype)
                count.scatter_add_(0, tgt, torch.ones(tgt.size(0), device=device, dtype=messages.dtype))
                count = count.clamp(min=1).unsqueeze(-1)
                aggregated = aggregated / count
            elif self.aggregation == 'max':
                aggregated.scatter_reduce_(
                    0, tgt.unsqueeze(-1).expand(-1, hidden_dim), messages,
                    reduce='amax', include_self=False
                )
            
            # Update: combine aggregated messages with original features
            update_input = torch.cat([aggregated, node_feats], dim=-1)
            out_feats = self.update_mlp(update_input)
            
            return out_feats


    class TangentMessagePassingStack(nn.Module):
        """
        Stack of Tangent Message Passing layers with residual connections.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            num_layers: Number of message passing layers
            aggregation: Aggregation method
            dropout: Dropout probability
            
        Example:
            >>> stack = TangentMessagePassingStack(3, 64, 32, num_layers=4)
            >>> out = stack(features, positions, normals, edge_index)
        """
        
        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            num_layers: int = 3,
            aggregation: str = 'mean',
            dropout: float = 0.0,
        ):
            super().__init__()
            
            self.input_proj = nn.Linear(in_features, hidden_features)
            
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(
                    TangentMessagePassing(
                        hidden_features, hidden_features,
                        hidden_dim=hidden_features,
                        aggregation=aggregation
                    )
                )
            
            self.output_proj = nn.Linear(hidden_features, out_features)
            self.layer_norm = nn.LayerNorm(hidden_features)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        def forward(
            self,
            node_feats: torch.Tensor,
            pos: torch.Tensor,
            normals: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
            """Forward with residual connections."""
            x = self.input_proj(node_feats)
            
            for layer in self.layers:
                residual = x
                x = layer(x, pos, normals, edge_index)
                x = self.layer_norm(x + residual)
                x = self.dropout(x)
            
            return self.output_proj(x)


    class TangentConv(nn.Module):
        """
        Tangent-space convolution with learned kernels.
        
        Unlike TangentMessagePassing which uses MLPs, this layer learns
        explicit 2D convolution kernels in tangent space.
        
        For each vertex:
        1. Project neighbors to local tangent coordinates
        2. Apply learned 2D kernel based on relative positions
        3. Aggregate weighted contributions
        
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            kernel_type: Type of kernel ('gaussian', 'polynomial', 'mlp')
            num_basis: Number of basis functions for kernel
            aggregation: Aggregation method
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_type: str = 'gaussian',
            num_basis: int = 8,
            aggregation: str = 'sum',
        ):
            super().__init__()
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_type = kernel_type
            self.num_basis = num_basis
            self.aggregation = aggregation
            
            if kernel_type == 'gaussian':
                # Learn centers and widths of Gaussian kernels
                self.centers = nn.Parameter(torch.randn(num_basis, 2) * 0.1)
                self.log_widths = nn.Parameter(torch.zeros(num_basis))
            elif kernel_type == 'polynomial':
                # Polynomial coefficients up to degree sqrt(num_basis)
                self.poly_coeffs = nn.Parameter(torch.randn(num_basis))
            elif kernel_type == 'mlp':
                # MLP to compute kernel from tangent coordinates
                self.kernel_mlp = nn.Sequential(
                    nn.Linear(2, num_basis),
                    nn.ReLU(),
                    nn.Linear(num_basis, num_basis),
                )
            
            # Feature transformation
            self.weight = nn.Parameter(torch.Tensor(num_basis, in_channels, out_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            
            self._init_parameters()
        
        def _init_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)
        
        def forward(
            self,
            x: torch.Tensor,
            pos: torch.Tensor,
            normals: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
            """
            Apply tangent-space convolution.
            
            Args:
                x: (N, in_channels) node features
                pos: (N, 3) node positions
                normals: (N, 3) surface normals
                edge_index: (2, E) edge indices
                
            Returns:
                out: (N, out_channels) convolved features
            """
            N = x.size(0)
            device = x.device
            
            # Compute tangent bases
            t1, t2 = compute_tangent_basis(normals)
            
            src, tgt = edge_index[0], edge_index[1]
            
            # Project edge vectors to tangent planes
            edge_vec = pos[src] - pos[tgt]
            n_tgt = normals[tgt]
            t1_tgt, t2_tgt = t1[tgt], t2[tgt]
            
            # Remove normal component
            edge_tangent = edge_vec - torch.sum(edge_vec * n_tgt, dim=-1, keepdim=True) * n_tgt
            
            # 2D coordinates
            tangent_coords = torch.stack([
                torch.sum(edge_tangent * t1_tgt, dim=-1),
                torch.sum(edge_tangent * t2_tgt, dim=-1),
            ], dim=-1)  # (E, 2)
            
            # Compute kernel values
            if self.kernel_type == 'gaussian':
                # Gaussian RBF kernels
                diff = tangent_coords.unsqueeze(1) - self.centers.unsqueeze(0)  # (E, num_basis, 2)
                widths = torch.exp(self.log_widths)  # (num_basis,)
                kernel_vals = torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * widths ** 2))  # (E, num_basis)
            elif self.kernel_type == 'polynomial':
                # Polynomial kernel
                r2 = torch.sum(tangent_coords ** 2, dim=-1, keepdim=True)  # (E, 1)
                powers = torch.arange(self.num_basis, device=device, dtype=x.dtype)
                kernel_vals = (r2 ** powers) * self.poly_coeffs  # (E, num_basis)
            else:  # mlp
                kernel_vals = self.kernel_mlp(tangent_coords)  # (E, num_basis)
            
            # Apply kernel-weighted transformation
            # kernel_vals: (E, num_basis), x[src]: (E, in_channels), weight: (num_basis, in_channels, out_channels)
            src_feats = x[src]  # (E, in_channels)
            
            # Weighted features: (E, num_basis, out_channels)
            weighted = torch.einsum('eb,bi,bio->eo', kernel_vals, src_feats.unsqueeze(1).expand(-1, self.num_basis, -1), self.weight)
            
            # This is inefficient - simplify:
            # messages = Σ_b kernel_vals[e,b] * (x[src[e]] @ weight[b])
            messages = torch.zeros(edge_index.size(1), self.out_channels, device=device, dtype=x.dtype)
            for b in range(self.num_basis):
                messages += kernel_vals[:, b:b+1] * (src_feats @ self.weight[b])
            
            # Aggregate
            out = torch.zeros(N, self.out_channels, device=device, dtype=x.dtype)
            
            if self.aggregation == 'sum':
                out.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.out_channels), messages)
            elif self.aggregation == 'mean':
                out.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.out_channels), messages)
                count = torch.zeros(N, device=device, dtype=x.dtype)
                count.scatter_add_(0, tgt, torch.ones(tgt.size(0), device=device, dtype=x.dtype))
                out = out / count.clamp(min=1).unsqueeze(-1)
            
            return out + self.bias

else:
    # Stub classes
    class TangentMessagePassing:
        def __init__(self, *args, **kwargs):
            raise ImportError("TangentMessagePassing requires PyTorch")
    
    class TangentMessagePassingStack:
        def __init__(self, *args, **kwargs):
            raise ImportError("TangentMessagePassingStack requires PyTorch")
    
    class TangentConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("TangentConv requires PyTorch")
