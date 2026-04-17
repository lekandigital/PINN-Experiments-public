"""
Chart-Aware Neural Network Modules
==================================

MLP and PINN architectures that operate in local chart coordinates
with smooth blending across chart boundaries.

Migrated from Project 01 (GeoPINN-Manifold) geopinn/layers/chart_atlas.py
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

from typing import Optional

if HAS_TORCH:
    from .atlas import TorchAtlas


    class ChartMLP(nn.Module):
        """
        MLP that operates in local chart coordinates.
        
        Takes 2D local coordinates and optionally chart index, outputs field value.
        
        Args:
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (1 for scalar field)
            num_layers: Number of hidden layers
            include_chart_id: If True, include one-hot chart encoding
            num_charts: Number of charts (required if include_chart_id=True)
            activation: Activation function ('tanh', 'relu', 'gelu', 'silu')
            
        Example:
            >>> mlp = ChartMLP(hidden_dim=64, out_dim=1, include_chart_id=True, num_charts=6)
            >>> values = mlp(local_coords, chart_idx)
        """
        
        def __init__(
            self,
            hidden_dim: int = 64,
            out_dim: int = 1,
            num_layers: int = 3,
            include_chart_id: bool = False,
            num_charts: int = 6,
            activation: str = 'tanh',
        ):
            super().__init__()
            
            self.include_chart_id = include_chart_id
            self.num_charts = num_charts
            
            in_dim = 2  # local (u, v) coordinates
            if include_chart_id:
                in_dim += num_charts  # one-hot chart encoding
            
            # Select activation
            if activation == 'tanh':
                act_fn = nn.Tanh
            elif activation == 'relu':
                act_fn = nn.ReLU
            elif activation == 'gelu':
                act_fn = nn.GELU
            elif activation == 'silu':
                act_fn = nn.SiLU
            else:
                act_fn = nn.Tanh
            
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn())
            
            layers.append(nn.Linear(hidden_dim, out_dim))
            
            self.net = nn.Sequential(*layers)
        
        def forward(
            self,
            local_coords: torch.Tensor,
            chart_idx: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Evaluate MLP at local coordinates.
            
            Args:
                local_coords: (M, 2) local (u, v) coordinates
                chart_idx: (M,) chart indices (required if include_chart_id=True)
                
            Returns:
                values: (M, out_dim) field values
            """
            if self.include_chart_id:
                assert chart_idx is not None, "chart_idx required when include_chart_id=True"
                # One-hot encoding
                one_hot = torch.zeros(
                    local_coords.shape[0], self.num_charts,
                    device=local_coords.device, dtype=local_coords.dtype
                )
                one_hot.scatter_(1, chart_idx.unsqueeze(-1).long(), 1.0)
                x = torch.cat([local_coords, one_hot], dim=-1)
            else:
                x = local_coords
            
            return self.net(x)


    class AtlasPINN(nn.Module):
        """
        Physics-Informed Neural Network using chart atlas.
        
        Evaluates a separate (or shared) MLP in each chart, then blends results
        using Gaussian partition-of-unity weights.
        
        Args:
            atlas: TorchAtlas instance
            hidden_dim: MLP hidden dimension
            out_dim: Output dimension
            num_layers: Number of MLP layers
            shared_mlp: If True, use single MLP with chart conditioning;
                        if False, use separate MLP per chart
            activation: Activation function
            
        Example:
            >>> atlas_torch = atlas.to_torch(device='cuda')
            >>> pinn = AtlasPINN(atlas_torch, hidden_dim=64, out_dim=1)
            >>> values = pinn(points_3d)
        """
        
        def __init__(
            self,
            atlas: TorchAtlas,
            hidden_dim: int = 64,
            out_dim: int = 1,
            num_layers: int = 3,
            shared_mlp: bool = True,
            activation: str = 'tanh',
        ):
            super().__init__()
            
            self.atlas = atlas
            self.shared_mlp = shared_mlp
            self.num_charts = atlas.num_charts
            self.out_dim = out_dim
            
            if shared_mlp:
                self.mlp = ChartMLP(
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    num_layers=num_layers,
                    include_chart_id=True,
                    num_charts=atlas.num_charts,
                    activation=activation,
                )
            else:
                self.mlps = nn.ModuleList([
                    ChartMLP(
                        hidden_dim=hidden_dim,
                        out_dim=out_dim,
                        num_layers=num_layers,
                        include_chart_id=False,
                        activation=activation,
                    )
                    for _ in range(atlas.num_charts)
                ])
        
        def forward(self, points: torch.Tensor) -> torch.Tensor:
            """
            Evaluate PINN at 3D points using chart blending.
            
            Args:
                points: (M, 3) query points on manifold
                
            Returns:
                values: (M, out_dim) predicted field values
            """
            M = points.shape[0]
            device = points.device
            
            # Get blending weights (partition of unity)
            weights = self.atlas.compute_chart_weights(points)  # (M, num_charts)
            
            # Evaluate each chart and blend
            blended = torch.zeros(M, self.out_dim, device=device, dtype=points.dtype)
            
            for c in range(self.num_charts):
                # Project to chart coordinates
                local_coords = self.atlas.project_to_chart(points, c)  # (M, 2)
                
                # Evaluate MLP
                if self.shared_mlp:
                    chart_idx = torch.full((M,), c, device=device, dtype=torch.long)
                    values = self.mlp(local_coords, chart_idx)
                else:
                    values = self.mlps[c](local_coords)
                
                # Weighted contribution
                blended += weights[:, c:c+1] * values
            
            return blended
        
        def forward_with_gradients(
            self,
            points: torch.Tensor
        ) -> tuple:
            """
            Evaluate PINN and compute spatial gradients.
            
            Useful for physics-informed losses that require ∇u.
            
            Args:
                points: (M, 3) query points (requires grad)
                
            Returns:
                values: (M, out_dim) field values
                gradients: (M, out_dim, 3) spatial gradients
            """
            points = points.requires_grad_(True)
            values = self.forward(points)
            
            gradients = []
            for d in range(self.out_dim):
                grad_d = torch.autograd.grad(
                    values[:, d].sum(),
                    points,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradients.append(grad_d)
            
            gradients = torch.stack(gradients, dim=1)  # (M, out_dim, 3)
            
            return values, gradients


    class MultiScaleAtlasPINN(nn.Module):
        """
        Multi-scale PINN using multiple chart atlases at different resolutions.
        
        Useful for problems with multi-scale features (e.g., turbulent flows,
        detailed coastal bathymetry).
        
        Args:
            atlases: List of TorchAtlas at different scales
            hidden_dims: Hidden dimensions per scale
            out_dim: Output dimension
        """
        
        def __init__(
            self,
            atlases: list,
            hidden_dims: list = None,
            out_dim: int = 1,
            activation: str = 'tanh',
        ):
            super().__init__()
            
            self.num_scales = len(atlases)
            
            if hidden_dims is None:
                hidden_dims = [64] * self.num_scales
            
            self.pinns = nn.ModuleList([
                AtlasPINN(atlas, hidden_dim=hd, out_dim=out_dim, activation=activation)
                for atlas, hd in zip(atlases, hidden_dims)
            ])
            
            # Learnable scale blending weights
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        def forward(self, points: torch.Tensor) -> torch.Tensor:
            """Evaluate multi-scale PINN with learned scale blending."""
            weights = torch.softmax(self.scale_weights, dim=0)
            
            output = torch.zeros(points.shape[0], self.pinns[0].out_dim, device=points.device)
            for w, pinn in zip(weights, self.pinns):
                output += w * pinn(points)
            
            return output

else:
    # Stub classes when PyTorch is not available
    class ChartMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChartMLP requires PyTorch")
    
    class AtlasPINN:
        def __init__(self, *args, **kwargs):
            raise ImportError("AtlasPINN requires PyTorch")
    
    class MultiScaleAtlasPINN:
        def __init__(self, *args, **kwargs):
            raise ImportError("MultiScaleAtlasPINN requires PyTorch")
