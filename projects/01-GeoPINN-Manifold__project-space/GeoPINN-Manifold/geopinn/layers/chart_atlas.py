"""
Chart Atlas for Manifold PDEs

Handles manifolds without global coordinates by partitioning into overlapping charts.
Each chart provides local 2D coordinates via tangent plane projection.

Key components:
- k-means clustering to partition manifold into patches
- Per-chart tangent basis computation
- Gaussian blending for smooth chart transitions
- Chart-aware MLP that operates in local coordinates

Reference: Section on "Chart Stitching / Atlas Approach" in GeoPINN documentation
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional


class Atlas:
    """
    Atlas of overlapping charts covering a manifold.
    
    Uses k-means clustering to define chart centers, then constructs
    local tangent coordinates at each chart.
    
    Args:
        points: [N, 3] point cloud on manifold
        normals: [N, 3] surface normals at each point
        num_charts: number of charts to use (default 6 for sphere-like)
        overlap_sigma: controls Gaussian blending width (relative to chart radius)
    """
    
    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        num_charts: int = 6,
        overlap_sigma: float = 0.5
    ):
        self.num_charts = num_charts
        self.overlap_sigma = overlap_sigma
        
        # Cluster points to find chart centers
        kmeans = KMeans(n_clusters=num_charts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        
        self.chart_centers = kmeans.cluster_centers_  # [num_charts, 3]
        self.labels = labels  # [N] cluster assignments
        
        # Compute chart properties
        self.chart_normals = []
        self.chart_tangent1 = []
        self.chart_tangent2 = []
        self.chart_radii = []
        
        for c in range(num_charts):
            mask = (labels == c)
            chart_points = points[mask]
            chart_normals = normals[mask]
            
            # Average normal for chart (then normalize)
            avg_normal = np.mean(chart_normals, axis=0)
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
            self.chart_normals.append(avg_normal)
            
            # Compute tangent basis using Gram-Schmidt
            t1, t2 = self._compute_tangent_basis(avg_normal)
            self.chart_tangent1.append(t1)
            self.chart_tangent2.append(t2)
            
            # Chart radius: max distance from center to any point in chart
            if len(chart_points) > 0:
                dists = np.linalg.norm(chart_points - self.chart_centers[c], axis=1)
                self.chart_radii.append(np.max(dists))
            else:
                self.chart_radii.append(1.0)
        
        self.chart_normals = np.array(self.chart_normals)
        self.chart_tangent1 = np.array(self.chart_tangent1)
        self.chart_tangent2 = np.array(self.chart_tangent2)
        self.chart_radii = np.array(self.chart_radii)
    
    def _compute_tangent_basis(self, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute orthonormal tangent basis from normal using Gram-Schmidt."""
        # Choose reference not parallel to normal
        if abs(normal[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        # Gram-Schmidt
        t1 = ref - np.dot(ref, normal) * normal
        t1 = t1 / (np.linalg.norm(t1) + 1e-10)
        
        # Cross product for second tangent
        t2 = np.cross(normal, t1)
        
        return t1, t2
    
    def project_to_chart(
        self,
        points: np.ndarray,
        chart_idx: int
    ) -> np.ndarray:
        """
        Project 3D points to 2D coordinates in specified chart.
        
        Args:
            points: [M, 3] points to project
            chart_idx: which chart to use
            
        Returns:
            coords: [M, 2] local (u, v) coordinates
        """
        center = self.chart_centers[chart_idx]
        t1 = self.chart_tangent1[chart_idx]
        t2 = self.chart_tangent2[chart_idx]
        
        # Vector from center to each point
        diff = points - center
        
        # Project onto tangent plane
        u = diff @ t1
        v = diff @ t2
        
        return np.stack([u, v], axis=-1)
    
    def compute_chart_weights(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Compute blending weights for each chart at given points.
        Uses Gaussian weights based on distance to chart centers.
        
        Args:
            points: [M, 3] query points
            
        Returns:
            weights: [M, num_charts] normalized blending weights
        """
        M = points.shape[0]
        weights = np.zeros((M, self.num_charts))
        
        for c in range(self.num_charts):
            center = self.chart_centers[c]
            radius = self.chart_radii[c]
            sigma = self.overlap_sigma * radius
            
            # Distance to chart center
            dist = np.linalg.norm(points - center, axis=1)
            
            # Gaussian weight
            weights[:, c] = np.exp(-0.5 * (dist / sigma) ** 2)
        
        # Normalize to sum to 1 (partition of unity)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
        
        return weights
    
    def to_torch(self, device: str = 'cpu') -> 'TorchAtlas':
        """Convert to PyTorch tensors for use in training."""
        return TorchAtlas(
            chart_centers=torch.tensor(self.chart_centers, dtype=torch.float32, device=device),
            chart_normals=torch.tensor(self.chart_normals, dtype=torch.float32, device=device),
            chart_tangent1=torch.tensor(self.chart_tangent1, dtype=torch.float32, device=device),
            chart_tangent2=torch.tensor(self.chart_tangent2, dtype=torch.float32, device=device),
            chart_radii=torch.tensor(self.chart_radii, dtype=torch.float32, device=device),
            overlap_sigma=self.overlap_sigma,
            num_charts=self.num_charts
        )


class TorchAtlas:
    """PyTorch version of Atlas for GPU training."""
    
    def __init__(
        self,
        chart_centers: torch.Tensor,
        chart_normals: torch.Tensor,
        chart_tangent1: torch.Tensor,
        chart_tangent2: torch.Tensor,
        chart_radii: torch.Tensor,
        overlap_sigma: float,
        num_charts: int
    ):
        self.chart_centers = chart_centers
        self.chart_normals = chart_normals
        self.chart_tangent1 = chart_tangent1
        self.chart_tangent2 = chart_tangent2
        self.chart_radii = chart_radii
        self.overlap_sigma = overlap_sigma
        self.num_charts = num_charts
    
    def project_to_chart(
        self,
        points: torch.Tensor,
        chart_idx: int
    ) -> torch.Tensor:
        """Project points to local 2D coordinates in chart."""
        center = self.chart_centers[chart_idx]
        t1 = self.chart_tangent1[chart_idx]
        t2 = self.chart_tangent2[chart_idx]
        
        diff = points - center
        u = (diff * t1).sum(dim=-1)
        v = (diff * t2).sum(dim=-1)
        
        return torch.stack([u, v], dim=-1)
    
    def compute_chart_weights(
        self,
        points: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized Gaussian blending weights."""
        M = points.shape[0]
        device = points.device
        weights = torch.zeros(M, self.num_charts, device=device)
        
        for c in range(self.num_charts):
            center = self.chart_centers[c]
            radius = self.chart_radii[c]
            sigma = self.overlap_sigma * radius
            
            dist = torch.norm(points - center, dim=-1)
            weights[:, c] = torch.exp(-0.5 * (dist / sigma) ** 2)
        
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        
        return weights


class ChartMLP(nn.Module):
    """
    MLP that operates in local chart coordinates.
    
    Takes 2D local coordinates and optionally chart index, outputs field value.
    
    Args:
        hidden_dim: hidden layer dimension
        out_dim: output dimension (1 for scalar field)
        num_layers: number of hidden layers
        include_chart_id: if True, include one-hot chart encoding
        num_charts: number of charts (required if include_chart_id=True)
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 3,
        include_chart_id: bool = False,
        num_charts: int = 6
    ):
        super().__init__()
        
        self.include_chart_id = include_chart_id
        self.num_charts = num_charts
        
        in_dim = 2  # local (u, v) coordinates
        if include_chart_id:
            in_dim += num_charts  # one-hot chart encoding
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
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
            local_coords: [M, 2] local (u, v) coordinates
            chart_idx: [M] chart indices (required if include_chart_id=True)
            
        Returns:
            values: [M, out_dim] field values
        """
        if self.include_chart_id:
            assert chart_idx is not None, "chart_idx required when include_chart_id=True"
            # One-hot encoding
            one_hot = torch.zeros(local_coords.shape[0], self.num_charts, 
                                 device=local_coords.device)
            one_hot.scatter_(1, chart_idx.unsqueeze(-1).long(), 1.0)
            x = torch.cat([local_coords, one_hot], dim=-1)
        else:
            x = local_coords
        
        return self.net(x)


class AtlasPINN(nn.Module):
    """
    Physics-Informed Neural Network using chart atlas.
    
    Evaluates a separate (or shared) MLP in each chart, then blends results.
    
    Args:
        atlas: TorchAtlas instance
        hidden_dim: MLP hidden dimension
        out_dim: output dimension
        shared_mlp: if True, use single MLP with chart conditioning;
                    if False, use separate MLP per chart
    """
    
    def __init__(
        self,
        atlas: TorchAtlas,
        hidden_dim: int = 64,
        out_dim: int = 1,
        shared_mlp: bool = True
    ):
        super().__init__()
        
        self.atlas = atlas
        self.shared_mlp = shared_mlp
        self.num_charts = atlas.num_charts
        
        if shared_mlp:
            self.mlp = ChartMLP(
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                include_chart_id=True,
                num_charts=atlas.num_charts
            )
        else:
            self.mlps = nn.ModuleList([
                ChartMLP(hidden_dim=hidden_dim, out_dim=out_dim, include_chart_id=False)
                for _ in range(atlas.num_charts)
            ])
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate PINN at 3D points using chart blending.
        
        Args:
            points: [M, 3] query points on manifold
            
        Returns:
            values: [M, out_dim] predicted field values
        """
        M = points.shape[0]
        device = points.device
        
        # Get blending weights
        weights = self.atlas.compute_chart_weights(points)  # [M, num_charts]
        
        # Evaluate each chart and blend
        out_dim = self.mlp.net[-1].out_features if self.shared_mlp else self.mlps[0].net[-1].out_features
        blended = torch.zeros(M, out_dim, device=device)
        
        for c in range(self.num_charts):
            # Project to chart coordinates
            local_coords = self.atlas.project_to_chart(points, c)  # [M, 2]
            
            # Evaluate MLP
            if self.shared_mlp:
                chart_idx = torch.full((M,), c, device=device)
                values = self.mlp(local_coords, chart_idx)
            else:
                values = self.mlps[c](local_coords)
            
            # Weighted contribution
            blended += weights[:, c:c+1] * values
        
        return blended
