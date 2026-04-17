"""
Chart Atlas Implementation
==========================

Atlas of overlapping charts covering a manifold. Uses k-means clustering
to define chart centers, then constructs local tangent coordinates at each chart.

Migrated from Project 01 (GeoPINN-Manifold) geopinn/layers/chart_atlas.py
with enhancements for Earth-surface and garment panel domains.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# Optional sklearn import
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _compute_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orthonormal tangent basis from normal using Gram-Schmidt.
    
    Args:
        normal: (3,) unit normal vector
        
    Returns:
        t1, t2: Orthonormal tangent vectors
    """
    # Choose reference not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt: t1 = ref - (ref·n)n, then normalize
    t1 = ref - np.dot(ref, normal) * normal
    t1 = t1 / (np.linalg.norm(t1) + 1e-10)
    
    # Cross product for second tangent
    t2 = np.cross(normal, t1)
    
    return t1, t2


@dataclass
class Chart:
    """
    Single chart in an atlas.
    
    A chart provides local 2D coordinates for a region of the manifold
    via projection onto a tangent plane.
    
    Attributes:
        center: (3,) chart center point in 3D
        normal: (3,) average surface normal at center
        tangent1: (3,) first tangent basis vector
        tangent2: (3,) second tangent basis vector
        radius: Maximum distance from center to covered points
        vertex_indices: Indices of vertices assigned to this chart
        distortion: Maximum metric distortion in this chart
    """
    center: np.ndarray
    normal: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    radius: float
    vertex_indices: Optional[np.ndarray] = None
    distortion: float = 0.0
    projection_type: str = "tangent_plane"
    
    def project_to_local(self, points: np.ndarray) -> np.ndarray:
        """
        Project 3D points to local 2D coordinates.
        
        Args:
            points: (M, 3) points to project
            
        Returns:
            coords: (M, 2) local (u, v) coordinates
        """
        diff = points - self.center
        u = diff @ self.tangent1
        v = diff @ self.tangent2
        return np.stack([u, v], axis=-1)
    
    def project_to_3d(self, local_coords: np.ndarray) -> np.ndarray:
        """
        Map local 2D coordinates back to 3D.
        
        Args:
            local_coords: (M, 2) local coordinates
            
        Returns:
            points: (M, 3) approximate 3D positions
        """
        u, v = local_coords[:, 0], local_coords[:, 1]
        return self.center + u[:, None] * self.tangent1 + v[:, None] * self.tangent2


class Atlas:
    """
    Atlas of overlapping charts covering a manifold.
    
    Uses k-means clustering to define chart centers, then constructs
    local tangent coordinates at each chart. Supports Gaussian blending
    for smooth transitions between charts.
    
    Args:
        points: (N, 3) point cloud on manifold
        normals: (N, 3) surface normals at each point
        num_charts: Number of charts to use (default 6 for sphere-like)
        overlap_sigma: Controls Gaussian blending width (relative to chart radius)
        
    Example:
        >>> atlas = Atlas(sphere_points, sphere_normals, num_charts=6)
        >>> local_coords = atlas.project_to_chart(query_points, chart_idx=0)
        >>> weights = atlas.compute_chart_weights(query_points)
    """
    
    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        num_charts: int = 6,
        overlap_sigma: float = 0.5,
    ):
        if not HAS_SKLEARN:
            raise ImportError(
                "Atlas requires scikit-learn. Install with: pip install scikit-learn"
            )
        
        self.num_charts = num_charts
        self.overlap_sigma = overlap_sigma
        
        # Cluster points to find chart centers
        kmeans = KMeans(n_clusters=num_charts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        
        self.chart_centers = kmeans.cluster_centers_  # (num_charts, 3)
        self.labels = labels  # (N,) cluster assignments
        
        # Build charts
        self.charts: List[Chart] = []
        
        for c in range(num_charts):
            mask = (labels == c)
            chart_points = points[mask]
            chart_normals = normals[mask]
            
            # Average normal for chart (then normalize)
            if len(chart_normals) > 0:
                avg_normal = np.mean(chart_normals, axis=0)
                avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
            else:
                avg_normal = np.array([0., 0., 1.])
            
            # Compute tangent basis using Gram-Schmidt
            t1, t2 = _compute_tangent_basis(avg_normal)
            
            # Chart radius: max distance from center to any point in chart
            if len(chart_points) > 0:
                dists = np.linalg.norm(chart_points - self.chart_centers[c], axis=1)
                radius = np.max(dists)
            else:
                radius = 1.0
            
            self.charts.append(Chart(
                center=self.chart_centers[c],
                normal=avg_normal,
                tangent1=t1,
                tangent2=t2,
                radius=radius,
                vertex_indices=np.where(mask)[0],
            ))
        
        # Convert to arrays for efficient access
        self.chart_normals = np.array([c.normal for c in self.charts])
        self.chart_tangent1 = np.array([c.tangent1 for c in self.charts])
        self.chart_tangent2 = np.array([c.tangent2 for c in self.charts])
        self.chart_radii = np.array([c.radius for c in self.charts])
    
    @classmethod
    def from_mesh(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_normals: Optional[np.ndarray] = None,
        num_charts: int = 6,
        overlap_sigma: float = 0.5,
    ) -> 'Atlas':
        """
        Build atlas from mesh data.
        
        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            vertex_normals: (V, 3) optional precomputed normals
            num_charts: Number of charts
            overlap_sigma: Blending width parameter
            
        Returns:
            Atlas covering the mesh
        """
        if vertex_normals is None:
            vertex_normals = _compute_vertex_normals(vertices, faces)
        
        return cls(vertices, vertex_normals, num_charts, overlap_sigma)
    
    @classmethod
    def from_earth_patch(
        cls,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution_deg: float = 0.5,
        earth_radius_m: float = 6371000.0,
        num_charts: Optional[int] = None,
        overlap_sigma: float = 0.5,
    ) -> 'Atlas':
        """
        Build atlas for a patch of Earth's surface.
        
        Uses gnomonic (tangent plane) projection which preserves directions
        from the chart center - standard in navigation.
        
        Args:
            lat_min, lat_max: Latitude bounds in degrees
            lon_min, lon_max: Longitude bounds in degrees
            resolution_deg: Grid resolution in degrees
            earth_radius_m: Earth radius in meters
            num_charts: Number of charts (auto-computed if None)
            overlap_sigma: Blending width parameter
            
        Returns:
            Atlas optimized for Earth-surface geometry
        """
        # Generate grid of points on the Earth patch
        lat_range = np.arange(lat_min, lat_max + resolution_deg, resolution_deg)
        lon_range = np.arange(lon_min, lon_max + resolution_deg, resolution_deg)
        
        lats, lons = np.meshgrid(lat_range, lon_range)
        lats = lats.flatten()
        lons = lons.flatten()
        
        # Convert to 3D Cartesian on sphere
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        
        x = earth_radius_m * np.cos(lat_rad) * np.cos(lon_rad)
        y = earth_radius_m * np.cos(lat_rad) * np.sin(lon_rad)
        z = earth_radius_m * np.sin(lat_rad)
        
        points = np.stack([x, y, z], axis=-1)
        
        # On a sphere, normals are just normalized positions
        normals = points / earth_radius_m
        
        # Auto-compute number of charts based on domain size
        if num_charts is None:
            lat_span = lat_max - lat_min
            lon_span = lon_max - lon_min
            # Roughly 1 chart per ~10 degrees
            num_charts = max(4, int(np.sqrt(lat_span * lon_span) / 10))
        
        return cls(points, normals, num_charts, overlap_sigma)
    
    @classmethod
    def from_garment_panel(
        cls,
        flat_vertices: np.ndarray,
        draped_vertices: np.ndarray,
        faces: np.ndarray,
        num_charts: int = 4,
        overlap_sigma: float = 0.6,
    ) -> Tuple['Atlas', 'Atlas']:
        """
        Build atlas pair for a garment panel (flat + draped states).
        
        Key insight: the flat (rest) state IS a single chart - flat fabric
        is intrinsically flat. Charts are needed for the draped state where
        the fabric has been curved in 3D.
        
        Args:
            flat_vertices: (V, 3) rest-state vertices (flat pattern)
            draped_vertices: (V, 3) draped-state vertices (on body)
            faces: (F, 3) triangle indices (same for both states)
            num_charts: Number of charts for draped state
            overlap_sigma: Blending width parameter
            
        Returns:
            (flat_atlas, draped_atlas) tuple
        """
        flat_normals = _compute_vertex_normals(flat_vertices, faces)
        draped_normals = _compute_vertex_normals(draped_vertices, faces)
        
        # Flat state: typically just needs 1 chart since it's flat
        flat_atlas = cls(flat_vertices, flat_normals, num_charts=1, overlap_sigma=1.0)
        
        # Draped state: needs multiple charts due to curvature
        draped_atlas = cls(draped_vertices, draped_normals, num_charts, overlap_sigma)
        
        return flat_atlas, draped_atlas
    
    def project_to_chart(
        self,
        points: np.ndarray,
        chart_idx: int
    ) -> np.ndarray:
        """
        Project 3D points to 2D coordinates in specified chart.
        
        Args:
            points: (M, 3) points to project
            chart_idx: Which chart to use
            
        Returns:
            coords: (M, 2) local (u, v) coordinates
        """
        return self.charts[chart_idx].project_to_local(points)
    
    def compute_chart_weights(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Compute blending weights for each chart at given points.
        Uses Gaussian weights based on distance to chart centers.
        
        Args:
            points: (M, 3) query points
            
        Returns:
            weights: (M, num_charts) normalized blending weights (partition of unity)
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
    
    def get_chart_for_vertex(self, vertex_idx: int) -> int:
        """Return the primary chart index for a vertex."""
        return int(self.labels[vertex_idx])
    
    def get_overlap_vertices(self, chart_a: int, chart_b: int) -> np.ndarray:
        """Return vertex indices in the overlap of two charts."""
        # Vertices belong to both charts if they have significant weight in both
        # For simplicity, use the label-based primary assignment
        verts_a = set(self.charts[chart_a].vertex_indices.tolist())
        verts_b = set(self.charts[chart_b].vertex_indices.tolist())
        
        # Get vertices near the boundary (within overlap_sigma * radius)
        overlap = []
        for v in verts_a:
            dist_to_b = np.linalg.norm(
                self.chart_centers[chart_b] - self.chart_centers[chart_a]
            )
            if dist_to_b < (self.chart_radii[chart_a] + self.chart_radii[chart_b]):
                overlap.append(v)
        
        return np.array(overlap, dtype=np.int64)
    
    def to_torch(self, device: str = 'cpu') -> 'TorchAtlas':
        """Convert to PyTorch tensors for use in training."""
        if not HAS_TORCH:
            raise ImportError("TorchAtlas requires PyTorch")
        
        return TorchAtlas(
            chart_centers=torch.tensor(self.chart_centers, dtype=torch.float32, device=device),
            chart_normals=torch.tensor(self.chart_normals, dtype=torch.float32, device=device),
            chart_tangent1=torch.tensor(self.chart_tangent1, dtype=torch.float32, device=device),
            chart_tangent2=torch.tensor(self.chart_tangent2, dtype=torch.float32, device=device),
            chart_radii=torch.tensor(self.chart_radii, dtype=torch.float32, device=device),
            overlap_sigma=self.overlap_sigma,
            num_charts=self.num_charts,
        )


class TorchAtlas:
    """
    PyTorch version of Atlas for GPU training.
    
    All operations are differentiable and support batched inputs.
    """
    
    def __init__(
        self,
        chart_centers: 'torch.Tensor',
        chart_normals: 'torch.Tensor',
        chart_tangent1: 'torch.Tensor',
        chart_tangent2: 'torch.Tensor',
        chart_radii: 'torch.Tensor',
        overlap_sigma: float,
        num_charts: int,
    ):
        if not HAS_TORCH:
            raise ImportError("TorchAtlas requires PyTorch")
        
        self.chart_centers = chart_centers
        self.chart_normals = chart_normals
        self.chart_tangent1 = chart_tangent1
        self.chart_tangent2 = chart_tangent2
        self.chart_radii = chart_radii
        self.overlap_sigma = overlap_sigma
        self.num_charts = num_charts
    
    def project_to_chart(
        self,
        points: 'torch.Tensor',
        chart_idx: int
    ) -> 'torch.Tensor':
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
        points: 'torch.Tensor'
    ) -> 'torch.Tensor':
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
    
    def to(self, device: str) -> 'TorchAtlas':
        """Move atlas to specified device."""
        return TorchAtlas(
            chart_centers=self.chart_centers.to(device),
            chart_normals=self.chart_normals.to(device),
            chart_tangent1=self.chart_tangent1.to(device),
            chart_tangent2=self.chart_tangent2.to(device),
            chart_radii=self.chart_radii.to(device),
            overlap_sigma=self.overlap_sigma,
            num_charts=self.num_charts,
        )


def _compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """Compute area-weighted vertex normals from mesh."""
    V = vertices.shape[0]
    vertex_normals = np.zeros((V, 3), dtype=np.float64)
    
    for face in faces:
        i, j, k = face
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        
        # Face normal (area-weighted)
        e1 = v1 - v0
        e2 = v2 - v0
        face_normal = np.cross(e1, e2)  # Area-weighted (length = 2 * area)
        
        # Add to vertices
        vertex_normals[i] += face_normal
        vertex_normals[j] += face_normal
        vertex_normals[k] += face_normal
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norms + 1e-10)
    
    return vertex_normals
