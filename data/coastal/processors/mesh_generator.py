"""
Coastal mesh generation from bathymetry and shoreline data.

Creates unstructured triangular meshes suitable for finite element
methods and graph neural networks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

from ..config import BoundingBox, RegionConfig
from ..datatypes import (
    BathymetryGrid,
    CoastalMesh,
    Shoreline,
)
from .coordinate_utils import (
    latlon_to_utm,
    utm_to_latlon,
    compute_distance_to_shore,
    great_circle_distance,
)

logger = logging.getLogger(__name__)


@dataclass
class MeshParameters:
    """Parameters controlling mesh generation."""
    
    # Resolution control (in meters)
    min_edge_length: float = 100.0
    max_edge_length: float = 5000.0
    
    # Grading based on depth
    depth_grading_factor: float = 0.1  # Finer mesh in shallow water
    
    # Grading based on distance to shore
    shore_grading_factor: float = 0.05  # Finer mesh near shore
    shore_grading_distance: float = 10000.0  # Distance over which shore grading applies
    
    # Quality constraints
    min_angle: float = 20.0  # Minimum triangle angle in degrees
    max_aspect_ratio: float = 3.0  # Maximum triangle aspect ratio
    
    # Boundary handling
    boundary_refinement: float = 0.5  # Refinement factor along boundaries
    
    # Simplification
    simplify_shoreline: bool = True
    simplify_tolerance: float = 50.0  # Meters


@dataclass 
class TriangulationResult:
    """Result of Delaunay triangulation."""
    
    vertices: np.ndarray  # (V, 2) vertex coordinates
    triangles: np.ndarray  # (T, 3) triangle vertex indices
    boundary_edges: np.ndarray  # (E, 2) boundary edge vertex indices
    boundary_markers: np.ndarray  # (V,) boundary type for each vertex


class CoastalMeshGenerator:
    """
    Generate computational meshes for coastal regions.
    
    Creates unstructured triangular meshes with:
    - Depth-dependent resolution grading
    - Shore-distance resolution grading
    - Boundary classification
    - Quality constraints
    
    Example:
        generator = CoastalMeshGenerator(params)
        mesh = generator.generate(bathymetry, shoreline, region)
    """
    
    def __init__(
        self,
        params: Optional[MeshParameters] = None,
    ):
        """
        Initialize mesh generator.
        
        Args:
            params: Mesh generation parameters (uses defaults if None)
        """
        self.params = params or MeshParameters()
        self._scipy_available = False
        self._triangle_available = False
        
        # Check for optional dependencies
        try:
            from scipy.spatial import Delaunay
            self._scipy_available = True
        except ImportError:
            logger.warning("scipy not available - using simple triangulation")
        
        try:
            import triangle
            self._triangle_available = True
        except ImportError:
            pass  # triangle library not available - fall back to scipy
    
    def generate(
        self,
        bathymetry: BathymetryGrid,
        shoreline: Optional[Shoreline] = None,
        region: Optional[RegionConfig] = None,
    ) -> CoastalMesh:
        """
        Generate a triangular mesh for the coastal region.
        
        Args:
            bathymetry: Bathymetry grid defining the domain
            shoreline: Optional shoreline for boundary constraints
            region: Optional region config for additional parameters
            
        Returns:
            CoastalMesh with vertices, triangles, and attributes
        """
        logger.info("Generating coastal mesh...")
        
        # 1. Generate initial vertex positions
        logger.info("  Creating initial vertices...")
        vertices_utm, depths, zone, hemisphere = self._create_initial_vertices(
            bathymetry
        )
        
        # 2. Compute desired edge lengths based on depth and shore distance
        logger.info("  Computing edge length field...")
        edge_lengths = self._compute_edge_length_field(
            vertices_utm, depths, shoreline
        )
        
        # 3. Remove vertices that are too close together
        logger.info("  Filtering vertices by spacing...")
        vertices_utm, depths, edge_lengths = self._filter_by_spacing(
            vertices_utm, depths, edge_lengths
        )
        
        # 4. Add boundary vertices from shoreline
        if shoreline is not None and shoreline.segments:
            logger.info("  Adding shoreline boundary vertices...")
            vertices_utm, depths, edge_lengths, boundary_markers = self._add_shoreline_vertices(
                vertices_utm, depths, edge_lengths, shoreline, zone, hemisphere
            )
        else:
            boundary_markers = np.zeros(len(vertices_utm), dtype=np.int32)
        
        # 5. Perform triangulation
        logger.info("  Performing Delaunay triangulation...")
        triangles = self._triangulate(vertices_utm)
        
        # 6. Remove triangles outside domain or over land
        logger.info("  Filtering triangles...")
        triangles = self._filter_triangles(
            vertices_utm, triangles, depths, shoreline
        )
        
        # 7. Build edge connectivity
        logger.info("  Building edge connectivity...")
        edges = self._build_edges(triangles)
        
        # 8. Identify boundary edges
        boundary_edges = self._identify_boundary_edges(triangles)
        
        # 9. Convert back to lat/lon
        logger.info("  Converting to lat/lon...")
        vertices_lat, vertices_lon = utm_to_latlon(
            vertices_utm[:, 0], vertices_utm[:, 1], zone, hemisphere
        )
        
        # 10. Classify boundary types
        logger.info("  Classifying boundaries...")
        boundary_types = self._classify_boundaries(
            vertices_lon, vertices_lat, boundary_markers, bathymetry.bbox
        )
        
        # 11. Compute additional node features
        logger.info("  Computing node features...")
        distance_to_shore = compute_distance_to_shore(
            vertices_lon, vertices_lat,
            shoreline.segments if shoreline else []
        )
        
        # Build the mesh object
        mesh = CoastalMesh(
            vertices=np.column_stack([vertices_lon, vertices_lat]),
            triangles=triangles,
            edges=edges,
            depths=depths,
            boundary_nodes=np.where(boundary_markers > 0)[0].astype(np.int64),
            boundary_types=boundary_types,
            distance_to_shore=distance_to_shore,
            utm_zone=zone,
            utm_hemisphere=hemisphere,
        )
        
        logger.info(f"  Generated mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, {mesh.n_edges} edges")
        
        return mesh
    
    def _create_initial_vertices(
        self,
        bathymetry: BathymetryGrid,
    ) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """
        Create initial vertex positions from bathymetry grid.
        
        Returns vertices in UTM coordinates, depths, and UTM zone info.
        """
        # Create meshgrid of points
        lon_grid, lat_grid = np.meshgrid(bathymetry.lon, bathymetry.lat)
        
        # Flatten
        lons = lon_grid.flatten()
        lats = lat_grid.flatten()
        depths = bathymetry.elevation.flatten()
        
        # Filter out NaN depths and land (positive elevation)
        valid_mask = ~np.isnan(depths) & (depths < 0)
        lons = lons[valid_mask]
        lats = lats[valid_mask]
        depths = -depths[valid_mask]  # Convert to positive depth
        
        # Convert to UTM
        x, y, zone, hemisphere = latlon_to_utm(lats, lons)
        vertices_utm = np.column_stack([x, y])
        
        return vertices_utm, depths, zone, hemisphere
    
    def _compute_edge_length_field(
        self,
        vertices_utm: np.ndarray,
        depths: np.ndarray,
        shoreline: Optional[Shoreline],
    ) -> np.ndarray:
        """
        Compute desired edge length at each vertex.
        
        Edge length varies based on:
        - Water depth (finer mesh in shallow water)
        - Distance to shore (finer mesh near coast)
        """
        n = len(vertices_utm)
        edge_lengths = np.full(n, self.params.max_edge_length)
        
        # Depth-based grading: smaller elements in shallow water
        if self.params.depth_grading_factor > 0:
            # Scale: 0m depth -> min_edge_length, deep water -> max_edge_length
            depth_factor = np.clip(depths / 100.0, 0, 1)  # Normalize by 100m
            depth_lengths = (
                self.params.min_edge_length + 
                depth_factor * (self.params.max_edge_length - self.params.min_edge_length)
            )
            edge_lengths = np.minimum(edge_lengths, depth_lengths)
        
        # Shore-distance grading
        if shoreline is not None and shoreline.segments and self.params.shore_grading_factor > 0:
            # Compute distance to shore for each vertex
            lats, lons = utm_to_latlon(
                vertices_utm[:, 0], vertices_utm[:, 1],
                zone=18, hemisphere='N'  # Will need proper zone
            )
            
            shore_distances = compute_distance_to_shore(lons, lats, shoreline.segments)
            
            # Scale: 0m distance -> min_edge_length, grading_distance -> no effect
            shore_factor = np.clip(
                shore_distances / self.params.shore_grading_distance, 
                0, 1
            )
            shore_lengths = (
                self.params.min_edge_length +
                shore_factor * (self.params.max_edge_length - self.params.min_edge_length)
            )
            edge_lengths = np.minimum(edge_lengths, shore_lengths)
        
        return edge_lengths
    
    def _filter_by_spacing(
        self,
        vertices: np.ndarray,
        depths: np.ndarray,
        edge_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove vertices that are too close together based on local edge length.
        
        Uses a simple greedy algorithm - could be improved with Poisson disk sampling.
        """
        n = len(vertices)
        if n < 10:
            return vertices, depths, edge_lengths
        
        # Sort by edge length (keep vertices with smaller desired spacing)
        order = np.argsort(edge_lengths)
        
        kept = np.ones(n, dtype=bool)
        
        # For each vertex, mark nearby vertices for removal
        for idx in order:
            if not kept[idx]:
                continue
            
            # Find vertices within half the edge length
            threshold = edge_lengths[idx] * 0.5
            distances = np.sqrt(np.sum((vertices - vertices[idx])**2, axis=1))
            
            # Mark close vertices for removal (except this one)
            too_close = (distances < threshold) & (distances > 0)
            kept[too_close] = False
        
        logger.info(f"    Kept {np.sum(kept)}/{n} vertices after spacing filter")
        
        return vertices[kept], depths[kept], edge_lengths[kept]
    
    def _add_shoreline_vertices(
        self,
        vertices_utm: np.ndarray,
        depths: np.ndarray,
        edge_lengths: np.ndarray,
        shoreline: Shoreline,
        zone: int,
        hemisphere: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Add vertices along shoreline boundary.
        
        Returns updated arrays plus boundary markers.
        """
        # Convert shoreline to UTM
        shore_vertices = []
        for segment in shoreline.segments:
            lons = segment[:, 0]
            lats = segment[:, 1]
            x, y, _, _ = latlon_to_utm(lats, lons, zone, hemisphere)
            shore_vertices.append(np.column_stack([x, y]))
        
        if not shore_vertices:
            boundary_markers = np.zeros(len(vertices_utm), dtype=np.int32)
            return vertices_utm, depths, edge_lengths, boundary_markers
        
        # Resample shoreline to desired spacing
        resampled_shore = []
        for segment in shore_vertices:
            if len(segment) < 2:
                continue
            
            # Compute cumulative distance along segment
            diffs = np.diff(segment, axis=0)
            seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = cumulative[-1]
            
            if total_length < self.params.min_edge_length:
                continue
            
            # Resample at regular intervals
            spacing = self.params.min_edge_length * self.params.boundary_refinement
            n_points = int(total_length / spacing) + 1
            new_distances = np.linspace(0, total_length, n_points)
            
            # Interpolate
            new_x = np.interp(new_distances, cumulative, segment[:, 0])
            new_y = np.interp(new_distances, cumulative, segment[:, 1])
            resampled_shore.append(np.column_stack([new_x, new_y]))
        
        if not resampled_shore:
            boundary_markers = np.zeros(len(vertices_utm), dtype=np.int32)
            return vertices_utm, depths, edge_lengths, boundary_markers
        
        shore_all = np.vstack(resampled_shore)
        
        # Combine with interior vertices
        n_interior = len(vertices_utm)
        n_shore = len(shore_all)
        
        all_vertices = np.vstack([vertices_utm, shore_all])
        all_depths = np.concatenate([depths, np.zeros(n_shore)])  # Zero depth at shore
        all_edge_lengths = np.concatenate([
            edge_lengths,
            np.full(n_shore, self.params.min_edge_length * self.params.boundary_refinement)
        ])
        
        # Boundary markers: 0 = interior, 1 = shoreline
        boundary_markers = np.zeros(n_interior + n_shore, dtype=np.int32)
        boundary_markers[n_interior:] = 1
        
        return all_vertices, all_depths, all_edge_lengths, boundary_markers
    
    def _triangulate(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """
        Perform Delaunay triangulation.
        
        Uses scipy.spatial.Delaunay or triangle library if available,
        otherwise falls back to simple incremental algorithm.
        """
        if len(vertices) < 3:
            return np.array([], dtype=np.int64).reshape(0, 3)
        
        if self._scipy_available:
            from scipy.spatial import Delaunay
            try:
                tri = Delaunay(vertices)
                return tri.simplices.astype(np.int64)
            except Exception as e:
                logger.warning(f"Delaunay triangulation failed: {e}")
                return self._simple_triangulation(vertices)
        else:
            return self._simple_triangulation(vertices)
    
    def _simple_triangulation(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """
        Simple grid-based triangulation fallback.
        
        Not a proper Delaunay triangulation, but works for simple cases.
        """
        n = len(vertices)
        if n < 3:
            return np.array([], dtype=np.int64).reshape(0, 3)
        
        # Find bounding box and create regular grid
        xmin, ymin = vertices.min(axis=0)
        xmax, ymax = vertices.max(axis=0)
        
        # Simple approach: find nearest neighbors and form triangles
        # This is not ideal but works as a fallback
        
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        
        triangles = []
        used_edges = set()
        
        for i in range(n):
            # Get k nearest neighbors
            distances, indices = tree.query(vertices[i], k=min(7, n))
            neighbors = indices[1:]  # Exclude self
            
            # Try to form triangles with pairs of neighbors
            for j_idx, j in enumerate(neighbors):
                for k in neighbors[j_idx+1:]:
                    # Check if this triangle is valid (all edges reasonable length)
                    edge1 = tuple(sorted([i, j]))
                    edge2 = tuple(sorted([j, k]))
                    edge3 = tuple(sorted([i, k]))
                    
                    tri_key = tuple(sorted([i, j, k]))
                    if tri_key not in used_edges:
                        triangles.append([i, j, k])
                        used_edges.add(tri_key)
        
        if not triangles:
            return np.array([], dtype=np.int64).reshape(0, 3)
        
        return np.array(triangles, dtype=np.int64)
    
    def _filter_triangles(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        depths: np.ndarray,
        shoreline: Optional[Shoreline],
    ) -> np.ndarray:
        """
        Remove triangles that are invalid (over land, outside domain, poor quality).
        """
        if len(triangles) == 0:
            return triangles
        
        n_original = len(triangles)
        keep_mask = np.ones(n_original, dtype=bool)
        
        # 1. Remove triangles where all vertices are on land (depth == 0)
        tri_depths = depths[triangles]
        all_land = np.all(tri_depths <= 0, axis=1)
        keep_mask &= ~all_land
        
        # 2. Remove triangles with very large edges (likely spanning gaps)
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        edge1_len = np.sqrt(np.sum((v1 - v0)**2, axis=1))
        edge2_len = np.sqrt(np.sum((v2 - v1)**2, axis=1))
        edge3_len = np.sqrt(np.sum((v0 - v2)**2, axis=1))
        
        max_edge = np.maximum(np.maximum(edge1_len, edge2_len), edge3_len)
        too_large = max_edge > self.params.max_edge_length * 2
        keep_mask &= ~too_large
        
        # 3. Remove degenerate triangles (very small area)
        # Area using cross product
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
                (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        area = np.abs(cross) / 2
        min_area = (self.params.min_edge_length ** 2) * 0.1
        degenerate = area < min_area
        keep_mask &= ~degenerate
        
        filtered = triangles[keep_mask]
        logger.info(f"    Kept {len(filtered)}/{n_original} triangles after filtering")
        
        return filtered
    
    def _build_edges(
        self,
        triangles: np.ndarray,
    ) -> np.ndarray:
        """
        Build edge list from triangles.
        
        Returns unique edges as (E, 2) array.
        """
        if len(triangles) == 0:
            return np.array([], dtype=np.int64).reshape(0, 2)
        
        # Extract all edges from triangles
        edges = []
        for i in range(3):
            j = (i + 1) % 3
            e = np.column_stack([
                triangles[:, i],
                triangles[:, j]
            ])
            edges.append(e)
        
        all_edges = np.vstack(edges)
        
        # Sort each edge so smaller index is first
        all_edges = np.sort(all_edges, axis=1)
        
        # Remove duplicates
        unique_edges = np.unique(all_edges, axis=0)
        
        return unique_edges.astype(np.int64)
    
    def _identify_boundary_edges(
        self,
        triangles: np.ndarray,
    ) -> np.ndarray:
        """
        Identify edges that are on the boundary (appear in only one triangle).
        """
        if len(triangles) == 0:
            return np.array([], dtype=np.int64).reshape(0, 2)
        
        # Count edge occurrences
        edge_count = {}
        
        for tri in triangles:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([tri[i], tri[j]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Boundary edges appear exactly once
        boundary = [edge for edge, count in edge_count.items() if count == 1]
        
        if not boundary:
            return np.array([], dtype=np.int64).reshape(0, 2)
        
        return np.array(boundary, dtype=np.int64)
    
    def _classify_boundaries(
        self,
        vertices_lon: np.ndarray,
        vertices_lat: np.ndarray,
        boundary_markers: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray:
        """
        Classify boundary nodes by type.
        
        Types:
            0: Interior
            1: Shoreline (land boundary)
            2: Ocean boundary (open water)
            3: River/inlet boundary
        """
        n = len(vertices_lon)
        boundary_types = np.zeros(n, dtype=np.int32)
        
        # Mark shoreline nodes
        boundary_types[boundary_markers == 1] = 1
        
        # Identify ocean boundaries (nodes near domain edge over water)
        edge_tolerance = 0.01  # degrees
        
        near_west = vertices_lon < bbox.lon_min + edge_tolerance
        near_east = vertices_lon > bbox.lon_max - edge_tolerance
        near_south = vertices_lat < bbox.lat_min + edge_tolerance
        near_north = vertices_lat > bbox.lat_max - edge_tolerance
        
        on_domain_edge = near_west | near_east | near_south | near_north
        
        # Ocean boundary: on domain edge but not already marked as shoreline
        ocean_boundary = on_domain_edge & (boundary_types == 0)
        boundary_types[ocean_boundary] = 2
        
        return boundary_types


def generate_mesh(
    bathymetry: BathymetryGrid,
    shoreline: Optional[Shoreline] = None,
    region: Optional[RegionConfig] = None,
    params: Optional[MeshParameters] = None,
) -> CoastalMesh:
    """
    Convenience function to generate a coastal mesh.
    
    Args:
        bathymetry: Bathymetry grid
        shoreline: Optional shoreline geometry
        region: Optional region configuration
        params: Optional mesh parameters
        
    Returns:
        CoastalMesh ready for use in simulations
    """
    generator = CoastalMeshGenerator(params)
    return generator.generate(bathymetry, shoreline, region)
