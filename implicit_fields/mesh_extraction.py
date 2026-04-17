"""
Mesh extraction from neural implicit fields using marching cubes.

This module provides progressive marching cubes mesh extraction with
post-processing (Laplacian smoothing) and export utilities.

Features:
- Progressive resolution refinement (64³ → 128³ → 256³)
- Stateful MeshExtractor class for latent-conditioned models
- Laplacian and Taubin smoothing
- Multiple export formats (OBJ, PLY)
- Mesh quality metrics

Example:
    >>> from implicit_fields import extract_mesh, export_mesh_obj
    >>> def sdf_fn(coords):
    ...     return model(coords)  # Your trained SDF network
    >>> vertices, faces = extract_mesh(sdf_fn, bounds=(min_corner, max_corner))
    >>> export_mesh_obj(vertices, faces, "output.obj")
    
    # For latent-conditioned models:
    >>> from implicit_fields import MeshExtractor, MeshExtractionConfig
    >>> config = MeshExtractionConfig(resolution=128)
    >>> extractor = MeshExtractor(model, config=config)
    >>> vertices, faces, normals = extractor.extract_mesh(latent_code)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@dataclass
class MeshExtractionConfig:
    """Configuration for mesh extraction from neural implicit fields.
    
    This config is used by both the functional `extract_mesh()` API and
    the stateful `MeshExtractor` class.
    
    Attributes:
        resolution: Grid resolution for marching cubes. Default: 128.
        bounds: Bounding box as (min, max) tuple. Default: (-1.5, 1.5).
        iso_level: Isosurface threshold (0.0 for SDF surface). Default: 0.0.
        batch_size: Points per batch for inference. Default: 65536.
        smooth_iterations: Laplacian smoothing iterations. Default: 0.
        smooth_lambda: Smoothing strength (0-1). Default: 0.5.
        simplify_ratio: Mesh simplification ratio (1.0 = no simplification). Default: 1.0.
        compute_normals: Whether to compute vertex normals. Default: True.
        progressive_resolutions: Resolutions for progressive extraction. Default: (64, 128, 256).
        device: Compute device. Default: 'cuda'.
    
    Example:
        >>> config = MeshExtractionConfig(resolution=256, smooth_iterations=3)
        >>> extractor = MeshExtractor(model, config=config)
    """
    resolution: int = 128
    bounds: Tuple[float, float] = (-1.5, 1.5)
    iso_level: float = 0.0
    batch_size: int = 65536
    smooth_iterations: int = 0
    smooth_lambda: float = 0.5
    simplify_ratio: float = 1.0
    compute_normals: bool = True
    progressive_resolutions: Tuple[int, ...] = (64, 128, 256)
    device: str = 'cuda'


class MeshExtractor:
    """
    Stateful mesh extractor for latent-conditioned neural implicit models.
    
    This class wraps a trained NIF model and provides mesh extraction
    with support for:
    - Latent-conditioned SDF evaluation
    - Batched inference for memory efficiency
    - Progressive multi-resolution extraction
    - Post-processing (smoothing, simplification)
    
    Ported from Project 04 (ClothGeom-NIF) for shared use.
    
    Args:
        model: Trained neural implicit model. Must have interface:
               `model(coords, latent) -> (sdf, variance)` or `model(coords, latent) -> sdf`
        device: Torch device for inference.
        config: Extraction configuration.
    
    Example:
        >>> from implicit_fields import MeshExtractor, MeshExtractionConfig
        >>> 
        >>> config = MeshExtractionConfig(resolution=128)
        >>> extractor = MeshExtractor(model, config=config)
        >>> 
        >>> # Extract mesh from latent code
        >>> latent = torch.randn(128)
        >>> vertices, faces, normals = extractor.extract_mesh(latent)
        >>> 
        >>> # Save the mesh
        >>> extractor.save_mesh(vertices, faces, "output.obj", normals)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[MeshExtractionConfig] = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or MeshExtractionConfig()
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_sdf_grid(
        self,
        latent: torch.Tensor,
        resolution: Optional[int] = None,
        time: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Evaluate SDF on a regular 3D grid.
        
        Args:
            latent: Latent code [latent_dim] or [1, latent_dim]
            resolution: Grid resolution (overrides config)
            time: Optional time value for temporal models [1] or [1, 1]
        
        Returns:
            sdf_volume: [R, R, R] numpy array of SDF values
        """
        resolution = resolution or self.config.resolution
        b = self.config.bounds[1]
        
        # Create grid coordinates
        coords_1d = np.linspace(-b, b, resolution)
        xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid_points).float().to(self.device)
        
        # Ensure latent is on device and properly shaped
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent = latent.to(self.device)
        
        # Handle time parameter
        if time is not None:
            if time.dim() == 0:
                time = time.unsqueeze(0).unsqueeze(0)
            elif time.dim() == 1:
                time = time.unsqueeze(0)
            time = time.to(self.device)
        
        # Batched inference
        sdf_values = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i+batch_size]
            batch_latent = latent.expand(batch.shape[0], -1)
            
            # Handle different model interfaces
            if time is not None:
                batch_time = time.expand(batch.shape[0], -1)
                result = self.model(batch, batch_latent, batch_time)
            else:
                result = self.model(batch, batch_latent)
            
            # Handle tuple return (sdf, variance) or just sdf
            if isinstance(result, tuple):
                sdf = result[0]
            else:
                sdf = result
            
            if sdf.dim() > 1:
                sdf = sdf.squeeze(-1)
            sdf_values.append(sdf.cpu().numpy())
        
        sdf_volume = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
        return sdf_volume.astype(np.float32)
    
    def extract_mesh(
        self,
        latent: torch.Tensor,
        resolution: Optional[int] = None,
        iso_level: Optional[float] = None,
        time: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract mesh from latent code using marching cubes.
        
        Args:
            latent: Latent code [latent_dim] or [1, latent_dim]
            resolution: Grid resolution (overrides config)
            iso_level: Isosurface level (overrides config)
            time: Optional time value for temporal models
        
        Returns:
            vertices: [V, 3] vertex positions
            faces: [F, 3] face indices
            normals: [V, 3] vertex normals (if compute_normals=True)
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for mesh extraction. "
                            "Install with: pip install scikit-image")
        
        resolution = resolution or self.config.resolution
        iso_level = iso_level if iso_level is not None else self.config.iso_level
        
        logger.info(f"Extracting mesh at resolution {resolution}...")
        
        # Evaluate SDF on grid
        sdf_volume = self.evaluate_sdf_grid(latent, resolution, time)
        
        # Check for valid surface
        if sdf_volume.min() >= 0 or sdf_volume.max() <= 0:
            logger.warning("No zero crossing found in SDF volume. Surface may not exist.")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))
        
        # Pad volume to avoid boundary artifacts
        sdf_padded = np.pad(sdf_volume, pad_width=1, mode='constant', constant_values=1.0)
        
        # Run marching cubes
        try:
            vertices, faces, normals, _ = marching_cubes(
                sdf_padded,
                level=iso_level,
                spacing=(1.0, 1.0, 1.0),
                gradient_direction='descent'
            )
        except Exception as e:
            logger.error(f"Marching cubes failed: {e}")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))
        
        # Adjust for padding and rescale to world coordinates
        b = self.config.bounds[1]
        vertices = vertices - 1  # Remove padding offset
        vertices = vertices / (resolution - 1) * 2 * b - b  # Scale to [-b, b]
        
        logger.info(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Apply post-processing
        if self.config.smooth_iterations > 0:
            vertices = laplacian_smooth(
                vertices, faces,
                iterations=self.config.smooth_iterations,
                lam=self.config.smooth_lambda,
            )
            # Recompute normals after smoothing
            normals = self._compute_vertex_normals(vertices, faces)
        
        if self.config.simplify_ratio < 1.0 and TRIMESH_AVAILABLE:
            vertices, faces, normals = self._simplify_mesh(vertices, faces, normals)
        
        return vertices, faces, normals
    
    def extract_mesh_progressive(
        self,
        latent: torch.Tensor,
        resolutions: Optional[List[int]] = None,
        time: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Progressive mesh extraction with increasing resolution.
        
        Starts at coarse resolution and refines near the surface,
        reducing computation time while maintaining quality.
        
        Args:
            latent: Latent code
            resolutions: List of resolutions (default from config)
            time: Optional time value for temporal models
        
        Returns:
            vertices, faces, normals at highest resolution
        """
        resolutions = resolutions or list(self.config.progressive_resolutions)
        
        for i, res in enumerate(resolutions):
            logger.info(f"Progressive extraction: resolution {res}³ ({i+1}/{len(resolutions)})")
            vertices, faces, normals = self.extract_mesh(
                latent, resolution=res, time=time
            )
            
            if len(vertices) == 0:
                logger.warning(f"No mesh at resolution {res}, trying next...")
                continue
        
        return vertices, faces, normals
    
    def extract_sequence(
        self,
        latent: torch.Tensor,
        times: torch.Tensor,
        resolution: Optional[int] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract mesh sequence for animation.
        
        Args:
            latent: Latent code [latent_dim]
            times: [T] time values
            resolution: Grid resolution
        
        Returns:
            List of dicts with 'vertices', 'faces', 'normals' for each frame
        """
        meshes = []
        
        for i, t in enumerate(times):
            logger.info(f"Extracting frame {i+1}/{len(times)}")
            t_tensor = t.unsqueeze(0) if t.dim() == 0 else t
            vertices, faces, normals = self.extract_mesh(
                latent, resolution=resolution, time=t_tensor
            )
            meshes.append({
                'time': float(t),
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
            })
        
        return meshes
    
    def _compute_vertex_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """Compute vertex normals from face normals."""
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
        
        # Accumulate face normals to vertices
        vertex_normals = np.zeros_like(vertices)
        for i, face in enumerate(faces):
            vertex_normals[face[0]] += face_normals[i]
            vertex_normals[face[1]] += face_normals[i]
            vertex_normals[face[2]] += face_normals[i]
        
        # Normalize
        vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-10)
        return vertex_normals
    
    def _simplify_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simplify mesh using quadric decimation."""
        if not TRIMESH_AVAILABLE:
            return vertices, faces, normals
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        target_faces = int(len(faces) * self.config.simplify_ratio)
        
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
            return mesh.vertices, mesh.faces, mesh.vertex_normals
        except Exception as e:
            logger.warning(f"Mesh simplification failed: {e}")
            return vertices, faces, normals
    
    @staticmethod
    def save_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: str,
        normals: Optional[np.ndarray] = None
    ):
        """
        Save mesh to file (OBJ, PLY, STL).
        
        Args:
            vertices: [V, 3] vertex positions
            faces: [F, 3] face indices
            output_path: Output file path
            normals: [V, 3] vertex normals (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if len(vertices) == 0:
            logger.warning(f"Empty mesh, not saving to {output_path}")
            return
        
        suffix = output_path.suffix.lower()
        
        if TRIMESH_AVAILABLE:
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals
            )
            mesh.export(str(output_path))
        elif suffix == '.obj':
            export_mesh_obj(vertices, faces, str(output_path))
        elif suffix == '.ply':
            export_mesh_ply(vertices, faces, str(output_path))
        else:
            raise ValueError(f"Unsupported format {suffix}. Install trimesh for more formats.")
        
        logger.info(f"Saved mesh to: {output_path}")


def extract_mesh(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]],
    resolutions: List[int] = [64, 128, 256],
    threshold: float = 0.0,
    batch_size: int = 32768,
    device: str = 'cuda',
    smoothing_iterations: int = 3,
    smoothing_lambda: float = 0.5,
    progressive: bool = True,
    padding: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mesh from SDF network using progressive marching cubes.
    
    Progressively refines mesh extraction from coarse to fine resolution,
    focusing computation on regions near the surface.
    
    Args:
        sdf_fn: Function mapping (N, 3) coordinates to (N, 1) or (N,) SDF values.
                Should handle batched inputs on the specified device.
        bounds: Tuple of (min_corner, max_corner) defining the extraction volume.
                Each is a 3D vector (tensor or array).
        resolutions: List of resolutions for progressive refinement.
                    Default: [64, 128, 256] (64³ → 128³ → 256³).
        threshold: Isosurface threshold (0.0 for SDF zero-crossing). Default: 0.0.
        batch_size: Points per batch for memory efficiency. Default: 32768.
        device: Device for computation ('cuda' or 'cpu'). Default: 'cuda'.
        smoothing_iterations: Laplacian smoothing passes. Default: 3.
        smoothing_lambda: Smoothing strength (0-1). Default: 0.5.
        progressive: If True, use progressive refinement. If False, use only
                    the highest resolution. Default: True.
        padding: Fractional padding for bounds refinement. Default: 0.1 (10%).
    
    Returns:
        Tuple of (vertices, faces):
            - vertices: (V, 3) float array of vertex positions
            - faces: (F, 3) int array of face vertex indices
    
    Example:
        >>> def sdf_fn(coords):
        ...     with torch.no_grad():
        ...         return model(coords)
        >>> 
        >>> bounds = (torch.tensor([-1, -1, -1]), torch.tensor([1, 1, 1]))
        >>> verts, faces = extract_mesh(sdf_fn, bounds, resolutions=[64, 128])
        >>> print(f"Extracted mesh: {len(verts)} vertices, {len(faces)} faces")
    
    Note:
        Requires scikit-image for marching cubes. Install with:
        pip install scikit-image
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for mesh extraction. "
            "Install with: pip install scikit-image"
        )
    
    # Convert bounds to numpy
    min_corner = _to_numpy(bounds[0])
    max_corner = _to_numpy(bounds[1])
    
    if not progressive:
        # Use only the highest resolution
        resolutions = [resolutions[-1]]
    
    current_min = min_corner.copy()
    current_max = max_corner.copy()
    vertices, faces = None, None
    
    for i, resolution in enumerate(resolutions):
        # Create 3D grid
        grid_points, grid_shape, spacing = _create_grid(
            current_min, current_max, resolution
        )
        
        # Evaluate SDF in batches
        sdf_values = _evaluate_sdf_batched(
            sdf_fn, grid_points, batch_size, device
        )
        
        # Reshape to 3D volume
        sdf_volume = sdf_values.reshape(grid_shape)
        
        # Run marching cubes
        try:
            verts, faces_mc, normals, values = marching_cubes(
                sdf_volume,
                level=threshold,
                spacing=spacing,
            )
        except ValueError as e:
            # No surface found (all inside or outside)
            if "Surface level" in str(e) or "No surface" in str(e):
                print(f"Warning: No surface found at resolution {resolution}")
                continue
            raise
        
        # Transform vertices to world coordinates
        vertices = verts + current_min
        faces = faces_mc
        
        # For progressive refinement, narrow bounds to surface region
        if i < len(resolutions) - 1 and len(vertices) > 0:
            # Find tight bounding box of surface
            surface_min = vertices.min(axis=0)
            surface_max = vertices.max(axis=0)
            
            # Add padding
            extent = surface_max - surface_min
            pad = padding * extent
            current_min = np.maximum(min_corner, surface_min - pad)
            current_max = np.minimum(max_corner, surface_max + pad)
    
    if vertices is None or faces is None:
        raise RuntimeError("Failed to extract mesh at any resolution")
    
    # Apply Laplacian smoothing
    if smoothing_iterations > 0 and len(vertices) > 0:
        vertices = laplacian_smooth(
            vertices, faces,
            iterations=smoothing_iterations,
            lam=smoothing_lambda,
        )
    
    return vertices, faces


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _create_grid(
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    resolution: int,
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[float, float, float]]:
    """Create a regular 3D grid of query points.
    
    Returns:
        Tuple of (grid_points, grid_shape, spacing)
    """
    # Create 1D arrays for each axis
    x = np.linspace(min_corner[0], max_corner[0], resolution)
    y = np.linspace(min_corner[1], max_corner[1], resolution)
    z = np.linspace(min_corner[2], max_corner[2], resolution)
    
    # Compute spacing
    spacing = (
        (max_corner[0] - min_corner[0]) / (resolution - 1),
        (max_corner[1] - min_corner[1]) / (resolution - 1),
        (max_corner[2] - min_corner[2]) / (resolution - 1),
    )
    
    # Create meshgrid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten to (N, 3) points
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    
    grid_shape = (resolution, resolution, resolution)
    
    return grid_points, grid_shape, spacing


def _evaluate_sdf_batched(
    sdf_fn: Callable,
    points: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Evaluate SDF function in batches for memory efficiency."""
    n_points = len(points)
    sdf_values = np.zeros(n_points, dtype=np.float32)
    
    # Check if CUDA is available when requested
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = torch.from_numpy(points[start:end]).float().to(device)
            
            sdf_batch = sdf_fn(batch)
            
            # Handle both (N, 1) and (N,) outputs
            if sdf_batch.dim() > 1:
                sdf_batch = sdf_batch.squeeze(-1)
            
            sdf_values[start:end] = sdf_batch.cpu().numpy()
    
    return sdf_values


def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    lam: float = 0.5,
    preserve_boundary: bool = True,
) -> np.ndarray:
    """Apply Laplacian smoothing to a mesh.
    
    Iteratively moves each vertex toward the average of its neighbors,
    reducing noise while potentially shrinking the mesh.
    
    Args:
        vertices: (V, 3) array of vertex positions.
        faces: (F, 3) array of face indices.
        iterations: Number of smoothing iterations. Default: 3.
        lam: Smoothing factor per iteration (0 = no smoothing, 1 = full average).
             Default: 0.5.
        preserve_boundary: If True, don't smooth boundary vertices. Default: True.
    
    Returns:
        Smoothed vertices array of shape (V, 3).
    
    Example:
        >>> verts_smooth = laplacian_smooth(vertices, faces, iterations=5, lam=0.3)
    
    Note:
        For volume-preserving smoothing, consider using Taubin smoothing
        (alternating positive and negative lambda values).
    """
    vertices = vertices.copy()
    n_vertices = len(vertices)
    
    # Build adjacency list
    adjacency = _build_adjacency(faces, n_vertices)
    
    # Find boundary vertices if preserving
    if preserve_boundary:
        boundary = _find_boundary_vertices(faces, n_vertices)
    else:
        boundary = set()
    
    # Iterative smoothing
    for _ in range(iterations):
        new_vertices = vertices.copy()
        
        for i in range(n_vertices):
            if i in boundary:
                continue
            
            neighbors = adjacency[i]
            if len(neighbors) == 0:
                continue
            
            # Compute average of neighbor positions
            neighbor_avg = vertices[list(neighbors)].mean(axis=0)
            
            # Move toward average
            new_vertices[i] = (1 - lam) * vertices[i] + lam * neighbor_avg
        
        vertices = new_vertices
    
    return vertices


def taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    lam: float = 0.5,
    mu: float = -0.53,
    preserve_boundary: bool = True,
) -> np.ndarray:
    """Apply Taubin smoothing (volume-preserving Laplacian smoothing).
    
    Alternates between positive lambda (shrinking) and negative mu (expanding)
    to smooth while preserving volume better than standard Laplacian.
    
    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        iterations: Number of smoothing iterations. Default: 3.
        lam: Positive smoothing factor. Default: 0.5.
        mu: Negative expansion factor. Should satisfy mu < -lam. Default: -0.53.
        preserve_boundary: Don't smooth boundary vertices. Default: True.
    
    Returns:
        Smoothed vertices.
    """
    vertices = vertices.copy()
    n_vertices = len(vertices)
    
    adjacency = _build_adjacency(faces, n_vertices)
    
    if preserve_boundary:
        boundary = _find_boundary_vertices(faces, n_vertices)
    else:
        boundary = set()
    
    for _ in range(iterations):
        # Shrinking step (positive lambda)
        vertices = _smooth_step(vertices, adjacency, boundary, lam)
        # Expanding step (negative mu, i.e., move away from neighbors)
        vertices = _smooth_step(vertices, adjacency, boundary, mu)
    
    return vertices


def _smooth_step(
    vertices: np.ndarray,
    adjacency: List[set],
    boundary: set,
    factor: float,
) -> np.ndarray:
    """Single Laplacian smoothing step."""
    new_vertices = vertices.copy()
    
    for i in range(len(vertices)):
        if i in boundary:
            continue
        
        neighbors = adjacency[i]
        if len(neighbors) == 0:
            continue
        
        neighbor_avg = vertices[list(neighbors)].mean(axis=0)
        new_vertices[i] = vertices[i] + factor * (neighbor_avg - vertices[i])
    
    return new_vertices


def _build_adjacency(faces: np.ndarray, n_vertices: int) -> List[set]:
    """Build vertex adjacency list from faces."""
    adjacency = [set() for _ in range(n_vertices)]
    
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    return adjacency


def _find_boundary_vertices(faces: np.ndarray, n_vertices: int) -> set:
    """Find vertices on mesh boundary (edges with only one adjacent face)."""
    edge_count = {}
    
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary = set()
    for (v1, v2), count in edge_count.items():
        if count == 1:  # Boundary edge
            boundary.add(v1)
            boundary.add(v2)
    
    return boundary


def export_mesh_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str,
    vertex_colors: Optional[np.ndarray] = None,
) -> None:
    """Export mesh to OBJ file format.
    
    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices (0-indexed, will be converted to 1-indexed).
        filepath: Output file path.
        vertex_colors: Optional (V, 3) RGB colors in [0, 1].
    
    Example:
        >>> export_mesh_obj(vertices, faces, "output.obj")
    """
    with open(filepath, 'w') as f:
        f.write("# OBJ file exported by implicit_fields\n")
        
        # Write vertices
        for i, v in enumerate(vertices):
            if vertex_colors is not None:
                c = vertex_colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-indexed vertices)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def export_mesh_ply(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str,
    vertex_colors: Optional[np.ndarray] = None,
    binary: bool = True,
) -> None:
    """Export mesh to PLY file format.
    
    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        filepath: Output file path.
        vertex_colors: Optional (V, 3) RGB colors in [0, 1].
        binary: If True, write binary PLY (smaller, faster). Default: True.
    
    Example:
        >>> export_mesh_ply(vertices, faces, "output.ply")
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    has_colors = vertex_colors is not None
    
    header = f"""ply
format {'binary_little_endian' if binary else 'ascii'} 1.0
element vertex {n_vertices}
property float x
property float y
property float z
"""
    if has_colors:
        header += """property uchar red
property uchar green
property uchar blue
"""
    header += f"""element face {n_faces}
property list uchar int vertex_indices
end_header
"""
    
    if binary:
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            
            # Write vertices
            for i, v in enumerate(vertices):
                f.write(np.array(v, dtype='<f4').tobytes())
                if has_colors:
                    c = (vertex_colors[i] * 255).astype(np.uint8)
                    f.write(c.tobytes())
            
            # Write faces
            for face in faces:
                f.write(np.uint8(3).tobytes())
                f.write(np.array(face, dtype='<i4').tobytes())
    else:
        with open(filepath, 'w') as f:
            f.write(header)
            
            for i, v in enumerate(vertices):
                line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                if has_colors:
                    c = (vertex_colors[i] * 255).astype(int)
                    line += f" {c[0]} {c[1]} {c[2]}"
                f.write(line + "\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def compute_mesh_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict:
    """Compute mesh quality metrics.
    
    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
    
    Returns:
        Dictionary with quality metrics:
            - vertex_count: Number of vertices
            - face_count: Number of faces
            - edge_count: Number of unique edges
            - min_edge_length: Minimum edge length
            - max_edge_length: Maximum edge length
            - mean_edge_length: Mean edge length
            - watertight: Whether mesh is watertight (closed)
            - genus: Topological genus (for closed meshes)
            - self_intersections: Number of self-intersecting faces (approximate)
    
    Example:
        >>> quality = compute_mesh_quality(vertices, faces)
        >>> print(f"Watertight: {quality['watertight']}")
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    # Compute edges and edge lengths
    edges = set()
    edge_lengths = []
    
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edges:
                edges.add(edge)
                length = np.linalg.norm(vertices[v1] - vertices[v2])
                edge_lengths.append(length)
    
    edge_lengths = np.array(edge_lengths)
    n_edges = len(edges)
    
    # Check watertightness: every edge should have exactly 2 adjacent faces
    edge_face_count = {}
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
    
    boundary_edges = sum(1 for count in edge_face_count.values() if count != 2)
    watertight = boundary_edges == 0
    
    # Euler characteristic: V - E + F = 2 - 2g for closed surfaces
    euler = n_vertices - n_edges + n_faces
    genus = (2 - euler) // 2 if watertight else -1
    
    return {
        'vertex_count': n_vertices,
        'face_count': n_faces,
        'edge_count': n_edges,
        'min_edge_length': float(edge_lengths.min()) if len(edge_lengths) > 0 else 0.0,
        'max_edge_length': float(edge_lengths.max()) if len(edge_lengths) > 0 else 0.0,
        'mean_edge_length': float(edge_lengths.mean()) if len(edge_lengths) > 0 else 0.0,
        'watertight': watertight,
        'boundary_edges': boundary_edges,
        'genus': genus,
        'euler_characteristic': euler,
    }


def extract_mesh_mcubes(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]],
    resolution: int = 128,
    threshold: float = 0.0,
    batch_size: int = 32768,
    device: str = 'cuda',
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mesh using the mcubes library (alternative backend).
    
    Some projects (like NIF-Cloth4D) use the mcubes library instead of
    scikit-image. This function provides that alternative.
    
    Args:
        sdf_fn: SDF function.
        bounds: (min_corner, max_corner) tuple.
        resolution: Grid resolution. Default: 128.
        threshold: Isosurface level. Default: 0.0.
        batch_size: Batch size for SDF evaluation. Default: 32768.
        device: Computation device. Default: 'cuda'.
    
    Returns:
        Tuple of (vertices, faces).
    
    Note:
        Requires mcubes library. Install with: pip install PyMCubes
    """
    try:
        import mcubes
    except ImportError:
        raise ImportError(
            "mcubes library is required for this function. "
            "Install with: pip install PyMCubes"
        )
    
    min_corner = _to_numpy(bounds[0])
    max_corner = _to_numpy(bounds[1])
    
    # Create grid and evaluate SDF
    grid_points, grid_shape, _ = _create_grid(min_corner, max_corner, resolution)
    sdf_values = _evaluate_sdf_batched(sdf_fn, grid_points, batch_size, device)
    sdf_volume = sdf_values.reshape(grid_shape)
    
    # Run mcubes
    vertices, faces = mcubes.marching_cubes(sdf_volume, threshold)
    
    # Scale vertices to world coordinates
    scale = (max_corner - min_corner) / (resolution - 1)
    vertices = vertices * scale + min_corner
    
    return vertices, faces
