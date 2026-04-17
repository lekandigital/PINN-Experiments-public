"""
Core data types for the cloth simulation pipeline.

These dataclasses represent the intermediate format used between
ingest, transform, and export stages.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class CollisionBody:
    """
    A collision object (e.g., character body) from the simulation.
    
    Attributes:
        vertices: Vertex positions. Shape is either:
            - (num_frames, num_vertices, 3) for animated bodies
            - (num_vertices, 3) for static bodies
        faces: Triangle face indices. Shape: (num_faces, 3)
        is_animated: True if body moves across frames
        name: Optional identifier for the collision body
    """
    vertices: np.ndarray
    faces: np.ndarray
    is_animated: bool = False
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate array shapes."""
        if self.is_animated:
            assert self.vertices.ndim == 3, \
                f"Animated body vertices must be (F, N, 3), got {self.vertices.shape}"
        else:
            assert self.vertices.ndim == 2, \
                f"Static body vertices must be (N, 3), got {self.vertices.shape}"
        assert self.faces.ndim == 2 and self.faces.shape[1] == 3, \
            f"Faces must be (F, 3), got {self.faces.shape}"
    
    @property
    def num_vertices(self) -> int:
        """Number of vertices in the collision body."""
        if self.is_animated:
            return self.vertices.shape[1]
        return self.vertices.shape[0]
    
    @property
    def num_faces(self) -> int:
        """Number of faces in the collision body."""
        return self.faces.shape[0]
    
    @property
    def num_frames(self) -> int:
        """Number of frames (1 for static bodies)."""
        if self.is_animated:
            return self.vertices.shape[0]
        return 1
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get vertex positions at a specific frame."""
        if self.is_animated:
            return self.vertices[frame_idx]
        return self.vertices


@dataclass
class ClothSequence:
    """
    One cloth simulation sequence ready for transformation.
    
    This is the unified intermediate format produced by all ingesters
    (Alembic, OBJ sequence, Blender) and consumed by all transformers.
    
    Attributes:
        vertices: Vertex positions over time. Shape: (num_frames, num_vertices, 3)
        faces: Face connectivity (triangles or quads). Shape: (num_faces, 3 or 4)
        normals: Optional vertex normals. Shape: (num_frames, num_vertices, 3)
        uvs: Optional UV coordinates. Shape: (num_vertices, 2)
        velocities: Optional vertex velocities. Shape: (num_frames, num_vertices, 3)
        vertex_attributes: Optional per-vertex simulation attributes (stretch, bend, etc.)
        fps: Source simulation framerate
        dt: Timestep between frames in seconds
        frame_range: (start_frame, end_frame) tuple
        metadata: Source tool info, sim settings, material properties, etc.
        collision_bodies: Optional list of collision body meshes
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    vertex_attributes: Optional[Dict[str, np.ndarray]] = None
    fps: float = 24.0
    dt: float = 1.0 / 24.0
    frame_range: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    collision_bodies: Optional[List[CollisionBody]] = None
    
    def __post_init__(self):
        """Validate array shapes and compute derived values."""
        # Validate vertices
        assert self.vertices.ndim == 3, \
            f"Vertices must be (F, N, 3), got shape {self.vertices.shape}"
        assert self.vertices.shape[2] == 3, \
            f"Vertices must have 3 coordinates, got {self.vertices.shape[2]}"
        
        # Validate faces
        assert self.faces.ndim == 2, \
            f"Faces must be (F, 3 or 4), got shape {self.faces.shape}"
        assert self.faces.shape[1] in (3, 4), \
            f"Faces must be triangles (3) or quads (4), got {self.faces.shape[1]}"
        
        # Validate normals if present
        if self.normals is not None:
            assert self.normals.shape == self.vertices.shape, \
                f"Normals shape {self.normals.shape} must match vertices {self.vertices.shape}"
        
        # Validate UVs if present
        if self.uvs is not None:
            assert self.uvs.ndim == 2 and self.uvs.shape[1] == 2, \
                f"UVs must be (N, 2), got {self.uvs.shape}"
            assert self.uvs.shape[0] == self.num_vertices, \
                f"UVs vertex count {self.uvs.shape[0]} must match {self.num_vertices}"
        
        # Validate velocities if present
        if self.velocities is not None:
            assert self.velocities.shape == self.vertices.shape, \
                f"Velocities shape {self.velocities.shape} must match vertices {self.vertices.shape}"
        
        # Update frame_range if not set
        if self.frame_range == (0, 0):
            self.frame_range = (0, self.num_frames - 1)
        
        # Ensure dt matches fps
        if self.dt == 1.0 / 24.0 and self.fps != 24.0:
            self.dt = 1.0 / self.fps
    
    @property
    def num_frames(self) -> int:
        """Number of frames in the sequence."""
        return self.vertices.shape[0]
    
    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[1]
    
    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return self.faces.shape[0]
    
    @property
    def is_triangulated(self) -> bool:
        """True if mesh uses triangles (not quads)."""
        return self.faces.shape[1] == 3
    
    @property
    def rest_pose(self) -> np.ndarray:
        """Get the rest pose (first frame) vertex positions."""
        return self.vertices[0]
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return (self.num_frames - 1) * self.dt
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get vertex positions at a specific frame."""
        return self.vertices[frame_idx]
    
    def get_displacement(self, frame_idx: int) -> np.ndarray:
        """Get displacement from rest pose at a specific frame."""
        return self.vertices[frame_idx] - self.rest_pose
    
    def compute_velocities(self, method: str = "central") -> np.ndarray:
        """
        Compute vertex velocities using finite differences.
        
        Args:
            method: 'forward', 'backward', or 'central' differencing
            
        Returns:
            Velocities array of shape (num_frames, num_vertices, 3)
        """
        if self.velocities is not None:
            return self.velocities
        
        velocities = np.zeros_like(self.vertices)
        
        if method == "forward":
            velocities[:-1] = (self.vertices[1:] - self.vertices[:-1]) / self.dt
            velocities[-1] = velocities[-2]  # Copy last valid velocity
        elif method == "backward":
            velocities[1:] = (self.vertices[1:] - self.vertices[:-1]) / self.dt
            velocities[0] = velocities[1]  # Copy first valid velocity
        elif method == "central":
            velocities[1:-1] = (self.vertices[2:] - self.vertices[:-2]) / (2 * self.dt)
            velocities[0] = (self.vertices[1] - self.vertices[0]) / self.dt
            velocities[-1] = (self.vertices[-1] - self.vertices[-2]) / self.dt
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.velocities = velocities
        return velocities
    
    def compute_accelerations(self) -> np.ndarray:
        """
        Compute vertex accelerations using second-order finite differences.
        
        Returns:
            Accelerations array of shape (num_frames, num_vertices, 3)
        """
        if self.velocities is None:
            self.compute_velocities()
        
        accelerations = np.zeros_like(self.vertices)
        accelerations[1:-1] = (self.velocities[2:] - self.velocities[:-2]) / (2 * self.dt)
        accelerations[0] = (self.velocities[1] - self.velocities[0]) / self.dt
        accelerations[-1] = (self.velocities[-1] - self.velocities[-2]) / self.dt
        
        return accelerations
    
    def triangulate(self) -> 'ClothSequence':
        """
        Convert quads to triangles if necessary.
        
        Returns:
            New ClothSequence with triangulated faces (or self if already triangles)
        """
        if self.is_triangulated:
            return self
        
        # Convert quads to triangles: each quad (a,b,c,d) -> (a,b,c), (a,c,d)
        quads = self.faces
        triangles = np.zeros((quads.shape[0] * 2, 3), dtype=quads.dtype)
        triangles[0::2] = quads[:, [0, 1, 2]]
        triangles[1::2] = quads[:, [0, 2, 3]]
        
        return ClothSequence(
            vertices=self.vertices,
            faces=triangles,
            normals=self.normals,
            uvs=self.uvs,
            velocities=self.velocities,
            vertex_attributes=self.vertex_attributes,
            fps=self.fps,
            dt=self.dt,
            frame_range=self.frame_range,
            metadata=self.metadata,
            collision_bodies=self.collision_bodies,
        )
    
    def get_bounding_box(self, frame_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box.
        
        Args:
            frame_idx: Specific frame, or None for bbox over all frames
            
        Returns:
            (min_corner, max_corner) each of shape (3,)
        """
        if frame_idx is not None:
            verts = self.vertices[frame_idx]
        else:
            verts = self.vertices.reshape(-1, 3)
        
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_centroid(self, frame_idx: int = 0) -> np.ndarray:
        """Get mesh centroid at a specific frame."""
        return self.vertices[frame_idx].mean(axis=0)
    
    def center_at_origin(self) -> 'ClothSequence':
        """
        Center the sequence so rest pose centroid is at origin.
        
        Returns:
            New ClothSequence centered at origin
        """
        centroid = self.get_centroid(0)
        centered_vertices = self.vertices - centroid
        
        # Also center collision bodies if present
        centered_bodies = None
        if self.collision_bodies:
            centered_bodies = []
            for body in self.collision_bodies:
                centered_verts = body.vertices - centroid
                centered_bodies.append(CollisionBody(
                    vertices=centered_verts,
                    faces=body.faces,
                    is_animated=body.is_animated,
                    name=body.name,
                ))
        
        return ClothSequence(
            vertices=centered_vertices,
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            velocities=self.velocities.copy() if self.velocities is not None else None,
            vertex_attributes=self.vertex_attributes.copy() if self.vertex_attributes else None,
            fps=self.fps,
            dt=self.dt,
            frame_range=self.frame_range,
            metadata={**self.metadata, 'centered': True, 'original_centroid': centroid.tolist()},
            collision_bodies=centered_bodies,
        )
    
    def slice_frames(self, start: int, end: int) -> 'ClothSequence':
        """
        Extract a subsequence of frames.
        
        Args:
            start: Start frame index (inclusive)
            end: End frame index (exclusive)
            
        Returns:
            New ClothSequence with sliced frames
        """
        sliced_bodies = None
        if self.collision_bodies:
            sliced_bodies = []
            for body in self.collision_bodies:
                if body.is_animated:
                    sliced_verts = body.vertices[start:end]
                else:
                    sliced_verts = body.vertices
                sliced_bodies.append(CollisionBody(
                    vertices=sliced_verts,
                    faces=body.faces,
                    is_animated=body.is_animated,
                    name=body.name,
                ))
        
        return ClothSequence(
            vertices=self.vertices[start:end],
            faces=self.faces.copy(),
            normals=self.normals[start:end] if self.normals is not None else None,
            uvs=self.uvs,
            velocities=self.velocities[start:end] if self.velocities is not None else None,
            vertex_attributes=self.vertex_attributes,
            fps=self.fps,
            dt=self.dt,
            frame_range=(self.frame_range[0] + start, self.frame_range[0] + end - 1),
            metadata=self.metadata,
            collision_bodies=sliced_bodies,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'vertices': self.vertices,
            'faces': self.faces,
            'fps': self.fps,
            'dt': self.dt,
            'frame_range': self.frame_range,
            'metadata': self.metadata,
        }
        if self.normals is not None:
            data['normals'] = self.normals
        if self.uvs is not None:
            data['uvs'] = self.uvs
        if self.velocities is not None:
            data['velocities'] = self.velocities
        if self.vertex_attributes:
            data['vertex_attributes'] = self.vertex_attributes
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClothSequence':
        """Create from dictionary."""
        return cls(
            vertices=data['vertices'],
            faces=data['faces'],
            normals=data.get('normals'),
            uvs=data.get('uvs'),
            velocities=data.get('velocities'),
            vertex_attributes=data.get('vertex_attributes'),
            fps=data.get('fps', 24.0),
            dt=data.get('dt', 1.0/24.0),
            frame_range=tuple(data.get('frame_range', (0, 0))),
            metadata=data.get('metadata', {}),
        )


@dataclass 
class TransformedData:
    """
    Container for all transformed data ready for export.
    
    This holds the results of all transform operations and is passed
    to project-specific exporters.
    """
    # Original sequence reference
    sequence: ClothSequence
    
    # Graph data (from transform/graph.py)
    edge_index: Optional[np.ndarray] = None  # (2, num_edges)
    edge_features: Optional[np.ndarray] = None  # (num_edges, feature_dim)
    rest_edge_lengths: Optional[np.ndarray] = None  # (num_edges,)
    rest_edge_directions: Optional[np.ndarray] = None  # (num_edges, 3)
    dihedral_angles: Optional[np.ndarray] = None  # (num_edges,)
    
    # Hierarchical graph data
    coarse_vertices: Optional[np.ndarray] = None  # (num_frames, num_coarse, 3)
    coarse_edge_index: Optional[np.ndarray] = None  # (2, num_coarse_edges)
    cluster_assignments: Optional[np.ndarray] = None  # (num_vertices,)
    
    # SDF data (from transform/sdf.py)
    sdf_volumes: Optional[np.ndarray] = None  # (num_frames, res, res, res) or single frame
    sdf_samples: Optional[np.ndarray] = None  # (num_samples, 4) - (x,y,z,sdf)
    sdf_resolution: int = 64
    sdf_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (min, max)
    
    # Temporal data (from transform/temporal.py)
    velocities: Optional[np.ndarray] = None  # (num_frames, num_vertices, 3)
    accelerations: Optional[np.ndarray] = None  # (num_frames, num_vertices, 3)
    
    # Vertex features (from transform/features.py)
    curvatures: Optional[np.ndarray] = None  # (num_frames, num_vertices)
    computed_normals: Optional[np.ndarray] = None  # (num_frames, num_vertices, 3)
    
    # Normalization stats (from transform/normalization.py)
    normalization_stats: Optional[Dict[str, Any]] = None


@dataclass
class ExportMetadata:
    """Metadata about an exported dataset."""
    source_file: str
    source_format: str
    num_frames: int
    num_vertices: int
    num_faces: int
    fps: float
    export_timestamp: str
    pipeline_version: str
    config_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source_file': self.source_file,
            'source_format': self.source_format,
            'num_frames': self.num_frames,
            'num_vertices': self.num_vertices,
            'num_faces': self.num_faces,
            'fps': self.fps,
            'export_timestamp': self.export_timestamp,
            'pipeline_version': self.pipeline_version,
            'config_hash': self.config_hash,
        }
