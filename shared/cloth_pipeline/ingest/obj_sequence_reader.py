"""
OBJ Sequence Reader for the cloth simulation pipeline.

Reads numbered OBJ sequences exported from Marvelous Designer, Blender, 
Houdini, or any other DCC tool.

Supports naming patterns like:
- frame_0001.obj, frame_0002.obj, ...
- cloth_001.obj, cloth_002.obj, ...
- mesh.0001.obj, mesh.0002.obj, ...
"""

import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass

from ..types import ClothSequence


@dataclass
class OBJMesh:
    """Parsed data from a single OBJ file."""
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (F, 3 or 4) int64
    normals: Optional[np.ndarray] = None  # (N, 3) or None
    uvs: Optional[np.ndarray] = None  # (N, 2) or None
    vertex_normals: Optional[np.ndarray] = None  # (N, 3) per-vertex normals
    

class OBJSequenceReader:
    """
    Reader for OBJ mesh sequences.
    
    Example usage:
        reader = OBJSequenceReader("/path/to/sequence/")
        sequence = reader.read(fps=24.0)
    """
    
    # Common OBJ sequence naming patterns
    DEFAULT_PATTERNS = [
        r'^frame[_-]?(\d+)\.obj$',      # frame_0001.obj, frame-1.obj
        r'^cloth[_-]?(\d+)\.obj$',       # cloth_001.obj
        r'^mesh[_-]?(\d+)\.obj$',        # mesh_001.obj
        r'^.*?[_\-\.](\d+)\.obj$',       # anything.0001.obj, thing_001.obj
    ]
    
    def __init__(self, directory: Union[str, Path], pattern: Optional[str] = None):
        """
        Initialize the OBJ sequence reader.
        
        Args:
            directory: Path to directory containing OBJ files
            pattern: Regex pattern with a capture group for the frame number.
                    If None, auto-detect from DEFAULT_PATTERNS.
        """
        self.directory = Path(directory)
        self.pattern = pattern
        self._file_list: Optional[List[Tuple[int, Path]]] = None
        
    def discover_files(self) -> List[Tuple[int, Path]]:
        """
        Discover and sort OBJ files in the directory.
        
        Returns:
            List of (frame_number, file_path) tuples, sorted by frame number
        """
        if self._file_list is not None:
            return self._file_list
        
        obj_files = list(self.directory.glob("*.obj"))
        
        if not obj_files:
            raise FileNotFoundError(f"No OBJ files found in {self.directory}")
        
        # Try to find a pattern that matches
        patterns_to_try = [self.pattern] if self.pattern else self.DEFAULT_PATTERNS
        
        for pattern in patterns_to_try:
            if pattern is None:
                continue
                
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []
            
            for filepath in obj_files:
                match = regex.match(filepath.name)
                if match:
                    frame_num = int(match.group(1))
                    matches.append((frame_num, filepath))
            
            if len(matches) == len(obj_files):
                # This pattern matches all files
                self._file_list = sorted(matches, key=lambda x: x[0])
                return self._file_list
        
        # Fallback: just sort alphabetically and use index as frame number
        obj_files.sort()
        self._file_list = [(i, f) for i, f in enumerate(obj_files)]
        return self._file_list
    
    def read(
        self,
        fps: float = 24.0,
        rest_frame: int = 0,
        validate_topology: bool = True,
        compute_normals: bool = True,
    ) -> ClothSequence:
        """
        Read the full OBJ sequence into a ClothSequence.
        
        Args:
            fps: Framerate of the sequence
            rest_frame: Which frame index represents the rest pose
            validate_topology: If True, verify all frames have same topology
            compute_normals: If True, compute vertex normals from faces
            
        Returns:
            ClothSequence containing all frames
            
        Raises:
            ValueError: If topology changes between frames (and validate_topology=True)
        """
        file_list = self.discover_files()
        
        if not file_list:
            raise ValueError("No OBJ files found")
        
        # Read first frame to get mesh structure
        first_frame_num, first_file = file_list[0]
        reference_mesh = self._parse_obj(first_file)
        
        num_frames = len(file_list)
        num_vertices = reference_mesh.vertices.shape[0]
        num_faces = reference_mesh.faces.shape[0]
        face_size = reference_mesh.faces.shape[1]
        
        # Pre-allocate arrays
        vertices = np.zeros((num_frames, num_vertices, 3), dtype=np.float32)
        vertices[0] = reference_mesh.vertices
        
        has_normals = reference_mesh.normals is not None
        normals = np.zeros((num_frames, num_vertices, 3), dtype=np.float32) if has_normals else None
        if has_normals:
            normals[0] = reference_mesh.normals
        
        # Read remaining frames
        for i, (frame_num, filepath) in enumerate(file_list[1:], start=1):
            mesh = self._parse_obj(filepath)
            
            # Validate topology consistency
            if validate_topology:
                if mesh.vertices.shape[0] != num_vertices:
                    raise ValueError(
                        f"Topology mismatch at frame {frame_num} ({filepath.name}): "
                        f"expected {num_vertices} vertices, got {mesh.vertices.shape[0]}"
                    )
                if mesh.faces.shape[0] != num_faces:
                    raise ValueError(
                        f"Topology mismatch at frame {frame_num} ({filepath.name}): "
                        f"expected {num_faces} faces, got {mesh.faces.shape[0]}"
                    )
                if not np.array_equal(mesh.faces, reference_mesh.faces):
                    raise ValueError(
                        f"Face connectivity changed at frame {frame_num} ({filepath.name})"
                    )
            
            vertices[i] = mesh.vertices
            
            if has_normals and mesh.normals is not None:
                normals[i] = mesh.normals
        
        # Compute normals if not present but requested
        if normals is None and compute_normals:
            normals = self._compute_all_normals(vertices, reference_mesh.faces)
        
        # Build frame range from actual frame numbers
        frame_numbers = [f[0] for f in file_list]
        frame_range = (min(frame_numbers), max(frame_numbers))
        
        return ClothSequence(
            vertices=vertices,
            faces=reference_mesh.faces,
            normals=normals,
            uvs=reference_mesh.uvs,
            velocities=None,  # Will be computed in transform stage
            fps=fps,
            dt=1.0 / fps,
            frame_range=frame_range,
            metadata={
                'source_format': 'obj_sequence',
                'source_directory': str(self.directory),
                'num_source_files': num_frames,
                'pattern_used': self.pattern,
            }
        )
    
    def _parse_obj(self, filepath: Path) -> OBJMesh:
        """
        Parse a single OBJ file.
        
        Uses a fast line-by-line parser optimized for common OBJ structure.
        """
        vertices = []
        faces = []
        normals = []
        uvs = []
        vertex_normals = []  # vn lines
        vertex_uvs = []  # vt lines
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                prefix = parts[0]
                
                if prefix == 'v':
                    # Vertex position
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                elif prefix == 'vn':
                    # Vertex normal
                    vertex_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                elif prefix == 'vt':
                    # Texture coordinate
                    vertex_uvs.append([float(parts[1]), float(parts[2])])
                
                elif prefix == 'f':
                    # Face - handle multiple formats:
                    # f v1 v2 v3
                    # f v1/vt1 v2/vt2 v3/vt3
                    # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                    # f v1//vn1 v2//vn2 v3//vn3
                    face_verts = []
                    for part in parts[1:]:
                        # Get vertex index (first number)
                        idx = part.split('/')[0]
                        face_verts.append(int(idx) - 1)  # OBJ is 1-indexed
                    faces.append(face_verts)
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Handle faces (might be triangles or quads)
        face_sizes = [len(f) for f in faces]
        if len(set(face_sizes)) > 1:
            # Mixed face sizes - triangulate quads
            triangulated = []
            for face in faces:
                if len(face) == 3:
                    triangulated.append(face)
                elif len(face) == 4:
                    # Quad -> 2 triangles
                    triangulated.append([face[0], face[1], face[2]])
                    triangulated.append([face[0], face[2], face[3]])
                else:
                    # Fan triangulation for polygons with >4 vertices
                    for i in range(1, len(face) - 1):
                        triangulated.append([face[0], face[i], face[i + 1]])
            faces = triangulated
        
        faces = np.array(faces, dtype=np.int64)
        
        # Process normals
        processed_normals = None
        if vertex_normals:
            processed_normals = np.array(vertex_normals, dtype=np.float32)
            # If we have per-face-vertex normals, we need to average them per vertex
            if len(processed_normals) != len(vertices):
                # This is a simplified approach - assumes normal indices match vertex indices
                # In complex OBJ files, this might not be true
                processed_normals = None
        
        # Process UVs
        processed_uvs = None
        if vertex_uvs and len(vertex_uvs) == len(vertices):
            processed_uvs = np.array(vertex_uvs, dtype=np.float32)
        
        return OBJMesh(
            vertices=vertices,
            faces=faces,
            normals=processed_normals,
            uvs=processed_uvs,
        )
    
    def _compute_all_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Compute vertex normals for all frames.
        
        Args:
            vertices: (F, N, 3) vertex positions
            faces: (num_faces, 3) face indices
            
        Returns:
            (F, N, 3) vertex normals
        """
        num_frames, num_vertices, _ = vertices.shape
        normals = np.zeros((num_frames, num_vertices, 3), dtype=np.float32)
        
        for f in range(num_frames):
            normals[f] = self._compute_vertex_normals(vertices[f], faces)
        
        return normals
    
    def _compute_vertex_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Compute vertex normals from face normals (area-weighted average).
        
        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices
            
        Returns:
            (N, 3) vertex normals (normalized)
        """
        num_vertices = vertices.shape[0]
        vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)
        
        # Get face vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Compute face normals (not normalized - magnitude is 2x triangle area)
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = np.cross(e1, e2)
        
        # Accumulate face normals to vertices (area-weighted)
        np.add.at(vertex_normals, faces[:, 0], face_normals)
        np.add.at(vertex_normals, faces[:, 1], face_normals)
        np.add.at(vertex_normals, faces[:, 2], face_normals)
        
        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        vertex_normals = vertex_normals / norms
        
        return vertex_normals


def read_obj_sequence(
    directory: Union[str, Path],
    fps: float = 24.0,
    rest_frame: int = 0,
    pattern: Optional[str] = None,
    validate_topology: bool = True,
    **kwargs
) -> ClothSequence:
    """
    Read an OBJ sequence from a directory.
    
    This is the main entry point for OBJ sequence ingestion.
    
    Args:
        directory: Path to directory containing OBJ files
        fps: Framerate of the sequence
        rest_frame: Which frame represents the rest pose
        pattern: Optional regex pattern for frame number extraction
        validate_topology: Verify consistent topology across frames
        
    Returns:
        ClothSequence ready for transformation
        
    Example:
        sequence = read_obj_sequence(
            "/path/to/obj_sequence/",
            fps=30.0,
            pattern=r"cloth_(\d+).obj"
        )
    """
    reader = OBJSequenceReader(directory, pattern=pattern)
    return reader.read(
        fps=fps,
        rest_frame=rest_frame,
        validate_topology=validate_topology,
        **kwargs
    )
