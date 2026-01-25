"""
Mesh Loading Utilities for HGNN-NIF-Cloth

Provides functions to load cloth meshes from various formats:
- OBJ: Standard Wavefront format
- PLY: Stanford polygon format
- Alembic: For animated sequences (optional dependency)
- NPZ: NumPy archive format
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings


def load_obj(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load OBJ mesh file.

    Supports:
    - Vertex positions (v)
    - Faces (f) with various formats: v, v/vt, v/vt/vn, v//vn
    - Triangles and quads (quads are triangulated)

    Args:
        filepath: Path to OBJ file

    Returns:
        vertices: (N, 3) float32 array of vertex positions
        faces: (F, 3) int64 array of triangle face indices
        edges: (2, E) int64 array of unique edge indices
    """
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v' and len(parts) >= 4:
                # Vertex position
                vertices.append([float(x) for x in parts[1:4]])

            elif parts[0] == 'f':
                # Face - handle various formats
                face_verts = []
                for p in parts[1:]:
                    # Split on '/' to handle v, v/vt, v/vt/vn, v//vn
                    v_idx = int(p.split('/')[0])
                    # OBJ is 1-indexed, convert to 0-indexed
                    # Handle negative indices
                    if v_idx < 0:
                        v_idx = len(vertices) + v_idx
                    else:
                        v_idx = v_idx - 1
                    face_verts.append(v_idx)

                # Triangulate if necessary
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    # Quad -> two triangles
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                elif len(face_verts) > 4:
                    # Fan triangulation for polygons
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64)

    # Extract unique edges
    edges = _extract_edges(faces)

    return vertices, faces, edges


def load_ply(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PLY mesh file (ASCII format).

    Args:
        filepath: Path to PLY file

    Returns:
        vertices: (N, 3) float32 array
        faces: (F, 3) int64 array
        edges: (2, E) int64 array
    """
    vertices = []
    faces = []
    vertex_count = 0
    face_count = 0
    in_header = True
    reading_vertices = False
    reading_faces = False
    vertex_props = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])
                elif line.startswith('property') and 'vertex' not in line:
                    vertex_props.append(line.split()[-1])
                elif line == 'end_header':
                    in_header = False
                    reading_vertices = True
                continue

            if reading_vertices:
                parts = line.split()
                # Take first 3 values as x, y, z
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(vertices) >= vertex_count:
                    reading_vertices = False
                    reading_faces = True
                continue

            if reading_faces:
                parts = line.split()
                n_verts = int(parts[0])
                face_verts = [int(parts[i + 1]) for i in range(n_verts)]

                if n_verts == 3:
                    faces.append(face_verts)
                elif n_verts == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                elif n_verts > 4:
                    for i in range(1, n_verts - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64)
    edges = _extract_edges(faces)

    return vertices, faces, edges


def load_npz(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load mesh data from NumPy archive.

    Expected keys:
    - 'vertices' or 'positions': (N, 3) vertex positions
    - 'faces': (F, 3) face indices
    - 'edges': (2, E) or (E, 2) edge indices (optional)

    Args:
        filepath: Path to .npz file

    Returns:
        Dictionary with mesh components
    """
    data = np.load(filepath)

    result = {}

    # Load vertices
    if 'vertices' in data:
        result['vertices'] = data['vertices'].astype(np.float32)
    elif 'positions' in data:
        result['vertices'] = data['positions'].astype(np.float32)
    else:
        raise ValueError("No vertices found in NPZ file")

    # Load faces
    if 'faces' in data:
        result['faces'] = data['faces'].astype(np.int64)
    else:
        raise ValueError("No faces found in NPZ file")

    # Load or compute edges
    if 'edges' in data:
        edges = data['edges'].astype(np.int64)
        # Ensure shape is (2, E)
        if edges.shape[0] != 2:
            edges = edges.T
        result['edges'] = edges
    else:
        result['edges'] = _extract_edges(result['faces'])

    return result


def load_alembic(
    filepath: str,
    frame: int = 0,
    mesh_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh from Alembic (.abc) file at specific frame.

    Requires: alembic package (pip install alembic)

    Args:
        filepath: Path to Alembic file
        frame: Frame number to load
        mesh_name: Specific mesh to load (if multiple in scene)

    Returns:
        vertices: (N, 3) float32 array
        faces: (F, 3) int64 array
        edges: (2, E) int64 array
    """
    try:
        import alembic
        from alembic import Abc, AbcGeom
    except ImportError:
        raise ImportError(
            "Alembic loading requires the alembic package. "
            "Install with: pip install alembic"
        )

    archive = Abc.IArchive(filepath)
    root = archive.getTop()

    def find_mesh(obj, name=None):
        """Recursively find mesh in hierarchy."""
        if AbcGeom.IPolyMesh.matches(obj.getHeader()):
            mesh = AbcGeom.IPolyMesh(obj)
            if name is None or obj.getName() == name:
                return mesh
        for i in range(obj.getNumChildren()):
            child = obj.getChild(i)
            result = find_mesh(child, name)
            if result is not None:
                return result
        return None

    mesh = find_mesh(root, mesh_name)
    if mesh is None:
        raise ValueError(f"No mesh found in {filepath}")

    schema = mesh.getSchema()
    sample = schema.getValue(Abc.ISampleSelector(frame))

    # Get positions
    positions = np.array(sample.getPositions())

    # Get face counts and indices
    face_counts = np.array(sample.getFaceCounts())
    face_indices = np.array(sample.getFaceIndices())

    # Build face array (triangulate if needed)
    faces = []
    idx = 0
    for count in face_counts:
        face_verts = face_indices[idx:idx + count]
        if count == 3:
            faces.append(face_verts)
        elif count == 4:
            faces.append([face_verts[0], face_verts[1], face_verts[2]])
            faces.append([face_verts[0], face_verts[2], face_verts[3]])
        elif count > 4:
            for i in range(1, count - 1):
                faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
        idx += count

    vertices = positions.astype(np.float32)
    faces = np.array(faces, dtype=np.int64)
    edges = _extract_edges(faces)

    return vertices, faces, edges


def load_alembic_sequence(
    filepath: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    mesh_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load animated mesh sequence from Alembic file.

    Args:
        filepath: Path to Alembic file
        start_frame: First frame to load
        end_frame: Last frame to load (None = all frames)
        step: Frame step
        mesh_name: Specific mesh to load

    Returns:
        vertices_sequence: (T, N, 3) float32 array
        faces: (F, 3) int64 array (constant topology assumed)
        edges: (2, E) int64 array
    """
    try:
        import alembic
        from alembic import Abc, AbcGeom
    except ImportError:
        raise ImportError("Alembic loading requires: pip install alembic")

    archive = Abc.IArchive(filepath)
    root = archive.getTop()

    def find_mesh(obj, name=None):
        if AbcGeom.IPolyMesh.matches(obj.getHeader()):
            mesh = AbcGeom.IPolyMesh(obj)
            if name is None or obj.getName() == name:
                return mesh
        for i in range(obj.getNumChildren()):
            result = find_mesh(obj.getChild(i), name)
            if result is not None:
                return result
        return None

    mesh = find_mesh(root, mesh_name)
    if mesh is None:
        raise ValueError(f"No mesh found in {filepath}")

    schema = mesh.getSchema()
    num_samples = schema.getNumSamples()

    if end_frame is None:
        end_frame = num_samples

    # Load first frame to get topology
    sample = schema.getValue(Abc.ISampleSelector(start_frame))
    face_counts = np.array(sample.getFaceCounts())
    face_indices = np.array(sample.getFaceIndices())

    # Build face array
    faces = []
    idx = 0
    for count in face_counts:
        face_verts = face_indices[idx:idx + count]
        if count == 3:
            faces.append(face_verts)
        elif count >= 4:
            for i in range(1, count - 1):
                faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
        idx += count
    faces = np.array(faces, dtype=np.int64)

    # Load all frames
    vertices_sequence = []
    for frame in range(start_frame, min(end_frame, num_samples), step):
        sample = schema.getValue(Abc.ISampleSelector(frame))
        positions = np.array(sample.getPositions())
        vertices_sequence.append(positions)

    vertices_sequence = np.stack(vertices_sequence, axis=0).astype(np.float32)
    edges = _extract_edges(faces)

    return vertices_sequence, faces, edges


def _extract_edges(faces: np.ndarray) -> np.ndarray:
    """
    Extract unique edges from face array.

    Args:
        faces: (F, 3) face indices

    Returns:
        edges: (2, E) unique edge indices
    """
    edge_set = set()
    for face in faces:
        for k in range(3):
            e = tuple(sorted([int(face[k]), int(face[(k + 1) % 3])]))
            edge_set.add(e)
    edges = np.array(list(edge_set), dtype=np.int64).T
    return edges


def normalize_mesh(
    vertices: np.ndarray,
    center: bool = True,
    scale: bool = True,
    target_size: float = 2.0
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize mesh to standard coordinates.

    Args:
        vertices: (N, 3) vertex positions
        center: Center mesh at origin
        scale: Scale to target size
        target_size: Target bounding box size

    Returns:
        normalized_vertices: (N, 3) normalized positions
        transform: Dict with 'center' and 'scale' for inverse transform
    """
    transform = {}

    if center:
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        transform['center'] = centroid
    else:
        transform['center'] = np.zeros(3)

    if scale:
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = extent.max()
        if max_extent > 0:
            scale_factor = target_size / max_extent
            vertices = vertices * scale_factor
            transform['scale'] = scale_factor
        else:
            transform['scale'] = 1.0
    else:
        transform['scale'] = 1.0

    return vertices.astype(np.float32), transform


def compute_mesh_info(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray
) -> Dict:
    """
    Compute mesh statistics and information.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        edges: (2, E) edge indices

    Returns:
        Dictionary with mesh statistics
    """
    # Bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Edge lengths
    edge_vecs = vertices[edges[1]] - vertices[edges[0]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=-1)

    # Face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=-1)

    return {
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'num_edges': edges.shape[1],
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'edge_length_mean': float(edge_lengths.mean()),
        'edge_length_std': float(edge_lengths.std()),
        'edge_length_min': float(edge_lengths.min()),
        'edge_length_max': float(edge_lengths.max()),
        'total_area': float(face_areas.sum()),
        'mean_face_area': float(face_areas.mean()),
    }


def auto_load(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Automatically load mesh based on file extension.

    Supported formats: .obj, .ply, .npz, .abc

    Args:
        filepath: Path to mesh file

    Returns:
        vertices: (N, 3) float32 array
        faces: (F, 3) int64 array
        edges: (2, E) int64 array
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == '.obj':
        return load_obj(filepath)
    elif ext == '.ply':
        return load_ply(filepath)
    elif ext == '.npz':
        data = load_npz(filepath)
        return data['vertices'], data['faces'], data['edges']
    elif ext == '.abc':
        return load_alembic(filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
