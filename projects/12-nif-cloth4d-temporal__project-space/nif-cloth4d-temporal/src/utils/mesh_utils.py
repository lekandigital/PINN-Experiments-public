"""
Mesh export utilities for NIF-Cloth4D-Temporal.

Supports exporting cloth sequences to:
    - USD (Universal Scene Description) for DCC tools
    - OBJ (Wavefront) for wide compatibility
    - PLY for point cloud workflows
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, List


def export_to_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    normals: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
) -> str:
    """
    Export mesh to OBJ format.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Triangle indices (F, 3), 0-indexed
        output_path: Output file path
        normals: Optional vertex normals (N, 3)
        uvs: Optional UV coordinates (N, 2)
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# NIF-Cloth4D-Temporal OBJ Export\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Normals
        if normals is not None:
            f.write("\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        # UVs
        if uvs is not None:
            f.write("\n")
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Faces (OBJ uses 1-indexed)
        f.write("\n")
        for face in faces:
            if normals is not None and uvs is not None:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                       f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                       f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            elif normals is not None:
                f.write(f"f {face[0]+1}//{face[0]+1} "
                       f"{face[1]+1}//{face[1]+1} "
                       f"{face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    return str(output_path)


def export_to_ply(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    colors: Optional[np.ndarray] = None,
    binary: bool = True,
) -> str:
    """
    Export mesh to PLY format.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Triangle indices (F, 3)
        output_path: Output file path
        colors: Optional vertex colors (N, 3) as uint8
        binary: If True, use binary format (smaller files)
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # Header
    header = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    
    header.extend([
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ])
    
    with open(output_path, 'wb' if binary else 'w') as f:
        # Write header
        header_str = '\n'.join(header) + '\n'
        if binary:
            f.write(header_str.encode('ascii'))
        else:
            f.write(header_str)
        
        # Write data
        if binary:
            # Vertices
            for i, v in enumerate(vertices):
                f.write(np.array(v, dtype=np.float32).tobytes())
                if colors is not None:
                    f.write(np.array(colors[i], dtype=np.uint8).tobytes())
            
            # Faces
            for face in faces:
                f.write(np.array([3], dtype=np.uint8).tobytes())
                f.write(np.array(face, dtype=np.int32).tobytes())
        else:
            # ASCII format
            for i, v in enumerate(vertices):
                line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                if colors is not None:
                    line += f" {colors[i][0]} {colors[i][1]} {colors[i][2]}"
                f.write(line + '\n')
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    return str(output_path)


def export_to_usd(
    vertices_sequence: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    mesh_name: str = "cloth",
) -> str:
    """
    Export animated mesh sequence to USD format.
    
    USD (Universal Scene Description) is the standard format for
    exchanging 3D data between DCC applications.
    
    Args:
        vertices_sequence: Vertex positions per frame (T, N, 3)
        faces: Triangle indices (F, 3), shared across frames
        output_path: Output file path (.usda or .usd)
        fps: Framerate
        mesh_name: Name of the mesh in USD hierarchy
        
    Returns:
        Path to saved file
    """
    try:
        from pxr import Usd, UsdGeom, Vt, Gf, Sdf
    except ImportError:
        raise ImportError(
            "USD Python bindings (pxr) not found. "
            "Install with: pip install usd-core"
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(vertices_sequence)
    
    # Create stage
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetTimeCodesPerSecond(fps)
    
    # Create mesh
    mesh_path = f"/World/{mesh_name}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Set face vertex counts (all triangles = 3)
    face_counts = [3] * len(faces)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    
    # Set face vertex indices
    face_indices = faces.flatten().tolist()
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    
    # Set animated vertex positions
    points_attr = mesh.GetPointsAttr()
    
    for frame, vertices in enumerate(vertices_sequence):
        # Convert to USD point format
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        points_attr.Set(Vt.Vec3fArray(points), Usd.TimeCode(frame))
    
    # Set subdivision scheme
    mesh.GetSubdivisionSchemeAttr().Set("none")
    
    # Save
    stage.Save()
    
    return str(output_path)


def export_mesh_sequence(
    vertices_sequence: Union[np.ndarray, List[np.ndarray]],
    faces: np.ndarray,
    output_dir: str,
    format: str = 'obj',
    fps: float = 30.0,
    prefix: str = 'frame',
) -> List[str]:
    """
    Export mesh sequence to individual files.
    
    Args:
        vertices_sequence: List of vertex arrays or (T, N, 3) array
        faces: Triangle indices (F, 3)
        output_dir: Output directory
        format: 'obj', 'ply', or 'usd'
        fps: Framerate (for USD)
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to list if numpy array
    if isinstance(vertices_sequence, np.ndarray):
        vertices_sequence = [vertices_sequence[i] for i in range(len(vertices_sequence))]
    
    saved_paths = []
    
    if format == 'usd':
        # Single USD file with animation
        usd_path = output_dir / f'{prefix}_sequence.usda'
        vertices_array = np.stack(vertices_sequence, axis=0)
        path = export_to_usd(vertices_array, faces, str(usd_path), fps=fps)
        saved_paths.append(path)
    else:
        # Individual files per frame
        for i, vertices in enumerate(vertices_sequence):
            if format == 'obj':
                path = output_dir / f'{prefix}_{i:04d}.obj'
                export_to_obj(vertices, faces, str(path))
            elif format == 'ply':
                path = output_dir / f'{prefix}_{i:04d}.ply'
                export_to_ply(vertices, faces, str(path))
            else:
                raise ValueError(f"Unknown format: {format}")
            
            saved_paths.append(str(path))
    
    return saved_paths


def compute_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex normals from mesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Triangle indices (F, 3)
        
    Returns:
        Vertex normals (N, 3), normalized
    """
    # Initialize normals
    normals = np.zeros_like(vertices)
    
    # Get face vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute face normals
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = np.cross(e1, e2)
    
    # Accumulate to vertices
    for i, face in enumerate(faces):
        normals[face[0]] += face_normals[i]
        normals[face[1]] += face_normals[i]
        normals[face[2]] += face_normals[i]
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    return normals
