"""
Mesh Generation Utilities
=========================

Functions for creating standard test meshes:
- Icosphere (subdivided icosahedron)
- Torus
- Flat rectangular grid
- Cubed sphere (no pole singularities)
"""

import numpy as np
from typing import Tuple


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an icosahedral sphere mesh via subdivision.
    
    Starts with a regular icosahedron and recursively subdivides each
    triangle into 4 smaller triangles, projecting vertices to the sphere.
    
    Args:
        subdivisions: Number of subdivision iterations (0 = icosahedron with 12 vertices)
        radius: Sphere radius
        
    Returns:
        vertices: (V, 3) vertex positions on sphere
        faces: (F, 3) triangle face indices
        
    Vertex counts by subdivision:
        0: 12 vertices, 20 faces
        1: 42 vertices, 80 faces
        2: 162 vertices, 320 faces
        3: 642 vertices, 1280 faces
        4: 2562 vertices, 5120 faces
    """
    # Golden ratio for icosahedron construction
    phi = (1 + np.sqrt(5)) / 2
    
    # Initial icosahedron vertices (normalized)
    vertices = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=np.float64)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Initial icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    
    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = _subdivide_icosphere(vertices, faces)
    
    # Scale to desired radius
    vertices = vertices * radius
    
    return vertices, faces


def _subdivide_icosphere(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide each triangle into 4 triangles and project to sphere.
    
    For each triangle with vertices (v0, v1, v2), add midpoint vertices
    (m01, m12, m20) and create 4 sub-triangles:
        (v0, m01, m20), (v1, m12, m01), (v2, m20, m12), (m01, m12, m20)
    """
    edge_midpoints = {}  # (min_idx, max_idx) -> midpoint_vertex_idx
    new_vertices = list(vertices)
    new_faces = []
    
    def get_midpoint(i: int, j: int) -> int:
        """Get or create midpoint vertex between vertices i and j."""
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        
        # Create new vertex at midpoint, projected to sphere
        midpoint = (vertices[i] + vertices[j]) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)  # Project to unit sphere
        
        new_idx = len(new_vertices)
        new_vertices.append(midpoint)
        edge_midpoints[key] = new_idx
        return new_idx
    
    for face in faces:
        v0, v1, v2 = face
        
        # Get midpoint vertices
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        
        # Create 4 sub-faces
        new_faces.extend([
            [v0, m01, m20],
            [v1, m12, m01],
            [v2, m20, m12],
            [m01, m12, m20],
        ])
    
    return np.array(new_vertices), np.array(new_faces, dtype=np.int64)


def torus_mesh(
    R: float = 1.0,
    r: float = 0.4,
    n_major: int = 32,
    n_minor: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a torus mesh.
    
    Parametrization:
        x = (R + r*cos(φ)) * cos(θ)
        y = (R + r*cos(φ)) * sin(θ)
        z = r * sin(φ)
    
    Args:
        R: Major radius (center of tube to center of torus)
        r: Minor radius (tube radius)
        n_major: Number of divisions around major circumference
        n_minor: Number of divisions around minor circumference
        
    Returns:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle face indices
    """
    # Create parameter grid
    theta = np.linspace(0, 2*np.pi, n_major, endpoint=False)
    phi = np.linspace(0, 2*np.pi, n_minor, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Compute vertex positions
    x = (R + r * np.cos(phi_grid)) * np.cos(theta_grid)
    y = (R + r * np.cos(phi_grid)) * np.sin(theta_grid)
    z = r * np.sin(phi_grid)
    
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    
    # Create faces (two triangles per grid cell)
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            # Vertex indices (with wrapping)
            i_next = (i + 1) % n_major
            j_next = (j + 1) % n_minor
            
            v00 = i * n_minor + j
            v10 = i_next * n_minor + j
            v01 = i * n_minor + j_next
            v11 = i_next * n_minor + j_next
            
            # Two triangles per quad
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    
    return vertices, np.array(faces, dtype=np.int64)


def flat_grid(
    nx: int = 10,
    ny: int = 10,
    lx: float = 1.0,
    ly: float = 1.0,
    z: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a flat rectangular triangular mesh.
    
    Useful for testing and as initial cloth shape.
    
    Args:
        nx: Number of vertices in x direction
        ny: Number of vertices in y direction
        lx: Total length in x
        ly: Total length in y
        z: Height (z coordinate of all vertices)
        
    Returns:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle face indices
    """
    # Create vertex grid
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = np.full_like(xx, z)
    
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    
    # Create faces (two triangles per grid cell)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Vertex indices
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)
            
            # Two triangles per quad
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    
    return vertices, np.array(faces, dtype=np.int64)


def cubed_sphere(n: int = 8, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a cubed-sphere mesh (no pole singularities).
    
    Projects a cube onto a sphere, giving 6 equal-area patches
    with no coordinate singularities.
    
    Args:
        n: Number of divisions per cube edge
        radius: Sphere radius
        
    Returns:
        vertices: (V, 3) vertex positions on sphere
        faces: (F, 3) triangle face indices
    """
    # Generate points on each cube face, then project to sphere
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Six faces: +x, -x, +y, -y, +z, -z
    face_axes = [
        (0, 1, 2, 1),   # +x: (y, z) parameterize, x = 1
        (0, 1, 2, -1),  # -x: (y, z) parameterize, x = -1
        (1, 0, 2, 1),   # +y: (x, z) parameterize, y = 1
        (1, 0, 2, -1),  # -y: (x, z) parameterize, y = -1
        (2, 0, 1, 1),   # +z: (x, y) parameterize, z = 1
        (2, 0, 1, -1),  # -z: (x, y) parameterize, z = -1
    ]
    
    for fixed_axis, axis1, axis2, sign in face_axes:
        # Create grid on this face
        t = np.linspace(-1, 1, n)
        t1_grid, t2_grid = np.meshgrid(t, t, indexing='ij')
        
        # Build vertices on cube face
        face_verts = np.zeros((n * n, 3))
        face_verts[:, fixed_axis] = sign
        face_verts[:, axis1] = t1_grid.ravel()
        face_verts[:, axis2] = t2_grid.ravel()
        
        # Project to sphere
        norms = np.linalg.norm(face_verts, axis=1, keepdims=True)
        face_verts = face_verts / norms * radius
        
        all_vertices.append(face_verts)
        
        # Create faces for this patch
        for i in range(n - 1):
            for j in range(n - 1):
                v00 = vertex_offset + i * n + j
                v10 = vertex_offset + (i + 1) * n + j
                v01 = vertex_offset + i * n + (j + 1)
                v11 = vertex_offset + (i + 1) * n + (j + 1)
                
                all_faces.append([v00, v10, v11])
                all_faces.append([v00, v11, v01])
        
        vertex_offset += n * n
    
    vertices = np.vstack(all_vertices)
    faces = np.array(all_faces, dtype=np.int64)
    
    # TODO: Merge duplicate vertices at cube edges for watertight mesh
    # For now, each face patch is independent
    
    return vertices, faces


def uv_sphere(
    n_lat: int = 16,
    n_lon: int = 32,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a latitude-longitude sphere mesh.
    
    Warning: Has degenerate triangles and clustering at poles.
    Prefer icosphere for uniform sampling.
    
    Args:
        n_lat: Number of latitude bands
        n_lon: Number of longitude divisions
        radius: Sphere radius
        
    Returns:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle face indices
    """
    # Create vertices
    vertices = []
    
    # North pole
    vertices.append([0, 0, radius])
    
    # Interior latitude bands
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat  # 0 to pi
        z = radius * np.cos(theta)
        r_xy = radius * np.sin(theta)
        
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = r_xy * np.cos(phi)
            y = r_xy * np.sin(phi)
            vertices.append([x, y, z])
    
    # South pole
    vertices.append([0, 0, -radius])
    
    vertices = np.array(vertices)
    
    # Create faces
    faces = []
    
    # Top cap (triangles from north pole)
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])
    
    # Middle bands (quads split into triangles)
    for i in range(n_lat - 2):
        row_start = 1 + i * n_lon
        next_row_start = 1 + (i + 1) * n_lon
        
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            
            v00 = row_start + j
            v01 = row_start + j_next
            v10 = next_row_start + j
            v11 = next_row_start + j_next
            
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    
    # Bottom cap (triangles to south pole)
    south_pole = len(vertices) - 1
    last_row_start = 1 + (n_lat - 2) * n_lon
    
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([last_row_start + j, south_pole, last_row_start + j_next])
    
    return vertices, np.array(faces, dtype=np.int64)
