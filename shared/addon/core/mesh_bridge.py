"""
Mesh Bridge - Converts between Blender mesh data and numpy arrays.

This is the only file that should import bpy for mesh access.
Backends never touch bpy — all data crosses as numpy arrays.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np


def blender_mesh_to_numpy(obj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mesh data from a Blender object.
    
    Args:
        obj: Blender object (bpy.types.Object) with mesh data
        
    Returns:
        Tuple of:
        - vertices: (V, 3) float32 array in world space
        - faces: (F, 3) int32 array of vertex indices
        - edges: (E, 2) int32 array of vertex indices
        
    Note:
        Applies the object's world transform so vertices are in world space.
        Backends work in world space, not object-local space.
    """
    import bpy
    
    mesh = obj.data
    world_matrix = np.array(obj.matrix_world, dtype=np.float32)
    
    # Ensure mesh data is up to date
    if obj.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    mesh.calc_loop_triangles()
    
    # Extract vertices
    num_verts = len(mesh.vertices)
    vertices = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", vertices)
    vertices = vertices.reshape(-1, 3)
    
    # Apply world transform
    # vertices_world = vertices @ world_matrix[:3, :3].T + world_matrix[:3, 3]
    vertices_4d = np.hstack([vertices, np.ones((num_verts, 1), dtype=np.float32)])
    vertices = (vertices_4d @ world_matrix.T)[:, :3]
    
    # Extract faces (triangulated)
    num_tris = len(mesh.loop_triangles)
    if num_tris > 0:
        triangles = np.empty(num_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", triangles)
        faces = triangles.reshape(-1, 3)
    else:
        faces = np.array([], dtype=np.int32).reshape(0, 3)
    
    # Extract edges
    num_edges = len(mesh.edges)
    if num_edges > 0:
        edges = np.empty(num_edges * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
    else:
        edges = np.array([], dtype=np.int32).reshape(0, 2)
    
    return vertices, faces, edges


def get_vertex_normals(obj) -> np.ndarray:
    """
    Extract vertex normals from a Blender object.
    
    Args:
        obj: Blender object with mesh data
        
    Returns:
        (V, 3) float32 array of vertex normals in world space
    """
    mesh = obj.data
    world_matrix = np.array(obj.matrix_world, dtype=np.float32)
    normal_matrix = np.linalg.inv(world_matrix[:3, :3]).T
    
    # Get normals
    mesh.calc_normals()
    num_verts = len(mesh.vertices)
    normals = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)
    
    # Transform to world space
    normals = normals @ normal_matrix.T
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, where=norms > 1e-8)
    
    return normals.astype(np.float32)


def numpy_displacements_to_blender(
    obj, 
    displacements: np.ndarray, 
    rest_vertices: np.ndarray
) -> None:
    """
    Apply displacement vectors to a Blender mesh.
    
    Computes: new_position = rest_position + displacement
    Then transforms back to object-local space.
    
    Args:
        obj: Blender object to modify
        displacements: (V, 3) displacement vectors in world space
        rest_vertices: (V, 3) rest positions in world space
    """
    world_matrix = np.array(obj.matrix_world, dtype=np.float32)
    inv_world = np.linalg.inv(world_matrix)
    
    # Compute world-space positions
    world_positions = rest_vertices + displacements
    
    # Transform back to object-local space
    num_verts = len(world_positions)
    positions_4d = np.hstack([world_positions, np.ones((num_verts, 1), dtype=np.float32)])
    local_positions = (positions_4d @ inv_world.T)[:, :3]
    
    # Write back to Blender mesh
    mesh = obj.data
    local_positions_flat = local_positions.flatten().astype(np.float32)
    mesh.vertices.foreach_set("co", local_positions_flat)
    mesh.update()


def numpy_positions_to_blender(obj, positions: np.ndarray) -> None:
    """
    Set absolute vertex positions on a Blender mesh.
    
    Args:
        obj: Blender object to modify
        positions: (V, 3) absolute positions in world space
    """
    world_matrix = np.array(obj.matrix_world, dtype=np.float32)
    inv_world = np.linalg.inv(world_matrix)
    
    # Transform to object-local space
    num_verts = len(positions)
    positions_4d = np.hstack([positions, np.ones((num_verts, 1), dtype=np.float32)])
    local_positions = (positions_4d @ inv_world.T)[:, :3]
    
    # Write back
    mesh = obj.data
    local_positions_flat = local_positions.flatten().astype(np.float32)
    mesh.vertices.foreach_set("co", local_positions_flat)
    mesh.update()


def sdf_grid_to_blender_mesh(
    sdf_grid: np.ndarray,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    obj,
    level: float = 0.0,
    smooth: bool = True,
    smooth_iterations: int = 2
) -> None:
    """
    Run marching cubes on an SDF grid and update a Blender object's mesh.
    
    Used for SDF-output backends (Projects 09, 13).
    
    Args:
        sdf_grid: (R, R, R) array of SDF values
        bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        obj: Blender object to update
        level: Isosurface level (default 0.0 for SDF)
        smooth: Whether to apply Laplacian smoothing
        smooth_iterations: Number of smoothing iterations
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for SDF mesh extraction. "
            "Install: pip install scikit-image"
        )
    
    # Run marching cubes
    try:
        verts, faces, normals, values = marching_cubes(
            sdf_grid, 
            level=level,
            spacing=(1.0, 1.0, 1.0)
        )
    except Exception as e:
        print(f"[NeuralSim] Marching cubes failed: {e}")
        return
    
    if len(verts) == 0:
        print("[NeuralSim] Warning: Marching cubes produced no vertices")
        return
    
    # Scale vertices from grid coordinates to world coordinates
    min_bound = np.array(bounds[0])
    max_bound = np.array(bounds[1])
    resolution = sdf_grid.shape[0]
    scale = (max_bound - min_bound) / (resolution - 1)
    verts = verts * scale + min_bound
    
    # Optional smoothing
    if smooth and len(verts) > 0 and len(faces) > 0:
        verts = laplacian_smooth(verts, faces, smooth_iterations, lambda_factor=0.5)
    
    # Update Blender mesh
    _update_blender_mesh(obj, verts, faces)


def laplacian_smooth(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    iterations: int = 2,
    lambda_factor: float = 0.5
) -> np.ndarray:
    """
    Apply Laplacian smoothing to a mesh.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing factor (0-1)
        
    Returns:
        Smoothed vertex positions
    """
    from collections import defaultdict
    
    # Build adjacency list
    adjacency = defaultdict(set)
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    verts = vertices.copy()
    
    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(len(verts)):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                centroid = np.mean(verts[neighbors], axis=0)
                new_verts[i] = verts[i] + lambda_factor * (centroid - verts[i])
        verts = new_verts
    
    return verts


def _update_blender_mesh(obj, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Replace a Blender object's mesh data with new vertices and faces.
    
    Args:
        obj: Blender object to update
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
    """
    import bpy
    import bmesh
    
    # Transform vertices to object local space
    world_matrix = np.array(obj.matrix_world, dtype=np.float32)
    inv_world = np.linalg.inv(world_matrix)
    
    num_verts = len(vertices)
    verts_4d = np.hstack([vertices, np.ones((num_verts, 1), dtype=np.float32)])
    local_verts = (verts_4d @ inv_world.T)[:, :3]
    
    # Create new mesh
    mesh = obj.data
    
    # Use bmesh for efficient mesh creation
    bm = bmesh.new()
    
    # Add vertices
    for v in local_verts:
        bm.verts.new(v)
    
    bm.verts.ensure_lookup_table()
    
    # Add faces
    for face in faces:
        try:
            bm.faces.new([bm.verts[i] for i in face])
        except ValueError:
            pass  # Skip duplicate faces
    
    # Write to mesh
    bm.to_mesh(mesh)
    bm.free()
    
    mesh.update()


def apply_joint_values_to_armature(
    armature_obj,
    joint_values: np.ndarray,
    joint_names: List[str],
    joint_mappings: Dict[str, str],
    is_position: bool = True
) -> None:
    """
    Apply motion prediction output to a Blender armature.
    
    Args:
        armature_obj: Blender armature object
        joint_values: (J, 3) or (J, 4) joint positions or quaternions
        joint_names: List of joint names from the model
        joint_mappings: Dict mapping model joint names → Blender bone names
        is_position: True for positions, False for rotations
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        print("[NeuralSim] Warning: Invalid armature object")
        return
    
    for i, model_joint in enumerate(joint_names):
        if i >= len(joint_values):
            break
            
        # Get mapped Blender bone name
        blender_bone = joint_mappings.get(model_joint)
        if not blender_bone:
            # Try direct name match
            blender_bone = model_joint
        
        pose_bone = armature_obj.pose.bones.get(blender_bone)
        if not pose_bone:
            continue
        
        if is_position:
            # Apply as location
            pose_bone.location = joint_values[i].tolist()
        else:
            # Apply as rotation (quaternion)
            pose_bone.rotation_mode = 'QUATERNION'
            if len(joint_values[i]) == 4:
                pose_bone.rotation_quaternion = joint_values[i].tolist()
            elif len(joint_values[i]) == 3:
                # Euler angles
                pose_bone.rotation_mode = 'XYZ'
                pose_bone.rotation_euler = joint_values[i].tolist()
    
    armature_obj.update_tag()


def get_mesh_bounds(obj) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get the bounding box of a mesh in world space.
    
    Args:
        obj: Blender object with mesh data
        
    Returns:
        ((min_x, min_y, min_z), (max_x, max_y, max_z))
    """
    vertices, _, _ = blender_mesh_to_numpy(obj)
    
    if len(vertices) == 0:
        return ((0, 0, 0), (1, 1, 1))
    
    min_bound = tuple(vertices.min(axis=0).tolist())
    max_bound = tuple(vertices.max(axis=0).tolist())
    
    # Add some padding
    padding = 0.1
    min_bound = tuple(v - padding for v in min_bound)
    max_bound = tuple(v + padding for v in max_bound)
    
    return (min_bound, max_bound)


def ensure_mesh_triangulated(obj) -> None:
    """
    Ensure a mesh is triangulated.
    
    Args:
        obj: Blender object with mesh data
    """
    import bpy
    import bmesh
    
    # Work in object mode
    if obj.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Triangulate
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
