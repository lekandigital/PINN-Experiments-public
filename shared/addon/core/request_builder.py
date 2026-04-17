"""
Request Builder - Constructs PredictionRequest from Blender scene state.

Reads the current scene properties, mesh data, and simulation state
to build a request object for the backend.
"""

from typing import Optional
import numpy as np


def build_request(scene, frame: int, state: Optional["SimulationState"] = None):
    """
    Build a PredictionRequest from Blender scene state.
    
    Args:
        scene: Blender scene
        frame: Current frame number
        state: Previous simulation state (or None for first frame)
        
    Returns:
        PredictionRequest populated with scene data
    """
    from ..backend import PredictionRequest, get_manager, InputRequirement
    from .mesh_bridge import blender_mesh_to_numpy, get_vertex_normals, get_mesh_bounds
    
    props = scene.neural_sim
    manager = get_manager()
    caps = manager.get_active_capabilities()
    
    if caps is None:
        return PredictionRequest(frame=frame, state=state)
    
    # Compute time values
    fps = scene.render.fps
    current_time = frame / fps
    delta_time = 1.0 / fps
    
    # Start building request
    request = PredictionRequest(
        time=current_time,
        delta_time=delta_time,
        frame=frame,
        state=state,
    )
    
    # Add mesh data if required
    if InputRequirement.MESH_VERTICES in caps.input_requirements:
        target_obj = _get_target_object(scene, caps)
        if target_obj is not None:
            vertices, faces, edges = blender_mesh_to_numpy(target_obj)
            request.vertices = vertices
            
            if InputRequirement.MESH_TOPOLOGY in caps.input_requirements:
                request.faces = faces
                request.edges = edges
            
            # Get normals if available
            try:
                request.vertex_normals = get_vertex_normals(target_obj)
            except:
                pass
    
    # Add forces if supported
    if InputRequirement.FORCES in caps.input_requirements:
        if caps.supports_wind and hasattr(props, "wind_strength"):
            wind_dir = np.array(props.wind_direction, dtype=np.float32)
            wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
            request.wind_velocity = wind_dir * props.wind_strength
        
        if caps.supports_gravity and hasattr(props, "gravity_enabled"):
            if props.gravity_enabled:
                gravity_strength = getattr(props, "gravity_strength", 9.81)
                request.gravity = np.array([0, 0, -gravity_strength], dtype=np.float32)
    
    # Add material parameters
    if InputRequirement.MATERIAL_PARAMS in caps.input_requirements:
        if caps.supports_material_params:
            request.material_params = {}
            for param_name in caps.material_param_names:
                prop_name = f"material_{param_name}"
                if hasattr(props, prop_name):
                    request.material_params[param_name] = getattr(props, prop_name)
                else:
                    request.material_params[param_name] = caps.material_param_defaults.get(param_name, 0.0)
    
    # Add SDF query configuration
    if InputRequirement.QUERY_POINTS in caps.input_requirements:
        if hasattr(props, "sdf_resolution"):
            request.query_resolution = props.sdf_resolution
        else:
            request.query_resolution = caps.default_resolution
        
        # Get bounds from target object or use defaults
        target_obj = _get_target_object(scene, caps)
        if target_obj is not None:
            request.query_bounds = get_mesh_bounds(target_obj)
        else:
            request.query_bounds = ((-1.5, -1.5, -0.5), (1.5, 1.5, 2.5))
    
    # Add collision body if required
    if InputRequirement.COLLISION_BODY in caps.input_requirements:
        body_obj = _get_body_object(scene)
        if body_obj is not None:
            coll_verts, coll_faces, _ = blender_mesh_to_numpy(body_obj)
            request.collision_vertices = coll_verts
            request.collision_faces = coll_faces
    
    # Add skeleton data for motion backends
    if InputRequirement.SKELETON_POSE in caps.input_requirements:
        armature_obj = _get_armature_object(scene)
        if armature_obj is not None:
            joint_positions, joint_rotations = _get_armature_pose(armature_obj)
            request.joint_positions = joint_positions
            request.joint_rotations = joint_rotations
    
    return request


def _get_target_object(scene, caps):
    """Get the primary target object based on backend category."""
    import bpy
    from ..backend import ModelCategory
    
    props = scene.neural_sim
    
    if caps.category == ModelCategory.CLOTH_SIMULATION:
        obj_name = getattr(props, "cloth_object", None)
    elif caps.category == ModelCategory.BODY_DEFORMATION:
        obj_name = getattr(props, "body_object", None)
    else:
        return None
    
    if obj_name and obj_name in scene.objects:
        return scene.objects[obj_name]
    
    return None


def _get_body_object(scene):
    """Get the collision body object."""
    import bpy
    
    props = scene.neural_sim
    obj_name = getattr(props, "body_object", None)
    
    if obj_name and obj_name in scene.objects:
        return scene.objects[obj_name]
    
    return None


def _get_armature_object(scene):
    """Get the armature object for motion backends."""
    import bpy
    
    props = scene.neural_sim
    obj_name = getattr(props, "armature_object", None)
    
    if obj_name and obj_name in scene.objects:
        obj = scene.objects[obj_name]
        if obj.type == 'ARMATURE':
            return obj
    
    return None


def _get_armature_pose(armature_obj):
    """
    Extract current pose from an armature.
    
    Returns:
        Tuple of (joint_positions, joint_rotations)
    """
    import numpy as np
    
    positions = []
    rotations = []
    
    for bone in armature_obj.pose.bones:
        # Get world-space head position
        world_pos = armature_obj.matrix_world @ bone.head
        positions.append([world_pos.x, world_pos.y, world_pos.z])
        
        # Get rotation as quaternion
        rot = bone.rotation_quaternion
        rotations.append([rot.w, rot.x, rot.y, rot.z])
    
    return (
        np.array(positions, dtype=np.float32),
        np.array(rotations, dtype=np.float32)
    )
