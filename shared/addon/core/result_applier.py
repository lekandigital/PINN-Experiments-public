"""
Result Applier - Applies PredictionResult to Blender objects.

Handles different output formats (displacements, positions, SDF, joints)
and applies them to the appropriate Blender objects.
"""

from typing import Optional
import numpy as np


def apply_result(scene, result: "PredictionResult", caps: "BackendCapabilities") -> None:
    """
    Apply prediction result to Blender scene objects.
    
    Args:
        scene: Blender scene
        result: PredictionResult from backend
        caps: Backend capabilities to determine output format
    """
    from ..backend import OutputFormat, ModelCategory
    from .mesh_bridge import (
        numpy_displacements_to_blender,
        numpy_positions_to_blender,
        sdf_grid_to_blender_mesh,
        apply_joint_values_to_armature,
    )
    
    props = scene.neural_sim
    
    # Check for errors
    if result.confidence < 0.5:
        if result.error_message:
            print(f"[NeuralSim] Warning: {result.error_message}")
        return
    
    # Get target object
    target_obj = _get_target_for_output(scene, caps)
    if target_obj is None and caps.output_format != OutputFormat.JOINT_POSITIONS:
        print("[NeuralSim] Warning: No target object for result application")
        return
    
    # Apply based on output format
    if caps.output_format == OutputFormat.VERTEX_DISPLACEMENTS:
        _apply_displacements(target_obj, result, props)
    
    elif caps.output_format == OutputFormat.VERTEX_POSITIONS:
        _apply_positions(target_obj, result)
    
    elif caps.output_format == OutputFormat.SDF_FIELD:
        _apply_sdf(target_obj, result, props)
    
    elif caps.output_format in (OutputFormat.JOINT_POSITIONS, OutputFormat.JOINT_ANGLES):
        _apply_joints(scene, result, caps)


def _get_target_for_output(scene, caps):
    """Get the appropriate target object based on backend category."""
    import bpy
    from ..backend import ModelCategory
    
    props = scene.neural_sim
    
    if caps.category == ModelCategory.CLOTH_SIMULATION:
        obj_name = getattr(props, "cloth_object", None)
    elif caps.category == ModelCategory.BODY_DEFORMATION:
        obj_name = getattr(props, "body_object", None)
    elif caps.category == ModelCategory.MOTION_PREDICTION:
        obj_name = getattr(props, "armature_object", None)
    else:
        return None
    
    if obj_name and obj_name in scene.objects:
        return scene.objects[obj_name]
    
    return None


def _apply_displacements(target_obj, result, props):
    """Apply vertex displacements to mesh."""
    from .mesh_bridge import numpy_displacements_to_blender, blender_mesh_to_numpy
    
    if result.displacements is None:
        return
    
    # Get rest vertices (stored or current)
    if hasattr(props, "_rest_vertices") and props._rest_vertices is not None:
        rest_vertices = props._rest_vertices
    else:
        rest_vertices, _, _ = blender_mesh_to_numpy(target_obj)
        # Store for future frames (via custom property)
        target_obj["_neuralsim_rest_vertices"] = rest_vertices.tobytes()
    
    # Check dimension match
    if len(result.displacements) != len(rest_vertices):
        print(f"[NeuralSim] Warning: Displacement size mismatch: "
              f"{len(result.displacements)} vs {len(rest_vertices)} vertices")
        return
    
    # Apply scaling if available
    scale = getattr(props, "output_scale", 1.0)
    displacements = result.displacements * scale
    
    numpy_displacements_to_blender(target_obj, displacements, rest_vertices)


def _apply_positions(target_obj, result):
    """Apply absolute vertex positions to mesh."""
    from .mesh_bridge import numpy_positions_to_blender
    
    if result.positions is None:
        return
    
    numpy_positions_to_blender(target_obj, result.positions)


def _apply_sdf(target_obj, result, props):
    """Apply SDF grid via marching cubes."""
    from .mesh_bridge import sdf_grid_to_blender_mesh
    
    if result.sdf_grid is None:
        print("[NeuralSim] Warning: No SDF grid in result")
        return
    
    if result.sdf_bounds is None:
        print("[NeuralSim] Warning: No SDF bounds in result")
        return
    
    # Get smoothing settings
    smooth = getattr(props, "mesh_smoothing", True)
    smooth_iters = getattr(props, "smooth_iterations", 2)
    
    sdf_grid_to_blender_mesh(
        result.sdf_grid,
        result.sdf_bounds,
        target_obj,
        level=0.0,
        smooth=smooth,
        smooth_iterations=smooth_iters
    )


def _apply_joints(scene, result, caps):
    """Apply joint values to armature."""
    from .mesh_bridge import apply_joint_values_to_armature
    from ..backend import OutputFormat
    
    if result.joint_values is None:
        return
    
    props = scene.neural_sim
    armature_name = getattr(props, "armature_object", None)
    
    if not armature_name or armature_name not in scene.objects:
        print("[NeuralSim] Warning: No armature object set")
        return
    
    armature_obj = scene.objects[armature_name]
    
    # Get joint mapping
    joint_mappings = _get_joint_mappings(props)
    
    # Get joint names
    joint_names = result.joint_names or []
    
    # Determine if position or rotation output
    is_position = caps.output_format == OutputFormat.JOINT_POSITIONS
    
    apply_joint_values_to_armature(
        armature_obj,
        result.joint_values,
        joint_names,
        joint_mappings,
        is_position=is_position
    )


def _get_joint_mappings(props) -> dict:
    """
    Get the joint name mapping from props.
    
    Returns dict mapping model joint names → Blender bone names.
    """
    mappings = {}
    
    # Check if joint mappings are defined
    if hasattr(props, "joint_mappings"):
        for mapping in props.joint_mappings:
            if mapping.model_joint and mapping.blender_bone:
                mappings[mapping.model_joint] = mapping.blender_bone
    
    return mappings


def cache_rest_vertices(scene) -> None:
    """
    Cache rest vertices for all target objects.
    
    Called when simulation starts to capture rest pose.
    """
    from ..backend import get_manager, ModelCategory
    from .mesh_bridge import blender_mesh_to_numpy
    
    props = scene.neural_sim
    manager = get_manager()
    caps = manager.get_active_capabilities()
    
    if caps is None:
        return
    
    # Get target object
    target_obj = _get_target_for_output(scene, caps)
    if target_obj is None:
        return
    
    # Cache vertices
    vertices, _, _ = blender_mesh_to_numpy(target_obj)
    target_obj["_neuralsim_rest_vertices"] = vertices.tobytes()
    
    print(f"[NeuralSim] Cached {len(vertices)} rest vertices for {target_obj.name}")


def restore_rest_vertices(scene) -> None:
    """
    Restore mesh to rest pose from cached vertices.
    
    Called when simulation is reset.
    """
    from ..backend import get_manager, ModelCategory
    from .mesh_bridge import numpy_positions_to_blender
    
    props = scene.neural_sim
    manager = get_manager()
    caps = manager.get_active_capabilities()
    
    if caps is None:
        return
    
    target_obj = _get_target_for_output(scene, caps)
    if target_obj is None:
        return
    
    # Restore from cache
    if "_neuralsim_rest_vertices" in target_obj:
        import numpy as np
        vertices_bytes = target_obj["_neuralsim_rest_vertices"]
        vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)
        numpy_positions_to_blender(target_obj, vertices)
        print(f"[NeuralSim] Restored rest pose for {target_obj.name}")
