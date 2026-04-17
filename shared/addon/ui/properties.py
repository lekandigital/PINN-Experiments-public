"""
Neural Simulation Properties - Blender property definitions.

Defines all properties used by the addon's UI panels and operators.
These properties are registered on bpy.types.Scene.
"""

import bpy
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    FloatVectorProperty,
    EnumProperty,
    CollectionProperty,
    PointerProperty,
)


def _get_backend_items(self, context):
    """
    Dynamic enum items for backend selection.
    
    Called by Blender to populate the backend dropdown.
    """
    items = [("NONE", "Select Model...", "Choose a neural simulation backend", 0)]
    
    try:
        from ..backend import get_manager
        manager = get_manager()
        backends = manager.get_all_backend_info()
        
        for i, (name, caps) in enumerate(backends.items(), start=1):
            description = caps.description[:80] + "..." if len(caps.description) > 80 else caps.description
            items.append((name, name, description, caps.icon, i))
    except Exception as e:
        print(f"[NeuralSim] Error getting backends: {e}")
    
    return items


def _get_checkpoint_items(self, context):
    """Dynamic enum items for checkpoint selection."""
    items = [("NONE", "Select Checkpoint...", "Choose a model checkpoint", 0)]
    
    try:
        from ..backend import get_manager
        manager = get_manager()
        
        if context.scene.neural_sim.active_backend != "NONE":
            caps = manager.get_backend_info(context.scene.neural_sim.active_backend)
            if caps:
                # Find available checkpoints
                backend_class = manager._backends.get(context.scene.neural_sim.active_backend)
                if backend_class:
                    temp = backend_class()
                    checkpoints = temp.get_available_checkpoints()
                    for i, path in enumerate(checkpoints[:10], start=1):  # Limit to 10
                        name = path.split("/")[-1]
                        items.append((path, name, path, i))
    except Exception as e:
        print(f"[NeuralSim] Error getting checkpoints: {e}")
    
    return items


def _get_device_items(self, context):
    """Dynamic enum items for device selection."""
    items = [
        ("auto", "Auto", "Automatically select best available device"),
        ("cpu", "CPU", "Run on CPU (slower but always available)"),
    ]
    
    try:
        import torch
        if torch.cuda.is_available():
            items.append(("cuda", "CUDA", "Run on NVIDIA GPU"))
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                items.append((f"cuda:{i}", f"CUDA:{i} ({name})", f"Run on GPU {i}"))
    except ImportError:
        pass
    
    return items


class NeuralSimJointMapping(bpy.types.PropertyGroup):
    """Property group for joint name mapping."""
    
    model_joint: StringProperty(
        name="Model Joint",
        description="Joint name from the motion model",
        default=""
    )
    
    blender_bone: StringProperty(
        name="Blender Bone",
        description="Corresponding Blender armature bone name",
        default=""
    )


class NeuralSimProperties(bpy.types.PropertyGroup):
    """Main property group for Neural Simulation addon."""
    
    # Backend selection
    active_backend: EnumProperty(
        name="Backend",
        description="Neural simulation backend to use",
        items=_get_backend_items,
    )
    
    checkpoint_path: StringProperty(
        name="Checkpoint",
        description="Path to model checkpoint file",
        default="",
        subtype='FILE_PATH',
    )
    
    device: EnumProperty(
        name="Device",
        description="Device to run inference on",
        items=_get_device_items,
    )
    
    is_loaded: BoolProperty(
        name="Is Loaded",
        description="Whether a backend is currently loaded",
        default=False,
    )
    
    # Target objects
    cloth_object: StringProperty(
        name="Cloth Object",
        description="Cloth mesh to simulate",
        default="",
    )
    
    body_object: StringProperty(
        name="Body Object",
        description="Body mesh for collision",
        default="",
    )
    
    armature_object: StringProperty(
        name="Armature",
        description="Armature for motion prediction",
        default="",
    )
    
    # Force parameters
    wind_strength: FloatProperty(
        name="Wind Strength",
        description="Wind force magnitude",
        default=0.0,
        min=0.0,
        max=100.0,
        soft_max=10.0,
    )
    
    wind_direction: FloatVectorProperty(
        name="Wind Direction",
        description="Wind direction vector",
        default=(1.0, 0.0, 0.0),
        size=3,
        subtype='DIRECTION',
    )
    
    gravity_enabled: BoolProperty(
        name="Enable Gravity",
        description="Apply gravity to simulation",
        default=True,
    )
    
    gravity_strength: FloatProperty(
        name="Gravity Strength",
        description="Gravity magnitude (m/s²)",
        default=9.81,
        min=0.0,
        max=100.0,
    )
    
    # Material parameters (base - others added dynamically)
    material_preset: EnumProperty(
        name="Material Preset",
        description="Predefined material settings",
        items=[
            ("custom", "Custom", "Custom material settings"),
            ("silk", "Silk", "Light, flowing silk"),
            ("cotton", "Cotton", "Medium weight cotton"),
            ("denim", "Denim", "Heavy, stiff denim"),
            ("leather", "Leather", "Thick leather"),
            ("chiffon", "Chiffon", "Ultra-light chiffon"),
        ],
        default="custom",
    )
    
    # Dynamic material params - defaults that may be overridden
    material_stiffness: FloatProperty(
        name="Stiffness",
        description="Material stiffness",
        default=50.0,
        min=0.1,
        max=200.0,
    )
    
    material_density: FloatProperty(
        name="Density",
        description="Material density",
        default=0.3,
        min=0.01,
        max=5.0,
    )
    
    material_damping: FloatProperty(
        name="Damping",
        description="Material damping",
        default=0.1,
        min=0.0,
        max=1.0,
    )
    
    material_type: FloatProperty(
        name="Material Type",
        description="Material type ID (for legacy backend)",
        default=0.0,
        min=0.0,
        max=4.0,
    )
    
    # SDF / Resolution
    sdf_resolution: IntProperty(
        name="SDF Resolution",
        description="Resolution of SDF grid for marching cubes",
        default=128,
        min=32,
        max=256,
    )
    
    adaptive_resolution: BoolProperty(
        name="Adaptive Resolution",
        description="Use lower resolution during playback for speed",
        default=False,
    )
    
    # Mesh processing
    mesh_smoothing: BoolProperty(
        name="Mesh Smoothing",
        description="Apply Laplacian smoothing to extracted mesh",
        default=True,
    )
    
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Number of smoothing iterations",
        default=2,
        min=0,
        max=10,
    )
    
    # Output scaling
    output_scale: FloatProperty(
        name="Output Scale",
        description="Scale factor for displacement output",
        default=1.0,
        min=0.0,
        max=10.0,
    )
    
    # Playback state
    is_playing: BoolProperty(
        name="Is Playing",
        description="Whether simulation is playing",
        default=False,
    )
    
    current_frame: IntProperty(
        name="Current Frame",
        description="Current simulation frame",
        default=0,
    )
    
    # Performance display
    last_inference_ms: FloatProperty(
        name="Last Inference Time",
        description="Time for last inference in milliseconds",
        default=0.0,
    )
    
    current_fps: FloatProperty(
        name="Current FPS",
        description="Current frames per second",
        default=0.0,
    )
    
    # Joint mappings for motion backend
    joint_mappings: CollectionProperty(
        type=NeuralSimJointMapping,
        name="Joint Mappings",
        description="Mapping from model joints to Blender bones",
    )
    
    # Bake settings
    bake_start_frame: IntProperty(
        name="Start Frame",
        description="Start frame for baking",
        default=1,
    )
    
    bake_end_frame: IntProperty(
        name="End Frame",
        description="End frame for baking",
        default=250,
    )


def register_properties():
    """Register properties with Blender."""
    bpy.utils.register_class(NeuralSimJointMapping)
    bpy.utils.register_class(NeuralSimProperties)
    bpy.types.Scene.neural_sim = PointerProperty(type=NeuralSimProperties)


def unregister_properties():
    """Unregister properties from Blender."""
    del bpy.types.Scene.neural_sim
    bpy.utils.unregister_class(NeuralSimProperties)
    bpy.utils.unregister_class(NeuralSimJointMapping)
