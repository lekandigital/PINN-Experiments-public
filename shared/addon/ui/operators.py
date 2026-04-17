"""
Operators for Neural Simulation addon.

Contains all Blender operators for loading backends, playback control,
and baking simulations to keyframes.
"""

import bpy
from bpy.types import Operator


class NEURALSIM_OT_LoadBackend(Operator):
    """Load the selected neural simulation backend."""
    
    bl_idname = "neuralsim.load_backend"
    bl_label = "Load Backend"
    bl_description = "Load the selected neural simulation backend"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ..backend import get_manager
        from ..core import register_frame_handler
        
        props = context.scene.neural_sim
        
        # Check backend selection
        if props.active_backend == "NONE":
            self.report({'ERROR'}, "Please select a backend first")
            return {'CANCELLED'}
        
        # Check checkpoint
        if not props.checkpoint_path:
            # Try to get recommended checkpoint
            manager = get_manager()
            caps = manager.get_backend_info(props.active_backend)
            if caps:
                backend_class = manager._backends.get(props.active_backend)
                if backend_class:
                    temp = backend_class()
                    recommended = temp.get_recommended_checkpoint()
                    if recommended:
                        props.checkpoint_path = recommended
                    else:
                        self.report({'ERROR'}, "Please specify a checkpoint file")
                        return {'CANCELLED'}
        
        # Load backend
        try:
            manager = get_manager()
            manager.activate_backend(
                props.active_backend,
                props.checkpoint_path,
                props.device
            )
            props.is_loaded = True
            
            # Register frame handler
            register_frame_handler()
            
            self.report({'INFO'}, f"Loaded {props.active_backend}")
            return {'FINISHED'}
            
        except FileNotFoundError as e:
            self.report({'ERROR'}, f"Checkpoint not found: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load backend: {e}")
            return {'CANCELLED'}


class NEURALSIM_OT_RefreshBackends(Operator):
    """Refresh the list of available backends."""
    
    bl_idname = "neuralsim.refresh_backends"
    bl_label = "Refresh Backends"
    bl_description = "Scan for available neural simulation backends"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from ..backend import get_manager
        
        manager = get_manager()
        backends = manager.discover_backends(force_refresh=True)
        
        self.report({'INFO'}, f"Found {len(backends)} backends")
        return {'FINISHED'}


class NEURALSIM_OT_Play(Operator):
    """Start simulation playback."""
    
    bl_idname = "neuralsim.play"
    bl_label = "Play"
    bl_description = "Start neural simulation playback"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.neural_sim
        
        if not props.is_loaded:
            self.report({'WARNING'}, "No backend loaded")
            return {'CANCELLED'}
        
        props.is_playing = True
        
        # Start Blender's animation playback
        bpy.ops.screen.animation_play()
        
        return {'FINISHED'}


class NEURALSIM_OT_Pause(Operator):
    """Pause simulation playback."""
    
    bl_idname = "neuralsim.pause"
    bl_label = "Pause"
    bl_description = "Pause neural simulation playback"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.neural_sim
        props.is_playing = False
        
        # Stop Blender's animation playback
        bpy.ops.screen.animation_cancel()
        
        return {'FINISHED'}


class NEURALSIM_OT_Reset(Operator):
    """Reset simulation to frame 0."""
    
    bl_idname = "neuralsim.reset"
    bl_label = "Reset"
    bl_description = "Reset simulation to initial state"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ..backend import get_manager
        from ..core import reset_simulation_state
        from ..core.result_applier import restore_rest_vertices
        
        props = context.scene.neural_sim
        
        # Stop playback
        props.is_playing = False
        bpy.ops.screen.animation_cancel()
        
        # Reset backend state
        manager = get_manager()
        manager.reset()
        
        # Reset frame handler state
        reset_simulation_state()
        
        # Restore mesh to rest pose
        restore_rest_vertices(context.scene)
        
        # Go to start frame
        context.scene.frame_set(context.scene.frame_start)
        
        self.report({'INFO'}, "Simulation reset")
        return {'FINISHED'}


class NEURALSIM_OT_StepForward(Operator):
    """Step simulation forward one frame."""
    
    bl_idname = "neuralsim.step_forward"
    bl_label = "Step Forward"
    bl_description = "Step simulation forward one frame"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        context.scene.frame_set(context.scene.frame_current + 1)
        return {'FINISHED'}


class NEURALSIM_OT_StepBack(Operator):
    """Step simulation backward one frame."""
    
    bl_idname = "neuralsim.step_back"
    bl_label = "Step Back"
    bl_description = "Step simulation backward one frame"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        context.scene.frame_set(context.scene.frame_current - 1)
        return {'FINISHED'}


class NEURALSIM_OT_BakeToKeyframes(Operator):
    """Bake neural simulation to keyframes."""
    
    bl_idname = "neuralsim.bake_to_keyframes"
    bl_label = "Bake Neural Simulation"
    bl_description = "Run simulation and bake results to keyframes"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from ..backend import get_manager, OutputFormat
        from ..core.request_builder import build_request
        from ..core.result_applier import apply_result, cache_rest_vertices
        
        scene = context.scene
        props = scene.neural_sim
        
        if not props.is_loaded:
            self.report({'ERROR'}, "No backend loaded")
            return {'CANCELLED'}
        
        manager = get_manager()
        caps = manager.get_active_capabilities()
        
        if caps is None:
            self.report({'ERROR'}, "Backend capabilities unavailable")
            return {'CANCELLED'}
        
        start_frame = scene.frame_start
        end_frame = scene.frame_end
        total_frames = end_frame - start_frame + 1
        
        # Cache rest vertices
        cache_rest_vertices(scene)
        
        # Reset simulation
        manager.reset()
        state = None
        
        # Progress tracking
        wm = context.window_manager
        wm.progress_begin(0, total_frames)
        
        # Stop playback during bake
        props.is_playing = False
        
        for i, frame in enumerate(range(start_frame, end_frame + 1)):
            scene.frame_set(frame)
            
            # Run prediction
            request = build_request(scene, frame, state)
            result = manager.predict(request)
            state = result.state
            
            # Apply result
            apply_result(scene, result, caps)
            
            # Insert keyframes based on output type
            self._insert_keyframes(scene, frame, caps)
            
            wm.progress_update(i)
        
        wm.progress_end()
        
        self.report({'INFO'}, f"Baked {total_frames} frames")
        return {'FINISHED'}
    
    def _insert_keyframes(self, scene, frame, caps):
        """Insert keyframes for the current frame."""
        from ..backend import OutputFormat
        
        props = scene.neural_sim
        
        if caps.output_format in (OutputFormat.VERTEX_DISPLACEMENTS, OutputFormat.VERTEX_POSITIONS):
            # Get target mesh
            obj_name = props.cloth_object or props.body_object
            if obj_name and obj_name in scene.objects:
                obj = scene.objects[obj_name]
                
                # Insert shape key or vertex animation
                # For now, use shape keys
                self._insert_mesh_keyframe(obj, frame)
        
        elif caps.output_format == OutputFormat.SDF_FIELD:
            # SDF creates new mesh topology each frame
            # Use mesh sequence or shape keys with matching topology
            obj_name = props.cloth_object
            if obj_name and obj_name in scene.objects:
                obj = scene.objects[obj_name]
                self._insert_mesh_keyframe(obj, frame)
        
        elif caps.output_format in (OutputFormat.JOINT_POSITIONS, OutputFormat.JOINT_ANGLES):
            # Keyframe armature bones
            obj_name = props.armature_object
            if obj_name and obj_name in scene.objects:
                armature = scene.objects[obj_name]
                for bone in armature.pose.bones:
                    bone.keyframe_insert(data_path="location", frame=frame)
                    bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    
    def _insert_mesh_keyframe(self, obj, frame):
        """Insert a shape key for the mesh at the given frame."""
        mesh = obj.data
        
        # Ensure basis shape key exists
        if mesh.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        
        # Add or update evaluation shape key
        eval_key = mesh.shape_keys.key_blocks.get("NeuralSim_Eval")
        if eval_key is None:
            eval_key = obj.shape_key_add(name="NeuralSim_Eval", from_mix=False)
        
        # Copy current vertex positions to shape key
        for i, vert in enumerate(mesh.vertices):
            eval_key.data[i].co = vert.co.copy()
        
        # Keyframe the shape key value
        eval_key.value = 1.0
        eval_key.keyframe_insert(data_path="value", frame=frame)


# Export all operator classes
__all__ = [
    "NEURALSIM_OT_LoadBackend",
    "NEURALSIM_OT_RefreshBackends",
    "NEURALSIM_OT_Play",
    "NEURALSIM_OT_Pause",
    "NEURALSIM_OT_Reset",
    "NEURALSIM_OT_StepForward",
    "NEURALSIM_OT_StepBack",
    "NEURALSIM_OT_BakeToKeyframes",
]
