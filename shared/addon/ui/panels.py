"""
UI Panels for Neural Simulation addon.

Each panel shows/hides based on the active backend's capabilities.
"""

import bpy
from bpy.types import Panel


def _get_caps():
    """Get active backend capabilities, handling import errors gracefully."""
    try:
        from ..backend import get_manager
        return get_manager().get_active_capabilities()
    except:
        return None


class NEURALSIM_PT_BackendPanel(Panel):
    """Main panel for backend selection and configuration."""
    
    bl_label = "Neural Simulation"
    bl_idname = "NEURALSIM_PT_backend"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Neural Sim'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.neural_sim
        
        # Backend selection
        layout.label(text="Model Backend:", icon='GHOST_ENABLED')
        layout.prop(props, "active_backend", text="")
        
        # Show backend info if selected
        if props.active_backend != "NONE":
            caps = _get_caps() if props.is_loaded else None
            
            if caps is None:
                try:
                    from ..backend import get_manager
                    caps = get_manager().get_backend_info(props.active_backend)
                except:
                    pass
            
            if caps:
                box = layout.box()
                box.label(text=caps.display_name, icon=caps.icon)
                
                # Wrap long description
                desc_words = caps.description.split()
                line = ""
                for word in desc_words:
                    if len(line) + len(word) > 40:
                        box.label(text=line)
                        line = word
                    else:
                        line = f"{line} {word}" if line else word
                if line:
                    box.label(text=line)
                
                # Stats row
                row = box.row()
                row.label(text=f"Quality: {caps.quality_tier.title()}")
                row.label(text=f"~{caps.typical_fps:.0f} FPS")
                
                row = box.row()
                row.label(text=f"{caps.parameter_count/1000:.0f}K params")
                row.label(text=f"~{caps.typical_memory_mb:.0f} MB")
        
        layout.separator()
        
        # Checkpoint selection
        layout.label(text="Checkpoint:", icon='FILE')
        layout.prop(props, "checkpoint_path", text="")
        
        # Device selection
        layout.prop(props, "device", text="Device")
        
        layout.separator()
        
        # Load button
        row = layout.row(align=True)
        row.scale_y = 1.5
        
        if props.is_loaded:
            row.operator("neuralsim.load_backend", text="Reload Model", icon='FILE_REFRESH')
        else:
            row.operator("neuralsim.load_backend", text="Load Model", icon='IMPORT')
        
        row.operator("neuralsim.refresh_backends", text="", icon='RECOVER_LAST')
        
        # Status indicator
        if props.is_loaded:
            layout.label(text="✓ Model loaded and ready", icon='CHECKMARK')
        elif props.active_backend != "NONE":
            layout.label(text="Model not loaded", icon='ERROR')


class NEURALSIM_PT_TargetPanel(Panel):
    """Panel for selecting simulation target objects."""
    
    bl_label = "Simulation Targets"
    bl_idname = "NEURALSIM_PT_target"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Neural Sim'
    bl_parent_id = "NEURALSIM_PT_backend"
    
    @classmethod
    def poll(cls, context):
        return context.scene.neural_sim.is_loaded
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.neural_sim
        caps = _get_caps()
        
        if caps is None:
            layout.label(text="No backend loaded")
            return
        
        from ..backend import ModelCategory, InputRequirement
        
        # Show appropriate object selectors based on category
        if caps.category == ModelCategory.CLOTH_SIMULATION:
            layout.prop_search(
                props, "cloth_object",
                context.scene, "objects",
                text="Cloth Mesh",
                icon='OUTLINER_OB_MESH'
            )
            
            if InputRequirement.COLLISION_BODY in caps.input_requirements:
                layout.prop_search(
                    props, "body_object",
                    context.scene, "objects",
                    text="Body Mesh",
                    icon='OUTLINER_OB_ARMATURE'
                )
        
        elif caps.category == ModelCategory.BODY_DEFORMATION:
            layout.prop_search(
                props, "body_object",
                context.scene, "objects",
                text="Body Mesh",
                icon='OUTLINER_OB_MESH'
            )
        
        elif caps.category == ModelCategory.MOTION_PREDICTION:
            layout.prop_search(
                props, "armature_object",
                context.scene, "objects",
                text="Armature",
                icon='ARMATURE_DATA'
            )
            
            # Joint mapping info
            if props.armature_object:
                box = layout.box()
                box.label(text="Joint Mapping", icon='BONE_DATA')
                box.label(text="Configure bone mappings in addon preferences")


class NEURALSIM_PT_ForcesPanel(Panel):
    """Panel for wind, gravity, and force controls."""
    
    bl_label = "Forces"
    bl_idname = "NEURALSIM_PT_forces"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Neural Sim'
    bl_parent_id = "NEURALSIM_PT_backend"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        caps = _get_caps()
        if not caps:
            return False
        return caps.supports_wind or caps.supports_gravity or caps.supports_custom_forces
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.neural_sim
        caps = _get_caps()
        
        if caps is None:
            return
        
        # Wind controls
        if caps.supports_wind:
            box = layout.box()
            box.label(text="Wind", icon='FORCE_WIND')
            box.prop(props, "wind_strength", slider=True)
            box.prop(props, "wind_direction", text="Direction")
        
        # Gravity controls
        if caps.supports_gravity:
            box = layout.box()
            box.label(text="Gravity", icon='FORCE_FORCE')
            row = box.row()
            row.prop(props, "gravity_enabled", text="")
            sub = row.row()
            sub.enabled = props.gravity_enabled
            sub.prop(props, "gravity_strength", text="Strength")


class NEURALSIM_PT_MaterialPanel(Panel):
    """Panel for material parameter controls."""
    
    bl_label = "Material"
    bl_idname = "NEURALSIM_PT_material"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Neural Sim'
    bl_parent_id = "NEURALSIM_PT_backend"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        caps = _get_caps()
        return caps and caps.supports_material_params
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.neural_sim
        caps = _get_caps()
        
        if caps is None:
            return
        
        # Material preset
        layout.prop(props, "material_preset", text="Preset")
        
        layout.separator()
        
        # Dynamic sliders for material parameters
        for param_name in caps.material_param_names:
            prop_name = f"material_{param_name}"
            if hasattr(props, prop_name):
                layout.prop(props, prop_name, text=param_name.replace('_', ' ').title())


class NEURALSIM_PT_ResolutionPanel(Panel):
    """Panel for SDF resolution and quality controls."""
    
    bl_label = "Quality"
    bl_idname = "NEURALSIM_PT_resolution"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Neural Sim'
    bl_parent_id = "NEURALSIM_PT_backend"
    
    @classmethod
    def poll(cls, context):
        caps = _get_caps()
        return caps and caps.supports_resolution_control
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.neural_sim
        caps = _get_caps()
        
        if caps is None:
            return
        
        # Resolution slider
        layout.prop(props, "sdf_resolution", text="SDF Resolution")
        
        # Estimate vertex count
        res = props.sdf_resolution
        estimated_verts = (res ** 2) * 2  # Rough estimate
        layout.label(text=f"~{estimated_verts:,} vertices (estimated)", icon='MESH_DATA')
        
        # Adaptive resolution
        layout.prop(props, "adaptive_resolution", text="Adaptive (faster playback)")
        
        layout.separator()
        
        # Mesh processing
        layout.label(text="Post-Processing:", icon='MOD_SMOOTH')
        layout.prop(props, "mesh_smoothing", text="Smooth Mesh")
        if props.mesh_smoothing:
            layout.prop(props, "smooth_iterations", text="Iterations")


class NEURALSIM_PT_PlaybackPanel(Panel):
    """Panel for play/pause/reset and performance display."""
    
    bl_label = "Playback"
    bl_idname = "NEURALSIM_PT_playback"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Neural Sim'
    bl_parent_id = "NEURALSIM_PT_backend"
    
    @classmethod
    def poll(cls, context):
        return context.scene.neural_sim.is_loaded
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.neural_sim
        caps = _get_caps()
        
        # Play/Pause/Reset controls
        row = layout.row(align=True)
        row.scale_y = 1.3
        
        if props.is_playing:
            row.operator("neuralsim.pause", text="Pause", icon='PAUSE')
        else:
            row.operator("neuralsim.play", text="Play", icon='PLAY')
        
        row.operator("neuralsim.reset", text="Reset", icon='REW')
        
        # Step controls
        row = layout.row(align=True)
        row.operator("neuralsim.step_back", text="", icon='PREV_KEYFRAME')
        row.operator("neuralsim.step_forward", text="", icon='NEXT_KEYFRAME')
        
        layout.separator()
        
        # Bake to keyframes
        layout.operator("neuralsim.bake_to_keyframes", text="Bake to Keyframes", icon='KEY_HLT')
        
        layout.separator()
        
        # Performance display
        box = layout.box()
        box.label(text="Performance", icon='TIME')
        
        col = box.column(align=True)
        col.label(text=f"Inference: {props.last_inference_ms:.1f} ms")
        col.label(text=f"FPS: {props.current_fps:.0f}")
        col.label(text=f"Frame: {props.current_frame}")
        
        # Stability warning for sequential models
        if caps and caps.needs_sequential_frames:
            progress = props.current_frame / caps.max_stable_frames
            if progress > 0.8:
                box.label(text="⚠ Approaching stability limit", icon='ERROR')


# Export all panel classes
__all__ = [
    "NEURALSIM_PT_BackendPanel",
    "NEURALSIM_PT_TargetPanel",
    "NEURALSIM_PT_ForcesPanel",
    "NEURALSIM_PT_MaterialPanel",
    "NEURALSIM_PT_ResolutionPanel",
    "NEURALSIM_PT_PlaybackPanel",
]
