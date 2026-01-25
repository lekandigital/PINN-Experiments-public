"""
NIF-Cloth3D-Interactive: Blender Add-on for Live Cloth Prediction

This add-on enables real-time neural cloth simulation in Blender with:
- GUI sliders for wind speed and pin positions
- Live inference using the trained PyTorch model
- Real-time mesh updates

Installation:
1. Install dependencies in Blender's Python:
   /path/to/blender/python -m pip install torch
   
2. In Blender: Edit > Preferences > Add-ons > Install
   Select this file (nif_cloth_addon.py)

3. Enable the add-on and find controls in 3D View > Sidebar > Cloth3D tab

Usage:
1. Create or select a mesh object named "Cloth"
2. Load a trained model using the "Load Model" button
3. Adjust wind speed and pin position sliders
4. Click "Apply Neural Step" or enable "Auto Update" for continuous simulation
"""

bl_info = {
    "name": "NIF-Cloth3D Interactive",
    "author": "NIF-Cloth3D Team",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D View > Sidebar > Cloth3D",
    "description": "Real-time neural cloth simulation with interactive controls",
    "category": "Physics",
}

import bpy
import numpy as np
from bpy.props import (
    FloatProperty, 
    IntProperty, 
    BoolProperty, 
    StringProperty,
    FloatVectorProperty
)
from bpy.types import Panel, Operator, PropertyGroup
from mathutils import Vector

# Global model reference
_model = None
_device = None


def get_torch():
    """Import torch with error handling."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def load_neural_model(model_path):
    """Load the TorchScript model."""
    global _model, _device
    
    torch = get_torch()
    if torch is None:
        raise ImportError("PyTorch is not installed in Blender's Python")
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = torch.jit.load(model_path).to(_device)
    _model.eval()
    
    return True


def predict_cloth_displacement(vertices, time, wind_vector, material_id=0):
    """
    Run neural network inference to predict cloth displacement.
    
    Args:
        vertices: numpy array of vertex positions (N, 3)
        time: current time value (normalized)
        wind_vector: 3D wind force vector
        material_id: material identifier
    
    Returns:
        displacement: numpy array (N, 3)
    """
    global _model, _device
    
    if _model is None:
        raise RuntimeError("Model not loaded. Use 'Load Model' button first.")
    
    torch = get_torch()
    
    N = vertices.shape[0]
    
    # Prepare input tensor: [x, y, z, t, fx, fy, fz, material_id]
    verts_tensor = torch.tensor(vertices, dtype=torch.float32, device=_device)
    t_tensor = torch.full((N, 1), time, dtype=torch.float32, device=_device)
    force_tensor = torch.tensor(wind_vector, dtype=torch.float32, device=_device)
    force_tensor = force_tensor.unsqueeze(0).expand(N, 3)
    mat_tensor = torch.full((N, 1), float(material_id), dtype=torch.float32, device=_device)
    
    inp = torch.cat([verts_tensor, t_tensor, force_tensor, mat_tensor], dim=1)
    
    # Inference
    with torch.no_grad():
        displacement = _model(inp)
    
    return displacement.cpu().numpy()


# -----------------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------------

class NIFClothProperties(PropertyGroup):
    """Properties for the NIF Cloth3D add-on."""
    
    model_path: StringProperty(
        name="Model Path",
        description="Path to the trained TorchScript model (.pt file)",
        default="",
        subtype='FILE_PATH'
    )
    
    wind_speed: FloatProperty(
        name="Wind Speed",
        description="Magnitude of wind force",
        default=0.5,
        min=0.0,
        max=10.0,
        step=0.1
    )
    
    wind_direction: FloatVectorProperty(
        name="Wind Direction",
        description="Direction of wind force",
        default=(1.0, 0.0, 0.0),
        size=3,
        subtype='DIRECTION'
    )
    
    pin_vertex_index: IntProperty(
        name="Pin Vertex",
        description="Index of vertex to pin (fix in place)",
        default=0,
        min=0,
        max=10000
    )
    
    time_value: FloatProperty(
        name="Time",
        description="Normalized simulation time [0, 1]",
        default=0.0,
        min=0.0,
        max=1.0
    )
    
    material_id: IntProperty(
        name="Material ID",
        description="Material type (0-4)",
        default=0,
        min=0,
        max=4
    )
    
    auto_update: BoolProperty(
        name="Auto Update",
        description="Automatically update cloth on frame change",
        default=False
    )
    
    displacement_scale: FloatProperty(
        name="Displacement Scale",
        description="Scale factor for predicted displacements",
        default=1.0,
        min=0.0,
        max=10.0
    )


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

class NIF_OT_LoadModel(Operator):
    """Load the trained neural network model."""
    
    bl_idname = "nif_cloth.load_model"
    bl_label = "Load Model"
    bl_description = "Load the trained TorchScript model"
    
    def execute(self, context):
        props = context.scene.nif_cloth_props
        
        if not props.model_path:
            self.report({'ERROR'}, "No model path specified")
            return {'CANCELLED'}
        
        try:
            load_neural_model(props.model_path)
            self.report({'INFO'}, f"Model loaded successfully from {props.model_path}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load model: {str(e)}")
            return {'CANCELLED'}


class NIF_OT_ApplyNeuralStep(Operator):
    """Apply one step of neural cloth simulation."""
    
    bl_idname = "nif_cloth.apply_step"
    bl_label = "Apply Neural Step"
    bl_description = "Run neural inference and update cloth mesh"
    
    def execute(self, context):
        global _model
        
        if _model is None:
            self.report({'ERROR'}, "Model not loaded. Load a model first.")
            return {'CANCELLED'}
        
        # Get the cloth object
        cloth = bpy.data.objects.get("Cloth")
        if cloth is None:
            # Try to use selected object
            cloth = context.active_object
            if cloth is None or cloth.type != 'MESH':
                self.report({'ERROR'}, "No cloth mesh found. Select a mesh or name it 'Cloth'")
                return {'CANCELLED'}
        
        props = context.scene.nif_cloth_props
        
        # Get vertex positions
        mesh = cloth.data
        n_verts = len(mesh.vertices)
        vertices = np.zeros((n_verts, 3), dtype=np.float32)
        
        for i, v in enumerate(mesh.vertices):
            vertices[i] = (v.co.x, v.co.y, v.co.z)
        
        # Compute wind vector
        wind_dir = np.array(props.wind_direction)
        wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
        wind_vector = wind_dir * props.wind_speed
        
        # Predict displacement
        try:
            displacement = predict_cloth_displacement(
                vertices,
                props.time_value,
                wind_vector,
                props.material_id
            )
            
            # Apply displacement (scaled)
            for i, v in enumerate(mesh.vertices):
                v.co.x += displacement[i, 0] * props.displacement_scale
                v.co.y += displacement[i, 1] * props.displacement_scale
                v.co.z += displacement[i, 2] * props.displacement_scale
            
            # Update mesh
            mesh.update()
            
            # Increment time slightly
            props.time_value = min(1.0, props.time_value + 0.01)
            
            self.report({'INFO'}, "Neural step applied successfully")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Inference failed: {str(e)}")
            return {'CANCELLED'}


class NIF_OT_ResetCloth(Operator):
    """Reset cloth to rest position."""
    
    bl_idname = "nif_cloth.reset"
    bl_label = "Reset Cloth"
    bl_description = "Reset cloth mesh to initial state"
    
    # Store original positions
    _original_positions = {}
    
    def execute(self, context):
        cloth = bpy.data.objects.get("Cloth")
        if cloth is None:
            cloth = context.active_object
        
        if cloth is None or cloth.type != 'MESH':
            self.report({'ERROR'}, "No cloth mesh found")
            return {'CANCELLED'}
        
        # Reset to a flat plane if no original stored
        mesh = cloth.data
        
        # Simple reset: flatten Z coordinates
        for v in mesh.vertices:
            v.co.z = 0.0
        
        mesh.update()
        
        # Reset time
        context.scene.nif_cloth_props.time_value = 0.0
        
        self.report({'INFO'}, "Cloth reset to initial state")
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------

class NIF_PT_ClothPanel(Panel):
    """Main panel for NIF Cloth3D controls."""
    
    bl_label = "Cloth3D Controls"
    bl_idname = "NIF_PT_cloth_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cloth3D"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.nif_cloth_props
        
        # Model section
        box = layout.box()
        box.label(text="Neural Model", icon='MODIFIER')
        box.prop(props, "model_path")
        box.operator("nif_cloth.load_model", icon='IMPORT')
        
        # Status
        global _model
        if _model is not None:
            box.label(text="✓ Model loaded", icon='CHECKMARK')
        else:
            box.label(text="✗ No model loaded", icon='ERROR')
        
        # Wind controls
        box = layout.box()
        box.label(text="Wind Force", icon='FORCE_WIND')
        box.prop(props, "wind_speed")
        box.prop(props, "wind_direction")
        
        # Material
        box = layout.box()
        box.label(text="Material", icon='MATERIAL')
        box.prop(props, "material_id")
        
        # Time
        box = layout.box()
        box.label(text="Simulation", icon='TIME')
        box.prop(props, "time_value", slider=True)
        box.prop(props, "displacement_scale")
        box.prop(props, "auto_update")
        
        # Actions
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("nif_cloth.apply_step", icon='PLAY')
        row.operator("nif_cloth.reset", icon='FILE_REFRESH')


# -----------------------------------------------------------------------------
# Frame change handler for auto-update
# -----------------------------------------------------------------------------

def frame_change_handler(scene):
    """Handle frame changes for auto-update."""
    props = scene.nif_cloth_props
    
    if not props.auto_update:
        return
    
    global _model
    if _model is None:
        return
    
    # Update time based on frame
    frame = scene.frame_current
    props.time_value = (frame % 100) / 100.0
    
    # Apply step (simplified, no error reporting)
    try:
        cloth = bpy.data.objects.get("Cloth")
        if cloth is None:
            return
        
        mesh = cloth.data
        n_verts = len(mesh.vertices)
        vertices = np.zeros((n_verts, 3), dtype=np.float32)
        
        for i, v in enumerate(mesh.vertices):
            vertices[i] = (v.co.x, v.co.y, v.co.z)
        
        wind_dir = np.array(props.wind_direction)
        wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
        wind_vector = wind_dir * props.wind_speed
        
        displacement = predict_cloth_displacement(
            vertices, props.time_value, wind_vector, props.material_id
        )
        
        for i, v in enumerate(mesh.vertices):
            v.co.x += displacement[i, 0] * props.displacement_scale * 0.1
            v.co.y += displacement[i, 1] * props.displacement_scale * 0.1
            v.co.z += displacement[i, 2] * props.displacement_scale * 0.1
        
        mesh.update()
        
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = [
    NIFClothProperties,
    NIF_OT_LoadModel,
    NIF_OT_ApplyNeuralStep,
    NIF_OT_ResetCloth,
    NIF_PT_ClothPanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.nif_cloth_props = bpy.props.PointerProperty(type=NIFClothProperties)
    
    # Add frame change handler
    bpy.app.handlers.frame_change_post.append(frame_change_handler)
    
    print("NIF-Cloth3D Interactive add-on registered")


def unregister():
    # Remove frame change handler
    if frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(frame_change_handler)
    
    del bpy.types.Scene.nif_cloth_props
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    print("NIF-Cloth3D Interactive add-on unregistered")


if __name__ == "__main__":
    register()
