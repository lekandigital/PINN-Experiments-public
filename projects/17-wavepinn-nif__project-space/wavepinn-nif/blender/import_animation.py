"""
Blender Import Script for WavePINN Wave Animation

Creates a complete animated wave surface from displacement maps.

Usage:
1. Open Blender (any version 2.8+)
2. Go to Scripting workspace (top menu)
3. Click "New" to create a new text block
4. Paste this entire script
5. Click "Run Script" or press Alt+P
6. Press Spacebar to play animation

The script will:
- Create an animated wave surface
- Set up camera and lighting
- Apply water-like material
- Configure render settings
"""

import bpy
import os
import json

# ============================================================================
# CONFIGURATION - Update this path to your blender_export folder
# ============================================================================
EXPORT_PATH = "/Users/lekanadeyeri/Dev/PINN-Experiments/projects/wavepinn-nif__project-space/wavepinn-nif/blender_export"
# ============================================================================


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def setup_wave_animation():
    """Create animated wave surface from exported displacement maps."""

    # Load metadata
    metadata_path = os.path.join(EXPORT_PATH, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata.json not found at {metadata_path}")
        print("Please update EXPORT_PATH in this script.")
        return False

    with open(metadata_path) as f:
        meta = json.load(f)

    resolution = meta["resolution"]
    n_frames = meta["n_frames"]
    fps = meta["fps"]
    strength = meta["displacement_scale"]

    print("\n" + "=" * 60)
    print("WavePINN Animation Setup")
    print("=" * 60)
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Frames: {n_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {n_frames/fps:.1f} seconds")
    print("=" * 60 + "\n")

    # Clear existing scene
    clear_scene()

    # Configure scene
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.fps = fps
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # Create subdivided grid mesh
    print("Creating wave surface mesh...")
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=resolution - 1,
        y_subdivisions=resolution - 1,
        size=4.0,
        location=(0, 0, 0)
    )
    grid = bpy.context.active_object
    grid.name = "WaveSurface"

    # UV unwrap for displacement texture mapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create displacement texture
    print("Setting up displacement texture...")
    tex = bpy.data.textures.new("WaveDisplacement", type='IMAGE')

    # Load image sequence
    images_path = os.path.join(EXPORT_PATH, "images")
    first_image = os.path.join(images_path, "wave_0000.png")

    if not os.path.exists(first_image):
        print(f"ERROR: Images not found at {images_path}")
        return False

    img = bpy.data.images.load(first_image)
    img.source = 'SEQUENCE'
    tex.image = img
    tex.image_user.frame_duration = n_frames
    tex.image_user.frame_start = 1
    tex.image_user.use_auto_refresh = True
    tex.image_user.use_cyclic = True

    # Add displacement modifier
    disp = grid.modifiers.new(name="WaveDisplace", type='DISPLACE')
    disp.texture = tex
    disp.texture_coords = 'UV'
    disp.strength = strength * 100  # Scale up for visibility
    disp.mid_level = 0.5
    disp.direction = 'Z'

    # Add subdivision surface for smoother waves
    subsurf = grid.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 1
    subsurf.render_levels = 2

    # Create ocean-like material
    print("Creating water material...")
    mat = bpy.data.materials.new(name="WaveMaterial")
    mat.use_nodes = True
    grid.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Material output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Principled BSDF (main shader)
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (200, 0)
    principled.inputs['Base Color'].default_value = (0.05, 0.25, 0.6, 1.0)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.15
    principled.inputs['IOR'].default_value = 1.33
    principled.inputs['Alpha'].default_value = 0.85

    # Enable transmission for glass-like effect
    if 'Transmission' in principled.inputs:
        principled.inputs['Transmission'].default_value = 0.3

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Enable material transparency
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'

    # Add camera
    print("Setting up camera...")
    bpy.ops.object.camera_add(location=(5, -5, 4))
    cam = bpy.context.active_object
    cam.name = "WaveCamera"
    cam.rotation_euler = (1.1, 0, 0.8)
    scene.camera = cam

    # Add tracking constraint to keep camera focused on center
    track = cam.constraints.new(type='TRACK_TO')
    track.target = grid
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Add sun light
    print("Setting up lighting...")
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 6))
    sun = bpy.context.active_object
    sun.name = "SunLight"
    sun.data.energy = 4.0
    sun.rotation_euler = (0.8, 0.3, 0.6)

    # Add fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 4))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 200.0
    fill.data.size = 5.0

    # Set viewport shading to material preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    space.clip_end = 1000

    # Add environment HDRI-like world
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs['Color'].default_value = (0.5, 0.7, 1.0, 1.0)
        bg.inputs['Strength'].default_value = 0.8

    # Set to first frame
    scene.frame_set(1)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"Animation: {n_frames} frames at {fps} fps")
    print(f"Duration: {n_frames/fps:.1f} seconds")
    print("")
    print("Controls:")
    print("  - SPACE: Play/pause animation")
    print("  - LEFT/RIGHT: Step through frames")
    print("  - 0 (numpad): Camera view")
    print("  - Z: Toggle wireframe")
    print("")
    print("To render animation:")
    print("  1. Output Properties -> Output path")
    print("  2. Render -> Render Animation (Ctrl+F12)")
    print("=" * 60 + "\n")

    return True


# Run the setup
if __name__ == "__main__":
    success = setup_wave_animation()
    if success:
        print("WavePINN animation ready to play!")
