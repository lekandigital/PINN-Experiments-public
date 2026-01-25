#!/usr/bin/env python3
"""
Create WavePINN Blender Animation File

This script creates a complete .blend file with the wave animation.
Run from command line with Blender:

    blender --background --python create_blend_file.py

Or on macOS with Blender installed:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python create_blend_file.py

The script will:
1. Set up the animated wave surface
2. Configure materials, lighting, and camera
3. Save as wavepinn_animation.blend
"""

import bpy
import os
import json
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXPORT_PATH = os.path.join(PROJECT_DIR, "blender_export")
OUTPUT_BLEND = os.path.join(PROJECT_DIR, "wavepinn_animation.blend")


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

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

    metadata_path = os.path.join(EXPORT_PATH, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata.json not found at {metadata_path}")
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
    print(f"Export path: {EXPORT_PATH}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Frames: {n_frames}")
    print(f"FPS: {fps}")
    print("=" * 60 + "\n")

    clear_scene()

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.fps = fps
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # Create grid
    print("Creating wave surface...")
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=resolution - 1,
        y_subdivisions=resolution - 1,
        size=4.0,
        location=(0, 0, 0)
    )
    grid = bpy.context.active_object
    grid.name = "WaveSurface"

    # UV unwrap
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Displacement texture
    print("Setting up displacement...")
    tex = bpy.data.textures.new("WaveDisplacement", type='IMAGE')

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

    # Displace modifier
    disp = grid.modifiers.new(name="WaveDisplace", type='DISPLACE')
    disp.texture = tex
    disp.texture_coords = 'UV'
    disp.strength = strength * 100
    disp.mid_level = 0.5
    disp.direction = 'Z'

    # Subdivision
    subsurf = grid.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 1
    subsurf.render_levels = 2

    # Material
    print("Creating material...")
    mat = bpy.data.materials.new(name="WaveMaterial")
    mat.use_nodes = True
    grid.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (200, 0)
    principled.inputs['Base Color'].default_value = (0.05, 0.25, 0.6, 1.0)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.15
    principled.inputs['IOR'].default_value = 1.33
    principled.inputs['Alpha'].default_value = 0.85

    if 'Transmission' in principled.inputs:
        principled.inputs['Transmission'].default_value = 0.3

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'

    # Camera
    print("Setting up camera...")
    bpy.ops.object.camera_add(location=(5, -5, 4))
    cam = bpy.context.active_object
    cam.name = "WaveCamera"
    cam.rotation_euler = (1.1, 0, 0.8)
    scene.camera = cam

    track = cam.constraints.new(type='TRACK_TO')
    track.target = grid
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Lights
    print("Setting up lighting...")
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 6))
    sun = bpy.context.active_object
    sun.name = "SunLight"
    sun.data.energy = 4.0
    sun.rotation_euler = (0.8, 0.3, 0.6)

    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 4))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 200.0
    fill.data.size = 5.0

    # World
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs['Color'].default_value = (0.5, 0.7, 1.0, 1.0)
        bg.inputs['Strength'].default_value = 0.8

    scene.frame_set(1)

    return True


def main():
    print("\n" + "=" * 60)
    print("WavePINN Blender File Generator")
    print("=" * 60 + "\n")

    if setup_wave_animation():
        # Save the blend file
        print(f"\nSaving to: {OUTPUT_BLEND}")
        bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
        print(f"Saved: {OUTPUT_BLEND}")

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Open in Blender: {OUTPUT_BLEND}")
        print("Press SPACE to play animation")
        print("=" * 60 + "\n")
        return 0
    else:
        print("ERROR: Setup failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
