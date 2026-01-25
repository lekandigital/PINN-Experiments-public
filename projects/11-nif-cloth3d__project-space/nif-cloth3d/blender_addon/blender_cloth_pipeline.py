"""
NIF-Cloth3D-Interactive: Blender Cloth Simulation Pipeline

This script runs inside Blender to:
1. Simulate cloth with wind and pinned corners
2. Export mesh sequences
3. Convert to SDF voxel grids
4. Save metadata (forces, wind, pins) in JSON

Usage:
    blender --background --python blender_cloth_pipeline.py
    blender --background --python blender_cloth_pipeline.py -- --output /path/to/output
"""

import bpy
import json
import os
import sys
import argparse
from pathlib import Path
import mathutils

# Parse command line arguments after --
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="/tmp/cloth_sim", help="Output directory")
parser.add_argument("--frames", type=int, default=50, help="Number of frames to simulate")
parser.add_argument("--resolution", type=int, default=20, help="Cloth subdivision resolution")
parser.add_argument("--wind-strength", type=float, default=20.0, help="Wind force strength")
args, _ = parser.parse_known_args(argv)


def setup_scene():
    """Clear scene and set up basic configuration."""
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = args.frames
    bpy.context.scene.frame_current = 1


def create_cloth():
    """Create a subdivided plane with cloth physics."""
    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 2))
    cloth = bpy.context.active_object
    cloth.name = "Cloth"
    
    # Subdivide for more detail
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=args.resolution)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add cloth modifier
    bpy.ops.object.modifier_add(type='CLOTH')
    cloth_mod = cloth.modifiers['Cloth']
    
    # Configure cloth physics
    cloth_mod.settings.quality = 5
    cloth_mod.settings.mass = 0.3
    cloth_mod.settings.air_damping = 0.5
    cloth_mod.settings.tension_stiffness = 15
    cloth_mod.settings.compression_stiffness = 15
    cloth_mod.settings.shear_stiffness = 15
    cloth_mod.settings.bending_stiffness = 0.5
    
    # Create pin group for top vertices
    vg = cloth.vertex_groups.new(name="PinGroup")
    
    # Pin top row of vertices (y > 0.9 in local coords)
    pinned_indices = []
    for v in cloth.data.vertices:
        if v.co.y > 0.9:
            vg.add([v.index], 1.0, 'ADD')
            pinned_indices.append(v.index)
    
    cloth_mod.settings.vertex_group_mass = "PinGroup"
    
    return cloth, pinned_indices


def create_wind():
    """Create wind force field."""
    bpy.ops.object.effector_add(type='WIND', location=(0, -3, 2))
    wind = bpy.context.active_object
    wind.name = "Wind"
    
    wind.field.strength = args.wind_strength
    wind.field.flow = 1.0
    
    # Point wind in +Y direction
    wind.rotation_euler = (1.5708, 0, 0)  # 90 degrees around X
    
    return wind


def create_ground():
    """Create ground plane for collision."""
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    
    # Add collision
    bpy.ops.object.modifier_add(type='COLLISION')
    
    return ground


def bake_simulation():
    """Bake the cloth simulation."""
    # Free any existing bake
    bpy.ops.ptcache.free_bake_all()
    
    # Bake
    bpy.ops.ptcache.bake_all(bake=True)


def export_meshes(output_dir):
    """Export cloth mesh for each frame."""
    cloth = bpy.data.objects.get("Cloth")
    if cloth is None:
        raise RuntimeError("Cloth object not found")
    
    mesh_dir = Path(output_dir) / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        
        # Select only cloth
        bpy.ops.object.select_all(action='DESELECT')
        cloth.select_set(True)
        bpy.context.view_layer.objects.active = cloth
        
        # Export as OBJ
        filepath = mesh_dir / f"cloth_frame_{frame:04d}.obj"
        bpy.ops.export_scene.obj(
            filepath=str(filepath),
            use_selection=True,
            use_mesh_modifiers=True,
            use_normals=True,
            use_uvs=False,
            use_materials=False
        )
        
        exported_files.append(str(filepath))
        print(f"Exported frame {frame}: {filepath}")
    
    return exported_files


def save_metadata(output_dir, pinned_indices, wind_strength):
    """Save simulation metadata to JSON."""
    cloth = bpy.data.objects.get("Cloth")
    wind = bpy.data.objects.get("Wind")
    
    # Get wind direction from rotation
    wind_dir = wind.matrix_world.to_3x3() @ mathutils.Vector((0, 0, -1))
    
    metadata = {
        "simulation": {
            "frames": args.frames,
            "fps": bpy.context.scene.render.fps,
            "cloth_resolution": args.resolution
        },
        "cloth": {
            "vertex_count": len(cloth.data.vertices),
            "face_count": len(cloth.data.polygons),
            "pinned_indices": pinned_indices
        },
        "forces": {
            "wind_strength": wind_strength,
            "wind_direction": [wind_dir.x, wind_dir.y, wind_dir.z],
            "gravity": [0, 0, -9.81]
        },
        "physics": {
            "mass": 0.3,
            "tension_stiffness": 15,
            "bending_stiffness": 0.5,
            "air_damping": 0.5
        }
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_path}")
    return metadata


def export_vertex_positions(output_dir):
    """Export vertex positions for each frame as numpy arrays."""
    try:
        import numpy as np
    except ImportError:
        print("NumPy not available, skipping vertex position export")
        return
    
    cloth = bpy.data.objects.get("Cloth")
    n_verts = len(cloth.data.vertices)
    n_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
    
    positions = np.zeros((n_frames, n_verts, 3), dtype=np.float32)
    
    for f_idx, frame in enumerate(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)):
        bpy.context.scene.frame_set(frame)
        
        # Get deformed mesh
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cloth_eval = cloth.evaluated_get(depsgraph)
        mesh = cloth_eval.to_mesh()
        
        for v in mesh.vertices:
            positions[f_idx, v.index, 0] = v.co.x
            positions[f_idx, v.index, 1] = v.co.y
            positions[f_idx, v.index, 2] = v.co.z
        
        cloth_eval.to_mesh_clear()
    
    np_path = Path(output_dir) / "vertex_positions.npy"
    np.save(np_path, positions)
    print(f"Saved vertex positions: {np_path} (shape: {positions.shape})")


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("NIF-Cloth3D-Interactive: Blender Simulation Pipeline")
    print("=" * 60)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Frames: {args.frames}")
    print(f"Resolution: {args.resolution}")
    print(f"Wind strength: {args.wind_strength}")
    
    # Setup
    print("\n[1/6] Setting up scene...")
    setup_scene()
    
    # Create objects
    print("[2/6] Creating cloth...")
    cloth, pinned_indices = create_cloth()
    print(f"       Cloth created with {len(cloth.data.vertices)} vertices")
    print(f"       Pinned {len(pinned_indices)} vertices")
    
    print("[3/6] Creating wind force...")
    wind = create_wind()
    
    print("[3.5/6] Creating ground collision...")
    ground = create_ground()
    
    # Bake simulation
    print("[4/6] Baking simulation...")
    bake_simulation()
    
    # Export
    print("[5/6] Exporting meshes...")
    exported_files = export_meshes(output_dir)
    
    print("[5.5/6] Exporting vertex positions...")
    export_vertex_positions(output_dir)
    
    print("[6/6] Saving metadata...")
    metadata = save_metadata(output_dir, pinned_indices, args.wind_strength)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Exported {len(exported_files)} frames to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
