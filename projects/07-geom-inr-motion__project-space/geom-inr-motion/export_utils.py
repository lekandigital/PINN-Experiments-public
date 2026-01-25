#!/usr/bin/env python3
"""
export_utils.py - Export Utilities for Geom-INR-Motion

This module provides utilities for exporting motion predictions to
various formats for visualization and integration with DCC tools.

Supported Formats:
1. USD (Universal Scene Description) - Pixar's format for Blender/Maya/Houdini
2. FBX - Autodesk format (via Blender export)
3. BVH - Biovision Hierarchy format for motion capture
4. NPZ - NumPy compressed arrays for analysis
5. JSON - Human-readable format for web visualization

Dependencies:
    pip install numpy

Optional (for USD export):
    pip install usd-core  # or install from Pixar

Optional (for FBX export):
    Blender must be installed and accessible from command line

Usage:
    from export_utils import MotionExporter
    
    exporter = MotionExporter()
    
    # Export to BVH
    exporter.export_bvh(
        trajectory,
        joint_names,
        filepath="motion.bvh",
        fps=30
    )
    
    # Export to USD (if available)
    exporter.export_usd(
        trajectory,
        joint_names,
        filepath="motion.usd"
    )
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import subprocess


# ============================================================================
# Joint Hierarchy Definition
# ============================================================================

# Default SMPL joint names (24 joints)
SMPL_JOINT_NAMES = [
    "pelvis",         # 0
    "left_hip",       # 1
    "right_hip",      # 2
    "spine1",         # 3
    "left_knee",      # 4
    "right_knee",     # 5
    "spine2",         # 6
    "left_ankle",     # 7
    "right_ankle",    # 8
    "spine3",         # 9
    "left_foot",      # 10
    "right_foot",     # 11
    "neck",           # 12
    "left_collar",    # 13
    "right_collar",   # 14
    "head",           # 15
    "left_shoulder",  # 16
    "right_shoulder", # 17
    "left_elbow",     # 18
    "right_elbow",    # 19
    "left_wrist",     # 20
    "right_wrist",    # 21
    "left_hand",      # 22
    "right_hand",     # 23
]

# SMPL parent indices (-1 for root)
SMPL_PARENTS = [
    -1,  # pelvis
    0,   # left_hip -> pelvis
    0,   # right_hip -> pelvis
    0,   # spine1 -> pelvis
    1,   # left_knee -> left_hip
    2,   # right_knee -> right_hip
    3,   # spine2 -> spine1
    4,   # left_ankle -> left_knee
    5,   # right_ankle -> right_knee
    6,   # spine3 -> spine2
    7,   # left_foot -> left_ankle
    8,   # right_foot -> right_ankle
    9,   # neck -> spine3
    9,   # left_collar -> spine3
    9,   # right_collar -> spine3
    12,  # head -> neck
    13,  # left_shoulder -> left_collar
    14,  # right_shoulder -> right_collar
    16,  # left_elbow -> left_shoulder
    17,  # right_elbow -> right_shoulder
    18,  # left_wrist -> left_elbow
    19,  # right_wrist -> right_elbow
    20,  # left_hand -> left_wrist
    21,  # right_hand -> right_wrist
]


# ============================================================================
# BVH Export
# ============================================================================

def export_bvh(
    positions: np.ndarray,
    joint_names: Optional[List[str]] = None,
    parents: Optional[List[int]] = None,
    filepath: str = "motion.bvh",
    fps: float = 30.0,
    scale: float = 100.0  # Convert to cm for BVH
) -> str:
    """
    Export motion to BVH format.
    
    BVH files contain both skeleton hierarchy and motion data.
    
    Args:
        positions: (T, J, 3) array of joint positions
        joint_names: List of joint names (default: SMPL names)
        parents: List of parent indices (default: SMPL parents)
        filepath: Output file path
        fps: Frames per second
        scale: Scale factor (BVH typically uses cm)
    
    Returns:
        Path to saved file
    """
    T, J, _ = positions.shape
    
    if joint_names is None:
        joint_names = SMPL_JOINT_NAMES[:J]
    if parents is None:
        parents = SMPL_PARENTS[:J]
    
    # Scale positions
    positions = positions * scale
    
    # Compute offsets (average bone vectors)
    offsets = np.zeros((J, 3))
    for j in range(J):
        if parents[j] >= 0:
            parent_pos = positions[:, parents[j], :].mean(axis=0)
            child_pos = positions[:, j, :].mean(axis=0)
            offsets[j] = child_pos - parent_pos
    
    # Build hierarchy string
    def build_hierarchy(joint_idx: int, indent: int = 0) -> str:
        name = joint_names[joint_idx]
        offset = offsets[joint_idx]
        prefix = "  " * indent
        
        # Find children
        children = [i for i, p in enumerate(parents) if p == joint_idx]
        
        if joint_idx == 0:
            s = f"ROOT {name}\n"
        else:
            s = f"{prefix}JOINT {name}\n"
        
        s += f"{prefix}{{\n"
        s += f"{prefix}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
        
        if joint_idx == 0:
            s += f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
        else:
            s += f"{prefix}  CHANNELS 3 Xrotation Yrotation Zrotation\n"
        
        if children:
            for child in children:
                s += build_hierarchy(child, indent + 1)
        else:
            s += f"{prefix}  End Site\n"
            s += f"{prefix}  {{\n"
            s += f"{prefix}    OFFSET 0.0 0.0 0.0\n"
            s += f"{prefix}  }}\n"
        
        s += f"{prefix}}}\n"
        return s
    
    # Write BVH file
    with open(filepath, 'w') as f:
        # Header
        f.write("HIERARCHY\n")
        f.write(build_hierarchy(0))
        
        # Motion section
        f.write("MOTION\n")
        f.write(f"Frames: {T}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")
        
        # Motion data
        # For simplicity, we output positions as root translation + zero rotations
        for t in range(T):
            root_pos = positions[t, 0, :]
            values = [root_pos[0], root_pos[1], root_pos[2]]
            
            # Add rotation channels (zeros for position-only export)
            for j in range(J):
                values.extend([0.0, 0.0, 0.0])
            
            f.write(" ".join(f"{v:.6f}" for v in values) + "\n")
    
    print(f"Exported BVH to: {filepath}")
    return filepath


# ============================================================================
# USD Export
# ============================================================================

def export_usd(
    positions: np.ndarray,
    joint_names: Optional[List[str]] = None,
    filepath: str = "motion.usd",
    fps: float = 30.0,
    scale: float = 1.0
) -> Optional[str]:
    """
    Export motion to USD format.
    
    Requires the `pxr` (USD) Python package.
    
    Args:
        positions: (T, J, 3) array of joint positions
        joint_names: List of joint names
        filepath: Output file path
        fps: Frames per second
        scale: Scale factor
    
    Returns:
        Path to saved file, or None if USD not available
    """
    try:
        from pxr import Usd, UsdGeom, Gf, UsdSkel
    except ImportError:
        print("Warning: USD (pxr) not available. Install with: pip install usd-core")
        print("Falling back to USDA (ASCII) format...")
        return export_usda(positions, joint_names, filepath, fps, scale)
    
    T, J, _ = positions.shape
    
    if joint_names is None:
        joint_names = SMPL_JOINT_NAMES[:J]
    
    # Create stage
    stage = Usd.Stage.CreateNew(filepath)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(T - 1)
    stage.SetFramesPerSecond(fps)
    
    # Create root xform
    root = UsdGeom.Xform.Define(stage, "/Root")
    
    # Create joint transforms
    for j, name in enumerate(joint_names):
        joint_path = f"/Root/{name}"
        xform = UsdGeom.Xform.Define(stage, joint_path)
        
        # Add translation animation
        translate_op = xform.AddTranslateOp()
        
        for t in range(T):
            pos = positions[t, j, :] * scale
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), t)
    
    # Save
    stage.GetRootLayer().Save()
    print(f"Exported USD to: {filepath}")
    return filepath


def export_usda(
    positions: np.ndarray,
    joint_names: Optional[List[str]] = None,
    filepath: str = "motion.usda",
    fps: float = 30.0,
    scale: float = 1.0
) -> str:
    """
    Export motion to USDA (ASCII USD) format.
    
    This is a fallback when the pxr library is not available.
    USDA files can be loaded in Blender with the USD importer.
    
    Args:
        positions: (T, J, 3) array of joint positions
        joint_names: List of joint names
        filepath: Output file path
        fps: Frames per second
        scale: Scale factor
    
    Returns:
        Path to saved file
    """
    T, J, _ = positions.shape
    
    if joint_names is None:
        joint_names = SMPL_JOINT_NAMES[:J]
    
    # Ensure .usda extension
    if not filepath.endswith('.usda'):
        filepath = filepath.rsplit('.', 1)[0] + '.usda'
    
    with open(filepath, 'w') as f:
        # Header
        f.write('#usda 1.0\n')
        f.write('(\n')
        f.write(f'    endTimeCode = {T - 1}\n')
        f.write(f'    framesPerSecond = {fps}\n')
        f.write('    metersPerUnit = 1\n')
        f.write(f'    startTimeCode = 0\n')
        f.write(')\n\n')
        
        # Root
        f.write('def Xform "Root"\n{\n')
        
        # Joints
        for j, name in enumerate(joint_names):
            # Sanitize name for USD
            safe_name = name.replace(' ', '_').replace('-', '_')
            
            f.write(f'    def Xform "{safe_name}"\n')
            f.write('    {\n')
            
            # Animated translation
            f.write('        double3 xformOp:translate.timeSamples = {\n')
            
            for t in range(T):
                pos = positions[t, j, :] * scale
                comma = ',' if t < T - 1 else ''
                f.write(f'            {t}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}){comma}\n')
            
            f.write('        }\n')
            f.write('        uniform token[] xformOpOrder = ["xformOp:translate"]\n')
            f.write('    }\n')
        
        f.write('}\n')
    
    print(f"Exported USDA to: {filepath}")
    return filepath


# ============================================================================
# NPZ Export
# ============================================================================

def export_npz(
    positions: np.ndarray,
    joint_names: Optional[List[str]] = None,
    parents: Optional[List[int]] = None,
    filepath: str = "motion.npz",
    fps: float = 30.0,
    metadata: Optional[Dict] = None
) -> str:
    """
    Export motion to NPZ format for analysis.
    
    Args:
        positions: (T, J, 3) array of joint positions
        joint_names: List of joint names
        parents: List of parent indices
        filepath: Output file path
        fps: Frames per second
        metadata: Additional metadata dictionary
    
    Returns:
        Path to saved file
    """
    T, J, _ = positions.shape
    
    if joint_names is None:
        joint_names = SMPL_JOINT_NAMES[:J]
    if parents is None:
        parents = SMPL_PARENTS[:J]
    
    save_dict = {
        'positions': positions.astype(np.float32),
        'joint_names': np.array(joint_names, dtype=object),
        'parents': np.array(parents, dtype=np.int32),
        'fps': np.array(fps, dtype=np.float32),
        'num_frames': T,
        'num_joints': J
    }
    
    if metadata:
        for k, v in metadata.items():
            save_dict[f'meta_{k}'] = v
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Exported NPZ to: {filepath}")
    return filepath


# ============================================================================
# JSON Export
# ============================================================================

def export_json(
    positions: np.ndarray,
    joint_names: Optional[List[str]] = None,
    parents: Optional[List[int]] = None,
    filepath: str = "motion.json",
    fps: float = 30.0,
    precision: int = 4
) -> str:
    """
    Export motion to JSON format for web visualization.
    
    Args:
        positions: (T, J, 3) array of joint positions
        joint_names: List of joint names
        parents: List of parent indices
        filepath: Output file path
        fps: Frames per second
        precision: Decimal precision for coordinates
    
    Returns:
        Path to saved file
    """
    T, J, _ = positions.shape
    
    if joint_names is None:
        joint_names = SMPL_JOINT_NAMES[:J]
    if parents is None:
        parents = SMPL_PARENTS[:J]
    
    # Round positions for smaller file size
    positions_list = np.round(positions, precision).tolist()
    
    data = {
        'version': '1.0',
        'fps': fps,
        'num_frames': T,
        'num_joints': J,
        'joint_names': joint_names,
        'parents': parents,
        'positions': positions_list  # [T][J][3]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"Exported JSON to: {filepath}")
    return filepath


# ============================================================================
# Motion Exporter Class
# ============================================================================

class MotionExporter:
    """
    Unified motion exporter supporting multiple formats.
    
    Usage:
        exporter = MotionExporter(joint_names=SMPL_JOINT_NAMES)
        
        # Export trajectory from model
        trajectory = model.predict_trajectory(actor_id=0, times=times)
        exporter.export(trajectory.cpu().numpy(), "output/motion", formats=['bvh', 'json'])
    """
    
    def __init__(
        self,
        joint_names: Optional[List[str]] = None,
        parents: Optional[List[int]] = None,
        fps: float = 30.0
    ):
        """
        Initialize exporter.
        
        Args:
            joint_names: Default joint names
            parents: Default parent indices
            fps: Default frames per second
        """
        self.joint_names = joint_names
        self.parents = parents
        self.fps = fps
    
    def export(
        self,
        positions: np.ndarray,
        filepath: str,
        formats: List[str] = ['bvh', 'npz'],
        **kwargs
    ) -> Dict[str, str]:
        """
        Export motion to multiple formats.
        
        Args:
            positions: (T, J, 3) array of joint positions
            filepath: Base filepath (without extension)
            formats: List of formats to export
            **kwargs: Additional arguments passed to exporters
        
        Returns:
            Dictionary mapping format to output path
        """
        joint_names = kwargs.pop('joint_names', self.joint_names)
        parents = kwargs.pop('parents', self.parents)
        fps = kwargs.pop('fps', self.fps)
        
        # Ensure output directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        for fmt in formats:
            fmt = fmt.lower()
            
            if fmt == 'bvh':
                path = export_bvh(positions, joint_names, parents,
                                  f"{filepath}.bvh", fps, **kwargs)
            elif fmt == 'usd':
                path = export_usd(positions, joint_names,
                                  f"{filepath}.usd", fps, **kwargs)
            elif fmt == 'usda':
                path = export_usda(positions, joint_names,
                                   f"{filepath}.usda", fps, **kwargs)
            elif fmt == 'npz':
                path = export_npz(positions, joint_names, parents,
                                  f"{filepath}.npz", fps, **kwargs)
            elif fmt == 'json':
                path = export_json(positions, joint_names, parents,
                                   f"{filepath}.json", fps, **kwargs)
            else:
                print(f"Warning: Unknown format '{fmt}'")
                continue
            
            if path:
                outputs[fmt] = path
        
        return outputs
    
    def export_from_model(
        self,
        model,
        actor_id: int,
        times: np.ndarray,
        filepath: str,
        formats: List[str] = ['bvh', 'npz'],
        device=None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Export motion directly from model prediction.
        
        Args:
            model: GeomINR model
            actor_id: Actor ID for prediction
            times: Array of time values
            filepath: Base filepath
            formats: Export formats
            device: Torch device
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping format to output path
        """
        import torch
        
        if device is None:
            device = next(model.parameters()).device
        
        times_tensor = torch.tensor(times, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            trajectory = model.predict_trajectory(actor_id, times_tensor, device)
        
        positions = trajectory.cpu().numpy()
        return self.export(positions, filepath, formats, **kwargs)


# ============================================================================
# Blender Script Generator
# ============================================================================

def generate_blender_import_script(
    motion_file: str,
    output_script: str = "import_motion.py"
) -> str:
    """
    Generate a Blender Python script to import motion data.
    
    Args:
        motion_file: Path to motion file (BVH, USD, or JSON)
        output_script: Output script path
    
    Returns:
        Path to generated script
    """
    ext = Path(motion_file).suffix.lower()
    motion_file_abs = os.path.abspath(motion_file)
    
    script = f'''"""
Blender Import Script for Geom-INR-Motion
Run this in Blender's Python console or as a script.
"""

import bpy
import os

# Motion file to import
MOTION_FILE = r"{motion_file_abs}"

def import_motion():
    """Import motion file into Blender."""
    ext = os.path.splitext(MOTION_FILE)[1].lower()
    
    if ext == '.bvh':
        bpy.ops.import_anim.bvh(filepath=MOTION_FILE)
        print(f"Imported BVH: {{MOTION_FILE}}")
    
    elif ext in ['.usd', '.usda', '.usdc']:
        bpy.ops.wm.usd_import(filepath=MOTION_FILE)
        print(f"Imported USD: {{MOTION_FILE}}")
    
    elif ext == '.json':
        # Custom JSON import
        import json
        with open(MOTION_FILE, 'r') as f:
            data = json.load(f)
        
        # Create armature
        bpy.ops.object.armature_add()
        armature = bpy.context.active_object
        armature.name = "GeomINR_Skeleton"
        
        print(f"Loaded JSON with {{data['num_frames']}} frames, {{data['num_joints']}} joints")
        # Note: Full JSON import would require more code to set keyframes
    
    else:
        print(f"Unsupported format: {{ext}}")

if __name__ == "__main__":
    import_motion()
'''
    
    with open(output_script, 'w') as f:
        f.write(script)
    
    print(f"Generated Blender script: {output_script}")
    return output_script


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Export Utilities Test")
    print("=" * 60)
    
    # Create test motion data
    print("\n[1] Creating test motion data...")
    T, J = 100, 24  # 100 frames, 24 joints
    t = np.linspace(0, 2 * np.pi, T)
    
    # Simple walking-like motion
    positions = np.zeros((T, J, 3))
    for j in range(J):
        phase = j * 0.2
        positions[:, j, 0] = t / (2 * np.pi) * 2  # Forward motion
        positions[:, j, 1] = 0.5 + j * 0.05  # Height varies by joint
        positions[:, j, 2] = 0.1 * np.sin(2 * t + phase)  # Side motion
    
    print(f"    Shape: {positions.shape}")
    
    # Create output directory
    output_dir = "test_exports"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Test BVH export
    print("\n[2] Testing BVH export...")
    bvh_path = export_bvh(
        positions,
        joint_names=SMPL_JOINT_NAMES,
        parents=SMPL_PARENTS,
        filepath=os.path.join(output_dir, "test_motion.bvh")
    )
    assert os.path.exists(bvh_path)
    print(f"    ✓ BVH file size: {os.path.getsize(bvh_path) / 1024:.1f} KB")
    
    # Test USDA export
    print("\n[3] Testing USDA export...")
    usda_path = export_usda(
        positions,
        joint_names=SMPL_JOINT_NAMES,
        filepath=os.path.join(output_dir, "test_motion.usda")
    )
    assert os.path.exists(usda_path)
    print(f"    ✓ USDA file size: {os.path.getsize(usda_path) / 1024:.1f} KB")
    
    # Test NPZ export
    print("\n[4] Testing NPZ export...")
    npz_path = export_npz(
        positions,
        joint_names=SMPL_JOINT_NAMES,
        parents=SMPL_PARENTS,
        filepath=os.path.join(output_dir, "test_motion.npz"),
        metadata={'actor_id': 0, 'action': 'walking'}
    )
    assert os.path.exists(npz_path)
    print(f"    ✓ NPZ file size: {os.path.getsize(npz_path) / 1024:.1f} KB")
    
    # Verify NPZ contents
    loaded = np.load(npz_path, allow_pickle=True)
    assert loaded['positions'].shape == positions.shape
    print(f"    ✓ NPZ verified: {list(loaded.keys())}")
    
    # Test JSON export
    print("\n[5] Testing JSON export...")
    json_path = export_json(
        positions,
        joint_names=SMPL_JOINT_NAMES,
        parents=SMPL_PARENTS,
        filepath=os.path.join(output_dir, "test_motion.json")
    )
    assert os.path.exists(json_path)
    print(f"    ✓ JSON file size: {os.path.getsize(json_path) / 1024:.1f} KB")
    
    # Test MotionExporter class
    print("\n[6] Testing MotionExporter class...")
    exporter = MotionExporter(
        joint_names=SMPL_JOINT_NAMES,
        parents=SMPL_PARENTS,
        fps=30.0
    )
    
    outputs = exporter.export(
        positions,
        os.path.join(output_dir, "exported_motion"),
        formats=['bvh', 'npz', 'json']
    )
    print(f"    ✓ Exported formats: {list(outputs.keys())}")
    
    # Test Blender script generation
    print("\n[7] Testing Blender script generation...")
    script_path = generate_blender_import_script(
        bvh_path,
        os.path.join(output_dir, "import_motion.py")
    )
    assert os.path.exists(script_path)
    
    print("\n✓ Export utilities test completed!")
    print(f"\nOutput files in: {os.path.abspath(output_dir)}")
