"""
Exporters for animation pipeline outputs.

Supports exporting to:
- OBJ: Single frame mesh export
- NPZ: Compressed numpy archive for full sequences
- BVH: Biovision hierarchy for skeleton animation
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from pathlib import Path


def export_obj(vertices: np.ndarray, faces: np.ndarray,
               output_path: str, normals: Optional[np.ndarray] = None,
               uvs: Optional[np.ndarray] = None,
               material_name: Optional[str] = None) -> None:
    """
    Export mesh to OBJ format.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices (0-indexed)
        output_path: Output file path
        normals: (V, 3) vertex normals (optional)
        uvs: (V, 2) texture coordinates (optional)
        material_name: Material name to reference
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write(f"# OBJ file exported from AnimationPipeline\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # Material reference
        if material_name:
            mtl_path = output_path.with_suffix('.mtl')
            f.write(f"mtllib {mtl_path.name}\n")
            f.write(f"usemtl {material_name}\n\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        # Normals
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("\n")
        
        # UVs
        if uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            f.write("\n")
        
        # Faces (1-indexed)
        for face in faces:
            if normals is not None and uvs is not None:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                       f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                       f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            elif normals is not None:
                f.write(f"f {face[0]+1}//{face[0]+1} "
                       f"{face[1]+1}//{face[1]+1} "
                       f"{face[2]+1}//{face[2]+1}\n")
            elif uvs is not None:
                f.write(f"f {face[0]+1}/{face[0]+1} "
                       f"{face[1]+1}/{face[1]+1} "
                       f"{face[2]+1}/{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def export_obj_sequence(frames: List[Dict], output_dir: str,
                        mesh_key: str = 'body',
                        prefix: str = 'frame') -> List[str]:
    """
    Export mesh sequence as numbered OBJ files.
    
    Args:
        frames: List of frame dictionaries with mesh data
        output_dir: Output directory
        mesh_key: Which mesh to export ('body' or 'cloth')
        prefix: Filename prefix
        
    Returns:
        List of exported file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported = []
    num_digits = len(str(len(frames)))
    
    for i, frame in enumerate(frames):
        filename = f"{prefix}_{str(i).zfill(num_digits)}.obj"
        output_path = output_dir / filename
        
        if mesh_key == 'body':
            vertices = frame.get('body_vertices', frame.get('body', {}).get('vertices'))
            normals = frame.get('body_normals', frame.get('body', {}).get('normals'))
            faces = frame.get('body_faces', frame.get('body', {}).get('faces'))
        else:
            vertices = frame.get('cloth_vertices', frame.get('cloth', {}).get('vertices'))
            normals = frame.get('cloth_normals', frame.get('cloth', {}).get('normals'))
            faces = frame.get('cloth_faces', frame.get('cloth', {}).get('faces'))
        
        if vertices is not None and faces is not None:
            export_obj(vertices, faces, str(output_path), normals=normals)
            exported.append(str(output_path))
    
    return exported


def export_npz(frames: List[Dict], output_path: str,
               compress: bool = True) -> None:
    """
    Export full animation sequence to compressed NPZ archive.
    
    The archive contains:
    - times: (N,) frame times
    - joint_positions: (N, J, 3) skeleton positions
    - joint_rotations: (N, J, 4) skeleton rotations
    - body_vertices: (N, V_body, 3) body mesh vertices
    - body_faces: (F_body, 3) body face indices
    - cloth_vertices: (N, V_cloth, 3) cloth mesh vertices
    - cloth_faces: (F_cloth, 3) cloth face indices
    - metadata: dict with config info
    
    Args:
        frames: List of frame dictionaries
        output_path: Output file path (.npz)
        compress: Whether to use compression
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect arrays
    times = np.array([f.get('time', i/30.0) for i, f in enumerate(frames)])
    
    # Stack frame data
    joint_positions = np.stack([f['joint_positions'] for f in frames])
    joint_rotations = np.stack([f['joint_rotations'] for f in frames])
    body_vertices = np.stack([f['body_vertices'] for f in frames])
    cloth_vertices = np.stack([f['cloth_vertices'] for f in frames])
    
    # Faces are constant
    body_faces = frames[0]['body_faces']
    cloth_faces = frames[0]['cloth_faces']
    
    # Optional data
    data = {
        'times': times,
        'joint_positions': joint_positions,
        'joint_rotations': joint_rotations,
        'body_vertices': body_vertices,
        'body_faces': body_faces,
        'cloth_vertices': cloth_vertices,
        'cloth_faces': cloth_faces,
    }
    
    # Add normals if available
    if 'body_normals' in frames[0]:
        data['body_normals'] = np.stack([f['body_normals'] for f in frames])
    if 'cloth_normals' in frames[0]:
        data['cloth_normals'] = np.stack([f['cloth_normals'] for f in frames])
    
    # Add root data
    if 'root_position' in frames[0]:
        data['root_positions'] = np.stack([f['root_position'] for f in frames])
    if 'root_rotation' in frames[0]:
        data['root_rotations'] = np.stack([f['root_rotation'] for f in frames])
    
    # Add diagnostics
    if 'cloth_strain_energy' in frames[0] and frames[0]['cloth_strain_energy'] is not None:
        data['cloth_strain_energy'] = np.array([
            f.get('cloth_strain_energy', 0) for f in frames
        ])
    
    # Save
    if compress:
        np.savez_compressed(output_path, **data)
    else:
        np.savez(output_path, **data)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load animation sequence from NPZ archive.
    
    Args:
        path: Path to NPZ file
        
    Returns:
        Dictionary with animation data
    """
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


class BVHExporter:
    """
    Export skeleton animation to BVH (Biovision Hierarchy) format.
    
    BVH is a standard format for skeleton animation that can be
    imported into most 3D software (Blender, Maya, etc.).
    """
    
    # Standard channel order
    CHANNEL_ORDER = ['Xposition', 'Yposition', 'Zposition',
                     'Zrotation', 'Xrotation', 'Yrotation']
    
    def __init__(self, joint_names: List[str], parent_indices: List[int],
                 rest_positions: np.ndarray):
        """
        Args:
            joint_names: List of joint names
            parent_indices: Parent index for each joint (-1 for root)
            rest_positions: (J, 3) rest pose positions
        """
        self.joint_names = joint_names
        self.parent_indices = parent_indices
        self.rest_positions = rest_positions
        self.num_joints = len(joint_names)
        
        # Build hierarchy
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Build joint hierarchy tree."""
        self.children = [[] for _ in range(self.num_joints)]
        for i, parent in enumerate(self.parent_indices):
            if parent >= 0:
                self.children[parent].append(i)
        
        # Find root (parent = -1)
        self.root_idx = self.parent_indices.index(-1) if -1 in self.parent_indices else 0
    
    def export(self, output_path: str,
               joint_positions: np.ndarray,
               joint_rotations: Optional[np.ndarray] = None,
               fps: float = 30.0):
        """
        Export animation to BVH file.
        
        Args:
            output_path: Output file path
            joint_positions: (N, J, 3) joint positions per frame
            joint_rotations: (N, J, 4) joint quaternions per frame (wxyz)
            fps: Frames per second
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_frames = joint_positions.shape[0]
        frame_time = 1.0 / fps
        
        with open(output_path, 'w') as f:
            # Write hierarchy
            f.write("HIERARCHY\n")
            self._write_joint(f, self.root_idx, 0, is_root=True)
            
            # Write motion
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")
            
            # Write frame data
            for frame_idx in range(num_frames):
                values = []
                self._collect_frame_values(
                    values, 
                    self.root_idx,
                    joint_positions[frame_idx],
                    joint_rotations[frame_idx] if joint_rotations is not None else None,
                    is_root=True
                )
                f.write(" ".join(f"{v:.6f}" for v in values) + "\n")
    
    def _write_joint(self, f, joint_idx: int, indent: int, is_root: bool = False):
        """Write joint hierarchy entry."""
        name = self.joint_names[joint_idx]
        prefix = "  " * indent
        
        if is_root:
            f.write(f"{prefix}ROOT {name}\n")
        else:
            f.write(f"{prefix}JOINT {name}\n")
        
        f.write(f"{prefix}{{\n")
        
        # Offset from parent
        if is_root:
            offset = self.rest_positions[joint_idx]
        else:
            parent = self.parent_indices[joint_idx]
            offset = self.rest_positions[joint_idx] - self.rest_positions[parent]
        
        f.write(f"{prefix}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Channels
        if is_root:
            f.write(f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        else:
            f.write(f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
        
        # Children
        if self.children[joint_idx]:
            for child_idx in self.children[joint_idx]:
                self._write_joint(f, child_idx, indent + 1)
        else:
            # End site
            f.write(f"{prefix}  End Site\n")
            f.write(f"{prefix}  {{\n")
            f.write(f"{prefix}    OFFSET 0.000000 0.100000 0.000000\n")
            f.write(f"{prefix}  }}\n")
        
        f.write(f"{prefix}}}\n")
    
    def _collect_frame_values(self, values: List[float], joint_idx: int,
                               positions: np.ndarray,
                               rotations: Optional[np.ndarray],
                               is_root: bool = False):
        """Collect motion values for a joint."""
        if is_root:
            # Root has position + rotation
            pos = positions[joint_idx]
            values.extend([pos[0], pos[1], pos[2]])
        
        # Rotation (convert quaternion to Euler ZXY)
        if rotations is not None:
            quat = rotations[joint_idx]  # wxyz
            euler = self._quat_to_euler_zxy(quat)
        else:
            euler = [0.0, 0.0, 0.0]
        
        values.extend(euler)
        
        # Recurse to children
        for child_idx in self.children[joint_idx]:
            self._collect_frame_values(values, child_idx, positions, rotations)
    
    def _quat_to_euler_zxy(self, quat: np.ndarray) -> List[float]:
        """
        Convert quaternion (wxyz) to Euler angles (ZXY order) in degrees.
        
        Args:
            quat: (4,) quaternion [w, x, y, z]
            
        Returns:
            [rz, rx, ry] in degrees
        """
        w, x, y, z = quat
        
        # ZXY Euler angles
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        rx = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1, 1)
        ry = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        rz = np.arctan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return [np.degrees(rz), np.degrees(rx), np.degrees(ry)]


def export_bvh(joint_names: List[str], parent_indices: List[int],
               rest_positions: np.ndarray,
               joint_positions: np.ndarray,
               output_path: str,
               joint_rotations: Optional[np.ndarray] = None,
               fps: float = 30.0):
    """
    Convenience function to export BVH file.
    
    Args:
        joint_names: List of joint names
        parent_indices: Parent index for each joint
        rest_positions: (J, 3) rest pose positions
        joint_positions: (N, J, 3) animated joint positions
        output_path: Output file path
        joint_rotations: (N, J, 4) joint quaternions (optional)
        fps: Frames per second
    """
    exporter = BVHExporter(joint_names, parent_indices, rest_positions)
    exporter.export(output_path, joint_positions, joint_rotations, fps)


def export_frame(frame, output_dir: str, frame_idx: int = 0,
                 export_body: bool = True, export_cloth: bool = True,
                 export_skeleton: bool = True):
    """
    Export a single frame to multiple formats.
    
    Args:
        frame: FrameResult or dict with frame data
        output_dir: Output directory
        frame_idx: Frame index for naming
        export_body: Export body mesh OBJ
        export_cloth: Export cloth mesh OBJ
        export_skeleton: Export skeleton positions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert FrameResult to dict if needed
    if hasattr(frame, 'to_dict'):
        frame = frame.to_dict()
    
    idx_str = str(frame_idx).zfill(4)
    
    if export_body and 'body_vertices' in frame:
        export_obj(
            frame['body_vertices'],
            frame['body_faces'],
            str(output_dir / f"body_{idx_str}.obj"),
            normals=frame.get('body_normals')
        )
    
    if export_cloth and 'cloth_vertices' in frame:
        export_obj(
            frame['cloth_vertices'],
            frame['cloth_faces'],
            str(output_dir / f"cloth_{idx_str}.obj"),
            normals=frame.get('cloth_normals')
        )
    
    if export_skeleton and 'joint_positions' in frame:
        np.save(
            str(output_dir / f"skeleton_{idx_str}.npy"),
            frame['joint_positions']
        )


class SequenceExporter:
    """
    High-level exporter for animation sequences.
    
    Handles exporting full sequences to various formats with
    consistent naming and organization.
    """
    
    def __init__(self, output_dir: str, name: str = "animation"):
        """
        Args:
            output_dir: Base output directory
            name: Animation name (used for subdirectories)
        """
        self.output_dir = Path(output_dir)
        self.name = name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_all(self, frames: List,
                   joint_names: Optional[List[str]] = None,
                   parent_indices: Optional[List[int]] = None,
                   rest_positions: Optional[np.ndarray] = None,
                   fps: float = 30.0):
        """
        Export sequence in all supported formats.
        
        Args:
            frames: List of FrameResult or dicts
            joint_names: Joint names for BVH export
            parent_indices: Parent indices for BVH export
            rest_positions: Rest positions for BVH export
            fps: Frames per second
        """
        # Convert frames to dicts if needed
        frame_dicts = []
        for f in frames:
            if hasattr(f, 'to_dict'):
                frame_dicts.append(f.to_dict())
            else:
                frame_dicts.append(f)
        
        # NPZ archive
        npz_path = self.output_dir / f"{self.name}.npz"
        export_npz(frame_dicts, str(npz_path))
        print(f"Exported NPZ: {npz_path}")
        
        # OBJ sequences
        body_dir = self.output_dir / f"{self.name}_body_obj"
        cloth_dir = self.output_dir / f"{self.name}_cloth_obj"
        
        export_obj_sequence(frame_dicts, str(body_dir), mesh_key='body', prefix='body')
        print(f"Exported body OBJs: {body_dir}")
        
        export_obj_sequence(frame_dicts, str(cloth_dir), mesh_key='cloth', prefix='cloth')
        print(f"Exported cloth OBJs: {cloth_dir}")
        
        # BVH skeleton
        if joint_names and parent_indices is not None and rest_positions is not None:
            bvh_path = self.output_dir / f"{self.name}.bvh"
            joint_positions = np.stack([f['joint_positions'] for f in frame_dicts])
            joint_rotations = None
            if 'joint_rotations' in frame_dicts[0]:
                joint_rotations = np.stack([f['joint_rotations'] for f in frame_dicts])
            
            export_bvh(
                joint_names, parent_indices, rest_positions,
                joint_positions, str(bvh_path),
                joint_rotations=joint_rotations, fps=fps
            )
            print(f"Exported BVH: {bvh_path}")
    
    def export_preview_frames(self, frames: List, indices: List[int] = None):
        """
        Export a few preview frames for quick visualization.
        
        Args:
            frames: Full frame list
            indices: Which frames to export (default: first, middle, last)
        """
        if indices is None:
            n = len(frames)
            indices = [0, n // 2, n - 1] if n > 2 else list(range(n))
        
        preview_dir = self.output_dir / f"{self.name}_preview"
        preview_dir.mkdir(exist_ok=True)
        
        for idx in indices:
            if idx < len(frames):
                export_frame(frames[idx], str(preview_dir), frame_idx=idx)
        
        print(f"Exported preview frames: {preview_dir}")
