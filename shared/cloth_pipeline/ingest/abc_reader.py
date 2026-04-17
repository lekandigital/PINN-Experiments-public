"""
Alembic (.abc) Reader for the cloth simulation pipeline.

Reads cloth simulation caches from Houdini Vellum, Maya, and other DCC tools
that export to the Alembic format.

Primary method: Use the alembic Python bindings (if installed)
Fallback: Use Blender as a subprocess to convert .abc -> OBJ sequence
"""

import subprocess
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import warnings

from ..types import ClothSequence, CollisionBody


class AlembicReader:
    """
    Reader for Alembic (.abc) files.
    
    Attempts to use native alembic Python bindings first.
    Falls back to Blender subprocess conversion if bindings are unavailable.
    """
    
    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize the Alembic reader.
        
        Args:
            filepath: Path to the .abc file
        """
        self.filepath = Path(filepath)
        self._native_available: Optional[bool] = None
        self._blender_path: Optional[str] = None
        
    @property
    def native_available(self) -> bool:
        """Check if native Alembic bindings are available."""
        if self._native_available is None:
            self._native_available = self._check_native_bindings()
        return self._native_available
    
    def _check_native_bindings(self) -> bool:
        """Check if alembic Python package is installed."""
        try:
            import alembic
            return True
        except ImportError:
            pass
        
        try:
            import imath
            import alembic
            return True
        except ImportError:
            pass
        
        return False
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable for fallback conversion."""
        if self._blender_path is not None:
            return self._blender_path
        
        # Common Blender locations
        candidates = [
            'blender',  # In PATH
            '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
            '/usr/bin/blender',  # Linux
            'C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe',  # Windows
            'C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe',
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, '--version'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self._blender_path = candidate
                    return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        return None
    
    def read(
        self,
        fps: float = 24.0,
        mesh_path: Optional[str] = None,
        use_fallback: bool = True,
    ) -> ClothSequence:
        """
        Read the Alembic file into a ClothSequence.
        
        Args:
            fps: Target framerate (for resampling if needed)
            mesh_path: Path within Alembic hierarchy to the mesh object.
                      If None, auto-detect the first polymesh.
            use_fallback: If True, use Blender fallback when native bindings fail
            
        Returns:
            ClothSequence containing the animation
        """
        if self.native_available:
            try:
                return self._read_native(fps=fps, mesh_path=mesh_path)
            except Exception as e:
                if use_fallback:
                    warnings.warn(f"Native Alembic read failed: {e}. Trying Blender fallback.")
                else:
                    raise
        
        if use_fallback:
            return self._read_via_blender(fps=fps)
        
        raise ImportError(
            "Alembic Python bindings not available. "
            "Install with: pip install alembic imath\n"
            "Or ensure Blender is installed for fallback conversion."
        )
    
    def _read_native(
        self,
        fps: float = 24.0,
        mesh_path: Optional[str] = None,
    ) -> ClothSequence:
        """Read Alembic using native Python bindings."""
        import alembic
        from alembic import Abc, AbcGeom
        
        archive = Abc.IArchive(str(self.filepath))
        top = archive.getTop()
        
        # Find the mesh object
        mesh_obj = None
        if mesh_path:
            mesh_obj = self._get_object_by_path(top, mesh_path)
        else:
            mesh_obj = self._find_first_mesh(top)
        
        if mesh_obj is None:
            raise ValueError(f"No polymesh found in {self.filepath}")
        
        # Get the mesh schema
        mesh = AbcGeom.IPolyMesh(mesh_obj, Abc.WrapExistingFlag.kWrapExisting)
        schema = mesh.getSchema()
        
        # Get time sampling
        time_sampling = schema.getTimeSampling()
        num_samples = schema.getNumSamples()
        
        if num_samples == 0:
            raise ValueError(f"No samples in mesh {mesh_path or mesh_obj.getName()}")
        
        # Read face connectivity from first sample (assuming constant topology)
        first_sample = schema.getValue(Abc.ISampleSelector(0))
        faces_counts = np.array(first_sample.getFaceCounts())
        faces_indices = np.array(first_sample.getFaceIndices())
        
        # Convert to triangle faces
        faces = self._convert_faces(faces_counts, faces_indices)
        
        # Read all vertex positions
        vertices_list = []
        for i in range(num_samples):
            sample = schema.getValue(Abc.ISampleSelector(i))
            positions = np.array(sample.getPositions(), dtype=np.float32)
            vertices_list.append(positions)
        
        vertices = np.stack(vertices_list, axis=0)
        
        # Read normals if available
        normals = None
        normals_param = schema.getNormalsParam()
        if normals_param.valid():
            normals_list = []
            for i in range(num_samples):
                sample = normals_param.getExpandedValue(Abc.ISampleSelector(i))
                n = np.array(sample.getVals(), dtype=np.float32)
                if len(n) == vertices.shape[1]:
                    normals_list.append(n)
            if len(normals_list) == num_samples:
                normals = np.stack(normals_list, axis=0)
        
        # Read UVs if available
        uvs = None
        uvs_param = schema.getUVsParam()
        if uvs_param.valid():
            sample = uvs_param.getExpandedValue(Abc.ISampleSelector(0))
            uv_vals = np.array(sample.getVals(), dtype=np.float32)
            if len(uv_vals) == vertices.shape[1]:
                uvs = uv_vals[:, :2]  # Take only UV, not UVW
        
        # Read velocities if available
        velocities = None
        vel_param = schema.getVelocitiesParam()
        if vel_param.valid():
            vel_list = []
            for i in range(num_samples):
                sample = vel_param.getValue(Abc.ISampleSelector(i))
                v = np.array(sample, dtype=np.float32)
                if len(v) == vertices.shape[1]:
                    vel_list.append(v)
            if len(vel_list) == num_samples:
                velocities = np.stack(vel_list, axis=0)
        
        # Determine actual fps from time sampling
        actual_fps = fps
        if num_samples > 1:
            t0 = time_sampling.getSampleTime(0)
            t1 = time_sampling.getSampleTime(1)
            if t1 > t0:
                actual_fps = 1.0 / (t1 - t0)
        
        return ClothSequence(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
            velocities=velocities,
            fps=actual_fps,
            dt=1.0 / actual_fps,
            frame_range=(0, num_samples - 1),
            metadata={
                'source_format': 'alembic',
                'source_file': str(self.filepath),
                'alembic_mesh_path': mesh_obj.getFullName(),
                'num_samples': num_samples,
            }
        )
    
    def _get_object_by_path(self, top, path: str):
        """Navigate to a specific object in the Alembic hierarchy."""
        from alembic import Abc
        
        parts = path.strip('/').split('/')
        current = top
        
        for part in parts:
            found = False
            for i in range(current.getNumChildren()):
                child = current.getChild(i)
                if child.getName() == part:
                    current = child
                    found = True
                    break
            if not found:
                return None
        
        return current
    
    def _find_first_mesh(self, obj):
        """Recursively find the first polymesh in the hierarchy."""
        from alembic import Abc, AbcGeom
        
        # Check if this object is a polymesh
        md = obj.getMetaData()
        if AbcGeom.IPolyMesh.matches(md):
            return obj
        
        # Recursively check children
        for i in range(obj.getNumChildren()):
            child = obj.getChild(i)
            result = self._find_first_mesh(child)
            if result is not None:
                return result
        
        return None
    
    def _convert_faces(
        self,
        face_counts: np.ndarray,
        face_indices: np.ndarray
    ) -> np.ndarray:
        """Convert Alembic face representation to triangle array."""
        triangles = []
        idx = 0
        
        for count in face_counts:
            if count == 3:
                triangles.append(face_indices[idx:idx+3])
            elif count == 4:
                # Quad -> 2 triangles
                quad = face_indices[idx:idx+4]
                triangles.append([quad[0], quad[1], quad[2]])
                triangles.append([quad[0], quad[2], quad[3]])
            else:
                # Fan triangulation
                face = face_indices[idx:idx+count]
                for i in range(1, count - 1):
                    triangles.append([face[0], face[i], face[i+1]])
            idx += count
        
        return np.array(triangles, dtype=np.int64)
    
    def _read_via_blender(self, fps: float = 24.0) -> ClothSequence:
        """
        Read Alembic by converting to OBJ sequence using Blender.
        
        This is the fallback when native Python bindings aren't available.
        """
        blender = self._find_blender()
        if blender is None:
            raise RuntimeError(
                "Cannot read Alembic: native bindings not available and Blender not found. "
                "Install Blender or the alembic Python package."
            )
        
        # Create temp directory for OBJ export
        temp_dir = tempfile.mkdtemp(prefix='cloth_pipeline_abc_')
        
        try:
            # Write conversion script
            script_path = Path(temp_dir) / 'convert_abc.py'
            self._write_conversion_script(script_path, temp_dir)
            
            # Run Blender
            result = subprocess.run(
                [blender, '--background', '--python', str(script_path), '--', str(self.filepath)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Blender conversion failed:\n{result.stderr}")
            
            # Read the OBJ sequence
            from .obj_sequence_reader import read_obj_sequence
            
            sequence = read_obj_sequence(temp_dir, fps=fps)
            
            # Update metadata
            sequence.metadata['source_format'] = 'alembic_via_blender'
            sequence.metadata['source_file'] = str(self.filepath)
            
            return sequence
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _write_conversion_script(self, script_path: Path, output_dir: str) -> None:
        """Write the Blender Python script for Alembic conversion."""
        script = f'''
import bpy
import sys
import os

# Get the input file from command line args
argv = sys.argv
abc_file = argv[argv.index('--') + 1] if '--' in argv else None

if not abc_file:
    print("ERROR: No Alembic file specified")
    sys.exit(1)

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import Alembic
bpy.ops.wm.alembic_import(filepath=abc_file)

# Find the mesh object
mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh_obj = obj
        break

if mesh_obj is None:
    print("ERROR: No mesh found in Alembic file")
    sys.exit(1)

# Get frame range
scene = bpy.context.scene
frame_start = scene.frame_start
frame_end = scene.frame_end

# Export each frame as OBJ
output_dir = r"{output_dir}"

for frame in range(frame_start, frame_end + 1):
    scene.frame_set(frame)
    
    # Evaluate the mesh at this frame
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = mesh_obj.evaluated_get(depsgraph)
    
    # Export to OBJ
    output_path = os.path.join(output_dir, f"frame_{{frame:04d}}.obj")
    
    # Select only this object
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,
    )

print(f"Exported {{frame_end - frame_start + 1}} frames to {{output_dir}}")
'''
        with open(script_path, 'w') as f:
            f.write(script)


def read_alembic(
    filepath: Union[str, Path],
    fps: float = 24.0,
    mesh_path: Optional[str] = None,
    use_fallback: bool = True,
    **kwargs
) -> ClothSequence:
    """
    Read an Alembic file into a ClothSequence.
    
    This is the main entry point for Alembic ingestion.
    
    Args:
        filepath: Path to the .abc file
        fps: Target framerate
        mesh_path: Path within Alembic hierarchy (auto-detect if None)
        use_fallback: Use Blender conversion if native bindings fail
        
    Returns:
        ClothSequence ready for transformation
        
    Example:
        sequence = read_alembic("/path/to/cloth_sim.abc", fps=24.0)
    """
    reader = AlembicReader(filepath)
    return reader.read(fps=fps, mesh_path=mesh_path, use_fallback=use_fallback)
