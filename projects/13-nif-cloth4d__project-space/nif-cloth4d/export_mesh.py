"""
NIF-Cloth4D: Mesh Export Script

This script exports trained model predictions to standard mesh formats:
1. Extracts meshes at specified timestamps using marching cubes
2. Exports to USD (Universal Scene Description) format
3. Optionally exports to OBJ format as fallback

Usage:
    python export_mesh.py --checkpoint ./checkpoints/model_best.pt --times 0.0,0.5,1.0 --output ./exports
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch

# Try to import mcubes
try:
    import mcubes
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False
    print("Warning: mcubes not installed. Install with: pip install PyMCubes")

# Try to import USD
try:
    from pxr import Usd, UsdGeom, Vt, Gf
    HAS_USD = True
except ImportError:
    HAS_USD = False
    print("Warning: USD (pxr) not installed. Install with: pip install usd-core")

# Local imports
from nif_cloth4d import FourierFeatureSIREN, create_model


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[FourierFeatureSIREN, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device
    
    Returns:
        model: Loaded model
        config: Training configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model, config


def query_sdf_grid(
    model: FourierFeatureSIREN,
    frame_time: float,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: int = 64,
    device: torch.device = torch.device('cpu'),
    chunk_size: int = 65536
) -> np.ndarray:
    """
    Query the model's SDF on a 3D grid.
    
    Args:
        model: Trained SDF model
        frame_time: Time value to query
        grid_min: Minimum coordinates (3,)
        grid_max: Maximum coordinates (3,)
        resolution: Grid resolution per axis
        device: Torch device
        chunk_size: Batch size for queries
    
    Returns:
        sdf_grid: SDF values as (resolution, resolution, resolution) array
    """
    model.eval()
    
    # Create coordinate grid
    xs = np.linspace(grid_min[0], grid_max[0], resolution)
    ys = np.linspace(grid_min[1], grid_max[1], resolution)
    zs = np.linspace(grid_min[2], grid_max[2], resolution)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    times = np.full((len(coords), 1), frame_time)
    coords_4d = np.hstack([coords, times]).astype(np.float32)
    
    # Query in chunks
    sdf_values = []
    with torch.no_grad():
        for i in range(0, len(coords_4d), chunk_size):
            batch = torch.from_numpy(coords_4d[i:i+chunk_size]).to(device)
            sdf = model(batch).cpu().numpy()
            sdf_values.append(sdf)
    
    sdf_values = np.concatenate(sdf_values, axis=0)
    sdf_grid = sdf_values.reshape((resolution, resolution, resolution))
    
    return sdf_grid


def extract_mesh(
    sdf_grid: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    iso_level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from SDF using marching cubes.
    
    Args:
        sdf_grid: SDF volume
        grid_min: Minimum coordinates
        grid_max: Maximum coordinates
        iso_level: ISO surface level (0 for SDF)
    
    Returns:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
    """
    if not HAS_MCUBES:
        raise ImportError("mcubes required for mesh extraction")
    
    # Run marching cubes
    vertices, faces = mcubes.marching_cubes(sdf_grid, iso_level)
    
    # Scale vertices to world coordinates
    resolution = sdf_grid.shape[0]
    scale = (grid_max - grid_min) / (resolution - 1)
    vertices = vertices * scale + grid_min
    
    return vertices.astype(np.float32), faces.astype(np.int32)


def export_to_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str
):
    """
    Export mesh to OBJ format.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        f.write("# NIF-Cloth4D exported mesh\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Exported OBJ: {filepath}")


def export_to_usd(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str,
    mesh_name: str = "ClothMesh"
):
    """
    Export mesh to USD format.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
        filepath: Output file path
        mesh_name: Name for the mesh prim
    """
    if not HAS_USD:
        raise ImportError("USD (pxr) required for USD export")
    
    # Create USD stage
    stage = Usd.Stage.CreateNew(filepath)
    
    # Define mesh
    mesh_path = f"/{mesh_name}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Set vertex positions
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    mesh.CreatePointsAttr(points)
    
    # Set face vertex indices
    face_indices = faces.flatten().tolist()
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    
    # Set face vertex counts (3 per triangle)
    face_counts = [3] * len(faces)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    
    # Set subdivision scheme to none (we want triangles, not subdivided)
    mesh.CreateSubdivisionSchemeAttr("none")
    
    # Save
    stage.GetRootLayer().Save()
    print(f"Exported USD: {filepath}")


def export_mesh_sequence(
    checkpoint_path: str,
    output_dir: str,
    times: List[float],
    resolution: int = 64,
    formats: List[str] = ['usd', 'obj']
) -> List[str]:
    """
    Export mesh sequence from trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory
        times: List of timestamps to export
        resolution: Marching cubes resolution
        formats: List of formats to export ('usd', 'obj')
    
    Returns:
        List of exported file paths
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Grid bounds
    grid_min = np.array([-1.0, -1.0, -1.0])
    grid_max = np.array([1.0, 1.0, 1.0])
    
    exported_files = []
    
    print(f"\nExporting {len(times)} meshes at resolution {resolution}³")
    print(f"Output directory: {output_dir}")
    print(f"Formats: {formats}")
    print("-" * 40)
    
    for i, t in enumerate(times):
        print(f"\nFrame {i+1}/{len(times)}: t = {t:.3f}")
        
        # Query SDF
        print("  Querying SDF...")
        sdf_grid = query_sdf_grid(
            model, t, grid_min, grid_max, resolution, device
        )
        print(f"  SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
        
        # Extract mesh
        if HAS_MCUBES:
            print("  Extracting mesh...")
            try:
                vertices, faces = extract_mesh(sdf_grid, grid_min, grid_max)
                print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")
                
                # Export in requested formats
                base_name = f"cloth_t{t:.3f}".replace('.', '_')
                
                if 'obj' in formats:
                    obj_path = output_path / f"{base_name}.obj"
                    export_to_obj(vertices, faces, str(obj_path))
                    exported_files.append(str(obj_path))
                
                if 'usd' in formats and HAS_USD:
                    usd_path = output_path / f"{base_name}.usda"
                    export_to_usd(vertices, faces, str(usd_path), f"Cloth_t{i}")
                    exported_files.append(str(usd_path))
                elif 'usd' in formats and not HAS_USD:
                    print("  Warning: USD export skipped (pxr not installed)")
                
            except Exception as e:
                print(f"  Error extracting mesh: {e}")
        else:
            print("  Warning: Mesh extraction skipped (mcubes not installed)")
    
    # Summary
    print("\n" + "="*40)
    print("EXPORT COMPLETE")
    print("="*40)
    print(f"Exported {len(exported_files)} files:")
    for f in exported_files:
        size = os.path.getsize(f) / 1024  # KB
        print(f"  {f} ({size:.1f} KB)")
    
    # Verify files
    print("\nVerification:")
    valid_count = 0
    for f in exported_files:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            valid_count += 1
            print(f"  ✓ {Path(f).name}")
        else:
            print(f"  ✗ {Path(f).name} (invalid or empty)")
    
    print(f"\n{valid_count}/{len(exported_files)} files valid")
    
    return exported_files


def main():
    parser = argparse.ArgumentParser(description='Export NIF-Cloth4D meshes')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./exports',
                        help='Output directory')
    parser.add_argument('--times', type=str, default='0.0,0.5,1.0',
                        help='Comma-separated list of times to export')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Marching cubes resolution')
    parser.add_argument('--formats', type=str, default='usd,obj',
                        help='Comma-separated list of formats (usd, obj)')
    args = parser.parse_args()
    
    times = [float(t) for t in args.times.split(',')]
    formats = [f.strip().lower() for f in args.formats.split(',')]
    
    export_mesh_sequence(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        times=times,
        resolution=args.resolution,
        formats=formats
    )


if __name__ == "__main__":
    main()
