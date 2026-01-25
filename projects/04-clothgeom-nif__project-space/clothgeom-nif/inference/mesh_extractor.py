"""
Mesh Extraction from Neural Implicit Field

Extracts watertight meshes from trained NIF decoder using
marching cubes algorithm with progressive resolution support.

Features:
- Single and multi-resolution extraction
- Configurable bounding box and isosurface level
- Progressive extraction (coarse-to-fine)
- Mesh export (OBJ, PLY, STL)
- Optional mesh post-processing (smoothing, simplification)
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Mesh extraction disabled.")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MeshExtractionConfig:
    """Configuration for mesh extraction."""
    resolution: int = 128           # Grid resolution for marching cubes
    bounds: Tuple[float, float] = (-1.5, 1.5)  # Bounding box [-b, b]³
    iso_level: float = 0.0          # Isosurface level (0 for SDF surface)
    batch_size: int = 65536         # Points per batch for inference
    smooth_iterations: int = 0      # Laplacian smoothing iterations
    simplify_ratio: float = 1.0     # Mesh simplification (1.0 = no simplification)
    compute_normals: bool = True    # Compute vertex normals


class MeshExtractor:
    """
    Extract watertight meshes from NIF decoder.
    
    Uses marching cubes on SDF volume predicted by the decoder.
    
    Args:
        model: Trained NIFDecoder model
        device: Torch device for inference
        config: Extraction configuration
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[MeshExtractionConfig] = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or MeshExtractionConfig()
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_sdf_grid(
        self,
        latent: torch.Tensor,
        resolution: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate SDF on a regular 3D grid.
        
        Args:
            latent: Latent code [latent_dim]
            resolution: Grid resolution (overrides config)
        
        Returns:
            sdf_volume: [R, R, R] numpy array of SDF values
        """
        resolution = resolution or self.config.resolution
        b = self.config.bounds[1]
        
        # Create grid coordinates
        coords_1d = np.linspace(-b, b, resolution)
        xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid_points).float().to(self.device)
        
        # Ensure latent is on device and properly shaped
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent = latent.to(self.device)
        
        # Batched inference
        sdf_values = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i+batch_size]
            batch_latent = latent.expand(batch.shape[0], -1)
            
            sdf, _ = self.model(batch, batch_latent, return_variance=False)
            sdf_values.append(sdf.cpu().numpy())
        
        sdf_volume = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
        return sdf_volume.astype(np.float32)
    
    def extract_mesh(
        self,
        latent: torch.Tensor,
        resolution: Optional[int] = None,
        iso_level: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract mesh from latent code using marching cubes.
        
        Args:
            latent: Latent code [latent_dim]
            resolution: Grid resolution (overrides config)
            iso_level: Isosurface level (overrides config)
        
        Returns:
            vertices: [V, 3] vertex positions
            faces: [F, 3] face indices
            normals: [V, 3] vertex normals (if compute_normals=True)
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for mesh extraction")
        
        resolution = resolution or self.config.resolution
        iso_level = iso_level if iso_level is not None else self.config.iso_level
        
        logger.info(f"Extracting mesh at resolution {resolution}...")
        
        # Evaluate SDF on grid
        sdf_volume = self.evaluate_sdf_grid(latent, resolution)
        
        # Check for valid surface
        if sdf_volume.min() >= 0 or sdf_volume.max() <= 0:
            logger.warning("No zero crossing found in SDF volume. Surface may not exist.")
            # Return empty mesh
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))
        
        # Pad volume to avoid boundary artifacts
        sdf_padded = np.pad(sdf_volume, pad_width=1, mode='constant', constant_values=1.0)
        
        # Run marching cubes
        try:
            vertices, faces, normals, _ = measure.marching_cubes(
                sdf_padded,
                level=iso_level,
                spacing=(1.0, 1.0, 1.0),
                gradient_direction='descent'
            )
        except Exception as e:
            logger.error(f"Marching cubes failed: {e}")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))
        
        # Adjust for padding and rescale to world coordinates
        b = self.config.bounds[1]
        vertices = vertices - 1  # Remove padding offset
        vertices = vertices / (resolution - 1) * 2 * b - b  # Scale to [-b, b]
        
        logger.info(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Apply post-processing
        if self.config.smooth_iterations > 0 and TRIMESH_AVAILABLE:
            vertices, normals = self._smooth_mesh(vertices, faces, normals)
        
        if self.config.simplify_ratio < 1.0 and TRIMESH_AVAILABLE:
            vertices, faces, normals = self._simplify_mesh(vertices, faces, normals)
        
        return vertices, faces, normals
    
    def progressive_extract(
        self,
        latent: torch.Tensor,
        resolutions: List[int] = [64, 128, 256]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract meshes at multiple resolutions (progressive LOD).
        
        Args:
            latent: Latent code [latent_dim]
            resolutions: List of resolutions to extract
        
        Returns:
            List of dicts with 'vertices', 'faces', 'normals' for each resolution
        """
        meshes = []
        
        for res in resolutions:
            logger.info(f"Extracting at resolution {res}...")
            vertices, faces, normals = self.extract_mesh(latent, resolution=res)
            
            meshes.append({
                'resolution': res,
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'num_vertices': len(vertices),
                'num_faces': len(faces)
            })
        
        return meshes
    
    def _smooth_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Laplacian smoothing."""
        if not TRIMESH_AVAILABLE:
            return vertices, normals
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        trimesh.smoothing.filter_laplacian(mesh, iterations=self.config.smooth_iterations)
        
        return mesh.vertices, mesh.vertex_normals
    
    def _simplify_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simplify mesh using quadric decimation."""
        if not TRIMESH_AVAILABLE:
            return vertices, faces, normals
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        target_faces = int(len(faces) * self.config.simplify_ratio)
        
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
            return mesh.vertices, mesh.faces, mesh.vertex_normals
        except Exception as e:
            logger.warning(f"Mesh simplification failed: {e}")
            return vertices, faces, normals
    
    def extract_to_trimesh(
        self,
        latent: torch.Tensor,
        resolution: Optional[int] = None
    ) -> 'trimesh.Trimesh':
        """
        Extract mesh and return as trimesh object.
        
        Args:
            latent: Latent code
            resolution: Grid resolution
        
        Returns:
            trimesh.Trimesh object
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for this function")
        
        vertices, faces, normals = self.extract_mesh(latent, resolution)
        
        if len(vertices) == 0:
            return trimesh.Trimesh()
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh
    
    @staticmethod
    def save_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: str,
        normals: Optional[np.ndarray] = None
    ):
        """
        Save mesh to file.
        
        Supports: OBJ, PLY, STL (via trimesh or manual OBJ export)
        
        Args:
            vertices: [V, 3] vertex positions
            faces: [F, 3] face indices
            output_path: Output file path
            normals: [V, 3] vertex normals (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if len(vertices) == 0:
            logger.warning(f"Empty mesh, not saving to {output_path}")
            return
        
        suffix = output_path.suffix.lower()
        
        if TRIMESH_AVAILABLE:
            # Use trimesh for export
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals if normals is not None else None
            )
            mesh.export(str(output_path))
        elif suffix == '.obj':
            # Manual OBJ export
            with open(output_path, 'w') as f:
                f.write("# ClothGeom-NIF mesh export\n")
                
                # Vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Normals
                if normals is not None:
                    for n in normals:
                        f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                
                # Faces (OBJ uses 1-indexed)
                for face in faces:
                    if normals is not None:
                        f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                    else:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        else:
            raise ValueError(f"Unsupported format {suffix}. Install trimesh for more formats.")
        
        logger.info(f"Saved mesh to: {output_path}")


def load_model_for_extraction(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load trained model from checkpoint for mesh extraction.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Torch device
    
    Returns:
        model: Loaded model in eval mode
        config: Model configuration
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import NIFDecoder
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = NIFDecoder(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_config


def batch_extract_meshes(
    model: torch.nn.Module,
    latents: torch.Tensor,
    output_dir: str,
    resolution: int = 128,
    file_format: str = 'obj'
) -> List[str]:
    """
    Extract meshes for multiple latent codes.
    
    Args:
        model: Trained model
        latents: [N, latent_dim] latent codes
        output_dir: Output directory
        resolution: Grid resolution
        file_format: Output format (obj, ply, stl)
    
    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = MeshExtractor(
        model,
        config=MeshExtractionConfig(resolution=resolution)
    )
    
    output_paths = []
    
    for i, latent in enumerate(latents):
        logger.info(f"Extracting mesh {i+1}/{len(latents)}...")
        
        vertices, faces, normals = extractor.extract_mesh(latent)
        
        output_path = output_dir / f'mesh_{i:04d}.{file_format}'
        MeshExtractor.save_mesh(vertices, faces, str(output_path), normals)
        output_paths.append(str(output_path))
    
    return output_paths


if __name__ == '__main__':
    # Test with random latent
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import create_nif_decoder
    
    # Create dummy model
    model = create_nif_decoder({'latent_dim': 128})
    
    # Create extractor
    extractor = MeshExtractor(
        model,
        config=MeshExtractionConfig(resolution=32)  # Low res for testing
    )
    
    # Random latent
    latent = torch.randn(128)
    
    # Evaluate SDF
    print("Evaluating SDF grid...")
    sdf = extractor.evaluate_sdf_grid(latent, resolution=32)
    print(f"SDF shape: {sdf.shape}")
    print(f"SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    
    # Extract mesh
    print("\nExtracting mesh...")
    vertices, faces, normals = extractor.extract_mesh(latent, resolution=32)
    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
