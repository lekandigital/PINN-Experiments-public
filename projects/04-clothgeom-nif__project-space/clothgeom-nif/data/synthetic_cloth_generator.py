"""
Synthetic Cloth Data Generator

Generates training data for ClothGeom-NIF:
1. Creates synthetic cloth meshes (grid topology)
2. Applies random deformations (sine waves, gravity, wind)
3. Computes node positions and edge strains as latent code
4. Generates ground-truth SDF volumes via distance computation
5. Outputs (latent_code, sdf_volume) pairs for training

The synthetic data mimics real cloth simulation outputs without
requiring an external simulator.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class ClothConfig:
    """Configuration for cloth mesh generation."""
    grid_size: int = 32  # N x N nodes
    cloth_width: float = 2.0  # Physical width in world units
    cloth_height: float = 2.0  # Physical height
    thickness: float = 0.02  # Cloth thickness for SDF
    rest_edge_length: Optional[float] = None  # Auto-computed if None


@dataclass 
class DeformationConfig:
    """Configuration for cloth deformations."""
    # Sine wave deformation
    wave_amplitude: float = 0.3
    wave_frequency: float = 2.0
    wave_phase_range: Tuple[float, float] = (0, 2 * np.pi)
    
    # Gravity drape
    gravity_strength: float = 0.5
    gravity_center: Tuple[float, float] = (0.5, 0.5)  # Relative position
    
    # Wind displacement
    wind_strength: float = 0.2
    wind_direction: Tuple[float, float, float] = (1, 0, 0.5)
    
    # Random noise
    noise_amplitude: float = 0.05
    
    # Pinning (fixed corners)
    pin_corners: bool = True
    pin_top_edge: bool = False


class SyntheticClothGenerator:
    """
    Generates synthetic cloth samples with SDF ground truth.
    
    Each sample consists of:
    - Node positions: [N*N, 3] array of 3D positions
    - Edge strains: [E, 1] array of strain values (length ratio - 1)
    - Latent code: Concatenated and normalized positions + strains
    - SDF volume: [R, R, R] grid of signed distances
    
    Args:
        cloth_config: Cloth mesh configuration
        deform_config: Deformation parameters
        sdf_resolution: Resolution of SDF volume (default: 64)
        sdf_bounds: Bounding box for SDF [-b, b]³ (default: 1.5)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        cloth_config: Optional[ClothConfig] = None,
        deform_config: Optional[DeformationConfig] = None,
        sdf_resolution: int = 64,
        sdf_bounds: float = 1.5,
        seed: Optional[int] = None
    ):
        self.cloth_cfg = cloth_config or ClothConfig()
        self.deform_cfg = deform_config or DeformationConfig()
        self.sdf_resolution = sdf_resolution
        self.sdf_bounds = sdf_bounds
        
        if seed is not None:
            np.random.seed(seed)
        
        # Compute rest edge length if not specified
        if self.cloth_cfg.rest_edge_length is None:
            self.cloth_cfg.rest_edge_length = (
                self.cloth_cfg.cloth_width / (self.cloth_cfg.grid_size - 1)
            )
        
        # Pre-compute grid for SDF evaluation
        self._setup_sdf_grid()
        
        # Pre-compute mesh topology
        self._setup_mesh_topology()
    
    def _setup_sdf_grid(self):
        """Create 3D grid for SDF evaluation."""
        r = self.sdf_resolution
        b = self.sdf_bounds
        
        # Create grid coordinates
        coords_1d = np.linspace(-b, b, r)
        xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        
        # Flatten for vectorized distance computation
        self.grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        self.grid_shape = (r, r, r)
    
    def _setup_mesh_topology(self):
        """Pre-compute mesh edges for strain calculation."""
        n = self.cloth_cfg.grid_size
        
        # Generate edge pairs (horizontal and vertical connections)
        edges = []
        
        for i in range(n):
            for j in range(n):
                node_idx = i * n + j
                
                # Horizontal edge (to right neighbor)
                if j < n - 1:
                    edges.append((node_idx, node_idx + 1))
                
                # Vertical edge (to bottom neighbor)
                if i < n - 1:
                    edges.append((node_idx, node_idx + n))
        
        self.edges = np.array(edges)
        self.num_edges = len(edges)
    
    def create_rest_mesh(self) -> np.ndarray:
        """
        Create flat cloth mesh in rest configuration.
        
        Returns:
            nodes: [N*N, 3] array of node positions
        """
        n = self.cloth_cfg.grid_size
        w = self.cloth_cfg.cloth_width
        h = self.cloth_cfg.cloth_height
        
        # Create grid in XY plane, centered at origin
        x = np.linspace(-w/2, w/2, n)
        y = np.linspace(-h/2, h/2, n)
        xx, yy = np.meshgrid(x, y)
        
        # Z = 0 for flat mesh
        zz = np.zeros_like(xx)
        
        nodes = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        return nodes
    
    def apply_deformation(
        self,
        nodes: np.ndarray,
        deform_type: str = 'random',
        params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply deformation to cloth mesh.
        
        Args:
            nodes: [N, 3] rest node positions
            deform_type: Type of deformation:
                - 'sine': Sinusoidal waves
                - 'gravity': Gravity drape (catenary-like)
                - 'wind': Wind displacement
                - 'random': Random combination
            params: Override deformation parameters
        
        Returns:
            deformed_nodes: [N, 3] deformed positions
        """
        cfg = self.deform_cfg
        if params:
            # Update config with provided params
            for k, v in params.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        
        deformed = nodes.copy()
        n = self.cloth_cfg.grid_size
        
        if deform_type == 'random':
            # Randomly choose which deformations to apply
            deform_type = np.random.choice(['sine', 'gravity', 'wind', 'combined'])
        
        if deform_type in ['sine', 'combined']:
            # Sinusoidal wave deformation
            phase = np.random.uniform(*cfg.wave_phase_range)
            freq = cfg.wave_frequency * (0.5 + np.random.random())
            amp = cfg.wave_amplitude * (0.3 + 0.7 * np.random.random())
            
            # Apply along random axis combination
            wave_dir = np.random.randn(2)
            wave_dir /= np.linalg.norm(wave_dir)
            
            wave_input = wave_dir[0] * nodes[:, 0] + wave_dir[1] * nodes[:, 1]
            deformed[:, 2] += amp * np.sin(freq * wave_input + phase)
        
        if deform_type in ['gravity', 'combined']:
            # Gravity-like drape
            strength = cfg.gravity_strength * (0.5 + np.random.random())
            cx, cy = cfg.gravity_center
            cx += 0.2 * (np.random.random() - 0.5)
            cy += 0.2 * (np.random.random() - 0.5)
            
            # Distance from center
            dx = (nodes[:, 0] / self.cloth_cfg.cloth_width + 0.5) - cx
            dy = (nodes[:, 1] / self.cloth_cfg.cloth_height + 0.5) - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            # Parabolic droop
            deformed[:, 2] -= strength * (1 - dist**2)
        
        if deform_type in ['wind', 'combined']:
            # Wind displacement
            strength = cfg.wind_strength * np.random.random()
            direction = np.array(cfg.wind_direction)
            direction += 0.3 * np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Height-dependent wind effect
            height_factor = 0.5 + 0.5 * (nodes[:, 1] / self.cloth_cfg.cloth_height + 0.5)
            deformed += strength * height_factor[:, None] * direction
        
        # Add small random noise
        deformed += cfg.noise_amplitude * np.random.randn(*deformed.shape)
        
        # Apply pinning constraints
        if cfg.pin_corners:
            corners = [0, n-1, n*(n-1), n*n-1]
            deformed[corners] = nodes[corners]
        
        if cfg.pin_top_edge:
            top_edge = list(range(n))
            deformed[top_edge] = nodes[top_edge]
        
        return deformed
    
    def compute_edge_strains(
        self,
        nodes: np.ndarray,
        rest_nodes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute edge strain values.
        
        Strain = (current_length / rest_length) - 1
        Positive = stretched, Negative = compressed
        
        Args:
            nodes: Current node positions [N, 3]
            rest_nodes: Rest positions (for rest length computation)
        
        Returns:
            strains: [E, 1] strain values per edge
        """
        if rest_nodes is None:
            rest_length = self.cloth_cfg.rest_edge_length
        else:
            # Compute rest lengths from rest configuration
            rest_edges = rest_nodes[self.edges[:, 1]] - rest_nodes[self.edges[:, 0]]
            rest_length = np.linalg.norm(rest_edges, axis=1, keepdims=True)
        
        # Current edge vectors
        edge_vectors = nodes[self.edges[:, 1]] - nodes[self.edges[:, 0]]
        current_length = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        
        # Strain computation
        if isinstance(rest_length, (int, float)):
            strains = (current_length / rest_length) - 1
        else:
            strains = (current_length / (rest_length + 1e-8)) - 1
        
        return strains.astype(np.float32)
    
    def nodes_to_latent(
        self,
        nodes: np.ndarray,
        strains: np.ndarray,
        latent_dim: int = 128
    ) -> np.ndarray:
        """
        Convert node positions and strains to latent code.
        
        The latent code is constructed by:
        1. Flattening node positions
        2. Concatenating strain values
        3. PCA or random projection to fixed dimension
        4. Normalization
        
        For simplicity, we use a fixed random projection matrix.
        
        Args:
            nodes: [N, 3] node positions
            strains: [E, 1] edge strains
            latent_dim: Target latent dimension
        
        Returns:
            latent: [latent_dim] normalized latent code
        """
        # Flatten inputs
        nodes_flat = nodes.flatten()
        strains_flat = strains.flatten()
        
        # Concatenate
        features = np.concatenate([nodes_flat, strains_flat])
        
        # Random projection (deterministic for reproducibility)
        np.random.seed(42)
        projection = np.random.randn(len(features), latent_dim)
        projection /= np.sqrt(len(features))  # Scale normalization
        np.random.seed(None)  # Reset seed
        
        # Project
        latent = features @ projection
        
        # L2 normalize
        latent = latent / (np.linalg.norm(latent) + 1e-8)
        
        return latent.astype(np.float32)
    
    def compute_sdf(
        self,
        nodes: np.ndarray,
        method: str = 'point_cloud'
    ) -> np.ndarray:
        """
        Compute signed distance field from cloth mesh.
        
        Args:
            nodes: [N, 3] deformed node positions
            method: SDF computation method:
                - 'point_cloud': Distance to nearest point (fast, approximate)
                - 'mesh': Signed distance to triangulated mesh (accurate)
        
        Returns:
            sdf: [R, R, R] SDF volume
        """
        if method == 'point_cloud':
            return self._compute_sdf_point_cloud(nodes)
        elif method == 'mesh':
            return self._compute_sdf_mesh(nodes)
        else:
            raise ValueError(f"Unknown SDF method: {method}")
    
    def _compute_sdf_point_cloud(self, nodes: np.ndarray) -> np.ndarray:
        """
        Fast approximate SDF using nearest point distance.
        
        Distance is unsigned, but we create a pseudo-signed field
        by checking if points are above or below the local surface.
        """
        from scipy.spatial import cKDTree
        
        # Build KD-tree for fast nearest neighbor queries
        tree = cKDTree(nodes)
        
        # Query distances for all grid points
        distances, indices = tree.query(self.grid_points, k=1)
        
        # Estimate sign using local normal direction
        # Approximate normal as average of nearby node normals
        n = self.cloth_cfg.grid_size
        
        # Compute per-node normals using grid structure
        node_normals = self._estimate_normals(nodes)
        
        # Get normal at nearest node for each grid point
        nearest_normals = node_normals[indices]
        
        # Vector from nearest node to grid point
        vectors = self.grid_points - nodes[indices]
        
        # Sign based on dot product with normal
        signs = np.sign(np.sum(vectors * nearest_normals, axis=1))
        signs[signs == 0] = 1  # Handle exactly on surface
        
        # Create signed distance
        signed_distances = signs * distances
        
        # Reshape to volume
        sdf = signed_distances.reshape(self.grid_shape)
        
        return sdf.astype(np.float32)
    
    def _estimate_normals(self, nodes: np.ndarray) -> np.ndarray:
        """Estimate surface normals for cloth nodes."""
        n = self.cloth_cfg.grid_size
        nodes_grid = nodes.reshape(n, n, 3)
        
        normals = np.zeros_like(nodes)
        normals_grid = normals.reshape(n, n, 3)
        
        # Use finite differences to compute tangent vectors
        for i in range(n):
            for j in range(n):
                # Tangent in u direction
                if j < n - 1:
                    du = nodes_grid[i, j+1] - nodes_grid[i, j]
                else:
                    du = nodes_grid[i, j] - nodes_grid[i, j-1]
                
                # Tangent in v direction
                if i < n - 1:
                    dv = nodes_grid[i+1, j] - nodes_grid[i, j]
                else:
                    dv = nodes_grid[i, j] - nodes_grid[i-1, j]
                
                # Normal as cross product
                normal = np.cross(du, dv)
                norm = np.linalg.norm(normal)
                if norm > 1e-8:
                    normal /= norm
                else:
                    normal = np.array([0, 0, 1])
                
                normals_grid[i, j] = normal
        
        return normals
    
    def _compute_sdf_mesh(self, nodes: np.ndarray) -> np.ndarray:
        """
        Accurate SDF using triangulated mesh.
        Requires trimesh library.
        """
        try:
            import trimesh
        except ImportError:
            warnings.warn("trimesh not available, falling back to point_cloud method")
            return self._compute_sdf_point_cloud(nodes)
        
        # Create triangulated mesh from grid
        n = self.cloth_cfg.grid_size
        faces = []
        
        for i in range(n - 1):
            for j in range(n - 1):
                # Two triangles per quad
                v0 = i * n + j
                v1 = v0 + 1
                v2 = v0 + n
                v3 = v2 + 1
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        faces = np.array(faces)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=nodes, faces=faces)
        
        # Compute signed distance
        # Note: This can be slow for high-resolution grids
        try:
            from trimesh.proximity import signed_distance
            sdf = signed_distance(mesh, self.grid_points)
        except ImportError:
            # Fallback to unsigned distance
            closest, distances, _ = mesh.nearest.on_surface(self.grid_points)
            # Estimate sign
            normals = mesh.face_normals[_]
            vectors = self.grid_points - closest
            signs = np.sign(np.sum(vectors * normals, axis=1))
            sdf = signs * distances
        
        return sdf.reshape(self.grid_shape).astype(np.float32)
    
    def generate_sample(
        self,
        deform_type: str = 'random',
        latent_dim: int = 128,
        sdf_method: str = 'point_cloud'
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete training sample.
        
        Returns:
            dict with keys:
                - 'nodes': [N, 3] deformed node positions
                - 'rest_nodes': [N, 3] rest configuration
                - 'strains': [E, 1] edge strains
                - 'latent': [latent_dim] latent code
                - 'sdf': [R, R, R] SDF volume
                - 'edges': [E, 2] edge connectivity
        """
        # Create rest mesh
        rest_nodes = self.create_rest_mesh()
        
        # Apply deformation
        nodes = self.apply_deformation(rest_nodes, deform_type)
        
        # Compute edge strains
        strains = self.compute_edge_strains(nodes, rest_nodes)
        
        # Create latent code
        latent = self.nodes_to_latent(nodes, strains, latent_dim)
        
        # Compute SDF
        sdf = self.compute_sdf(nodes, method=sdf_method)
        
        return {
            'nodes': nodes.astype(np.float32),
            'rest_nodes': rest_nodes.astype(np.float32),
            'strains': strains.astype(np.float32),
            'latent': latent.astype(np.float32),
            'sdf': sdf.astype(np.float32),
            'edges': self.edges.astype(np.int32)
        }
    
    def generate_batch(
        self,
        num_samples: int,
        latent_dim: int = 128,
        sdf_method: str = 'point_cloud',
        progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate multiple training samples.
        
        Args:
            num_samples: Number of samples to generate
            latent_dim: Latent code dimension
            sdf_method: SDF computation method
            progress: Show progress bar
        
        Returns:
            dict with batched arrays
        """
        samples = {
            'nodes': [],
            'strains': [],
            'latent': [],
            'sdf': []
        }
        
        iterator = range(num_samples)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating samples")
            except ImportError:
                pass
        
        for _ in iterator:
            sample = self.generate_sample(
                deform_type='random',
                latent_dim=latent_dim,
                sdf_method=sdf_method
            )
            
            for key in samples:
                samples[key].append(sample[key])
        
        # Stack into batched arrays
        return {k: np.stack(v, axis=0) for k, v in samples.items()}


def generate_dataset(
    output_path: str,
    num_samples: int = 100,
    sdf_resolution: int = 64,
    latent_dim: int = 128,
    grid_size: int = 32,
    seed: int = 42
) -> str:
    """
    Generate and save training dataset to HDF5 file.
    
    Args:
        output_path: Path to save HDF5 file
        num_samples: Number of samples to generate
        sdf_resolution: Resolution of SDF volumes
        latent_dim: Latent code dimension
        grid_size: Cloth grid size (N×N nodes)
        seed: Random seed
    
    Returns:
        Path to saved file
    """
    import h5py
    import os
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Configure generator
    cloth_cfg = ClothConfig(grid_size=grid_size)
    generator = SyntheticClothGenerator(
        cloth_config=cloth_cfg,
        sdf_resolution=sdf_resolution,
        seed=seed
    )
    
    # Generate samples
    print(f"Generating {num_samples} samples with {sdf_resolution}³ SDF volumes...")
    data = generator.generate_batch(
        num_samples=num_samples,
        latent_dim=latent_dim,
        sdf_method='point_cloud',
        progress=True
    )
    
    # Save to HDF5
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Store arrays
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression='gzip')
        
        # Store metadata
        f.attrs['num_samples'] = num_samples
        f.attrs['sdf_resolution'] = sdf_resolution
        f.attrs['latent_dim'] = latent_dim
        f.attrs['grid_size'] = grid_size
        f.attrs['sdf_bounds'] = generator.sdf_bounds
    
    print(f"✅ Dataset saved: {output_path}")
    print(f"   Samples: {num_samples}")
    print(f"   SDF shape: {data['sdf'].shape}")
    print(f"   Latent shape: {data['latent'].shape}")
    
    return output_path


if __name__ == '__main__':
    # Quick test
    generator = SyntheticClothGenerator(sdf_resolution=32)
    sample = generator.generate_sample()
    
    print("Sample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")
