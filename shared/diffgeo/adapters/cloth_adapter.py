"""
Cloth Manifold Adapter
======================

Bridge differential geometry infrastructure to cloth/garment simulation.
Targets Projects 05 (ClothGNN), 09 (HGNN-NIF-Cloth), 11 (NIF-Cloth3D).

Domain context:
- Cloth is a thin shell (2D manifold in 3D)
- Material properties: stretch, shear, bending stiffness
- Deformation: large rotations, moderate strains
- Boundary conditions: seams, attachment points, body collision
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..mesh.trimesh import TriangleMesh

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class MaterialType(Enum):
    """Common cloth material types with typical parameters."""
    COTTON = 'cotton'
    SILK = 'silk'
    DENIM = 'denim'
    LEATHER = 'leather'
    POLYESTER = 'polyester'
    CUSTOM = 'custom'


@dataclass
class ClothMaterialConfig:
    """Physical parameters for cloth material."""
    
    # Material type (for presets)
    material_type: MaterialType = MaterialType.CUSTOM
    
    # Stiffness parameters (N/m or N·m for bending)
    stretch_stiffness: float = 1000.0  # In-plane stretch
    shear_stiffness: float = 500.0     # In-plane shear
    bending_stiffness: float = 0.01    # Out-of-plane bending
    
    # Damping coefficients
    stretch_damping: float = 0.1
    shear_damping: float = 0.05
    bending_damping: float = 0.001
    
    # Physical properties
    density: float = 0.3  # kg/m² (area density)
    thickness: float = 0.001  # m
    friction: float = 0.4  # Coefficient of friction
    
    # Poisson ratio (coupling between stretch directions)
    poisson_ratio: float = 0.3
    
    @classmethod
    def from_material(cls, material: MaterialType) -> 'ClothMaterialConfig':
        """Create config from material preset."""
        presets = {
            MaterialType.COTTON: cls(
                material_type=MaterialType.COTTON,
                stretch_stiffness=800.0,
                shear_stiffness=400.0,
                bending_stiffness=0.02,
                density=0.2,
            ),
            MaterialType.SILK: cls(
                material_type=MaterialType.SILK,
                stretch_stiffness=500.0,
                shear_stiffness=200.0,
                bending_stiffness=0.005,
                density=0.1,
            ),
            MaterialType.DENIM: cls(
                material_type=MaterialType.DENIM,
                stretch_stiffness=2000.0,
                shear_stiffness=1000.0,
                bending_stiffness=0.1,
                density=0.4,
            ),
            MaterialType.LEATHER: cls(
                material_type=MaterialType.LEATHER,
                stretch_stiffness=5000.0,
                shear_stiffness=2500.0,
                bending_stiffness=0.5,
                density=0.8,
            ),
            MaterialType.POLYESTER: cls(
                material_type=MaterialType.POLYESTER,
                stretch_stiffness=1200.0,
                shear_stiffness=600.0,
                bending_stiffness=0.015,
                density=0.15,
            ),
        }
        return presets.get(material, cls())


@dataclass 
class GarmentTopology:
    """Topology information for a garment (seams, panels, etc.)."""
    
    # Panel information
    panel_ids: Optional[np.ndarray] = None  # Per-face panel assignment
    panel_names: Optional[List[str]] = None
    
    # Seam information: pairs of edges that should be sewn together
    seam_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    
    # Boundary types
    boundary_vertex_ids: Optional[np.ndarray] = None
    boundary_types: Optional[np.ndarray] = None  # 0=free, 1=seam, 2=attachment
    
    # Attachment points (pinned vertices)
    pinned_vertices: Optional[np.ndarray] = None


class ClothManifoldAdapter:
    """
    Adapter connecting diffgeo infrastructure to cloth simulation.
    
    Provides:
    1. Rest shape metric for strain computation
    2. Material property handling
    3. Panel/seam topology for garments
    4. Spectral/chart setup for deformed meshes
    
    Example:
        >>> adapter = ClothManifoldAdapter.from_obj(
        ...     'garment.obj',
        ...     material=MaterialType.COTTON,
        ... )
        >>> strain = adapter.compute_strain(deformed_vertices)
        >>> loss_fn = adapter.create_loss_function()
    """
    
    def __init__(
        self,
        mesh: 'TriangleMesh',
        rest_vertices: Optional[np.ndarray] = None,
        material: ClothMaterialConfig = None,
        topology: Optional[GarmentTopology] = None,
    ):
        self.mesh = mesh
        self.rest_vertices = rest_vertices if rest_vertices is not None else mesh.vertices.copy()
        self.material = material if material is not None else ClothMaterialConfig()
        self.topology = topology
        
        # Precompute rest shape properties
        self._rest_edge_lengths = None
        self._rest_face_areas = None
        self._rest_metric = None
        self._dihedral_angles_rest = None
        
        # Lazy-initialized components
        self._atlas = None
        self._spectral_basis = None
        
        self._precompute_rest_shape()
    
    def _precompute_rest_shape(self):
        """Precompute rest shape geometric quantities."""
        V = self.rest_vertices
        F = self.mesh.faces
        
        # Rest edge lengths
        if hasattr(self.mesh, 'edges'):
            self._rest_edge_lengths = np.linalg.norm(
                V[self.mesh.edges[:, 0]] - V[self.mesh.edges[:, 1]],
                axis=1,
            )
        
        # Rest face areas
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        self._rest_face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Rest metric tensor per face (2x2 in tangent space)
        # First fundamental form: [E F; F G] where E=e1·e1, F=e1·e2, G=e2·e2
        e1 = v1 - v0  # First edge
        e2 = v2 - v0  # Second edge
        
        E = np.sum(e1 * e1, axis=1)
        F_coef = np.sum(e1 * e2, axis=1)
        G = np.sum(e2 * e2, axis=1)
        
        # Store as (n_faces, 2, 2) metric tensors
        self._rest_metric = np.stack([
            np.stack([E, F_coef], axis=1),
            np.stack([F_coef, G], axis=1),
        ], axis=2)  # (n_faces, 2, 2)
        
        # Compute rest dihedral angles for bending
        self._compute_rest_dihedral_angles()
    
    def _compute_rest_dihedral_angles(self):
        """Compute dihedral angles between adjacent faces in rest shape."""
        if not hasattr(self.mesh, 'edges') or not hasattr(self.mesh, 'edge_faces'):
            return
        
        angles = []
        V = self.rest_vertices
        F = self.mesh.faces
        
        for e_idx, (f1_idx, f2_idx) in enumerate(self.mesh.edge_faces):
            if f1_idx < 0 or f2_idx < 0:
                angles.append(np.pi)  # Boundary edge
                continue
            
            f1, f2 = F[f1_idx], F[f2_idx]
            
            # Face normals
            def face_normal(face):
                v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                return n / (np.linalg.norm(n) + 1e-10)
            
            n1 = face_normal(f1)
            n2 = face_normal(f2)
            
            # Dihedral angle
            cos_angle = np.clip(np.dot(n1, n2), -1, 1)
            angles.append(np.arccos(cos_angle))
        
        self._dihedral_angles_rest = np.array(angles)
    
    @classmethod
    def from_obj(
        cls,
        obj_path: str,
        material: MaterialType = MaterialType.CUSTOM,
        material_config: Optional[ClothMaterialConfig] = None,
    ) -> 'ClothManifoldAdapter':
        """Load cloth mesh from OBJ file."""
        from ..mesh.trimesh import TriangleMesh
        
        mesh = TriangleMesh.from_obj(obj_path)
        
        if material_config is None:
            material_config = ClothMaterialConfig.from_material(material)
        
        return cls(mesh=mesh, material=material_config)
    
    @classmethod
    def from_pattern(
        cls,
        pattern_vertices: np.ndarray,  # 2D pattern
        pattern_faces: np.ndarray,
        seams: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        material: MaterialType = MaterialType.CUSTOM,
    ) -> 'ClothManifoldAdapter':
        """
        Create adapter from 2D sewing pattern.
        
        The 2D pattern serves as the rest shape (UV coordinates),
        and can be sewn/deformed to create 3D garment.
        """
        from ..mesh.trimesh import TriangleMesh
        
        # Embed 2D pattern in 3D (z=0)
        if pattern_vertices.shape[1] == 2:
            vertices_3d = np.hstack([
                pattern_vertices,
                np.zeros((pattern_vertices.shape[0], 1))
            ])
        else:
            vertices_3d = pattern_vertices
        
        mesh = TriangleMesh(vertices=vertices_3d, faces=pattern_faces)
        
        topology = GarmentTopology(seam_pairs=seams)
        material_config = ClothMaterialConfig.from_material(material)
        
        return cls(
            mesh=mesh,
            rest_vertices=vertices_3d,  # 2D pattern is rest shape
            material=material_config,
            topology=topology,
        )
    
    def compute_strain_tensor(
        self,
        deformed_vertices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Green-Lagrange strain tensor for each face.
        
        E = (1/2)(C - I) where C = F^T F is the Cauchy-Green tensor
        and F is the deformation gradient.
        
        In 2D tangent space, this gives a (n_faces, 2, 2) tensor.
        
        Args:
            deformed_vertices: (V, 3) deformed vertex positions
            
        Returns:
            strain: (n_faces, 2, 2) strain tensor per face
        """
        V_rest = self.rest_vertices
        V_def = deformed_vertices
        F = self.mesh.faces
        
        strain = np.zeros((len(F), 2, 2))
        
        for f_idx, face in enumerate(F):
            # Rest shape edges
            e1_rest = V_rest[face[1]] - V_rest[face[0]]
            e2_rest = V_rest[face[2]] - V_rest[face[0]]
            
            # Deformed edges
            e1_def = V_def[face[1]] - V_def[face[0]]
            e2_def = V_def[face[2]] - V_def[face[0]]
            
            # First fundamental forms
            # Rest: [E_r F_r; F_r G_r]
            E_r = np.dot(e1_rest, e1_rest)
            F_r = np.dot(e1_rest, e2_rest)
            G_r = np.dot(e2_rest, e2_rest)
            
            # Deformed: [E_d F_d; F_d G_d]
            E_d = np.dot(e1_def, e1_def)
            F_d = np.dot(e1_def, e2_def)
            G_d = np.dot(e2_def, e2_def)
            
            # Green-Lagrange strain: E = (1/2)(I_def - I_rest)
            # In metric form: g_deformed - g_rest
            strain[f_idx, 0, 0] = 0.5 * (E_d - E_r) / E_r if E_r > 1e-10 else 0
            strain[f_idx, 0, 1] = 0.5 * (F_d - F_r) / np.sqrt(E_r * G_r) if E_r * G_r > 1e-10 else 0
            strain[f_idx, 1, 0] = strain[f_idx, 0, 1]
            strain[f_idx, 1, 1] = 0.5 * (G_d - G_r) / G_r if G_r > 1e-10 else 0
        
        return strain
    
    def compute_curvature_change(
        self,
        deformed_vertices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute change in dihedral angles (bending).
        
        For each interior edge, compute how much the dihedral angle
        has changed from the rest configuration.
        
        Args:
            deformed_vertices: (V, 3) deformed vertex positions
            
        Returns:
            angle_change: (n_edges,) change in dihedral angle per edge
        """
        if self._dihedral_angles_rest is None:
            raise ValueError("Dihedral angles not computed for this mesh")
        
        V = deformed_vertices
        F = self.mesh.faces
        
        angles_def = []
        for e_idx, (f1_idx, f2_idx) in enumerate(self.mesh.edge_faces):
            if f1_idx < 0 or f2_idx < 0:
                angles_def.append(np.pi)
                continue
            
            f1, f2 = F[f1_idx], F[f2_idx]
            
            def face_normal(face):
                v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                return n / (np.linalg.norm(n) + 1e-10)
            
            n1 = face_normal(f1)
            n2 = face_normal(f2)
            
            cos_angle = np.clip(np.dot(n1, n2), -1, 1)
            angles_def.append(np.arccos(cos_angle))
        
        angles_def = np.array(angles_def)
        return angles_def - self._dihedral_angles_rest
    
    def create_chart_atlas(
        self,
        n_charts: int = 8,
        use_uv: bool = True,
    ):
        """
        Create chart atlas for the cloth mesh.
        
        For garments, charts can correspond to panels if topology is available.
        Otherwise, uses k-means clustering on the rest shape.
        
        Args:
            n_charts: Number of charts (ignored if using panel topology)
            use_uv: Use UV coordinates for chart creation if available
            
        Returns:
            Atlas object
        """
        from ..charts import Atlas
        
        # If we have panel topology, use panels as charts
        if self.topology is not None and self.topology.panel_ids is not None:
            n_panels = len(set(self.topology.panel_ids))
            self._atlas = Atlas.from_mesh(
                vertices=self.rest_vertices,
                faces=self.mesh.faces,
                n_charts=n_panels,
                face_labels=self.topology.panel_ids,
            )
        else:
            self._atlas = Atlas.from_mesh(
                vertices=self.rest_vertices,
                faces=self.mesh.faces,
                n_charts=n_charts,
            )
        
        return self._atlas
    
    def create_spectral_basis(
        self,
        n_eigenpairs: int = 30,
    ):
        """
        Create spectral basis for the cloth mesh.
        
        Useful for:
        - Low-frequency deformation modes
        - Spectral filtering of high-frequency wrinkles
        - Physics-informed regularization
        """
        from ..spectral import SpectralBasis, compute_laplacian_eigenpairs
        
        eigenvalues, eigenvectors = compute_laplacian_eigenpairs(
            self.mesh,
            k=n_eigenpairs,
        )
        
        self._spectral_basis = SpectralBasis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mass_matrix=self.mesh.mass_matrix,
        )
        return self._spectral_basis
    
    def get_pinned_vertices(self) -> Optional[np.ndarray]:
        """Get indices of pinned/attachment vertices."""
        if self.topology is not None and self.topology.pinned_vertices is not None:
            return self.topology.pinned_vertices
        return None


if HAS_TORCH:
    class ClothManifoldLoss(nn.Module):
        """
        Physics-informed loss for cloth simulation on manifold.
        
        Energy terms:
        - Membrane (stretch + shear): Based on first fundamental form change
        - Bending: Based on dihedral angle change
        - Gravity: Potential energy
        - Collision: Penetration penalty
        
        The manifold structure enters through:
        - Metric tensor for strain computation
        - Curvature for bending energy
        - Geodesic distances for collision handling
        """
        
        def __init__(
            self,
            adapter: ClothManifoldAdapter,
            membrane_weight: float = 1.0,
            bending_weight: float = 0.01,
            gravity_weight: float = 0.1,
            collision_weight: float = 10.0,
            gravity: Tuple[float, float, float] = (0, 0, -9.81),
        ):
            super().__init__()
            self.adapter = adapter
            self.membrane_weight = membrane_weight
            self.bending_weight = bending_weight
            self.gravity_weight = gravity_weight
            self.collision_weight = collision_weight
            
            # Register buffers for GPU
            self.register_buffer(
                'gravity',
                torch.tensor(gravity, dtype=torch.float32)
            )
            self.register_buffer(
                'rest_vertices',
                torch.from_numpy(adapter.rest_vertices).float()
            )
            self.register_buffer(
                'faces',
                torch.from_numpy(adapter.mesh.faces).long()
            )
            
            if adapter._rest_edge_lengths is not None:
                self.register_buffer(
                    'rest_edge_lengths',
                    torch.from_numpy(adapter._rest_edge_lengths).float()
                )
            
            if adapter._rest_face_areas is not None:
                self.register_buffer(
                    'rest_face_areas',
                    torch.from_numpy(adapter._rest_face_areas).float()
                )
            
            if adapter._dihedral_angles_rest is not None:
                self.register_buffer(
                    'rest_dihedral_angles',
                    torch.from_numpy(adapter._dihedral_angles_rest).float()
                )
            
            # Material parameters
            mat = adapter.material
            self.stretch_stiffness = mat.stretch_stiffness
            self.shear_stiffness = mat.shear_stiffness
            self.bending_stiffness = mat.bending_stiffness
            self.density = mat.density
        
        def forward(
            self,
            vertices: torch.Tensor,  # (V, 3) or (batch, V, 3)
            velocities: Optional[torch.Tensor] = None,  # (V, 3) for damping
            collision_sdf: Optional[Callable] = None,  # SDF for collision
        ) -> Dict[str, torch.Tensor]:
            """
            Compute cloth physics losses.
            
            Args:
                vertices: Current vertex positions
                velocities: Current vertex velocities (for damping)
                collision_sdf: Signed distance function for collision body
                
            Returns:
                losses: Dict with 'membrane', 'bending', 'gravity', 'collision', 'total'
            """
            batch = vertices.dim() == 3
            if not batch:
                vertices = vertices.unsqueeze(0)
            
            B = vertices.shape[0]
            
            # Membrane energy (stretch + shear)
            membrane_loss = self._membrane_energy(vertices)
            
            # Bending energy
            bending_loss = self._bending_energy(vertices)
            
            # Gravity potential
            gravity_loss = self._gravity_potential(vertices)
            
            # Collision penalty
            if collision_sdf is not None:
                collision_loss = self._collision_penalty(vertices, collision_sdf)
            else:
                collision_loss = torch.zeros(B, device=vertices.device)
            
            # Total loss
            total = (
                self.membrane_weight * membrane_loss +
                self.bending_weight * bending_loss +
                self.gravity_weight * gravity_loss +
                self.collision_weight * collision_loss
            )
            
            if not batch:
                return {
                    'membrane': membrane_loss.squeeze(0),
                    'bending': bending_loss.squeeze(0),
                    'gravity': gravity_loss.squeeze(0),
                    'collision': collision_loss.squeeze(0),
                    'total': total.squeeze(0),
                }
            
            return {
                'membrane': membrane_loss,
                'bending': bending_loss,
                'gravity': gravity_loss,
                'collision': collision_loss,
                'total': total,
            }
        
        def _membrane_energy(self, vertices: torch.Tensor) -> torch.Tensor:
            """
            Compute membrane (stretch + shear) energy.
            
            Uses St. Venant-Kirchhoff model:
            W = (λ/2)(tr E)² + μ tr(E²)
            """
            B, V, _ = vertices.shape
            F = self.faces
            rest_V = self.rest_vertices
            
            # Get triangle vertices
            v0 = vertices[:, F[:, 0]]  # (B, n_faces, 3)
            v1 = vertices[:, F[:, 1]]
            v2 = vertices[:, F[:, 2]]
            
            r0 = rest_V[F[:, 0]]  # (n_faces, 3)
            r1 = rest_V[F[:, 1]]
            r2 = rest_V[F[:, 2]]
            
            # Deformed edges
            e1_def = v1 - v0  # (B, n_faces, 3)
            e2_def = v2 - v0
            
            # Rest edges
            e1_rest = r1 - r0  # (n_faces, 3)
            e2_rest = r2 - r0
            
            # First fundamental form - deformed
            E_d = torch.sum(e1_def * e1_def, dim=-1)  # (B, n_faces)
            F_d = torch.sum(e1_def * e2_def, dim=-1)
            G_d = torch.sum(e2_def * e2_def, dim=-1)
            
            # First fundamental form - rest
            E_r = torch.sum(e1_rest * e1_rest, dim=-1)  # (n_faces,)
            F_r = torch.sum(e1_rest * e2_rest, dim=-1)
            G_r = torch.sum(e2_rest * e2_rest, dim=-1)
            
            # Green-Lagrange strain components
            eps_11 = 0.5 * (E_d - E_r.unsqueeze(0)) / (E_r.unsqueeze(0) + 1e-8)
            eps_12 = 0.5 * (F_d - F_r.unsqueeze(0)) / (torch.sqrt(E_r * G_r).unsqueeze(0) + 1e-8)
            eps_22 = 0.5 * (G_d - G_r.unsqueeze(0)) / (G_r.unsqueeze(0) + 1e-8)
            
            # Energy density: stretch + shear
            stretch_energy = self.stretch_stiffness * (eps_11 ** 2 + eps_22 ** 2)
            shear_energy = self.shear_stiffness * (eps_12 ** 2)
            
            # Integrate over faces (weighted by rest area)
            areas = self.rest_face_areas.unsqueeze(0)  # (1, n_faces)
            energy = torch.sum((stretch_energy + shear_energy) * areas, dim=-1)  # (B,)
            
            return energy
        
        def _bending_energy(self, vertices: torch.Tensor) -> torch.Tensor:
            """
            Compute bending energy based on dihedral angle change.
            
            W_bend = k_bend * Σ_e |θ_e - θ_e^rest|² * l_e
            
            where θ_e is dihedral angle at edge e and l_e is edge length.
            """
            if not hasattr(self, 'rest_dihedral_angles'):
                return torch.zeros(vertices.shape[0], device=vertices.device)
            
            # This is a simplified version - full implementation would
            # compute dihedral angles from vertices
            # For now, use edge-based strain as proxy
            
            B, V, _ = vertices.shape
            edges = torch.from_numpy(self.adapter.mesh.edges).long().to(vertices.device)
            rest_lengths = self.rest_edge_lengths
            
            # Current edge lengths
            edge_vecs = vertices[:, edges[:, 1]] - vertices[:, edges[:, 0]]  # (B, n_edges, 3)
            edge_lengths = torch.norm(edge_vecs, dim=-1)  # (B, n_edges)
            
            # Edge strain (simplified bending proxy)
            edge_strain = (edge_lengths - rest_lengths.unsqueeze(0)) / (rest_lengths.unsqueeze(0) + 1e-8)
            
            # Bending energy (quadratic in strain)
            energy = self.bending_stiffness * torch.sum(edge_strain ** 2, dim=-1)
            
            return energy
        
        def _gravity_potential(self, vertices: torch.Tensor) -> torch.Tensor:
            """
            Compute gravitational potential energy.
            
            U = m * g * h (integrated over cloth)
            """
            B, V, _ = vertices.shape
            
            # Height (z-component relative to reference)
            h = vertices[:, :, 2]  # (B, V)
            
            # Vertex masses (from face areas)
            # Each vertex gets 1/3 of adjacent face areas
            vertex_masses = torch.zeros(V, device=vertices.device)
            for i in range(3):
                vertex_masses.scatter_add_(
                    0,
                    self.faces[:, i],
                    self.rest_face_areas / 3 * self.density
                )
            
            # Potential energy: m * g * z
            g_mag = torch.norm(self.gravity)
            potential = torch.sum(vertex_masses.unsqueeze(0) * g_mag * h, dim=-1)  # (B,)
            
            return potential
        
        def _collision_penalty(
            self,
            vertices: torch.Tensor,
            collision_sdf: Callable,
        ) -> torch.Tensor:
            """
            Compute collision penalty using signed distance function.
            
            Penalizes vertices that penetrate the collision body (negative SDF).
            """
            B, V, _ = vertices.shape
            
            # Evaluate SDF at all vertices
            sdf_values = collision_sdf(vertices.reshape(-1, 3))  # (B*V,)
            sdf_values = sdf_values.reshape(B, V)
            
            # Penalty for penetration (negative SDF)
            penetration = F.relu(-sdf_values)  # Only penalize negative values
            penalty = torch.sum(penetration ** 2, dim=-1)  # (B,)
            
            return penalty
    
    
    def create_cloth_pipeline(
        adapter: ClothManifoldAdapter,
        model_type: str = 'tangent_gnn',
        hidden_dims: List[int] = [64, 64, 64],
        use_spectral: bool = True,
        n_eigenpairs: int = 30,
    ) -> Tuple[nn.Module, ClothManifoldLoss]:
        """
        Create a complete cloth simulation pipeline.
        
        Builds:
        1. Neural network model for predicting vertex displacements
        2. Physics-informed loss function
        
        Args:
            adapter: ClothManifoldAdapter with mesh and material
            model_type: 'tangent_gnn', 'atlas_pinn', or 'spectral'
            hidden_dims: Hidden layer dimensions
            use_spectral: Whether to use spectral features
            n_eigenpairs: Number of spectral basis functions
            
        Returns:
            model: Neural network predicting (Δx, Δy, Δz) displacements
            loss_fn: ClothManifoldLoss
        """
        if adapter.mesh is None:
            raise ValueError("Adapter must have a mesh")
        
        # Optionally create spectral basis
        if use_spectral:
            adapter.create_spectral_basis(n_eigenpairs=n_eigenpairs)
        
        # Create model based on type
        if model_type == 'tangent_gnn':
            from ..tangent import TangentMessagePassingStack
            
            model = TangentMessagePassingStack(
                in_channels=3,  # Input: current position or displacement
                hidden_channels=hidden_dims[0],
                out_channels=3,  # Output: displacement prediction
                n_layers=len(hidden_dims),
            )
        
        elif model_type == 'atlas_pinn':
            from ..charts import AtlasPINN
            
            atlas = adapter.create_chart_atlas()
            
            model = AtlasPINN(
                atlas=atlas,
                input_dim=2,  # Local chart coordinates
                hidden_dims=hidden_dims,
                output_dim=3,  # Displacement
            )
        
        elif model_type == 'spectral':
            from ..spectral import SpectralConvStack
            
            if adapter._spectral_basis is None:
                adapter.create_spectral_basis(n_eigenpairs=n_eigenpairs)
            
            model = SpectralConvStack(
                in_channels=3,
                hidden_channels=hidden_dims[0],
                out_channels=3,
                n_layers=len(hidden_dims),
                eigenvectors=torch.from_numpy(
                    adapter._spectral_basis.eigenvectors
                ).float(),
                eigenvalues=torch.from_numpy(
                    adapter._spectral_basis.eigenvalues
                ).float(),
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create loss function
        loss_fn = ClothManifoldLoss(adapter=adapter)
        
        return model, loss_fn

else:
    # No-torch placeholders
    class ClothManifoldLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for ClothManifoldLoss")
    
    def create_cloth_pipeline(*args, **kwargs):
        raise ImportError("PyTorch required for create_cloth_pipeline")
