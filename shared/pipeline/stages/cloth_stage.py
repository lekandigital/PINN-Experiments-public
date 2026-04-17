"""
Cloth Stage - Project 09 (HGNN-NIF-Cloth) Wrapper

Wraps the hierarchical GNN + SIREN cloth simulation model.
Handles cloth mesh graph construction, body collision, and mesh extraction.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ..config import ClothSpec, PipelineConfig

# Try importing Project 09 model
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3] / "projects" / "09-hgnn-nif-cloth__project-space"))
    from models.hgnn_nif import HGNN_NIF_ClothModel
    HAS_PROJECT09 = True
except ImportError:
    HAS_PROJECT09 = False
    HGNN_NIF_ClothModel = None


@dataclass
class ClothOutput:
    """Output from cloth simulation stage."""
    vertices: torch.Tensor          # (V_cloth, 3) cloth vertex positions
    normals: torch.Tensor           # (V_cloth, 3) vertex normals
    faces: torch.Tensor             # (F, 3) triangle indices
    velocities: torch.Tensor        # (V_cloth, 3) vertex velocities
    sdf_grid: Optional[torch.Tensor] = None  # (D, H, W) SDF volume if requested
    penetration_mask: Optional[torch.Tensor] = None  # (V_cloth,) bool mask of corrected vertices
    strain_energy: Optional[float] = None  # Total cloth strain energy


class ClothGraphBuilder:
    """Builds graph structure for cloth mesh."""
    
    def __init__(self, faces: np.ndarray, num_vertices: int):
        """
        Args:
            faces: (F, 3) triangle indices
            num_vertices: Total number of vertices
        """
        self.faces = faces
        self.num_vertices = num_vertices
        self._build_edges()
    
    def _build_edges(self):
        """Build edge list and adjacency from faces."""
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                v0, v1 = int(f[i]), int(f[(i + 1) % 3])
                edges_set.add((min(v0, v1), max(v0, v1)))
        
        self.edges = np.array(list(edges_set), dtype=np.int64)
        
        # Build adjacency list
        self.adjacency = [[] for _ in range(self.num_vertices)]
        for e in self.edges:
            self.adjacency[e[0]].append(e[1])
            self.adjacency[e[1]].append(e[0])
    
    def get_edge_index(self, device: torch.device) -> torch.Tensor:
        """Get edge_index in PyG format (2, E)."""
        # Bidirectional edges
        edge_index = np.concatenate([
            self.edges,
            self.edges[:, ::-1]
        ], axis=0).T
        return torch.tensor(edge_index, dtype=torch.long, device=device)
    
    def build_coarse_graph(self, num_coarse: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build coarse graph via farthest point sampling.
        
        Returns:
            coarse_indices: (num_coarse,) indices into fine vertices
            coarse_edges: (E_coarse, 2) edges for coarse graph
        """
        # Farthest point sampling
        coarse_indices = self._farthest_point_sample(num_coarse)
        
        # Build coarse edges based on original adjacency
        coarse_set = set(coarse_indices)
        coarse_edges_set = set()
        
        for idx in coarse_indices:
            # Find coarse neighbors within 2-hop
            neighbors = set()
            frontier = {idx}
            for _ in range(3):  # 3-hop neighborhood
                next_frontier = set()
                for v in frontier:
                    for n in self.adjacency[v]:
                        if n in coarse_set and n != idx:
                            neighbors.add(n)
                        next_frontier.add(n)
                frontier = next_frontier
            
            for n in neighbors:
                coarse_edges_set.add((min(idx, n), max(idx, n)))
        
        coarse_edges = np.array(list(coarse_edges_set), dtype=np.int64)
        return coarse_indices, coarse_edges
    
    def _farthest_point_sample(self, num_samples: int) -> np.ndarray:
        """Farthest point sampling on graph."""
        # Use graph distance approximation
        selected = [0]  # Start with vertex 0
        min_distances = np.full(self.num_vertices, np.inf)
        
        for _ in range(num_samples - 1):
            # Update distances from last selected point
            last = selected[-1]
            distances = self._bfs_distances(last)
            min_distances = np.minimum(min_distances, distances)
            
            # Select farthest point
            min_distances[selected] = -1  # Exclude already selected
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)
        
        return np.array(selected, dtype=np.int64)
    
    def _bfs_distances(self, start: int) -> np.ndarray:
        """BFS to compute graph distances from start vertex."""
        distances = np.full(self.num_vertices, np.inf)
        distances[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            v = queue[head]
            head += 1
            for n in self.adjacency[v]:
                if distances[n] == np.inf:
                    distances[n] = distances[v] + 1
                    queue.append(n)
        
        return distances


class MockClothModel(nn.Module):
    """Mock cloth model for testing without trained weights."""
    
    def __init__(self, rest_vertices: torch.Tensor, faces: torch.Tensor,
                 stiffness: float = 100.0, damping: float = 0.99):
        super().__init__()
        self.register_buffer('rest_vertices', rest_vertices)
        self.register_buffer('faces', faces)
        self.stiffness = stiffness
        self.damping = damping
        
        # Build edge data
        self._build_edges()
        
        # State
        self.vertices = rest_vertices.clone()
        self.velocities = torch.zeros_like(rest_vertices)
    
    def _build_edges(self):
        """Build edge list and rest lengths."""
        edges_set = set()
        for f in self.faces.cpu().numpy():
            for i in range(3):
                v0, v1 = int(f[i]), int(f[(i + 1) % 3])
                edges_set.add((min(v0, v1), max(v0, v1)))
        
        self.edges = torch.tensor(list(edges_set), dtype=torch.long,
                                   device=self.rest_vertices.device)
        
        # Compute rest lengths
        v0 = self.rest_vertices[self.edges[:, 0]]
        v1 = self.rest_vertices[self.edges[:, 1]]
        self.rest_lengths = (v1 - v0).norm(dim=1)
    
    def forward(self, body_vertices: torch.Tensor, body_sdf_fn: callable,
                pinned_mask: torch.Tensor, pinned_positions: torch.Tensor,
                dt: float = 1/30) -> torch.Tensor:
        """
        Simple mass-spring cloth simulation step.
        
        Args:
            body_vertices: (V_body, 3) body mesh for collision
            body_sdf_fn: Function to query body SDF
            pinned_mask: (V_cloth,) bool mask of pinned vertices
            pinned_positions: (V_cloth, 3) target positions for pinned verts
            dt: Time step
            
        Returns:
            vertices: (V_cloth, 3) new cloth positions
        """
        # Spring forces
        forces = torch.zeros_like(self.vertices)
        
        v0_idx, v1_idx = self.edges[:, 0], self.edges[:, 1]
        v0, v1 = self.vertices[v0_idx], self.vertices[v1_idx]
        
        edge_vec = v1 - v0
        edge_len = edge_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        edge_dir = edge_vec / edge_len
        
        strain = edge_len.squeeze() - self.rest_lengths
        spring_force = self.stiffness * strain.unsqueeze(1) * edge_dir
        
        # Accumulate forces
        forces.index_add_(0, v0_idx, spring_force)
        forces.index_add_(0, v1_idx, -spring_force)
        
        # Gravity
        forces[:, 1] -= 9.81  # Assuming Y-up
        
        # Integration
        self.velocities = self.damping * self.velocities + dt * forces
        self.vertices = self.vertices + dt * self.velocities
        
        # Pin constraints
        self.vertices[pinned_mask] = pinned_positions[pinned_mask]
        self.velocities[pinned_mask] = 0
        
        # Collision with body
        if body_sdf_fn is not None:
            sdf_vals = body_sdf_fn(self.vertices)
            penetrating = sdf_vals < 0
            if penetrating.any():
                # Simple push-out (approximate)
                normals = self._estimate_collision_normals(
                    self.vertices[penetrating], body_vertices
                )
                push_dist = -sdf_vals[penetrating].unsqueeze(1) + 0.001
                self.vertices[penetrating] += push_dist * normals
                self.velocities[penetrating] *= 0.1  # Friction
        
        return self.vertices.clone()
    
    def _estimate_collision_normals(self, points: torch.Tensor,
                                     body_vertices: torch.Tensor) -> torch.Tensor:
        """Estimate collision normals by finding closest body vertex."""
        dists = torch.cdist(points, body_vertices)
        closest_idx = dists.argmin(dim=1)
        closest_body = body_vertices[closest_idx]
        
        normals = points - closest_body
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
        return normals
    
    def reset(self, new_rest: Optional[torch.Tensor] = None):
        """Reset simulation state."""
        if new_rest is not None:
            self.rest_vertices = new_rest
            self._build_edges()
        self.vertices = self.rest_vertices.clone()
        self.velocities = torch.zeros_like(self.rest_vertices)


class ClothStage:
    """
    Cloth simulation stage wrapping Project 09's HGNN-NIF model.
    
    Handles:
    - Cloth mesh graph construction (fine + coarse levels)
    - Body collision detection and correction
    - Optional SDF volume output for rendering
    """
    
    def __init__(self, config: PipelineConfig, device: torch.device = None):
        """
        Args:
            config: Pipeline configuration
            device: Compute device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cloth_spec = config.cloth
        
        self.model: Optional[nn.Module] = None
        self.graph_builder: Optional[ClothGraphBuilder] = None
        
        # Cloth state
        self.rest_vertices: Optional[torch.Tensor] = None
        self.faces: Optional[torch.Tensor] = None
        self.pinned_mask: Optional[torch.Tensor] = None
        self.prev_vertices: Optional[torch.Tensor] = None
        self.prev_time: float = 0.0
        
        # Coarse graph for HGNN
        self.coarse_indices: Optional[np.ndarray] = None
        self.coarse_edge_index: Optional[torch.Tensor] = None
    
    def load(self, checkpoint_path: Optional[str] = None,
             cloth_mesh_path: Optional[str] = None,
             pinned_vertex_ids: Optional[List[int]] = None):
        """
        Load cloth model and mesh.
        
        Args:
            checkpoint_path: Path to model checkpoint
            cloth_mesh_path: Path to rest-pose cloth mesh (OBJ format)
            pinned_vertex_ids: Vertex indices to pin (e.g., shoulder attachment)
        """
        # Load cloth mesh
        if cloth_mesh_path and Path(cloth_mesh_path).exists():
            self._load_cloth_mesh(cloth_mesh_path)
        else:
            self._create_default_cloth()
        
        # Set pinned vertices
        if pinned_vertex_ids:
            self.pinned_mask = torch.zeros(self.rest_vertices.shape[0],
                                            dtype=torch.bool, device=self.device)
            self.pinned_mask[pinned_vertex_ids] = True
        else:
            self._auto_pin_top_vertices()
        
        # Build graph structure
        faces_np = self.faces.cpu().numpy()
        self.graph_builder = ClothGraphBuilder(faces_np, self.rest_vertices.shape[0])
        
        # Build coarse graph for HGNN
        self.coarse_indices, coarse_edges = self.graph_builder.build_coarse_graph(
            num_coarse=self.cloth_spec.coarse_graph_size
        )
        
        # Load model
        if checkpoint_path and Path(checkpoint_path).exists() and HAS_PROJECT09:
            self._load_hgnn_model(checkpoint_path)
        else:
            print(f"[ClothStage] Using mock cloth model (checkpoint not found or Project 09 unavailable)")
            self._create_mock_model()
        
        # Initialize state
        self.prev_vertices = self.rest_vertices.clone()
        self.prev_time = 0.0
    
    def _load_cloth_mesh(self, path: str):
        """Load cloth mesh from OBJ file."""
        vertices = []
        faces = []
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    face_verts = []
                    for p in parts[1:4]:
                        face_verts.append(int(p.split('/')[0]) - 1)
                    faces.append(face_verts)
        
        self.rest_vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)
    
    def _create_default_cloth(self):
        """Create a default rectangular cloth mesh."""
        res = self.cloth_spec.resolution
        width = self.cloth_spec.width
        height = self.cloth_spec.height
        
        # Create grid
        xs = torch.linspace(-width/2, width/2, res, device=self.device)
        zs = torch.linspace(0, -height, res, device=self.device)
        
        grid_x, grid_z = torch.meshgrid(xs, zs, indexing='xy')
        vertices = torch.stack([
            grid_x.flatten(),
            torch.full((res * res,), 1.5, device=self.device),
            grid_z.flatten()
        ], dim=1)
        
        self.rest_vertices = vertices
        
        # Create faces
        faces = []
        for i in range(res - 1):
            for j in range(res - 1):
                v0 = i * res + j
                v1 = v0 + 1
                v2 = v0 + res
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)
    
    def _auto_pin_top_vertices(self):
        """Automatically pin top-most vertices."""
        y_coords = self.rest_vertices[:, 1]
        y_max = y_coords.max()
        threshold = y_max - 0.05 * (y_max - y_coords.min())
        self.pinned_mask = y_coords >= threshold
    
    def _load_hgnn_model(self, checkpoint_path: str):
        """Load the HGNN-NIF model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_config = checkpoint.get('config', {})
        
        self.model = HGNN_NIF_ClothModel(
            input_dim=model_config.get('input_dim', 3),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 4),
            output_dim=model_config.get('output_dim', 1),
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def _create_mock_model(self):
        """Create mock cloth model."""
        self.model = MockClothModel(
            self.rest_vertices,
            self.faces,
            stiffness=self.cloth_spec.stiffness,
            damping=self.cloth_spec.damping
        ).to(self.device)
    
    def step(self, body_output, motion_output, t: float) -> ClothOutput:
        """
        Run cloth simulation for one frame.
        
        Args:
            body_output: Output from body stage (deformed mesh + SDF)
            motion_output: Output from motion stage (skeleton pose)
            t: Current time
            
        Returns:
            ClothOutput with simulated cloth state
        """
        dt = t - self.prev_time if self.prev_time > 0 else 1/30
        dt = max(dt, 1e-4)
        
        # Get pinned positions from body
        pinned_positions = self._compute_pinned_positions(body_output, motion_output)
        
        # Create body SDF query function
        body_sdf_fn = self._create_sdf_query(body_output)
        
        # Run simulation
        if isinstance(self.model, MockClothModel):
            new_vertices = self.model(
                body_vertices=body_output.deformed_vertices,
                body_sdf_fn=body_sdf_fn,
                pinned_mask=self.pinned_mask,
                pinned_positions=pinned_positions,
                dt=dt
            )
            velocities = (new_vertices - self.prev_vertices) / dt
        else:
            new_vertices, velocities = self._forward_hgnn(
                body_output, pinned_positions, dt
            )
        
        # Collision correction pass
        new_vertices, penetration_mask = self._collision_correction(
            new_vertices, body_sdf_fn
        )
        
        # Compute normals
        normals = self._compute_normals(new_vertices)
        
        # Update state
        self.prev_vertices = new_vertices.clone()
        self.prev_time = t
        
        return ClothOutput(
            vertices=new_vertices,
            normals=normals,
            faces=self.faces,
            velocities=velocities,
            penetration_mask=penetration_mask,
            strain_energy=self._compute_strain_energy(new_vertices)
        )
    
    def _compute_pinned_positions(self, body_output, motion_output) -> torch.Tensor:
        """Compute target positions for pinned cloth vertices."""
        pinned_positions = self.rest_vertices.clone()
        
        if not self.pinned_mask.any():
            return pinned_positions
        
        if hasattr(motion_output, 'joint_positions'):
            # Get shoulder joints
            if motion_output.joint_positions.shape[0] > 17:
                left_shoulder = motion_output.joint_positions[16]
                right_shoulder = motion_output.joint_positions[17]
                
                rest_shoulder_y = 1.4
                current_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                offset_y = current_shoulder_y - rest_shoulder_y
                
                pinned_positions[self.pinned_mask, 1] += offset_y
        
        return pinned_positions
    
    def _create_sdf_query(self, body_output) -> callable:
        """Create SDF query function from body output."""
        if hasattr(body_output, 'body_sdf') and body_output.body_sdf is not None:
            sdf_obj = body_output.body_sdf
            if callable(sdf_obj):
                return sdf_obj
            elif hasattr(sdf_obj, 'query'):
                return sdf_obj.query
        
        def simple_sdf(points: torch.Tensor) -> torch.Tensor:
            dists = torch.cdist(points, body_output.deformed_vertices)
            min_dists = dists.min(dim=1).values
            return min_dists - 0.02
        
        return simple_sdf
    
    def _forward_hgnn(self, body_output, pinned_positions: torch.Tensor,
                      dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through HGNN-NIF model."""
        velocities = (self.prev_vertices - self.rest_vertices) / max(self.prev_time, 1e-4)
        
        node_features = torch.cat([
            self.prev_vertices,
            velocities,
            self.pinned_mask.float().unsqueeze(1)
        ], dim=1)
        
        edge_index = self.graph_builder.get_edge_index(self.device)
        
        with torch.no_grad():
            sdf_output = self.model(
                x=node_features,
                edge_index=edge_index,
                query_points=self.prev_vertices
            )
        
        new_vertices = self.prev_vertices - 0.1 * sdf_output.unsqueeze(1) * self._compute_normals(self.prev_vertices)
        new_vertices[self.pinned_mask] = pinned_positions[self.pinned_mask]
        
        velocities = (new_vertices - self.prev_vertices) / dt
        velocities[self.pinned_mask] = 0
        
        return new_vertices, velocities
    
    def _collision_correction(self, vertices: torch.Tensor,
                               body_sdf_fn: callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """Correct cloth vertices that penetrate the body."""
        sdf_values = body_sdf_fn(vertices)
        penetrating = sdf_values < 0
        
        if not penetrating.any():
            return vertices, penetrating
        
        corrected = vertices.clone()
        
        # Estimate surface normal using finite differences
        eps = 0.001
        normals = torch.zeros_like(vertices[penetrating])
        
        for dim in range(3):
            perturb = torch.zeros(3, device=self.device)
            perturb[dim] = eps
            
            sdf_plus = body_sdf_fn(vertices[penetrating] + perturb)
            sdf_minus = body_sdf_fn(vertices[penetrating] - perturb)
            normals[:, dim] = (sdf_plus - sdf_minus) / (2 * eps)
        
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
        
        push_dist = -sdf_values[penetrating] + self.cloth_spec.collision_margin
        corrected[penetrating] = vertices[penetrating] + push_dist.unsqueeze(1) * normals
        
        return corrected, penetrating
    
    def _compute_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute per-vertex normals from mesh."""
        v0 = vertices[self.faces[:, 0]]
        v1 = vertices[self.faces[:, 1]]
        v2 = vertices[self.faces[:, 2]]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = face_normals / (face_normals.norm(dim=1, keepdim=True) + 1e-8)
        
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.index_add_(0, self.faces[:, i], face_normals)
        
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=1, keepdim=True) + 1e-8)
        return vertex_normals
    
    def _compute_strain_energy(self, vertices: torch.Tensor) -> float:
        """Compute total strain energy in cloth."""
        edges = torch.tensor(self.graph_builder.edges, device=self.device)
        v0 = vertices[edges[:, 0]]
        v1 = vertices[edges[:, 1]]
        
        current_lengths = (v1 - v0).norm(dim=1)
        
        rest_v0 = self.rest_vertices[edges[:, 0]]
        rest_v1 = self.rest_vertices[edges[:, 1]]
        rest_lengths = (rest_v1 - rest_v0).norm(dim=1)
        
        strain = (current_lengths - rest_lengths) / (rest_lengths + 1e-8)
        energy = 0.5 * self.cloth_spec.stiffness * (strain ** 2).sum()
        
        return energy.item()
    
    def reset(self, new_rest_mesh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Reset cloth simulation state."""
        if new_rest_mesh is not None:
            self.rest_vertices, self.faces = new_rest_mesh
            self.graph_builder = ClothGraphBuilder(
                self.faces.cpu().numpy(),
                self.rest_vertices.shape[0]
            )
            if isinstance(self.model, MockClothModel):
                self.model.reset(self.rest_vertices)
        
        self.prev_vertices = self.rest_vertices.clone()
        self.prev_time = 0.0
        
        if isinstance(self.model, MockClothModel):
            self.model.vertices = self.rest_vertices.clone()
            self.model.velocities = torch.zeros_like(self.rest_vertices)
    
    def get_rest_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get rest pose mesh."""
        return self.rest_vertices, self.faces
    
    def set_pinned_vertices(self, vertex_ids: List[int]):
        """Update pinned vertex set."""
        self.pinned_mask = torch.zeros(self.rest_vertices.shape[0],
                                        dtype=torch.bool, device=self.device)
        self.pinned_mask[vertex_ids] = True
