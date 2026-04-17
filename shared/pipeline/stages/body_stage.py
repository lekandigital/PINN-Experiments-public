"""
Body Stage - Project 14 (PEGNN-Deform) Wrapper with LBS

Combines Linear Blend Skinning for coarse deformation with
PEGNN for physics-based soft tissue dynamics.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ..config import SkinningSpec, PipelineConfig
from ..transforms import linear_blend_skinning_torch, forward_kinematics

# Try importing Project 14 model
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3] / "projects" / "14-pegnn-deform__project-space"))
    from models.pegnn import PEGNNDeform
    HAS_PROJECT14 = True
except ImportError:
    HAS_PROJECT14 = False
    PEGNNDeform = None

# Try importing collision module
try:
    from shared.collision import SDFCollider
    HAS_COLLISION = True
except ImportError:
    HAS_COLLISION = False
    SDFCollider = None


@dataclass
class BodyOutput:
    """Output from body deformation stage."""
    deformed_vertices: torch.Tensor  # (V, 3) deformed mesh vertices
    normals: torch.Tensor            # (V, 3) vertex normals
    velocities: torch.Tensor         # (V, 3) vertex velocities
    faces: torch.Tensor              # (F, 3) face indices (unchanged)
    body_sdf: Optional[Any] = None   # SDF for collision queries


class SimpleSDF:
    """Simple SDF approximation from mesh vertices."""
    
    def __init__(self, vertices: torch.Tensor, margin: float = 0.02):
        self.vertices = vertices
        self.margin = margin
    
    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Query SDF at points."""
        dists = torch.cdist(points, self.vertices)
        min_dists = dists.min(dim=1).values
        return min_dists - self.margin
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return self.query(points)


class BodyStage:
    """
    Body deformation stage combining LBS and PEGNN.
    
    Pipeline:
    1. Apply Linear Blend Skinning from skeleton pose
    2. Apply PEGNN soft tissue deformation corrections
    3. Compute vertex normals and SDF for cloth collision
    """
    
    def __init__(self, config: PipelineConfig, device: torch.device = None):
        """
        Args:
            config: Pipeline configuration
            device: Compute device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.skinning = config.skinning
        
        self.model: Optional[nn.Module] = None
        
        # Mesh data
        self.rest_vertices: Optional[torch.Tensor] = None
        self.faces: Optional[torch.Tensor] = None
        self.skinning_weights: Optional[torch.Tensor] = None
        
        # State for velocity computation
        self.prev_vertices: Optional[torch.Tensor] = None
        self.prev_time: float = 0.0
        
        # SDF collider
        self.sdf_collider: Optional[Any] = None
    
    def load(self, checkpoint_path: Optional[str] = None,
             body_mesh_path: Optional[str] = None,
             skinning_weights_path: Optional[str] = None):
        """
        Load body model and mesh data.
        
        Args:
            checkpoint_path: Path to PEGNN checkpoint
            body_mesh_path: Path to rest pose body mesh (OBJ)
            skinning_weights_path: Path to skinning weights (NPZ)
        """
        # Load body mesh
        if body_mesh_path and Path(body_mesh_path).exists():
            self._load_body_mesh(body_mesh_path)
        else:
            self._create_default_body()
        
        # Load skinning weights
        if skinning_weights_path and Path(skinning_weights_path).exists():
            self._load_skinning_weights(skinning_weights_path)
        else:
            self._generate_default_weights()
        
        # Load PEGNN model
        if checkpoint_path and Path(checkpoint_path).exists() and HAS_PROJECT14:
            self._load_pegnn(checkpoint_path)
        else:
            print(f"[BodyStage] No PEGNN model loaded - using LBS only")
            self.model = None
        
        # Initialize SDF collider
        if HAS_COLLISION:
            self.sdf_collider = SDFCollider(self.rest_vertices, self.faces)
        else:
            self.sdf_collider = None
        
        self.prev_vertices = self.rest_vertices.clone()
    
    def _load_body_mesh(self, path: str):
        """Load body mesh from OBJ file."""
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
                    for p in parts[1:]:
                        if len(face_verts) >= 3:
                            break
                        face_verts.append(int(p.split('/')[0]) - 1)
                    if len(face_verts) == 3:
                        faces.append(face_verts)
        
        self.rest_vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)
    
    def _create_default_body(self):
        """Create a simple default body mesh (capsule approximation)."""
        # Create a simple humanoid mesh from capsules
        # This is just for testing; real usage should load SMPL or similar
        
        vertices = []
        faces = []
        
        # Simple torso (box)
        torso_verts = [
            [-0.15, 0.9, -0.1], [0.15, 0.9, -0.1], [0.15, 0.9, 0.1], [-0.15, 0.9, 0.1],
            [-0.15, 1.4, -0.1], [0.15, 1.4, -0.1], [0.15, 1.4, 0.1], [-0.15, 1.4, 0.1],
        ]
        vertices.extend(torso_verts)
        
        # Box faces
        box_faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 1], [1, 4, 5],  # front
            [2, 6, 3], [3, 6, 7],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 2], [2, 5, 6],  # right
        ]
        faces.extend(box_faces)
        
        self.rest_vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)
    
    def _load_skinning_weights(self, path: str):
        """Load skinning weights from NPZ file."""
        data = np.load(path)
        
        if 'weights' in data:
            weights = data['weights']
        elif 'skinning_weights' in data:
            weights = data['skinning_weights']
        else:
            # Try first array
            weights = data[list(data.keys())[0]]
        
        self.skinning_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def _generate_default_weights(self):
        """Generate default skinning weights based on vertex positions."""
        num_vertices = self.rest_vertices.shape[0]
        num_joints = self.config.skeleton.num_joints
        
        # Simple distance-based weighting
        # In practice, would use geodesic distances or learned weights
        
        joint_positions = torch.tensor(
            self.config.skeleton.rest_positions,
            dtype=torch.float32, device=self.device
        )
        
        # Compute distances from each vertex to each joint
        # Shape: (V, J)
        dists = torch.cdist(self.rest_vertices, joint_positions)
        
        # Convert to weights using inverse distance (with softmax)
        inv_dists = 1.0 / (dists + 0.1)
        weights = torch.softmax(inv_dists * 5.0, dim=1)  # Temperature scaling
        
        # Sparsify: keep only top-k weights per vertex
        k = min(4, num_joints)
        topk_weights, topk_indices = torch.topk(weights, k, dim=1)
        
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(1, topk_indices, topk_weights)
        sparse_weights = sparse_weights / sparse_weights.sum(dim=1, keepdim=True)
        
        self.skinning_weights = sparse_weights
    
    def _load_pegnn(self, checkpoint_path: str):
        """Load PEGNN model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_config = checkpoint.get('config', {})
        
        self.model = PEGNNDeform(
            input_dim=model_config.get('input_dim', 3),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 4),
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def step(self, motion_output, t: float) -> BodyOutput:
        """
        Deform body mesh based on skeleton pose.
        
        Args:
            motion_output: Output from motion stage
            t: Current time
            
        Returns:
            BodyOutput with deformed mesh
        """
        # Get joint transforms
        joint_positions = motion_output.joint_positions
        joint_rotations = motion_output.joint_rotations
        
        # Compute joint transforms for LBS
        joint_transforms = self._compute_joint_transforms(
            joint_positions, joint_rotations
        )
        
        # Apply LBS
        deformed = linear_blend_skinning_torch(
            self.rest_vertices,
            self.skinning_weights,
            joint_transforms
        )
        
        # Apply PEGNN soft tissue corrections if available
        if self.model is not None:
            deformed = self._apply_soft_tissue(deformed, motion_output)
        
        # Compute velocity
        dt = t - self.prev_time if self.prev_time > 0 else 1/30
        dt = max(dt, 1e-6)
        velocities = (deformed - self.prev_vertices) / dt
        
        # Compute normals
        normals = self._compute_normals(deformed)
        
        # Update SDF
        body_sdf = self._create_sdf(deformed)
        
        # Update state
        self.prev_vertices = deformed.clone()
        self.prev_time = t
        
        return BodyOutput(
            deformed_vertices=deformed,
            normals=normals,
            velocities=velocities,
            faces=self.faces,
            body_sdf=body_sdf
        )
    
    def _compute_joint_transforms(self, positions: torch.Tensor,
                                   rotations: torch.Tensor) -> torch.Tensor:
        """
        Compute 4x4 transform matrices for each joint.
        
        Args:
            positions: (J, 3) global joint positions
            rotations: (J, 4) local quaternions (wxyz)
            
        Returns:
            transforms: (J, 4, 4) transformation matrices
        """
        num_joints = positions.shape[0]
        transforms = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_joints, 1, 1)
        
        # Convert quaternions to rotation matrices
        for j in range(num_joints):
            q = rotations[j]
            w, x, y, z = q[0], q[1], q[2], q[3]
            
            # Rotation matrix from quaternion
            R = torch.tensor([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ], device=self.device, dtype=torch.float32)
            
            transforms[j, :3, :3] = R
            transforms[j, :3, 3] = positions[j]
        
        return transforms
    
    def _apply_soft_tissue(self, lbs_vertices: torch.Tensor,
                            motion_output) -> torch.Tensor:
        """
        Apply PEGNN soft tissue deformation on top of LBS.
        
        Args:
            lbs_vertices: (V, 3) LBS-deformed vertices
            motion_output: Motion data for dynamics context
            
        Returns:
            deformed: (V, 3) vertices with soft tissue effects
        """
        # Build input features for PEGNN
        # Include position, velocity hints, etc.
        
        if motion_output.velocity is not None:
            # Propagate joint velocities to vertices
            joint_vel = motion_output.velocity  # (J, 3)
            # Simple: weight by skinning weights
            vertex_vel_hint = torch.matmul(self.skinning_weights, joint_vel)
        else:
            vertex_vel_hint = torch.zeros_like(lbs_vertices)
        
        # Node features
        node_features = torch.cat([
            lbs_vertices,
            vertex_vel_hint,
        ], dim=1)  # (V, 6)
        
        # Build edge index from faces
        edge_index = self._faces_to_edges()
        
        with torch.no_grad():
            # PEGNN predicts displacement corrections
            displacement = self.model(
                x=node_features,
                edge_index=edge_index,
                batch=None
            )
        
        # Apply displacement (scaled for stability)
        deformed = lbs_vertices + 0.1 * displacement[:, :3]
        
        return deformed
    
    def _faces_to_edges(self) -> torch.Tensor:
        """Convert faces to edge index for GNN."""
        edges_set = set()
        for f in self.faces.cpu().numpy():
            for i in range(3):
                v0, v1 = int(f[i]), int(f[(i + 1) % 3])
                edges_set.add((min(v0, v1), max(v0, v1)))
        
        edges = list(edges_set)
        # Bidirectional
        edge_index = []
        for e in edges:
            edge_index.append([e[0], e[1]])
            edge_index.append([e[1], e[0]])
        
        return torch.tensor(edge_index, dtype=torch.long, device=self.device).T
    
    def _compute_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute per-vertex normals."""
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
    
    def _create_sdf(self, vertices: torch.Tensor):
        """Create SDF representation of body for collision."""
        if self.sdf_collider is not None:
            self.sdf_collider.update(vertices)
            return self.sdf_collider
        else:
            return SimpleSDF(vertices)
    
    def reset(self):
        """Reset body stage state."""
        self.prev_vertices = self.rest_vertices.clone()
        self.prev_time = 0.0
    
    def get_rest_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get rest pose mesh."""
        return self.rest_vertices, self.faces
