"""
Interface between body deformation (PEGNN) and collision system.

Wraps body meshes with auto-updating SDF fields and velocity estimation.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F

from .sdf_field import SDFField
from .config import BodyConfig, SDFConfig


class DeformableBody:
    """
    Wraps a body mesh (potentially driven by PEGNN-Deform) and maintains
    an up-to-date SDF field as the body deforms.
    
    Example:
        >>> body = DeformableBody(rest_vertices, faces, sdf_resolution=128)
        >>> # Each frame:
        >>> body.update_from_deformation(pegnn_output)
        >>> sdf = body.get_sdf()
        >>> collisions = detector.detect(cloth_vertices, sdf)
    """
    
    def __init__(
        self,
        rest_vertices: torch.Tensor,
        faces: torch.Tensor,
        sdf_resolution: int = 128,
        device: Optional[torch.device] = None,
        config: Optional[BodyConfig] = None,
    ):
        """
        Initialize deformable body.
        
        Args:
            rest_vertices: (V, 3) rest pose vertex positions
            faces: (F, 3) triangle face indices (topology, doesn't change)
            sdf_resolution: Resolution for SDF computation
            device: Target device
            config: Full body configuration
        """
        self.device = device or rest_vertices.device
        
        self.rest_vertices = rest_vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.num_vertices = rest_vertices.shape[0]
        
        if config is not None:
            self.sdf_resolution = config.sdf_resolution
            self.velocity_smoothing = config.velocity_smoothing
            self.cache_frames = config.cache_previous_frames
        else:
            self.sdf_resolution = sdf_resolution
            self.velocity_smoothing = 0.5
            self.cache_frames = 2
        
        # Current state
        self.current_vertices = self.rest_vertices.clone()
        self._sdf: Optional[SDFField] = None
        
        # Frame history for velocity estimation
        self._vertex_history: List[torch.Tensor] = []
        self._velocity: Optional[torch.Tensor] = None
        
        # Skinning weights for LBS fallback
        self._skinning_weights: Optional[torch.Tensor] = None
        self._bone_transforms: Optional[torch.Tensor] = None
        
        # Build initial SDF
        self._update_sdf()
    
    def update_from_deformation(
        self,
        deformed_vertices: torch.Tensor,
        dt: float = 1.0 / 30.0,
    ) -> None:
        """
        Update body from deformed vertex positions (e.g., PEGNN output).
        
        Args:
            deformed_vertices: (V, 3) new vertex positions
            dt: Time step for velocity estimation
        """
        deformed_vertices = deformed_vertices.to(self.device)
        
        # Update vertex history for velocity estimation
        self._vertex_history.append(self.current_vertices.clone())
        if len(self._vertex_history) > self.cache_frames:
            self._vertex_history.pop(0)
        
        # Estimate velocity
        if len(self._vertex_history) >= 1:
            prev_vertices = self._vertex_history[-1]
            raw_velocity = (deformed_vertices - prev_vertices) / dt
            
            if self._velocity is not None:
                # Temporal smoothing
                self._velocity = (
                    self.velocity_smoothing * self._velocity +
                    (1 - self.velocity_smoothing) * raw_velocity
                )
            else:
                self._velocity = raw_velocity
        
        # Update current state
        self.current_vertices = deformed_vertices
        
        # Recompute SDF
        self._update_sdf()
    
    def update_from_skeleton(
        self,
        joint_transforms: torch.Tensor,
        skinning_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update body from skeleton joint transforms using Linear Blend Skinning.
        
        Alternative to PEGNN for basic body animation.
        
        Args:
            joint_transforms: (J, 4, 4) transformation matrices for each joint
            skinning_weights: (V, J) blend weights per vertex per joint
                             If not provided, uses cached weights
        """
        if skinning_weights is not None:
            self._skinning_weights = skinning_weights.to(self.device)
        
        if self._skinning_weights is None:
            raise ValueError("Skinning weights must be provided at least once")
        
        joint_transforms = joint_transforms.to(self.device)
        
        # Linear Blend Skinning
        # deformed_v = sum_j(w_j * T_j * rest_v)
        num_joints = joint_transforms.shape[0]
        
        # Homogeneous coordinates for rest vertices
        rest_homo = F.pad(self.rest_vertices, (0, 1), value=1.0)  # (V, 4)
        
        # Apply each joint transform weighted by skinning weights
        deformed = torch.zeros_like(self.rest_vertices)
        for j in range(num_joints):
            # Transform rest vertices by joint j
            transformed = torch.matmul(rest_homo, joint_transforms[j].T)[:, :3]  # (V, 3)
            # Weight by skinning weight for joint j
            deformed = deformed + self._skinning_weights[:, j:j+1] * transformed
        
        # Update via standard deformation path
        self.update_from_deformation(deformed)
    
    def _update_sdf(self) -> None:
        """Recompute SDF from current vertex positions."""
        self._sdf = SDFField.from_mesh(
            self.current_vertices,
            self.faces,
            resolution=self.sdf_resolution,
            device=self.device,
        )
    
    def get_sdf(self) -> SDFField:
        """Get current SDF field for collision queries."""
        if self._sdf is None:
            self._update_sdf()
        return self._sdf
    
    def get_velocity_field(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Get body surface velocity at query points.
        
        Interpolates vertex velocities to arbitrary points using
        closest-point correspondence.
        
        Args:
            query_points: (N, 3) query positions
            
        Returns:
            (N, 3) velocity vectors at query points
        """
        if self._velocity is None:
            return torch.zeros_like(query_points)
        
        query_points = query_points.to(self.device)
        
        # Find closest vertex to each query point
        # (Simple approach - for better accuracy, use barycentric interpolation)
        distances = torch.cdist(query_points, self.current_vertices)  # (N, V)
        closest_vertex_idx = distances.argmin(dim=1)  # (N,)
        
        # Get velocity at closest vertex
        velocities = self._velocity[closest_vertex_idx]  # (N, 3)
        
        return velocities
    
    def get_surface_point(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Project query points onto the body surface.
        
        Args:
            query_points: (N, 3) query positions
            
        Returns:
            (N, 3) closest points on body surface
        """
        sdf = self.get_sdf()
        sdf_values = sdf.query(query_points)
        gradients = sdf.gradient(query_points)
        normals = F.normalize(gradients, dim=-1)
        
        # Project to surface: p_surface = p - sdf * normal
        surface_points = query_points - sdf_values.unsqueeze(-1) * normals
        
        return surface_points
    
    def reset(self) -> None:
        """Reset to rest pose."""
        self.current_vertices = self.rest_vertices.clone()
        self._vertex_history.clear()
        self._velocity = None
        self._update_sdf()
    
    @property
    def vertices(self) -> torch.Tensor:
        """Current vertex positions."""
        return self.current_vertices
    
    @property
    def velocity(self) -> Optional[torch.Tensor]:
        """Current vertex velocities (if available)."""
        return self._velocity


class PEGNNBodyAdapter:
    """
    Adapter to connect PEGNN-Deform model output to DeformableBody.
    
    Example:
        >>> adapter = PEGNNBodyAdapter(pegnn_model, rest_vertices, faces)
        >>> # Each frame:
        >>> body = adapter.step(input_features)
        >>> sdf = body.get_sdf()
    """
    
    def __init__(
        self,
        pegnn_model: torch.nn.Module,
        rest_vertices: torch.Tensor,
        faces: torch.Tensor,
        sdf_resolution: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize PEGNN adapter.
        
        Args:
            pegnn_model: Trained PEGNN-Deform model
            rest_vertices: (V, 3) body rest pose
            faces: (F, 3) body mesh faces
            sdf_resolution: SDF grid resolution
            device: Target device
        """
        self.pegnn_model = pegnn_model
        self.body = DeformableBody(
            rest_vertices, faces,
            sdf_resolution=sdf_resolution,
            device=device,
        )
        self.device = self.body.device
        
        # GRU hidden state for temporal continuity
        self._hidden_state = None
    
    def step(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dt: float = 1.0 / 30.0,
    ) -> DeformableBody:
        """
        Run one PEGNN step and update body SDF.
        
        Args:
            node_features: (V, F_node) node features for PEGNN
            edge_index: (2, E) edge connectivity
            edge_attr: (E, F_edge) edge features
            dt: Time step
            
        Returns:
            Updated DeformableBody with new SDF
        """
        # Run PEGNN forward pass
        with torch.no_grad():
            deformed_vertices, self._hidden_state = self.pegnn_model(
                node_features.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device),
                hidden=self._hidden_state,
            )
        
        # Update body
        self.body.update_from_deformation(deformed_vertices, dt=dt)
        
        return self.body
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._hidden_state = None
        self.body.reset()
