"""
Collision response: resolve detected penetrations via position or force corrections.

Provides position-based dynamics (PBD) style correction, force-based correction
for physics simulations, and friction/damping models.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from .detection import CollisionResult
from .config import CollisionConfig


class CollisionResponse:
    """
    Computes corrective forces or position adjustments to resolve detected collisions.
    
    Example:
        >>> response = CollisionResponse(stiffness=1000.0, friction=0.3)
        >>> corrected = response.resolve_positions(cloth_vertices, collision_result)
        >>> # or for force-based simulation:
        >>> forces = response.resolve_forces(cloth_vertices, collision_result)
    """
    
    def __init__(
        self,
        stiffness: float = 1000.0,
        friction: float = 0.3,
        damping: float = 0.1,
        max_correction: float = 0.1,
        config: Optional[CollisionConfig] = None,
    ):
        """
        Initialize collision response.
        
        Args:
            stiffness: Force magnitude per unit penetration depth
            friction: Coulomb friction coefficient [0, 1]
            damping: Velocity damping for near-contact vertices [0, 1]
            max_correction: Maximum position correction per step
            config: Full collision config (overrides other params)
        """
        if config is not None:
            self.stiffness = config.stiffness
            self.friction = config.friction
            self.damping = config.damping
            self.max_correction = config.max_correction
        else:
            self.stiffness = stiffness
            self.friction = friction
            self.damping = damping
            self.max_correction = max_correction
    
    def resolve_positions(
        self,
        cloth_vertices: torch.Tensor,
        collision_result: CollisionResult,
    ) -> torch.Tensor:
        """
        Resolve collisions by projecting penetrating vertices to the surface.
        
        For position-based dynamics (PBD) style simulations like Projects 09, 11, 13.
        
        Args:
            cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
            collision_result: Result from CollisionDetector.detect()
            
        Returns:
            (V, 3) or (B, V, 3) corrected vertex positions
        """
        corrected = cloth_vertices.clone()
        
        # Get penetration data
        mask = collision_result.penetrating_mask
        depths = collision_result.penetration_depths  # Negative for penetrating
        normals = collision_result.surface_normals
        
        if not collision_result.has_collisions:
            return corrected
        
        # Correction: push vertex out by penetration depth along surface normal
        # depth is negative, normal points outward, so we add |depth| * normal
        correction = -depths.unsqueeze(-1) * normals  # (V, 3) or (B, V, 3)
        
        # Clamp correction magnitude
        correction_magnitude = torch.norm(correction, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_correction / (correction_magnitude + 1e-8), max=1.0)
        correction = correction * scale
        
        # Apply correction only to penetrating vertices
        corrected = torch.where(
            mask.unsqueeze(-1).expand_as(corrected),
            corrected + correction,
            corrected,
        )
        
        return corrected
    
    def resolve_forces(
        self,
        cloth_vertices: torch.Tensor,
        collision_result: CollisionResult,
    ) -> torch.Tensor:
        """
        Compute repulsive forces to resolve collisions.
        
        For force-based simulations like Projects 05, 08.
        
        Args:
            cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
            collision_result: Result from CollisionDetector.detect()
            
        Returns:
            (V, 3) or (B, V, 3) collision forces to apply
        """
        device = cloth_vertices.device
        forces = torch.zeros_like(cloth_vertices)
        
        if not collision_result.has_collisions:
            return forces
        
        # Get penetration data
        mask = collision_result.penetrating_mask
        depths = collision_result.penetration_depths  # Negative for penetrating
        normals = collision_result.surface_normals
        
        # Force = stiffness * |penetration_depth| * surface_normal
        # depth is negative, so -depth gives positive magnitude
        force_magnitude = self.stiffness * torch.abs(depths)
        collision_forces = force_magnitude.unsqueeze(-1) * normals
        
        # Apply forces only to penetrating vertices
        forces = torch.where(
            mask.unsqueeze(-1).expand_as(forces),
            collision_forces,
            forces,
        )
        
        return forces
    
    def apply_friction(
        self,
        cloth_velocities: torch.Tensor,
        collision_result: CollisionResult,
        normal_force_magnitude: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Coulomb friction to cloth velocities at contact points.
        
        Decomposes velocity into normal and tangential components,
        then applies friction to the tangential component.
        
        Args:
            cloth_velocities: (V, 3) or (B, V, 3) cloth vertex velocities
            collision_result: Result from CollisionDetector.detect()
            normal_force_magnitude: Optional (V,) normal force magnitudes
            
        Returns:
            (V, 3) or (B, V, 3) friction-adjusted velocities
        """
        adjusted = cloth_velocities.clone()
        
        # Get contact vertices (penetrating or proximal)
        contact_mask = collision_result.penetrating_mask | collision_result.proximity_mask
        if not contact_mask.any():
            return adjusted
        
        normals = collision_result.surface_normals
        
        # Decompose velocity into normal and tangential components
        # v_n = (v · n) * n
        # v_t = v - v_n
        normal_component = (cloth_velocities * normals).sum(dim=-1, keepdim=True) * normals
        tangential_component = cloth_velocities - normal_component
        
        # Compute tangential speed
        tangential_speed = torch.norm(tangential_component, dim=-1, keepdim=True) + 1e-8
        
        # Apply Coulomb friction: reduce tangential velocity
        # friction_reduction = min(1, μ * |F_n| / |v_t|)
        if normal_force_magnitude is None:
            # Use penetration depth as proxy for normal force
            normal_force_magnitude = torch.abs(collision_result.penetration_depths).unsqueeze(-1)
        
        friction_reduction = torch.clamp(
            self.friction * normal_force_magnitude / tangential_speed,
            max=1.0,
        )
        
        # Reduce tangential velocity
        adjusted_tangential = tangential_component * (1 - friction_reduction)
        
        # Reconstruct velocity (remove normal component for penetrating vertices)
        # Keep normal component only if moving away from surface
        normal_speed = (cloth_velocities * normals).sum(dim=-1, keepdim=True)
        keep_normal = (normal_speed > 0).float()  # Moving away from surface
        adjusted_normal = normal_component * keep_normal
        
        new_velocity = adjusted_normal + adjusted_tangential
        
        # Apply only to contact vertices
        adjusted = torch.where(
            contact_mask.unsqueeze(-1).expand_as(adjusted),
            new_velocity,
            adjusted,
        )
        
        return adjusted
    
    def apply_damping(
        self,
        cloth_velocities: torch.Tensor,
        collision_result: CollisionResult,
    ) -> torch.Tensor:
        """
        Apply velocity damping near contact surfaces to prevent jittering.
        
        Damping increases as vertices get closer to the surface.
        
        Args:
            cloth_velocities: (V, 3) or (B, V, 3) cloth vertex velocities
            collision_result: Result from CollisionDetector.detect()
            
        Returns:
            (V, 3) or (B, V, 3) damped velocities
        """
        damped = cloth_velocities.clone()
        
        contact_mask = collision_result.penetrating_mask | collision_result.proximity_mask
        if not contact_mask.any():
            return damped
        
        # Compute proximity factor (1 at surface, 0 far away)
        # Use abs(sdf) since penetrating vertices have negative sdf
        distances = torch.abs(collision_result.penetration_depths)
        
        # Normalize by some reference distance (e.g., proximity threshold)
        reference_distance = 0.01  # 1cm in normalized coordinates
        proximity_factor = torch.clamp(1 - distances / reference_distance, min=0, max=1)
        
        # Apply damping: velocity *= (1 - damping * proximity_factor)
        damping_factor = 1 - self.damping * proximity_factor
        
        damped = torch.where(
            contact_mask.unsqueeze(-1).expand_as(damped),
            cloth_velocities * damping_factor.unsqueeze(-1),
            damped,
        )
        
        return damped
    
    def full_response(
        self,
        cloth_vertices: torch.Tensor,
        cloth_velocities: Optional[torch.Tensor],
        collision_result: CollisionResult,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply full collision response: position correction + friction + damping.
        
        Args:
            cloth_vertices: (V, 3) or (B, V, 3) cloth vertex positions
            cloth_velocities: Optional (V, 3) or (B, V, 3) cloth velocities
            collision_result: Result from CollisionDetector.detect()
            
        Returns:
            Tuple of (corrected_vertices, adjusted_velocities)
        """
        corrected_vertices = self.resolve_positions(cloth_vertices, collision_result)
        
        if cloth_velocities is not None:
            adjusted_velocities = self.apply_friction(cloth_velocities, collision_result)
            adjusted_velocities = self.apply_damping(adjusted_velocities, collision_result)
        else:
            adjusted_velocities = None
        
        return corrected_vertices, adjusted_velocities
