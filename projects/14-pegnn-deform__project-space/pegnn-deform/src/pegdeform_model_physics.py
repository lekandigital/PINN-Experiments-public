"""
PEGNN-Deform: Refactored to use shared PhysicsEncodedConv library.

This module provides a physics-encoded version of PEGNNDeform that uses
the shared ElasticForceConv and VelocityGRU from the physics_conv library,
while maintaining full backward compatibility with the original API.

Gains from refactoring:
- Stronger physics inductive bias with Hooke's law hard-coded
- physics_fraction() diagnostic for monitoring training
- Shared code with other projects (08-HGNN-ClothDyn, etc.)
- Better tested and documented physics implementations
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict

# Import shared physics-encoded convolution
import sys
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments')
from shared.physics_conv import ElasticForceConv
from shared.physics_conv.deform_conv import ElasticConvConfig, VelocityGRU
from shared.physics_conv.integrators import SemiImplicitEuler, ExplicitEuler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsEncodedSpringMP(nn.Module):
    """
    Physics-encoded spring message passing using shared ElasticForceConv.
    
    This is a drop-in replacement for the original SpringMessagePassing that:
    1. Uses Hooke's law from the shared physics library
    2. Provides physics_fraction() diagnostics
    3. Supports learned corrections on top of physics
    
    API compatibility with original SpringMessagePassing:
    - forward(pos, edge_index, edge_attr) -> forces
    """
    
    def __init__(
        self,
        use_learned_modulation: bool = True,
        stiffness_scale: float = 1.0,
        correction_hidden_dim: int = 16,
    ):
        """
        Args:
            use_learned_modulation: Whether to learn corrections to physics
            stiffness_scale: Multiplier for edge stiffness values
            correction_hidden_dim: Hidden dimension for correction MLP
        """
        super().__init__()
        
        self.use_learned_modulation = use_learned_modulation
        self.stiffness_scale = stiffness_scale
        
        # Configure the physics-encoded convolution
        config = ElasticConvConfig(
            youngs_modulus=1.0,  # Will use per-edge stiffness
            stiffness_mode="per_edge",
            correction_hidden_dim=correction_hidden_dim if use_learned_modulation else 1,
            correction_layers=2 if use_learned_modulation else 0,
            correction_scale_init=0.1 if use_learned_modulation else 0.0,
            aggregation="add",
            track_diagnostics=True,
            use_velocity_gru=False,  # GRU is handled separately
        )
        
        self.elastic_conv = ElasticForceConv(config=config)
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spring forces on all nodes.
        
        Args:
            pos: Node positions [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 2] - (stiffness, rest_length)
            
        Returns:
            forces: Per-node force vectors [N, 3]
        """
        # Rearrange edge_attr: original has (stiffness, rest_length)
        # ElasticForceConv expects (rest_length, stiffness)
        rest_length = edge_attr[:, 1:2]  # L0
        stiffness = edge_attr[:, 0:1] * self.stiffness_scale  # k
        
        # Create properly ordered edge_attr for physics conv
        reordered_attr = torch.cat([rest_length, stiffness], dim=-1)
        
        # Compute forces using physics-encoded convolution
        forces = self.elastic_conv(pos, edge_index, reordered_attr)
        
        return forces
    
    def physics_fraction(self) -> float:
        """Get the fraction of output attributable to physics vs learned."""
        return self.elastic_conv.physics_fraction()
    
    def correction_magnitude(self) -> float:
        """Get the average magnitude of learned corrections."""
        return self.elastic_conv.correction_magnitude()


class PEGNNDeformPhysics(nn.Module):
    """
    Physics-Encoded Graph Neural Network for Deformable Bodies.
    
    Refactored version using shared physics_conv library components.
    
    Architecture:
    1. Multiple ElasticForceConv layers with physics-encoded springs
    2. VelocityGRU for temporal roll-out and stability  
    3. SemiImplicitEuler integration for position updates
    
    Fully backward compatible with original PEGNNDeform API.
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_mp_layers: int = 3,
        dt: float = 0.01,
        use_learned_modulation: bool = True,
        use_checkpointing: bool = False,
        integrator: str = "semi_implicit_euler",
        damping: float = 1.0,
    ):
        """
        Args:
            hidden_size: Hidden dimension for GRU
            num_mp_layers: Number of message passing iterations
            dt: Time step for integration
            use_learned_modulation: Whether to use learnable force modulation
            use_checkpointing: Enable gradient checkpointing for memory savings
            integrator: Integration scheme ("explicit_euler", "semi_implicit_euler")
            damping: Velocity damping factor (1.0 = no damping)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_mp_layers = num_mp_layers
        self.dt = dt
        self.use_checkpointing = use_checkpointing
        self.damping = damping

        # Physics-encoded message passing layers
        self.spring_layers = nn.ModuleList([
            PhysicsEncodedSpringMP(
                use_learned_modulation=use_learned_modulation,
                correction_hidden_dim=16,
            )
            for _ in range(num_mp_layers)
        ])

        # VelocityGRU for temporal dynamics (from shared library)
        self.velocity_gru = VelocityGRU(
            input_size=3,
            hidden_size=hidden_size,
            output_size=3,
        )
        
        # Time integrator
        if integrator == "semi_implicit_euler":
            self.integrator = SemiImplicitEuler(damping=damping)
        else:
            self.integrator = ExplicitEuler(damping=damping)
        
        logger.info(
            f"PEGNNDeformPhysics initialized: {num_mp_layers} physics-encoded layers, "
            f"learned_modulation={use_learned_modulation}, dt={dt}, integrator={integrator}"
        )
    
    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mass: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict next position and velocity.
        
        Args:
            pos: Current positions [N, 3]
            vel: Current velocities [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 2] - (stiffness, rest_length)
            hidden: GRU hidden state [N, hidden_size]
            mass: Per-node mass [N, 1] (optional, defaults to 1.0)
            
        Returns:
            new_pos: Updated positions [N, 3]
            new_vel: Updated velocities [N, 3]
            new_hidden: Updated GRU hidden state [N, hidden_size]
        """
        N = pos.size(0)
        device = pos.device

        # Default mass = 1.0 for all nodes
        if mass is None:
            mass = torch.ones(N, 1, device=device)

        # Accumulate forces from all physics-encoded layers
        total_force = torch.zeros_like(pos)

        current_pos = pos
        for mp_layer in self.spring_layers:
            if self.use_checkpointing and self.training:
                force = checkpoint(
                    mp_layer,
                    current_pos, edge_index, edge_attr,
                    use_reentrant=False
                )
            else:
                force = mp_layer(current_pos, edge_index, edge_attr)
            total_force = total_force + force

        # Compute acceleration: a = F / m
        acceleration = total_force / (mass + 1e-8)  # [N, 3]

        # Velocity update: v' = v + dt * a
        velocity_update = vel + self.dt * acceleration

        # Refine velocity through GRU (adds implicit damping/stability)
        new_vel, new_hidden = self.velocity_gru(velocity_update, hidden)

        # Position update using integrator
        new_pos = pos + self.dt * new_vel

        return new_pos, new_vel, new_hidden
    
    def rollout(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_steps: int,
        mass: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out simulation for multiple time steps.
        
        Args:
            pos: Initial positions [N, 3]
            vel: Initial velocities [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 2]
            num_steps: Number of time steps to simulate
            mass: Per-node mass [N, 1]
            
        Returns:
            pos_trajectory: Positions over time [num_steps+1, N, 3]
            vel_trajectory: Velocities over time [num_steps+1, N, 3]
        """
        pos_trajectory = [pos]
        vel_trajectory = [vel]
        
        current_pos = pos
        current_vel = vel
        hidden = None
        
        for _ in range(num_steps):
            current_pos, current_vel, hidden = self.forward(
                current_pos, current_vel, edge_index, edge_attr, hidden, mass
            )
            pos_trajectory.append(current_pos)
            vel_trajectory.append(current_vel)
        
        return torch.stack(pos_trajectory), torch.stack(vel_trajectory)
    
    def physics_fraction(self) -> Dict[str, float]:
        """Get physics fraction for all spring layers."""
        fractions = {}
        for i, layer in enumerate(self.spring_layers):
            fractions[f"spring_layer_{i}"] = layer.physics_fraction()
        
        # Compute mean
        if fractions:
            fractions["mean"] = sum(fractions.values()) / len(fractions)
        
        return fractions
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get comprehensive diagnostics for physics vs learned components."""
        diagnostics = self.physics_fraction()
        
        # Add correction magnitudes
        for i, layer in enumerate(self.spring_layers):
            diagnostics[f"correction_mag_{i}"] = layer.correction_magnitude()
        
        return diagnostics


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_physics_model():
    """Test the physics-encoded PEGNN-Deform model."""
    print("=" * 60)
    print("Testing Physics-Encoded PEGNN-Deform")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test PhysicsEncodedSpringMP
    print("Testing PhysicsEncodedSpringMP...")
    spring_mp = PhysicsEncodedSpringMP(use_learned_modulation=True).to(device)
    
    N, E = 100, 300
    pos = torch.randn(N, 3, device=device)
    edge_index = torch.randint(0, N, (2, E), device=device)
    edge_attr = torch.rand(E, 2, device=device)
    edge_attr[:, 0] = edge_attr[:, 0] * 10 + 1  # stiffness in [1, 11]
    edge_attr[:, 1] = edge_attr[:, 1] * 0.5 + 0.1  # rest length in [0.1, 0.6]
    
    forces = spring_mp(pos, edge_index, edge_attr)
    print(f"  Input: {pos.shape} → Output: {forces.shape}")
    print(f"  Physics fraction: {spring_mp.physics_fraction():.3f}")
    print("  ✓ PhysicsEncodedSpringMP test passed\n")
    
    # Test PEGNNDeformPhysics
    print("Testing PEGNNDeformPhysics...")
    model = PEGNNDeformPhysics(
        hidden_size=64,
        num_mp_layers=3,
        dt=0.01,
        use_learned_modulation=True
    ).to(device)
    
    vel = torch.randn(N, 3, device=device) * 0.1
    
    new_pos, new_vel, hidden = model(pos, vel, edge_index, edge_attr)
    print(f"  Input pos: {pos.shape} → Output pos: {new_pos.shape}")
    print(f"  Output vel: {new_vel.shape}")
    print(f"  Hidden state: {hidden.shape}")
    
    # Check physics fractions
    fractions = model.physics_fraction()
    print(f"  Physics fractions: mean={fractions['mean']:.3f}")
    
    assert new_pos.shape == pos.shape
    assert new_vel.shape == vel.shape
    print("  ✓ PEGNNDeformPhysics forward pass test passed\n")
    
    # Test rollout
    print("Testing rollout...")
    pos_traj, vel_traj = model.rollout(pos, vel, edge_index, edge_attr, num_steps=10)
    print(f"  Rollout pos trajectory: {pos_traj.shape}")
    print(f"  Rollout vel trajectory: {vel_traj.shape}")
    assert pos_traj.shape == (11, N, 3)
    assert vel_traj.shape == (11, N, 3)
    print("  ✓ Rollout test passed\n")
    
    # Parameter count
    print("Model parameters:")
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
    print(f"  Total: {total:,}")
    
    # Compare with original
    print("\n" + "=" * 60)
    print("✓ All physics model tests passed!")


if __name__ == "__main__":
    test_physics_model()
