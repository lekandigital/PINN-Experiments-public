"""
PEGNN-Deform: Physics-Encoded Graph Neural Network Model

Core architecture with:
- SpringMessagePassing: Hard-coded Hooke's law in message function
- PEGNNDeform: Main model with GRU temporal roll-out
- Mixed-precision (AMP) support for efficient training/inference

Author: PEGNN-Deform Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple


class SpringMessagePassing(MessagePassing):
    """
    Physics-encoded message passing layer implementing Hooke's law.
    
    Each edge represents a spring with:
    - Stiffness k (edge_attr[:, 0])
    - Rest length L0 (edge_attr[:, 1])
    
    Spring force: F = k * (||x_j - x_i|| - L0) * unit_vector(x_j - x_i)
    
    The force is hard-coded (not learned) to guarantee physical consistency.
    An optional learnable modulation factor provides fine-tuning capability.
    """
    
    def __init__(self, use_learned_modulation: bool = True):
        """
        Args:
            use_learned_modulation: If True, add learnable scaling to forces
        """
        super().__init__(aggr='add')  # Sum forces from all neighbors
        
        self.use_learned_modulation = use_learned_modulation
        
        if use_learned_modulation:
            # Small MLP to modulate force magnitude (e.g., damping effects)
            # Output is sigmoid to keep modulation in [0, 1]
            self.edge_mlp = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
    
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
        return self.propagate(edge_index, x=pos, edge_attr=edge_attr)
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spring force from node j to node i (Hooke's law).
        
        The force pulls i toward j when stretched, pushes away when compressed.
        
        Args:
            x_i: Positions of destination nodes [E, 3]
            x_j: Positions of source nodes [E, 3]
            edge_attr: [E, 2] - (stiffness k, rest_length L0)
            
        Returns:
            force: Force vectors on destination nodes [E, 3]
        """
        # Vector from i to j
        diff = x_j - x_i  # [E, 3]
        
        # Current spring length
        dist = torch.norm(diff, dim=1, keepdim=True)  # [E, 1]
        
        # Extract spring parameters
        k = edge_attr[:, 0:1]   # stiffness [E, 1]
        L0 = edge_attr[:, 1:2]  # rest length [E, 1]
        
        # Hooke's law: F = k * (d - L0)
        # Positive when stretched (d > L0), negative when compressed
        force_magnitude = k * (dist - L0)  # [E, 1]
        
        # Optional learned modulation (for damping, etc.)
        if self.use_learned_modulation:
            alpha = self.edge_mlp(edge_attr)  # [E, 1]
            force_magnitude = force_magnitude * alpha
        
        # Unit direction vector (from i toward j)
        direction = diff / (dist + 1e-8)  # [E, 3]
        
        # Force vector
        force = force_magnitude * direction  # [E, 3]
        
        return force


class VelocityGRU(nn.Module):
    """
    GRU-based temporal processor for velocity updates.
    
    Maintains hidden state per node to capture temporal dynamics
    and provide implicit damping/stabilization.
    """
    
    def __init__(self, input_size: int = 3, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Project input to hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # GRU for temporal processing
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        
        # Project back to velocity space
        self.output_proj = nn.Linear(hidden_size, 3)
    
    def forward(
        self,
        velocity_update: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process velocity update through GRU.
        
        Args:
            velocity_update: Proposed velocity change [N, 3]
            hidden: Previous hidden state [N, hidden_size]
            
        Returns:
            refined_velocity: Processed velocity [N, 3]
            new_hidden: Updated hidden state [N, hidden_size]
        """
        N = velocity_update.size(0)
        device = velocity_update.device
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(N, self.hidden_size, device=device)
        
        # Project input
        x = self.input_proj(velocity_update)  # [N, hidden_size]
        
        # GRU update
        new_hidden = self.gru(x, hidden)  # [N, hidden_size]
        
        # Project to velocity
        refined_velocity = self.output_proj(new_hidden)  # [N, 3]
        
        return refined_velocity, new_hidden


class PEGNNDeform(nn.Module):
    """
    Physics-Encoded Graph Neural Network for Deformable Bodies.
    
    Architecture:
    1. Multiple message-passing layers with physics-encoded springs
    2. GRU for temporal roll-out and stability
    3. Explicit Euler integration for position updates
    
    Supports mixed-precision training via torch.amp.autocast.
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_mp_layers: int = 3,
        dt: float = 0.01,
        use_learned_modulation: bool = True,
        use_checkpointing: bool = False
    ):
        """
        Args:
            hidden_size: Hidden dimension for GRU
            num_mp_layers: Number of message passing iterations
            dt: Time step for integration
            use_learned_modulation: Whether to use learnable force modulation
            use_checkpointing: Enable gradient checkpointing for memory savings
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_mp_layers = num_mp_layers
        self.dt = dt
        self.use_checkpointing = use_checkpointing

        # Multiple message passing layers (shared or separate)
        self.spring_layers = nn.ModuleList([
            SpringMessagePassing(use_learned_modulation)
            for _ in range(num_mp_layers)
        ])

        # Velocity GRU for temporal dynamics
        self.velocity_gru = VelocityGRU(input_size=3, hidden_size=hidden_size)
    
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

        # Accumulate forces from all message passing layers
        total_force = torch.zeros_like(pos)

        current_pos = pos
        for mp_layer in self.spring_layers:
            if self.use_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
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

        # Position update: x' = x + dt * v'
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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    print("Testing PEGNN-Deform Model...")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = PEGNNDeform(hidden_size=64, num_mp_layers=3, dt=0.01).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test data
    N = 100  # nodes
    E = 300  # edges
    
    pos = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device) * 0.1
    edge_index = torch.randint(0, N, (2, E), device=device)
    edge_attr = torch.rand(E, 2, device=device)
    edge_attr[:, 0] = edge_attr[:, 0] * 10 + 1  # stiffness in [1, 11]
    edge_attr[:, 1] = edge_attr[:, 1] * 0.5 + 0.1  # rest length in [0.1, 0.6]
    
    # Test forward pass with mixed precision
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        new_pos, new_vel, hidden = model(pos, vel, edge_index, edge_attr)
    
    print(f"Input pos shape: {pos.shape}")
    print(f"Output pos shape: {new_pos.shape}")
    print(f"Output vel shape: {new_vel.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    
    assert new_pos.shape == pos.shape
    assert new_vel.shape == vel.shape
    
    # Test rollout
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        pos_traj, vel_traj = model.rollout(pos, vel, edge_index, edge_attr, num_steps=10)
    
    print(f"Rollout pos trajectory shape: {pos_traj.shape}")
    print(f"Rollout vel trajectory shape: {vel_traj.shape}")
    
    print("✓ All model tests passed!")
