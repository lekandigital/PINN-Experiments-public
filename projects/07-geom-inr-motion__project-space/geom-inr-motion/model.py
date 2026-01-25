#!/usr/bin/env python3
"""
model.py - Geom-INR Neural Network Architecture

This module implements the core Implicit Neural Representation (INR) model
for human motion prediction with differential geometry regularization.

Architecture:
    Input: (actor_id, joint_index, time)
           ↓
    Actor Embedding (dim=32) + Joint Index [0,1] + Time
           ↓
    SIREN MLP (4 hidden layers, 256 dim, sine activations)
           ↓
    Optional Graph Convolution (skeleton adjacency smoothing)
           ↓
    Output: (x, y, z) joint position

Key Features:
- SIREN-style initialization for sine activations
- Actor conditioning via learned embeddings
- Optional graph-based skeletal consistency
- Supports batch inference and trajectory generation

References:
- SIREN: Sitzmann et al., "Implicit Neural Representations with Periodic
  Activation Functions", NeurIPS 2020
- NeRMo: Wei et al., "Learning Implicit Neural Representations for 3D
  Human Motion Prediction", ECCV 2024

Usage:
    from model import GeomINR
    
    model = GeomINR(num_joints=24, num_actors=100)
    
    # Forward pass
    actor_ids = torch.tensor([0, 0, 0])  # Actor 0
    joint_idxs = torch.tensor([0, 1, 2])  # Joints 0, 1, 2
    times = torch.tensor([0.5, 0.5, 0.5])  # Time 0.5
    
    positions = model(actor_ids, joint_idxs, times)  # (3, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np


# ============================================================================
# Sine Activation Layer
# ============================================================================

class Sine(nn.Module):
    """
    Sine activation function for SIREN networks.
    
    y = sin(omega * x)
    
    The omega parameter controls the frequency. Higher omega allows
    representing higher-frequency signals but may cause training instability.
    """
    
    def __init__(self, omega: float = 30.0):
        """
        Initialize sine activation.
        
        Args:
            omega: Frequency multiplier (default 30.0 from SIREN paper)
        """
        super().__init__()
        self.omega = omega
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)
    
    def __repr__(self):
        return f"Sine(omega={self.omega})"


# ============================================================================
# SIREN Layer with Proper Initialization
# ============================================================================

class SIRENLayer(nn.Module):
    """
    A single SIREN layer: Linear + Sine activation with proper initialization.
    
    Initialization follows Sitzmann et al. 2020:
    - First layer: weights ~ Uniform(-1/d_in, 1/d_in)
    - Hidden layers: weights ~ Uniform(-sqrt(6/d_in)/omega, sqrt(6/d_in)/omega)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False,
        bias: bool = True
    ):
        """
        Initialize SIREN layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega: Frequency for sine activation
            is_first: Whether this is the first layer (different init)
            bias: Whether to include bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = Sine(omega)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform(-1/d, 1/d)
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform(-sqrt(6/d)/omega, sqrt(6/d)/omega)
                bound = math.sqrt(6.0 / self.in_features) / self.omega
            
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


# ============================================================================
# Skeleton Graph Module
# ============================================================================

class SkeletonGraphModule(nn.Module):
    """
    Graph-based module for enforcing skeletal structure.
    
    Applies graph convolution using skeleton adjacency matrix to:
    1. Smooth predictions among connected joints
    2. Encourage consistent bone lengths
    
    Uses simple aggregation: x' = x + lambda * A @ x - lambda * x
    where A is the normalized adjacency matrix.
    """
    
    def __init__(
        self,
        num_joints: int = 24,
        learnable_scale: bool = True,
        adjacency: Optional[torch.Tensor] = None
    ):
        """
        Initialize graph module.
        
        Args:
            num_joints: Number of joints in skeleton
            learnable_scale: Whether smoothing scale is learnable
            adjacency: Pre-computed adjacency matrix (J, J)
        """
        super().__init__()
        
        self.num_joints = num_joints
        
        # Build adjacency matrix
        if adjacency is not None:
            self.register_buffer('adjacency', adjacency.float())
        else:
            # Default: simple chain skeleton
            adj = self._build_default_adjacency(num_joints)
            self.register_buffer('adjacency', adj)
        
        # Learnable smoothing factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('scale', torch.tensor(0.1))
    
    def _build_default_adjacency(self, num_joints: int) -> torch.Tensor:
        """
        Build default skeleton adjacency matrix.
        
        Uses SMPL-like parent structure for typical humanoid skeleton.
        """
        # SMPL parent indices (simplified for generic skeleton)
        parents = [
            -1,  # 0: pelvis (root)
            0, 0, 0,  # 1-3: hips and spine
            1, 2, 3,  # 4-6: knees and spine2
            4, 5, 6,  # 7-9: ankles and spine3
            7, 8, 9,  # 10-12: feet and neck
            9, 9, 12,  # 13-15: shoulders and head
            13, 14,  # 16-17: elbows
            16, 17,  # 18-19: wrists
            18, 19,  # 20-21: hands
            20, 21,  # 22-23: fingers
        ]
        
        # Truncate to actual number of joints
        parents = parents[:num_joints]
        
        # Build adjacency matrix (symmetric)
        adj = torch.zeros(num_joints, num_joints)
        for child, parent in enumerate(parents):
            if parent >= 0:
                adj[child, parent] = 1.0
                adj[parent, child] = 1.0
        
        # Add self-connections
        adj = adj + torch.eye(num_joints)
        
        # Normalize rows
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        adj = adj / row_sum
        
        return adj
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply graph smoothing to joint positions.
        
        Args:
            positions: (B, J, 3) or (J, 3) joint positions
        
        Returns:
            Smoothed positions with same shape
        """
        squeeze = False
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            squeeze = True
        
        B, J, D = positions.shape
        
        # Ensure adjacency matches
        if J != self.num_joints:
            # Resize adjacency if needed (simple interpolation)
            adj = F.interpolate(
                self.adjacency.unsqueeze(0).unsqueeze(0),
                size=(J, J),
                mode='bilinear'
            ).squeeze()
        else:
            adj = self.adjacency
        
        # Graph convolution: aggregate neighbor information
        # positions: (B, J, 3) -> (B, 3, J) for matmul
        pos_t = positions.transpose(1, 2)  # (B, 3, J)
        smoothed_t = torch.matmul(pos_t, adj.T)  # (B, 3, J)
        smoothed = smoothed_t.transpose(1, 2)  # (B, J, 3)
        
        # Residual connection with learnable scale
        output = positions + self.scale * (smoothed - positions)
        
        if squeeze:
            output = output.squeeze(0)
        
        return output


# ============================================================================
# Positional Encoding
# ============================================================================

class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for time and joint index.
    
    Encodes scalar inputs as:
        [sin(2^0 * pi * x), cos(2^0 * pi * x),
         sin(2^1 * pi * x), cos(2^1 * pi * x),
         ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    
    This helps the network learn high-frequency details.
    """
    
    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        """
        Initialize Fourier encoding.
        
        Args:
            num_frequencies: Number of frequency bands (L)
            include_input: Whether to include original input
        """
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer('freqs', freqs * math.pi)
    
    @property
    def output_dim(self) -> int:
        """Output dimension after encoding."""
        dim = 2 * self.num_frequencies  # sin + cos for each freq
        if self.include_input:
            dim += 1
        return dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding.
        
        Args:
            x: (...,) or (..., 1) input tensor
        
        Returns:
            (..., output_dim) encoded tensor
        """
        if x.dim() == 1 or (x.dim() > 1 and x.size(-1) != 1):
            x = x.unsqueeze(-1)  # (..., 1)
        
        # Compute sin and cos at each frequency
        x_freq = x * self.freqs  # (..., L)
        encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)  # (..., 2L)
        
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)  # (..., 2L+1)
        
        return encoded


# ============================================================================
# Main Geom-INR Model
# ============================================================================

class GeomINR(nn.Module):
    """
    Geom-INR: Implicit Neural Representation for Human Motion.
    
    This model maps (actor_id, joint_index, time) to 3D joint positions
    using a SIREN network with actor conditioning and optional graph refinement.
    
    Architecture:
        1. Actor embedding: Embed discrete actor ID
        2. Input encoding: Normalize time and joint index to [0, 1]
        3. SIREN MLP: Process concatenated features through sine-activated layers
        4. Graph module (optional): Apply skeletal smoothing
        5. Output: 3D (x, y, z) position
    """
    
    def __init__(
        self,
        num_joints: int = 24,
        num_actors: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        actor_embed_dim: int = 32,
        omega_0: float = 30.0,
        use_positional_encoding: bool = False,
        positional_encoding_freqs: int = 6,
        use_graph_module: bool = True,
        max_time: float = 100.0
    ):
        """
        Initialize Geom-INR model.
        
        Args:
            num_joints: Number of joints in skeleton
            num_actors: Maximum number of actor IDs
            hidden_dim: Hidden layer dimension
            num_layers: Number of SIREN layers
            actor_embed_dim: Dimension of actor embedding
            omega_0: Frequency for sine activations
            use_positional_encoding: Whether to use Fourier encoding
            positional_encoding_freqs: Number of Fourier frequencies
            use_graph_module: Whether to apply graph smoothing
            max_time: Maximum time value (for normalization)
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.num_actors = num_actors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega_0 = omega_0
        self.max_time = max_time
        self.use_graph_module = use_graph_module
        
        # Actor embedding
        self.actor_embed = nn.Embedding(num_actors, actor_embed_dim)
        
        # Positional encoding (optional)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.time_encoder = FourierPositionalEncoding(positional_encoding_freqs)
            self.joint_encoder = FourierPositionalEncoding(positional_encoding_freqs)
            input_dim = (
                self.time_encoder.output_dim +
                self.joint_encoder.output_dim +
                actor_embed_dim
            )
        else:
            input_dim = 1 + 1 + actor_embed_dim  # time + joint_idx + actor
        
        # Build SIREN network
        self.layers = nn.ModuleList()
        
        # First layer (different initialization)
        self.layers.append(SIRENLayer(
            input_dim, hidden_dim,
            omega=omega_0, is_first=True
        ))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(SIRENLayer(
                hidden_dim, hidden_dim,
                omega=omega_0, is_first=False
            ))
        
        # Output layer (no activation)
        self.output_layer = nn.Linear(hidden_dim, 3)
        self._init_output_layer()
        
        # Graph module for skeletal consistency
        if use_graph_module:
            self.graph_module = SkeletonGraphModule(num_joints)
        else:
            self.graph_module = None
    
    def _init_output_layer(self):
        """Initialize output layer with small weights."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_dim) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()
    
    def forward(
        self,
        actor_ids: torch.Tensor,
        joint_indices: torch.Tensor,
        times: torch.Tensor,
        apply_graph: bool = True
    ) -> torch.Tensor:
        """
        Forward pass: predict joint positions.
        
        Args:
            actor_ids: (B,) tensor of actor IDs (long)
            joint_indices: (B,) tensor of joint indices (long or float)
            times: (B,) tensor of continuous time values (float)
            apply_graph: Whether to apply graph smoothing
        
        Returns:
            positions: (B, 3) tensor of predicted positions
        """
        B = actor_ids.shape[0]
        device = actor_ids.device
        
        # Embed actor ID
        actor_feat = self.actor_embed(actor_ids)  # (B, actor_embed_dim)
        
        # Normalize time and joint index to [0, 1]
        time_norm = times / self.max_time  # (B,)
        joint_norm = joint_indices.float() / max(1, self.num_joints - 1)  # (B,)
        
        # Encode inputs
        if self.use_positional_encoding:
            time_feat = self.time_encoder(time_norm)  # (B, time_dim)
            joint_feat = self.joint_encoder(joint_norm)  # (B, joint_dim)
        else:
            time_feat = time_norm.unsqueeze(-1)  # (B, 1)
            joint_feat = joint_norm.unsqueeze(-1)  # (B, 1)
        
        # Concatenate features
        x = torch.cat([time_feat, joint_feat, actor_feat], dim=-1)  # (B, input_dim)
        
        # Pass through SIREN layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        positions = self.output_layer(x)  # (B, 3)
        
        # Apply graph smoothing if enabled and batch represents full skeleton
        if apply_graph and self.graph_module is not None:
            # Check if batch represents a full pose (all joints at same time)
            if B == self.num_joints:
                # Reshape to (1, J, 3) for graph module
                positions = self.graph_module(positions.unsqueeze(0)).squeeze(0)
        
        return positions
    
    def predict_pose(
        self,
        actor_id: int,
        time: float,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Predict full pose at a given time.
        
        Args:
            actor_id: Actor ID
            time: Time value
            device: Device for computation
        
        Returns:
            positions: (J, 3) tensor of joint positions
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Create inputs for all joints
        actor_ids = torch.full((self.num_joints,), actor_id, dtype=torch.long, device=device)
        joint_indices = torch.arange(self.num_joints, dtype=torch.long, device=device)
        times = torch.full((self.num_joints,), time, dtype=torch.float32, device=device)
        
        return self.forward(actor_ids, joint_indices, times)
    
    def predict_trajectory(
        self,
        actor_id: int,
        times: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Predict motion trajectory over multiple time steps.
        
        Args:
            actor_id: Actor ID
            times: (T,) tensor of time values
            device: Device for computation
        
        Returns:
            trajectory: (T, J, 3) tensor of joint positions
        """
        if device is None:
            device = next(self.parameters()).device
        
        times = times.to(device)
        T = times.shape[0]
        
        trajectory = torch.zeros(T, self.num_joints, 3, device=device)
        
        for t_idx, t in enumerate(times):
            trajectory[t_idx] = self.predict_pose(actor_id, t.item(), device)
        
        return trajectory
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Model Variants
# ============================================================================

class GeomINRSmall(GeomINR):
    """Small variant for quick testing."""
    
    def __init__(self, num_joints: int = 24, **kwargs):
        super().__init__(
            num_joints=num_joints,
            hidden_dim=128,
            num_layers=3,
            actor_embed_dim=16,
            **kwargs
        )


class GeomINRLarge(GeomINR):
    """Large variant for better capacity."""
    
    def __init__(self, num_joints: int = 24, **kwargs):
        super().__init__(
            num_joints=num_joints,
            hidden_dim=512,
            num_layers=6,
            actor_embed_dim=64,
            use_positional_encoding=True,
            **kwargs
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_model(
    variant: str = "base",
    num_joints: int = 24,
    num_actors: int = 100,
    **kwargs
) -> GeomINR:
    """
    Create a Geom-INR model variant.
    
    Args:
        variant: Model variant ("small", "base", "large")
        num_joints: Number of skeleton joints
        num_actors: Number of actor IDs to support
        **kwargs: Additional model arguments
    
    Returns:
        GeomINR model instance
    """
    variants = {
        "small": GeomINRSmall,
        "base": GeomINR,
        "large": GeomINRLarge,
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    return variants[variant](num_joints=num_joints, num_actors=num_actors, **kwargs)


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Geom-INR Model Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    print("\n[1] Creating model...")
    model = GeomINR(
        num_joints=24,
        num_actors=100,
        hidden_dim=256,
        num_layers=4
    ).to(device)
    
    print(f"    Parameters: {model.get_num_params():,}")
    
    # Test forward pass
    print("\n[2] Testing forward pass...")
    batch_size = 24  # Full skeleton
    
    actor_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    joint_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    times = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        positions = model(actor_ids, joint_indices, times)
    
    print(f"    Input: actor_ids={actor_ids.shape}, joints={joint_indices.shape}, times={times.shape}")
    print(f"    Output: {positions.shape}")
    print(f"    Sample output: {positions[0].cpu().numpy()}")
    
    # Test gradient flow
    print("\n[3] Testing gradient flow...")
    actor_ids.requires_grad = False
    joint_indices_float = joint_indices.float()
    times.requires_grad = False
    
    positions = model(actor_ids, joint_indices, times)
    loss = positions.mean()
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    print(f"    Gradient norms (first 3):")
    for name, norm in grad_norms[:3]:
        print(f"      {name}: {norm:.6f}")
    
    # Test pose prediction
    print("\n[4] Testing pose prediction...")
    with torch.no_grad():
        pose = model.predict_pose(actor_id=0, time=10.0, device=device)
    
    print(f"    Pose shape: {pose.shape}")
    
    # Test trajectory prediction
    print("\n[5] Testing trajectory prediction...")
    with torch.no_grad():
        times_seq = torch.linspace(0, 50, 10)
        trajectory = model.predict_trajectory(actor_id=0, times=times_seq, device=device)
    
    print(f"    Trajectory shape: {trajectory.shape}")
    
    # Test model variants
    print("\n[6] Testing model variants...")
    for variant in ["small", "base", "large"]:
        m = create_model(variant, num_joints=24)
        print(f"    {variant}: {m.get_num_params():,} parameters")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\n    GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    print("\n✓ Model test completed!")
