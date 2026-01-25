"""
Temporal Dynamics Module for HGNN-NIF-Cloth Animation

Extends the base HGNN-NIF model to predict frame-to-frame dynamics
for smooth cloth animation generation.

Given frames [t-2, t-1, t], predict frame [t+1] using velocity-based
conditioning for physically plausible motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .hybrid_model import HGNN_NIF_ClothModel


class TemporalHGNN_NIF(nn.Module):
    """
    Temporal extension of HGNN-NIF for cloth dynamics prediction.

    Uses a GRU to encode velocity history and predict next-frame
    displacements, enabling smooth animation generation.

    Args:
        base_model: Pre-trained HGNN_NIF_ClothModel for static encoding
        history_frames: Number of historical frames to consider (default: 3)
        temporal_hidden: Hidden dimension for temporal encoder (default: 64)
        predict_velocity: If True, predict velocity; else predict position directly

    Example:
        >>> base = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64)
        >>> temporal = TemporalHGNN_NIF(base, history_frames=3)
        >>> history = torch.randn(1, 3, 400, 3)  # (B, T, N, 3)
        >>> output = temporal(history, fine_edges, coarse_edges, query_points)
        >>> next_pos = output['positions']  # (B, N, 3)
    """

    def __init__(
        self,
        base_model: HGNN_NIF_ClothModel,
        history_frames: int = 3,
        temporal_hidden: int = 64,
        predict_velocity: bool = True
    ):
        super().__init__()

        self.base_model = base_model
        self.history_frames = history_frames
        self.temporal_hidden = temporal_hidden
        self.predict_velocity = predict_velocity
        self.latent_dim = base_model.latent_dim

        # Temporal encoder: processes velocity history
        self.velocity_encoder = nn.GRU(
            input_size=3,  # dx, dy, dz per vertex
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Velocity predictor: predicts next frame displacement
        self.velocity_predictor = nn.Sequential(
            nn.Linear(temporal_hidden + base_model.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Predict per-vertex velocity
        )

        # Optional: physics-aware feature encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(6, 32),  # velocity + position offset
            nn.ReLU(),
            nn.Linear(32, temporal_hidden)
        )

    def forward(
        self,
        frame_history: torch.Tensor,
        fine_edges: torch.Tensor,
        coarse_edges: torch.Tensor,
        query_points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for temporal prediction.

        Args:
            frame_history: (B, T, N_fine, 3) - T frames of fine positions
            fine_edges: (2, E_fine) edge indices
            coarse_edges: (2, E_coarse) edge indices
            query_points: Optional (B, Q, 3) points to query SDF

        Returns:
            Dict containing:
                - 'positions': (B, N_fine, 3) - Predicted next frame positions
                - 'sdf': (B*Q,) - SDF at query points (if provided)
                - 'velocity': (B, N_fine, 3) - Predicted velocities
                - 'latent': (B, latent_dim) - Latent code from current frame
        """
        B, T, N, _ = frame_history.shape
        device = frame_history.device

        # Compute velocities from position differences
        velocities = frame_history[:, 1:] - frame_history[:, :-1]  # (B, T-1, N, 3)
        velocities = velocities.reshape(B * N, T - 1, 3)

        # Encode velocity history with GRU
        _, h_n = self.velocity_encoder(velocities)
        temporal_feat = h_n[-1].reshape(B, N, -1)  # (B, N, temporal_hidden)

        # Get current frame and decimate to coarse
        current_frame = frame_history[:, -1]  # (B, N, 3)
        coarse_frame = self._decimate(current_frame)  # (B, N_coarse, 3)

        # Encode with base HGNN
        latent = self.base_model.encode(
            (current_frame, fine_edges),
            (coarse_frame, coarse_edges)
        )  # (B, latent_dim)

        # Expand latent to match vertex count
        latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)

        # Combine temporal and spatial features
        combined = torch.cat([temporal_feat, latent_expanded], dim=-1)

        # Predict velocity for next frame
        pred_velocity = self.velocity_predictor(combined)  # (B, N, 3)

        # Next frame = current + predicted velocity
        pred_positions = current_frame + pred_velocity

        output = {
            'positions': pred_positions,
            'velocity': pred_velocity,
            'latent': latent
        }

        # Decode SDF at query points if provided
        if query_points is not None:
            pred_sdf = self.base_model.decode(query_points, latent)
            output['sdf'] = pred_sdf

        return output

    def _decimate(self, fine_pos: torch.Tensor, stride: int = 2) -> torch.Tensor:
        """
        Subsample fine mesh to coarse mesh.

        Assumes the mesh is a regular grid that can be subsampled by stride.

        Args:
            fine_pos: (B, N, 3) fine vertex positions
            stride: Subsampling stride (default: 2)

        Returns:
            coarse_pos: (B, N_coarse, 3) coarse vertex positions
        """
        B, N, _ = fine_pos.shape
        side = int(np.sqrt(N))

        if side * side != N:
            # Non-square mesh: use simple subsampling
            return fine_pos[:, ::stride**2, :]

        # Reshape to grid and subsample
        grid = fine_pos.reshape(B, side, side, 3)
        coarse = grid[:, ::stride, ::stride, :].reshape(B, -1, 3)
        return coarse

    def autoregressive_rollout(
        self,
        initial_history: torch.Tensor,
        fine_edges: torch.Tensor,
        coarse_edges: torch.Tensor,
        num_frames: int,
        fixed_vertices: Optional[torch.Tensor] = None,
        initial_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate animation through autoregressive rollout.

        Args:
            initial_history: (B, T, N, 3) initial frame history
            fine_edges: (2, E_fine) edge indices
            coarse_edges: (2, E_coarse) edge indices
            num_frames: Number of frames to generate
            fixed_vertices: Optional (M,) indices of vertices to keep fixed
            initial_positions: Optional (B, N, 3) positions for fixed vertices

        Returns:
            positions_sequence: (B, num_frames, N, 3) generated animation
        """
        self.eval()
        B, T, N, _ = initial_history.shape
        device = initial_history.device

        # Build frame history buffer
        history = list(initial_history.unbind(dim=1))
        positions_sequence = []

        with torch.no_grad():
            for _ in range(num_frames):
                # Stack history
                history_tensor = torch.stack(history[-T:], dim=1)

                # Predict next frame
                output = self.forward(history_tensor, fine_edges, coarse_edges)
                pred_pos = output['positions']

                # Apply fixed vertex constraints
                if fixed_vertices is not None and initial_positions is not None:
                    pred_pos[:, fixed_vertices, :] = initial_positions[:, fixed_vertices, :]

                # Store and update history
                positions_sequence.append(pred_pos)
                history.append(pred_pos)

        return torch.stack(positions_sequence, dim=1)


class TemporalAttentionBlock(nn.Module):
    """
    Attention-based temporal feature aggregation.

    Alternative to GRU for capturing long-range temporal dependencies.
    """

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) temporal features

        Returns:
            out: (B, T, D) attended features
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TemporalHGNN_NIF_Attention(TemporalHGNN_NIF):
    """
    Temporal HGNN-NIF variant using attention instead of GRU.

    May capture longer-range temporal dependencies better for
    complex cloth dynamics.
    """

    def __init__(
        self,
        base_model: HGNN_NIF_ClothModel,
        history_frames: int = 3,
        temporal_hidden: int = 64,
        num_heads: int = 4
    ):
        super().__init__(base_model, history_frames, temporal_hidden)

        # Replace GRU with attention
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, temporal_hidden),
            TemporalAttentionBlock(temporal_hidden, num_heads),
            TemporalAttentionBlock(temporal_hidden, num_heads)
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        frame_history: torch.Tensor,
        fine_edges: torch.Tensor,
        coarse_edges: torch.Tensor,
        query_points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using attention-based temporal encoding."""
        B, T, N, _ = frame_history.shape

        # Compute velocities
        velocities = frame_history[:, 1:] - frame_history[:, :-1]
        velocities = velocities.reshape(B * N, T - 1, 3)

        # Encode with attention
        temporal_feat = self.velocity_encoder(velocities)  # (B*N, T-1, hidden)

        # Pool over time
        temporal_feat = temporal_feat.transpose(1, 2)  # (B*N, hidden, T-1)
        temporal_feat = self.temporal_pool(temporal_feat).squeeze(-1)  # (B*N, hidden)
        temporal_feat = temporal_feat.reshape(B, N, -1)

        # Rest is same as parent
        current_frame = frame_history[:, -1]
        coarse_frame = self._decimate(current_frame)

        latent = self.base_model.encode(
            (current_frame, fine_edges),
            (coarse_frame, coarse_edges)
        )

        latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([temporal_feat, latent_expanded], dim=-1)
        pred_velocity = self.velocity_predictor(combined)
        pred_positions = current_frame + pred_velocity

        output = {
            'positions': pred_positions,
            'velocity': pred_velocity,
            'latent': latent
        }

        if query_points is not None:
            output['sdf'] = self.base_model.decode(query_points, latent)

        return output
