"""
Temporal GRU module for conditioning the neural implicit field.

The GRU maintains hidden state across temporal rollouts, enabling the model
to capture temporal hysteresis and history-dependent cloth behavior (e.g.,
the cloth "remembers" its previous motion for more realistic dynamics).

This is optional but improves temporal smoothness in long sequences.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemporalGRU(nn.Module):
    """
    GRU-based temporal conditioning module.
    
    Takes features from the main MLP and conditions them on temporal history.
    The hidden state is passed between frames during rollout.
    
    Args:
        input_dim: Dimension of input features (from MLP hidden layer)
        hidden_dim: GRU hidden state dimension
        num_layers: Number of stacked GRU layers
        dropout: Dropout probability (applied between layers if num_layers > 1)
        bidirectional: Whether to use bidirectional GRU (only for offline processing)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Main GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Output projection to match input dimension for residual connection
        output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Linear(output_dim, input_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize GRU with orthogonal initialization for stability."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 3 : 2 * n // 3].fill_(1.0)
        
        # Output projection initialized to near-identity
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize hidden state to zeros.
        
        Args:
            batch_size: Batch size for the hidden state
            device: Device to create tensor on
            dtype: Data type (float32 or float16 for AMP)
            
        Returns:
            Zero-initialized hidden state of shape (num_layers * num_directions, batch, hidden_dim)
        """
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process features through GRU with optional hidden state.
        
        Args:
            x: Input features of shape (batch, input_dim)
               Treated as a single timestep sequence
            hidden: Optional hidden state from previous timestep
                   Shape: (num_layers * num_directions, batch, hidden_dim)
                   If None, initialized to zeros
                   
        Returns:
            output: Conditioned features of shape (batch, input_dim)
            new_hidden: Updated hidden state for next timestep
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, device, dtype)
        
        # Reshape to (batch, seq_len=1, features) for GRU
        x_seq = x.unsqueeze(1)
        
        # GRU forward pass
        gru_out, new_hidden = self.gru(x_seq, hidden)
        
        # Remove sequence dimension: (batch, 1, hidden*directions) -> (batch, hidden*directions)
        gru_out = gru_out.squeeze(1)
        
        # Project back to input dimension
        projected = self.output_proj(gru_out)
        
        # Residual connection with layer norm
        output = self.layer_norm(x + projected)
        
        return output, new_hidden
    
    def forward_sequence(
        self,
        x_sequence: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process an entire sequence at once (for training efficiency).
        
        Args:
            x_sequence: Input sequence of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            output_sequence: Conditioned features (batch, seq_len, input_dim)
            final_hidden: Final hidden state
        """
        batch_size = x_sequence.size(0)
        device = x_sequence.device
        dtype = x_sequence.dtype
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, device, dtype)
        
        # GRU over full sequence
        gru_out, final_hidden = self.gru(x_sequence, hidden)
        
        # Project and residual
        projected = self.output_proj(gru_out)
        output_sequence = self.layer_norm(x_sequence + projected)
        
        return output_sequence, final_hidden


class TemporalAttention(nn.Module):
    """
    Alternative temporal module using self-attention.
    
    More expressive than GRU but requires storing all previous frames.
    Best used for offline processing where all frames are available.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Positional encoding
        self.register_buffer(
            'positional_encoding',
            self._create_positional_encoding(max_seq_len, feature_dim),
        )
    
    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int,
    ) -> torch.Tensor:
        """Sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal attention over sequence.
        
        Args:
            x: Input sequence (batch, seq_len, feature_dim)
            mask: Optional attention mask
            
        Returns:
            Output sequence (batch, seq_len, feature_dim)
        """
        seq_len = x.size(1)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Self-attention with residual
        attended, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))
        
        return x
