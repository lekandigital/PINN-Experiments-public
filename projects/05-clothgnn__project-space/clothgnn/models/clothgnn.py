"""
ClothGNN: Encoder -> GRU -> Decoder architecture.
Recurrent GNN for cloth and hair dynamics prediction.

Based on MeshGraphNetRP architecture with temporal consistency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class ClothEncoder(MessagePassing):
    """
    GNN-based encoder processing mesh node/edge features.
    Uses mean aggregation for neighbor features.
    
    Args:
        in_channels: Input node feature dimension
        hidden_channels: Hidden/output dimension
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__(aggr="mean")
        
        # Node feature projection
        self.lin_node = nn.Linear(in_channels, hidden_channels)
        
        # Edge feature projection (for edge-based messages)
        # Edge features: relative position (3) + distance (1) = 4
        self.lin_edge = nn.Linear(4, hidden_channels)
        
        # Message MLP
        self.lin_msg = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, pos=None):
        """
        Forward pass through the encoder.
        
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connectivity
            pos: (N, 3) optional node positions for edge features
            
        Returns:
            h: (N, hidden_channels) encoded node embeddings
        """
        # Project node features
        h = self.lin_node(x)
        
        # Compute edge features if positions provided
        if pos is not None:
            row, col = edge_index
            rel_pos = pos[col] - pos[row]  # (E, 3)
            dist = rel_pos.norm(dim=1, keepdim=True)  # (E, 1)
            edge_attr = torch.cat([rel_pos, dist], dim=1)  # (E, 4)
        else:
            edge_attr = None
        
        # Message passing
        out = self.propagate(edge_index, x=h, edge_attr=edge_attr)
        
        # Residual connection + normalization
        out = self.norm(h + out)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages from neighbors.
        
        Args:
            x_i: (E, hidden) target node features
            x_j: (E, hidden) source node features
            edge_attr: (E, 4) edge features
        """
        if edge_attr is not None:
            edge_feat = self.lin_edge(edge_attr)
            msg_input = torch.cat([x_i, x_j, edge_feat], dim=1)
        else:
            msg_input = torch.cat([x_i, x_j, torch.zeros_like(x_i)], dim=1)
        
        return self.lin_msg(msg_input)


class ClothDecoder(nn.Module):
    """
    MLP-based decoder mapping latent embeddings to displacements.
    Outputs 3D vertex displacement predictions.
    
    Args:
        hidden_channels: Input embedding dimension
        out_channels: Output dimension (3 for 3D displacements)
    """
    def __init__(self, hidden_channels, out_channels=3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, h):
        """
        Decode embeddings to displacements.
        
        Args:
            h: (N, hidden_channels) node embeddings
            
        Returns:
            displacements: (N, out_channels) predicted displacements
        """
        return self.mlp(h)


class ClothGNNModel(nn.Module):
    """
    Complete ClothGNN model: Encoder -> GRU -> Decoder.
    Predicts vertex displacements with temporal consistency.
    
    Args:
        node_feat_dim: Input node feature dimension (default: 16)
        hidden_dim: Hidden state dimension (default: 64)
    """
    def __init__(self, node_feat_dim=16, hidden_dim=64):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Encoder: process node features into latent space
        self.encoder = ClothEncoder(node_feat_dim, hidden_dim)
        
        # GRU: temporal dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Decoder: predict displacements from hidden state
        self.decoder = ClothDecoder(hidden_dim, out_channels=3)
        
    def forward(self, data, h):
        """
        Forward pass for one timestep.
        
        Args:
            data: PyG Data object with:
                - x: (N, node_feat_dim) node features
                - edge_index: (2, E) edge connectivity
                - pos: (N, 3) optional positions for edge features
            h: (N, hidden_dim) current hidden state
            
        Returns:
            predictions: (N, 3) displacement predictions
            h_next: (N, hidden_dim) updated hidden state
        """
        # Get position if available
        pos = data.pos if hasattr(data, "pos") else None
        
        # Encode current state
        encoded = self.encoder(data.x, data.edge_index, pos)
        
        # Update hidden state with GRU
        h_next = self.gru(encoded, h)
        
        # Decode to displacements
        predictions = self.decoder(h_next)
        
        return predictions, h_next
    
    def init_hidden(self, num_nodes, device):
        """
        Initialize hidden state for a new sequence.
        
        Args:
            num_nodes: Number of nodes in the graph
            device: Target device
            
        Returns:
            h0: (num_nodes, hidden_dim) zero-initialized hidden state
        """
        return torch.zeros(num_nodes, self.hidden_dim, device=device)
    
    def rollout(self, data, h, num_steps, return_hidden=False):
        """
        Multi-step prediction rollout.
        
        Args:
            data: Initial PyG Data object (x will be updated each step)
            h: Initial hidden state
            num_steps: Number of steps to roll out
            return_hidden: Whether to return hidden states
            
        Returns:
            predictions: List of (N, 3) displacement predictions
            h_final: Final hidden state (or list if return_hidden=True)
        """
        predictions = []
        hidden_states = [] if return_hidden else None
        
        current_pos = data.pos.clone() if hasattr(data, "pos") else None
        
        for step in range(num_steps):
            # Forward pass
            pred, h = self.forward(data, h)
            predictions.append(pred)
            
            if return_hidden:
                hidden_states.append(h)
            
            # Update positions if available (for next step edge features)
            if current_pos is not None:
                current_pos = current_pos + pred
                data.pos = current_pos
        
        if return_hidden:
            return predictions, hidden_states
        return predictions, h

    def save_checkpoint(self, path, optimizer=None, epoch=None, loss=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'config': {
                'node_feat_dim': self.encoder.lin_node.in_features,
                'hidden_dim': self.hidden_dim,
            }
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device='cuda'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
