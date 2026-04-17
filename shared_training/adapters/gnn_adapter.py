"""
GNN Adapter for Graph Neural Network models.

This adapter handles models where the state includes graph structure
that may change between timesteps (e.g., mesh deformation updates
edge connectivity).

Designed for:
- Project 05 (ClothGNN): GNN with PyG Data objects
- Project 08 (HGNN-ClothDyn): Hierarchical GNN with physics-encoded edges
- Project 06 (CoastFlow-GNN): Multi-scale GNN for coastal flow
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base_adapter import RolloutModelAdapter

# Try to import PyG, but don't fail if not available
try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None
    Batch = None


class GNNAdapter(RolloutModelAdapter):
    """Adapter for Graph Neural Network models.
    
    Handles models that operate on graph-structured data where:
    - Nodes represent physical entities (particles, mesh vertices)
    - Edges represent interactions (springs, proximity)
    - Graph structure may evolve over time
    
    Supports both:
    - Fixed graph structure (edges don't change)
    - Dynamic graph structure (edges updated based on positions)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        position_key: str = 'positions',
        velocity_key: str = 'velocities',
        edge_key: str = 'edge_index',
        rest_lengths_key: str = 'rest_lengths',
        node_features_key: Optional[str] = None,
        use_hierarchical: bool = False,
        coarse_edge_key: str = 'coarse_edge_index',
        cluster_map_key: str = 'cluster_map',
        update_edges: bool = False,
        edge_radius: float = 0.1,
        dt: float = 1.0 / 30.0,
        damping: float = 0.99,
        physics_losses: Optional[Dict[str, Callable]] = None,
        physics_loss_weights: Optional[Dict[str, float]] = None,
    ):
        if not HAS_PYG:
            raise ImportError(
                "GNNAdapter requires torch_geometric. "
                "Install with: pip install torch-geometric"
            )
        
        self._model = model
        self.loss_fn = loss_fn
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.edge_key = edge_key
        self.rest_lengths_key = rest_lengths_key
        self.node_features_key = node_features_key
        self.use_hierarchical = use_hierarchical
        self.coarse_edge_key = coarse_edge_key
        self.cluster_map_key = cluster_map_key
        self.update_edges = update_edges
        self.edge_radius = edge_radius
        self.dt = dt
        self.damping = damping
        self.physics_losses = physics_losses or {}
        self.physics_loss_weights = physics_loss_weights or {}
        
        # State tracking
        self._current_pos: Optional[Tensor] = None
        self._current_vel: Optional[Tensor] = None
        self._prev_pos: Optional[Tensor] = None
        self._edge_index: Optional[Tensor] = None
        self._rest_lengths: Optional[Tensor] = None
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    def init_hidden(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        positions = batch[self.position_key]
        
        if positions.dim() == 3:
            self._current_pos = positions.clone()
        elif positions.dim() == 4:
            self._current_pos = positions[:, 0].clone()
        
        self._current_vel = torch.zeros_like(self._current_pos)
        self._prev_pos = None
        
        self._edge_index = batch.get(self.edge_key)
        self._rest_lengths = batch.get(self.rest_lengths_key)
        
        if hasattr(self._model, 'init_hidden'):
            batch_size = self._current_pos.size(0)
            device = self._current_pos.device
            return self._model.init_hidden(batch_size, device)
        
        return None
    
    def step(
        self,
        state: Tensor,
        hidden: Optional[Tensor],
        t: int,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.update_edges and self._current_pos is not None:
            self._edge_index = self._compute_edges(state)
        
        edge_index = self._edge_index or batch.get(self.edge_key)
        
        if self._prev_pos is not None:
            velocity = (state - self._prev_pos) / self.dt
        elif self.velocity_key in batch:
            velocities = batch[self.velocity_key]
            velocity = velocities[:, t] if velocities.dim() == 4 else velocities
        else:
            velocity = torch.zeros_like(state)
        
        node_features = self._build_node_features(state, velocity, batch)
        data = self._build_pyg_data(node_features, state, edge_index, batch)
        
        if self.use_hierarchical:
            coarse_data = self._build_coarse_data(batch)
            cluster_map = batch.get(self.cluster_map_key)
            
            if hidden is not None:
                delta_vel, new_hidden = self._model(data, coarse_data, cluster_map, hidden)
            else:
                delta_vel = self._model(data, coarse_data, cluster_map)
                new_hidden = None
        else:
            if hidden is not None:
                delta_vel, new_hidden = self._model(data, hidden)
            else:
                delta_vel = self._model(data)
                new_hidden = None
        
        new_vel = (velocity + delta_vel) * self.damping
        prediction = state + new_vel * self.dt
        
        self._prev_pos = state.detach()
        self._current_pos = prediction.detach()
        self._current_vel = new_vel.detach()
        
        return prediction, new_hidden
    
    def _build_node_features(
        self,
        position: Tensor,
        velocity: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        features = [position, velocity]
        
        if self.node_features_key and self.node_features_key in batch:
            extra_features = batch[self.node_features_key]
            features.append(extra_features)
        
        return torch.cat(features, dim=-1)
    
    def _build_pyg_data(
        self,
        node_features: Tensor,
        position: Tensor,
        edge_index: Tensor,
        batch: Dict[str, Tensor],
    ) -> 'Data':
        if node_features.dim() == 3:
            batch_size, num_nodes, feat_dim = node_features.shape
            
            x = node_features.view(-1, feat_dim)
            pos = position.view(-1, 3)
            
            if edge_index.dim() == 2:
                edge_indices = []
                for b in range(batch_size):
                    offset = b * num_nodes
                    edge_indices.append(edge_index + offset)
                edge_index = torch.cat(edge_indices, dim=1)
            
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = batch_idx.repeat_interleave(num_nodes)
            
            data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch_idx)
        else:
            data = Data(x=node_features, pos=position, edge_index=edge_index)
        
        if self._rest_lengths is not None:
            data.rest_lengths = self._rest_lengths
        
        return data
    
    def _build_coarse_data(self, batch: Dict[str, Tensor]) -> Optional['Data']:
        coarse_edge_index = batch.get(self.coarse_edge_key)
        if coarse_edge_index is None:
            return None
        
        coarse_pos = batch.get('coarse_positions')
        if coarse_pos is None and self._current_pos is not None:
            cluster_map = batch.get(self.cluster_map_key)
            if cluster_map is not None:
                num_clusters = cluster_map.max() + 1
                coarse_pos = torch.zeros(
                    num_clusters, 3, device=self._current_pos.device
                )
                for c in range(num_clusters):
                    mask = cluster_map == c
                    coarse_pos[c] = self._current_pos[:, mask].mean(dim=1)
        
        return Data(edge_index=coarse_edge_index, pos=coarse_pos)
    
    def _compute_edges(self, position: Tensor) -> Tensor:
        from torch_geometric.nn import radius_graph
        
        if position.dim() == 3:
            pos_flat = position.view(-1, 3)
            batch_idx = torch.arange(
                position.size(0), device=position.device
            ).repeat_interleave(position.size(1))
        else:
            pos_flat = position
            batch_idx = None
        
        edge_index = radius_graph(
            pos_flat, r=self.edge_radius, batch=batch_idx
        )
        
        return edge_index
    
    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        is_teacher_forced: bool = True,
        **kwargs,
    ) -> Tensor:
        loss = self.loss_fn(prediction, target)
        
        for name, loss_fn in self.physics_losses.items():
            weight = self.physics_loss_weights.get(name, 1.0)
            
            physics_kwargs = {
                'prediction': prediction,
                'target': target,
                'velocity': self._current_vel,
                'prev_position': self._prev_pos,
                'edge_index': self._edge_index,
                'rest_lengths': self._rest_lengths,
                'dt': self.dt,
            }
            
            try:
                physics_loss = loss_fn(**physics_kwargs)
            except TypeError:
                physics_loss = loss_fn(prediction, target)
            
            loss = loss + weight * physics_loss
        
        return loss
    
    def get_initial_state(self, batch: Dict[str, Tensor]) -> Tensor:
        positions = batch[self.position_key]
        return positions[:, 0] if positions.dim() == 4 else positions
    
    def get_target(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        positions = batch[self.position_key]
        return positions[:, t]
    
    def get_teacher_input(self, batch: Dict[str, Tensor], t: int) -> Tensor:
        return self.get_target(batch, t)
    
    def prediction_to_input(self, prediction: Tensor) -> Tensor:
        return prediction


class Project08Adapter(GNNAdapter):
    """Specialized adapter for Project 08 (HGNN-ClothDyn)."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        **kwargs,
    ):
        physics_losses = {
            'edge_length': self._edge_length_loss,
            'velocity_smoothness': self._velocity_smoothness_loss,
        }
        physics_loss_weights = {
            'edge_length': 0.1,
            'velocity_smoothness': 0.01,
        }
        
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            use_hierarchical=True,
            physics_losses=physics_losses,
            physics_loss_weights=physics_loss_weights,
            **kwargs,
        )
    
    def _edge_length_loss(
        self,
        prediction: Tensor,
        edge_index: Tensor,
        rest_lengths: Tensor,
        **kwargs,
    ) -> Tensor:
        if edge_index is None or rest_lengths is None:
            return torch.tensor(0.0, device=prediction.device)
        
        if prediction.dim() == 3:
            prediction = prediction.view(-1, 3)
        
        src, dst = edge_index
        edge_vec = prediction[dst] - prediction[src]
        current_lengths = torch.norm(edge_vec, dim=-1)
        
        strain = (current_lengths - rest_lengths) / (rest_lengths + 1e-8)
        
        return (strain ** 2).mean()
    
    def _velocity_smoothness_loss(
        self,
        velocity: Tensor,
        **kwargs,
    ) -> Tensor:
        if velocity is None:
            return torch.tensor(0.0)
        
        return (velocity ** 2).mean()
