# HGNN-NIF-Cloth Model Components
"""
Core neural network modules for hybrid cloth simulation.

Components:
- GraphConv: Message-passing graph convolution layer
- CrossLevelAttention: Multi-resolution attention mechanism
- AdaptiveHGNN: Hierarchical GNN with energy-based resolution control
- SIRENDecoder: Latent-conditioned sinusoidal implicit decoder
- HGNN_NIF_ClothModel: End-to-end hybrid model
- TemporalHGNN_NIF: Temporal extension for animation prediction
"""

from .graph_conv import GraphConv
from .attention import CrossLevelAttention
from .hgnn import AdaptiveHGNN
from .siren import SirenLayer, SIRENDecoder
from .hybrid_model import HGNN_NIF_ClothModel
from .temporal import TemporalHGNN_NIF, TemporalHGNN_NIF_Attention

__all__ = [
    "GraphConv",
    "CrossLevelAttention",
    "AdaptiveHGNN",
    "SirenLayer",
    "SIRENDecoder",
    "HGNN_NIF_ClothModel",
    "TemporalHGNN_NIF",
    "TemporalHGNN_NIF_Attention",
]
