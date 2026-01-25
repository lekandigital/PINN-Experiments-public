"""
GeoPINN Layers - Geometry-aware neural network layers for manifold PDEs
"""

from .tangent_message_passing import TangentMessagePassing
from .spectral_conv import SpectralGraphConv
from .dec_operators import build_dec_operators, DECLaplacian
from .chart_atlas import Atlas, ChartMLP

__all__ = [
    'TangentMessagePassing',
    'SpectralGraphConv', 
    'build_dec_operators',
    'DECLaplacian',
    'Atlas',
    'ChartMLP'
]
