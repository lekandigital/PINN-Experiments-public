"""
Tangent Space Operations Module
===============================

Operations in local tangent planes of manifolds, including:
- Tangent basis computation
- Parallel transport between tangent planes
- Tangent-space message passing for GNNs

Migrated from Project 01 (GeoPINN-Manifold) for use across multiple projects.

Key Components:
- compute_tangent_basis: Gram-Schmidt orthonormalization from normals
- TangentMessagePassing: GNN layer with tangent-plane awareness
- TangentMessagePassingStack: Multi-layer stack with residuals
- parallel_transport: Transport vectors between tangent planes

Usage:
    >>> from diffgeo.tangent import TangentMessagePassing, compute_tangent_basis
    >>>
    >>> # Build tangent basis from normals
    >>> t1, t2 = compute_tangent_basis(normals)
    >>>
    >>> # Tangent-aware message passing
    >>> layer = TangentMessagePassing(in_features=16, out_features=32)
    >>> out = layer(features, positions, normals, edge_index)
"""

from .basis import (
    compute_tangent_basis,
    compute_tangent_basis_np,
    parallel_transport,
    parallel_transport_batch,
)
from .message_passing import (
    TangentMessagePassing,
    TangentMessagePassingStack,
    TangentConv,
)

__all__ = [
    # Tangent basis
    "compute_tangent_basis",
    "compute_tangent_basis_np",
    # Parallel transport
    "parallel_transport",
    "parallel_transport_batch",
    # Message passing layers
    "TangentMessagePassing",
    "TangentMessagePassingStack",
    "TangentConv",
]
