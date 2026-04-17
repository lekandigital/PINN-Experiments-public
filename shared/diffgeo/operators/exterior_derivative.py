"""
Exterior Derivative Operators
=============================

Build the discrete exterior derivative matrices d0 and d1.

These are the fundamental building blocks of DEC:
    d0: 0-forms → 1-forms  (gradient, maps vertex values to edge differences)
    d1: 1-forms → 2-forms  (curl, maps edge values to face circulations)

Key property: d1 ∘ d0 = 0 (curl of gradient is zero)
"""

import numpy as np
from scipy import sparse
from typing import Dict, Tuple


def build_d0(
    edges: np.ndarray,
    num_vertices: int
) -> sparse.csr_matrix:
    """
    Build the d0 (discrete gradient) matrix.
    
    d0 is an (E, V) matrix where:
        d0[e, i] = -1  if vertex i is the source of edge e
        d0[e, j] = +1  if vertex j is the target of edge e
    
    For edge e = (i, j) with i < j (canonical ordering):
        (d0 @ f)[e] = f[j] - f[i]
    
    Args:
        edges: (E, 2) edge vertex indices, each row [i, j] with i < j
        num_vertices: Total number of vertices V
        
    Returns:
        d0: (E, V) sparse CSR matrix
    """
    E = edges.shape[0]
    
    # Two entries per edge: -1 at source, +1 at target
    rows = np.repeat(np.arange(E), 2)
    cols = edges.ravel()
    data = np.tile([-1.0, 1.0], E)
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(E, num_vertices))


def build_d1(
    faces: np.ndarray,
    edges: np.ndarray,
    edge_to_idx: Dict[Tuple[int, int], int] = None
) -> sparse.csr_matrix:
    """
    Build the d1 (discrete curl) matrix.
    
    d1 is an (F, E) matrix where:
        d1[f, e] = +1  if edge e is traversed positively in face f
        d1[f, e] = -1  if edge e is traversed negatively in face f
        d1[f, e] = 0   otherwise
    
    For face f with vertices (i, j, k) in counter-clockwise order:
        (d1 @ ω)[f] = ω[e_ij] + ω[e_jk] + ω[e_ki]  (with appropriate signs)
    
    Args:
        faces: (F, 3) face vertex indices
        edges: (E, 2) edge vertex indices
        edge_to_idx: Optional dict mapping (i, j) tuple to edge index
        
    Returns:
        d1: (F, E) sparse CSR matrix
    """
    F = faces.shape[0]
    E = edges.shape[0]
    
    if edge_to_idx is None:
        edge_to_idx = {tuple(e): idx for idx, e in enumerate(edges)}
    
    rows, cols, data = [], [], []
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        
        # Three edges of face in winding order
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            e_idx = edge_to_idx[edge]
            
            # Sign: +1 if edge direction (a→b) matches canonical (min→max)
            sign = 1.0 if a < b else -1.0
            
            rows.append(f_idx)
            cols.append(e_idx)
            data.append(sign)
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(F, E))


def verify_complex_exactness(
    d0: sparse.csr_matrix,
    d1: sparse.csr_matrix,
    tol: float = 1e-10
) -> bool:
    """
    Verify that d1 @ d0 = 0 (exactness of the discrete de Rham complex).
    
    This is a fundamental property that must hold for any valid DEC discretization.
    
    Args:
        d0: (E, V) gradient matrix
        d1: (F, E) curl matrix
        tol: Numerical tolerance
        
    Returns:
        True if ||d1 @ d0||_F < tol
    """
    d1_d0 = d1 @ d0
    norm = sparse.linalg.norm(d1_d0, 'fro')
    return norm < tol
