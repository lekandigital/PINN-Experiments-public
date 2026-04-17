"""
Dual Mesh Utilities
===================

Compute dual mesh quantities for discrete exterior calculus:
- Barycentric dual areas (robust for any triangulation)
- Voronoi dual areas (more accurate but can be negative for obtuse triangles)
- Dual edge lengths
"""

import numpy as np
from typing import Tuple, Dict, List


def compute_dual_areas(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = 'barycentric'
) -> np.ndarray:
    """
    Compute dual cell areas for each vertex.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        method: 'barycentric' (1/3 of adjacent face areas, always positive)
                'voronoi' (circumcentric dual, more accurate but can be negative)
                'mixed' (voronoi for acute, barycentric for obtuse triangles)
                
    Returns:
        dual_areas: (V,) dual cell area per vertex
    """
    V = vertices.shape[0]
    F = faces.shape[0]
    dual_areas = np.zeros(V, dtype=np.float64)
    
    if method == 'barycentric':
        # Simple: each vertex gets 1/3 of each adjacent face's area
        for f_idx in range(F):
            i, j, k = faces[f_idx]
            v0, v1, v2 = vertices[i], vertices[j], vertices[k]
            
            # Face area via cross product
            e1 = v1 - v0
            e2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            
            dual_areas[i] += area / 3
            dual_areas[j] += area / 3
            dual_areas[k] += area / 3
    
    elif method == 'voronoi':
        # Voronoi dual: connect circumcenters to edge midpoints
        for f_idx in range(F):
            i, j, k = faces[f_idx]
            v0, v1, v2 = vertices[i], vertices[j], vertices[k]
            
            # Compute circumcenter
            cc = _circumcenter(v0, v1, v2)
            
            # Edge midpoints
            m01 = (v0 + v1) / 2
            m12 = (v1 + v2) / 2
            m20 = (v2 + v0) / 2
            
            # Dual cell area contributions
            # For vertex i: quadrilateral (m01, cc, m20)
            # Area = 0.5 * |(m01-cc) x (m20-cc)|
            dual_areas[i] += 0.5 * np.linalg.norm(np.cross(m01 - cc, m20 - cc))
            dual_areas[j] += 0.5 * np.linalg.norm(np.cross(m12 - cc, m01 - cc))
            dual_areas[k] += 0.5 * np.linalg.norm(np.cross(m20 - cc, m12 - cc))
    
    elif method == 'mixed':
        # Meyer et al. "Discrete Differential-Geometry Operators"
        # Use Voronoi for non-obtuse triangles, barycentric for obtuse
        for f_idx in range(F):
            i, j, k = faces[f_idx]
            v0, v1, v2 = vertices[i], vertices[j], vertices[k]
            
            # Check for obtuse angles
            e01 = v1 - v0
            e12 = v2 - v1
            e20 = v0 - v2
            
            dot0 = np.dot(-e20, e01)  # angle at v0
            dot1 = np.dot(-e01, e12)  # angle at v1
            dot2 = np.dot(-e12, e20)  # angle at v2
            
            area = 0.5 * np.linalg.norm(np.cross(e01, -e20))
            
            if dot0 < 0 or dot1 < 0 or dot2 < 0:
                # Obtuse triangle: use barycentric
                dual_areas[i] += area / 3
                dual_areas[j] += area / 3
                dual_areas[k] += area / 3
            else:
                # Non-obtuse: use Voronoi (cotangent formula)
                # A_i = (1/8) * (|e01|² cot(angle at k) + |e20|² cot(angle at j))
                len01_sq = np.dot(e01, e01)
                len12_sq = np.dot(e12, e12)
                len20_sq = np.dot(e20, e20)
                
                # Cotangents from dot products
                cross_norm = 2 * area
                if cross_norm > 1e-12:
                    cot0 = dot0 / cross_norm
                    cot1 = dot1 / cross_norm
                    cot2 = dot2 / cross_norm
                else:
                    cot0 = cot1 = cot2 = 0
                
                dual_areas[i] += (len01_sq * cot2 + len20_sq * cot1) / 8
                dual_areas[j] += (len12_sq * cot0 + len01_sq * cot2) / 8
                dual_areas[k] += (len20_sq * cot1 + len12_sq * cot0) / 8
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure positive (numerical safety)
    dual_areas = np.maximum(dual_areas, 1e-12)
    
    return dual_areas


def _circumcenter(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute circumcenter of triangle (v0, v1, v2)."""
    # Use barycentric coordinate formula
    a = v1 - v0
    b = v2 - v0
    
    a_dot_a = np.dot(a, a)
    b_dot_b = np.dot(b, b)
    a_dot_b = np.dot(a, b)
    
    denom = 2 * (a_dot_a * b_dot_b - a_dot_b * a_dot_b)
    
    if abs(denom) < 1e-12:
        # Degenerate triangle, return centroid
        return (v0 + v1 + v2) / 3
    
    s = (a_dot_a * b_dot_b - b_dot_b * a_dot_b) / denom
    t = (b_dot_b * a_dot_a - a_dot_a * a_dot_b) / denom
    
    return v0 + s * a + t * b


def compute_dual_edge_lengths(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]]
) -> np.ndarray:
    """
    Compute dual edge lengths for Hodge star on 1-forms.
    
    For interior edges: distance between circumcenters of adjacent faces.
    For boundary edges: distance from circumcenter to edge midpoint.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        edges: (E, 2) edge vertex indices
        edge_to_faces: Dict mapping edge tuple to list of adjacent face indices
        
    Returns:
        dual_lengths: (E,) dual edge length per edge
    """
    E = edges.shape[0]
    F = faces.shape[0]
    
    # Precompute face circumcenters
    circumcenters = np.zeros((F, 3), dtype=np.float64)
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        circumcenters[f_idx] = _circumcenter(vertices[i], vertices[j], vertices[k])
    
    # Compute dual edge lengths
    dual_lengths = np.zeros(E, dtype=np.float64)
    
    for e_idx in range(E):
        edge_tuple = tuple(edges[e_idx])
        adjacent_faces = edge_to_faces.get(edge_tuple, [])
        
        if len(adjacent_faces) == 2:
            # Interior edge: distance between circumcenters
            f1, f2 = adjacent_faces
            dual_lengths[e_idx] = np.linalg.norm(
                circumcenters[f1] - circumcenters[f2]
            )
        elif len(adjacent_faces) == 1:
            # Boundary edge: distance from circumcenter to midpoint
            f1 = adjacent_faces[0]
            midpoint = (vertices[edges[e_idx, 0]] + vertices[edges[e_idx, 1]]) / 2
            dual_lengths[e_idx] = np.linalg.norm(circumcenters[f1] - midpoint)
        else:
            # Edge not in any face (shouldn't happen for valid mesh)
            dual_lengths[e_idx] = 0
    
    # Ensure positive
    dual_lengths = np.maximum(dual_lengths, 1e-12)
    
    return dual_lengths
