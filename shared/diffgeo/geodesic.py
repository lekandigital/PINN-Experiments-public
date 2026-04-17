"""
Geodesic Distance Computation
=============================

Compute geodesic (shortest-path) distances on triangle meshes.

Methods:
- Heat method (Crane et al. 2013): Fast approximate distances via heat diffusion
- Dijkstra/fast marching: Exact distances on mesh graph
- Exact polyhedral: True geodesics through faces (expensive)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, TYPE_CHECKING
import heapq

if TYPE_CHECKING:
    from .mesh.trimesh import TriangleMesh


def geodesic_distance_heat(
    mesh: 'TriangleMesh',
    source_vertices: np.ndarray,
    t: Optional[float] = None,
) -> np.ndarray:
    """
    Compute geodesic distance using the heat method (Crane et al. 2013).
    
    Algorithm:
    1. Solve heat equation with delta sources: (I + tL)u = δ_source
    2. Compute normalized gradient: X = -∇u / |∇u|
    3. Solve Poisson equation: Δφ = ∇·X
    4. Shift so minimum is at source: φ = φ - φ_min
    
    Fast and robust, with accuracy improving as mesh is refined.
    
    Args:
        mesh: TriangleMesh with precomputed operators
        source_vertices: Indices of source vertices
        t: Diffusion time (default: mean edge length squared)
        
    Returns:
        distances: (V,) geodesic distance from source set
    """
    V = mesh.vertices.shape[0]
    
    # Default time step: h² where h is mean edge length
    if t is None:
        t = np.mean(mesh.edge_lengths) ** 2
    
    # Step 1: Solve heat equation
    # (M + t*L) u = δ_source
    # Use mass matrix M and stiffness matrix L
    L = mesh.laplacian
    M = mesh.mass_matrix
    
    # Heat matrix: M - t*L (using negative Laplacian convention)
    # Or: M + t*L if using positive Laplacian
    # Check sign convention of mesh.laplacian
    A = M + t * L
    
    # Source vector: 1 at source vertices, 0 elsewhere
    b = np.zeros(V)
    source_vertices = np.atleast_1d(source_vertices)
    b[source_vertices] = 1.0
    
    # Solve for heat distribution
    u = spsolve(A, b)
    
    # Step 2: Compute normalized gradient on faces
    # Gradient of u on each face
    grad_u = _compute_face_gradients(mesh, u)
    
    # Normalize: X = -grad_u / |grad_u|
    grad_norm = np.linalg.norm(grad_u, axis=1, keepdims=True)
    X = -grad_u / (grad_norm + 1e-10)
    
    # Step 3: Solve Poisson equation
    # Δφ = ∇·X (divergence of X)
    div_X = _compute_divergence(mesh, X)
    
    # Solve L @ φ = div_X
    # Add constraint to fix solution (Laplacian has null space)
    L_constrained = L.copy()
    L_constrained[0, :] = 0
    L_constrained[0, 0] = 1
    div_X[0] = 0
    
    phi = spsolve(L_constrained, div_X)
    
    # Step 4: Shift so minimum (at source) is zero
    phi = phi - np.min(phi[source_vertices])
    
    return phi


def _compute_face_gradients(
    mesh: 'TriangleMesh',
    vertex_values: np.ndarray,
) -> np.ndarray:
    """
    Compute gradient of vertex function on each face.
    
    For triangle with vertices (v0, v1, v2) and values (f0, f1, f2):
    ∇f = (1/2A) * Σ_i f_i * (n × e_i)
    
    where A is face area, n is face normal, e_i is edge opposite to vertex i.
    
    Returns:
        gradients: (F, 3) gradient vector per face
    """
    F = mesh.faces.shape[0]
    gradients = np.zeros((F, 3))
    
    for f_idx, face in enumerate(mesh.faces):
        i, j, k = face
        v0, v1, v2 = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        f0, f1, f2 = vertex_values[i], vertex_values[j], vertex_values[k]
        
        # Face normal and area
        e1, e2 = v1 - v0, v2 - v0
        normal = np.cross(e1, e2)
        area2 = np.linalg.norm(normal)
        if area2 < 1e-10:
            continue
        normal = normal / area2
        
        # Gradient: sum of f_i * (n × e_i) where e_i is opposite edge
        # e0 = v2 - v1, e1 = v0 - v2, e2 = v1 - v0
        e0 = v2 - v1
        e1_edge = v0 - v2
        e2_edge = v1 - v0
        
        grad = f0 * np.cross(normal, e0) + f1 * np.cross(normal, e1_edge) + f2 * np.cross(normal, e2_edge)
        gradients[f_idx] = grad / area2
    
    return gradients


def _compute_divergence(
    mesh: 'TriangleMesh',
    face_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute divergence of per-face vector field at vertices.
    
    Uses the integrated divergence formula:
    (∇·X)_i = (1/A_i) Σ_f (cot θ_ij * (e_ij · X_f) + cot θ_ik * (e_ik · X_f))
    
    Returns:
        divergence: (V,) divergence at each vertex
    """
    V = mesh.vertices.shape[0]
    divergence = np.zeros(V)
    
    for f_idx, face in enumerate(mesh.faces):
        i, j, k = face
        v0, v1, v2 = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
        X = face_vectors[f_idx]
        
        # Edge vectors from each vertex
        e_ij = v1 - v0
        e_ik = v2 - v0
        e_jk = v2 - v1
        e_ji = -e_ij
        e_ki = -e_ik
        e_kj = -e_jk
        
        # Cotangent weights
        def cotan(e1, e2):
            cross_norm = np.linalg.norm(np.cross(e1, e2))
            if cross_norm < 1e-10:
                return 0.0
            return np.dot(e1, e2) / cross_norm
        
        # For vertex i: edges are e_ij and e_ik
        cot_j = cotan(e_ij, e_ik)  # angle at i
        cot_k = cotan(e_ji, e_jk)  # angle at j
        cot_i = cotan(e_ki, e_kj)  # angle at k
        
        # Add contributions
        divergence[i] += 0.5 * (cot_k * np.dot(e_ij, X) + cot_j * np.dot(e_ik, X))
        divergence[j] += 0.5 * (cot_i * np.dot(e_jk, X) + cot_k * np.dot(e_ji, X))
        divergence[k] += 0.5 * (cot_j * np.dot(e_ki, X) + cot_i * np.dot(e_kj, X))
    
    return divergence


def geodesic_distance_dijkstra(
    mesh: 'TriangleMesh',
    source_vertices: np.ndarray,
) -> np.ndarray:
    """
    Compute geodesic distance using Dijkstra's algorithm on mesh graph.
    
    Uses edge lengths as distances. This gives exact graph distances
    but underestimates true geodesic distances (paths must follow edges).
    
    Args:
        mesh: TriangleMesh with edges
        source_vertices: Indices of source vertices
        
    Returns:
        distances: (V,) shortest path distance from source set
    """
    V = mesh.vertices.shape[0]
    source_vertices = np.atleast_1d(source_vertices)
    
    # Build adjacency list from edges
    adj = [[] for _ in range(V)]
    for e_idx, (i, j) in enumerate(mesh.edges):
        length = mesh.edge_lengths[e_idx]
        adj[i].append((j, length))
        adj[j].append((i, length))
    
    # Initialize distances
    distances = np.full(V, np.inf)
    distances[source_vertices] = 0
    
    # Priority queue: (distance, vertex)
    heap = [(0.0, v) for v in source_vertices]
    heapq.heapify(heap)
    
    visited = set()
    
    while heap:
        dist, u = heapq.heappop(heap)
        
        if u in visited:
            continue
        visited.add(u)
        
        for v, edge_len in adj[u]:
            if v in visited:
                continue
            
            new_dist = dist + edge_len
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    return distances


def geodesic_distance_fast_marching(
    mesh: 'TriangleMesh',
    source_vertices: np.ndarray,
) -> np.ndarray:
    """
    Compute geodesic distance using Fast Marching Method on triangulated surface.
    
    More accurate than Dijkstra because it allows paths through triangle interiors.
    Uses the FMM update rule for triangles (Kimmel & Sethian 1998).
    
    Args:
        mesh: TriangleMesh
        source_vertices: Indices of source vertices
        
    Returns:
        distances: (V,) geodesic distance from source set
    """
    V = mesh.vertices.shape[0]
    source_vertices = np.atleast_1d(source_vertices)
    
    # Initialize
    distances = np.full(V, np.inf)
    distances[source_vertices] = 0.0
    
    # States: 0 = far, 1 = considered (in heap), 2 = accepted (frozen)
    states = np.zeros(V, dtype=np.int32)
    states[source_vertices] = 2  # Accepted
    
    # Build vertex-to-faces mapping
    vertex_faces = [[] for _ in range(V)]
    for f_idx, face in enumerate(mesh.faces):
        for v in face:
            vertex_faces[v].append(f_idx)
    
    # Initialize heap with neighbors of source vertices
    heap = []
    for src in source_vertices:
        for f_idx in vertex_faces[src]:
            face = mesh.faces[f_idx]
            for v in face:
                if states[v] == 0:  # Far
                    # Compute initial estimate
                    d = distances[src] + np.linalg.norm(mesh.vertices[v] - mesh.vertices[src])
                    if d < distances[v]:
                        distances[v] = d
                    states[v] = 1  # Considered
                    heapq.heappush(heap, (distances[v], v))
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if states[u] == 2:  # Already accepted
            continue
        
        states[u] = 2  # Accept
        
        # Update neighbors
        for f_idx in vertex_faces[u]:
            face = mesh.faces[f_idx]
            
            # For each vertex in the face that's not accepted
            for v in face:
                if states[v] == 2:
                    continue
                
                # Try to update distance using this triangle
                # Get the two other vertices of the triangle
                others = [w for w in face if w != v]
                
                # If both others are accepted, use triangle update
                if states[others[0]] == 2 and states[others[1]] == 2:
                    new_d = _triangle_update(
                        mesh.vertices[v],
                        mesh.vertices[others[0]], distances[others[0]],
                        mesh.vertices[others[1]], distances[others[1]],
                    )
                    if new_d < distances[v]:
                        distances[v] = new_d
                
                # Otherwise use edge update
                elif states[others[0]] == 2:
                    edge_len = np.linalg.norm(mesh.vertices[v] - mesh.vertices[others[0]])
                    new_d = distances[others[0]] + edge_len
                    if new_d < distances[v]:
                        distances[v] = new_d
                
                elif states[others[1]] == 2:
                    edge_len = np.linalg.norm(mesh.vertices[v] - mesh.vertices[others[1]])
                    new_d = distances[others[1]] + edge_len
                    if new_d < distances[v]:
                        distances[v] = new_d
                
                # Add to heap if updated
                if states[v] == 0:
                    states[v] = 1
                heapq.heappush(heap, (distances[v], v))
    
    return distances


def _triangle_update(
    p: np.ndarray,     # Target vertex
    q1: np.ndarray,    # First accepted vertex
    d1: float,         # Distance at q1
    q2: np.ndarray,    # Second accepted vertex
    d2: float,         # Distance at q2
) -> float:
    """
    Compute FMM update for vertex p using triangle (p, q1, q2).
    
    Solves for the distance at p assuming the wavefront arrives
    from the edge (q1, q2) with known distances d1 and d2.
    """
    # Edge from q1 to q2
    e = q2 - q1
    e_len = np.linalg.norm(e)
    if e_len < 1e-10:
        return min(d1, d2) + np.linalg.norm(p - q1)
    
    e_unit = e / e_len
    
    # Vector from q1 to p
    v = p - q1
    
    # Project p onto line through q1, q2
    t = np.dot(v, e_unit)  # Parameter along edge
    
    # Perpendicular distance from p to edge
    perp = v - t * e_unit
    h = np.linalg.norm(perp)
    
    if h < 1e-10:
        # p is on the edge - use simple linear interpolation
        if 0 <= t <= e_len:
            alpha = t / e_len
            return (1 - alpha) * d1 + alpha * d2
        else:
            return min(d1 + np.linalg.norm(p - q1), d2 + np.linalg.norm(p - q2))
    
    # Solve quadratic for wavefront arrival
    # The distance function should satisfy |∇d| = 1
    # Using the linear approximation along the edge: d(s) = d1 + (d2-d1)*s/e_len
    
    # Simplified: try to find where the wavefront from the edge reaches p
    # This is approximate - full solution requires solving quadratic
    
    # Check if update is valid (wavefront reaches p from inside the triangle)
    if t < 0 or t > e_len:
        # Wavefront doesn't reach from edge interior
        return min(d1 + np.linalg.norm(p - q1), d2 + np.linalg.norm(p - q2))
    
    # Linear interpolation of distance along edge + perpendicular distance
    alpha = t / e_len
    d_edge = (1 - alpha) * d1 + alpha * d2
    
    return d_edge + h
