"""
Discrete Exterior Calculus (DEC) Operators

Builds discrete differential operators on triangle meshes:
- B0 (gradient): E × V incidence matrix mapping 0-forms (vertices) → 1-forms (edges)  
- B1 (curl): F × E incidence matrix mapping 1-forms (edges) → 2-forms (faces)
- Hodge star operators for 0-forms, 1-forms, 2-forms

The discrete Laplacian is constructed as:
    Δ₀ = B₀ᵀ · Hodge₁⁻¹ · B₀ · Hodge₀

References:
- Desbrun et al., "Discrete Differential Forms for Computational Modeling"
- Crane et al., "Digital Geometry Processing with Discrete Exterior Calculus"
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix, diags
from typing import Dict, Tuple, Optional


def compute_face_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute area of triangle with vertices v0, v1, v2."""
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    return 0.5 * np.linalg.norm(cross)


def compute_edge_length(v0: np.ndarray, v1: np.ndarray) -> float:
    """Compute length of edge from v0 to v1."""
    return np.linalg.norm(v1 - v0)


def build_dec_operators(
    vertices: np.ndarray,
    faces: np.ndarray,
    return_tensors: bool = True,
    device: str = 'cpu'
) -> Dict:
    """
    Build DEC operators from a triangle mesh.
    
    Args:
        vertices: [N, 3] vertex positions
        faces: [F, 3] triangle face indices
        return_tensors: if True, return PyTorch tensors; else numpy/scipy
        device: PyTorch device for tensors
        
    Returns:
        dict with keys:
            - 'B0': Gradient operator (E × V)
            - 'B1': Curl operator (F × E)
            - 'Hodge0': 0-form Hodge star (V × V diagonal)
            - 'Hodge1': 1-form Hodge star (E × E diagonal)
            - 'Hodge2': 2-form Hodge star (F × F diagonal)
            - 'edge_to_idx': dict mapping (i,j) -> edge index
            - 'edges': list of (i, j) tuples
    """
    N = vertices.shape[0]
    F = faces.shape[0]
    
    # =====================
    # Step 1: Build edge list and mapping
    # =====================
    edge_set = set()
    face_edges = []  # For each face, list of 3 edge indices
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        # Edges of face (ordered consistently)
        face_e = []
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            edge_set.add(edge)
            face_e.append(edge)
        face_edges.append(face_e)
    
    edges = sorted(list(edge_set))
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    E = len(edges)
    
    # =====================
    # Step 2: Build B0 (gradient) - E × V
    # =====================
    # For edge e = (a, b) with a < b:
    # B0[e, a] = -1, B0[e, b] = +1
    # Represents gradient: (B0 @ f)[e] = f[b] - f[a]
    
    B0_rows, B0_cols, B0_data = [], [], []
    
    for e_idx, (a, b) in enumerate(edges):
        B0_rows.extend([e_idx, e_idx])
        B0_cols.extend([a, b])
        B0_data.extend([-1.0, 1.0])
    
    B0_sparse = csr_matrix((B0_data, (B0_rows, B0_cols)), shape=(E, N))
    
    # =====================
    # Step 3: Build B1 (curl) - F × E
    # =====================
    # For face f with edges e1, e2, e3 (in order):
    # B1[f, e] = ±1 based on edge orientation matching face winding
    
    B1_rows, B1_cols, B1_data = [], [], []
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        
        # Face edges in order: (i,j), (j,k), (k,i)
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            e_idx = edge_to_idx[edge]
            
            # Sign: +1 if edge direction matches (a < b matches face winding)
            # -1 if edge is reversed relative to face
            sign = 1.0 if a < b else -1.0
            
            B1_rows.append(f_idx)
            B1_cols.append(e_idx)
            B1_data.append(sign)
    
    B1_sparse = csr_matrix((B1_data, (B1_rows, B1_cols)), shape=(F, E))
    
    # =====================
    # Step 4: Build Hodge0 (vertex areas)
    # =====================
    # Hodge0[i] = barycentric dual cell area = (1/3) * sum of adjacent face areas
    
    vertex_areas = np.zeros(N)
    face_areas = np.zeros(F)
    
    for f_idx in range(F):
        i, j, k = faces[f_idx]
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        area = compute_face_area(v0, v1, v2)
        face_areas[f_idx] = area
        
        # Each vertex gets 1/3 of face area
        vertex_areas[i] += area / 3.0
        vertex_areas[j] += area / 3.0
        vertex_areas[k] += area / 3.0
    
    Hodge0_diag = vertex_areas
    
    # =====================
    # Step 5: Build Hodge1 (edge lengths / 2)
    # =====================
    # Hodge1[e] = edge_length / 2 (simplified; full DEC uses dual edge length)
    
    edge_lengths = np.zeros(E)
    
    for e_idx, (a, b) in enumerate(edges):
        edge_lengths[e_idx] = compute_edge_length(vertices[a], vertices[b])
    
    Hodge1_diag = edge_lengths / 2.0
    
    # =====================
    # Step 6: Build Hodge2 (1 / face area)
    # =====================
    Hodge2_diag = 1.0 / (face_areas + 1e-10)
    
    # =====================
    # Convert to PyTorch tensors if requested
    # =====================
    if return_tensors:
        def sparse_to_dense_tensor(sp_matrix, device):
            dense = sp_matrix.toarray()
            return torch.tensor(dense, dtype=torch.float32, device=device)
        
        result = {
            'B0': sparse_to_dense_tensor(B0_sparse, device),
            'B1': sparse_to_dense_tensor(B1_sparse, device),
            'Hodge0': torch.tensor(Hodge0_diag, dtype=torch.float32, device=device),
            'Hodge1': torch.tensor(Hodge1_diag, dtype=torch.float32, device=device),
            'Hodge2': torch.tensor(Hodge2_diag, dtype=torch.float32, device=device),
            'edge_to_idx': edge_to_idx,
            'edges': edges,
            'face_areas': torch.tensor(face_areas, dtype=torch.float32, device=device),
        }
    else:
        result = {
            'B0': B0_sparse,
            'B1': B1_sparse,
            'Hodge0': Hodge0_diag,
            'Hodge1': Hodge1_diag,
            'Hodge2': Hodge2_diag,
            'edge_to_idx': edge_to_idx,
            'edges': edges,
            'face_areas': face_areas,
        }
    
    return result


class DECLaplacian(torch.nn.Module):
    """
    Discrete Laplace-Beltrami operator using DEC.
    
    Computes: Δ₀ = Hodge₀⁻¹ · B₀ᵀ · Hodge₁ · B₀
    
    This is the cotan Laplacian divided by vertex areas (mass-normalized).
    
    Args:
        vertices: [N, 3] mesh vertices (numpy)
        faces: [F, 3] mesh faces (numpy)
        normalized: if True, apply mass matrix normalization
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normalized: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        
        # Build DEC operators
        ops = build_dec_operators(vertices, faces, return_tensors=True, device=device)
        
        self.register_buffer('B0', ops['B0'])
        self.register_buffer('Hodge0', ops['Hodge0'])
        self.register_buffer('Hodge1', ops['Hodge1'])
        
        self.normalized = normalized
        
        # Precompute Laplacian matrix
        # L = B0^T @ diag(Hodge1) @ B0
        Hodge1_mat = torch.diag(ops['Hodge1'])
        L = ops['B0'].T @ Hodge1_mat @ ops['B0']
        
        if normalized:
            # Mass-normalized Laplacian: M^(-1) @ L
            Hodge0_inv = 1.0 / (ops['Hodge0'] + 1e-10)
            L = torch.diag(Hodge0_inv) @ L
        
        self.register_buffer('L', L)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Apply Laplacian to vertex function.
        
        Args:
            f: [N] or [N, C] scalar or vector field on vertices
            
        Returns:
            Lf: [N] or [N, C] Laplacian of f
        """
        if f.dim() == 1:
            return self.L @ f
        else:
            return (self.L @ f.T).T


def build_graph_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Build the graph (combinatorial) Laplacian from mesh connectivity.
    
    L = D - A where D is degree matrix and A is adjacency.
    
    This is simpler than the cotangent Laplacian but doesn't encode geometry.
    
    Args:
        vertices: [N, 3] vertex positions (only used for sizing)
        faces: [F, 3] face indices
        device: PyTorch device
        
    Returns:
        L: [N, N] graph Laplacian matrix
    """
    N = vertices.shape[0]
    
    # Build adjacency from faces
    adjacency = np.zeros((N, N))
    
    for f in faces:
        i, j, k = f
        adjacency[i, j] = adjacency[j, i] = 1
        adjacency[j, k] = adjacency[k, j] = 1
        adjacency[k, i] = adjacency[i, k] = 1
    
    # Degree matrix
    degree = np.sum(adjacency, axis=1)
    
    # Laplacian L = D - A
    L = np.diag(degree) - adjacency
    
    return torch.tensor(L, dtype=torch.float32, device=device)
