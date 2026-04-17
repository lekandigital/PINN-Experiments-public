"""
Triangle Mesh Data Structure for Discrete Exterior Calculus
============================================================

Core mesh representation storing both primal mesh (vertices, faces, edges)
and precomputed DEC infrastructure (Hodge stars, boundary operators, Laplacian).

The mesh topology (connectivity) is separated from geometry (vertex positions)
to support efficient updates for deforming meshes (cloth simulation).
"""

import numpy as np
from scipy import sparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import warnings


@dataclass
class TriangleMesh:
    """
    Triangle mesh with precomputed DEC operators.
    
    Stores both primal mesh (vertices, faces, edges) and precomputed
    DEC infrastructure (dual areas, cotangent weights, operator matrices).
    
    All topology arrays are integer numpy arrays.
    All geometry arrays are float64 numpy arrays.
    
    Attributes:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices, consistently oriented
        edges: (E, 2) edge vertex indices, canonical ordering (i < j)
        
        # Topology (computed once, shared on update_vertices)
        edge_to_idx: Dict mapping (i,j) tuple to edge index
        face_edges: (F, 3) edge indices for each face
        vertex_faces: List of face indices per vertex
        boundary_edges: Indices of boundary edges (1 adjacent face)
        boundary_vertices: Indices of boundary vertices
        
        # Geometry (recomputed when vertices change)
        edge_lengths: (E,) length of each edge
        face_areas: (F,) area of each face
        face_normals: (F, 3) unit face normals
        vertex_normals: (V, 3) area-weighted vertex normals
        
        # DEC dual mesh (recomputed when vertices change)
        dual_areas: (V,) barycentric dual area per vertex
        cotangent_weights: (E,) cotangent weights per edge
        
        # DEC operators (sparse matrices, recomputed when vertices change)
        d0: (E, V) discrete exterior derivative 0→1 (gradient)
        d1: (F, E) discrete exterior derivative 1→2 (curl)
        star0: (V, V) Hodge star on 0-forms (diagonal, dual areas)
        star1: (E, E) Hodge star on 1-forms (diagonal, cotan weights)
        star2: (F, F) Hodge star on 2-forms (diagonal, 1/face_area)
        laplacian: (V, V) cotangent Laplace-Beltrami operator
    """
    # Core mesh data
    vertices: np.ndarray
    faces: np.ndarray
    edges: np.ndarray = field(default=None, repr=False)
    
    # Topology (connectivity)
    edge_to_idx: Dict[Tuple[int, int], int] = field(default=None, repr=False)
    face_edges: np.ndarray = field(default=None, repr=False)
    vertex_faces: List[List[int]] = field(default=None, repr=False)
    vertex_edges: List[List[int]] = field(default=None, repr=False)
    boundary_edges: np.ndarray = field(default=None, repr=False)
    boundary_vertices: np.ndarray = field(default=None, repr=False)
    is_closed: bool = field(default=None, repr=False)
    
    # Geometry
    edge_lengths: np.ndarray = field(default=None, repr=False)
    face_areas: np.ndarray = field(default=None, repr=False)
    face_normals: np.ndarray = field(default=None, repr=False)
    vertex_normals: np.ndarray = field(default=None, repr=False)
    
    # DEC dual mesh
    dual_areas: np.ndarray = field(default=None, repr=False)
    cotangent_weights: np.ndarray = field(default=None, repr=False)
    
    # DEC operators (sparse)
    d0: sparse.csr_matrix = field(default=None, repr=False)
    d1: sparse.csr_matrix = field(default=None, repr=False)
    star0: sparse.dia_matrix = field(default=None, repr=False)
    star1: sparse.dia_matrix = field(default=None, repr=False)
    star2: sparse.dia_matrix = field(default=None, repr=False)
    laplacian: sparse.csr_matrix = field(default=None, repr=False)
    mass_matrix: sparse.dia_matrix = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate and compute derived quantities if not provided."""
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {self.faces.shape}")
    
    @classmethod
    def from_vertices_faces(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        compute_operators: bool = True
    ) -> 'TriangleMesh':
        """
        Construct complete mesh infrastructure from vertices and faces.
        
        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle face indices
            compute_operators: If True, precompute all DEC operators
            
        Returns:
            TriangleMesh with all derived quantities computed
        """
        mesh = cls(vertices=vertices, faces=faces)
        
        # Build topology (doesn't depend on vertex positions)
        mesh._build_topology()
        
        if compute_operators:
            # Compute geometry and operators
            mesh._compute_geometry()
            mesh._compute_dec_operators()
        
        return mesh
    
    def _build_topology(self) -> None:
        """Build edge list and adjacency structures from face connectivity."""
        V = self.vertices.shape[0]
        F = self.faces.shape[0]
        
        # Build edge set from faces
        edge_set = set()
        face_edge_list = []
        edge_faces = {}  # Track which faces each edge belongs to
        
        for f_idx in range(F):
            i, j, k = self.faces[f_idx]
            face_e = []
            for a, b in [(i, j), (j, k), (k, i)]:
                edge = (min(a, b), max(a, b))  # Canonical order
                edge_set.add(edge)
                face_e.append(edge)
                
                # Track face adjacency for boundary detection
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(f_idx)
                
            face_edge_list.append(face_e)
        
        # Create sorted edge list and mapping
        self.edges = np.array(sorted(edge_set), dtype=np.int64)
        self.edge_to_idx = {tuple(e): idx for idx, e in enumerate(self.edges)}
        
        # Convert face edges to indices
        self.face_edges = np.array([
            [self.edge_to_idx[e] for e in fe]
            for fe in face_edge_list
        ], dtype=np.int64)
        
        # Vertex-to-face adjacency
        self.vertex_faces = [[] for _ in range(V)]
        for f_idx in range(F):
            for v_idx in self.faces[f_idx]:
                self.vertex_faces[v_idx].append(f_idx)
        
        # Vertex-to-edge adjacency
        self.vertex_edges = [[] for _ in range(V)]
        for e_idx, (a, b) in enumerate(self.edges):
            self.vertex_edges[a].append(e_idx)
            self.vertex_edges[b].append(e_idx)
        
        # Boundary detection: edges with only 1 adjacent face
        boundary_edge_list = []
        for e_idx, edge in enumerate(self.edges):
            if len(edge_faces[tuple(edge)]) == 1:
                boundary_edge_list.append(e_idx)
        
        self.boundary_edges = np.array(boundary_edge_list, dtype=np.int64)
        
        # Boundary vertices: vertices on boundary edges
        boundary_vertex_set = set()
        for e_idx in self.boundary_edges:
            boundary_vertex_set.add(self.edges[e_idx, 0])
            boundary_vertex_set.add(self.edges[e_idx, 1])
        self.boundary_vertices = np.array(sorted(boundary_vertex_set), dtype=np.int64)
        
        self.is_closed = len(self.boundary_edges) == 0
    
    def _compute_geometry(self) -> None:
        """Compute geometric quantities from vertex positions."""
        V = self.vertices.shape[0]
        F = self.faces.shape[0]
        E = self.edges.shape[0]
        
        # Edge lengths
        self.edge_lengths = np.zeros(E, dtype=np.float64)
        for e_idx, (a, b) in enumerate(self.edges):
            self.edge_lengths[e_idx] = np.linalg.norm(
                self.vertices[b] - self.vertices[a]
            )
        
        # Face areas and normals
        self.face_areas = np.zeros(F, dtype=np.float64)
        self.face_normals = np.zeros((F, 3), dtype=np.float64)
        
        for f_idx in range(F):
            i, j, k = self.faces[f_idx]
            v0, v1, v2 = self.vertices[i], self.vertices[j], self.vertices[k]
            
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(cross)
            
            self.face_areas[f_idx] = area
            if area > 1e-12:
                self.face_normals[f_idx] = cross / (2 * area)
            else:
                self.face_normals[f_idx] = np.array([0., 0., 1.])
        
        # Vertex normals (area-weighted average of adjacent face normals)
        self.vertex_normals = np.zeros((V, 3), dtype=np.float64)
        for v_idx in range(V):
            for f_idx in self.vertex_faces[v_idx]:
                self.vertex_normals[v_idx] += self.face_areas[f_idx] * self.face_normals[f_idx]
            norm = np.linalg.norm(self.vertex_normals[v_idx])
            if norm > 1e-12:
                self.vertex_normals[v_idx] /= norm
            else:
                self.vertex_normals[v_idx] = np.array([0., 0., 1.])
        
        # Dual areas (barycentric: 1/3 of sum of adjacent face areas)
        self.dual_areas = np.zeros(V, dtype=np.float64)
        for f_idx in range(F):
            area_third = self.face_areas[f_idx] / 3.0
            for v_idx in self.faces[f_idx]:
                self.dual_areas[v_idx] += area_third
        
        # Cotangent weights per edge
        self._compute_cotangent_weights()
    
    def _compute_cotangent_weights(self) -> None:
        """
        Compute cotangent weights for each edge.
        
        For interior edge (i,j) shared by triangles with opposite angles α, β:
            w_ij = (cot α + cot β) / 2
        
        For boundary edge with one adjacent triangle and opposite angle α:
            w_ij = cot α / 2
        
        Negative weights (from obtuse triangles) are clamped to small positive value.
        """
        E = self.edges.shape[0]
        self.cotangent_weights = np.zeros(E, dtype=np.float64)
        
        # Build edge-to-faces mapping
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {
            tuple(e): [] for e in self.edges
        }
        for f_idx, face in enumerate(self.faces):
            i, j, k = face
            for a, b in [(i, j), (j, k), (k, i)]:
                edge = (min(a, b), max(a, b))
                edge_to_faces[edge].append(f_idx)
        
        # Compute cotangent weight for each edge
        for e_idx, edge in enumerate(self.edges):
            edge_tuple = tuple(edge)
            adjacent_faces = edge_to_faces[edge_tuple]
            
            total_cotan = 0.0
            for f_idx in adjacent_faces:
                # Find the vertex opposite to this edge in this face
                face = self.faces[f_idx]
                opposite_vertex = None
                for v in face:
                    if v not in edge:
                        opposite_vertex = v
                        break
                
                if opposite_vertex is None:
                    continue
                
                # Get the three vertices
                vi, vj = edge
                vk = opposite_vertex
                
                # Compute cotangent of angle at opposite vertex
                # angle at vk in triangle (vi, vj, vk)
                pi = self.vertices[vi]
                pj = self.vertices[vj]
                pk = self.vertices[vk]
                
                # Vectors from vk to vi and vj
                eki = pi - pk
                ekj = pj - pk
                
                # cot(angle) = (e1 · e2) / |e1 × e2|
                dot = np.dot(eki, ekj)
                cross_norm = np.linalg.norm(np.cross(eki, ekj))
                
                if cross_norm > 1e-12:
                    cotan = dot / cross_norm
                else:
                    cotan = 0.0
                
                total_cotan += cotan
            
            # Average cotangent weight
            self.cotangent_weights[e_idx] = total_cotan / 2.0
        
        # Clamp negative weights with warning
        negative_mask = self.cotangent_weights < 0
        if np.any(negative_mask):
            n_negative = np.sum(negative_mask)
            warnings.warn(
                f"Clamped {n_negative} negative cotangent weights (obtuse triangles). "
                "Consider using intrinsic Delaunay for better accuracy."
            )
            self.cotangent_weights = np.maximum(self.cotangent_weights, 1e-6)
    
    def _compute_dec_operators(self) -> None:
        """Build discrete exterior calculus operator matrices."""
        V = self.vertices.shape[0]
        F = self.faces.shape[0]
        E = self.edges.shape[0]
        
        # d0: Discrete gradient (E × V)
        # (d0 f)[e] = f[j] - f[i] for edge e = (i, j)
        d0_rows, d0_cols, d0_data = [], [], []
        for e_idx, (i, j) in enumerate(self.edges):
            d0_rows.extend([e_idx, e_idx])
            d0_cols.extend([i, j])
            d0_data.extend([-1.0, 1.0])
        
        self.d0 = sparse.csr_matrix(
            (d0_data, (d0_rows, d0_cols)), shape=(E, V)
        )
        
        # d1: Discrete curl (F × E)
        # (d1 ω)[f] = sum of ω on boundary edges with signs
        d1_rows, d1_cols, d1_data = [], [], []
        for f_idx in range(F):
            i, j, k = self.faces[f_idx]
            for a, b in [(i, j), (j, k), (k, i)]:
                edge = (min(a, b), max(a, b))
                e_idx = self.edge_to_idx[edge]
                # Sign: +1 if edge orientation matches face winding
                sign = 1.0 if a < b else -1.0
                d1_rows.append(f_idx)
                d1_cols.append(e_idx)
                d1_data.append(sign)
        
        self.d1 = sparse.csr_matrix(
            (d1_data, (d1_rows, d1_cols)), shape=(F, E)
        )
        
        # Hodge star operators (diagonal matrices)
        # star0: dual areas
        self.star0 = sparse.diags(self.dual_areas, format='dia')
        
        # star1: cotangent weights (ratio of dual to primal edge length)
        self.star1 = sparse.diags(self.cotangent_weights, format='dia')
        
        # star2: 1 / face_area
        star2_data = 1.0 / np.maximum(self.face_areas, 1e-12)
        self.star2 = sparse.diags(star2_data, format='dia')
        
        # Mass matrix (same as star0, but named for clarity)
        self.mass_matrix = self.star0.copy()
        
        # Cotangent Laplacian: L = d0^T @ star1 @ d0
        # This is the weak Laplacian (integrated against test functions)
        self.laplacian = self.d0.T @ self.star1 @ self.d0
        
        # Make symmetric (numerical cleanup)
        self.laplacian = 0.5 * (self.laplacian + self.laplacian.T)
    
    def update_vertices(self, new_vertices: np.ndarray) -> 'TriangleMesh':
        """
        Create new mesh with updated vertex positions, preserving topology.
        
        This is the key method for deforming meshes (cloth simulation) where
        topology is fixed but geometry changes.
        
        Args:
            new_vertices: (V, 3) new vertex positions
            
        Returns:
            New TriangleMesh with updated geometry, shared topology
        """
        new_vertices = np.asarray(new_vertices, dtype=np.float64)
        if new_vertices.shape != self.vertices.shape:
            raise ValueError(
                f"new_vertices shape {new_vertices.shape} doesn't match "
                f"original {self.vertices.shape}"
            )
        
        # Create new mesh sharing topology
        new_mesh = TriangleMesh(
            vertices=new_vertices,
            faces=self.faces,
            edges=self.edges,
            edge_to_idx=self.edge_to_idx,
            face_edges=self.face_edges,
            vertex_faces=self.vertex_faces,
            vertex_edges=self.vertex_edges,
            boundary_edges=self.boundary_edges,
            boundary_vertices=self.boundary_vertices,
            is_closed=self.is_closed,
        )
        
        # Recompute geometry-dependent quantities
        new_mesh._compute_geometry()
        new_mesh._compute_dec_operators()
        
        return new_mesh
    
    def update_vertices_fast(self, new_vertices: np.ndarray) -> 'TriangleMesh':
        """
        Fast vertex update that only recomputes cotangent weights and Laplacian.
        
        Skips recomputing d0, d1, boundary info (topology-dependent).
        Use for high-frequency updates in simulation.
        
        Args:
            new_vertices: (V, 3) new vertex positions
            
        Returns:
            New TriangleMesh with updated Laplacian
        """
        new_vertices = np.asarray(new_vertices, dtype=np.float64)
        if new_vertices.shape != self.vertices.shape:
            raise ValueError(
                f"new_vertices shape {new_vertices.shape} doesn't match "
                f"original {self.vertices.shape}"
            )
        
        # Create new mesh sharing most data
        new_mesh = TriangleMesh(
            vertices=new_vertices,
            faces=self.faces,
            edges=self.edges,
            edge_to_idx=self.edge_to_idx,
            face_edges=self.face_edges,
            vertex_faces=self.vertex_faces,
            vertex_edges=self.vertex_edges,
            boundary_edges=self.boundary_edges,
            boundary_vertices=self.boundary_vertices,
            is_closed=self.is_closed,
            d0=self.d0,  # Topology-dependent, reuse
            d1=self.d1,  # Topology-dependent, reuse
        )
        
        # Recompute geometry
        new_mesh._compute_geometry()
        
        # Recompute operators (star0, star1, star2, Laplacian)
        V = new_mesh.vertices.shape[0]
        F = new_mesh.faces.shape[0]
        E = new_mesh.edges.shape[0]
        
        new_mesh.star0 = sparse.diags(new_mesh.dual_areas, format='dia')
        new_mesh.star1 = sparse.diags(new_mesh.cotangent_weights, format='dia')
        star2_data = 1.0 / np.maximum(new_mesh.face_areas, 1e-12)
        new_mesh.star2 = sparse.diags(star2_data, format='dia')
        new_mesh.mass_matrix = new_mesh.star0.copy()
        
        # Recompute Laplacian
        new_mesh.laplacian = self.d0.T @ new_mesh.star1 @ self.d0
        new_mesh.laplacian = 0.5 * (new_mesh.laplacian + new_mesh.laplacian.T)
        
        return new_mesh
    
    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]
    
    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]
    
    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]
    
    @property
    def total_area(self) -> float:
        return float(np.sum(self.face_areas))
    
    def __repr__(self) -> str:
        return (
            f"TriangleMesh(V={self.num_vertices}, F={self.num_faces}, "
            f"E={self.num_edges}, closed={self.is_closed})"
        )
