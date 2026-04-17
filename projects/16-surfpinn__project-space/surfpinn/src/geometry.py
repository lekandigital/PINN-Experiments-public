"""
Intrinsic Geometry Module for SurfPINN
======================================

Provides intrinsic (mesh-based) differential operators using the shared
diffgeo module with JAX backend.

Advantages over finite-difference operators:
- Works on unstructured meshes (not just regular grids)
- Handles curved surfaces (e.g., free-surface simulation)
- Automatic boundary handling via DEC
- Spectral accuracy for smooth fields

Usage:
    from geometry import MeshOperators, make_laplacian_loss
    
    # Create operators from mesh
    ops = MeshOperators(vertices, faces)
    
    # Use in physics loss
    def physics_loss(params, x):
        f_pred = model.apply(params, x)
        lap_f = ops.laplacian(f_pred)
        return jnp.mean((lap_f - target) ** 2)
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable
import numpy as np

# Conditional import for shared diffgeo module
try:
    import sys
    from pathlib import Path
    shared_path = Path(__file__).parents[4] / "shared"
    if str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))
    
    from diffgeo.mesh import TriangleMesh
    from diffgeo.backends.jax_backend import JAXBackend, make_laplacian_fn
    DIFFGEO_AVAILABLE = True
except ImportError:
    DIFFGEO_AVAILABLE = False
    TriangleMesh = None
    JAXBackend = None


class MeshOperators:
    """
    Intrinsic differential operators on triangle mesh using DEC.
    
    Provides JAX-compatible operators that can be JIT-compiled and
    differentiated with jax.grad.
    
    Args:
        vertices: Vertex positions [V, 3] as numpy array
        faces: Triangle faces [F, 3] as numpy array
        use_dense: If True, use dense matrices (faster for small meshes)
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        use_dense: bool = True
    ):
        if not DIFFGEO_AVAILABLE:
            raise ImportError(
                "MeshOperators requires the diffgeo module. "
                "Make sure shared/diffgeo is in your Python path."
            )
        
        # Create mesh with precomputed DEC operators
        self.mesh = TriangleMesh.from_vertices_faces(vertices, faces)
        self.backend = JAXBackend(
            use_dense_fallback=use_dense,
            dense_threshold=100_000
        )
        
        # Pre-convert to JAX format
        self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
        self._d0 = self.backend.sparse_to_tensor(self.mesh.d0)
        self._dual_areas = jnp.array(self.mesh.dual_areas)
        
        # Cache for JIT-compiled functions
        self._laplacian_fn = None
    
    def laplacian(self, f: jnp.ndarray, strong_form: bool = True) -> jnp.ndarray:
        """
        Apply Laplace-Beltrami operator.
        
        Args:
            f: Scalar field [V] or [V, C]
            strong_form: If True, normalize by dual areas
            
        Returns:
            Laplacian of f, same shape as input
        """
        Lf = self._laplacian @ f
        
        if strong_form:
            if f.ndim > 1:
                Lf = Lf / (self._dual_areas[:, None] + 1e-12)
            else:
                Lf = Lf / (self._dual_areas + 1e-12)
        
        return Lf
    
    def gradient(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Apply discrete gradient (d0).
        
        Args:
            f: Scalar field [V]
            
        Returns:
            Edge 1-form [E] (gradient projected onto edges)
        """
        if f.ndim > 1:
            f = f.squeeze(-1)
        return self._d0 @ f
    
    def mean_curvature(self, vertices: jnp.ndarray) -> jnp.ndarray:
        """
        Compute mean curvature from vertex positions.
        
        H = ||Δx|| / 2
        
        Args:
            vertices: Vertex positions [V, 3]
            
        Returns:
            Mean curvature [V]
        """
        Lx = self.laplacian(vertices, strong_form=True)
        H = jnp.linalg.norm(Lx, axis=-1) / 2
        return H
    
    def mean_curvature_vector(self, vertices: jnp.ndarray) -> jnp.ndarray:
        """
        Compute mean curvature vector: H_vec = Δx / 2
        
        Points in direction of mean curvature normal.
        
        Args:
            vertices: Vertex positions [V, 3]
            
        Returns:
            Mean curvature vector [V, 3]
        """
        Lx = self.laplacian(vertices, strong_form=True)
        return Lx / 2
    
    def get_jit_laplacian(self) -> Callable:
        """
        Get JIT-compiled Laplacian function.
        
        More efficient for repeated evaluations (e.g., in training loop).
        
        Returns:
            JIT-compiled function: f -> Laplacian(f)
        """
        if self._laplacian_fn is None:
            L = self._laplacian
            dual_areas = self._dual_areas
            
            @jax.jit
            def laplacian_fn(f):
                Lf = L @ f
                if f.ndim > 1:
                    return Lf / (dual_areas[:, None] + 1e-12)
                return Lf / (dual_areas + 1e-12)
            
            self._laplacian_fn = laplacian_fn
        
        return self._laplacian_fn
    
    def update_geometry(self, vertices: np.ndarray):
        """
        Update vertex positions (for deforming surfaces).
        
        Recomputes geometry-dependent quantities.
        """
        self.mesh = self.mesh.update_vertices(vertices)
        self._laplacian = self.backend.sparse_to_tensor(self.mesh.laplacian)
        self._dual_areas = jnp.array(self.mesh.dual_areas)
        self._laplacian_fn = None  # Clear cached JIT function


def make_laplacian_loss(
    mesh_ops: MeshOperators,
    target: jnp.ndarray
) -> Callable:
    """
    Create a loss function that penalizes Laplacian residual.
    
    Useful for Poisson equation: ∇²f = target
    
    Args:
        mesh_ops: MeshOperators instance
        target: Target Laplacian values [V]
        
    Returns:
        Loss function: f -> scalar loss
    """
    laplacian_fn = mesh_ops.get_jit_laplacian()
    
    @jax.jit
    def loss_fn(f: jnp.ndarray) -> jnp.ndarray:
        lap_f = laplacian_fn(f)
        return jnp.mean((lap_f - target) ** 2)
    
    return loss_fn


def make_diffusion_loss(
    mesh_ops: MeshOperators,
    diffusivity: float = 1.0
) -> Callable:
    """
    Create diffusion equation loss: ∂f/∂t = D∇²f
    
    For steady state: ∇²f = 0
    
    Args:
        mesh_ops: MeshOperators instance
        diffusivity: Diffusion coefficient
        
    Returns:
        Loss function for diffusion residual
    """
    laplacian_fn = mesh_ops.get_jit_laplacian()
    
    @jax.jit
    def loss_fn(f: jnp.ndarray, f_prev: jnp.ndarray, dt: float) -> jnp.ndarray:
        # Time derivative
        df_dt = (f - f_prev) / dt
        
        # Diffusion term
        lap_f = laplacian_fn(f)
        diffusion = diffusivity * lap_f
        
        # Residual: ∂f/∂t - D∇²f
        residual = df_dt - diffusion
        
        return jnp.mean(residual ** 2)
    
    return loss_fn


def make_mean_curvature_regularizer(
    mesh_ops: MeshOperators,
    weight: float = 0.01
) -> Callable:
    """
    Create mean curvature regularization loss.
    
    Penalizes high curvature for surface smoothness.
    E_curv = weight * ∫ H² dA
    
    Args:
        mesh_ops: MeshOperators instance
        weight: Regularization weight
        
    Returns:
        Loss function: vertices -> scalar loss
    """
    laplacian_fn = mesh_ops.get_jit_laplacian()
    dual_areas = mesh_ops._dual_areas
    
    @jax.jit
    def loss_fn(vertices: jnp.ndarray) -> jnp.ndarray:
        # Mean curvature vector: H_vec = Δx / 2
        Lx = laplacian_fn(vertices)
        H_vec = Lx / 2
        
        # Mean curvature magnitude
        H_sq = jnp.sum(H_vec ** 2, axis=-1)
        
        # Weighted integral
        energy = jnp.dot(H_sq, dual_areas)
        
        return weight * energy
    
    return loss_fn
