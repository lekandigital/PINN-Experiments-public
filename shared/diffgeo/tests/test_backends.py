"""
Tests for backend implementations.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add shared module to path
shared_path = Path(__file__).parents[1]
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from diffgeo.mesh import TriangleMesh
from diffgeo.mesh.generation import icosphere
from diffgeo.backends.numpy_backend import NumPyBackend


class TestNumPyBackend:
    """Test NumPy backend operations."""
    
    @pytest.fixture
    def mesh_and_backend(self):
        mesh = icosphere(subdivisions=2)
        backend = NumPyBackend()
        return mesh, backend
    
    def test_sparse_to_tensor(self, mesh_and_backend):
        mesh, backend = mesh_and_backend
        
        L = backend.sparse_to_tensor(mesh.laplacian)
        
        # Should be CSR matrix
        from scipy.sparse import isspmatrix_csr
        assert isspmatrix_csr(L)
    
    def test_spmv(self, mesh_and_backend):
        mesh, backend = mesh_and_backend
        
        f = np.ones(len(mesh.vertices))
        L = backend.sparse_to_tensor(mesh.laplacian)
        
        Lf = backend.spmv(L, f)
        
        # Laplacian of constant should be ~0
        assert np.allclose(Lf, 0, atol=1e-10)
    
    def test_apply_laplacian(self, mesh_and_backend):
        mesh, backend = mesh_and_backend
        
        # Coordinate function
        f = mesh.vertices[:, 0]  # x coordinate
        
        Lf = backend.apply_laplacian(mesh, f)
        
        # Should be non-zero for non-constant field
        assert not np.allclose(Lf, 0)


class TestTorchBackend:
    """Test PyTorch backend operations."""
    
    @pytest.fixture
    def mesh_and_backend(self):
        pytest.importorskip("torch")
        from diffgeo.backends.torch_backend import TorchBackend
        
        mesh = icosphere(subdivisions=2)
        backend = TorchBackend(device='cpu')
        return mesh, backend
    
    def test_sparse_conversion(self, mesh_and_backend):
        import torch
        mesh, backend = mesh_and_backend
        
        L = backend.sparse_to_tensor(mesh.laplacian)
        
        assert isinstance(L, torch.Tensor)
        assert L.is_sparse
    
    def test_spmv_differentiable(self, mesh_and_backend):
        import torch
        mesh, backend = mesh_and_backend
        
        f = backend.numpy_to_tensor(mesh.vertices[:, 0], requires_grad=True)
        L = backend.sparse_to_tensor(mesh.laplacian)
        
        Lf = backend.spmv(L, f)
        loss = (Lf ** 2).sum()
        loss.backward()
        
        # Gradient should exist
        assert f.grad is not None
        assert not torch.isnan(f.grad).any()
    
    def test_bending_energy(self, mesh_and_backend):
        import torch
        mesh, backend = mesh_and_backend
        
        vertices_tensor = backend.numpy_to_tensor(mesh.vertices)
        
        energy = backend.compute_mean_curvature_energy(mesh, vertices_tensor)
        
        # Sphere has constant curvature, energy should be non-zero
        assert energy.item() > 0


class TestJAXBackend:
    """Test JAX backend operations."""
    
    @pytest.fixture
    def mesh_and_backend(self):
        pytest.importorskip("jax")
        from diffgeo.backends.jax_backend import JAXBackend
        
        mesh = icosphere(subdivisions=2)
        backend = JAXBackend(use_dense_fallback=True)  # Dense for small test
        return mesh, backend
    
    def test_conversion(self, mesh_and_backend):
        import jax.numpy as jnp
        mesh, backend = mesh_and_backend
        
        L = backend.sparse_to_tensor(mesh.laplacian)
        
        # Should be JAX array (dense for small mesh)
        assert isinstance(L, jnp.ndarray)
    
    def test_spmv(self, mesh_and_backend):
        import jax.numpy as jnp
        mesh, backend = mesh_and_backend
        
        f = jnp.array(mesh.vertices[:, 0])
        
        Lf = backend.apply_laplacian(mesh, f)
        
        assert not jnp.allclose(Lf, 0)
    
    def test_differentiable(self, mesh_and_backend):
        import jax
        import jax.numpy as jnp
        mesh, backend = mesh_and_backend
        
        # Cache Laplacian
        L = backend.sparse_to_tensor(mesh.laplacian)
        dual_areas = jnp.array(mesh.dual_areas)
        
        def loss_fn(f):
            Lf = L @ f
            Lf = Lf / (dual_areas + 1e-12)
            return jnp.sum(Lf ** 2)
        
        f = jnp.array(mesh.vertices[:, 0])
        
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(f)
        
        assert not jnp.isnan(grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
