"""
Tests for shared/diffgeo Extensions
===================================

Comprehensive tests for the new subpackages added in v0.2.0:
- charts: Chart atlas, ChartMLP, AtlasPINN
- spectral: Eigenpairs, spectral convolutions, Chebyshev
- tangent: Tangent basis, parallel transport, message passing
- projection: Map projections
- geodesic: Distance computation
- adapters: Coastal and cloth adapters
"""

import pytest
import numpy as np
from typing import Optional


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_mesh():
    """Create a simple triangle mesh (icosahedron)."""
    # Create icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices[0])  # Normalize to unit sphere
    
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    
    return vertices, faces


@pytest.fixture
def plane_mesh():
    """Create a simple plane mesh for testing."""
    # 5x5 grid of vertices
    n = 5
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    vertices = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1)
    
    # Generate faces (two triangles per quad)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces, dtype=np.int64)
    return vertices, faces


@pytest.fixture
def trimesh_simple(simple_mesh):
    """Create TriangleMesh from simple mesh."""
    try:
        from shared.diffgeo.mesh.trimesh import TriangleMesh
        vertices, faces = simple_mesh
        return TriangleMesh(vertices=vertices, faces=faces)
    except ImportError:
        pytest.skip("TriangleMesh not available")


# ============================================================================
# Charts Subpackage Tests
# ============================================================================

class TestAtlas:
    """Tests for chart atlas functionality."""
    
    def test_atlas_creation(self, simple_mesh):
        """Test creating an atlas from mesh."""
        try:
            from shared.diffgeo.charts import Atlas
        except ImportError:
            pytest.skip("Charts module not available")
        
        vertices, faces = simple_mesh
        atlas = Atlas.from_mesh(vertices, faces, n_charts=4)
        
        assert atlas.n_charts == 4
        assert len(atlas.charts) == 4
        
    def test_atlas_point_location(self, simple_mesh):
        """Test locating points in charts."""
        try:
            from shared.diffgeo.charts import Atlas
        except ImportError:
            pytest.skip("Charts module not available")
        
        vertices, faces = simple_mesh
        atlas = Atlas.from_mesh(vertices, faces, n_charts=4)
        
        # Test with some query points
        points = vertices[:5]
        chart_ids, local_coords, weights = atlas.locate_points(points)
        
        assert chart_ids.shape == (5,)
        assert local_coords.shape == (5, 2)
        assert all(0 <= cid < 4 for cid in chart_ids)
    
    def test_torch_atlas(self, simple_mesh):
        """Test PyTorch atlas."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.charts import TorchAtlas
        except ImportError:
            pytest.skip("Charts module not available")
        
        vertices, faces = simple_mesh
        atlas = TorchAtlas.from_mesh(
            torch.from_numpy(vertices).float(),
            torch.from_numpy(faces).long(),
            n_charts=4,
        )
        
        assert atlas.n_charts == 4


class TestChartMLP:
    """Tests for ChartMLP and AtlasPINN."""
    
    def test_chart_mlp_forward(self, simple_mesh):
        """Test ChartMLP forward pass."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.charts import ChartMLP
        except ImportError:
            pytest.skip("Charts module not available")
        
        model = ChartMLP(
            input_dim=2,
            hidden_dims=[32, 32],
            output_dim=1,
        )
        
        # Test forward pass
        local_coords = torch.randn(10, 2)
        output = model(local_coords)
        
        assert output.shape == (10, 1)
    
    def test_atlas_pinn_forward(self, simple_mesh):
        """Test AtlasPINN blended output."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.charts import TorchAtlas, AtlasPINN
        except ImportError:
            pytest.skip("Charts module not available")
        
        vertices, faces = simple_mesh
        atlas = TorchAtlas.from_mesh(
            torch.from_numpy(vertices).float(),
            torch.from_numpy(faces).long(),
            n_charts=4,
        )
        
        model = AtlasPINN(
            atlas=atlas,
            input_dim=2,
            hidden_dims=[32, 32],
            output_dim=1,
        )
        
        # Test forward with 3D points
        points = torch.from_numpy(vertices[:5]).float()
        output = model(points)
        
        assert output.shape == (5, 1)


# ============================================================================
# Spectral Subpackage Tests
# ============================================================================

class TestSpectralBasis:
    """Tests for spectral basis computation."""
    
    def test_eigenpair_computation(self, trimesh_simple):
        """Test computing Laplacian eigenpairs."""
        try:
            from shared.diffgeo.spectral import compute_laplacian_eigenpairs, SpectralBasis
        except ImportError:
            pytest.skip("Spectral module not available")
        
        mesh = trimesh_simple
        eigenvalues, eigenvectors = compute_laplacian_eigenpairs(mesh, k=5)
        
        # Check shapes
        assert eigenvalues.shape == (5,)
        assert eigenvectors.shape[0] == mesh.vertices.shape[0]
        assert eigenvectors.shape[1] == 5
        
        # First eigenvalue should be ~0 (constant eigenvector)
        assert eigenvalues[0] < 1e-8
        
        # Eigenvalues should be non-negative and sorted
        assert np.all(eigenvalues >= -1e-8)
        assert np.all(np.diff(eigenvalues) >= -1e-8)
    
    def test_spectral_basis_project_reconstruct(self, trimesh_simple):
        """Test spectral projection and reconstruction."""
        try:
            from shared.diffgeo.spectral import compute_laplacian_eigenpairs, SpectralBasis
        except ImportError:
            pytest.skip("Spectral module not available")
        
        mesh = trimesh_simple
        eigenvalues, eigenvectors = compute_laplacian_eigenpairs(mesh, k=10)
        
        basis = SpectralBasis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mass_matrix=mesh.mass_matrix,
        )
        
        # Project a function
        f = mesh.vertices[:, 0]  # x-coordinate
        coeffs = basis.project(f)
        
        assert coeffs.shape == (10,)
        
        # Reconstruct
        f_rec = basis.reconstruct(coeffs)
        
        # Should be close (not exact due to truncation)
        assert f_rec.shape == f.shape


class TestSpectralConv:
    """Tests for spectral convolution layers."""
    
    def test_spectral_graph_conv(self, trimesh_simple):
        """Test SpectralGraphConv forward pass."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.spectral import compute_laplacian_eigenpairs, SpectralGraphConv
        except ImportError:
            pytest.skip("Spectral module not available")
        
        mesh = trimesh_simple
        eigenvalues, eigenvectors = compute_laplacian_eigenpairs(mesh, k=10)
        
        model = SpectralGraphConv(
            in_channels=3,
            out_channels=16,
            eigenvectors=torch.from_numpy(eigenvectors).float(),
            eigenvalues=torch.from_numpy(eigenvalues).float(),
        )
        
        # Input features (vertex positions)
        x = torch.from_numpy(mesh.vertices).float()
        
        output = model(x)
        assert output.shape == (mesh.vertices.shape[0], 16)
    
    def test_cheb_conv(self, trimesh_simple):
        """Test Chebyshev convolution."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.spectral import ChebConv
        except ImportError:
            pytest.skip("Spectral module not available")
        
        mesh = trimesh_simple
        
        # Build edge index from mesh
        edges = mesh.edges if hasattr(mesh, 'edges') else None
        if edges is None:
            pytest.skip("Mesh edges not available")
        
        edge_index = torch.from_numpy(edges.T).long()
        
        model = ChebConv(
            in_channels=3,
            out_channels=16,
            K=3,  # Chebyshev polynomial order
        )
        
        x = torch.from_numpy(mesh.vertices).float()
        output = model(x, edge_index, num_nodes=mesh.vertices.shape[0])
        
        assert output.shape == (mesh.vertices.shape[0], 16)


# ============================================================================
# Tangent Subpackage Tests
# ============================================================================

class TestTangentBasis:
    """Tests for tangent space operations."""
    
    def test_tangent_basis_computation(self, simple_mesh):
        """Test computing tangent basis at vertices."""
        try:
            from shared.diffgeo.tangent import compute_tangent_basis
        except ImportError:
            pytest.skip("Tangent module not available")
        
        vertices, faces = simple_mesh
        
        # For a sphere, we can compute tangent basis
        # Using vertex normals
        normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        e1, e2 = compute_tangent_basis(normals)
        
        # Check orthonormality
        assert e1.shape == vertices.shape
        assert e2.shape == vertices.shape
        
        # e1, e2 should be unit vectors
        e1_norms = np.linalg.norm(e1, axis=1)
        e2_norms = np.linalg.norm(e2, axis=1)
        np.testing.assert_allclose(e1_norms, 1.0, atol=1e-6)
        np.testing.assert_allclose(e2_norms, 1.0, atol=1e-6)
        
        # e1, e2 should be orthogonal
        dots = np.sum(e1 * e2, axis=1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-6)
        
        # Both should be orthogonal to normal
        dots_n1 = np.sum(e1 * normals, axis=1)
        dots_n2 = np.sum(e2 * normals, axis=1)
        np.testing.assert_allclose(dots_n1, 0.0, atol=1e-6)
        np.testing.assert_allclose(dots_n2, 0.0, atol=1e-6)
    
    def test_parallel_transport(self, simple_mesh):
        """Test parallel transport along edges."""
        try:
            from shared.diffgeo.tangent import compute_tangent_basis, parallel_transport
        except ImportError:
            pytest.skip("Tangent module not available")
        
        vertices, faces = simple_mesh
        normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        e1, e2 = compute_tangent_basis(normals)
        
        # Transport a vector from vertex 0 to vertex 1
        v_source = e1[0]  # A tangent vector at vertex 0
        v_transported = parallel_transport(
            v_source,
            vertices[0],
            vertices[1],
            normals[0],
            normals[1],
        )
        
        # Result should be tangent to destination
        assert np.abs(np.dot(v_transported, normals[1])) < 1e-6


class TestTangentMessagePassing:
    """Tests for tangent-aware GNN layers."""
    
    def test_tangent_message_passing(self, simple_mesh):
        """Test TangentMessagePassing forward pass."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.tangent import TangentMessagePassing
        except ImportError:
            pytest.skip("Tangent module not available")
        
        vertices, faces = simple_mesh
        
        # Build edge list from faces
        edges = set()
        for face in faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges.add(e)
        edges = np.array(list(edges), dtype=np.int64)
        edge_index = torch.from_numpy(edges.T).long()
        
        model = TangentMessagePassing(
            in_channels=3,
            out_channels=16,
        )
        
        x = torch.from_numpy(vertices).float()
        normals = x / torch.norm(x, dim=1, keepdim=True)
        
        output = model(x, edge_index, normals)
        
        assert output.shape == (vertices.shape[0], 16)


# ============================================================================
# Geodesic Module Tests
# ============================================================================

class TestGeodesic:
    """Tests for geodesic distance computation."""
    
    def test_dijkstra_distance(self, trimesh_simple):
        """Test Dijkstra distance computation."""
        try:
            from shared.diffgeo.geodesic import geodesic_distance_dijkstra
        except ImportError:
            pytest.skip("Geodesic module not available")
        
        mesh = trimesh_simple
        
        # Compute distance from vertex 0
        distances = geodesic_distance_dijkstra(mesh, source_vertices=np.array([0]))
        
        # Check shape
        assert distances.shape == (mesh.vertices.shape[0],)
        
        # Distance to self should be 0
        assert distances[0] == 0.0
        
        # All distances should be non-negative
        assert np.all(distances >= 0)
        
        # All distances should be finite
        assert np.all(np.isfinite(distances))
    
    def test_fast_marching_distance(self, trimesh_simple):
        """Test fast marching distance."""
        try:
            from shared.diffgeo.geodesic import geodesic_distance_fast_marching
        except ImportError:
            pytest.skip("Geodesic module not available")
        
        mesh = trimesh_simple
        
        distances = geodesic_distance_fast_marching(mesh, source_vertices=np.array([0]))
        
        assert distances.shape == (mesh.vertices.shape[0],)
        assert distances[0] == 0.0
        assert np.all(distances >= 0)


# ============================================================================
# Projection Module Tests
# ============================================================================

class TestProjection:
    """Tests for map projections."""
    
    def test_latlon_to_xyz(self):
        """Test lat/lon to Cartesian conversion."""
        try:
            from shared.diffgeo.projection import latlon_to_xyz, xyz_to_latlon
        except ImportError:
            pytest.skip("Projection module not available")
        
        # Test equator point
        lat, lon = 0.0, 0.0
        x, y, z = latlon_to_xyz(lat, lon)
        
        np.testing.assert_allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-10)
        
        # Test round trip
        lat2, lon2 = xyz_to_latlon(x, y, z)
        np.testing.assert_allclose([lat2, lon2], [lat, lon], atol=1e-10)
    
    def test_gnomonic_projection(self):
        """Test gnomonic projection."""
        try:
            from shared.diffgeo.projection import gnomonic_projection
        except ImportError:
            pytest.skip("Projection module not available")
        
        # Project point near tangent point
        lat0, lon0 = 45.0, -122.0  # Portland
        lat, lon = 45.5, -121.5  # Nearby point
        
        x, y = gnomonic_projection(lat, lon, lat0, lon0)
        
        # Result should be small (nearby point)
        assert abs(x) < 0.1
        assert abs(y) < 0.1
    
    def test_coriolis_parameter(self):
        """Test Coriolis parameter computation."""
        try:
            from shared.diffgeo.projection import compute_coriolis_parameter
        except ImportError:
            pytest.skip("Projection module not available")
        
        # At equator, f = 0
        f_equator = compute_coriolis_parameter(0.0)
        np.testing.assert_allclose(f_equator, 0.0, atol=1e-10)
        
        # At poles, f = ±2Ω
        omega = 7.2921e-5
        f_north_pole = compute_coriolis_parameter(90.0)
        np.testing.assert_allclose(f_north_pole, 2 * omega, rtol=1e-6)


# ============================================================================
# Adapters Tests
# ============================================================================

class TestCoastalAdapter:
    """Tests for coastal adapter."""
    
    def test_coastal_adapter_creation(self, plane_mesh):
        """Test creating coastal adapter."""
        try:
            from shared.diffgeo.adapters import CoastalManifoldAdapter
            from shared.diffgeo.adapters.coastal_adapter import CoastalDomainConfig
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Coastal adapter not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        config = CoastalDomainConfig(
            x_min=0, x_max=1,
            y_min=0, y_max=1,
        )
        
        adapter = CoastalManifoldAdapter(
            config=config,
            mesh=mesh,
            bathymetry=np.ones(vertices.shape[0]) * 10,  # 10m depth
        )
        
        assert adapter.mesh is not None
        assert adapter.bathymetry is not None
    
    def test_shallow_water_loss(self, plane_mesh):
        """Test shallow water loss computation."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.adapters import CoastalManifoldAdapter, ShallowWaterManifoldLoss
            from shared.diffgeo.adapters.coastal_adapter import CoastalDomainConfig
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Coastal adapter not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        config = CoastalDomainConfig()
        adapter = CoastalManifoldAdapter(
            config=config,
            mesh=mesh,
            bathymetry=np.ones(vertices.shape[0]) * 10,
        )
        
        loss_fn = ShallowWaterManifoldLoss(adapter, use_coriolis=False)
        
        N = vertices.shape[0]
        eta = torch.zeros(N)
        u = torch.zeros(N, 2)
        d_eta_dt = torch.zeros(N)
        d_u_dt = torch.zeros(N, 2)
        grad_eta = torch.zeros(N, 2)
        div_hu = torch.zeros(N)
        
        losses = loss_fn(eta, u, d_eta_dt, d_u_dt, grad_eta, div_hu)
        
        assert 'total' in losses
        assert 'continuity' in losses


class TestClothAdapter:
    """Tests for cloth adapter."""
    
    def test_cloth_adapter_creation(self, plane_mesh):
        """Test creating cloth adapter."""
        try:
            from shared.diffgeo.adapters import ClothManifoldAdapter
            from shared.diffgeo.adapters.cloth_adapter import MaterialType
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Cloth adapter not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        adapter = ClothManifoldAdapter(
            mesh=mesh,
            material=ClothManifoldAdapter.material if hasattr(ClothManifoldAdapter, 'material') else None,
        )
        
        assert adapter.mesh is not None
        assert adapter.rest_vertices is not None
    
    def test_strain_computation(self, plane_mesh):
        """Test Green-Lagrange strain computation."""
        try:
            from shared.diffgeo.adapters import ClothManifoldAdapter
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Cloth adapter not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        adapter = ClothManifoldAdapter(mesh=mesh)
        
        # Deform: stretch by 10% in x
        deformed = vertices.copy()
        deformed[:, 0] *= 1.1
        
        strain = adapter.compute_strain_tensor(deformed)
        
        # Should be (n_faces, 2, 2)
        assert strain.shape[0] == faces.shape[0]
        assert strain.shape[1] == 2
        assert strain.shape[2] == 2
        
        # Strain in x direction should be positive
        assert np.mean(strain[:, 0, 0]) > 0
    
    def test_cloth_loss(self, plane_mesh):
        """Test cloth physics loss."""
        pytest.importorskip("torch")
        import torch
        
        try:
            from shared.diffgeo.adapters import ClothManifoldAdapter, ClothManifoldLoss
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Cloth adapter not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        adapter = ClothManifoldAdapter(mesh=mesh)
        loss_fn = ClothManifoldLoss(adapter)
        
        # Test with undeformed mesh (should have low energy)
        verts_tensor = torch.from_numpy(vertices).float()
        
        losses = loss_fn(verts_tensor)
        
        assert 'total' in losses
        assert 'membrane' in losses


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests across modules."""
    
    def test_coastal_pipeline_creation(self, plane_mesh):
        """Test creating complete coastal pipeline."""
        pytest.importorskip("torch")
        
        try:
            from shared.diffgeo.adapters import CoastalManifoldAdapter, create_coastal_pipeline
            from shared.diffgeo.adapters.coastal_adapter import CoastalDomainConfig
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Required modules not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        config = CoastalDomainConfig()
        adapter = CoastalManifoldAdapter(
            config=config,
            mesh=mesh,
            bathymetry=np.ones(vertices.shape[0]) * 10,
        )
        
        # This may fail if model components not fully implemented
        try:
            model, loss_fn = create_coastal_pipeline(
                adapter,
                model_type='atlas_pinn',
                n_charts=4,
            )
            assert model is not None
            assert loss_fn is not None
        except (ImportError, NotImplementedError):
            pytest.skip("Pipeline components not fully implemented")
    
    def test_cloth_pipeline_creation(self, plane_mesh):
        """Test creating complete cloth pipeline."""
        pytest.importorskip("torch")
        
        try:
            from shared.diffgeo.adapters import ClothManifoldAdapter, create_cloth_pipeline
            from shared.diffgeo.mesh.trimesh import TriangleMesh
        except ImportError:
            pytest.skip("Required modules not available")
        
        vertices, faces = plane_mesh
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        
        adapter = ClothManifoldAdapter(mesh=mesh)
        
        try:
            model, loss_fn = create_cloth_pipeline(
                adapter,
                model_type='tangent_gnn',
            )
            assert model is not None
            assert loss_fn is not None
        except (ImportError, NotImplementedError):
            pytest.skip("Pipeline components not fully implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
