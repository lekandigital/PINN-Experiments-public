"""
Unit tests for physics loss functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from src.losses import (
    PhysicsLossStack,
    compute_stretch_loss,
    compute_bend_loss,
    compute_momentum_loss,
    compute_collision_loss,
)
from src.losses.physics_losses import (
    compute_edge_lengths,
    compute_dihedral_angles,
    create_grid_mesh_topology,
)


class TestEdgeLengthComputation:
    """Tests for edge length computation."""
    
    def test_basic_computation(self):
        """Test edge length computation on simple mesh."""
        # 3 vertices forming a triangle
        vertices = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
        edges = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)
        
        lengths = compute_edge_lengths(vertices, edges)
        
        assert lengths.shape == (1, 3)
        assert torch.allclose(lengths[0, 0], torch.tensor(1.0))  # Edge 0-1
        assert torch.allclose(lengths[0, 2], torch.tensor(1.0))  # Edge 0-2
    
    def test_batched_computation(self):
        """Test batch processing."""
        batch_size = 4
        num_vertices = 10
        num_edges = 15
        
        vertices = torch.randn(batch_size, num_vertices, 3)
        edges = torch.randint(0, num_vertices, (num_edges, 2))
        
        lengths = compute_edge_lengths(vertices, edges)
        
        assert lengths.shape == (batch_size, num_edges)
        assert (lengths >= 0).all()


class TestStretchLoss:
    """Tests for stretch loss computation."""
    
    def test_zero_stretch(self):
        """Test that identical meshes have zero stretch loss."""
        vertices = torch.randn(4, 16, 3)
        edges = torch.randint(0, 16, (20, 2))
        rest_lengths = compute_edge_lengths(vertices, edges)
        
        loss = compute_stretch_loss(vertices, edges, rest_lengths.squeeze(0))
        
        assert loss.item() < 1e-6
    
    def test_stretched_mesh(self):
        """Test that stretched mesh has positive loss."""
        vertices = torch.randn(4, 16, 3)
        edges = torch.randint(0, 16, (20, 2))
        rest_lengths = compute_edge_lengths(vertices, edges)
        
        # Scale mesh (stretch)
        stretched = vertices * 1.5
        
        loss = compute_stretch_loss(stretched, edges, rest_lengths.squeeze(0))
        
        assert loss.item() > 0
    
    def test_gradient_flow(self):
        """Test that gradients flow through stretch loss."""
        vertices = torch.randn(4, 16, 3, requires_grad=True)
        edges = torch.randint(0, 16, (20, 2))
        rest_lengths = torch.ones(20)
        
        loss = compute_stretch_loss(vertices, edges, rest_lengths)
        loss.backward()
        
        assert vertices.grad is not None
        assert not torch.isnan(vertices.grad).any()


class TestBendLoss:
    """Tests for bending loss computation."""
    
    def test_flat_mesh(self):
        """Test that flat mesh has zero bend loss from flat rest."""
        # Create flat 4x4 grid
        edges, faces, edge_to_faces = create_grid_mesh_topology(4)
        
        # Flat vertices
        vertices = torch.zeros(1, 16, 3)
        for i in range(4):
            for j in range(4):
                vertices[0, i * 4 + j, 0] = i
                vertices[0, i * 4 + j, 2] = j
        
        # Compute rest angles
        rest_angles = compute_dihedral_angles(vertices, faces, edge_to_faces).squeeze(0)
        
        loss = compute_bend_loss(vertices, faces, edge_to_faces, rest_angles)
        
        # Should be near zero (flat mesh, flat rest)
        assert loss.item() < 1e-4


class TestMomentumLoss:
    """Tests for momentum conservation loss."""
    
    def test_stationary_mesh(self):
        """Test that stationary mesh with zero acceleration has zero momentum loss."""
        v_curr = torch.randn(4, 16, 3)
        v_prev = v_curr.clone()  # Same position = velocity zero
        a_pred = torch.zeros_like(v_curr)
        
        loss = compute_momentum_loss(v_curr, v_prev, a_pred, dt=1/30)
        
        assert loss.item() < 1e-6
    
    def test_accelerating_mesh(self):
        """Test that accelerating mesh with wrong prediction has positive loss."""
        v_prev = torch.randn(4, 16, 3)
        v_curr = v_prev + 0.1  # Moving
        a_pred = torch.zeros_like(v_curr)  # Wrong prediction
        
        loss = compute_momentum_loss(v_curr, v_prev, a_pred, dt=1/30)
        
        assert loss.item() > 0


class TestCollisionLoss:
    """Tests for collision loss computation."""
    
    def test_above_ground(self):
        """Test that vertices above ground have zero collision loss."""
        vertices = torch.ones(4, 16, 3)  # All at y=1
        
        loss = compute_collision_loss(vertices, ground_height=0.0)
        
        assert loss.item() < 1e-6
    
    def test_below_ground(self):
        """Test that vertices below ground have positive collision loss."""
        vertices = torch.zeros(4, 16, 3)
        vertices[..., 1] = -0.5  # Below ground
        
        loss = compute_collision_loss(vertices, ground_height=0.0)
        
        assert loss.item() > 0
    
    def test_margin(self):
        """Test collision margin."""
        vertices = torch.zeros(4, 16, 3)
        vertices[..., 1] = 0.005  # Just above ground but within margin
        
        loss = compute_collision_loss(vertices, ground_height=0.0, margin=0.01)
        
        # Should have some loss due to margin
        assert loss.item() > 0


class TestPhysicsLossStack:
    """Tests for combined physics loss stack."""
    
    def test_initialization(self):
        """Test loss stack initialization."""
        loss_stack = PhysicsLossStack(
            lambda_stretch=1.0,
            lambda_bend=0.1,
            lambda_momentum=0.1,
            lambda_collision=10.0,
        )
        
        assert loss_stack.lambda_stretch == 1.0
        assert loss_stack.lambda_collision == 10.0
    
    def test_set_mesh_topology(self):
        """Test setting mesh topology."""
        loss_stack = PhysicsLossStack()
        
        edges, faces, edge_to_faces = create_grid_mesh_topology(4)
        rest_vertices = torch.randn(16, 3)
        
        loss_stack.set_mesh_topology(edges, faces, edge_to_faces, rest_vertices)
        
        assert loss_stack.edges is not None
        assert loss_stack.faces is not None
        assert loss_stack.rest_lengths is not None
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_stack = PhysicsLossStack()
        
        pred = torch.randn(4, 16, 3)
        target = torch.randn(4, 16, 3)
        
        total_loss, loss_dict = loss_stack(
            pred_vertices=pred,
            target_vertices=target,
        )
        
        assert 'total' in loss_dict
        assert 'data' in loss_dict
        assert 'collision' in loss_dict
        assert total_loss.item() > 0
    
    def test_forward_with_topology(self):
        """Test forward with mesh topology set."""
        loss_stack = PhysicsLossStack(
            lambda_stretch=1.0,
            lambda_bend=0.1,
        )
        
        edges, faces, edge_to_faces = create_grid_mesh_topology(4)
        rest_vertices = torch.randn(16, 3)
        loss_stack.set_mesh_topology(edges, faces, edge_to_faces, rest_vertices)
        
        pred = torch.randn(4, 16, 3)
        target = torch.randn(4, 16, 3)
        
        total_loss, loss_dict = loss_stack(pred, target)
        
        assert 'stretch' in loss_dict
        assert 'bend' in loss_dict


class TestGridMeshTopology:
    """Tests for grid mesh topology generation."""
    
    def test_vertex_count(self):
        """Test correct number of edges/faces for grid."""
        grid_size = 4
        edges, faces, edge_to_faces = create_grid_mesh_topology(grid_size)
        
        # Expected face count: 2 * (grid_size - 1)^2
        expected_faces = 2 * (grid_size - 1) ** 2
        assert faces.shape[0] == expected_faces
    
    def test_face_indices_valid(self):
        """Test face indices are within valid range."""
        grid_size = 8
        edges, faces, edge_to_faces = create_grid_mesh_topology(grid_size)
        
        num_vertices = grid_size * grid_size
        assert faces.max() < num_vertices
        assert faces.min() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
