"""
Test ClothGNN model functionality on synthetic data.
Verifies forward pass, gradient computation, and basic training.
"""
import sys
import os
import time

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import h5py
from torch_geometric.data import Data

from models.clothgnn import ClothGNNModel, ClothEncoder, ClothDecoder
from utils.losses import position_loss, edge_length_loss, compute_rest_lengths, total_loss
from utils.collision import OccupancyGrid, sphere_sdf


def load_test_data(data_path):
    """Load synthetic test dataset."""
    with h5py.File(data_path, "r") as f:
        trajectory = torch.tensor(f["trajectory"][:])
        displacements = torch.tensor(f["displacements"][:])
        edge_index = torch.tensor(f["edge_index"][:])
        faces = torch.tensor(f["faces"][:])
        rest_positions = torch.tensor(f["rest_positions"][:])
    return trajectory, displacements, edge_index, faces, rest_positions


def test_encoder(device="cuda"):
    """Test ClothEncoder independently."""
    print("\n=== Testing ClothEncoder ===")
    
    num_nodes = 100
    in_channels = 16
    hidden_channels = 64
    
    encoder = ClothEncoder(in_channels, hidden_channels).to(device)
    
    # Random input
    x = torch.randn(num_nodes, in_channels).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 300)).to(device)
    pos = torch.randn(num_nodes, 3).to(device)
    
    # Forward pass
    out = encoder(x, edge_index, pos)
    
    print(f"✓ Encoder forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (num_nodes, hidden_channels), "Encoder output shape mismatch"
    
    return encoder


def test_decoder(device="cuda"):
    """Test ClothDecoder independently."""
    print("\n=== Testing ClothDecoder ===")
    
    num_nodes = 100
    hidden_channels = 64
    out_channels = 3
    
    decoder = ClothDecoder(hidden_channels, out_channels).to(device)
    
    # Random input
    h = torch.randn(num_nodes, hidden_channels).to(device)
    
    # Forward pass
    out = decoder(h)
    
    print(f"✓ Decoder forward pass successful")
    print(f"  Input shape: {h.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (num_nodes, out_channels), "Decoder output shape mismatch"
    
    return decoder


def test_forward_pass(device="cuda"):
    """Test full model forward pass with minimal data."""
    print("\n=== Testing Full Model Forward Pass ===")
    
    # Create minimal graph
    num_nodes = 100
    node_feat_dim = 16
    hidden_dim = 64
    
    node_features = torch.randn(num_nodes, node_feat_dim).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 300)).to(device)
    pos = torch.randn(num_nodes, 3).to(device)
    
    # Initialize model
    model = ClothGNNModel(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim).to(device)
    h0 = model.init_hidden(num_nodes, device)
    
    # Forward pass
    data = Data(x=node_features, edge_index=edge_index, pos=pos)
    pred, h_next = model(data, h0)
    
    print(f"✓ Forward pass successful")
    print(f"  Input features: {node_features.shape}")
    print(f"  Output predictions: {pred.shape}")
    print(f"  Hidden state: {h_next.shape}")
    assert pred.shape == (num_nodes, 3), "Output shape mismatch"
    assert h_next.shape == (num_nodes, hidden_dim), "Hidden state shape mismatch"
    
    # Check for NaN
    assert not torch.isnan(pred).any(), "NaN in predictions"
    assert not torch.isnan(h_next).any(), "NaN in hidden state"
    
    return model


def test_collision_module(device="cuda"):
    """Test occupancy grid collision detection."""
    print("\n=== Testing Collision Module ===")
    
    voxelizer = OccupancyGrid(grid_dim=32, bounds=((-1, 1), (-1, 2), (-1, 1))).to(device)
    
    # Test with random positions
    node_positions = torch.randn(100, 3).to(device)
    occupancy = voxelizer(node_positions, sphere_sdf)
    
    print(f"✓ Collision module working")
    print(f"  Node positions: {node_positions.shape}")
    print(f"  Occupancy features: {occupancy.shape}")
    print(f"  Occupied nodes: {occupancy.sum().item():.0f} / {len(occupancy)}")
    
    # Test grid computation
    grid = voxelizer.compute_grid_occupancy(sphere_sdf)
    print(f"  Full grid shape: {grid.shape}")
    print(f"  Occupied voxels: {grid.sum().item():.0f}")
    
    return voxelizer


def test_loss_computation(device="cuda"):
    """Test loss function calculations."""
    print("\n=== Testing Loss Computation ===")
    
    num_nodes = 100
    num_edges = 300
    
    # Create leaf tensor for gradient tracking
    pred = torch.randn(num_nodes, 3).to(device)
    pred.requires_grad_(True)
    target = torch.randn(num_nodes, 3).to(device)
    
    # Test position loss
    loss = position_loss(pred, target)
    loss.backward()
    
    print(f"✓ Position loss and backprop successful")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Gradient computed: {pred.grad is not None}")
    assert pred.grad is not None, "Gradient not computed"
    
    # Test edge length loss
    positions = torch.randn(num_nodes, 3).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
    rest_lengths = compute_rest_lengths(positions, edge_index)
    
    pred_pos = positions + torch.randn_like(positions) * 0.1
    pred_pos = pred_pos.detach().requires_grad_(True)
    
    l_edge = edge_length_loss(pred_pos, edge_index, rest_lengths)
    l_edge.backward()
    
    print(f"✓ Edge length loss successful")
    print(f"  Edge loss value: {l_edge.item():.6f}")
    
    return loss


def test_training_step(model, data_path, device="cuda"):
    """Test one training iteration with real data."""
    print("\n=== Testing Training Step ===")
    
    # Load synthetic data
    trajectory, displacements, edge_index, faces, rest_positions = load_test_data(data_path)
    
    # Move to device
    edge_index = edge_index.to(device)
    rest_positions = rest_positions.to(device)
    rest_lengths = compute_rest_lengths(rest_positions, edge_index)
    
    # Prepare one training sample (frame t -> frame t+1)
    t = 10
    current_pos = trajectory[t].to(device)
    target_disp = displacements[t].to(device)
    
    # Create node features: position (3) + velocity placeholder (3) + padding (10) = 16
    velocity = torch.zeros_like(current_pos)
    node_features = torch.cat([
        current_pos,
        velocity,
        torch.zeros(len(current_pos), 10, device=device)
    ], dim=1)
    
    data = Data(x=node_features, edge_index=edge_index, pos=current_pos)
    
    # Forward pass
    h0 = model.init_hidden(len(current_pos), device)
    pred_disp, h_next = model(data, h0)
    
    # Compute total loss
    pred_pos = current_pos + pred_disp
    loss, loss_dict = total_loss(
        pred_disp, target_disp, pred_pos, edge_index, rest_lengths,
        triangles=faces.to(device)
    )
    
    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Total loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.6f}")
    print(f"  Prediction range: [{pred_disp.min().item():.4f}, {pred_disp.max().item():.4f}]")
    
    return loss.item()


def test_multi_step_rollout(model, data_path, num_steps=10, device="cuda"):
    """Test multi-step prediction rollout."""
    print("\n=== Testing Multi-Step Rollout ===")
    
    trajectory, displacements, edge_index, faces, rest_positions = load_test_data(data_path)
    
    # Initialize
    current_pos = trajectory[0].to(device)
    edge_index = edge_index.to(device)
    h = model.init_hidden(len(current_pos), device)
    
    predictions = [current_pos.cpu()]
    
    model.eval()
    with torch.no_grad():
        for step in range(num_steps):
            # Create features
            velocity = torch.zeros_like(current_pos)
            node_features = torch.cat([
                current_pos, velocity,
                torch.zeros(len(current_pos), 10, device=device)
            ], dim=1)
            
            data = Data(x=node_features, edge_index=edge_index, pos=current_pos)
            
            # Predict
            pred_disp, h = model(data, h)
            
            # Update position
            current_pos = current_pos + pred_disp
            predictions.append(current_pos.cpu())
            
            # Check for NaN/Inf
            assert not torch.isnan(pred_disp).any(), f"NaN at step {step}"
            assert not torch.isinf(pred_disp).any(), f"Inf at step {step}"
    
    model.train()
    predictions = torch.stack(predictions)
    
    print(f"✓ Multi-step rollout successful")
    print(f"  Rolled out {num_steps} steps")
    print(f"  Final position range: [{predictions[-1].min():.4f}, {predictions[-1].max():.4f}]")
    
    # Check displacement magnitude is reasonable
    total_disp = (predictions[-1] - predictions[0]).norm(dim=1).mean()
    print(f"  Average total displacement: {total_disp:.4f}")
    
    return predictions


def test_gpu_performance(model, num_nodes=1000, num_iters=100, device="cuda"):
    """Benchmark inference speed on GPU."""
    print("\n=== Testing GPU Performance ===")
    
    # Create larger graph for benchmark
    node_features = torch.randn(num_nodes, 16).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 6)).to(device)
    pos = torch.randn(num_nodes, 3).to(device)
    data = Data(x=node_features, edge_index=edge_index, pos=pos)
    h = model.init_hidden(num_nodes, device)
    
    model.eval()
    
    # Warm up
    print("  Warming up...")
    with torch.no_grad():
        for _ in range(20):
            pred, h = model(data, h)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iters):
            pred, h = model(data, h)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    fps = num_iters / elapsed
    ms_per_frame = 1000 * elapsed / num_iters
    
    print(f"✓ Performance test complete")
    print(f"  Mesh size: {num_nodes} vertices, {edge_index.shape[1]} edges")
    print(f"  Iterations: {num_iters}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Time per frame: {ms_per_frame:.2f}ms")
    
    model.train()
    return fps


def test_larger_mesh_performance(model, device="cuda"):
    """Test with larger mesh sizes."""
    print("\n=== Testing Performance at Different Mesh Sizes ===")
    
    mesh_sizes = [100, 500, 1000, 2000, 5000]
    results = []
    
    model.eval()
    
    for num_nodes in mesh_sizes:
        num_edges = min(num_nodes * 6, num_nodes * (num_nodes - 1) // 2)
        
        node_features = torch.randn(num_nodes, 16).to(device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
        pos = torch.randn(num_nodes, 3).to(device)
        data = Data(x=node_features, edge_index=edge_index, pos=pos)
        h = model.init_hidden(num_nodes, device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                pred, h = model(data, h)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        num_iters = max(10, 100 // (num_nodes // 100))
        with torch.no_grad():
            for _ in range(num_iters):
                pred, h = model(data, h)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = num_iters / elapsed
        
        results.append((num_nodes, num_edges, fps))
        print(f"  {num_nodes:5d} vertices, {num_edges:6d} edges: {fps:7.1f} FPS")
    
    model.train()
    return results


def run_all_tests():
    """Execute complete test suite."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"ClothGNN Test Suite")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"{'='*60}")
    
    try:
        # Generate test data first
        print("\n>>> Generating synthetic test data...")
        from generate_test_data import generate_all_test_data
        generate_all_test_data(SCRIPT_DIR)

        data_path = os.path.join(SCRIPT_DIR, "gravity_10x10.h5")
        
        # Run component tests
        test_encoder(device)
        test_decoder(device)
        
        # Run full model tests
        model = test_forward_pass(device)
        test_collision_module(device)
        test_loss_computation(device)
        loss = test_training_step(model, data_path, device)
        predictions = test_multi_step_rollout(model, data_path, num_steps=10, device=device)
        
        # Performance benchmarks
        fps_1k = test_gpu_performance(model, num_nodes=1000, device=device)
        perf_results = test_larger_mesh_performance(model, device)
        
        # Memory usage
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
        else:
            mem_allocated = mem_reserved = 0
        
        # Summary
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED")
        print(f"{'='*60}")
        print(f"  Final training loss: {loss:.6f}")
        print(f"  Inference speed (1K vertices): {fps_1k:.1f} FPS")
        print(f"  GPU Memory allocated: {mem_allocated:.2f} GB")
        print(f"  GPU Memory reserved: {mem_reserved:.2f} GB")
        
        # Save results
        results_dir = os.path.join(PROJECT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "test_results.txt")
        with open(results_path, "w") as f:
            f.write("ClothGNN Test Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Device: {device}\n")
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"VRAM: {props.total_memory / 1e9:.1f} GB\n")
            f.write(f"\nFinal training loss: {loss:.6f}\n")
            f.write(f"Inference speed (1K vertices): {fps_1k:.1f} FPS\n")
            f.write(f"GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved\n")
            f.write("\nPerformance by mesh size:\n")
            for nodes, edges, fps in perf_results:
                f.write(f"  {nodes} vertices: {fps:.1f} FPS\n")
            f.write("\nALL TESTS PASSED\n")
        
        print(f"\nResults saved to: {results_path}")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
