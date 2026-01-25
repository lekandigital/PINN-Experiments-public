"""
Integration Test: Full Forward Pass

Tests the complete HGNN-NIF-Cloth forward pass with synthetic data.
Run: pytest tests/test_forward.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import HGNN_NIF_ClothModel


def test_forward_pass():
    """Test full forward pass with synthetic data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = HGNN_NIF_ClothModel(
        node_feat_dim=3, 
        latent_dim=64, 
        hidden_dim=64,
        siren_hidden_dim=128,
        siren_layers=3
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate small batch of synthetic data
    batch_size = 2
    fine_pos = torch.randn(batch_size, 400, 3).to(device)
    fine_edges = torch.randint(0, 400, (2, 300)).to(device)
    coarse_pos = torch.randn(batch_size, 100, 3).to(device)
    coarse_edges = torch.randint(0, 100, (2, 80)).to(device)
    query_points = torch.randn(batch_size, 1000, 3).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(
            (fine_pos, fine_edges), 
            (coarse_pos, coarse_edges), 
            query_points
        )
    
    # Assertions
    assert 'latent' in output, "Output should contain 'latent'"
    assert 'sdf' in output, "Output should contain 'sdf'"
    assert 'energy' in output, "Output should contain 'energy'"
    
    # Check shapes
    assert output['latent'].shape == (batch_size, 64), \
        f"Latent shape mismatch: {output['latent'].shape}"
    assert output['sdf'].shape == (batch_size, 1000), \
        f"SDF shape mismatch: {output['sdf'].shape}"
    assert output['energy'].shape == (batch_size,), \
        f"Energy shape mismatch: {output['energy'].shape}"
    
    # Check no NaNs
    assert not torch.isnan(output['latent']).any(), "Latent contains NaN"
    assert not torch.isnan(output['sdf']).any(), "SDF contains NaN"
    
    print("\n✓ Forward pass successful!")
    print(f"  Latent shape: {output['latent'].shape}")
    print(f"  SDF shape: {output['sdf'].shape}")
    print(f"  Energy: {output['energy'].cpu().tolist()}")
    
    from src.training.losses import PhysicsLoss
    
    # Create model and loss
    model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64, hidden_dim=64)
    model = model.to(device)
    model.train()
    
    loss_fn = PhysicsLoss(lambda_spring=0.1, lambda_sdf=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate data
    batch_size = 2
    fine_pos = torch.randn(batch_size, 400, 3, requires_grad=True).to(device)
    fine_edges = torch.randint(0, 400, (2, 300)).to(device)
    coarse_pos = torch.randn(batch_size, 100, 3).to(device)
    coarse_edges = torch.randint(0, 100, (2, 80)).to(device)
    query_points = torch.randn(batch_size, 1000, 3).to(device)
    gt_sdf = torch.randn(batch_size, 1000).to(device)
    
    # Forward
    output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
    
    # Compute loss
    total_loss, loss_dict = loss_fn(
        pred_pos=fine_pos,
        pred_sdf=output['sdf'],
        gt_sdf=gt_sdf,
        edge_index=fine_edges
    )
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Assertions
    assert total_loss.item() > 0, "Loss should be positive"
    assert 'spring' in loss_dict, "Loss dict should contain 'spring'"
    assert 'sdf' in loss_dict, "Loss dict should contain 'sdf'"
    
    print("\n✓ Training step successful!")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Spring loss: {loss_dict['spring'].item():.4f}")
    print(f"  SDF loss: {loss_dict['sdf'].item():.4f}")
    
    return True


def test_inference_speed():
    """Benchmark inference speed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64, hidden_dim=64)
    model = model.to(device)
    model.eval()
    
    # Generate data
    fine_pos = torch.randn(1, 400, 3).to(device)
    fine_edges = torch.randint(0, 400, (2, 300)).to(device)
    coarse_pos = torch.randn(1, 100, 3).to(device)
    coarse_edges = torch.randint(0, 100, (2, 80)).to(device)
    query_points = torch.randn(1, 1000, 3).to(device)
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
    
    # Benchmark
    import time
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    
    print(f"\n✓ Inference benchmark complete!")
    print(f"  Device: {device}")
    print(f"  Average inference time: {avg_time*1000:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    
    return fps


def test_memory_usage():
    """Check GPU memory usage."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = torch.device('cuda')
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = HGNN_NIF_ClothModel(node_feat_dim=3, latent_dim=64, hidden_dim=64)
    model = model.to(device)
    model.train()
    
    # Larger batch for memory test
    batch_size = 8
    fine_pos = torch.randn(batch_size, 400, 3).to(device)
    fine_edges = torch.randint(0, 400, (2, 300)).to(device)
    coarse_pos = torch.randn(batch_size, 100, 3).to(device)
    coarse_edges = torch.randint(0, 100, (2, 80)).to(device)
    query_points = torch.randn(batch_size, 2000, 3).to(device)
    
    # Forward + backward
    output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
    loss = output['sdf'].mean()
    loss.backward()
    
    # Get memory stats
    current = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n✓ Memory usage check complete!")
    print(f"  Current memory: {current:.2f} GB")
    print(f"  Peak memory: {peak:.2f} GB")
    print(f"  Batch size: {batch_size}")
    print(f"  Query points: 2000 per sample")
    
    # Should fit in 48GB with headroom
    assert peak < 40, f"Peak memory {peak:.1f}GB exceeds 40GB threshold"
    
    return peak


if __name__ == '__main__':
    print("=" * 60)
    print("HGNN-NIF-Cloth Integration Tests")
    print("=" * 60)
    
    test_forward_pass()
    test_training_step()
    fps = test_inference_speed()
    
    if torch.cuda.is_available():
        test_memory_usage()
        
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
