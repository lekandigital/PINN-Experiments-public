"""
Integration Test: Training Loop

Tests the complete training pipeline with a small dataset
to verify everything works together.
"""

import pytest
import torch
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader
from src.models.coastflow_gnn import CoastFlowGNN
from src.models.physics_losses import physics_informed_loss
from src.data.dataset import SyntheticCoastalDataset, create_data_loaders


def test_training_loop():
    """Run 5 epochs on small dataset to verify training works."""
    print("\n" + "=" * 50)
    print("Testing Training Loop")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create small synthetic dataset
        print("Creating synthetic dataset...")
        dataset = SyntheticCoastalDataset(
            root=tmp_dir,
            num_samples=20,
            num_nodes=50,
            seed=42,
        )
        
        train_loader, val_loader, _ = create_data_loaders(
            dataset, batch_size=4
        )
        print(f"Dataset: {len(dataset)} samples, "
              f"{len(train_loader)} train batches")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CoastFlowGNN(
            in_channels=6,
            hidden_channels=32,  # Smaller for quick test
            out_channels=4
        ).to(device)
        
        print(f"Model: {model.count_parameters():,} parameters")
        print(f"Device: {device}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 5 epochs
        print("\nTraining:")
        model.train()
        losses = []
        
        for epoch in range(5):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                pred = model(batch.x, batch.edge_index, batch.batch)
                loss_dict = physics_informed_loss(batch, pred)
                loss = loss_dict['total']
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"  Epoch {epoch+1}/5: Loss = {avg_loss:.6f}")
        
        # Validate
        print("\nValidation:")
        model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch)
                loss_dict = physics_informed_loss(batch, pred)
                val_loss += loss_dict['total'].item()
                num_batches += 1
        
        avg_val_loss = val_loss / max(num_batches, 1)
        print(f"  Validation Loss: {avg_val_loss:.6f}")
        
        # Checks
        assert not any(torch.isnan(torch.tensor(losses))), "Training loss contains NaN"
        assert losses[-1] < losses[0] * 2, "Loss should not explode"
        
        print("\n" + "=" * 50)
        print("✓ Training loop test PASSED")
        print("=" * 50 + "\n")
        
        return True


def test_inference_speed():
    """Benchmark inference speed."""
    print("\n" + "=" * 50)
    print("Testing Inference Speed")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CoastFlowGNN(
        in_channels=6,
        hidden_channels=64,
        out_channels=4
    ).to(device)
    model.eval()
    
    # Create test data
    num_nodes = 200
    x = torch.randn(num_nodes, 6, device=device)
    edge_index = torch.randint(0, num_nodes, (2, 600), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x, edge_index, batch)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    num_runs = 100
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = model(x, edge_index, batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    time_per_sample_ms = (end - start) / num_runs * 1000
    
    print(f"  Device: {device}")
    print(f"  Nodes per sample: {num_nodes}")
    print(f"  Time per sample: {time_per_sample_ms:.2f} ms")
    print(f"  Throughput: {1000/time_per_sample_ms:.1f} samples/sec")
    
    # Check meets target
    target_ms = 100
    passed = time_per_sample_ms < target_ms
    
    print(f"\n  Target: < {target_ms} ms")
    print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 50 + "\n")
    
    return passed


def test_model_save_load():
    """Test model checkpointing."""
    print("\n" + "=" * 50)
    print("Testing Model Save/Load")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create and save model
        model1 = CoastFlowGNN(in_channels=6, hidden_channels=32, out_channels=4)
        
        # Run forward pass to initialize
        x = torch.randn(50, 6)
        edge_index = torch.randint(0, 50, (2, 150))
        batch = torch.zeros(50, dtype=torch.long)
        
        out1 = model1(x, edge_index, batch)
        
        # Save checkpoint
        checkpoint_path = Path(tmp_dir) / 'model.pth'
        torch.save({
            'model_state_dict': model1.state_dict(),
            'epoch': 10,
        }, checkpoint_path)
        print(f"  Saved checkpoint to: {checkpoint_path}")
        
        # Load into new model
        model2 = CoastFlowGNN(in_channels=6, hidden_channels=32, out_channels=4)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Verify same output
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x, edge_index, batch)
            out2 = model2(x, edge_index, batch)
        
        assert torch.allclose(out1, out2), "Outputs should match after loading"
        print("  Outputs match after save/load")
        
        print("\n✓ Model save/load test PASSED")
        print("=" * 50 + "\n")
        
        return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CoastFlow-GNN Integration Tests")
    print("=" * 60)
    
    results = {}
    
    results['training_loop'] = test_training_loop()
    results['inference_speed'] = test_inference_speed()
    results['save_load'] = test_model_save_load()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 60 + "\n")
