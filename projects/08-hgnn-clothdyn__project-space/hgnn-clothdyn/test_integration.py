"""
HGNN-ClothDyn Integration Test Suite

Quick integration tests to verify:
- Model architecture correctness
- Training loop functionality
- Benchmark execution
- GPU utilization

Author: HGNN-ClothDyn
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import tempfile
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import HGNNClothDyn, EdgeForceConv, ClothSimulator
from mesh_to_graph import (
    create_grid_mesh, 
    compute_rest_lengths, 
    mesh_to_graph, 
    build_graph_pyramid
)
from synthetic_data import generate_cloth_sequence, save_to_hdf5
from train import ClothDataset, Trainer, set_seed
from benchmark import Benchmarker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """
    Integration test suite for HGNN-ClothDyn.
    """
    
    def __init__(self, device: torch.device):
        """
        Args:
            device: Compute device
        """
        self.device = device
        self.results = {}
        self.passed = 0
        self.failed = 0
        
        logger.info(f"IntegrationTester initialized on {device}")
    
    def run_test(self, name: str, test_fn) -> bool:
        """
        Run a single test with error handling.
        
        Args:
            name: Test name
            test_fn: Test function
            
        Returns:
            True if passed, False if failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {name}")
        logger.info('='*60)
        
        try:
            start_time = time.time()
            result = test_fn()
            elapsed = time.time() - start_time
            
            if result:
                self.passed += 1
                self.results[name] = {'status': 'PASS', 'time': elapsed}
                logger.info(f"✓ PASSED ({elapsed:.2f}s)")
                return True
            else:
                self.failed += 1
                self.results[name] = {'status': 'FAIL', 'time': elapsed}
                logger.error(f"✗ FAILED ({elapsed:.2f}s)")
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            self.failed += 1
            self.results[name] = {'status': 'ERROR', 'error': str(e), 'time': elapsed}
            logger.error(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_mesh_to_graph(self) -> bool:
        """Test mesh-to-graph conversion."""
        # Create grid mesh
        vertices, faces, edges = create_grid_mesh(size=10, spacing=0.1)
        
        assert vertices.shape == (100, 3), f"Wrong vertex shape: {vertices.shape}"
        assert len(edges) > 0, "No edges created"
        
        # Compute rest lengths
        rest_lengths = compute_rest_lengths(vertices, edges)
        assert len(rest_lengths) == len(edges), "Rest length count mismatch"
        assert np.all(rest_lengths > 0), "Invalid rest lengths"
        
        # Convert to graph
        data = mesh_to_graph(vertices, edges=edges, rest_lengths=rest_lengths)
        assert data.num_nodes == 100, f"Wrong node count: {data.num_nodes}"
        assert data.x.shape == (100, 6), f"Wrong feature shape: {data.x.shape}"
        
        # Build pyramid
        graphs, cluster_maps = build_graph_pyramid(data, num_levels=2)
        assert len(graphs) == 2, f"Wrong pyramid depth: {len(graphs)}"
        assert graphs[1].num_nodes < graphs[0].num_nodes, "Coarse not smaller"
        
        logger.info(f"Pyramid: {[g.num_nodes for g in graphs]} nodes per level")
        
        return True
    
    def test_edge_force_conv(self) -> bool:
        """Test EdgeForceConv layer."""
        num_nodes = 50
        num_edges = 200
        
        x = torch.randn(num_nodes, 64, device=self.device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=self.device)
        pos = torch.randn(num_nodes, 3, device=self.device)
        rest_lengths = torch.ones(num_edges, 1, device=self.device) * 0.1
        
        conv = EdgeForceConv(64, 128).to(self.device)
        
        out = conv(x, edge_index, pos, rest_lengths)
        
        assert out.shape == (num_nodes, 128), f"Wrong output shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"
        
        # Test backward pass
        loss = out.sum()
        loss.backward()
        
        for name, param in conv.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        
        return True
    
    def test_model_forward(self) -> bool:
        """Test HGNNClothDyn forward pass."""
        num_nodes = 100
        num_edges = 400
        
        model = HGNNClothDyn(
            input_dim=6,
            hidden_dim=64,
            output_dim=3,
            num_message_passes=2,
            num_levels=2
        ).to(self.device)
        
        x = torch.randn(num_nodes, 6, device=self.device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=self.device)
        rest_lengths = torch.ones(num_edges, 1, device=self.device) * 0.1
        pos = x[:, :3]
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=rest_lengths,
            pos=pos,
            num_nodes=num_nodes
        ).to(self.device)
        
        # Single-level forward
        output = model(data)
        assert output.shape == (num_nodes, 3), f"Wrong output shape: {output.shape}"
        
        # With latent extraction
        output, latent = model(data, return_latent=True)
        assert latent.shape == (num_nodes, 64), f"Wrong latent shape: {latent.shape}"
        
        # Test backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"
        
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    
    def test_model_hierarchical(self) -> bool:
        """Test hierarchical message passing."""
        num_fine = 100
        num_coarse = 25
        
        model = HGNNClothDyn(
            input_dim=6,
            hidden_dim=64,
            output_dim=3
        ).to(self.device)
        
        # Fine level
        fine_x = torch.randn(num_fine, 6, device=self.device)
        fine_edges = torch.randint(0, num_fine, (2, 400), device=self.device)
        
        data_fine = Data(
            x=fine_x,
            edge_index=fine_edges,
            edge_attr=torch.ones(400, 1, device=self.device) * 0.1,
            pos=fine_x[:, :3],
            num_nodes=num_fine
        ).to(self.device)
        
        # Coarse level
        coarse_x = torch.randn(num_coarse, 6, device=self.device)
        coarse_edges = torch.randint(0, num_coarse, (2, 50), device=self.device)
        
        data_coarse = Data(
            x=coarse_x,
            edge_index=coarse_edges,
            edge_attr=torch.ones(50, 1, device=self.device) * 0.2,
            pos=coarse_x[:, :3],
            num_nodes=num_coarse
        ).to(self.device)
        
        # Cluster map
        cluster_map = torch.randint(0, num_coarse, (num_fine,), device=self.device)
        
        # Forward with hierarchy
        output = model(data_fine, data_coarse, cluster_map)
        
        assert output.shape == (num_fine, 3), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN in hierarchical output"
        
        return True
    
    def test_synthetic_data_generation(self) -> bool:
        """Test synthetic data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_data.h5'
            
            data = generate_cloth_sequence(
                num_frames=10,
                mesh_size=10,
                output_path=str(output_path)
            )
            
            assert 'positions' in data, "Missing positions"
            assert 'velocities' in data, "Missing velocities"
            assert 'edges' in data, "Missing edges"
            
            assert data['positions'].shape == (10, 100, 3), \
                f"Wrong positions shape: {data['positions'].shape}"
            
            # Check file was saved
            assert output_path.exists(), "HDF5 file not saved"
            
            # Load and verify
            import h5py
            with h5py.File(output_path, 'r') as f:
                assert 'positions' in f, "Missing positions in file"
                assert f['positions'].shape == (10, 100, 3)
            
            logger.info(f"Generated {data['positions'].shape[0]} frames")
        
        return True
    
    def test_training_loop(self, epochs: int = 3) -> bool:
        """Test training loop execution."""
        set_seed(42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data_path = Path(tmpdir) / 'train_data.h5'
            generate_cloth_sequence(
                num_frames=20,
                mesh_size=10,
                output_path=str(data_path)
            )
            
            # Create dataset
            dataset = ClothDataset(str(data_path), device=self.device)
            
            # Create model
            model = HGNNClothDyn(
                input_dim=6,
                hidden_dim=32,  # Small for testing
                output_dim=3,
                num_message_passes=2
            )
            
            # Training config
            config = {
                'learning_rate': 1e-3,
                'use_amp': torch.cuda.is_available(),
                'use_scheduled_sampling': True,
                'sampling_start_epoch': 2,
                'gradient_clip': 1.0
            }
            
            # Create trainer
            trainer = Trainer(model, dataset, config, self.device)
            
            # Train
            checkpoint_dir = Path(tmpdir) / 'checkpoints'
            history = trainer.train(
                num_epochs=epochs,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_interval=1
            )
            
            # Verify training worked
            assert len(history['train_loss']) == epochs, "Wrong history length"
            assert history['train_loss'][-1] < history['train_loss'][0] or epochs < 3, \
                "Loss didn't decrease"
            
            # Check checkpoints
            assert (checkpoint_dir / 'final_model.pt').exists(), "No final checkpoint"
            
            logger.info(f"Training losses: {history['train_loss']}")
        
        return True
    
    def test_benchmark_execution(self) -> bool:
        """Test benchmark suite execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data_path = Path(tmpdir) / 'bench_data.h5'
            generate_cloth_sequence(
                num_frames=20,
                mesh_size=10,
                output_path=str(data_path)
            )
            
            # Create model
            model = HGNNClothDyn(
                input_dim=6,
                hidden_dim=32,
                output_dim=3
            ).to(self.device)
            
            # Create dataset
            dataset = ClothDataset(str(data_path), device=self.device)
            
            # Run benchmarks
            benchmarker = Benchmarker(model, dataset, self.device)
            results = benchmarker.run_all_benchmarks(
                rollout_frames=5,
                fps_iters=10
            )
            
            # Verify results
            assert 'summary' in results, "Missing summary"
            assert 'benchmarks' in results, "Missing benchmarks"
            assert results['summary']['fps'] > 0, "Invalid FPS"
            
            logger.info(f"Benchmark FPS: {results['summary']['fps']:.1f}")
        
        return True
    
    def test_gpu_utilization(self) -> bool:
        """Test GPU utilization and memory."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU test")
            return True
        
        torch.cuda.reset_peak_memory_stats()
        
        # Create larger model for GPU test
        model = HGNNClothDyn(
            input_dim=6,
            hidden_dim=128,
            output_dim=3,
            num_message_passes=3
        ).to(self.device)
        
        # Large batch
        num_nodes = 400
        x = torch.randn(num_nodes, 6, device=self.device)
        edges = torch.randint(0, num_nodes, (2, 2000), device=self.device)
        
        data = Data(
            x=x,
            edge_index=edges,
            edge_attr=torch.ones(2000, 1, device=self.device),
            pos=x[:, :3],
            num_nodes=num_nodes
        ).to(self.device)
        
        # Multiple forward passes
        for _ in range(10):
            output = model(data)
            loss = output.sum()
            loss.backward()
        
        # Check memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")
        
        # Verify GPU is being used
        assert peak_memory > 0.01, "GPU memory not used"
        
        # Check no memory leaks (simple check)
        torch.cuda.empty_cache()
        current = torch.cuda.memory_allocated() / 1e6
        logger.info(f"Current GPU memory after cleanup: {current:.1f} MB")
        
        return True
    
    def test_numerical_stability(self) -> bool:
        """Test for NaN/Inf in model outputs."""
        model = HGNNClothDyn(
            input_dim=6,
            hidden_dim=64,
            output_dim=3
        ).to(self.device)
        
        # Test with various inputs
        test_cases = [
            ('normal', torch.randn(100, 6)),
            ('zeros', torch.zeros(100, 6)),
            ('small', torch.randn(100, 6) * 1e-6),
            ('large', torch.randn(100, 6) * 1e3),
        ]
        
        for name, x in test_cases:
            x = x.to(self.device)
            edges = torch.randint(0, 100, (2, 400), device=self.device)
            
            data = Data(
                x=x,
                edge_index=edges,
                edge_attr=torch.ones(400, 1, device=self.device),
                pos=x[:, :3],
                num_nodes=100
            ).to(self.device)
            
            output = model(data)
            
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            
            if has_nan or has_inf:
                logger.error(f"Numerical instability with {name} input")
                return False
            
            logger.info(f"  {name}: output range [{output.min():.4f}, {output.max():.4f}]")
        
        return True
    
    def run_all_tests(
        self,
        quick: bool = False,
        epochs: int = 5
    ) -> Dict[str, Any]:
        """
        Run all integration tests.
        
        Args:
            quick: Run quick tests only
            epochs: Number of training epochs for training test
            
        Returns:
            Test results dictionary
        """
        logger.info("\n" + "=" * 60)
        logger.info("HGNN-ClothDyn Integration Test Suite")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Quick mode: {quick}")
        logger.info("=" * 60)
        
        # Core tests (always run)
        self.run_test("Mesh to Graph Conversion", self.test_mesh_to_graph)
        self.run_test("EdgeForceConv Layer", self.test_edge_force_conv)
        self.run_test("Model Forward Pass", self.test_model_forward)
        self.run_test("Hierarchical Message Passing", self.test_model_hierarchical)
        self.run_test("Numerical Stability", self.test_numerical_stability)
        
        if not quick:
            # Extended tests
            self.run_test("Synthetic Data Generation", self.test_synthetic_data_generation)
            self.run_test(f"Training Loop ({epochs} epochs)", 
                         lambda: self.test_training_loop(epochs))
            self.run_test("Benchmark Execution", self.test_benchmark_execution)
        
        if torch.cuda.is_available():
            self.run_test("GPU Utilization", self.test_gpu_utilization)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        total = self.passed + self.failed
        logger.info(f"Passed: {self.passed}/{total}")
        logger.info(f"Failed: {self.failed}/{total}")
        
        for name, result in self.results.items():
            status = result['status']
            icon = "✓" if status == "PASS" else "✗"
            logger.info(f"  {icon} {name}: {status}")
        
        logger.info("=" * 60)
        
        if self.failed == 0:
            logger.info("✓ All tests passed!")
        else:
            logger.error(f"✗ {self.failed} test(s) failed")
        
        return {
            'passed': self.passed,
            'failed': self.failed,
            'results': self.results
        }


def main():
    parser = argparse.ArgumentParser(description='Run HGNN-ClothDyn integration tests')
    
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick tests only (skip training/benchmark)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs for training test')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (not used in current implementation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run tests
    tester = IntegrationTester(device)
    results = tester.run_all_tests(
        quick=args.quick_test,
        epochs=args.epochs
    )
    
    # Exit code based on results
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
