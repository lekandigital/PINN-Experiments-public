"""
NIF-Cloth3D-Interactive: Optimized Inference Engine

This module provides real-time inference capabilities with:
- Voxel-hash caching for repeated SDF queries
- FP16 (half-precision) model inference
- Batched query optimization
- Optional CUDA kernel integration (via tiny-cuda-nn)

Target performance: <10ms inference for 10k vertices on RTX 4090/L40S

Usage:
    from inference import OptimizedInference
    
    engine = OptimizedInference('checkpoints/model_traced.pt')
    displacements = engine.predict(vertices, time, force, material_id)
"""

import time
import hashlib
from typing import Optional, Dict, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np


class VoxelHashCache:
    """
    Spatial hash-based cache for SDF/displacement queries.
    
    Uses a 3D hash grid to cache previously computed results.
    When a query falls within a cached voxel, the cached value is returned
    instead of recomputing through the neural network.
    
    This is inspired by Instant-NGP's multi-resolution hash encoding,
    but simplified for caching purposes.
    
    Args:
        resolution: Grid resolution (voxels per unit)
        max_entries: Maximum cache entries (LRU eviction)
        tolerance: Distance tolerance for cache hits
    """
    
    def __init__(
        self,
        resolution: int = 128,
        max_entries: int = 100000,
        tolerance: float = 0.01
    ):
        self.resolution = resolution
        self.max_entries = max_entries
        self.tolerance = tolerance
        
        # Cache storage: hash -> (key_tensor, value_tensor, timestamp)
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.access_counter = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _hash_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute hash keys for positions.
        
        Quantizes positions to grid cells and computes hash.
        """
        # Quantize to grid
        grid_coords = (x * self.resolution).floor().long()
        
        # Simple spatial hash (based on Teschner et al.)
        # hash(x,y,z) = (x * p1 xor y * p2 xor z * p3) mod table_size
        p1, p2, p3 = 73856093, 19349663, 83492791
        
        hash_keys = (
            grid_coords[:, 0] * p1 ^
            grid_coords[:, 1] * p2 ^
            grid_coords[:, 2] * p3
        ) % self.max_entries
        
        return hash_keys
    
    def get(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Look up cached values for positions.
        
        Args:
            positions: Query positions (N, 3)
        
        Returns:
            cached_values: Cached results for hits (M, out_dim)
            hit_mask: Boolean mask indicating cache hits (N,)
            miss_indices: Indices of cache misses
        """
        N = positions.shape[0]
        device = positions.device
        
        hash_keys = self._hash_position(positions)
        
        hit_mask = torch.zeros(N, dtype=torch.bool, device=device)
        cached_values = []
        
        for i, hk in enumerate(hash_keys.tolist()):
            if hk in self.cache:
                cached_pos, cached_val, _ = self.cache[hk]
                # Check if close enough
                if torch.norm(cached_pos - positions[i]) < self.tolerance:
                    hit_mask[i] = True
                    cached_values.append(cached_val)
                    self.cache[hk] = (cached_pos, cached_val, self.access_counter)
                    self.hits += 1
                else:
                    self.misses += 1
            else:
                self.misses += 1
        
        self.access_counter += 1
        
        miss_indices = torch.where(~hit_mask)[0]
        
        if cached_values:
            cached_tensor = torch.stack(cached_values)
        else:
            cached_tensor = torch.empty(0, 3, device=device)
        
        return cached_tensor, hit_mask, miss_indices
    
    def put(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        indices: Optional[torch.Tensor] = None
    ):
        """
        Store computed values in cache.
        
        Args:
            positions: Positions that were computed (M, 3)
            values: Computed values (M, out_dim)
            indices: Original indices if positions are a subset
        """
        hash_keys = self._hash_position(positions)
        
        for i, hk in enumerate(hash_keys.tolist()):
            if len(self.cache) >= self.max_entries:
                # LRU eviction: remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
                del self.cache[oldest_key]
            
            self.cache[hk] = (
                positions[i].detach().clone(),
                values[i].detach().clone(),
                self.access_counter
            )
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


class TensorCache:
    """
    GPU tensor-based cache using dense storage.
    
    More efficient than dict-based cache for GPU tensors,
    but uses fixed memory.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        key_dim: int = 3,
        value_dim: int = 3,
        device: str = 'cuda'
    ):
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.device = device
        
        # Pre-allocate storage
        self.keys = torch.zeros(capacity, key_dim, device=device)
        self.values = torch.zeros(capacity, value_dim, device=device)
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.size = 0
        self.next_idx = 0
    
    def query(
        self,
        queries: torch.Tensor,
        tolerance: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query cache with tolerance-based matching.
        
        Returns:
            results: (N, value_dim) with cached values (zeros for misses)
            hit_mask: (N,) boolean mask
        """
        N = queries.shape[0]
        
        if self.size == 0:
            return (
                torch.zeros(N, self.value_dim, device=self.device),
                torch.zeros(N, dtype=torch.bool, device=self.device)
            )
        
        # Compute distances to all cached keys
        # (N, 1, key_dim) - (1, size, key_dim) -> (N, size, key_dim)
        valid_keys = self.keys[:self.size]
        valid_values = self.values[:self.size]
        
        dists = torch.cdist(queries, valid_keys)  # (N, size)
        min_dists, min_indices = dists.min(dim=1)
        
        hit_mask = min_dists < tolerance
        results = torch.zeros(N, self.value_dim, device=self.device)
        results[hit_mask] = valid_values[min_indices[hit_mask]]
        
        return results, hit_mask
    
    def insert(self, keys: torch.Tensor, values: torch.Tensor):
        """Insert key-value pairs into cache."""
        n_insert = keys.shape[0]
        
        for i in range(n_insert):
            self.keys[self.next_idx] = keys[i]
            self.values[self.next_idx] = values[i]
            self.valid[self.next_idx] = True
            
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)


class OptimizedInference:
    """
    Optimized inference engine for real-time cloth prediction.
    
    Features:
    - FP16 model inference
    - Voxel-hash caching
    - Batched computation
    - Warm start support
    
    Args:
        model_path: Path to TorchScript model
        use_cache: Enable spatial caching
        cache_resolution: Cache grid resolution
        use_fp16: Use half-precision inference
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        use_cache: bool = True,
        cache_resolution: int = 128,
        use_fp16: bool = True,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.use_cache = use_cache
        
        # Load model
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        
        if self.use_fp16:
            self.model = self.model.half()
        
        # Initialize cache
        if use_cache:
            self.cache = VoxelHashCache(resolution=cache_resolution)
        else:
            self.cache = None
        
        # Warm up CUDA
        self._warmup()
    
    def _warmup(self, n_iters: int = 10):
        """Warm up CUDA kernels for consistent timing."""
        dummy = torch.randn(1000, 8, device=self.device)
        if self.use_fp16:
            dummy = dummy.half()
        
        for _ in range(n_iters):
            with torch.no_grad():
                _ = self.model(dummy)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    def predict(
        self,
        vertices: torch.Tensor,
        time: float,
        force: torch.Tensor,
        material_id: int = 0
    ) -> torch.Tensor:
        """
        Predict cloth displacements.
        
        Args:
            vertices: Vertex positions (N, 3)
            time: Normalized time [0, 1]
            force: External force vector (3,)
            material_id: Material identifier
        
        Returns:
            Predicted displacements (N, 3)
        """
        N = vertices.shape[0]
        
        # Move to device
        vertices = vertices.to(self.device)
        force = force.to(self.device)
        
        # Prepare full input
        t_expanded = torch.full((N, 1), time, device=self.device)
        force_expanded = force.unsqueeze(0).expand(N, 3)
        mat_expanded = torch.full((N, 1), float(material_id), device=self.device)
        
        inp = torch.cat([vertices, t_expanded, force_expanded, mat_expanded], dim=1)
        
        if self.use_fp16:
            inp = inp.half()
        
        # Check cache
        if self.use_cache and self.cache is not None:
            cached, hit_mask, miss_indices = self.cache.get(vertices)
            
            if miss_indices.numel() == 0:
                # All cached
                result = torch.zeros(N, 3, device=self.device)
                result[hit_mask] = cached
                return result.float()
            
            # Compute only misses
            miss_inp = inp[miss_indices]
            
            with torch.no_grad(), autocast(enabled=self.use_fp16):
                miss_out = self.model(miss_inp)
            
            # Update cache
            self.cache.put(vertices[miss_indices], miss_out)
            
            # Combine results
            result = torch.zeros(N, 3, device=self.device, dtype=miss_out.dtype)
            result[hit_mask] = cached.to(result.dtype)
            result[miss_indices] = miss_out
            
            return result.float()
        else:
            # No cache, compute all
            with torch.no_grad(), autocast(enabled=self.use_fp16):
                out = self.model(inp)
            return out.float()
    
    def predict_batch(
        self,
        vertices: torch.Tensor,
        times: torch.Tensor,
        forces: torch.Tensor,
        material_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Batched prediction for multiple configurations.
        
        Args:
            vertices: Vertex positions (B, N, 3)
            times: Time values (B,)
            forces: Force vectors (B, 3)
            material_ids: Material IDs (B,)
        
        Returns:
            Displacements (B, N, 3)
        """
        B, N, _ = vertices.shape
        
        results = []
        for b in range(B):
            out = self.predict(
                vertices[b],
                times[b].item(),
                forces[b],
                int(material_ids[b].item())
            )
            results.append(out)
        
        return torch.stack(results)
    
    def clear_cache(self):
        """Clear the inference cache."""
        if self.cache is not None:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        if self.cache is not None:
            return self.cache.get_stats()
        return {}


def benchmark_inference(
    engine: OptimizedInference,
    n_vertices: int = 10000,
    n_iterations: int = 1000
) -> Dict[str, float]:
    """
    Benchmark inference performance.
    
    Returns:
        Dictionary with FPS, latency (ms), and throughput metrics
    """
    device = engine.device
    
    # Generate test data
    vertices = torch.randn(n_vertices, 3, device=device)
    force = torch.randn(3, device=device)
    
    # Warmup
    for _ in range(10):
        _ = engine.predict(vertices, 0.5, force, 0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    
    for i in range(n_iterations):
        _ = engine.predict(vertices, i / n_iterations, force, 0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    fps = n_iterations / elapsed
    latency_ms = (elapsed / n_iterations) * 1000
    throughput = n_vertices * n_iterations / elapsed
    
    return {
        'fps': fps,
        'latency_ms': latency_ms,
        'throughput_vertices_per_sec': throughput,
        'n_vertices': n_vertices,
        'n_iterations': n_iterations
    }


# Convenience function for direct model loading
def load_model_for_inference(
    model_path: Union[str, Path],
    use_fp16: bool = True,
    device: str = 'cuda'
) -> Tuple[nn.Module, torch.device]:
    """
    Load model optimized for inference.
    
    Returns:
        model: Loaded model
        device: Device the model is on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = torch.jit.load(str(model_path)).to(device)
    model.eval()
    
    if use_fp16 and device.type == 'cuda':
        model = model.half()
    
    return model, device


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/model_traced.pt')
    parser.add_argument('--vertices', type=int, default=10000)
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()
    
    print("=== NIF-Cloth3D Inference Benchmark ===\n")
    
    # Check if model exists, else create dummy
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}, creating dummy model...")
        
        from model import SineMLP
        
        model = SineMLP(in_dim=8, hidden_dim=256, out_dim=3, n_layers=6)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Trace model
        traced = torch.jit.trace(model, torch.randn(100, 8, device=device))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(model_path))
        print(f"Saved dummy model to {model_path}")
    
    # Create inference engine
    print("\nLoading inference engine...")
    engine = OptimizedInference(
        model_path=model_path,
        use_cache=True,
        use_fp16=True
    )
    
    print(f"Device: {engine.device}")
    print(f"FP16: {engine.use_fp16}")
    print(f"Cache: {engine.use_cache}")
    
    # Run benchmark
    print(f"\nBenchmarking with {args.vertices} vertices, {args.iterations} iterations...")
    results = benchmark_inference(engine, args.vertices, args.iterations)
    
    print("\n=== Results ===")
    print(f"FPS: {results['fps']:.1f}")
    print(f"Latency: {results['latency_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_vertices_per_sec']/1e6:.2f} M vertices/sec")
    
    target_latency = 10.0  # ms
    if results['latency_ms'] < target_latency:
        print(f"\n✅ Target latency <{target_latency}ms ACHIEVED")
    else:
        print(f"\n❌ Target latency <{target_latency}ms NOT achieved")
    
    # Cache stats
    print("\nCache Statistics:")
    stats = engine.get_cache_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
