#!/bin/bash
#
# NIF-Cloth3D-Interactive: Benchmark Script
# Runs performance benchmarks and reports results
#

set -e

# Default values
MODEL="${MODEL:-checkpoints/model_traced.pt}"
VERTICES="${VERTICES:-10000}"
ITERATIONS="${ITERATIONS:-1000}"

echo "================================================="
echo "NIF-Cloth3D-Interactive Performance Benchmark"
echo "================================================="

# Check for CUDA
echo ""
echo "System Information:"
python3 -c "
import torch
import platform

print(f'  Python: {platform.python_version()}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU Memory: {mem:.1f} GB')
"

# Run benchmark
echo ""
echo "Running benchmark..."
echo "  Model: $MODEL"
echo "  Vertices: $VERTICES"
echo "  Iterations: $ITERATIONS"
echo ""

python3 -c "
import sys
sys.path.insert(0, 'src')

from inference import OptimizedInference, benchmark_inference

# Create engine
print('Loading model...')
try:
    engine = OptimizedInference(
        model_path='$MODEL',
        use_cache=True,
        use_fp16=True
    )
except Exception as e:
    print(f'Warning: Could not load model: {e}')
    print('Creating fresh model for benchmarking...')
    
    import torch
    from model import SineMLP
    
    model = SineMLP(in_dim=8, hidden_dim=256, out_dim=3, n_layers=6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    traced = torch.jit.trace(model, torch.randn(100, 8, device=device))
    traced.save('/tmp/bench_model.pt')
    
    engine = OptimizedInference(
        model_path='/tmp/bench_model.pt',
        use_cache=True,
        use_fp16=True
    )

# Run benchmark
print('')
print('=' * 50)
print('Benchmark Results')
print('=' * 50)

results = benchmark_inference(
    engine, 
    n_vertices=$VERTICES, 
    n_iterations=$ITERATIONS
)

print(f'')
print(f'  Latency: {results[\"latency_ms\"]:.2f} ms')
print(f'  FPS: {results[\"fps\"]:.1f}')
print(f'  Throughput: {results[\"throughput_vertices_per_sec\"]/1e6:.2f} M vertices/sec')
print(f'')

# Check targets
print('Target Compliance:')
if results['latency_ms'] < 10:
    print(f'  ✓ Latency < 10ms: PASS ({results[\"latency_ms\"]:.2f} ms)')
else:
    print(f'  ✗ Latency < 10ms: FAIL ({results[\"latency_ms\"]:.2f} ms)')

if results['fps'] > 60:
    print(f'  ✓ FPS > 60: PASS ({results[\"fps\"]:.1f} fps)')
else:
    print(f'  ✗ FPS > 60: FAIL ({results[\"fps\"]:.1f} fps)')

print('')
print('=' * 50)
"

echo ""
echo "Benchmark complete!"
