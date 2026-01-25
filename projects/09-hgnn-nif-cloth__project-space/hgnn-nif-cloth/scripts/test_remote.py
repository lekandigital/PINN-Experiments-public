"""
Remote Testing Script for Vast.ai Instance

Run tests and benchmarks on a deployed Vast.ai instance.

Usage:
    python scripts/test_remote.py --instance_id <ID>
    python scripts/test_remote.py --instance_id <ID> --benchmark
"""

import argparse
import subprocess
import sys
import json
import os
from typing import Optional, Tuple

# Set API key
VASTAI_API_KEY = "REDACTED_VASTAI_API_KEY"
os.environ['VASTAI_API_KEY'] = VASTAI_API_KEY


def get_ssh_info(instance_id: int) -> Tuple[Optional[str], Optional[int]]:
    """Get SSH connection info for instance."""
    result = subprocess.run(
        ['vastai', 'show', 'instance', str(instance_id), '--raw'],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return None, None
    
    try:
        data = json.loads(result.stdout)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        return data.get('ssh_host'), data.get('ssh_port')
    except (json.JSONDecodeError, KeyError):
        return None, None


def ssh_exec(ssh_host: str, ssh_port: int, command: str) -> int:
    """Execute command on remote instance via SSH."""
    cmd = [
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-p', str(ssh_port),
        f'root@{ssh_host}',
        command
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_validation_tests(ssh_host: str, ssh_port: int) -> bool:
    """Run model validation tests."""
    print("\n" + "=" * 60)
    print("Running Validation Tests")
    print("=" * 60)
    
    command = """
cd /workspace/hgnn-nif-cloth && \
python -c "
import torch
print('=== System Info ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Memory: {props.total_memory / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
" && \
echo "" && \
python -m pytest tests/test_forward.py -v
"""
    
    return ssh_exec(ssh_host, ssh_port, command) == 0


def run_benchmark(ssh_host: str, ssh_port: int) -> bool:
    """Run inference benchmark."""
    print("\n" + "=" * 60)
    print("Running Inference Benchmark")
    print("=" * 60)
    
    command = """
cd /workspace/hgnn-nif-cloth && \
python -c "
import torch
import time
import sys
sys.path.insert(0, '.')

from src.models.hybrid_model import HGNN_NIF_ClothModel

device = torch.device('cuda')
print(f'Device: {device}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Create model
model = HGNN_NIF_ClothModel(
    node_feat_dim=3,
    latent_dim=64,
    hidden_dim=64,
    siren_hidden_dim=128
)
model = model.to(device)
model.eval()

# Generate test data
fine_pos = torch.randn(1, 400, 3).to(device)
fine_edges = torch.randint(0, 400, (2, 300)).to(device)
coarse_pos = torch.randn(1, 100, 3).to(device)
coarse_edges = torch.randint(0, 100, (2, 80)).to(device)
query_points = torch.randn(1, 1000, 3).to(device)

print('\\nWarm-up...')
for _ in range(20):
    with torch.no_grad():
        _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)

print('Benchmarking (100 iterations)...')
torch.cuda.synchronize()
times = []

for _ in range(100):
    start = time.time()
    with torch.no_grad():
        output = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
    torch.cuda.synchronize()
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
fps = 1.0 / avg_time

print('\\n=== Results ===')
print(f'Average inference time: {avg_time * 1000:.2f} ms (± {std_time * 1000:.2f})')
print(f'FPS: {fps:.1f}')
print(f'Min time: {min(times) * 1000:.2f} ms')
print(f'Max time: {max(times) * 1000:.2f} ms')

# Memory usage
mem_allocated = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
print(f'\\nMemory allocated: {mem_allocated:.2f} GB')
print(f'Memory reserved: {mem_reserved:.2f} GB')

# Batch size test
print('\\n=== Batch Size Scaling ===')
for batch_size in [1, 2, 4, 8, 16]:
    try:
        fine_pos = torch.randn(batch_size, 400, 3).to(device)
        coarse_pos = torch.randn(batch_size, 100, 3).to(device)
        query_points = torch.randn(batch_size, 1000, 3).to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model((fine_pos, fine_edges), (coarse_pos, coarse_edges), query_points)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f'Batch {batch_size:2d}: {elapsed*1000:6.2f} ms, Memory: {mem:.2f} GB')
        
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError as e:
        print(f'Batch {batch_size:2d}: OOM')
        break
"
"""
    
    return ssh_exec(ssh_host, ssh_port, command) == 0


def run_training_test(ssh_host: str, ssh_port: int) -> bool:
    """Run quick training test."""
    print("\n" + "=" * 60)
    print("Running Training Test (3 epochs)")
    print("=" * 60)
    
    command = """
cd /workspace/hgnn-nif-cloth && \
python scripts/train_local.py --test_mode --output_dir outputs/remote_test
"""
    
    return ssh_exec(ssh_host, ssh_port, command) == 0


def main():
    parser = argparse.ArgumentParser(description='Test HGNN-NIF-Cloth on Vast.ai')
    parser.add_argument('--instance_id', type=int, required=True,
                        help='Vast.ai instance ID')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark')
    parser.add_argument('--train', action='store_true',
                        help='Run training test')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    
    args = parser.parse_args()
    
    print(f"Connecting to instance {args.instance_id}...")
    
    ssh_host, ssh_port = get_ssh_info(args.instance_id)
    
    if not ssh_host or not ssh_port:
        print(f"ERROR: Could not get SSH info for instance {args.instance_id}")
        print("Make sure the instance is running: vastai show instances")
        sys.exit(1)
    
    print(f"SSH: ssh -p {ssh_port} root@{ssh_host}")
    
    success = True
    
    # Validation tests
    if not run_validation_tests(ssh_host, ssh_port):
        print("\n✗ Validation tests failed")
        success = False
    else:
        print("\n✓ Validation tests passed")
    
    # Benchmark
    if args.benchmark or args.all:
        if not run_benchmark(ssh_host, ssh_port):
            print("\n✗ Benchmark failed")
            success = False
        else:
            print("\n✓ Benchmark complete")
    
    # Training test
    if args.train or args.all:
        if not run_training_test(ssh_host, ssh_port):
            print("\n✗ Training test failed")
            success = False
        else:
            print("\n✓ Training test passed")
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed ✗")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
