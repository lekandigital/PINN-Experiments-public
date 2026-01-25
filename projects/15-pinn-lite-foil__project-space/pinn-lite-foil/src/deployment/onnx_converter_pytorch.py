"""
ONNX Model Converter for PyTorch PINN Models
Converts PyTorch PINN models to ONNX format for edge deployment.

Usage:
    python onnx_converter_pytorch.py --input models/student/compressed_pinn.pt --config models/student/distillation_config.json --output models/onnx/student.onnx
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from baseline_pinn_pytorch import NavierStokesPINN

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Required packages not found. Install with:")
    print("  pip install onnx onnxruntime")
    sys.exit(1)


def convert_pytorch_to_onnx(
    model_path: str,
    config_path: str,
    onnx_output_path: str,
    opset_version: int = 13
) -> Tuple[str, dict]:
    """
    Convert PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch model (.pt)
        config_path: Path to model config JSON
        onnx_output_path: Output path for ONNX model
        opset_version: ONNX opset version

    Returns:
        Tuple of (output_path, metadata_dict)
    """
    print(f"\nLoading PyTorch model from {model_path}...")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Determine model architecture from config
    if 'student_layers' in config:
        hidden_layers = config['student_layers']
        hidden_units = config['student_units']
    else:
        hidden_layers = config['hidden_layers']
        hidden_units = config['hidden_units']

    nu = config.get('nu', 1e-3)

    # Create and load model
    model = NavierStokesPINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        nu=nu
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print(f"  Architecture: {hidden_layers} layers x {hidden_units} units")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create output directory
    Path(onnx_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, dtype=torch.float32)

    print(f"\nConverting to ONNX (opset {opset_version})...")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"  ONNX export complete")

    # Verify ONNX model
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Model validation passed")

    # Test inference consistency
    sess = ort.InferenceSession(onnx_output_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Test with dummy input
    test_input = np.random.randn(10, 3).astype(np.float32)

    # PyTorch prediction
    with torch.no_grad():
        pytorch_output = model(torch.tensor(test_input)).numpy()

    # ONNX prediction
    onnx_output = sess.run([output_name], {input_name: test_input})[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"  Max difference (PyTorch vs ONNX): {max_diff:.2e}")

    if max_diff > 1e-5:
        print(f"  WARNING: Large difference detected!")
    else:
        print(f"  Verification passed!")

    # Get model size
    model_size_kb = Path(onnx_output_path).stat().st_size / 1024

    # Metadata
    metadata = {
        'input_name': input_name,
        'output_name': output_name,
        'input_shape': list(sess.get_inputs()[0].shape),
        'output_shape': list(sess.get_outputs()[0].shape),
        'opset_version': opset_version,
        'model_size_kb': float(model_size_kb),
        'hidden_layers': hidden_layers,
        'hidden_units': hidden_units,
        'max_verification_diff': float(max_diff)
    }

    print(f"\nONNX model saved to {onnx_output_path}")
    print(f"  Size: {model_size_kb:.1f} KB")

    return onnx_output_path, metadata


def benchmark_onnx_inference(
    onnx_path: str,
    n_iterations: int = 1000,
    batch_sizes: list = [1, 4, 8, 16],
    use_gpu: bool = True
) -> dict:
    """Benchmark ONNX inference performance."""
    import time

    print(f"\nBenchmarking ONNX inference...")

    # Session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # Providers
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

    active_provider = sess.get_providers()[0]
    print(f"  Provider: {active_provider}")

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    results = {'provider': active_provider, 'benchmarks': {}}

    for batch_size in batch_sizes:
        inputs = np.random.randn(batch_size, 3).astype(np.float32)

        # Warmup
        for _ in range(50):
            sess.run([output_name], {input_name: inputs})

        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            sess.run([output_name], {input_name: inputs})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        times = np.array(times)

        batch_results = {
            'batch_size': batch_size,
            'n_iterations': n_iterations,
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_samples_per_sec': float(batch_size * 1000 / np.mean(times))
        }

        results['benchmarks'][f'batch_{batch_size}'] = batch_results

        print(f"\n  Batch size {batch_size}:")
        print(f"    Mean: {batch_results['mean_ms']:.3f} ms")
        print(f"    P99:  {batch_results['p99_ms']:.3f} ms")
        print(f"    Throughput: {batch_results['throughput_samples_per_sec']:.0f} samples/sec")

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch PINN to ONNX')
    parser.add_argument('--input', type=str, required=True, help='Path to PyTorch model')
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--benchmark_iterations', type=int, default=1000, help='Benchmark iterations')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for benchmarks')

    args = parser.parse_args()

    # Convert model
    onnx_path, metadata = convert_pytorch_to_onnx(
        model_path=args.input,
        config_path=args.config,
        onnx_output_path=args.output,
        opset_version=args.opset
    )

    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark_onnx_inference(
            onnx_path=onnx_path,
            n_iterations=args.benchmark_iterations,
            use_gpu=args.gpu
        )
        metadata['benchmark'] = benchmark_results

    # Save metadata
    metadata_path = Path(args.output).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")


if __name__ == '__main__':
    main()
