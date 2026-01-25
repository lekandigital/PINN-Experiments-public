"""
ONNX Model Converter
Converts TensorFlow/Keras PINN models to ONNX format for edge deployment.

Usage:
    python onnx_converter.py --input models/student/compressed_pinn.keras --output models/onnx/student.onnx
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import tf2onnx for conversion
try:
    import tf2onnx
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort
except ImportError:
    print("Required packages not found. Install with:")
    print("  pip install tf2onnx onnx onnxruntime")
    sys.exit(1)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.baseline_pinn import NavierStokesPINN


def convert_keras_to_onnx(
    keras_model_path: str,
    onnx_output_path: str,
    opset_version: int = 13,
    optimize: bool = True,
    simplify: bool = True
) -> Tuple[str, dict]:
    """
    Convert Keras model to ONNX format.
    
    Args:
        keras_model_path: Path to Keras model (.keras or .h5)
        onnx_output_path: Output path for ONNX model
        opset_version: ONNX opset version
        optimize: Apply ONNX graph optimizations
        simplify: Simplify ONNX graph
    
    Returns:
        Tuple of (output_path, metadata_dict)
    """
    print(f"\nLoading Keras model from {keras_model_path}...")
    
    # Load model with custom objects
    model = keras.models.load_model(
        keras_model_path,
        custom_objects={'NavierStokesPINN': NavierStokesPINN}
    )
    
    # Get model info
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output_shape}")
    
    # Define input signature for conversion
    # Input: [batch, 3] = [x, y, aoa]
    input_signature = [
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name='input')
    ]
    
    # Convert to ONNX
    print(f"\nConverting to ONNX (opset {opset_version})...")
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset_version,
        output_path=onnx_output_path
    )
    
    print(f"  Initial conversion complete")
    
    # Load and optimize
    if optimize or simplify:
        print(f"\nOptimizing ONNX model...")
        onnx_model = onnx.load(onnx_output_path)
        
        # Check model validity
        onnx.checker.check_model(onnx_model)
        
        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                onnx_model, check = onnx_simplify(onnx_model)
                if check:
                    print("  ONNX simplification successful")
                else:
                    print("  ONNX simplification failed, using original")
            except ImportError:
                print("  onnx-simplifier not installed, skipping simplification")
        
        # Save optimized model
        onnx.save(onnx_model, onnx_output_path)
    
    # Get model size
    model_size_mb = Path(onnx_output_path).stat().st_size / (1024 * 1024)
    
    # Test inference
    print(f"\nVerifying ONNX model...")
    sess = ort.InferenceSession(onnx_output_path)
    
    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Test with dummy input
    dummy_input = np.random.randn(1, 3).astype(np.float32)
    
    # Keras prediction
    keras_output = model.predict(dummy_input, verbose=0)
    
    # ONNX prediction
    onnx_output = sess.run([output_name], {input_name: dummy_input})[0]
    
    # Compare
    max_diff = np.max(np.abs(keras_output - onnx_output))
    print(f"  Max difference (Keras vs ONNX): {max_diff:.2e}")
    
    if max_diff > 1e-5:
        print(f"  WARNING: Large difference detected!")
    else:
        print(f"  ✓ Verification passed")
    
    # Metadata
    metadata = {
        'input_name': input_name,
        'output_name': output_name,
        'input_shape': list(sess.get_inputs()[0].shape),
        'output_shape': list(sess.get_outputs()[0].shape),
        'opset_version': opset_version,
        'model_size_mb': float(model_size_mb),
        'max_verification_diff': float(max_diff)
    }
    
    print(f"\n✓ ONNX model saved to {onnx_output_path}")
    print(f"  Size: {model_size_mb:.2f} MB")
    
    return onnx_output_path, metadata


def benchmark_onnx_inference(
    onnx_path: str,
    n_iterations: int = 1000,
    batch_size: int = 1
) -> dict:
    """
    Benchmark ONNX inference performance.
    
    Args:
        onnx_path: Path to ONNX model
        n_iterations: Number of inference iterations
        batch_size: Batch size for inference
    
    Returns:
        Benchmark results dictionary
    """
    import time
    
    print(f"\nBenchmarking ONNX inference ({n_iterations} iterations)...")
    
    # Create session with optimizations
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = 1  # Single-threaded for edge deployment
    
    # Try different execution providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=providers
    )
    
    active_provider = sess.get_providers()[0]
    print(f"  Using provider: {active_provider}")
    
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Generate random inputs
    inputs = np.random.randn(batch_size, 3).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        sess.run([output_name], {input_name: inputs})
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        sess.run([output_name], {input_name: inputs})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    results = {
        'provider': active_provider,
        'batch_size': batch_size,
        'n_iterations': n_iterations,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }
    
    print(f"\n  Results:")
    print(f"    Mean: {results['mean_ms']:.3f} ms")
    print(f"    Std:  {results['std_ms']:.3f} ms")
    print(f"    Min:  {results['min_ms']:.3f} ms")
    print(f"    P95:  {results['p95_ms']:.3f} ms")
    print(f"    P99:  {results['p99_ms']:.3f} ms")
    
    if results['mean_ms'] < 1.0:
        print(f"  ✓ Sub-millisecond inference achieved!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert Keras PINN to ONNX format'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to Keras model (.keras or .h5)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--opset', type=int, default=13,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--no_optimize', action='store_true',
        help='Disable ONNX optimizations'
    )
    parser.add_argument(
        '--no_simplify', action='store_true',
        help='Disable ONNX simplification'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run inference benchmark after conversion'
    )
    parser.add_argument(
        '--benchmark_iterations', type=int, default=1000,
        help='Number of benchmark iterations'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert model
    onnx_path, metadata = convert_keras_to_onnx(
        keras_model_path=args.input,
        onnx_output_path=args.output,
        opset_version=args.opset,
        optimize=not args.no_optimize,
        simplify=not args.no_simplify
    )
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark_onnx_inference(
            onnx_path=onnx_path,
            n_iterations=args.benchmark_iterations
        )
        metadata['benchmark'] = benchmark_results
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to {metadata_path}")


if __name__ == '__main__':
    main()
