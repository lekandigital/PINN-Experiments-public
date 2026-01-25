"""
Accuracy Test for PINN-Lite-Foil (Flat HDF5 Format)
Validates model predictions against ground truth data.

Usage:
    python accuracy_test_flat.py --model models/onnx/student.onnx --test_data data/processed/test_data.h5 --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)


def load_test_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load test data from flat HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
        aoa = f['aoa'][:]
        u = f['u'][:]
        v = f['v'][:]
        p = f['p'][:]

        metadata = {
            'n_samples': f.attrs.get('n_samples', len(x)),
            'naca_codes': f.attrs.get('naca_codes', 'unknown'),
            'aoa_min': f.attrs.get('aoa_min', aoa.min()),
            'aoa_max': f.attrs.get('aoa_max', aoa.max())
        }

    # Create input/target tensors
    inputs = np.stack([x, y, aoa], axis=-1).astype(np.float32)
    targets = np.stack([u, v, p], axis=-1).astype(np.float32)

    return inputs, targets, metadata


def run_accuracy_test(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    normalization_stats_path: str = None,
    use_gpu: bool = False
) -> Dict:
    """
    Run comprehensive accuracy test.

    Args:
        model_path: Path to ONNX model
        test_data_path: Path to HDF5 test data
        output_dir: Output directory for results
        normalization_stats_path: Optional path to normalization stats
        use_gpu: Use CUDA provider if available

    Returns:
        Dictionary of test results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("PINN-Lite-Foil Accuracy Test")
    print("="*60)

    # Load model
    print(f"\nLoading model: {model_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"  Provider: {session.get_providers()[0]}")

    # Load test data
    print(f"\nLoading test data: {test_data_path}")
    inputs, targets, metadata = load_test_data(test_data_path)
    print(f"  Samples: {len(inputs):,}")
    print(f"  AoA range: [{metadata['aoa_min']:.1f}, {metadata['aoa_max']:.1f}] degrees")

    # Load normalization stats if provided
    if normalization_stats_path and Path(normalization_stats_path).exists():
        with open(normalization_stats_path, 'r') as f:
            stats = json.load(f)

        # Normalize inputs
        input_mean = np.array(stats['input_mean'])
        input_std = np.array(stats['input_std'])
        inputs_normalized = (inputs - input_mean) / input_std

        # For denormalizing predictions
        target_mean = np.array(stats['target_mean'])
        target_std = np.array(stats['target_std'])
    else:
        # Simple normalization
        input_mean = inputs.mean(axis=0)
        input_std = inputs.std(axis=0) + 1e-8
        inputs_normalized = (inputs - input_mean) / input_std

        target_mean = targets.mean(axis=0)
        target_std = targets.std(axis=0) + 1e-8

    # Ensure float32 for ONNX
    inputs_normalized = inputs_normalized.astype(np.float32)

    # Run predictions in batches
    print("\nRunning predictions...")
    batch_size = 10000
    all_predictions = []

    for i in range(0, len(inputs_normalized), batch_size):
        batch = inputs_normalized[i:i+batch_size].astype(np.float32)
        preds = session.run([output_name], {input_name: batch})[0]
        all_predictions.append(preds)

        if (i + batch_size) % 50000 == 0 or i + batch_size >= len(inputs_normalized):
            print(f"  Processed {min(i + batch_size, len(inputs_normalized)):,}/{len(inputs_normalized):,}")

    predictions = np.concatenate(all_predictions, axis=0)

    # Denormalize predictions
    predictions_denorm = predictions * target_std + target_mean

    # Compute errors
    u_pred, v_pred, p_pred = predictions_denorm[:, 0], predictions_denorm[:, 1], predictions_denorm[:, 2]
    u_true, v_true, p_true = targets[:, 0], targets[:, 1], targets[:, 2]

    # Field errors
    u_mse = float(np.mean((u_pred - u_true) ** 2))
    v_mse = float(np.mean((v_pred - v_true) ** 2))
    p_mse = float(np.mean((p_pred - p_true) ** 2))

    u_mae = float(np.mean(np.abs(u_pred - u_true)))
    v_mae = float(np.mean(np.abs(v_pred - v_true)))
    p_mae = float(np.mean(np.abs(p_pred - p_true)))

    # Relative errors
    u_rel = float(np.mean(np.abs(u_pred - u_true) / (np.abs(u_true) + 1e-8)))
    v_rel = float(np.mean(np.abs(v_pred - v_true) / (np.abs(v_true) + 1e-8)))
    p_rel = float(np.mean(np.abs(p_pred - p_true) / (np.abs(p_true) + 1e-8)))

    # RMSE
    u_rmse = float(np.sqrt(u_mse))
    v_rmse = float(np.sqrt(v_mse))
    p_rmse = float(np.sqrt(p_mse))

    # Per-AoA analysis
    unique_aoa = np.unique(inputs[:, 2])
    aoa_results = []

    for aoa_val in unique_aoa:
        mask = inputs[:, 2] == aoa_val
        aoa_u_mae = float(np.mean(np.abs(u_pred[mask] - u_true[mask])))
        aoa_v_mae = float(np.mean(np.abs(v_pred[mask] - v_true[mask])))
        aoa_p_mae = float(np.mean(np.abs(p_pred[mask] - p_true[mask])))

        aoa_results.append({
            'aoa': float(aoa_val),
            'u_mae': aoa_u_mae,
            'v_mae': aoa_v_mae,
            'p_mae': aoa_p_mae,
            'n_samples': int(np.sum(mask))
        })

    # Results summary
    results = {
        'summary': {
            'n_samples': int(len(inputs)),
            'u_mse': u_mse, 'v_mse': v_mse, 'p_mse': p_mse,
            'u_mae': u_mae, 'v_mae': v_mae, 'p_mae': p_mae,
            'u_rmse': u_rmse, 'v_rmse': v_rmse, 'p_rmse': p_rmse,
            'u_relative_error': u_rel,
            'v_relative_error': v_rel,
            'p_relative_error': p_rel,
            'pressure_rmse': p_rmse
        },
        'per_aoa': aoa_results,
        'metadata': metadata
    }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nField Errors (RMSE):")
    print(f"  u-velocity: {u_rmse:.6f}")
    print(f"  v-velocity: {v_rmse:.6f}")
    print(f"  pressure:   {p_rmse:.6f}")

    print("\nField Errors (MAE):")
    print(f"  u-velocity: {u_mae:.6f}")
    print(f"  v-velocity: {v_mae:.6f}")
    print(f"  pressure:   {p_mae:.6f}")

    print("\nRelative Errors:")
    print(f"  u-velocity: {u_rel*100:.2f}%")
    print(f"  v-velocity: {v_rel*100:.2f}%")
    print(f"  pressure:   {p_rel*100:.2f}%")

    # Generate plots
    generate_accuracy_plots(results, inputs, predictions_denorm, targets, output_dir)

    # Save results
    with open(output_dir / 'accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return results


def generate_accuracy_plots(
    results: Dict,
    inputs: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Path
) -> None:
    """Generate accuracy visualization plots."""

    # 1. Prediction vs True scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sample for visualization
    n_sample = min(10000, len(inputs))
    idx = np.random.choice(len(inputs), n_sample, replace=False)

    for i, (name, ax) in enumerate(zip(['u', 'v', 'p'], axes)):
        pred_vals = predictions[idx, i]
        true_vals = targets[idx, i]

        ax.scatter(true_vals, pred_vals, alpha=0.3, s=1)
        ax.plot([true_vals.min(), true_vals.max()],
                [true_vals.min(), true_vals.max()], 'r--', lw=2)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}-component Prediction vs Truth')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_vs_true.png', dpi=150)
    plt.close()

    # 2. Error vs AoA
    aoa_results = results['per_aoa']
    aoa_values = [r['aoa'] for r in aoa_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(aoa_values, [r['u_mae'] for r in aoa_results], 'b-o')
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('u-velocity MAE')
    axes[0].set_title('u-Velocity Error vs AoA')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(aoa_values, [r['v_mae'] for r in aoa_results], 'g-o')
    axes[1].set_xlabel('Angle of Attack (deg)')
    axes[1].set_ylabel('v-velocity MAE')
    axes[1].set_title('v-Velocity Error vs AoA')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(aoa_values, [r['p_mae'] for r in aoa_results], 'r-o')
    axes[2].set_xlabel('Angle of Attack (deg)')
    axes[2].set_ylabel('pressure MAE')
    axes[2].set_title('Pressure Error vs AoA')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_aoa.png', dpi=150)
    plt.close()

    # 3. Error histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    u_errors = np.abs(predictions[:, 0] - targets[:, 0])
    v_errors = np.abs(predictions[:, 1] - targets[:, 1])
    p_errors = np.abs(predictions[:, 2] - targets[:, 2])

    axes[0].hist(u_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('u-velocity Absolute Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title('u-Velocity Error Distribution')
    axes[0].axvline(np.mean(u_errors), color='r', linestyle='--', label=f'Mean: {np.mean(u_errors):.4f}')
    axes[0].legend()

    axes[1].hist(v_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('v-velocity Absolute Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('v-Velocity Error Distribution')
    axes[1].axvline(np.mean(v_errors), color='r', linestyle='--', label=f'Mean: {np.mean(v_errors):.4f}')
    axes[1].legend()

    axes[2].hist(p_errors, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[2].set_xlabel('pressure Absolute Error')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Pressure Error Distribution')
    axes[2].axvline(np.mean(p_errors), color='r', linestyle='--', label=f'Mean: {np.mean(p_errors):.4f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'error_histograms.png', dpi=150)
    plt.close()

    print(f"  Generated plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='PINN accuracy test (flat HDF5)')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    parser.add_argument('--norm_stats', type=str, help='Normalization stats JSON')
    parser.add_argument('--gpu', action='store_true', help='Use CUDA')

    args = parser.parse_args()

    run_accuracy_test(
        model_path=args.model,
        test_data_path=args.test_data,
        output_dir=args.output,
        normalization_stats_path=args.norm_stats,
        use_gpu=args.gpu
    )


if __name__ == '__main__':
    main()
