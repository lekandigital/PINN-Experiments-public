"""
Accuracy Test for PINN-Lite-Foil
Validates model predictions against CFD ground truth data.

Metrics:
    - Lift coefficient MAE
    - Drag coefficient MAE
    - Pressure field MSE
    - Velocity field MSE

Usage:
    python accuracy_test.py --model models/onnx/student.onnx --test_data data/processed/test.h5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)


def load_test_data(
    filepath: str,
    split: str = 'test'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load test data from HDF5 file.
    
    Returns:
        Tuple of (inputs, targets, metadata, coefficients)
    """
    all_inputs = []
    all_targets = []
    metadata = []
    
    with h5py.File(filepath, 'r') as f:
        grp = f[split]
        
        for case_name in grp.keys():
            case = grp[case_name]
            
            x = case['x'][:]
            y = case['y'][:]
            u = case['u'][:]
            v = case['v'][:]
            p = case['p'][:]
            aoa = case.attrs['aoa']
            
            # Create input tensor [x, y, aoa]
            aoa_arr = np.full_like(x, aoa)
            inputs = np.stack([x, y, aoa_arr], axis=-1).astype(np.float32)
            targets = np.stack([u, v, p], axis=-1).astype(np.float32)
            
            all_inputs.append(inputs)
            all_targets.append(targets)
            
            case_meta = {
                'name': case_name,
                'aoa': float(aoa),
                'n_points': len(x)
            }
            
            if 'cl' in case.attrs:
                case_meta['cl_cfd'] = float(case.attrs['cl'])
            if 'cd' in case.attrs:
                case_meta['cd_cfd'] = float(case.attrs['cd'])
            if 'reynolds' in case.attrs:
                case_meta['reynolds'] = float(case.attrs['reynolds'])
            
            metadata.append(case_meta)
    
    return all_inputs, all_targets, metadata


def compute_lift_drag(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    aoa_deg: float,
    chord: float = 1.0,
    rho: float = 1.0,
    u_inf: float = 10.0
) -> Tuple[float, float]:
    """
    Compute lift and drag coefficients from pressure distribution.
    
    Simplified integration assuming surface pressure distribution.
    
    Args:
        x, y: Coordinates
        p: Pressure values
        aoa_deg: Angle of attack in degrees
        chord: Chord length
        rho: Air density
        u_inf: Freestream velocity
    
    Returns:
        Tuple of (Cl, Cd)
    """
    aoa_rad = np.radians(aoa_deg)
    
    # Dynamic pressure
    q = 0.5 * rho * u_inf**2
    
    # Sort points by x to separate upper/lower surfaces
    # This is a simplified approach - real implementation would need proper surface detection
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    p_sorted = p[sorted_idx]
    
    # Find leading edge (minimum x)
    le_idx = np.argmin(x_sorted)
    
    # Split into upper and lower surfaces based on y
    # Points with y > 0 are upper surface, y < 0 are lower
    upper_mask = y_sorted >= 0
    lower_mask = y_sorted < 0
    
    # Compute pressure coefficient
    cp = p_sorted / q
    
    # Integrate pressure forces (simplified trapezoidal)
    # Normal force coefficient
    cn = 0.0
    ca = 0.0
    
    if np.sum(upper_mask) > 1:
        x_upper = x_sorted[upper_mask]
        cp_upper = cp[upper_mask]
        sort_idx = np.argsort(x_upper)
        cn -= np.trapz(cp_upper[sort_idx], x_upper[sort_idx]) / chord
    
    if np.sum(lower_mask) > 1:
        x_lower = x_sorted[lower_mask]
        cp_lower = cp[lower_mask]
        sort_idx = np.argsort(x_lower)
        cn += np.trapz(cp_lower[sort_idx], x_lower[sort_idx]) / chord
    
    # Convert to lift and drag
    cl = cn * np.cos(aoa_rad) - ca * np.sin(aoa_rad)
    cd = cn * np.sin(aoa_rad) + ca * np.cos(aoa_rad)
    
    # Add friction drag estimate (flat plate approximation)
    cd += 0.01  # Simple skin friction estimate
    
    return float(cl), float(cd)


def run_accuracy_test(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    use_gpu: bool = False
) -> Dict:
    """
    Run comprehensive accuracy test.
    
    Args:
        model_path: Path to ONNX model
        test_data_path: Path to HDF5 test data
        output_dir: Output directory for results
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
    all_inputs, all_targets, metadata = load_test_data(test_data_path)
    
    n_cases = len(all_inputs)
    print(f"  Test cases: {n_cases}")
    
    # Run predictions and compute metrics
    results = {
        'cases': [],
        'field_metrics': {
            'u_mse': [], 'v_mse': [], 'p_mse': [],
            'u_mae': [], 'v_mae': [], 'p_mae': []
        },
        'coefficient_metrics': {
            'cl_error': [], 'cd_error': [],
            'cl_mae': 0.0, 'cd_mae': 0.0
        }
    }
    
    print("\nRunning predictions...")
    
    for i, (inputs, targets, meta) in enumerate(zip(all_inputs, all_targets, metadata)):
        # Run inference
        predictions = session.run([output_name], {input_name: inputs})[0]
        
        u_pred, v_pred, p_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        u_true, v_true, p_true = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Field errors
        u_mse = float(np.mean((u_pred - u_true)**2))
        v_mse = float(np.mean((v_pred - v_true)**2))
        p_mse = float(np.mean((p_pred - p_true)**2))
        
        u_mae = float(np.mean(np.abs(u_pred - u_true)))
        v_mae = float(np.mean(np.abs(v_pred - v_true)))
        p_mae = float(np.mean(np.abs(p_pred - p_true)))
        
        results['field_metrics']['u_mse'].append(u_mse)
        results['field_metrics']['v_mse'].append(v_mse)
        results['field_metrics']['p_mse'].append(p_mse)
        results['field_metrics']['u_mae'].append(u_mae)
        results['field_metrics']['v_mae'].append(v_mae)
        results['field_metrics']['p_mae'].append(p_mae)
        
        # Coefficient errors (if ground truth available)
        x, y = inputs[:, 0], inputs[:, 1]
        aoa = meta['aoa']
        
        cl_pred, cd_pred = compute_lift_drag(x, y, p_pred, aoa)
        
        case_result = {
            'name': meta['name'],
            'aoa': aoa,
            'u_mse': u_mse, 'v_mse': v_mse, 'p_mse': p_mse,
            'u_mae': u_mae, 'v_mae': v_mae, 'p_mae': p_mae,
            'cl_pred': cl_pred, 'cd_pred': cd_pred
        }
        
        if 'cl_cfd' in meta:
            cl_error = abs(cl_pred - meta['cl_cfd'])
            cd_error = abs(cd_pred - meta['cd_cfd']) if 'cd_cfd' in meta else 0
            
            case_result['cl_cfd'] = meta['cl_cfd']
            case_result['cl_error'] = cl_error
            results['coefficient_metrics']['cl_error'].append(cl_error)
            
            if 'cd_cfd' in meta:
                case_result['cd_cfd'] = meta['cd_cfd']
                case_result['cd_error'] = cd_error
                results['coefficient_metrics']['cd_error'].append(cd_error)
        
        results['cases'].append(case_result)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_cases} cases")
    
    # Compute summary statistics
    results['summary'] = {
        'n_cases': n_cases,
        'field_metrics': {
            'u_mse_mean': float(np.mean(results['field_metrics']['u_mse'])),
            'v_mse_mean': float(np.mean(results['field_metrics']['v_mse'])),
            'p_mse_mean': float(np.mean(results['field_metrics']['p_mse'])),
            'u_mae_mean': float(np.mean(results['field_metrics']['u_mae'])),
            'v_mae_mean': float(np.mean(results['field_metrics']['v_mae'])),
            'p_mae_mean': float(np.mean(results['field_metrics']['p_mae']))
        }
    }
    
    if results['coefficient_metrics']['cl_error']:
        results['summary']['cl_mae'] = float(np.mean(results['coefficient_metrics']['cl_error']))
        results['summary']['cd_mae'] = float(np.mean(results['coefficient_metrics']['cd_error']))
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nField Errors (Mean):")
    print(f"  u-velocity MSE: {results['summary']['field_metrics']['u_mse_mean']:.6f}")
    print(f"  v-velocity MSE: {results['summary']['field_metrics']['v_mse_mean']:.6f}")
    print(f"  pressure MSE:   {results['summary']['field_metrics']['p_mse_mean']:.6f}")
    
    print("\nField Errors (MAE):")
    print(f"  u-velocity MAE: {results['summary']['field_metrics']['u_mae_mean']:.6f}")
    print(f"  v-velocity MAE: {results['summary']['field_metrics']['v_mae_mean']:.6f}")
    print(f"  pressure MAE:   {results['summary']['field_metrics']['p_mae_mean']:.6f}")
    
    if 'cl_mae' in results['summary']:
        print("\nCoefficient Errors (MAE):")
        print(f"  Lift (Cl) MAE:  {results['summary']['cl_mae']:.4f}")
        print(f"  Drag (Cd) MAE:  {results['summary']['cd_mae']:.4f}")
        
        # Check targets
        if results['summary']['cl_mae'] < 0.05:
            print("  ✓ Lift MAE < 5% target")
        else:
            print("  ✗ Lift MAE exceeds 5% target")
        
        if results['summary']['cd_mae'] < 0.05:
            print("  ✓ Drag MAE < 5% target")
        else:
            print("  ✗ Drag MAE exceeds 5% target")
    
    # Generate plots
    generate_accuracy_plots(results, output_dir)
    
    # Save results
    with open(output_dir / 'accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    
    return results


def generate_accuracy_plots(results: Dict, output_dir: Path) -> None:
    """Generate accuracy visualization plots."""
    
    # Extract data
    aoa_values = [c['aoa'] for c in results['cases']]
    
    # 1. Field MSE vs AoA
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(aoa_values, results['field_metrics']['u_mse'], alpha=0.7)
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('u-velocity MSE')
    axes[0].set_title('u-Velocity Error vs AoA')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(aoa_values, results['field_metrics']['v_mse'], alpha=0.7, color='orange')
    axes[1].set_xlabel('Angle of Attack (deg)')
    axes[1].set_ylabel('v-velocity MSE')
    axes[1].set_title('v-Velocity Error vs AoA')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(aoa_values, results['field_metrics']['p_mse'], alpha=0.7, color='green')
    axes[2].set_xlabel('Angle of Attack (deg)')
    axes[2].set_ylabel('pressure MSE')
    axes[2].set_title('Pressure Error vs AoA')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'field_mse_vs_aoa.png', dpi=150)
    plt.close()
    
    # 2. Coefficient errors if available
    if results['coefficient_metrics']['cl_error']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].scatter(aoa_values, results['coefficient_metrics']['cl_error'], alpha=0.7)
        axes[0].axhline(y=0.05, color='r', linestyle='--', label='5% target')
        axes[0].set_xlabel('Angle of Attack (deg)')
        axes[0].set_ylabel('Lift Coefficient Error')
        axes[0].set_title('Lift (Cl) Error vs AoA')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(aoa_values, results['coefficient_metrics']['cd_error'], alpha=0.7, color='orange')
        axes[1].axhline(y=0.05, color='r', linestyle='--', label='5% target')
        axes[1].set_xlabel('Angle of Attack (deg)')
        axes[1].set_ylabel('Drag Coefficient Error')
        axes[1].set_title('Drag (Cd) Error vs AoA')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'coefficient_errors_vs_aoa.png', dpi=150)
        plt.close()
    
    # 3. Error histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(results['field_metrics']['u_mae'], bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('u-velocity MAE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('u-Velocity MAE Distribution')
    
    axes[1].hist(results['field_metrics']['v_mae'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('v-velocity MAE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('v-Velocity MAE Distribution')
    
    axes[2].hist(results['field_metrics']['p_mae'], bins=20, alpha=0.7, edgecolor='black', color='green')
    axes[2].set_xlabel('pressure MAE')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Pressure MAE Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_histograms.png', dpi=150)
    plt.close()
    
    print(f"  Generated plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='PINN-Lite-Foil accuracy test'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--test_data', type=str, required=True,
        help='Path to HDF5 test data'
    )
    parser.add_argument(
        '--output', type=str, default='results/accuracy/',
        help='Output directory'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use CUDA provider'
    )
    
    args = parser.parse_args()
    
    run_accuracy_test(
        model_path=args.model,
        test_data_path=args.test_data,
        output_dir=args.output,
        use_gpu=args.gpu
    )


if __name__ == '__main__':
    main()
