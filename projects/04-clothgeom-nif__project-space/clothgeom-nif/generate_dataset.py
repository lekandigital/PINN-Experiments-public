#!/usr/bin/env python3
"""
Generate Training Dataset for ClothGeom-NIF

Creates synthetic cloth samples with ground-truth SDF volumes
and saves to HDF5 format for training.

Usage:
    python generate_dataset.py --num_samples 100 --output data/generated/cloth_dataset.h5
    python generate_dataset.py --num_samples 10 --resolution 32  # Quick test
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_cloth_generator import generate_dataset, ClothConfig, SyntheticClothGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training dataset for ClothGeom-NIF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        default=100,
        help='Number of cloth samples to generate'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/generated/cloth_dataset.h5',
        help='Output HDF5 file path'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=64,
        help='SDF volume resolution (R×R×R)'
    )
    
    parser.add_argument(
        '--latent_dim', '-d',
        type=int,
        default=128,
        help='Latent code dimension'
    )
    
    parser.add_argument(
        '--grid_size', '-g',
        type=int,
        default=32,
        help='Cloth grid size (N×N nodes)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation on generated data'
    )
    
    return parser.parse_args()


def validate_dataset(path: str) -> bool:
    """
    Validate generated dataset.
    
    Checks:
    - File can be opened
    - Expected keys exist
    - Data has valid shapes
    - SDF values are reasonable
    """
    import h5py
    import numpy as np
    
    print("\n🔍 Validating dataset...")
    
    try:
        with h5py.File(path, 'r') as f:
            # Check keys
            expected_keys = ['sdf', 'latent', 'nodes', 'strains']
            for key in expected_keys:
                if key not in f:
                    print(f"  ❌ Missing key: {key}")
                    return False
                print(f"  ✓ {key}: {f[key].shape}")
            
            # Check metadata
            print(f"  ✓ num_samples: {f.attrs['num_samples']}")
            print(f"  ✓ sdf_resolution: {f.attrs['sdf_resolution']}")
            print(f"  ✓ latent_dim: {f.attrs['latent_dim']}")
            
            # Validate SDF values
            sdf = f['sdf'][:]
            sdf_min, sdf_max = sdf.min(), sdf.max()
            sdf_mean = np.abs(sdf).mean()
            
            print(f"  ✓ SDF range: [{sdf_min:.3f}, {sdf_max:.3f}]")
            print(f"  ✓ SDF mean abs: {sdf_mean:.3f}")
            
            # Check for surface (zero crossing)
            has_surface = np.any(sdf < 0) and np.any(sdf > 0)
            if has_surface:
                print(f"  ✓ SDF has zero crossing (surface exists)")
            else:
                print(f"  ⚠ SDF may not have valid surface")
            
            # Validate latent codes
            latent = f['latent'][:]
            latent_norms = np.linalg.norm(latent, axis=1)
            print(f"  ✓ Latent norms: mean={latent_norms.mean():.3f}, std={latent_norms.std():.3f}")
            
            # Check nodes
            nodes = f['nodes'][:]
            node_bounds = (nodes.min(), nodes.max())
            print(f"  ✓ Node bounds: [{node_bounds[0]:.3f}, {node_bounds[1]:.3f}]")
        
        print("\n✅ Dataset validation passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


def visualize_sample(path: str, sample_idx: int = 0):
    """Visualize a sample from the dataset."""
    import h5py
    import numpy as np
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    with h5py.File(path, 'r') as f:
        sdf = f['sdf'][sample_idx]
        nodes = f['nodes'][sample_idx]
    
    # Create figure
    fig = plt.figure(figsize=(12, 4))
    
    # Plot 1: Cloth mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='blue', s=1)
    ax1.set_title('Cloth Nodes')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: SDF slice (XY plane at Z=0)
    ax2 = fig.add_subplot(132)
    z_mid = sdf.shape[2] // 2
    im = ax2.imshow(sdf[:, :, z_mid].T, origin='lower', cmap='RdBu')
    ax2.set_title(f'SDF Slice (Z={z_mid})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: SDF histogram
    ax3 = fig.add_subplot(133)
    ax3.hist(sdf.flatten(), bins=50, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', label='Surface (SDF=0)')
    ax3.set_title('SDF Distribution')
    ax3.set_xlabel('SDF Value')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save figure
    viz_path = path.replace('.h5', '_sample.png')
    plt.savefig(viz_path, dpi=150)
    print(f"📊 Visualization saved to: {viz_path}")
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ClothGeom-NIF Dataset Generator")
    print("=" * 60)
    print(f"  Samples: {args.num_samples}")
    print(f"  SDF Resolution: {args.resolution}³")
    print(f"  Latent Dim: {args.latent_dim}")
    print(f"  Grid Size: {args.grid_size}×{args.grid_size}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Estimate size
    sdf_size = args.num_samples * (args.resolution ** 3) * 4  # float32
    latent_size = args.num_samples * args.latent_dim * 4
    nodes_size = args.num_samples * (args.grid_size ** 2) * 3 * 4
    total_size_mb = (sdf_size + latent_size + nodes_size) / (1024 * 1024)
    
    print(f"💾 Estimated uncompressed size: {total_size_mb:.1f} MB")
    print()
    
    # Time generation
    start_time = time.time()
    
    # Generate dataset
    generate_dataset(
        output_path=args.output,
        num_samples=args.num_samples,
        sdf_resolution=args.resolution,
        latent_dim=args.latent_dim,
        grid_size=args.grid_size,
        seed=args.seed
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Generation time: {elapsed:.1f}s ({elapsed/args.num_samples:.2f}s per sample)")
    
    # Get actual file size
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"💾 Actual file size: {file_size_mb:.1f} MB (with compression)")
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output)
        
        # Try to visualize
        try:
            visualize_sample(args.output)
        except Exception as e:
            print(f"⚠️ Visualization skipped: {e}")
    
    print("\n🎉 Done!")
    print(f"\nNext steps:")
    print(f"  1. Train model: python train.py --data {args.output}")
    print(f"  2. Run demo: python demo.py")


if __name__ == '__main__':
    main()
