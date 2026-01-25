#!/usr/bin/env python
"""
Generate Large-Scale Synthetic Cloth Dataset for Production Training

Creates high-resolution cloth simulation data with:
- 40x40 fine mesh (1600 vertices) instead of 20x20 (400 vertices)
- 64^3 SDF grid instead of 32^3
- Chunked HDF5 writing for memory efficiency

Usage:
    python scripts/generate_large_dataset.py --num_samples 10000 --fine_res 40 --sdf_res 64
    python scripts/generate_large_dataset.py --config configs/rtx3090_large.yaml --num_samples 10000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
from tqdm import tqdm

from src.data.synthetic_data import (
    create_cloth_mesh,
    apply_cloth_deformation,
    decimate_mesh,
    compute_sdf_volume_fast
)


def generate_large_dataset(
    num_samples: int = 10000,
    fine_resolution: int = 40,
    coarse_stride: int = 2,
    sdf_resolution: int = 64,
    cloth_size: float = 1.0,
    output_path: str = 'data/production_train.h5',
    chunk_size: int = 100,
    seed: int = 42
) -> None:
    """
    Generate large-scale cloth dataset with chunked writing.

    Args:
        num_samples: Total number of samples to generate
        fine_resolution: Vertices per side for fine mesh (total = fine_resolution^2)
        coarse_stride: Decimation stride for coarse mesh
        sdf_resolution: SDF grid resolution per axis
        cloth_size: Physical size of cloth
        output_path: Path to output HDF5 file
        chunk_size: Number of samples per chunk (for memory efficiency)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Calculate dimensions
    num_fine = fine_resolution ** 2
    num_coarse = (fine_resolution // coarse_stride) ** 2

    print("=" * 60)
    print("HGNN-NIF-Cloth: Large Dataset Generation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Fine mesh: {fine_resolution}x{fine_resolution} = {num_fine:,} vertices")
    print(f"  Coarse mesh: {fine_resolution // coarse_stride}x{fine_resolution // coarse_stride} = {num_coarse:,} vertices")
    print(f"  SDF resolution: {sdf_resolution}^3 = {sdf_resolution**3:,} voxels")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Output: {output_path}")

    # Estimate storage
    bytes_per_sample = (
        num_fine * 3 * 4 +                    # fine_positions (float32)
        num_coarse * 3 * 4 +                  # coarse_positions (float32)
        sdf_resolution ** 3 * 4              # sdf_volumes (float32)
    )
    estimated_size_gb = (bytes_per_sample * num_samples) / 1e9
    print(f"  Estimated raw size: {estimated_size_gb:.1f} GB (compressed ~{estimated_size_gb * 0.3:.1f} GB)")

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create base mesh (for consistent topology)
    base_vertices, fine_edges = create_cloth_mesh(fine_resolution, cloth_size)
    _, coarse_edges, _ = decimate_mesh(base_vertices, fine_edges, coarse_stride)

    print(f"\nMesh topology:")
    print(f"  Fine edges: {len(fine_edges):,}")
    print(f"  Coarse edges: {len(coarse_edges):,}")

    # Create HDF5 file with chunked, resizable datasets
    print(f"\nCreating HDF5 file...")

    with h5py.File(output_path, 'w') as f:
        # Create resizable datasets with compression
        fine_positions_ds = f.create_dataset(
            'fine_positions',
            shape=(0, num_fine, 3),
            maxshape=(None, num_fine, 3),
            dtype='float32',
            chunks=(min(chunk_size, num_samples), num_fine, 3),
            compression='gzip',
            compression_opts=4
        )

        coarse_positions_ds = f.create_dataset(
            'coarse_positions',
            shape=(0, num_coarse, 3),
            maxshape=(None, num_coarse, 3),
            dtype='float32',
            chunks=(min(chunk_size, num_samples), num_coarse, 3),
            compression='gzip',
            compression_opts=4
        )

        sdf_volumes_ds = f.create_dataset(
            'sdf_volumes',
            shape=(0, sdf_resolution, sdf_resolution, sdf_resolution),
            maxshape=(None, sdf_resolution, sdf_resolution, sdf_resolution),
            dtype='float32',
            chunks=(1, sdf_resolution, sdf_resolution, sdf_resolution),
            compression='gzip',
            compression_opts=4
        )

        # Store edge indices (same for all samples)
        f.create_dataset('fine_edges', data=fine_edges.T, compression='gzip')  # Transpose to (2, E) format
        f.create_dataset('coarse_edges', data=coarse_edges.T, compression='gzip')

        # Store metadata
        f.attrs['num_samples'] = num_samples
        f.attrs['fine_resolution'] = fine_resolution
        f.attrs['coarse_stride'] = coarse_stride
        f.attrs['sdf_resolution'] = sdf_resolution
        f.attrs['cloth_size'] = cloth_size
        f.attrs['num_fine_vertices'] = num_fine
        f.attrs['num_coarse_vertices'] = num_coarse

        # Generate data in chunks
        print(f"\nGenerating samples...")
        pbar = tqdm(total=num_samples, desc="Generating")

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            actual_chunk_size = chunk_end - chunk_start

            # Allocate chunk buffers
            fine_chunk = np.zeros((actual_chunk_size, num_fine, 3), dtype=np.float32)
            coarse_chunk = np.zeros((actual_chunk_size, num_coarse, 3), dtype=np.float32)
            sdf_chunk = np.zeros((actual_chunk_size, sdf_resolution, sdf_resolution, sdf_resolution), dtype=np.float32)

            # Generate samples in this chunk
            for i in range(actual_chunk_size):
                global_idx = chunk_start + i

                # Time progresses through simulation
                t = global_idx / max(num_samples - 1, 1)

                # Randomize deformation parameters for diversity
                amplitude = 0.1 + np.random.rand() * 0.2
                freq_space = 1.0 + np.random.rand() * 2.0
                freq_time = 0.5 + np.random.rand() * 1.0
                gravity = 0.03 + np.random.rand() * 0.1
                noise_scale = 0.01 + np.random.rand() * 0.02

                # Apply deformation
                deformed = apply_cloth_deformation(
                    base_vertices,
                    time=t,
                    amplitude=amplitude,
                    frequency_space=freq_space,
                    frequency_time=freq_time,
                    gravity=gravity,
                    noise_scale=noise_scale
                )

                fine_chunk[i] = deformed

                # Decimate for coarse mesh
                coarse_verts, _, _ = decimate_mesh(deformed, fine_edges, coarse_stride)
                coarse_chunk[i] = coarse_verts

                # Compute SDF volume
                sdf = compute_sdf_volume_fast(deformed, sdf_resolution)
                sdf_chunk[i] = sdf

                pbar.update(1)

            # Resize datasets and append chunk
            fine_positions_ds.resize(chunk_end, axis=0)
            coarse_positions_ds.resize(chunk_end, axis=0)
            sdf_volumes_ds.resize(chunk_end, axis=0)

            fine_positions_ds[chunk_start:chunk_end] = fine_chunk
            coarse_positions_ds[chunk_start:chunk_end] = coarse_chunk
            sdf_volumes_ds[chunk_start:chunk_end] = sdf_chunk

            # Flush to disk periodically
            f.flush()

        pbar.close()

    # Report final file size
    file_size = output_path.stat().st_size
    print(f"\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"  Output: {output_path}")
    print(f"  File size: {file_size / 1e6:.1f} MB ({file_size / 1e9:.2f} GB)")
    print(f"  Samples: {num_samples:,}")
    print(f"  Bytes/sample: {file_size / num_samples:.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate large-scale cloth simulation dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--fine_res', type=int, default=40,
                        help='Fine mesh resolution (vertices per side)')
    parser.add_argument('--coarse_stride', type=int, default=2,
                        help='Coarse mesh decimation stride')
    parser.add_argument('--sdf_res', type=int, default=64,
                        help='SDF grid resolution')

    # Output configuration
    parser.add_argument('--output', type=str, default='data/production_train.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Samples per chunk (for memory efficiency)')

    # Config file option
    parser.add_argument('--config', type=str, default=None,
                        help='Load data config from YAML file (overrides other data args)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get('data', {})
        args.fine_res = data_cfg.get('fine_resolution', args.fine_res)
        args.coarse_stride = data_cfg.get('coarse_stride', args.coarse_stride)
        args.sdf_res = data_cfg.get('sdf_resolution', args.sdf_res)
        print(f"Loaded config from {args.config}")

    generate_large_dataset(
        num_samples=args.num_samples,
        fine_resolution=args.fine_res,
        coarse_stride=args.coarse_stride,
        sdf_resolution=args.sdf_res,
        output_path=args.output,
        chunk_size=args.chunk_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
