#!/usr/bin/env python3
"""
Animation Pipeline Demo

Demonstrates the full animation pipeline with mock models.
Run this script to verify the pipeline is working correctly.

Usage:
    python demo.py                    # Run with defaults
    python demo.py --frames 60        # Specify number of frames
    python demo.py --output ./out     # Specify output directory
    python demo.py --device cuda      # Use GPU
"""

import argparse
import sys
from pathlib import Path
import time

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import numpy as np

from shared.pipeline import (
    AnimationPipeline,
    PipelineConfig,
    FrameResult,
    create_pipeline,
)
from shared.pipeline.exporters import (
    export_obj,
    export_npz,
    export_bvh,
    SequenceExporter,
)
from shared.pipeline.config import SMPL_JOINT_NAMES, SMPL_PARENTS


def parse_args():
    parser = argparse.ArgumentParser(description="Animation Pipeline Demo")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second")
    parser.add_argument("--output", type=str, default="./demo_output",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device")
    parser.add_argument("--profile", action="store_true",
                        help="Enable timing profiling")
    parser.add_argument("--export-all", action="store_true",
                        help="Export all formats (OBJ, NPZ, BVH)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"=" * 60)
    print(f"Animation Pipeline Demo")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Frames: {args.frames}")
    print(f"FPS: {args.fps}")
    print(f"Output: {args.output}")
    print()
    
    # Create pipeline
    print("Creating pipeline...")
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    pipeline = AnimationPipeline(config, device=device)
    
    # Load models (will use mock models since no checkpoints provided)
    print("Loading models (using mock models for demo)...")
    pipeline.load()
    
    print(f"Pipeline ready: {pipeline}")
    print()
    
    # Generate animation
    print(f"Generating {args.frames} frames...")
    start_time = time.time()
    
    t_start = 0.0
    t_end = args.frames / args.fps
    
    frames = []
    timing_stats = {
        'motion': [],
        'body': [],
        'cloth': [],
        'total': []
    }
    
    for i, frame in enumerate(pipeline.stream_sequence(t_start, t_end, args.fps, profile=args.profile)):
        frames.append(frame)
        
        if args.profile:
            stats = pipeline.get_timing_stats()
            for key in timing_stats:
                timing_stats[key].append(stats[key])
        
        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i+1}/{args.frames} (t={frame.time:.3f}s)")
    
    total_time = time.time() - start_time
    print(f"\nGeneration complete in {total_time:.2f}s ({args.frames/total_time:.1f} fps)")
    
    # Timing statistics
    if args.profile and timing_stats['total']:
        print(f"\nTiming Statistics (per frame):")
        print(f"  Motion:  {np.mean(timing_stats['motion']):.2f}ms (±{np.std(timing_stats['motion']):.2f})")
        print(f"  Body:    {np.mean(timing_stats['body']):.2f}ms (±{np.std(timing_stats['body']):.2f})")
        print(f"  Cloth:   {np.mean(timing_stats['cloth']):.2f}ms (±{np.std(timing_stats['cloth']):.2f})")
        print(f"  Total:   {np.mean(timing_stats['total']):.2f}ms (±{np.std(timing_stats['total']):.2f})")
    
    # Output statistics
    print(f"\nFrame Statistics:")
    print(f"  Body vertices:  {frames[0].body_vertices.shape[0]}")
    print(f"  Body faces:     {frames[0].body_faces.shape[0]}")
    print(f"  Cloth vertices: {frames[0].cloth_vertices.shape[0]}")
    print(f"  Cloth faces:    {frames[0].cloth_faces.shape[0]}")
    print(f"  Skeleton joints: {frames[0].joint_positions.shape[0]}")
    
    # Check for cloth strain
    if frames[-1].cloth_strain_energy is not None:
        energies = [f.cloth_strain_energy for f in frames]
        print(f"  Cloth strain energy: {np.mean(energies):.2f} (max: {np.max(energies):.2f})")
    
    # Export
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert frames to numpy dicts for export
    frame_dicts = [f.to_dict() for f in frames]
    
    if args.export_all:
        print(f"\nExporting to {output_dir}...")
        exporter = SequenceExporter(str(output_dir), name="demo")
        
        # Get skeleton info for BVH
        rest_positions = np.array(config.skeleton.rest_positions)
        
        exporter.export_all(
            frame_dicts,
            joint_names=SMPL_JOINT_NAMES,
            parent_indices=SMPL_PARENTS,
            rest_positions=rest_positions,
            fps=args.fps
        )
    else:
        # Just export NPZ
        npz_path = output_dir / "demo.npz"
        print(f"\nExporting NPZ: {npz_path}")
        export_npz(frame_dicts, str(npz_path))
        
        # Export first and last frame as OBJ
        print("Exporting preview frames...")
        export_obj(
            frame_dicts[0]['body_vertices'],
            frame_dicts[0]['body_faces'],
            str(output_dir / "body_first.obj"),
            normals=frame_dicts[0].get('body_normals')
        )
        export_obj(
            frame_dicts[-1]['body_vertices'],
            frame_dicts[-1]['body_faces'],
            str(output_dir / "body_last.obj"),
            normals=frame_dicts[-1].get('body_normals')
        )
        export_obj(
            frame_dicts[-1]['cloth_vertices'],
            frame_dicts[-1]['cloth_faces'],
            str(output_dir / "cloth_last.obj"),
            normals=frame_dicts[-1].get('cloth_normals')
        )
    
    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
