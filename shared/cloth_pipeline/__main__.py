"""
CLI entry point for the cloth simulation pipeline.

Usage:
    python -m cloth_pipeline run --config cloth_pipeline_config.yaml
    python -m cloth_pipeline ingest --source /path/to/sim.abc --info
    python -m cloth_pipeline export --config config.yaml --target project_09
    python -m cloth_pipeline validate --data /path/to/training_data/
    python -m cloth_pipeline init-config --output config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

from .config import load_config, save_config, create_default_config, create_example_config_yaml, PipelineConfig
from .types import ClothSequence


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline: ingest → transform → export."""
    from .pipeline import run_pipeline
    
    config = load_config(args.config)
    
    if args.verbose:
        print(f"[cloth_pipeline] Running full pipeline with config: {args.config}")
        print(f"[cloth_pipeline] Config hash: {config.get_hash()}")
    
    try:
        results = run_pipeline(config, verbose=args.verbose)
        
        if args.verbose:
            print(f"\n[cloth_pipeline] Pipeline completed successfully!")
            print(f"[cloth_pipeline] Processed {results.get('num_frames', 0)} frames")
            print(f"[cloth_pipeline] Exported to: {config.export.output_root}")
        
        return 0
    except Exception as e:
        print(f"[cloth_pipeline] ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest data and optionally display info without transforming."""
    from .ingest import ingest_source
    
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"[cloth_pipeline] ERROR: Source not found: {source_path}", file=sys.stderr)
        return 1
    
    # Auto-detect format if not specified
    format_type = args.format
    if format_type is None:
        if source_path.suffix == '.abc':
            format_type = 'abc'
        elif source_path.is_dir():
            format_type = 'obj_sequence'
        else:
            print("[cloth_pipeline] ERROR: Could not auto-detect format. Use --format.", file=sys.stderr)
            return 1
    
    if args.verbose:
        print(f"[cloth_pipeline] Ingesting: {source_path}")
        print(f"[cloth_pipeline] Format: {format_type}")
    
    try:
        sequence = ingest_source(
            source_path=str(source_path),
            format_type=format_type,
            fps=args.fps,
            rest_frame=args.rest_frame,
        )
        
        if args.info:
            print_sequence_info(sequence)
        
        if args.output:
            import numpy as np
            output_path = Path(args.output)
            np.savez_compressed(
                output_path,
                **sequence.to_dict()
            )
            print(f"[cloth_pipeline] Saved to: {output_path}")
        
        return 0
    except Exception as e:
        print(f"[cloth_pipeline] ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Export to a specific project format."""
    from .pipeline import run_pipeline
    
    config = load_config(args.config)
    
    # Enable only the specified target
    target = args.target.lower().replace('-', '_')
    valid_targets = ['project_04', 'project_05', 'project_08', 'project_09', 
                     'project_11', 'project_12', 'project_13']
    
    if target not in valid_targets:
        print(f"[cloth_pipeline] ERROR: Invalid target '{target}'", file=sys.stderr)
        print(f"[cloth_pipeline] Valid targets: {', '.join(valid_targets)}", file=sys.stderr)
        return 1
    
    # Disable all targets except the specified one
    for t in valid_targets:
        target_config = getattr(config.export.targets, t)
        target_config.enabled = (t == target)
    
    if args.verbose:
        print(f"[cloth_pipeline] Exporting to: {target}")
    
    try:
        results = run_pipeline(config, verbose=args.verbose)
        return 0
    except Exception as e:
        print(f"[cloth_pipeline] ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate exported dataset."""
    from .validation import validate_dataset
    
    data_path = Path(args.data)
    
    if not data_path.exists():
        print(f"[cloth_pipeline] ERROR: Data path not found: {data_path}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"[cloth_pipeline] Validating: {data_path}")
    
    try:
        issues = validate_dataset(data_path, verbose=args.verbose)
        
        if issues:
            print(f"\n[cloth_pipeline] Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print(f"[cloth_pipeline] Validation passed!")
            return 0
    except Exception as e:
        print(f"[cloth_pipeline] ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_init_config(args: argparse.Namespace) -> int:
    """Generate a default configuration file."""
    output_path = Path(args.output)
    
    if output_path.exists() and not args.force:
        print(f"[cloth_pipeline] ERROR: File exists: {output_path}", file=sys.stderr)
        print("[cloth_pipeline] Use --force to overwrite.", file=sys.stderr)
        return 1
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(create_example_config_yaml())
    
    print(f"[cloth_pipeline] Created config: {output_path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display package information."""
    from . import __version__
    
    print(f"cloth_pipeline v{__version__}")
    print()
    print("A professional cloth simulation data pipeline for ML training.")
    print()
    print("Supported input formats:")
    print("  - Alembic (.abc) - Houdini Vellum, Maya, etc.")
    print("  - OBJ sequences  - Marvelous Designer, Blender, etc.")
    print("  - Blender cloth  - Direct recording from Blender")
    print()
    print("Supported export targets:")
    print("  - project_04: ClothGeom-NIF (Static SDF)")
    print("  - project_05: ClothGNN (Graph Sequence)")
    print("  - project_08: HGNN-ClothDyn (Hierarchical Graph)")
    print("  - project_09: HGNN-NIF-Cloth (Hybrid Graph+SDF)")
    print("  - project_11: NIF-Cloth3D-Interactive (Blender)")
    print("  - project_12: NIF-Cloth4D-Temporal (Spacetime SDF)")
    print("  - project_13: NIF-Cloth4D (Compact 4D SDF)")
    print()
    print("Usage:")
    print("  python -m cloth_pipeline run --config config.yaml")
    print("  python -m cloth_pipeline ingest --source /path/to/data --info")
    print("  python -m cloth_pipeline init-config --output config.yaml")
    
    return 0


def print_sequence_info(sequence: ClothSequence) -> None:
    """Print detailed information about a ClothSequence."""
    print("\n" + "=" * 60)
    print("CLOTH SEQUENCE INFO")
    print("=" * 60)
    
    print(f"\nMesh Statistics:")
    print(f"  Vertices:     {sequence.num_vertices:,}")
    print(f"  Faces:        {sequence.num_faces:,} ({'triangles' if sequence.is_triangulated else 'quads'})")
    print(f"  Frames:       {sequence.num_frames:,}")
    print(f"  Frame range:  {sequence.frame_range[0]} - {sequence.frame_range[1]}")
    
    print(f"\nTiming:")
    print(f"  FPS:          {sequence.fps}")
    print(f"  Timestep:     {sequence.dt:.6f}s")
    print(f"  Duration:     {sequence.duration:.3f}s")
    
    bbox_min, bbox_max = sequence.get_bounding_box()
    bbox_size = bbox_max - bbox_min
    print(f"\nBounding Box:")
    print(f"  Min:          [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
    print(f"  Max:          [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
    print(f"  Size:         [{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}]")
    
    print(f"\nOptional Data:")
    print(f"  Normals:      {'Yes' if sequence.normals is not None else 'No'}")
    print(f"  UVs:          {'Yes' if sequence.uvs is not None else 'No'}")
    print(f"  Velocities:   {'Yes' if sequence.velocities is not None else 'No'}")
    print(f"  Attributes:   {list(sequence.vertex_attributes.keys()) if sequence.vertex_attributes else 'None'}")
    print(f"  Collision:    {len(sequence.collision_bodies) if sequence.collision_bodies else 0} bodies")
    
    if sequence.metadata:
        print(f"\nMetadata:")
        for key, value in sequence.metadata.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='cloth_pipeline',
        description='Professional cloth simulation data pipeline for ML training',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--config', '-c', required=True, help='Path to config YAML')
    run_parser.set_defaults(func=cmd_run)
    
    # ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest and inspect data')
    ingest_parser.add_argument('--source', '-s', required=True, help='Source path')
    ingest_parser.add_argument('--format', '-f', choices=['abc', 'obj_sequence', 'blender'],
                               help='Input format (auto-detected if not specified)')
    ingest_parser.add_argument('--fps', type=float, default=24.0, help='Source FPS')
    ingest_parser.add_argument('--rest-frame', type=int, default=0, help='Rest pose frame')
    ingest_parser.add_argument('--info', '-i', action='store_true', help='Print sequence info')
    ingest_parser.add_argument('--output', '-o', help='Save ingested data to NPZ')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export to specific project')
    export_parser.add_argument('--config', '-c', required=True, help='Path to config YAML')
    export_parser.add_argument('--target', '-t', required=True, 
                               help='Target project (e.g., project_09)')
    export_parser.set_defaults(func=cmd_export)
    
    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate exported dataset')
    validate_parser.add_argument('--data', '-d', required=True, help='Path to exported data')
    validate_parser.set_defaults(func=cmd_validate)
    
    # init-config command
    init_parser = subparsers.add_parser('init-config', help='Create default config file')
    init_parser.add_argument('--output', '-o', default='cloth_pipeline_config.yaml',
                             help='Output config path')
    init_parser.add_argument('--force', '-f', action='store_true', help='Overwrite if exists')
    init_parser.set_defaults(func=cmd_init_config)
    
    # info command
    info_parser = subparsers.add_parser('info', help='Display package information')
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
