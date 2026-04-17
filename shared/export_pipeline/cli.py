#!/usr/bin/env python
"""
Export Pipeline CLI.

Command-line interface for exporting models to ONNX and various deployment targets.

Usage:
    python -m shared.export_pipeline.cli --project nif_cloth4d --checkpoint model.pt --config config.yaml --output exports/

Examples:
    # Export Project 13 (NIF-Cloth4D) for browser
    python -m shared.export_pipeline.cli \
        --project nif_cloth4d \
        --checkpoint projects/13-nif-cloth4d__project-space/nif-cloth4d/checkpoints/model.pt \
        --config shared/export_pipeline/configs/cloth4d_browser.yaml \
        --output exports/cloth4d/
    
    # Export Project 15 (PINN-Lite-Foil) for edge
    python -m shared.export_pipeline.cli \
        --project pinn_lite_foil \
        --checkpoint projects/15-pinn-lite-foil__project-space/pinn-lite-foil/models/student/model.pt \
        --config shared/export_pipeline/configs/pinn_lite_edge.yaml \
        --output exports/pinn_lite/
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.export_pipeline.config import ExportConfig
from shared.export_pipeline.pipeline import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Project-specific model loaders
def load_model_generic(checkpoint_path: str, config: ExportConfig) -> torch.nn.Module:
    """
    Generic model loader that attempts to load from checkpoint.
    
    This is a fallback - project-specific adapters should provide better loading.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint
    elif isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            return checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            raise ValueError(
                "Checkpoint contains state_dict but no model architecture. "
                "Use a project-specific adapter."
            )
    
    raise ValueError(f"Cannot load model from checkpoint: {checkpoint_path}")


def load_model_with_adapter(
    project_name: str,
    checkpoint_path: str,
    config: ExportConfig,
) -> torch.nn.Module:
    """
    Load model using project-specific adapter if available.
    """
    # Try to import project-specific adapter
    adapter_module_name = f"shared.export_pipeline.adapters.{project_name}_adapter"
    
    try:
        adapter = importlib.import_module(adapter_module_name)
        if hasattr(adapter, 'load_model'):
            logger.info(f"Using adapter: {adapter_module_name}")
            return adapter.load_model(checkpoint_path, config)
    except ImportError:
        logger.info(f"No adapter found for {project_name}, using generic loader")
    
    return load_model_generic(checkpoint_path, config)


def main():
    parser = argparse.ArgumentParser(
        description='Export PINN models to ONNX and deployment targets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='Project name (e.g., nif_cloth4d, cloth_gnn, pinn_lite_foil)'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pt, .pth)'
    )
    
    parser.add_argument(
        '--config', '-f',
        type=str,
        required=True,
        help='Path to export config file (.yaml or .json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for exported artifacts'
    )
    
    parser.add_argument(
        '--skip-benchmarks',
        action='store_true',
        help='Skip benchmark phase'
    )
    
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip platform-specific optimization phase'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config without running export'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    config_path = Path(args.config)
    if config_path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert to ExportConfig
            from shared.export_pipeline.config import (
                TensorSpec, MeshReconstructionConfig, QuantizationConfig,
                TemporalConfig, PhysicsParameter, TargetPlatform, OutputType,
                QuantizationMethod
            )
            
            # Parse input/output specs
            config_data['input_specs'] = [
                TensorSpec(**s) for s in config_data.get('input_specs', [])
            ]
            config_data['output_specs'] = [
                TensorSpec(**s) for s in config_data.get('output_specs', [])
            ]
            
            # Parse nested configs
            if config_data.get('mesh_config'):
                config_data['mesh_config'] = MeshReconstructionConfig(**config_data['mesh_config'])
            if config_data.get('quantization_config'):
                qc = config_data['quantization_config']
                if 'method' in qc and isinstance(qc['method'], str):
                    qc['method'] = QuantizationMethod(qc['method'])
                config_data['quantization_config'] = QuantizationConfig(**qc)
            if config_data.get('temporal_config'):
                config_data['temporal_config'] = TemporalConfig(**config_data['temporal_config'])
            if config_data.get('physics_parameters'):
                config_data['physics_parameters'] = [
                    PhysicsParameter(**p) for p in config_data['physics_parameters']
                ]
            
            # Parse enums
            if 'targets' in config_data:
                config_data['targets'] = [
                    TargetPlatform(t) if isinstance(t, str) else t 
                    for t in config_data['targets']
                ]
            if 'expected_output_type' in config_data:
                if isinstance(config_data['expected_output_type'], str):
                    config_data['expected_output_type'] = OutputType(config_data['expected_output_type'])
            
            config = ExportConfig(**config_data)
            
        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install pyyaml")
            sys.exit(1)
    else:
        config = ExportConfig.load(str(config_path))
    
    logger.info(f"Loaded config for {config.project_name} v{config.model_version}")
    logger.info(f"  Targets: {[t.value for t in config.targets]}")
    logger.info(f"  Output type: {config.expected_output_type.value}")
    
    if args.dry_run:
        logger.info("Dry run - config validated successfully")
        logger.info(f"  Input specs: {[s.name for s in config.input_specs]}")
        logger.info(f"  Output specs: {[s.name for s in config.output_specs]}")
        return
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model_with_adapter(args.project, args.checkpoint, config)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {param_count:,}")
    
    # Run pipeline
    logger.info(f"Running export pipeline...")
    result = run_pipeline(
        model=model,
        config=config,
        output_dir=args.output,
        skip_benchmarks=args.skip_benchmarks,
        skip_optimization=args.skip_optimization,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Project: {config.project_name} v{config.model_version}")
    print(f"Output directory: {result.output_dir}")
    print(f"Total time: {result.total_time_seconds:.1f}s")
    print()
    print("Artifacts:")
    print(f"  FP32 ONNX: {Path(result.fp32_onnx_path).name} ({result.fp32_export_result.model_size_mb:.2f} MB)")
    if result.int8_onnx_path:
        print(f"  INT8 ONNX: {Path(result.int8_onnx_path).name} ({result.int8_result.quantized_size_bytes / 1024:.1f} KB)")
    if result.fp16_onnx_path:
        print(f"  FP16 ONNX: {Path(result.fp16_onnx_path).name}")
    for platform, opt_result in result.platform_results.items():
        status = "✓" if opt_result.success else "✗"
        print(f"  {platform}: {Path(opt_result.output_path).name} [{status}]")
    print()
    print(f"Report: {result.report_path}")
    print(f"Metadata: {result.metadata_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
