"""
Pipeline Orchestrator.

Main entry point for the export pipeline. Coordinates export, quantization,
optimization, and benchmarking into a single workflow.
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import ExportConfig, TargetPlatform, OutputType
from .exporter import ONNXExporter, ExportResult
from .quantize import quantize_model, quantize_fp16, QuantizationResult
from .optimize import optimize_for_platform, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of export pipeline execution."""
    
    # Core artifacts
    fp32_onnx_path: str
    fp32_export_result: ExportResult
    
    # Quantized artifacts (if enabled)
    int8_onnx_path: Optional[str] = None
    int8_result: Optional[QuantizationResult] = None
    fp16_onnx_path: Optional[str] = None
    fp16_result: Optional[QuantizationResult] = None
    
    # Platform-optimized artifacts
    platform_results: Dict[str, OptimizationResult] = field(default_factory=dict)
    
    # Benchmark results
    benchmark_results: List[Any] = field(default_factory=list)
    
    # Metadata
    metadata_path: str = ""
    report_path: str = ""
    output_dir: str = ""
    
    # Timing
    total_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fp32_onnx_path": self.fp32_onnx_path,
            "fp32_export_result": self.fp32_export_result.to_dict(),
            "int8_onnx_path": self.int8_onnx_path,
            "int8_result": self.int8_result.to_dict() if self.int8_result else None,
            "fp16_onnx_path": self.fp16_onnx_path,
            "fp16_result": self.fp16_result.to_dict() if self.fp16_result else None,
            "platform_results": {k: v.to_dict() for k, v in self.platform_results.items()},
            "benchmark_results": [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.benchmark_results],
            "metadata_path": self.metadata_path,
            "report_path": self.report_path,
            "output_dir": self.output_dir,
            "total_time_seconds": self.total_time_seconds,
        }
    
    def save(self, path: str) -> None:
        """Save pipeline results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def _generate_report(
    config: ExportConfig,
    result: PipelineResult,
    output_path: str,
) -> str:
    """Generate a human-readable Markdown report."""
    
    report_lines = [
        f"# Export Pipeline Report: {config.project_name}",
        f"",
        f"**Version:** {config.model_version}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Time:** {result.total_time_seconds:.1f}s",
        f"",
        f"## Model Information",
        f"",
        f"- **Physics Domain:** {config.physics_domain or 'N/A'}",
        f"- **Description:** {config.description or 'N/A'}",
        f"- **Output Type:** {config.expected_output_type.value}",
        f"- **Parameters:** {result.fp32_export_result.param_count:,}",
        f"",
        f"## Exported Artifacts",
        f"",
        f"### FP32 ONNX Model",
        f"",
        f"- **Path:** `{result.fp32_onnx_path}`",
        f"- **Size:** {result.fp32_export_result.model_size_mb:.2f} MB",
        f"- **Opset Version:** {result.fp32_export_result.opset_version}",
    ]
    
    if result.fp32_export_result.verification_metrics:
        vm = result.fp32_export_result.verification_metrics
        report_lines.extend([
            f"- **Verification:** ✅ Passed",
            f"  - Max Abs Error: {vm['max_abs_error']:.2e}",
            f"  - Max Rel Error: {vm['max_rel_error']:.2e}",
        ])
    
    if result.int8_onnx_path:
        report_lines.extend([
            f"",
            f"### INT8 Quantized Model",
            f"",
            f"- **Path:** `{result.int8_onnx_path}`",
            f"- **Size:** {result.int8_result.quantized_size_bytes / 1024:.1f} KB",
            f"- **Compression:** {result.int8_result.compression_ratio:.2f}x",
        ])
        if result.int8_result.accuracy_metrics:
            am = result.int8_result.accuracy_metrics
            status = "✅ Passed" if result.int8_result.passed_accuracy_check else "⚠️ Warning"
            report_lines.extend([
                f"- **Accuracy Check:** {status}",
                f"  - Max Rel Error: {am.get('max_rel_error', 0):.4f}",
            ])
    
    if result.fp16_onnx_path:
        report_lines.extend([
            f"",
            f"### FP16 Model",
            f"",
            f"- **Path:** `{result.fp16_onnx_path}`",
            f"- **Compression:** {result.fp16_result.compression_ratio:.2f}x",
        ])
    
    if result.platform_results:
        report_lines.extend([
            f"",
            f"## Platform-Specific Artifacts",
            f"",
        ])
        
        for platform, opt_result in result.platform_results.items():
            status = "✅" if opt_result.success else "❌"
            report_lines.extend([
                f"### {platform.upper()}",
                f"",
                f"- **Status:** {status}",
                f"- **Path:** `{opt_result.output_path}`",
                f"- **Size:** {opt_result.optimized_size_bytes / 1024:.1f} KB",
            ])
            if opt_result.warnings:
                report_lines.append(f"- **Warnings:** {', '.join(opt_result.warnings)}")
            if opt_result.unsupported_ops:
                report_lines.append(f"- **Unsupported Ops:** {', '.join(opt_result.unsupported_ops)}")
            report_lines.append("")
    
    if result.benchmark_results:
        report_lines.extend([
            f"",
            f"## Benchmark Results",
            f"",
            f"| Provider | Batch | Latency (median) | Latency (p99) | Throughput |",
            f"|----------|-------|------------------|---------------|------------|",
        ])
        
        for br in result.benchmark_results:
            if hasattr(br, 'latency'):
                report_lines.append(
                    f"| {br.provider} | {br.batch_size} | "
                    f"{br.latency.median_ms:.2f} ms | {br.latency.p99_ms:.2f} ms | "
                    f"{br.throughput.samples_per_second:.0f} fps |"
                )
    
    report_lines.extend([
        f"",
        f"## Files Generated",
        f"",
        f"```",
        f"{result.output_dir}/",
    ])
    
    # List files
    for artifact in [
        result.fp32_onnx_path,
        result.int8_onnx_path,
        result.fp16_onnx_path,
    ]:
        if artifact:
            report_lines.append(f"├── {Path(artifact).name}")
    
    for opt_result in result.platform_results.values():
        if opt_result.output_path:
            report_lines.append(f"├── {Path(opt_result.output_path).name}")
    
    report_lines.extend([
        f"├── metadata.json",
        f"└── REPORT.md",
        f"```",
        f"",
        f"---",
        f"*Generated by PINN-Experiments Export Pipeline*",
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return output_path


def run_pipeline(
    model: nn.Module,
    config: ExportConfig,
    output_dir: str,
    skip_benchmarks: bool = False,
    skip_optimization: bool = False,
) -> PipelineResult:
    """
    Run the complete export pipeline.
    
    This is the main entry point for the export pipeline. It:
    1. Exports the model to FP32 ONNX
    2. Optionally quantizes to INT8 and/or FP16
    3. Runs platform-specific optimizations
    4. Runs benchmarks
    5. Generates metadata and reports
    
    Args:
        model: PyTorch model to export
        config: Export configuration
        output_dir: Directory for output artifacts
        skip_benchmarks: Skip benchmark phase
        skip_optimization: Skip platform optimization phase
        
    Returns:
        PipelineResult with all artifacts and metrics
    """
    start_time = time.time()
    
    logger.info(f"Starting export pipeline for {config.project_name} v{config.model_version}")
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build artifact names
    base_name = f"{config.project_name}_v{config.model_version.replace('.', '_')}"
    
    # Step 1: Export FP32 ONNX
    logger.info("Step 1: Exporting FP32 ONNX model")
    fp32_path = output_dir / f"{base_name}_fp32.onnx"
    
    exporter = ONNXExporter(config)
    fp32_result = exporter.export(model, str(fp32_path))
    
    result = PipelineResult(
        fp32_onnx_path=str(fp32_path),
        fp32_export_result=fp32_result,
        output_dir=str(output_dir),
    )
    
    # Step 2: Quantization (if enabled)
    if config.enable_int8:
        logger.info("Step 2a: INT8 quantization")
        int8_path = output_dir / f"{base_name}_int8.onnx"
        
        try:
            int8_result = quantize_model(str(fp32_path), str(int8_path), config)
            result.int8_onnx_path = str(int8_path)
            result.int8_result = int8_result
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
    
    if config.enable_fp16:
        logger.info("Step 2b: FP16 conversion")
        fp16_path = output_dir / f"{base_name}_fp16.onnx"
        
        try:
            fp16_result = quantize_fp16(str(fp32_path), str(fp16_path))
            result.fp16_onnx_path = str(fp16_path)
            result.fp16_result = fp16_result
        except Exception as e:
            logger.error(f"FP16 conversion failed: {e}")
    
    # Step 3: Platform optimization (if not skipped)
    if not skip_optimization:
        logger.info("Step 3: Platform-specific optimization")
        
        for target in config.targets:
            logger.info(f"  Optimizing for {target.value}")
            
            # Choose source model (prefer quantized for mobile/edge)
            if target in (TargetPlatform.MOBILE, TargetPlatform.EDGE_TENSORRT, TargetPlatform.EDGE_OPENVINO):
                source_path = result.int8_onnx_path or str(fp32_path)
            else:
                source_path = str(fp32_path)
            
            # Build output path
            suffix_map = {
                TargetPlatform.BROWSER: "_web.onnx",
                TargetPlatform.MOBILE: "_mobile.ort",
                TargetPlatform.EDGE_TENSORRT: "_tensorrt.engine",
                TargetPlatform.EDGE_OPENVINO: "_openvino.xml",
                TargetPlatform.DESKTOP: "_desktop.onnx",
            }
            opt_path = output_dir / f"{base_name}{suffix_map.get(target, '.onnx')}"
            
            try:
                opt_result = optimize_for_platform(
                    source_path, str(opt_path), target, config
                )
                result.platform_results[target.value] = opt_result
            except Exception as e:
                logger.error(f"Optimization for {target.value} failed: {e}")
    
    # Step 4: Benchmarks (if not skipped)
    if not skip_benchmarks:
        logger.info("Step 4: Running benchmarks")
        
        try:
            from ..benchmarks import BenchmarkSuite
            
            suite = BenchmarkSuite(
                str(fp32_path),
                input_specs=[s.to_dict() for s in config.input_specs],
            )
            
            benchmark_results = suite.run(
                batch_sizes=[1, 4, 8],
                num_iterations=500,  # Reduced for faster pipeline
            )
            result.benchmark_results = benchmark_results
            
            # Save benchmark results
            benchmarks_dir = output_dir / "benchmarks"
            benchmarks_dir.mkdir(exist_ok=True)
            
            for br in benchmark_results:
                br.save(str(benchmarks_dir / f"{br.provider}_{br.batch_size}.json"))
            
            # Generate browser benchmark HTML
            if TargetPlatform.BROWSER in config.targets:
                from ..benchmarks.benchmark import generate_browser_benchmark_html
                
                web_model = result.platform_results.get("browser", None)
                if web_model:
                    html_path = benchmarks_dir / "browser_benchmark.html"
                    metadata_path = output_dir / "metadata.json"
                    generate_browser_benchmark_html(
                        web_model.output_path,
                        str(metadata_path),
                        str(html_path),
                    )
                    
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
    
    # Step 5: Generate metadata
    logger.info("Step 5: Generating metadata")
    
    metadata = config.to_metadata_dict()
    metadata["artifacts"] = {
        "fp32": str(fp32_path),
        "int8": result.int8_onnx_path,
        "fp16": result.fp16_onnx_path,
    }
    metadata["platforms"] = {
        k: v.output_path for k, v in result.platform_results.items()
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    result.metadata_path = str(metadata_path)
    
    # Step 6: Generate report
    logger.info("Step 6: Generating report")
    
    report_path = output_dir / "REPORT.md"
    _generate_report(config, result, str(report_path))
    result.report_path = str(report_path)
    
    # Done
    result.total_time_seconds = time.time() - start_time
    
    # Save full pipeline result
    result.save(str(output_dir / "pipeline_result.json"))
    
    logger.info(f"Pipeline complete in {result.total_time_seconds:.1f}s")
    logger.info(f"Artifacts saved to: {output_dir}")
    
    return result
