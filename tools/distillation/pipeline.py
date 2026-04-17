"""
Distillation Pipeline Orchestrator

Chains all pipeline stages: distill → export → quantize → benchmark → package
with unified configuration and reporting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .config import DistillConfig, load_config, save_config
from .registry import ModelRegistry
from .utils import setup_logging, get_device, set_seed
from .distill import DistillationTrainer, DistillationResult
from .export import ONNXExporter, ExportResult
from .quantize import ModelQuantizer, QuantizationResult
from .benchmark import BenchmarkRunner, BenchmarkResult
from .package import DeploymentPackager, PackageResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from running the distillation pipeline."""
    config_path: str
    stages_run: list[str]
    total_time_seconds: float
    timestamp: str
    
    # Stage results
    distillation: DistillationResult | None = None
    export: ExportResult | None = None
    quantization: list[QuantizationResult] | None = None
    benchmark: dict[str, BenchmarkResult] | None = None
    package: PackageResult | None = None
    
    # Summary
    final_model_path: str | None = None
    final_model_size_mb: float | None = None
    final_latency_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config_path': self.config_path,
            'stages_run': self.stages_run,
            'total_time_seconds': self.total_time_seconds,
            'timestamp': self.timestamp,
            'distillation': self.distillation.to_dict() if self.distillation else None,
            'export': self.export.to_dict() if self.export else None,
            'quantization': [q.to_dict() for q in self.quantization] if self.quantization else None,
            'benchmark': {k: v.to_dict() for k, v in self.benchmark.items()} if self.benchmark else None,
            'package': {
                'target': self.package.target,
                'output_dir': self.package.output_dir,
                'files_created': self.package.files_created,
            } if self.package else None,
            'final_model_path': self.final_model_path,
            'final_model_size_mb': self.final_model_size_mb,
            'final_latency_ms': self.final_latency_ms,
        }
    
    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DistillationPipeline:
    """
    End-to-end distillation pipeline orchestrator.
    
    Usage:
        pipeline = DistillationPipeline(config_path="config.yaml")
        result = pipeline.run()  # Run all stages
        result = pipeline.run(stages=["distill", "export"])  # Run specific stages
    """
    
    ALL_STAGES = ["distill", "export", "quantize", "benchmark", "package"]
    
    def __init__(
        self,
        config: DistillConfig | str | Path,
        registry: ModelRegistry | None = None,
        output_dir: str | Path | None = None,
    ):
        """
        Args:
            config: Configuration object or path to YAML file
            registry: Model registry (uses global if not provided)
            output_dir: Override output directory
        """
        # Load config
        if isinstance(config, (str, Path)):
            self.config_path = str(config)
            self.config = load_config(config)
        else:
            self.config_path = "inline_config"
            self.config = config
        
        # Registry
        self.registry = registry or ModelRegistry.get_global()
        
        # Output directory
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(self.output_dir / "pipeline.log")
        
        # State
        self.teacher = None
        self.student = None
        self.trained_student = None
        self.export_result = None
        self.quantization_results = []
        self.benchmark_results = {}
    
    def run(
        self,
        stages: list[str] | None = None,
        skip_existing: bool = False,
    ) -> PipelineResult:
        """
        Run the distillation pipeline.
        
        Args:
            stages: List of stages to run. If None, run all stages.
            skip_existing: Skip stages that have existing outputs
            
        Returns:
            Complete pipeline results
        """
        stages = stages or self.ALL_STAGES
        
        # Validate stages
        for stage in stages:
            if stage not in self.ALL_STAGES:
                raise ValueError(f"Unknown stage: {stage}. Valid: {self.ALL_STAGES}")
        
        logger.info(f"Starting pipeline with stages: {stages}")
        start_time = time.time()
        
        # Set seed for reproducibility
        set_seed(self.config.training.seed)
        
        # Initialize result
        result = PipelineResult(
            config_path=self.config_path,
            stages_run=[],
            total_time_seconds=0,
            timestamp=datetime.now().isoformat(),
        )
        
        try:
            # Run stages
            if "distill" in stages:
                result.distillation = self._run_distillation(skip_existing)
                result.stages_run.append("distill")
            else:
                # Load existing student if skipping distillation
                self._load_existing_student()
            
            if "export" in stages:
                result.export = self._run_export(skip_existing)
                result.stages_run.append("export")
            else:
                self._load_existing_export()
            
            if "quantize" in stages:
                result.quantization = self._run_quantization(skip_existing)
                result.stages_run.append("quantize")
            
            if "benchmark" in stages:
                result.benchmark = self._run_benchmark()
                result.stages_run.append("benchmark")
            
            if "package" in stages:
                result.package = self._run_packaging()
                result.stages_run.append("package")
            
            # Finalize result
            result.total_time_seconds = time.time() - start_time
            self._populate_final_metrics(result)
            
            # Save result
            result.save(self.output_dir / "pipeline_result.json")
            
            # Generate report
            self._generate_report(result)
            
            logger.info(f"Pipeline completed in {result.total_time_seconds:.1f}s")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.total_time_seconds = time.time() - start_time
            result.save(self.output_dir / "pipeline_result_failed.json")
            raise
        
        return result
    
    def _run_distillation(self, skip_existing: bool = False) -> DistillationResult:
        """Run the distillation stage."""
        logger.info("=== Stage: Distillation ===")
        
        # Check for existing
        checkpoint_path = self.output_dir / "checkpoints" / "best_student.pt"
        if skip_existing and checkpoint_path.exists():
            logger.info("Skipping distillation, loading existing checkpoint")
            self._load_existing_student()
            return DistillationResult(
                student_path=str(checkpoint_path),
                final_loss=0.0,
                final_metrics={},
                training_time_seconds=0.0,
                epochs_trained=0,
            )
        
        # Create models
        self.teacher = self.registry.load_model(
            self.config.teacher.name,
            self.config.teacher.checkpoint,
        )
        
        self.student = self.registry.create_model(
            self.config.student.name,
            **self.config.student.init_args,
        )
        
        # Create trainer
        trainer = DistillationTrainer(self.config, self.registry)
        
        # Run training
        result = trainer.train(
            teacher=self.teacher,
            student=self.student,
            output_dir=self.output_dir,
        )
        
        # Keep trained student
        self.trained_student = trainer.student
        
        return result
    
    def _load_existing_student(self) -> None:
        """Load existing trained student model."""
        checkpoint_path = self.output_dir / "checkpoints" / "best_student.pt"
        if checkpoint_path.exists():
            self.student = self.registry.create_model(
                self.config.student.name,
                **self.config.student.init_args,
            )
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.trained_student = self.student
            logger.info(f"Loaded existing student from {checkpoint_path}")
    
    def _run_export(self, skip_existing: bool = False) -> ExportResult:
        """Run the export stage."""
        logger.info("=== Stage: Export ===")
        
        # Get export path
        export_path = self.output_dir / "export" / "student.onnx"
        
        # Check for existing
        if skip_existing and export_path.exists():
            logger.info("Skipping export, using existing")
            self.export_result = ExportResult(
                model_path=str(export_path),
                format="onnx",
                file_size_mb=export_path.stat().st_size / (1024 * 1024),
                validated=True,
            )
            return self.export_result
        
        # Ensure we have a student model
        if self.trained_student is None:
            raise RuntimeError("No trained student model. Run distillation first.")
        
        # Create exporter
        exporter = ONNXExporter(self.config.export)
        
        # Get input spec
        input_spec = self.registry.get(self.config.student.name).input_spec
        
        # Export
        self.export_result = exporter.export(
            model=self.trained_student,
            output_path=export_path,
            input_shape=input_spec.shape,
            input_dtype=input_spec.dtype,
        )
        
        return self.export_result
    
    def _load_existing_export(self) -> None:
        """Load existing export result."""
        export_path = self.output_dir / "export" / "student.onnx"
        if export_path.exists():
            self.export_result = ExportResult(
                model_path=str(export_path),
                format="onnx",
                file_size_mb=export_path.stat().st_size / (1024 * 1024),
                validated=True,
            )
    
    def _run_quantization(self, skip_existing: bool = False) -> list[QuantizationResult]:
        """Run the quantization stage."""
        logger.info("=== Stage: Quantization ===")
        
        if not self.export_result:
            raise RuntimeError("No exported model. Run export first.")
        
        # Create quantizer
        quantizer = ModelQuantizer(self.config.quantization)
        
        # Run quantization for each requested method
        results = []
        for method in self.config.quantization.methods:
            output_path = self.output_dir / "quantized" / f"student_{method}.onnx"
            
            # Check for existing
            if skip_existing and output_path.exists():
                logger.info(f"Skipping {method} quantization, using existing")
                result = QuantizationResult(
                    model_path=str(output_path),
                    method=method,
                    original_size_mb=self.export_result.file_size_mb,
                    quantized_size_mb=output_path.stat().st_size / (1024 * 1024),
                    size_reduction_percent=0.0,
                )
                results.append(result)
                continue
            
            # Get calibration data if needed
            calibration_data = None
            if method == "static_int8":
                calibration_data = self._get_calibration_data()
            
            result = quantizer.quantize(
                model_path=self.export_result.model_path,
                output_path=output_path,
                method=method,
                calibration_data=calibration_data,
            )
            results.append(result)
        
        self.quantization_results = results
        return results
    
    def _get_calibration_data(self) -> list:
        """Get calibration data for static quantization."""
        sampler = create_sampler(self.config.sampling, self.registry)
        
        input_spec = self.registry.get(self.config.student.name).input_spec
        data = []
        
        for i in range(self.config.quantization.calibration_samples):
            batch = sampler.sample(batch_size=1, input_shape=input_spec.shape[1:])
            data.append({'input': batch.numpy()})
        
        return data
    
    def _run_benchmark(self) -> dict[str, BenchmarkResult]:
        """Run the benchmark stage."""
        logger.info("=== Stage: Benchmark ===")
        
        # Create benchmark runner
        runner = BenchmarkRunner(self.config.benchmark)
        
        results = {}
        
        # Get input spec
        input_spec = self.registry.get(self.config.student.name).input_spec
        
        # Benchmark original ONNX
        if self.export_result and Path(self.export_result.model_path).exists():
            for backend in self.config.benchmark.backends:
                key = f"onnx_{backend}"
                try:
                    results[key] = runner.benchmark_onnx(
                        model_path=self.export_result.model_path,
                        input_shape=input_spec.shape,
                        backend=backend,
                    )
                except Exception as e:
                    logger.warning(f"Benchmark failed for {key}: {e}")
        
        # Benchmark quantized models
        for quant_result in self.quantization_results:
            if Path(quant_result.model_path).exists():
                key = f"quantized_{quant_result.method}"
                try:
                    results[key] = runner.benchmark_onnx(
                        model_path=quant_result.model_path,
                        input_shape=input_spec.shape,
                        backend="cpu",  # Quantized models typically run on CPU
                    )
                except Exception as e:
                    logger.warning(f"Benchmark failed for {key}: {e}")
        
        # Benchmark PyTorch model
        if self.trained_student is not None:
            try:
                results["pytorch_cpu"] = runner.benchmark_pytorch(
                    model=self.trained_student,
                    input_shape=input_spec.shape,
                    device="cpu",
                )
                
                if torch.cuda.is_available():
                    results["pytorch_cuda"] = runner.benchmark_pytorch(
                        model=self.trained_student,
                        input_shape=input_spec.shape,
                        device="cuda",
                    )
            except Exception as e:
                logger.warning(f"PyTorch benchmark failed: {e}")
        
        self.benchmark_results = results
        return results
    
    def _run_packaging(self) -> PackageResult:
        """Run the packaging stage."""
        logger.info("=== Stage: Packaging ===")
        
        # Determine best model to package
        model_path = self._select_best_model()
        
        # Get metadata
        input_spec = self.registry.get(self.config.student.name).input_spec
        output_spec = self.registry.get(self.config.student.name).output_spec
        
        metadata = {
            'model_name': self.config.student.name,
            'input_shape': list(input_spec.shape),
            'output_shape': list(output_spec.shape) if output_spec else None,
            'input_dtype': input_spec.dtype,
            'distillation_config': self.config_path,
        }
        
        # Add benchmark results if available
        if self.benchmark_results:
            best_backend = min(
                self.benchmark_results.keys(),
                key=lambda k: self.benchmark_results[k].latency.mean_ms
            )
            metadata['benchmark'] = {
                'best_backend': best_backend,
                'latency_ms': self.benchmark_results[best_backend].latency.mean_ms,
            }
        
        # Create packager
        packager = DeploymentPackager(
            model_path=model_path,
            metadata=metadata,
            model_name=self.config.student.name,
        )
        
        # Package based on target
        target = self.config.package.target
        package_dir = self.output_dir / "package" / target
        
        if target == "web":
            result = packager.package_for_web(
                output_dir=package_dir,
                include_demo=self.config.package.include_demo,
            )
        elif target == "mobile":
            result = packager.package_for_mobile(
                output_dir=package_dir,
                include_example=self.config.package.include_example,
            )
        elif target == "edge":
            result = packager.package_for_edge(
                output_dir=package_dir,
                target_platform=self.config.package.target_platform,
                include_cpp=self.config.package.include_cpp,
            )
        else:
            # Default to edge packaging
            result = packager.package_for_edge(
                output_dir=package_dir,
                target_platform="x86_64",
            )
        
        return result
    
    def _select_best_model(self) -> str:
        """Select the best model for packaging based on benchmark results."""
        candidates = []
        
        # Original ONNX
        if self.export_result:
            candidates.append(self.export_result.model_path)
        
        # Quantized models
        for qr in self.quantization_results:
            candidates.append(qr.model_path)
        
        if not candidates:
            raise RuntimeError("No models available for packaging")
        
        # If we have benchmarks, select based on latency
        if self.benchmark_results:
            # Find the model with best latency that meets size constraints
            best_path = None
            best_latency = float('inf')
            
            for path in candidates:
                # Find corresponding benchmark
                path_name = Path(path).stem
                for key, result in self.benchmark_results.items():
                    if path_name in key or key in path_name:
                        if result.latency.mean_ms < best_latency:
                            best_latency = result.latency.mean_ms
                            best_path = path
            
            if best_path:
                return best_path
        
        # Default to smallest model
        return min(candidates, key=lambda p: Path(p).stat().st_size)
    
    def _populate_final_metrics(self, result: PipelineResult) -> None:
        """Populate final metrics in the result."""
        # Find final model
        if result.package:
            for f in result.package.files_created:
                if f.endswith('.onnx'):
                    result.final_model_path = f
                    result.final_model_size_mb = Path(f).stat().st_size / (1024 * 1024)
                    break
        elif self.export_result:
            result.final_model_path = self.export_result.model_path
            result.final_model_size_mb = self.export_result.file_size_mb
        
        # Get best latency
        if self.benchmark_results:
            best_result = min(
                self.benchmark_results.values(),
                key=lambda r: r.latency.mean_ms
            )
            result.final_latency_ms = best_result.latency.mean_ms
    
    def _generate_report(self, result: PipelineResult) -> None:
        """Generate human-readable report."""
        report_path = self.output_dir / "pipeline_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Distillation Pipeline Report\n\n")
            f.write(f"**Config:** {result.config_path}\n")
            f.write(f"**Timestamp:** {result.timestamp}\n")
            f.write(f"**Total Time:** {result.total_time_seconds:.1f}s\n")
            f.write(f"**Stages Run:** {', '.join(result.stages_run)}\n\n")
            
            # Distillation
            if result.distillation:
                f.write("## Distillation\n\n")
                f.write(f"- **Epochs:** {result.distillation.epochs_trained}\n")
                f.write(f"- **Final Loss:** {result.distillation.final_loss:.6f}\n")
                f.write(f"- **Training Time:** {result.distillation.training_time_seconds:.1f}s\n")
                f.write(f"- **Output:** {result.distillation.student_path}\n\n")
            
            # Export
            if result.export:
                f.write("## Export\n\n")
                f.write(f"- **Format:** {result.export.format}\n")
                f.write(f"- **Size:** {result.export.file_size_mb:.2f} MB\n")
                f.write(f"- **Validated:** {result.export.validated}\n")
                f.write(f"- **Output:** {result.export.model_path}\n\n")
            
            # Quantization
            if result.quantization:
                f.write("## Quantization\n\n")
                f.write("| Method | Original (MB) | Quantized (MB) | Reduction |\n")
                f.write("|--------|---------------|----------------|----------|\n")
                for qr in result.quantization:
                    f.write(f"| {qr.method} | {qr.original_size_mb:.2f} | "
                           f"{qr.quantized_size_mb:.2f} | {qr.size_reduction_percent:.1f}% |\n")
                f.write("\n")
            
            # Benchmark
            if result.benchmark:
                f.write("## Benchmark\n\n")
                f.write("| Backend | Latency (ms) | P95 (ms) | Throughput (samples/s) |\n")
                f.write("|---------|--------------|----------|------------------------|\n")
                for name, br in result.benchmark.items():
                    f.write(f"| {name} | {br.latency.mean_ms:.2f} | "
                           f"{br.latency.p95_ms:.2f} | {br.throughput.samples_per_second:.1f} |\n")
                f.write("\n")
            
            # Package
            if result.package:
                f.write("## Package\n\n")
                f.write(f"- **Target:** {result.package.target}\n")
                f.write(f"- **Output Directory:** {result.package.output_dir}\n")
                f.write(f"- **Files Created:**\n")
                for fp in result.package.files_created:
                    f.write(f"  - {fp}\n")
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            if result.final_model_path:
                f.write(f"- **Final Model:** {result.final_model_path}\n")
            if result.final_model_size_mb:
                f.write(f"- **Final Size:** {result.final_model_size_mb:.2f} MB\n")
            if result.final_latency_ms:
                f.write(f"- **Final Latency:** {result.final_latency_ms:.2f} ms\n")
        
        logger.info(f"Report saved to {report_path}")


def create_sampler(config, registry):
    """Create sampler from config (import helper)."""
    from .sampling import (
        RandomSampling, GridQuerySampling, DatasetSampling, TeacherGuidedSampling
    )
    
    strategy = config.strategy
    
    if strategy == "random":
        return RandomSampling(bounds=config.bounds)
    elif strategy == "grid":
        return GridQuerySampling(bounds=config.bounds, resolution=config.resolution)
    elif strategy == "dataset":
        return DatasetSampling(
            data_path=config.data_path,
            augment=config.augment,
        )
    elif strategy == "teacher_guided":
        return TeacherGuidedSampling()
    else:
        return RandomSampling()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run distillation pipeline")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--stages", nargs="+", default=None,
                       help="Stages to run (distill, export, quantize, benchmark, package)")
    parser.add_argument("--output", "-o", help="Output directory override")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip stages with existing outputs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)
    
    # Run pipeline
    pipeline = DistillationPipeline(
        config=args.config,
        output_dir=args.output,
    )
    
    result = pipeline.run(
        stages=args.stages,
        skip_existing=args.skip_existing,
    )
    
    print(f"\n✓ Pipeline completed in {result.total_time_seconds:.1f}s")
    print(f"  Stages: {', '.join(result.stages_run)}")
    if result.final_model_path:
        print(f"  Model: {result.final_model_path}")
    if result.final_latency_ms:
        print(f"  Latency: {result.final_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
