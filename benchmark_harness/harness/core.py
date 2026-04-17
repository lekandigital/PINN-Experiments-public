"""
Core data structures for the benchmarking harness.

Defines BenchmarkResult (the output schema) and BenchmarkConfig (runtime settings).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class TimingStats:
    """Statistical summary of timing measurements."""
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    num_runs: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    peak_gpu_gb: Optional[float] = None
    model_size_gb: float = 0.0
    inference_peak_gb: float = 0.0
    peak_cpu_rss_gb: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ModelInfo:
    """Model metadata."""
    parameter_count: int
    parameter_count_non_trainable: int = 0
    model_size_mb: float = 0.0
    onnx_size_mb: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TrainingInfo:
    """Training metadata extracted from logs or documentation."""
    total_time_seconds: Optional[float] = None
    epochs: Optional[int] = None
    time_per_epoch_seconds: Optional[float] = None
    hardware: Optional[str] = None
    convergence_metric: Optional[str] = None
    convergence_value: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AccuracyMetrics:
    """Standard accuracy metrics."""
    rmse: Optional[float] = None
    rmse_units: Optional[str] = None
    nrmse: Optional[float] = None
    l2_relative_error: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HardwareInfo:
    """Hardware environment information."""
    gpu: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    cpu: str = "Unknown"
    ram_gb: float = 0.0
    os: str = "Unknown"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SoftwareInfo:
    """Software versions."""
    python: str
    framework: str
    framework_version: str
    cuda_toolkit: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result for a single project.
    
    This is the canonical output format for all benchmarks.
    JSON-serializable via to_dict() / to_json().
    """
    # Identity
    project_id: str
    project_name: str
    framework: str  # "pytorch", "jax", "onnx", "tensorflow"
    
    # Timestamp and versioning
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_hash: Optional[str] = None
    harness_version: str = "1.0.0"
    
    # Hardware and software environment
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    software: SoftwareInfo = field(default_factory=lambda: SoftwareInfo(
        python="Unknown", framework="Unknown", framework_version="Unknown"
    ))
    
    # Model information
    model: ModelInfo = field(default_factory=lambda: ModelInfo(parameter_count=0))
    
    # Performance metrics
    inference_time_gpu: Optional[TimingStats] = None
    inference_time_cpu: Optional[TimingStats] = None
    throughput_samples_per_sec: Optional[float] = None
    throughput_fps: Optional[float] = None
    memory: MemoryStats = field(default_factory=MemoryStats)
    
    # Training information
    training: TrainingInfo = field(default_factory=TrainingInfo)
    
    # Accuracy metrics
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    domain_metrics: dict[str, Any] = field(default_factory=dict)
    
    # Test data description
    test_dataset: str = "Unknown"
    test_data_synthetic_fallback: bool = False
    
    # Status and notes
    status: str = "success"  # "success", "failed", "partial"
    error_message: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "framework": self.framework,
            "timestamp": self.timestamp,
            "git_hash": self.git_hash,
            "harness_version": self.harness_version,
            "hardware": self.hardware.to_dict(),
            "software": self.software.to_dict(),
            "model": self.model.to_dict(),
            "performance": {
                "inference_time_gpu_ms": self.inference_time_gpu.to_dict() if self.inference_time_gpu else None,
                "inference_time_cpu_ms": self.inference_time_cpu.to_dict() if self.inference_time_cpu else None,
                "throughput_samples_per_sec": self.throughput_samples_per_sec,
                "throughput_fps": self.throughput_fps,
                "memory": self.memory.to_dict(),
            },
            "training": self.training.to_dict(),
            "accuracy": self.accuracy.to_dict(),
            "domain_metrics": self.domain_metrics,
            "test_dataset": self.test_dataset,
            "test_data_synthetic_fallback": self.test_data_synthetic_fallback,
            "status": self.status,
            "error_message": self.error_message,
            "notes": self.notes,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Path | str) -> None:
        """Save to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkResult:
        """Reconstruct from a dictionary."""
        # Handle nested dataclasses
        hardware = HardwareInfo(**data.get("hardware", {}))
        software = SoftwareInfo(**data.get("software", {
            "python": "Unknown", "framework": "Unknown", "framework_version": "Unknown"
        }))
        model = ModelInfo(**data.get("model", {"parameter_count": 0}))
        training = TrainingInfo(**data.get("training", {}))
        accuracy = AccuracyMetrics(**data.get("accuracy", {}))
        memory = MemoryStats(**data.get("performance", {}).get("memory", {}))
        
        perf = data.get("performance", {})
        inference_gpu = None
        if perf.get("inference_time_gpu_ms"):
            inference_gpu = TimingStats(**perf["inference_time_gpu_ms"])
        inference_cpu = None
        if perf.get("inference_time_cpu_ms"):
            inference_cpu = TimingStats(**perf["inference_time_cpu_ms"])
        
        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            framework=data["framework"],
            timestamp=data.get("timestamp", ""),
            git_hash=data.get("git_hash"),
            harness_version=data.get("harness_version", "1.0.0"),
            hardware=hardware,
            software=software,
            model=model,
            inference_time_gpu=inference_gpu,
            inference_time_cpu=inference_cpu,
            throughput_samples_per_sec=perf.get("throughput_samples_per_sec"),
            throughput_fps=perf.get("throughput_fps"),
            memory=memory,
            training=training,
            accuracy=accuracy,
            domain_metrics=data.get("domain_metrics", {}),
            test_dataset=data.get("test_dataset", "Unknown"),
            test_data_synthetic_fallback=data.get("test_data_synthetic_fallback", False),
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
            notes=data.get("notes", ""),
        )
    
    @classmethod
    def load(cls, path: Path | str) -> BenchmarkResult:
        """Load from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def create_failed(
        cls,
        project_id: str,
        project_name: str,
        framework: str,
        error_message: str
    ) -> BenchmarkResult:
        """Create a failed result entry."""
        return cls(
            project_id=project_id,
            project_name=project_name,
            framework=framework,
            status="failed",
            error_message=error_message,
        )


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark runs.
    
    Loaded from config.yaml or set programmatically.
    """
    # Timing settings
    num_warmup_runs: int = 100
    num_timed_runs: int = 1000
    
    # Quick mode (for development)
    quick_mode: bool = False
    quick_warmup: int = 10
    quick_runs: int = 100
    
    # Batch sizes to test
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32])
    
    # Device settings
    device: str = "cuda"  # "cuda", "cpu", "auto"
    force_cpu_benchmark: bool = True  # Also run CPU benchmarks
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("results"))
    save_raw: bool = True
    save_aggregated: bool = True
    generate_tables: bool = True
    generate_dashboard: bool = True
    
    # Project selection
    include_projects: Optional[list[str]] = None  # None = all
    exclude_projects: list[str] = field(default_factory=list)
    
    # Framework isolation (run JAX in subprocess)
    isolate_jax: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    @property
    def effective_warmup(self) -> int:
        return self.quick_warmup if self.quick_mode else self.num_warmup_runs
    
    @property
    def effective_runs(self) -> int:
        return self.quick_runs if self.quick_mode else self.num_timed_runs
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkConfig:
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> BenchmarkConfig:
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def save_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
