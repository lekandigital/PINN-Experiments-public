"""
Configuration dataclasses and YAML loading for the distillation pipeline.

All pipeline behavior is driven by YAML configuration files. This module
defines the schema and provides validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class TeacherConfig:
    """Configuration for the teacher model."""
    name: str
    checkpoint: str
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def __post_init__(self):
        # Resolve relative paths
        if not os.path.isabs(self.checkpoint):
            # Will be resolved relative to workspace root during loading
            pass


@dataclass
class StudentConfig:
    """Configuration for the student model."""
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    pretrained_checkpoint: str | None = None  # Optional warm start


@dataclass
class SamplingConfig:
    """Configuration for input sampling during distillation."""
    strategy: Literal["random", "dataset", "grid_query", "teacher_guided"]
    
    # For dataset strategy
    source_dataset: str | None = None
    
    # For random/grid_query strategy  
    num_samples_per_epoch: int = 50000
    spatial_range: list[float] | list[list[float]] = field(default_factory=lambda: [-1.0, 1.0])
    temporal_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    
    # For grid_query (SIREN distillation)
    num_query_points_per_sample: int = 4096
    num_cloth_configs: int = 200
    
    # For teacher_guided
    complexity_metric: str = "output_gradient_norm"
    oversample_ratio: float = 3.0
    
    # Augmentation
    augmentation: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Configuration for distillation loss components."""
    weights: dict[str, float] = field(default_factory=lambda: {"soft_target": 1.0})
    temperature: float = 2.0
    
    # Feature matching settings
    feature_matching: dict[str, str] | None = None  # {teacher_layer: student_layer}
    
    # Physics loss settings (domain-specific, provided by model registration)
    physics_config: dict[str, Any] = field(default_factory=dict)


@dataclass 
class CurriculumStage:
    """A single stage in curriculum/progressive distillation."""
    epochs: list[int]  # [start, end]
    description: str
    # Domain-specific parameters (e.g., forcing_range for coastal)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for distillation training."""
    num_epochs: int = 500
    batch_size: int = 4096
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    
    # Scheduler
    scheduler: Literal["cosine_annealing", "reduce_on_plateau", "cosine_annealing_warm_restarts", "none"] = "cosine_annealing"
    scheduler_params: dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    eval_every: int = 25
    save_every: int = 50
    
    # Early stopping
    patience: int | None = None
    
    # Reproducibility
    seed: int = 42
    
    # Curriculum/progressive distillation
    curriculum: list[CurriculumStage] = field(default_factory=list)


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    strategy: Literal["direct", "flatten", "decoder_only", "torchscript"] = "direct"
    opset_version: int = 17
    
    # For flatten strategy (GNN to dense)
    max_nodes: int | None = None
    
    # Input/output shapes (inferred if not provided)
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    
    # Dynamic axes
    dynamic_axes: dict[str, dict[int, str]] | None = None
    
    # Also export TorchScript variant
    also_export_torchscript: bool = False
    
    # Validation tolerance
    validation_tolerance: float = 1e-5
    num_validation_samples: int = 100


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    methods: list[Literal["dynamic_int8", "static_int8", "fp16"]] = field(
        default_factory=lambda: ["dynamic_int8"]
    )
    calibration_samples: int = 500
    
    # Static quantization settings
    per_channel: bool = True
    
    # Accuracy threshold - warn if degradation exceeds this
    accuracy_warning_threshold: float = 0.05


@dataclass
class BenchmarkTargets:
    """Performance targets for pass/fail determination."""
    latency_ms: float | None = None
    model_size_mb: float | None = None
    peak_memory_mb: float | None = None
    accuracy_relative: float | None = None
    throughput_samples_per_sec: float | None = None
    
    # Domain-specific targets
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    backends: list[str] = field(default_factory=lambda: ["onnx_cpu"])
    input_sizes: list[int] = field(default_factory=lambda: [1, 100, 1000])
    
    num_warmup: int = 50
    num_runs: int = 1000
    throughput_duration: float = 10.0  # seconds
    
    targets: BenchmarkTargets = field(default_factory=BenchmarkTargets)


@dataclass
class WebPackageConfig:
    """Configuration for web deployment package."""
    include_demo: bool = True
    demo_type: str = "basic"


@dataclass
class MobilePackageConfig:
    """Configuration for mobile deployment package."""
    framework: Literal["onnx", "pytorch", "both"] = "onnx"
    include_example_app: bool = False


@dataclass
class EdgePackageConfig:
    """Configuration for edge deployment package."""
    target_platform: Literal["arm64", "x86_64", "armv7"] = "arm64"
    include_cpp_inference: bool = True
    include_python_inference: bool = True
    include_sample_data: bool = False
    offline_mode: bool = True


@dataclass
class PackageConfig:
    """Configuration for deployment packaging."""
    targets: list[Literal["web", "mobile", "edge", "server"]] = field(default_factory=list)
    output_dir: str = "dist"
    
    web: WebPackageConfig = field(default_factory=WebPackageConfig)
    mobile: MobilePackageConfig = field(default_factory=MobilePackageConfig)
    edge: EdgePackageConfig = field(default_factory=EdgePackageConfig)


@dataclass
class DistillConfig:
    """
    Complete configuration for a distillation pipeline run.
    
    This is the top-level config that gets loaded from YAML.
    """
    # Metadata
    name: str = "distillation"
    description: str = ""
    
    # Core components
    teacher: TeacherConfig | None = None
    student: StudentConfig | None = None
    
    # Distillation settings
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Pipeline stages
    export: ExportConfig = field(default_factory=ExportConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    package: PackageConfig = field(default_factory=PackageConfig)
    
    # Output paths
    output_dir: str = "distillation_output"
    
    # Workspace root (set during loading)
    workspace_root: str = ""
    
    def resolve_paths(self, workspace_root: str):
        """Resolve all relative paths against workspace root."""
        self.workspace_root = workspace_root
        
        if self.teacher and not os.path.isabs(self.teacher.checkpoint):
            self.teacher.checkpoint = os.path.join(workspace_root, self.teacher.checkpoint)
        
        if self.sampling.source_dataset and not os.path.isabs(self.sampling.source_dataset):
            self.sampling.source_dataset = os.path.join(workspace_root, self.sampling.source_dataset)
        
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(workspace_root, self.output_dir)


def _dict_to_dataclass(data: dict, cls: type) -> Any:
    """Recursively convert a dict to a dataclass, handling nested structures."""
    if data is None:
        return None
    
    # Get field types
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()} if hasattr(cls, '__dataclass_fields__') else {}
    
    processed = {}
    for key, value in data.items():
        if key not in field_types:
            continue
            
        field_type = field_types[key]
        
        # Handle nested dataclasses
        if isinstance(value, dict):
            # Check if field type is a dataclass
            origin = getattr(field_type, '__origin__', None)
            if origin is None and hasattr(field_type, '__dataclass_fields__'):
                value = _dict_to_dataclass(value, field_type)
            # Handle Optional[SomeDataclass]
            elif origin is type(None) or str(field_type).startswith('typing.Optional'):
                args = getattr(field_type, '__args__', ())
                for arg in args:
                    if hasattr(arg, '__dataclass_fields__'):
                        value = _dict_to_dataclass(value, arg)
                        break
        
        # Handle list of dataclasses (e.g., curriculum stages)
        elif isinstance(value, list) and value:
            origin = getattr(field_type, '__origin__', None)
            if origin is list:
                args = getattr(field_type, '__args__', ())
                if args and hasattr(args[0], '__dataclass_fields__'):
                    value = [_dict_to_dataclass(item, args[0]) for item in value]
        
        processed[key] = value
    
    return cls(**processed)


def load_config(config_path: str, workspace_root: str | None = None) -> DistillConfig:
    """
    Load a distillation configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        workspace_root: Root directory for resolving relative paths.
                       If None, uses the parent of the config file's directory.
    
    Returns:
        DistillConfig with all settings populated
    """
    config_path = Path(config_path)
    
    if workspace_root is None:
        # Default: assume config is in tools/distillation/configs/
        # and workspace root is 3 levels up
        workspace_root = str(config_path.parent.parent.parent.parent)
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Handle nested 'distillation' key if present (some configs wrap everything)
    if 'distillation' in raw_config and 'teacher' in raw_config['distillation']:
        # Flatten: merge distillation settings into top level
        distill_settings = raw_config.pop('distillation')
        raw_config.update(distill_settings)
    
    # Convert to dataclass
    config = _dict_to_dataclass(raw_config, DistillConfig)
    
    # Resolve relative paths
    config.resolve_paths(workspace_root)
    
    return config


def save_config(config: DistillConfig, output_path: str):
    """Save a configuration to YAML."""
    import dataclasses
    
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    with open(output_path, 'w') as f:
        yaml.dump(to_dict(config), f, default_flow_style=False, sort_keys=False)
