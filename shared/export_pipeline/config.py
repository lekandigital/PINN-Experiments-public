"""
Export Pipeline Configuration Schema.

Defines dataclasses for configuring model export, including tensor specifications,
quantization settings, mesh reconstruction parameters, and target platforms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any
from enum import Enum
import json


class OutputType(str, Enum):
    """Type of model output, determines how the viewer interprets results."""
    MESH_SDF = "mesh_sdf"           # SDF field -> marching cubes -> mesh
    MESH_VERTICES = "mesh_vertices"  # Direct vertex positions
    FIELD_SCALAR = "field_scalar"    # Scalar field visualization
    FIELD_VECTOR = "field_vector"    # Vector field visualization
    IMAGE = "image"                  # Image output
    SCALAR = "scalar"                # Single scalar values


class QuantizationMethod(str, Enum):
    """Quantization approach."""
    DYNAMIC = "dynamic"  # Dynamic quantization (weights INT8, activations FP32 at runtime)
    STATIC = "static"    # Static quantization (both weights and activations INT8)


class TargetPlatform(str, Enum):
    """Deployment target platforms."""
    BROWSER = "browser"           # ONNX Runtime Web (WASM/WebGL/WebGPU)
    MOBILE = "mobile"             # ONNX Runtime Mobile (.ort format)
    EDGE_TENSORRT = "edge_tensorrt"  # TensorRT for NVIDIA edge devices
    EDGE_OPENVINO = "edge_openvino"  # OpenVINO for Intel devices
    DESKTOP = "desktop"           # Standard ONNX Runtime (CPU/CUDA)


@dataclass
class TensorSpec:
    """
    Specification for an input or output tensor.
    
    Attributes:
        name: Tensor name in the ONNX model
        shape: Tensor shape, use -1 for dynamic dimensions
        dtype: Data type string (float32, float16, int32, int64, etc.)
        description: Human-readable description
        min_value: Optional minimum expected value (for validation/normalization)
        max_value: Optional maximum expected value
    """
    name: str
    shape: List[int]
    dtype: str = "float32"
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorSpec":
        return cls(**data)


@dataclass
class PhysicsParameter:
    """
    A physics parameter that can be controlled in the viewer UI.
    
    Attributes:
        name: Parameter name (must match model input)
        display_name: Human-readable name for UI
        input_index: Index in the input tensor (or separate input name)
        min_value: Minimum slider value
        max_value: Maximum slider value
        default_value: Default value
        step: Slider step size
        unit: Unit string for display (e.g., "m/s", "degrees")
    """
    name: str
    display_name: str
    input_index: Optional[int] = None
    input_name: Optional[str] = None
    min_value: float = 0.0
    max_value: float = 1.0
    default_value: float = 0.5
    step: float = 0.01
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "input_index": self.input_index,
            "input_name": self.input_name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "step": self.step,
            "unit": self.unit,
        }


@dataclass
class MeshReconstructionConfig:
    """
    Configuration for reconstructing meshes from model output.
    
    Attributes:
        method: Reconstruction method
        grid_resolution: Resolution for marching cubes (32, 64, 128, 256)
        iso_value: Iso-surface level for SDF (typically 0.0)
        bounds: Spatial domain bounds (min, max) for each axis
        smooth_iterations: Laplacian smoothing passes
        decimate_target: Target face count for decimation (None = no decimation)
        template_mesh_path: Path to template mesh file (for vertex-based models)
    """
    method: Literal["marching_cubes", "direct_vertices", "displacement_map"] = "marching_cubes"
    grid_resolution: int = 64
    iso_value: float = 0.0
    bounds: Tuple[float, float] = (-1.0, 1.0)
    smooth_iterations: int = 2
    decimate_target: Optional[int] = None
    template_mesh_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "grid_resolution": self.grid_resolution,
            "iso_value": self.iso_value,
            "bounds": list(self.bounds),
            "smooth_iterations": self.smooth_iterations,
            "decimate_target": self.decimate_target,
            "template_mesh_path": self.template_mesh_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshReconstructionConfig":
        if "bounds" in data:
            data["bounds"] = tuple(data["bounds"])
        return cls(**data)


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.
    
    Attributes:
        method: Quantization method (dynamic or static)
        calibration_data_path: Path to calibration data (required for static)
        calibration_num_samples: Number of samples for calibration
        per_channel: Use per-channel quantization for weights
        reduce_range: Use reduced range for better accuracy on some hardware
        accuracy_threshold: Max acceptable relative error (warn if exceeded)
    """
    method: QuantizationMethod = QuantizationMethod.DYNAMIC
    calibration_data_path: Optional[str] = None
    calibration_num_samples: int = 100
    per_channel: bool = True
    reduce_range: bool = False
    accuracy_threshold: float = 0.05  # 5% relative error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "calibration_data_path": self.calibration_data_path,
            "calibration_num_samples": self.calibration_num_samples,
            "per_channel": self.per_channel,
            "reduce_range": self.reduce_range,
            "accuracy_threshold": self.accuracy_threshold,
        }


@dataclass 
class TemporalConfig:
    """
    Configuration for models with temporal/recurrent state.
    
    Attributes:
        has_temporal_state: Whether model has hidden states between frames
        hidden_state_names: Names of hidden state inputs/outputs
        hidden_state_shapes: Shapes of hidden state tensors
        sequence_length: Default sequence length for export
    """
    has_temporal_state: bool = False
    hidden_state_names: List[str] = field(default_factory=list)
    hidden_state_shapes: List[List[int]] = field(default_factory=list)
    sequence_length: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_temporal_state": self.has_temporal_state,
            "hidden_state_names": self.hidden_state_names,
            "hidden_state_shapes": self.hidden_state_shapes,
            "sequence_length": self.sequence_length,
        }


@dataclass
class ExportConfig:
    """
    Complete configuration for model export pipeline.
    
    This is the main configuration object that drives the entire export process.
    It specifies what model to export, how to export it, and what platforms to target.
    
    Example:
        config = ExportConfig(
            project_name="nif_cloth4d",
            model_version="1.0.0",
            input_specs=[TensorSpec("coords", [-1, 4], "float32", "Spacetime coordinates (x,y,z,t)")],
            output_specs=[TensorSpec("sdf", [-1, 1], "float32", "Signed distance field value")],
            targets=[TargetPlatform.BROWSER, TargetPlatform.MOBILE],
            expected_output_type=OutputType.MESH_SDF,
            mesh_config=MeshReconstructionConfig(grid_resolution=64),
        )
    """
    # Model identification
    project_name: str
    model_version: str
    
    # Input/output specification
    input_specs: List[TensorSpec]
    output_specs: List[TensorSpec]
    
    # Dynamic axes for ONNX export (maps input/output name to axis mapping)
    # Example: {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=dict)
    
    # Export options
    opset_version: int = 17  # ONNX opset (17 is widely supported as of 2025)
    enable_fp16: bool = False
    enable_int8: bool = False
    
    # Quantization config (used if enable_int8=True)
    quantization_config: Optional[QuantizationConfig] = None
    
    # Target platforms
    targets: List[TargetPlatform] = field(default_factory=lambda: [TargetPlatform.DESKTOP])
    
    # Metadata
    description: str = ""
    physics_domain: str = ""  # "cloth_simulation", "coastal_flow", "aerodynamics", etc.
    
    # Output type (informs the viewer how to interpret model output)
    expected_output_type: OutputType = OutputType.SCALAR
    
    # Mesh reconstruction config (only used if expected_output_type is mesh-related)
    mesh_config: Optional[MeshReconstructionConfig] = None
    
    # Temporal/recurrent model config
    temporal_config: Optional[TemporalConfig] = None
    
    # Physics parameters exposed in viewer UI
    physics_parameters: List[PhysicsParameter] = field(default_factory=list)
    
    # Verification settings
    verification_atol: float = 1e-5  # Absolute tolerance for PyTorch vs ONNX comparison
    verification_rtol: float = 1e-4  # Relative tolerance
    verification_samples: int = 10   # Number of random samples for verification
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Convert string targets to enum if needed
        if self.targets and isinstance(self.targets[0], str):
            self.targets = [TargetPlatform(t) for t in self.targets]
        
        # Convert string output type to enum if needed
        if isinstance(self.expected_output_type, str):
            self.expected_output_type = OutputType(self.expected_output_type)
        
        # Set default dynamic axes if not provided
        if not self.dynamic_axes:
            self.dynamic_axes = {}
            for spec in self.input_specs:
                axes = {}
                for i, dim in enumerate(spec.shape):
                    if dim == -1:
                        axes[i] = f"dim_{i}"
                if axes:
                    self.dynamic_axes[spec.name] = axes
            for spec in self.output_specs:
                axes = {}
                for i, dim in enumerate(spec.shape):
                    if dim == -1:
                        axes[i] = f"dim_{i}"
                if axes:
                    self.dynamic_axes[spec.name] = axes
        
        # Set default quantization config if INT8 enabled but no config provided
        if self.enable_int8 and self.quantization_config is None:
            self.quantization_config = QuantizationConfig()
        
        # Validate mesh config for mesh output types
        if self.expected_output_type in (OutputType.MESH_SDF, OutputType.MESH_VERTICES):
            if self.mesh_config is None:
                self.mesh_config = MeshReconstructionConfig()
    
    def get_input_names(self) -> List[str]:
        """Get list of input tensor names."""
        return [spec.name for spec in self.input_specs]
    
    def get_output_names(self) -> List[str]:
        """Get list of output tensor names."""
        return [spec.name for spec in self.output_specs]
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Convert config to metadata dictionary for embedding in ONNX model
        and saving alongside exported artifacts.
        """
        return {
            "project_name": self.project_name,
            "model_version": self.model_version,
            "input_specs": [s.to_dict() for s in self.input_specs],
            "output_specs": [s.to_dict() for s in self.output_specs],
            "opset_version": self.opset_version,
            "description": self.description,
            "physics_domain": self.physics_domain,
            "expected_output_type": self.expected_output_type.value,
            "mesh_config": self.mesh_config.to_dict() if self.mesh_config else None,
            "temporal_config": self.temporal_config.to_dict() if self.temporal_config else None,
            "physics_parameters": [p.to_dict() for p in self.physics_parameters],
        }
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        data = {
            "project_name": self.project_name,
            "model_version": self.model_version,
            "input_specs": [s.to_dict() for s in self.input_specs],
            "output_specs": [s.to_dict() for s in self.output_specs],
            "dynamic_axes": self.dynamic_axes,
            "opset_version": self.opset_version,
            "enable_fp16": self.enable_fp16,
            "enable_int8": self.enable_int8,
            "quantization_config": self.quantization_config.to_dict() if self.quantization_config else None,
            "targets": [t.value for t in self.targets],
            "description": self.description,
            "physics_domain": self.physics_domain,
            "expected_output_type": self.expected_output_type.value,
            "mesh_config": self.mesh_config.to_dict() if self.mesh_config else None,
            "temporal_config": self.temporal_config.to_dict() if self.temporal_config else None,
            "physics_parameters": [p.to_dict() for p in self.physics_parameters],
            "verification_atol": self.verification_atol,
            "verification_rtol": self.verification_rtol,
            "verification_samples": self.verification_samples,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExportConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert nested objects
        data["input_specs"] = [TensorSpec.from_dict(s) for s in data["input_specs"]]
        data["output_specs"] = [TensorSpec.from_dict(s) for s in data["output_specs"]]
        
        if data.get("quantization_config"):
            qc = data["quantization_config"]
            qc["method"] = QuantizationMethod(qc["method"])
            data["quantization_config"] = QuantizationConfig(**qc)
        
        if data.get("mesh_config"):
            data["mesh_config"] = MeshReconstructionConfig.from_dict(data["mesh_config"])
        
        if data.get("temporal_config"):
            data["temporal_config"] = TemporalConfig(**data["temporal_config"])
        
        if data.get("physics_parameters"):
            data["physics_parameters"] = [PhysicsParameter(**p) for p in data["physics_parameters"]]
        
        data["targets"] = [TargetPlatform(t) for t in data["targets"]]
        data["expected_output_type"] = OutputType(data["expected_output_type"])
        
        return cls(**data)
