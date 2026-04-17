"""
Abstract interface that all model backends must implement.
The Blender addon interacts exclusively through this interface.

This is the core contract between the addon's UI layer and the neural
simulation backends. All data crosses this interface as numpy arrays,
keeping the Blender side framework-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any
import numpy as np


class ModelCategory(Enum):
    """What kind of simulation does this backend perform?"""
    CLOTH_SIMULATION = auto()     # Projects 05, 09, 11, 13 — cloth on a body
    BODY_DEFORMATION = auto()     # Project 14 — soft body/skin deformation
    MOTION_PREDICTION = auto()    # Project 07 — skeleton/joint motion
    # Future extensions:
    # HAIR_SIMULATION = auto()
    # FLUID_SURFACE = auto()
    # FACIAL_ANIMATION = auto()


class OutputFormat(Enum):
    """How does the model represent its output?"""
    VERTEX_DISPLACEMENTS = auto()  # Δx, Δy, Δz per vertex (Projects 05, 11, 14)
    VERTEX_POSITIONS = auto()      # Absolute x, y, z per vertex (alternative)
    SDF_FIELD = auto()             # Signed distance field, needs marching cubes (Projects 09, 13)
    JOINT_ANGLES = auto()          # Skeleton joint rotations (Project 07)
    JOINT_POSITIONS = auto()       # Skeleton joint 3D positions (Project 07 variant)


class InputRequirement(Enum):
    """What inputs does this backend need from the scene?"""
    MESH_VERTICES = auto()         # Needs the cloth/body mesh vertex positions
    MESH_TOPOLOGY = auto()         # Needs edges/faces (for GNN backends)
    TIME = auto()                  # Needs current time/frame
    FORCES = auto()                # Supports external forces (wind, gravity)
    MATERIAL_PARAMS = auto()       # Supports material parameter control
    SKELETON_POSE = auto()         # Needs skeleton/armature pose
    COLLISION_BODY = auto()        # Needs a body mesh for collision
    QUERY_POINTS = auto()          # Needs spatial query points (for SDF backends)


@dataclass
class BackendCapabilities:
    """
    Declares what a backend can and cannot do.
    The UI reads this to show/hide relevant controls.
    
    This dataclass is the contract between backend and UI. The UI reads it
    to decide which panels and controls to show. If a backend says
    `supports_wind=False`, the wind panel is hidden. If it says
    `output_format=SDF_FIELD`, the addon knows to run marching cubes.
    """
    category: ModelCategory
    output_format: OutputFormat
    input_requirements: set[InputRequirement]
    
    # Force support
    supports_wind: bool = False
    supports_gravity: bool = False
    supports_custom_forces: bool = False
    wind_dimensions: int = 3  # 2D or 3D wind vector
    
    # Material support
    supports_material_params: bool = False
    material_param_names: list[str] = field(default_factory=list)
    # e.g., ["stiffness", "damping", "density"] — UI creates a slider for each
    material_param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # e.g., {"stiffness": (0.01, 100.0), "damping": (0.0, 1.0)}
    material_param_defaults: dict[str, float] = field(default_factory=dict)
    
    # Temporal behavior
    is_temporal: bool = True           # Does the model evolve over time?
    supports_continuous_time: bool = False  # Can query arbitrary t, or only discrete steps?
    needs_sequential_frames: bool = False   # Must be called in order (GRU state)?
    max_stable_frames: int = 1000      # How many frames before rollout diverges?
    
    # Resolution / quality
    supports_resolution_control: bool = False  # Can the user control output resolution?
    resolution_range: tuple[int, int] = (64, 256)  # Min/max for resolution slider
    default_resolution: int = 128
    
    # Performance characteristics (displayed to user)
    typical_fps: float = 30.0
    typical_memory_mb: float = 100.0
    parameter_count: int = 0
    model_size_mb: float = 0.0
    
    # Mesh requirements
    needs_fixed_topology: bool = False  # Must mesh topology stay constant across frames?
    max_vertices: int = 100_000
    
    # Human-readable metadata
    display_name: str = ""
    description: str = ""
    quality_tier: str = "standard"  # "preview", "standard", "high", "production"
    icon: str = "MESH_GRID"  # Blender icon name for the dropdown
    
    # Project reference
    project_id: str = ""  # e.g., "11-nif-cloth3d"
    project_path: str = ""  # Relative path to project directory


@dataclass
class SimulationState:
    """
    Opaque state object that backends use to carry information between frames.
    
    For GRU-based models this holds the hidden state. For stateless models
    this may be empty. The addon stores this and passes it back each frame —
    it never inspects the contents.
    """
    hidden: Optional[Any] = None  # Backend-specific hidden state (GRU state, etc.)
    frame: int = 0                # Current frame number
    time: float = 0.0             # Current time in seconds
    _backend_data: dict = field(default_factory=dict)  # Arbitrary backend-specific data
    
    def clone(self) -> "SimulationState":
        """Create a shallow copy of this state."""
        return SimulationState(
            hidden=self.hidden,  # Note: hidden state itself is not deep-copied
            frame=self.frame,
            time=self.time,
            _backend_data=dict(self._backend_data),
        )


@dataclass
class PredictionRequest:
    """
    Everything the addon sends to the backend for one frame of simulation.
    
    Backends read only the fields they need (declared in capabilities.input_requirements).
    All arrays are numpy float32 for vertices/positions, int32 for indices.
    """
    # Mesh data
    vertices: Optional[np.ndarray] = None        # (V, 3) rest or current vertex positions
    faces: Optional[np.ndarray] = None           # (F, 3) face indices (triangle mesh)
    edges: Optional[np.ndarray] = None           # (E, 2) edge indices
    vertex_normals: Optional[np.ndarray] = None  # (V, 3) vertex normals
    
    # Time
    time: float = 0.0           # Current time in seconds
    delta_time: float = 1/30    # Time since last frame
    frame: int = 0              # Current frame number
    
    # Forces
    wind_velocity: Optional[np.ndarray] = None   # (3,) wind vector in world space
    gravity: Optional[np.ndarray] = None         # (3,) gravity vector, default (0, 0, -9.81)
    custom_forces: Optional[np.ndarray] = None   # (V, 3) per-vertex external forces
    
    # Material
    material_params: dict[str, float] = field(default_factory=dict)
    # e.g., {"stiffness": 50.0, "damping": 0.1, "density": 0.3}
    
    # Skeleton (for motion backends)
    joint_positions: Optional[np.ndarray] = None  # (J, 3) current joint positions
    joint_rotations: Optional[np.ndarray] = None  # (J, 4) quaternions or (J, 3) euler
    
    # Collision body (for cloth backends)
    collision_vertices: Optional[np.ndarray] = None  # (Vb, 3) body mesh vertices
    collision_faces: Optional[np.ndarray] = None     # (Fb, 3) body mesh faces
    
    # SDF query points (for implicit field backends)
    query_points: Optional[np.ndarray] = None  # (Q, 3) points to evaluate SDF at
    query_resolution: int = 128                # Grid resolution for marching cubes
    query_bounds: Optional[tuple] = None       # ((min_x,min_y,min_z), (max_x,max_y,max_z))
    
    # State from previous frame
    state: Optional[SimulationState] = None


@dataclass
class PredictionResult:
    """
    Everything the backend returns to the addon for one frame.
    
    The addon reads the appropriate fields based on the backend's output_format.
    """
    # For VERTEX_DISPLACEMENTS output
    displacements: Optional[np.ndarray] = None  # (V, 3) displacement from rest pose
    
    # For VERTEX_POSITIONS output
    positions: Optional[np.ndarray] = None  # (V, 3) absolute world-space positions
    
    # For SDF_FIELD output
    sdf_grid: Optional[np.ndarray] = None   # (R, R, R) SDF values on regular grid
    sdf_bounds: Optional[tuple] = None       # ((min_x,min_y,min_z), (max_x,max_y,max_z))
    # The addon runs marching cubes on this grid to extract a mesh
    
    # For JOINT_ANGLES / JOINT_POSITIONS output
    joint_values: Optional[np.ndarray] = None  # (J, 3) or (J, 4) joint angles/positions
    joint_names: Optional[list[str]] = None    # Names mapping to Blender armature bones
    
    # Updated state (pass back next frame)
    state: SimulationState = field(default_factory=SimulationState)
    
    # Metadata
    inference_time_ms: float = 0.0  # How long the forward pass took
    confidence: float = 1.0          # Backend's self-assessed confidence (for rollout stability)
    error_message: Optional[str] = None  # If something went wrong


class ModelBackend(ABC):
    """
    Abstract base class that all model backends implement.
    The addon only ever holds a reference to this type.
    
    Implementation Notes:
    - All data crosses this interface as numpy arrays, not framework tensors
    - Backends convert to/from their internal format inside predict()
    - The Blender UI code never imports torch or any ML framework
    """
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """
        Return this backend's capabilities. Called once when the backend is selected.
        
        The addon uses this to configure its UI — showing/hiding panels and controls
        based on what the backend supports.
        
        Returns:
            BackendCapabilities declaring what this backend can do
        """
        ...
    
    @abstractmethod
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """
        Load model weights from a checkpoint file.
        
        Called when the user selects this backend or changes the checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pt, .onnx, etc.)
            device: "cpu", "cuda", or "cuda:N"
            
        Raises:
            FileNotFoundError: If checkpoint_path doesn't exist
            RuntimeError: If the checkpoint is incompatible or corrupted
        """
        ...
    
    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Run one frame of simulation.
        
        This is the hot path — called every frame during playback.
        Must be as fast as possible. All tensor conversions between numpy
        and the model's internal format happen inside this method.
        
        The addon guarantees:
        - request.vertices is set if MESH_VERTICES is in input_requirements
        - request.time and request.frame are always set
        - request.state is the PredictionResult.state from the previous frame
          (or None on the first frame)
        - All arrays are numpy float32 for floats, int32 for indices
        
        The backend must:
        - Return a PredictionResult with the appropriate output fields set
        - Return an updated state object for the next frame
        - Handle the first frame (state=None) gracefully by initializing state
        - Not crash or return NaN — if something goes wrong, return the input
          unchanged and set confidence=0.0 with an error_message
        
        Args:
            request: PredictionRequest with all input data
            
        Returns:
            PredictionResult with simulation output and updated state
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state (GRU hidden states, accumulators, etc.).
        
        Called when the user scrubs the timeline to frame 0, changes the mesh,
        or switches parameters that invalidate the simulation history.
        """
        ...
    
    def warmup(self, example_request: PredictionRequest) -> None:
        """
        Optional warmup call. Run a dummy prediction to trigger JIT compilation,
        CUDA kernel caching, etc. Called once after load().
        
        Default implementation just calls predict() once and discards the result.
        
        Args:
            example_request: A representative request for warmup
        """
        _ = self.predict(example_request)
    
    def get_recommended_checkpoint(self) -> Optional[str]:
        """
        Return the path to a recommended/default checkpoint, if one is bundled
        with the project. Returns None if the user must provide their own.
        
        Returns:
            Path to default checkpoint, or None
        """
        return None
    
    def get_available_checkpoints(self) -> list[str]:
        """
        Return a list of available checkpoint files for this backend.
        
        The addon can display these in a dropdown for easy selection.
        
        Returns:
            List of checkpoint file paths
        """
        return []
    
    def get_device_recommendation(self) -> str:
        """
        Return the recommended device for this backend.
        Most backends prefer CUDA if available.
        
        Returns:
            Device string: "cuda", "cuda:N", or "cpu"
        """
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def cleanup(self) -> None:
        """
        Release GPU memory, close file handles, etc.
        Called when the user switches to a different backend or disables the addon.
        """
        pass
    
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded and ready for inference.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return False
    
    def get_model_info(self) -> dict:
        """
        Return additional model information for display in the UI.
        
        Returns:
            Dict with keys like 'architecture', 'training_config', 'dataset', etc.
        """
        return {}


# Type alias for backend factory functions
BackendFactory = type[ModelBackend]
