#!/usr/bin/env python3
"""
data_pipeline.py - Data loading and preprocessing for Geom-INR-Motion

This module provides:
1. AMASS dataset loader (SMPL format)
2. Human3.6M dataset loader
3. Skeleton normalizer (zero-center, scale by bone length)
4. Continuous-time interpolators (cubic spline)
5. PyTorch Dataset class for training

The pipeline converts discrete motion capture frames to continuous-time
representations that can be queried at arbitrary timestamps.

Dependencies:
    pip install numpy scipy torch
    
Optional (for AMASS with proper SMPL conversion):
    pip install smplx

Usage:
    from data_pipeline import MotionDataset, create_synthetic_dataset
    
    # For testing without real data:
    dataset = create_synthetic_dataset(num_sequences=10, num_frames=100)
    
    # For real AMASS data:
    dataset = MotionDataset.from_amass('/path/to/AMASS')
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Callable, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
import glob
import os


# ============================================================================
# Configuration
# ============================================================================

# Standard skeleton joint counts
SMPL_NUM_JOINTS = 24  # SMPL body model
SMPL_FULL_JOINTS = 52  # SMPL with hands
H36M_NUM_JOINTS = 17  # Human3.6M skeleton

# Default joint hierarchy for SMPL (parent indices)
SMPL_PARENTS = [
    -1,  # 0: pelvis (root)
    0,   # 1: left hip
    0,   # 2: right hip
    0,   # 3: spine1
    1,   # 4: left knee
    2,   # 5: right knee
    3,   # 6: spine2
    4,   # 7: left ankle
    5,   # 8: right ankle
    6,   # 9: spine3
    7,   # 10: left foot
    8,   # 11: right foot
    9,   # 12: neck
    9,   # 13: left collar
    9,   # 14: right collar
    12,  # 15: head
    13,  # 16: left shoulder
    14,  # 17: right shoulder
    16,  # 18: left elbow
    17,  # 19: right elbow
    18,  # 20: left wrist
    19,  # 21: right wrist
    20,  # 22: left hand
    21,  # 23: right hand
]


# ============================================================================
# Motion Sequence Container
# ============================================================================

class MotionSequence:
    """
    Container for a single motion sequence with metadata.
    
    Stores joint positions over time and provides interpolation.
    """
    
    def __init__(
        self,
        positions: np.ndarray,
        fps: float = 30.0,
        actor_id: int = 0,
        name: str = "unnamed",
        joint_names: Optional[List[str]] = None
    ):
        """
        Initialize motion sequence.
        
        Args:
            positions: (T, J, 3) array of joint positions over time
            fps: Frames per second of the original capture
            actor_id: Unique identifier for the actor/subject
            name: Human-readable name for the sequence
            joint_names: Optional list of joint names
        """
        assert positions.ndim == 3 and positions.shape[2] == 3, \
            f"Expected (T, J, 3) array, got {positions.shape}"
        
        self.positions = positions.astype(np.float32)
        self.fps = fps
        self.actor_id = actor_id
        self.name = name
        self.joint_names = joint_names or [f"joint_{i}" for i in range(positions.shape[1])]
        
        # Cached interpolators (built on demand)
        self._splines: Optional[Dict] = None
    
    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]
    
    @property
    def num_joints(self) -> int:
        return self.positions.shape[1]
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return (self.num_frames - 1) / self.fps
    
    @property
    def times(self) -> np.ndarray:
        """Frame timestamps."""
        return np.arange(self.num_frames, dtype=np.float32)
    
    def _build_splines(self):
        """Build cubic spline interpolators for each joint/dimension."""
        if self._splines is not None:
            return
        
        self._splines = {}
        times = self.times
        
        for j in range(self.num_joints):
            self._splines[j] = {}
            for d in range(3):
                self._splines[j][d] = CubicSpline(times, self.positions[:, j, d])
    
    def interpolate(self, t_query: Union[float, np.ndarray]) -> np.ndarray:
        """
        Query joint positions at arbitrary continuous time(s).
        
        Args:
            t_query: Scalar or array of time values in [0, num_frames-1]
        
        Returns:
            positions: (len(t_query), J, 3) or (J, 3) array of positions
        """
        self._build_splines()
        
        t_query = np.atleast_1d(t_query).astype(np.float32)
        result = np.zeros((len(t_query), self.num_joints, 3), dtype=np.float32)
        
        for j in range(self.num_joints):
            for d in range(3):
                result[:, j, d] = self._splines[j][d](t_query)
        
        # Return squeezed if single query
        if len(t_query) == 1:
            return result[0]
        return result
    
    def get_motion_func(self) -> Callable:
        """
        Return a callable function that queries motion at any time.
        
        Returns:
            motion_func: Callable that takes time(s) and returns positions
        """
        return self.interpolate
    
    def normalize(self, root_joint: int = 0, bone_scale: Optional[float] = None) -> 'MotionSequence':
        """
        Return a normalized copy of this sequence.
        
        Normalizes by:
        1. Centering root joint at origin (per frame)
        2. Scaling by average bone length
        
        Args:
            root_joint: Index of root joint for centering
            bone_scale: If provided, scale by this value. Otherwise compute from data.
        
        Returns:
            Normalized MotionSequence
        """
        positions = self.positions.copy()
        
        # Center root at origin for each frame
        root_positions = positions[:, root_joint:root_joint+1, :]  # (T, 1, 3)
        positions = positions - root_positions
        
        # Compute scale from bone lengths if not provided
        if bone_scale is None:
            # Use first bone (root to child) as reference
            if self.num_joints > 1:
                bone_vectors = positions[:, 1, :] - positions[:, 0, :]
                bone_scale = np.linalg.norm(bone_vectors, axis=1).mean()
            else:
                bone_scale = 1.0
        
        if bone_scale > 1e-6:
            positions = positions / bone_scale
        
        return MotionSequence(
            positions=positions,
            fps=self.fps,
            actor_id=self.actor_id,
            name=f"{self.name}_normalized",
            joint_names=self.joint_names
        )


# ============================================================================
# AMASS Dataset Loader
# ============================================================================

def load_amass_sequence(filepath: str, convert_to_positions: bool = True) -> Optional[MotionSequence]:
    """
    Load a single AMASS sequence from .npz file.
    
    AMASS stores data in SMPL format with:
    - 'poses': (T, 156) or (T, 72) axis-angle rotations
    - 'trans': (T, 3) root translations
    - 'betas': (16,) body shape parameters
    - 'gender': 'male', 'female', or 'neutral'
    
    Args:
        filepath: Path to .npz file
        convert_to_positions: If True, convert rotations to 3D positions
                             (requires smplx, otherwise uses placeholder)
    
    Returns:
        MotionSequence or None if loading fails
    """
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None
    
    # Get pose data
    if 'poses' not in data:
        print(f"No 'poses' key in {filepath}")
        return None
    
    poses = data['poses']  # (T, 156) or (T, 72)
    trans = data.get('trans', np.zeros((poses.shape[0], 3)))  # (T, 3)
    
    T = poses.shape[0]
    fps = data.get('mocap_framerate', 30.0)
    if isinstance(fps, np.ndarray):
        fps = float(fps)
    
    # Determine number of joints from pose dimension
    pose_dim = poses.shape[1]
    if pose_dim == 156:
        num_joints = 52
    elif pose_dim == 72:
        num_joints = 24
    else:
        num_joints = pose_dim // 3
    
    if convert_to_positions:
        # Try to use SMPL model for proper forward kinematics
        try:
            import smplx
            # This would require SMPL model files
            # For now, use simplified placeholder
            raise ImportError("SMPL conversion not implemented")
        except ImportError:
            # Simplified: treat axis-angle as pseudo-positions
            # This is NOT correct but allows testing without SMPL
            poses_reshaped = poses.reshape(T, -1, 3)
            
            # Create pseudo-positions from rotations + translation
            # In practice, you need proper forward kinematics
            positions = np.zeros((T, num_joints, 3), dtype=np.float32)
            
            # Place joints in a simple chain for visualization
            for j in range(min(num_joints, poses_reshaped.shape[1])):
                # Offset each joint from root based on joint index
                positions[:, j, 0] = trans[:, 0] + j * 0.1  # X offset
                positions[:, j, 1] = trans[:, 1]
                positions[:, j, 2] = trans[:, 2] + np.sin(poses_reshaped[:, j, 0]) * 0.1
    else:
        # Return rotations directly (for rotation-based models)
        positions = poses.reshape(T, -1, 3)
    
    # Extract actor ID from filename
    basename = os.path.basename(filepath)
    actor_id = hash(basename) % 1000
    
    return MotionSequence(
        positions=positions,
        fps=fps,
        actor_id=actor_id,
        name=basename
    )


def load_amass_dataset(
    amass_dir: str,
    max_sequences: Optional[int] = None,
    min_frames: int = 50
) -> List[MotionSequence]:
    """
    Load multiple sequences from AMASS dataset directory.
    
    Args:
        amass_dir: Path to AMASS dataset root
        max_sequences: Maximum number of sequences to load (None = all)
        min_frames: Minimum number of frames required
    
    Returns:
        List of MotionSequence objects
    """
    sequences = []
    
    # Find all .npz files recursively
    patterns = [
        os.path.join(amass_dir, "**/*.npz"),
        os.path.join(amass_dir, "*.npz")
    ]
    
    npz_files = []
    for pattern in patterns:
        npz_files.extend(glob.glob(pattern, recursive=True))
    
    npz_files = sorted(set(npz_files))
    
    if max_sequences:
        npz_files = npz_files[:max_sequences * 2]  # Load extra in case some fail
    
    print(f"Found {len(npz_files)} .npz files in {amass_dir}")
    
    for filepath in npz_files:
        if max_sequences and len(sequences) >= max_sequences:
            break
        
        seq = load_amass_sequence(filepath)
        if seq is not None and seq.num_frames >= min_frames:
            sequences.append(seq)
    
    print(f"Loaded {len(sequences)} valid sequences")
    return sequences


# ============================================================================
# Human3.6M Dataset Loader
# ============================================================================

def load_h36m_sequence(filepath: str) -> Optional[MotionSequence]:
    """
    Load a Human3.6M sequence.
    
    Supports various formats:
    - .npz with 'positions' or 'poses' key
    - .h5 with 'positions' dataset
    
    Args:
        filepath: Path to motion file
    
    Returns:
        MotionSequence or None if loading fails
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.npz':
            data = np.load(filepath, allow_pickle=True)
            
            # Try different keys
            for key in ['positions', 'poses', 'joints3d', 'keypoints3d']:
                if key in data:
                    positions = data[key]
                    break
            else:
                print(f"No position data found in {filepath}")
                return None
            
        elif ext == '.h5':
            import h5py
            with h5py.File(filepath, 'r') as f:
                positions = f['positions'][:]
        else:
            print(f"Unsupported format: {ext}")
            return None
        
        # Reshape if needed
        if positions.ndim == 2:
            # Assume (T, J*3) format
            T = positions.shape[0]
            J = positions.shape[1] // 3
            positions = positions.reshape(T, J, 3)
        
        return MotionSequence(
            positions=positions,
            fps=50.0,  # H3.6M is 50 Hz
            actor_id=hash(filepath) % 1000,
            name=os.path.basename(filepath)
        )
        
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None


# ============================================================================
# Synthetic Data Generator
# ============================================================================

def create_synthetic_motion(
    num_frames: int = 100,
    num_joints: int = 24,
    motion_type: str = "walking",
    fps: float = 30.0,
    actor_id: int = 0
) -> MotionSequence:
    """
    Create synthetic motion sequence for testing.
    
    Args:
        num_frames: Number of frames
        num_joints: Number of joints
        motion_type: Type of motion ("walking", "waving", "random")
        fps: Frames per second
        actor_id: Actor identifier
    
    Returns:
        MotionSequence with synthetic motion
    """
    T, J = num_frames, num_joints
    times = np.arange(T) / fps  # Time in seconds
    
    positions = np.zeros((T, J, 3), dtype=np.float32)
    
    if motion_type == "walking":
        # Simple walking cycle
        cycle_freq = 1.0  # Hz
        stride_length = 0.5  # meters
        
        for j in range(J):
            # Base position along skeleton chain
            base_x = (j // 2) * 0.15  # Alternating left/right
            base_y = (j % 6) * 0.2    # Height
            base_z = 0.0
            
            # Add walking motion
            phase = j * np.pi / 6  # Phase offset per joint
            
            positions[:, j, 0] = base_x + times * stride_length  # Forward motion
            positions[:, j, 1] = base_y + 0.05 * np.sin(2 * np.pi * cycle_freq * times + phase)
            positions[:, j, 2] = base_z + 0.03 * np.sin(4 * np.pi * cycle_freq * times + phase)
            
    elif motion_type == "waving":
        # Arm waving motion
        wave_freq = 2.0  # Hz
        
        for j in range(J):
            base_y = j * 0.1
            
            positions[:, j, 0] = 0.0
            positions[:, j, 1] = base_y
            
            # Waving in Z direction, amplitude varies with joint
            if j >= J // 2:  # Upper body
                positions[:, j, 2] = 0.3 * np.sin(2 * np.pi * wave_freq * times + j * 0.5)
            else:
                positions[:, j, 2] = 0.05 * np.sin(2 * np.pi * wave_freq * times)
                
    else:  # "random" or default
        # Smooth random motion using cumulative sum
        np.random.seed(actor_id)
        increments = np.random.randn(T, J, 3) * 0.02
        positions = np.cumsum(increments, axis=0)
        positions -= positions.mean(axis=0, keepdims=True)
    
    return MotionSequence(
        positions=positions,
        fps=fps,
        actor_id=actor_id,
        name=f"synthetic_{motion_type}_{actor_id}"
    )


def create_synthetic_dataset(
    num_sequences: int = 10,
    num_frames: int = 100,
    num_joints: int = 24,
    motion_types: Optional[List[str]] = None
) -> List[MotionSequence]:
    """
    Create a dataset of synthetic motion sequences.
    
    Args:
        num_sequences: Number of sequences to generate
        num_frames: Frames per sequence
        num_joints: Joints per skeleton
        motion_types: List of motion types (cycles through if shorter)
    
    Returns:
        List of MotionSequence objects
    """
    if motion_types is None:
        motion_types = ["walking", "waving", "random"]
    
    sequences = []
    for i in range(num_sequences):
        motion_type = motion_types[i % len(motion_types)]
        seq = create_synthetic_motion(
            num_frames=num_frames,
            num_joints=num_joints,
            motion_type=motion_type,
            actor_id=i
        )
        sequences.append(seq)
    
    return sequences


# ============================================================================
# Skeleton Normalizer
# ============================================================================

class SkeletonNormalizer:
    """
    Normalizes skeleton data across different sources.
    
    Handles:
    1. Zero-centering at root joint
    2. Scaling by reference bone length
    3. Mapping between different skeleton topologies
    """
    
    def __init__(
        self,
        target_num_joints: int = 24,
        root_joint: int = 0,
        reference_bone: Tuple[int, int] = (0, 1)
    ):
        """
        Initialize normalizer.
        
        Args:
            target_num_joints: Target number of joints
            root_joint: Index of root joint for centering
            reference_bone: Joint indices for computing reference bone length
        """
        self.target_num_joints = target_num_joints
        self.root_joint = root_joint
        self.reference_bone = reference_bone
        
        # Statistics computed from data
        self.mean_bone_length: Optional[float] = None
        self.joint_means: Optional[np.ndarray] = None
    
    def fit(self, sequences: List[MotionSequence]):
        """
        Compute normalization statistics from data.
        
        Args:
            sequences: List of motion sequences to fit on
        """
        bone_lengths = []
        
        for seq in sequences:
            # Compute bone lengths
            j1, j2 = self.reference_bone
            if j1 < seq.num_joints and j2 < seq.num_joints:
                bones = seq.positions[:, j2, :] - seq.positions[:, j1, :]
                lengths = np.linalg.norm(bones, axis=1)
                bone_lengths.extend(lengths.tolist())
        
        if bone_lengths:
            self.mean_bone_length = np.mean(bone_lengths)
        else:
            self.mean_bone_length = 1.0
        
        print(f"SkeletonNormalizer: mean bone length = {self.mean_bone_length:.4f}")
    
    def normalize_sequence(self, seq: MotionSequence) -> MotionSequence:
        """
        Normalize a single sequence.
        
        Args:
            seq: Input motion sequence
        
        Returns:
            Normalized motion sequence
        """
        if self.mean_bone_length is None:
            self.fit([seq])
        
        return seq.normalize(
            root_joint=self.root_joint,
            bone_scale=self.mean_bone_length
        )
    
    def normalize_all(self, sequences: List[MotionSequence]) -> List[MotionSequence]:
        """
        Normalize multiple sequences.
        
        Args:
            sequences: List of input sequences
        
        Returns:
            List of normalized sequences
        """
        if self.mean_bone_length is None:
            self.fit(sequences)
        
        return [self.normalize_sequence(seq) for seq in sequences]


# ============================================================================
# PyTorch Dataset
# ============================================================================

class MotionDataset(Dataset):
    """
    PyTorch Dataset for training Geom-INR-Motion.
    
    Each sample returns:
    - actor_id: (B,) long tensor of actor IDs
    - joint_idx: (B,) long tensor of joint indices
    - time: (B,) float tensor of continuous time values
    - target: (B, 3) float tensor of target positions
    
    Samples are drawn by:
    1. Randomly selecting a sequence
    2. Randomly selecting time points within that sequence
    3. Returning all joints at those time points
    """
    
    def __init__(
        self,
        sequences: List[MotionSequence],
        samples_per_sequence: int = 100,
        time_window: Optional[int] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of MotionSequence objects
            samples_per_sequence: Number of training samples per sequence
            time_window: If set, sample contiguous windows of this length
            normalize: Whether to normalize sequences
        """
        self.samples_per_sequence = samples_per_sequence
        self.time_window = time_window
        
        # Normalize sequences
        if normalize:
            normalizer = SkeletonNormalizer()
            self.sequences = normalizer.normalize_all(sequences)
        else:
            self.sequences = sequences
        
        # Build interpolators
        self.motion_funcs = [seq.get_motion_func() for seq in self.sequences]
        
        # Precompute metadata
        self.num_sequences = len(sequences)
        self.num_joints = sequences[0].num_joints if sequences else 24
        self.actor_ids = [seq.actor_id for seq in self.sequences]
        self.num_frames = [seq.num_frames for seq in self.sequences]
    
    def __len__(self) -> int:
        return self.num_sequences * self.samples_per_sequence
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a training sample.
        
        Returns:
            actor_ids: (J,) tensor of actor IDs (repeated for each joint)
            joint_idxs: (J,) tensor of joint indices
            times: (J,) tensor of time values (same time, all joints)
            targets: (J, 3) tensor of target positions
        """
        # Determine which sequence
        seq_idx = idx // self.samples_per_sequence
        seq_idx = seq_idx % self.num_sequences
        
        seq = self.sequences[seq_idx]
        motion_func = self.motion_funcs[seq_idx]
        
        # Sample a random time
        max_time = seq.num_frames - 1
        t = np.random.uniform(0, max_time)
        
        # Get positions for all joints at this time
        positions = motion_func(t)  # (J, 3)
        
        # Create tensors
        J = seq.num_joints
        actor_ids = torch.full((J,), seq.actor_id, dtype=torch.long)
        joint_idxs = torch.arange(J, dtype=torch.long)
        times = torch.full((J,), t, dtype=torch.float32)
        targets = torch.from_numpy(positions).float()
        
        return actor_ids, joint_idxs, times, targets
    
    @classmethod
    def from_synthetic(
        cls,
        num_sequences: int = 10,
        num_frames: int = 100,
        num_joints: int = 24,
        **kwargs
    ) -> 'MotionDataset':
        """
        Create dataset from synthetic motion.
        
        Args:
            num_sequences: Number of sequences
            num_frames: Frames per sequence
            num_joints: Joints per skeleton
            **kwargs: Additional arguments for MotionDataset
        
        Returns:
            MotionDataset with synthetic data
        """
        sequences = create_synthetic_dataset(
            num_sequences=num_sequences,
            num_frames=num_frames,
            num_joints=num_joints
        )
        return cls(sequences, **kwargs)
    
    @classmethod
    def from_amass(
        cls,
        amass_dir: str,
        max_sequences: Optional[int] = None,
        **kwargs
    ) -> 'MotionDataset':
        """
        Create dataset from AMASS data.
        
        Args:
            amass_dir: Path to AMASS dataset
            max_sequences: Maximum sequences to load
            **kwargs: Additional arguments for MotionDataset
        
        Returns:
            MotionDataset with AMASS data
        """
        sequences = load_amass_dataset(amass_dir, max_sequences=max_sequences)
        if not sequences:
            print("Warning: No AMASS sequences loaded, using synthetic data")
            return cls.from_synthetic(**kwargs)
        return cls(sequences, **kwargs)


# ============================================================================
# Collate Function for DataLoader
# ============================================================================

def motion_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for MotionDataset.
    
    Flattens all joints from all samples into a single batch.
    
    Args:
        batch: List of (actor_ids, joint_idxs, times, targets) tuples
    
    Returns:
        Batched tensors
    """
    actor_ids = torch.cat([b[0] for b in batch], dim=0)
    joint_idxs = torch.cat([b[1] for b in batch], dim=0)
    times = torch.cat([b[2] for b in batch], dim=0)
    targets = torch.cat([b[3] for b in batch], dim=0)
    
    return actor_ids, joint_idxs, times, targets


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Data Pipeline Test")
    print("=" * 60)
    
    # Test synthetic data
    print("\n[1] Creating synthetic dataset...")
    sequences = create_synthetic_dataset(num_sequences=5, num_frames=100, num_joints=24)
    print(f"    Created {len(sequences)} sequences")
    print(f"    First sequence: {sequences[0].num_frames} frames, {sequences[0].num_joints} joints")
    
    # Test interpolation
    print("\n[2] Testing interpolation...")
    seq = sequences[0]
    motion_func = seq.get_motion_func()
    
    # Query at fractional timestamps
    test_times = [0.0, 5.5, 25.75, 50.0, 99.0]
    for t in test_times:
        pos = motion_func(t)
        print(f"    t={t:5.2f}: joint 0 position = [{pos[0, 0]:.3f}, {pos[0, 1]:.3f}, {pos[0, 2]:.3f}]")
    
    # Test normalization
    print("\n[3] Testing normalization...")
    normalizer = SkeletonNormalizer()
    normalized = normalizer.normalize_all(sequences)
    print(f"    Normalized {len(normalized)} sequences")
    print(f"    Mean bone length: {normalizer.mean_bone_length:.4f}")
    
    # Test PyTorch Dataset
    print("\n[4] Testing PyTorch Dataset...")
    dataset = MotionDataset(sequences, samples_per_sequence=10)
    print(f"    Dataset size: {len(dataset)}")
    
    actor_ids, joint_idxs, times, targets = dataset[0]
    print(f"    Sample shape: actor_ids={actor_ids.shape}, joints={joint_idxs.shape}, "
          f"times={times.shape}, targets={targets.shape}")
    
    # Test DataLoader
    print("\n[5] Testing DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=motion_collate_fn
    )
    
    batch = next(iter(loader))
    print(f"    Batch shapes: {[b.shape for b in batch]}")
    
    print("\n✓ Data pipeline test completed!")
