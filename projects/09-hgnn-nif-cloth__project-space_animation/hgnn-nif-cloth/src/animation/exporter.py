"""
Animation Export Pipeline for HGNN-NIF-Cloth

Provides utilities for exporting cloth animations to various formats:
- OBJ sequence (for rendering software)
- NumPy arrays (for further processing)
- MP4 video (for quick preview)
- USD/Alembic (for professional pipelines)
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, List, Union
import json


class AnimationExporter:
    """
    Export cloth animation to various formats.

    Supports:
    - OBJ sequence: Individual OBJ files per frame
    - NumPy arrays: Efficient storage for processing
    - MP4 video: Quick preview rendering
    - Metadata: JSON with animation parameters

    Args:
        output_dir: Base directory for all exports

    Example:
        >>> exporter = AnimationExporter('output/animation')
        >>> exporter.export_obj_sequence(positions, faces)
        >>> exporter.export_video(positions, faces, fps=30)
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_obj_sequence(
        self,
        positions_sequence: np.ndarray,
        faces: np.ndarray,
        prefix: str = "cloth",
        normals: Optional[np.ndarray] = None,
        uvs: Optional[np.ndarray] = None
    ) -> Path:
        """
        Export animation as OBJ sequence.

        Args:
            positions_sequence: (T, N, 3) vertex positions over time
            faces: (F, 3) face indices
            prefix: Filename prefix
            normals: Optional (T, N, 3) or (N, 3) vertex normals
            uvs: Optional (N, 2) UV coordinates

        Returns:
            Path to output directory
        """
        obj_dir = self.output_dir / "obj_sequence"
        obj_dir.mkdir(exist_ok=True)

        T = len(positions_sequence)
        for t, positions in enumerate(positions_sequence):
            filename = obj_dir / f"{prefix}_{t:05d}.obj"

            # Get normals for this frame
            frame_normals = None
            if normals is not None:
                if normals.ndim == 3:
                    frame_normals = normals[t]
                else:
                    frame_normals = normals

            self._write_obj(filename, positions, faces, frame_normals, uvs)

            if (t + 1) % 100 == 0:
                print(f"Exported frame {t + 1}/{T}")

        print(f"Exported {T} OBJ files to {obj_dir}")
        return obj_dir

    def _write_obj(
        self,
        filename: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        uvs: Optional[np.ndarray] = None
    ):
        """Write single OBJ file."""
        with open(filename, 'w') as f:
            f.write("# HGNN-NIF-Cloth Animation Export\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write UVs if provided
            if uvs is not None:
                f.write("\n")
                for uv in uvs:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Write normals if provided
            if normals is not None:
                f.write("\n")
                for n in normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write faces (OBJ is 1-indexed)
            f.write("\n")
            has_uvs = uvs is not None
            has_normals = normals is not None

            for face in faces:
                if has_uvs and has_normals:
                    f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                           f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                           f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                elif has_uvs:
                    f.write(f"f {face[0]+1}/{face[0]+1} "
                           f"{face[1]+1}/{face[1]+1} "
                           f"{face[2]+1}/{face[2]+1}\n")
                elif has_normals:
                    f.write(f"f {face[0]+1}//{face[0]+1} "
                           f"{face[1]+1}//{face[1]+1} "
                           f"{face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def export_numpy(
        self,
        positions_sequence: np.ndarray,
        velocities_sequence: Optional[np.ndarray] = None,
        sdf_sequence: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        compress: bool = True
    ) -> Path:
        """
        Export raw numpy arrays for further processing.

        Args:
            positions_sequence: (T, N, 3) vertex positions
            velocities_sequence: Optional (T, N, 3) velocities
            sdf_sequence: Optional (T, R, R, R) SDF volumes
            faces: Optional (F, 3) face indices
            edges: Optional (2, E) edge indices
            compress: Use compressed npz format

        Returns:
            Path to saved file
        """
        data = {'positions': positions_sequence}

        if velocities_sequence is not None:
            data['velocities'] = velocities_sequence
        if sdf_sequence is not None:
            data['sdf'] = sdf_sequence
        if faces is not None:
            data['faces'] = faces
        if edges is not None:
            data['edges'] = edges

        if compress:
            output_path = self.output_dir / "animation_data.npz"
            np.savez_compressed(output_path, **data)
        else:
            output_path = self.output_dir / "animation_data.npz"
            np.savez(output_path, **data)

        # Also save individual files for convenience
        np.save(self.output_dir / "positions.npy", positions_sequence)
        if velocities_sequence is not None:
            np.save(self.output_dir / "velocities.npy", velocities_sequence)

        print(f"Exported numpy arrays to {self.output_dir}")
        return output_path

    def export_video(
        self,
        positions_sequence: np.ndarray,
        faces: np.ndarray,
        filename: str = "animation.mp4",
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        camera_angle: Tuple[float, float] = (30, 45),
        show_edges: bool = True,
        colormap: str = 'viridis'
    ) -> Path:
        """
        Render animation to MP4 video using matplotlib.

        Args:
            positions_sequence: (T, N, 3) vertex positions
            faces: (F, 3) face indices
            filename: Output filename
            fps: Frames per second
            resolution: (width, height) in pixels
            camera_angle: (elevation, azimuth) in degrees
            show_edges: Draw mesh edges
            colormap: Matplotlib colormap for height coloring

        Returns:
            Path to video file

        Requires: matplotlib, ffmpeg
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            raise ImportError("Video export requires matplotlib. Install with: pip install matplotlib")

        # Compute bounds across all frames
        all_positions = positions_sequence.reshape(-1, 3)
        min_bounds = all_positions.min(axis=0)
        max_bounds = all_positions.max(axis=0)
        center = (min_bounds + max_bounds) / 2
        extent = (max_bounds - min_bounds).max() / 2 * 1.2

        fig = plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        def init():
            ax.set_xlim(center[0] - extent, center[0] + extent)
            ax.set_ylim(center[1] - extent, center[1] + extent)
            ax.set_zlim(center[2] - extent, center[2] + extent)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=camera_angle[0], azim=camera_angle[1])
            return []

        def update(frame):
            ax.clear()
            init()

            verts = positions_sequence[frame]
            triangles = verts[faces]

            # Color by height (Z coordinate)
            z_values = verts[:, 2]
            face_z = z_values[faces].mean(axis=1)
            face_colors = plt.cm.get_cmap(colormap)(
                (face_z - face_z.min()) / (face_z.max() - face_z.min() + 1e-8)
            )

            mesh = Poly3DCollection(triangles, alpha=0.9)
            mesh.set_facecolor(face_colors)
            if show_edges:
                mesh.set_edgecolor('darkgray')
                mesh.set_linewidth(0.3)
            ax.add_collection3d(mesh)

            ax.set_title(f'Frame {frame + 1}/{len(positions_sequence)}')
            return []

        print(f"Rendering {len(positions_sequence)} frames...")
        ani = animation.FuncAnimation(
            fig, update, frames=len(positions_sequence),
            init_func=init, blit=False, interval=1000 / fps
        )

        output_path = self.output_dir / filename
        try:
            ani.save(str(output_path), writer='ffmpeg', fps=fps,
                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        except Exception as e:
            print(f"FFmpeg failed: {e}")
            print("Trying with pillow writer...")
            gif_path = self.output_dir / filename.replace('.mp4', '.gif')
            ani.save(str(gif_path), writer='pillow', fps=fps)
            output_path = gif_path

        plt.close()

        print(f"Exported video to {output_path}")
        return output_path

    def export_metadata(
        self,
        num_frames: int,
        num_vertices: int,
        num_faces: int,
        fps: int = 30,
        model_config: Optional[dict] = None,
        extra_info: Optional[dict] = None
    ) -> Path:
        """
        Export animation metadata as JSON.

        Args:
            num_frames: Total number of frames
            num_vertices: Number of mesh vertices
            num_faces: Number of mesh faces
            fps: Animation frame rate
            model_config: Model configuration dict
            extra_info: Additional metadata

        Returns:
            Path to metadata file
        """
        metadata = {
            'animation': {
                'num_frames': num_frames,
                'fps': fps,
                'duration_seconds': num_frames / fps
            },
            'mesh': {
                'num_vertices': num_vertices,
                'num_faces': num_faces
            }
        }

        if model_config:
            metadata['model'] = model_config
        if extra_info:
            metadata['extra'] = extra_info

        output_path = self.output_dir / "metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return output_path

    def export_ply_sequence(
        self,
        positions_sequence: np.ndarray,
        faces: np.ndarray,
        prefix: str = "cloth",
        colors: Optional[np.ndarray] = None
    ) -> Path:
        """
        Export animation as PLY sequence (supports vertex colors).

        Args:
            positions_sequence: (T, N, 3) vertex positions
            faces: (F, 3) face indices
            prefix: Filename prefix
            colors: Optional (T, N, 3) or (N, 3) RGB colors [0-255]

        Returns:
            Path to output directory
        """
        ply_dir = self.output_dir / "ply_sequence"
        ply_dir.mkdir(exist_ok=True)

        T = len(positions_sequence)
        for t, positions in enumerate(positions_sequence):
            filename = ply_dir / f"{prefix}_{t:05d}.ply"

            frame_colors = None
            if colors is not None:
                frame_colors = colors[t] if colors.ndim == 3 else colors

            self._write_ply(filename, positions, faces, frame_colors)

        print(f"Exported {T} PLY files to {ply_dir}")
        return ply_dir

    def _write_ply(
        self,
        filename: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray] = None
    ):
        """Write single PLY file."""
        has_colors = colors is not None

        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Vertices
            for i, v in enumerate(vertices):
                if has_colors:
                    c = colors[i].astype(int)
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
                else:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


class MeshGenerator:
    """Generate cloth mesh topology for testing and demos."""

    @staticmethod
    def create_grid_mesh(
        rows: int = 20,
        cols: int = 20,
        size: float = 2.0,
        center: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a planar grid mesh.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            size: Total size of the mesh
            center: Center position (x, y, z)

        Returns:
            vertices: (N, 3) initial positions
            faces: (F, 3) triangle indices
            edges: (2, E) edge indices
        """
        # Create vertices
        x = np.linspace(-size / 2, size / 2, cols) + center[0]
        y = np.linspace(-size / 2, size / 2, rows) + center[1]
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, center[2])
        vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        # Create faces (two triangles per quad)
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v00 = i * cols + j
                v01 = i * cols + j + 1
                v10 = (i + 1) * cols + j
                v11 = (i + 1) * cols + j + 1
                faces.append([v00, v01, v11])
                faces.append([v00, v11, v10])
        faces = np.array(faces, dtype=np.int64)

        # Create unique edges
        edge_set = set()
        for face in faces:
            for k in range(3):
                e = tuple(sorted([face[k], face[(k + 1) % 3]]))
                edge_set.add(e)
        edges = np.array(list(edge_set), dtype=np.int64).T

        return vertices.astype(np.float32), faces, edges

    @staticmethod
    def create_uv_coordinates(rows: int, cols: int) -> np.ndarray:
        """
        Create UV coordinates for a grid mesh.

        Args:
            rows: Number of rows
            cols: Number of columns

        Returns:
            uvs: (N, 2) UV coordinates in [0, 1]
        """
        u = np.linspace(0, 1, cols)
        v = np.linspace(0, 1, rows)
        uu, vv = np.meshgrid(u, v)
        uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        return uvs.astype(np.float32)

    @staticmethod
    def compute_vertex_normals(
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Compute vertex normals from mesh geometry.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices

        Returns:
            normals: (N, 3) normalized vertex normals
        """
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = np.cross(e1, e2)

        # Accumulate face normals to vertices
        vertex_normals = np.zeros_like(vertices)
        for i, face in enumerate(faces):
            for v_idx in face:
                vertex_normals[v_idx] += face_normals[i]

        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
        vertex_normals = vertex_normals / (norms + 1e-8)

        return vertex_normals.astype(np.float32)

    @staticmethod
    def compute_rest_lengths(
        vertices: np.ndarray,
        edges: np.ndarray
    ) -> np.ndarray:
        """
        Compute rest lengths for all edges.

        Args:
            vertices: (N, 3) vertex positions
            edges: (2, E) edge indices

        Returns:
            rest_lengths: (E,) edge rest lengths
        """
        v0 = vertices[edges[0]]
        v1 = vertices[edges[1]]
        rest_lengths = np.linalg.norm(v1 - v0, axis=-1)
        return rest_lengths.astype(np.float32)

    @staticmethod
    def decimate_grid(
        vertices: np.ndarray,
        rows: int,
        cols: int,
        stride: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a coarser version of a grid mesh.

        Args:
            vertices: (N, 3) fine mesh vertices
            rows: Number of rows in fine mesh
            cols: Number of columns in fine mesh
            stride: Decimation stride

        Returns:
            coarse_vertices: (N_coarse, 3) coarse mesh vertices
            coarse_faces: (F_coarse, 3) coarse mesh faces
            coarse_edges: (2, E_coarse) coarse mesh edges
        """
        # Subsample vertices
        grid = vertices.reshape(rows, cols, 3)
        coarse_grid = grid[::stride, ::stride, :]
        coarse_rows, coarse_cols = coarse_grid.shape[:2]
        coarse_vertices = coarse_grid.reshape(-1, 3)

        # Create coarse faces
        coarse_faces = []
        for i in range(coarse_rows - 1):
            for j in range(coarse_cols - 1):
                v00 = i * coarse_cols + j
                v01 = i * coarse_cols + j + 1
                v10 = (i + 1) * coarse_cols + j
                v11 = (i + 1) * coarse_cols + j + 1
                coarse_faces.append([v00, v01, v11])
                coarse_faces.append([v00, v11, v10])
        coarse_faces = np.array(coarse_faces, dtype=np.int64)

        # Create coarse edges
        edge_set = set()
        for face in coarse_faces:
            for k in range(3):
                e = tuple(sorted([face[k], face[(k + 1) % 3]]))
                edge_set.add(e)
        coarse_edges = np.array(list(edge_set), dtype=np.int64).T

        return coarse_vertices.astype(np.float32), coarse_faces, coarse_edges
