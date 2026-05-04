"""
Heightfield extraction + mesh construction used by both the Prompt-1 export
pipeline and the Prompt-2 Taichi viewer.

No Taichi import lives in this module so it can be consumed from pure-numpy
contexts (reference-data stacking, metric computation, offline checks). The
torch import is lazy and only required when ``extract_heightfield_from_model``
is called with ``--live_query``.

Conventions
-----------
- ``sdf`` volumes are ordered ``(Nx, Ny, Nz)`` with the same per-axis
  ``coords`` array applied along every axis. This matches how
  ``create_drape_over_sphere_sdf`` and ``export_validation.query_volume``
  write their outputs.
- ``heights[i, j]`` is the zero-crossing ``z`` value for column ``(x=coords[i],
  y=coords[j])``. When no sign change exists along that column the
  ``argmin(|sdf|)`` voxel z is used as a fallback, which matches the historical
  extractor and gives a sensible boundary result.
- ``build_heightfield_mesh`` returns vertices with the **source axis order**
  ``(x, y, z=height)``. The viewer remaps to its own y-up convention; the
  extractor stays source-native.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def extract_heightfield_from_volume(
    sdf: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Linear zero-crossing along ``z`` for every ``(x, y)`` column.

    Args:
        sdf: ``(..., Nx, Ny, Nz)`` float array of SDF values on a regular grid.
        coords: ``(Nz,)`` float array of z coordinates (assumed identical across
            all three axes by the reference pipeline).

    Returns:
        heights: ``(..., Nx, Ny)`` float32 — the interpolated z of the zero
        level-set along each column.

    Implementation notes
    --------------------
    ``np.sign(a) * np.sign(b) < 0`` catches any strict sign change, including
    transitions through exact zero. ``np.argmax`` along the last axis returns
    the index of the first ``True`` — the first sign change from top to
    bottom. Linear interpolation between the two bracketing SDF values refines
    to sub-voxel precision. Columns without any sign change fall back to
    ``argmin(|sdf|)`` so boundary columns never produce NaN.
    """
    coords = np.asarray(coords, dtype=np.float32)
    nz = int(sdf.shape[-1])
    if coords.shape[0] != nz:
        raise ValueError(
            f"coords length {coords.shape[0]} does not match sdf z-axis {nz}"
        )

    abs_sdf = np.abs(sdf)
    fallback_idx = np.argmin(abs_sdf, axis=-1)
    z = coords[fallback_idx].astype(np.float32)

    s = np.sign(sdf)
    crossings = s[..., :-1] * s[..., 1:] < 0
    has_cross = crossings.any(axis=-1)
    if not has_cross.any():
        return z.astype(np.float32)

    first_cross = np.argmax(crossings, axis=-1)
    i0 = first_cross
    i1 = first_cross + 1
    s0 = np.take_along_axis(sdf, i0[..., None], axis=-1)[..., 0]
    s1 = np.take_along_axis(sdf, i1[..., None], axis=-1)[..., 0]
    z0 = coords[i0]
    z1 = coords[i1]
    denom = s1 - s0
    safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    z_interp = z0 - s0 * (z1 - z0) / safe
    return np.where(has_cross, z_interp, z).astype(np.float32)


def extract_heightfield_from_model(
    model,
    t: float,
    coords: np.ndarray,
    device=None,
    z_samples: int = 96,
    z_bounds: Optional[Tuple[float, float]] = None,
    chunk: int = 65536,
) -> np.ndarray:
    """
    Query a trained FourierFeatureMLP on a ``(Nx, Ny, K_z)`` lattice at a
    single time value ``t`` and extract the heightfield via the shared
    zero-crossing routine.

    Args:
        model: ``FourierFeatureMLP`` (GRU off expected for hero path). Should
            already be in ``eval`` mode.
        t: scalar time value in the same normalization as training.
        coords: ``(N,)`` xy axis coordinates. Used for both x and y — the
            lattice is square.
        device: torch device; defaults to the model's current device.
        z_samples: number of z taps per column. 96 is enough for
            ``bounds=(-1, 1)`` given the bump's 0.6 vertical range (pitch
            ≈ 0.021, comfortably below the cloth thickness).
        z_bounds: z range to sample. Defaults to ``(coords[0], coords[-1])``.
        chunk: forward-pass batch size.

    Returns:
        heights: ``(N, N)`` float32 heightfield.
    """
    import torch  # local import keeps the module torch-free by default

    coords = np.asarray(coords, dtype=np.float32)
    if device is None:
        device = next(model.parameters()).device

    if z_bounds is None:
        z_bounds = (float(coords[0]), float(coords[-1]))
    zs = np.linspace(z_bounds[0], z_bounds[1], z_samples, dtype=np.float32)

    X, Y, Z = np.meshgrid(coords, coords, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

    out = np.empty(pts.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            xyz = torch.from_numpy(pts[i:i + chunk]).to(device)
            tt = torch.full((xyz.shape[0], 1), float(t), device=device)
            sdf = model.forward_batch(xyz, tt).squeeze(-1)
            out[i:i + chunk] = sdf.detach().cpu().numpy()
    sdf_vol = out.reshape(coords.shape[0], coords.shape[0], z_samples)
    return extract_heightfield_from_volume(sdf_vol, zs)


def build_heightfield_mesh(
    heights: np.ndarray,
    coords_x: Optional[np.ndarray] = None,
    coords_y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a fixed-topology triangle mesh from a heightfield.

    Args:
        heights: ``(Nx, Ny)`` z values.
        coords_x: ``(Nx,)`` x coordinates. Defaults to ``linspace(-1, 1, Nx)``.
        coords_y: ``(Ny,)`` y coordinates. Defaults to ``linspace(-1, 1, Ny)``.

    Returns:
        vertices: ``(Nx * Ny, 3)`` float32 — columns ``(x, y, z=height)``.
        faces: ``(2 * (Nx - 1) * (Ny - 1), 3)`` int32 — two triangles per
            quad, CCW when viewed from +z.
        line_indices: ``(E,)`` int32 — flat edge index pairs suitable for
            ``ti.ui.Scene.lines``. One pair per unique grid edge.

    Notes
    -----
    Topology depends only on the grid resolution, not on the heights, so
    ``faces`` / ``line_indices`` can be computed once and reused across frames
    during playback.
    """
    heights = np.asarray(heights, dtype=np.float32)
    if heights.ndim != 2:
        raise ValueError(f"heights must be 2D, got shape {heights.shape}")
    nx, ny = heights.shape

    if coords_x is None:
        coords_x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    if coords_y is None:
        coords_y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)

    X, Y = np.meshgrid(coords_x, coords_y, indexing="ij")
    vertices = np.stack([X, Y, heights], axis=-1).reshape(-1, 3).astype(np.float32)

    # Triangle indices: two triangles per quad (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1).
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    tl = (i * ny + j).ravel()
    tr = (i * ny + (j + 1)).ravel()
    bl = ((i + 1) * ny + j).ravel()
    br = ((i + 1) * ny + (j + 1)).ravel()
    faces_tri_a = np.stack([tl, bl, tr], axis=-1)           # (quad_count, 3)
    faces_tri_b = np.stack([tr, bl, br], axis=-1)
    faces = np.concatenate([faces_tri_a, faces_tri_b], axis=0).astype(np.int32)

    # Line indices (unique edges). Horizontal edges along y, vertical along x.
    h_i, h_j = np.meshgrid(np.arange(nx), np.arange(ny - 1), indexing="ij")
    horiz = np.stack(
        [(h_i * ny + h_j).ravel(), (h_i * ny + h_j + 1).ravel()], axis=-1
    )
    v_i, v_j = np.meshgrid(np.arange(nx - 1), np.arange(ny), indexing="ij")
    vert = np.stack(
        [(v_i * ny + v_j).ravel(), ((v_i + 1) * ny + v_j).ravel()], axis=-1
    )
    line_pairs = np.concatenate([horiz, vert], axis=0).astype(np.int32)
    line_indices = line_pairs.reshape(-1)

    return vertices, faces, line_indices
