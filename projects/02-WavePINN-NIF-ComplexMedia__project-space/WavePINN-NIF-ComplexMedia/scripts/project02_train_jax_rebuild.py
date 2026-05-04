#!/usr/bin/env python3
"""Train a real JAX/Haiku WavePINN-NIF surrogate for Project 02."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
REFERENCE_DIR = PROJECT_ROOT / "outputs" / "project02_reference"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "project02_validation_model"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PRIVATE_DIR = PROJECT_ROOT / "private"
APPLY_RNG = jax.random.PRNGKey(0)

from project02_fd_reference import save_medium, save_snapshots, save_slice_xt
from src.model import count_parameters, create_model


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def psnr_db(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1.0e-30:
        return 120.0
    peak = float(np.max(target) - np.min(target))
    peak = max(peak, 1.0e-12)
    return float(10.0 * math.log10((peak * peak) / mse))


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    denom = float(np.mean(target ** 2)) + 1.0e-30
    return mse / denom


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1.0e-30
    return float(np.dot(aa, bb) / denom)


def interpolate_medium(medium: np.ndarray, xy: np.ndarray) -> np.ndarray:
    ny, nx = medium.shape
    x = np.clip(xy[:, 0], 0.0, 1.0) * (nx - 1)
    y = np.clip(xy[:, 1], 0.0, 1.0) * (ny - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    wx = x - x0
    wy = y - y0
    v00 = medium[y0, x0]
    v10 = medium[y0, x1]
    v01 = medium[y1, x0]
    v11 = medium[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    ).astype(np.float32)


def interpolate_wavefield(wavefield: np.ndarray, times: np.ndarray, coords: np.ndarray) -> np.ndarray:
    n_frames, ny, nx = wavefield.shape
    x = np.clip(coords[:, 0], 0.0, 1.0) * (nx - 1)
    y = np.clip(coords[:, 1], 0.0, 1.0) * (ny - 1)
    t = np.clip(coords[:, 2], float(times[0]), float(times[-1]))

    ti = np.searchsorted(times, t, side="right") - 1
    ti = np.clip(ti, 0, n_frames - 2)
    t0 = times[ti]
    t1 = times[ti + 1]
    wt = (t - t0) / np.maximum(t1 - t0, 1.0e-12)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    wx = x - x0
    wy = y - y0

    def bilinear(frame_indices: np.ndarray) -> np.ndarray:
        field = wavefield[frame_indices]
        v00 = field[np.arange(len(coords)), y0, x0]
        v10 = field[np.arange(len(coords)), y0, x1]
        v01 = field[np.arange(len(coords)), y1, x0]
        v11 = field[np.arange(len(coords)), y1, x1]
        return (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v10
            + (1.0 - wx) * wy * v01
            + wx * wy * v11
        )

    return ((1.0 - wt) * bilinear(ti) + wt * bilinear(ti + 1)).astype(np.float32)


def ricker_jax(t: jnp.ndarray, frequency: float, t0: float) -> jnp.ndarray:
    arg = (jnp.pi * frequency * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * jnp.exp(-arg)


def source_term_jax(x: jnp.ndarray, source: dict[str, Any]) -> jnp.ndarray:
    pos = jnp.asarray(source["position"], dtype=x.dtype)
    sigma = jnp.asarray(source["sigma"], dtype=x.dtype)
    radius2 = jnp.sum((x[..., :2] - pos[None, :]) ** 2, axis=-1)
    spatial = jnp.exp(-radius2 / (2.0 * sigma * sigma)) * jnp.asarray(source["spatial_normalization"], dtype=x.dtype)
    temporal = ricker_jax(x[..., 2], source["frequency"], source["t0"])
    return jnp.asarray(source["amplitude"], dtype=x.dtype) * spatial * temporal


def make_model_config(args: argparse.Namespace, c_min: float, c_max: float) -> dict[str, Any]:
    return {
        "wave": {
            "hidden_dims": [args.hidden_dim] * args.hidden_layers,
            "use_fourier": True,
            "fourier_dim": args.fourier_dim,
            "fourier_sigma": args.fourier_sigma,
            "activation": "tanh",
            "use_residual": True,
        },
        "media": {
            "hidden_dims": [args.media_hidden_dim] * args.media_hidden_layers,
            "activation": "softplus",
            "c_min": float(c_min),
            "c_max": float(c_max),
            "use_fourier": True,
            "fourier_dim": args.media_fourier_dim,
            "fourier_sigma": args.media_fourier_sigma,
        },
    }


def sample_batch(
    rng: np.random.Generator,
    wavefield: np.ndarray,
    medium: np.ndarray,
    times: np.ndarray,
    train_frames: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    ny, nx = medium.shape

    frame_idx = rng.choice(train_frames, size=args.batch_data, replace=True)
    x_idx = rng.integers(0, nx, size=args.batch_data)
    y_idx = rng.integers(0, ny, size=args.batch_data)
    x_data = np.stack([x_grid[x_idx], y_grid[y_idx], times[frame_idx]], axis=-1).astype(np.float32)
    u_data = wavefield[frame_idx, y_idx, x_idx].astype(np.float32)

    if args.batch_velocity > 0:
        velocity_frames = train_frames[(train_frames > 0) & (train_frames < train_frames[-1])]
        v_frame_idx = rng.choice(velocity_frames, size=args.batch_velocity, replace=True)
        v_x_idx = rng.integers(0, nx, size=args.batch_velocity)
        v_y_idx = rng.integers(0, ny, size=args.batch_velocity)
        x_velocity = np.stack([x_grid[v_x_idx], y_grid[v_y_idx], times[v_frame_idx]], axis=-1).astype(np.float32)
        dt_v = (times[v_frame_idx + 1] - times[v_frame_idx - 1]).astype(np.float32)
        v_data = (
            (wavefield[v_frame_idx + 1, v_y_idx, v_x_idx] - wavefield[v_frame_idx - 1, v_y_idx, v_x_idx])
            / np.maximum(dt_v, 1.0e-12)
        ).astype(np.float32)
    else:
        x_velocity = np.zeros((0, 3), dtype=np.float32)
        v_data = np.zeros((0,), dtype=np.float32)

    m_x_idx = rng.integers(0, nx, size=args.batch_medium)
    m_y_idx = rng.integers(0, ny, size=args.batch_medium)
    x_media = np.stack([x_grid[m_x_idx], y_grid[m_y_idx]], axis=-1).astype(np.float32)
    c_data = medium[m_y_idx, m_x_idx].astype(np.float32)

    pde_xy = rng.uniform(0.04, 0.96, size=(args.batch_pde, 2)).astype(np.float32)
    pde_t = rng.uniform(float(times[0]), float(times[-1]), size=(args.batch_pde, 1)).astype(np.float32)
    x_pde = np.concatenate([pde_xy, pde_t], axis=-1)

    ic_x_idx = rng.integers(0, nx, size=args.batch_ic)
    ic_y_idx = rng.integers(0, ny, size=args.batch_ic)
    x_ic = np.stack([x_grid[ic_x_idx], y_grid[ic_y_idx], np.zeros(args.batch_ic)], axis=-1).astype(np.float32)
    u0 = wavefield[0, ic_y_idx, ic_x_idx].astype(np.float32)
    v0_field = (wavefield[1] - wavefield[0]) / max(float(times[1] - times[0]), 1.0e-12)
    v0 = v0_field[ic_y_idx, ic_x_idx].astype(np.float32)

    bc_count = args.batch_bc
    edge = rng.integers(0, 4, size=bc_count)
    bx = rng.uniform(0.0, 1.0, size=bc_count)
    by = rng.uniform(0.0, 1.0, size=bc_count)
    bx = np.where(edge == 0, 0.0, np.where(edge == 1, 1.0, bx))
    by = np.where(edge == 2, 0.0, np.where(edge == 3, 1.0, by))
    bt = rng.uniform(float(times[0]), float(times[train_frames[-1]]), size=bc_count)
    x_bc = np.stack([bx, by, bt], axis=-1).astype(np.float32)
    u_bc = interpolate_wavefield(wavefield, times, x_bc).astype(np.float32)

    return {
        "x_data": x_data,
        "u_data": u_data,
        "x_velocity": x_velocity,
        "v_data": v_data,
        "x_media": x_media,
        "c_data": c_data,
        "x_pde": x_pde,
        "x_ic": x_ic,
        "u0": u0,
        "v0": v0,
        "x_bc": x_bc,
        "u_bc": u_bc,
    }


def make_loss_and_step(model: Any, optimizer: optax.GradientTransformation, source: dict[str, Any], weights: dict[str, float]):
    def pde_residual(params, x_pde):
        def single(x_single):
            def u_at(z):
                return model.apply(params, APPLY_RNG, z[None, :], return_media=False)[0, 0]

            hess = jax.hessian(u_at)(x_single)
            u_tt = hess[2, 2]
            lap = hess[0, 0] + hess[1, 1]
            _, c_val = model.apply(params, APPLY_RNG, x_single[None, :], return_media=True)
            return u_tt - c_val[0, 0] ** 2 * lap - source_term_jax(x_single[None, :], source)[0]

        return jax.vmap(single)(x_pde)

    def initial_time_derivative(params, x_ic):
        def single(x_single):
            def u_at(z):
                return model.apply(params, APPLY_RNG, z[None, :], return_media=False)[0, 0]

            grad = jax.grad(u_at)(x_single)
            return u_at(x_single), grad[2]

        return jax.vmap(single)(x_ic)

    def time_derivative(params, x_points):
        def single(x_single):
            def u_at(z):
                return model.apply(params, APPLY_RNG, z[None, :], return_media=False)[0, 0]

            return jax.grad(u_at)(x_single)[2]

        return jax.vmap(single)(x_points)

    def loss_fn(params, batch):
        u_pred = model.apply(params, APPLY_RNG, batch["x_data"], return_media=False).squeeze()
        loss_data = jnp.mean((u_pred - batch["u_data"]) ** 2)

        if batch["x_velocity"].shape[0] > 0:
            v_pred = time_derivative(params, batch["x_velocity"])
            v_scale = jnp.asarray(source.get("velocity_normalization", 1.0), dtype=v_pred.dtype)
            loss_velocity = jnp.mean(((v_pred - batch["v_data"]) / v_scale) ** 2)
        else:
            loss_velocity = jnp.array(0.0)

        dummy_t = jnp.zeros((batch["x_media"].shape[0], 1), dtype=batch["x_media"].dtype)
        _, c_pred = model.apply(params, APPLY_RNG, jnp.concatenate([batch["x_media"], dummy_t], axis=-1), return_media=True)
        loss_medium = jnp.mean((c_pred.squeeze() - batch["c_data"]) ** 2)

        residual = pde_residual(params, batch["x_pde"])
        pde_scale = jnp.asarray(source.get("pde_normalization", 1.0), dtype=residual.dtype)
        loss_pde = jnp.mean((residual / pde_scale) ** 2)

        u0_pred, v0_pred = initial_time_derivative(params, batch["x_ic"])
        loss_ic = jnp.mean((u0_pred - batch["u0"]) ** 2) + jnp.mean((v0_pred - batch["v0"]) ** 2)

        u_bc_pred = model.apply(params, APPLY_RNG, batch["x_bc"], return_media=False).squeeze()
        loss_bc = jnp.mean((u_bc_pred - batch["u_bc"]) ** 2)

        total = (
            weights["data"] * loss_data
            + weights["velocity"] * loss_velocity
            + weights["medium"] * loss_medium
            + weights["pde"] * loss_pde
            + weights["ic"] * loss_ic
            + weights["bc"] * loss_bc
        )
        return total, {
            "loss_total": total,
            "loss_data": loss_data,
            "loss_velocity": loss_velocity,
            "loss_medium": loss_medium,
            "loss_pde": loss_pde,
            "loss_ic": loss_ic,
            "loss_bc": loss_bc,
        }

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        parts = dict(parts)
        parts["grad_norm"] = grad_norm
        return new_params, new_opt_state, parts

    return loss_fn, train_step, pde_residual


def predict_points(model: Any, params: Any, coords: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    outputs = []
    for start in range(0, len(coords), batch_size):
        chunk = jnp.asarray(coords[start:start + batch_size], dtype=jnp.float32)
        pred = model.apply(params, APPLY_RNG, chunk, return_media=False).squeeze()
        outputs.append(np.asarray(jax.device_get(pred), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def predict_medium_points(model: Any, params: Any, xy: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    outputs = []
    for start in range(0, len(xy), batch_size):
        chunk_xy = jnp.asarray(xy[start:start + batch_size], dtype=jnp.float32)
        chunk = jnp.concatenate([chunk_xy, jnp.zeros((chunk_xy.shape[0], 1), dtype=jnp.float32)], axis=-1)
        _, pred = model.apply(params, APPLY_RNG, chunk, return_media=True)
        outputs.append(np.asarray(jax.device_get(pred.squeeze()), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def predict_full_grid(model: Any, params: Any, times: np.ndarray, nx: int, ny: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    frames = []
    for t in times:
        coords = np.concatenate([base, np.full((base.shape[0], 1), t, dtype=np.float32)], axis=-1)
        frames.append(predict_points(model, params, coords).reshape(ny, nx))
    return np.stack(frames, axis=0).astype(np.float32)


def heldout_subset(rng: np.random.Generator, wavefield: np.ndarray, times: np.ndarray, heldout_frames: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    _, ny, nx = wavefield.shape
    x_grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y_grid = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    frame_idx = rng.choice(heldout_frames, size=count, replace=True)
    x_idx = rng.integers(0, nx, size=count)
    y_idx = rng.integers(0, ny, size=count)
    coords = np.stack([x_grid[x_idx], y_grid[y_idx], times[frame_idx]], axis=-1).astype(np.float32)
    targets = wavefield[frame_idx, y_idx, x_idx].astype(np.float32)
    return coords, targets


def evaluate_pde_residual(
    pde_residual_fn: Any,
    params: Any,
    points: np.ndarray,
    batch_size: int = 128,
) -> tuple[float, float]:
    vals = []
    for start in range(0, len(points), batch_size):
        chunk = jnp.asarray(points[start:start + batch_size], dtype=jnp.float32)
        residual = pde_residual_fn(params, chunk)
        vals.append(np.asarray(jax.device_get(residual), dtype=np.float64))
    residuals = np.concatenate(vals, axis=0)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    scale = float(np.sqrt(np.mean(points[:, 2] ** 0)) + 1.0e-12)
    return rms, rms / scale


def contiguous_pass_fraction(pass_mask: np.ndarray) -> tuple[int, int, float]:
    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0
    for idx, ok in enumerate(pass_mask):
        if ok:
            if current_len == 0:
                current_start = idx
            current_len += 1
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
        else:
            current_len = 0
    end = best_start + max(best_len - 1, 0)
    frac = best_len / max(len(pass_mask), 1)
    return best_start, end, float(frac)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(jax.device_get(payload), f)


def write_metrics_summary(path: Path, metrics: dict[str, Any], checkpoint_path: Path, parameter_count: int) -> None:
    lines = [
        "# Project 02 Rebuild Metrics Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Training Artifact",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Parameter count: `{parameter_count:,}`",
        f"- Framework: `{metrics['framework']}`",
        f"- Training frames: `{metrics['train_frame_start']}..{metrics['train_frame_end']}`",
        f"- Held-out frames: `{metrics['heldout_frame_start']}..{metrics['heldout_frame_end']}`",
        "",
        "## Held-Out Field Metrics",
        "",
        f"- NMSE: `{metrics['time_heldout_nmse']:.6e}`",
        f"- PSNR: `{metrics['time_heldout_psnr_db']:.3f} dB`",
        f"- Energy ratio: `{metrics['energy_ratio']:.6f}`",
        f"- Dynamic-range ratio: `{metrics['dynamic_range_ratio']:.6f}`",
        f"- Learned temporal motion variance: `{metrics['learned_temporal_motion_variance']:.6e}`",
        f"- Trust-window fraction: `{metrics['trust_window_fraction']:.3f}`",
        "",
        "## Off-Lattice Metrics",
        "",
        f"- Samples: `{metrics['off_lattice_sample_count']}`",
        f"- NMSE: `{metrics['off_lattice_nmse']:.6e}`",
        f"- PSNR: `{metrics['off_lattice_psnr_db']:.3f} dB`",
        f"- MAE: `{metrics['off_lattice_mae']:.6e}`",
        "",
        "## Physics And Medium",
        "",
        f"- PDE residual RMS: `{metrics['heldout_pde_residual_rms']:.6e}`",
        f"- PDE residual normalized RMS: `{metrics['heldout_pde_residual_normalized_rms']:.6e}`",
        f"- Medium NMSE: `{metrics['medium_nmse']:.6e}`",
        f"- Medium correlation: `{metrics['medium_correlation']:.6f}`",
        f"- Medium off-lattice NMSE: `{metrics['medium_off_lattice_nmse']:.6e}`",
        f"- Source-event visible: `{metrics['source_event_visible']}`",
        "",
        "## Anti-Tautology",
        "",
        f"- Byte-identical to reference: `{metrics['anti_tautology']['byte_identical']}`",
        f"- Allclose at 1e-6: `{metrics['anti_tautology']['allclose_at_1e_6']}`",
        f"- Held-out NMSE above leakage floor: `{metrics['anti_tautology']['heldout_nmse_above_leakage_floor']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--batch-data", type=int, default=8192)
    parser.add_argument("--batch-medium", type=int, default=4096)
    parser.add_argument("--batch-pde", type=int, default=64)
    parser.add_argument("--batch-ic", type=int, default=256)
    parser.add_argument("--batch-bc", type=int, default=512)
    parser.add_argument("--batch-velocity", type=int, default=512)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-velocity", type=float, default=0.1)
    parser.add_argument("--lambda-medium", type=float, default=1.0)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-ic", type=float, default=0.01)
    parser.add_argument("--lambda-bc", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--fourier-dim", type=int, default=256)
    parser.add_argument("--fourier-sigma", type=float, default=4.0)
    parser.add_argument("--media-hidden-dim", type=int, default=128)
    parser.add_argument("--media-hidden-layers", type=int, default=3)
    parser.add_argument("--media-fourier-dim", type=int, default=128)
    parser.add_argument("--media-fourier-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=20000)
    parser.add_argument("--pde-eval-samples", type=int, default=5000)
    parser.add_argument("--smoke-only", action="store_true", help="Compile and run training steps, then exit before writing artifacts.")
    args = parser.parse_args()

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    wavefield = np.load(REFERENCE_DIR / "reference_wavefield.npy").astype(np.float32)
    medium = np.load(REFERENCE_DIR / "reference_medium.npy").astype(np.float32)
    times = np.load(REFERENCE_DIR / "times.npy").astype(np.float32)
    metadata = json.loads((REFERENCE_DIR / "reference_metadata.json").read_text(encoding="utf-8"))

    n_frames, ny, nx = wavefield.shape
    train_frames = np.arange(0, int(math.floor(0.75 * n_frames)), dtype=np.int64)
    heldout_frames = np.arange(train_frames[-1] + 1, n_frames, dtype=np.int64)
    x_grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y_grid = np.linspace(0.0, 1.0, ny, dtype=np.float32)

    source = dict(metadata["source"])
    xx, yy = np.meshgrid(x_grid, y_grid)
    spatial = np.exp(-((xx - source["position"][0]) ** 2 + (yy - source["position"][1]) ** 2) / (2.0 * source["sigma"] ** 2))
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    source["spatial_normalization"] = float(1.0 / (np.sum(spatial) * dx * dy))
    train_vel = (wavefield[2:train_frames[-1] + 1] - wavefield[:train_frames[-1] - 1]) / np.maximum(
        (times[2:train_frames[-1] + 1] - times[:train_frames[-1] - 1])[:, None, None],
        1.0e-12,
    )
    source["velocity_normalization"] = float(np.std(train_vel) + 1.0e-6)

    model_config = make_model_config(args, float(medium.min()), float(medium.max()))
    model = create_model(model_config)
    params = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 3), dtype=jnp.float32), return_media=True)
    parameter_count = count_parameters(params)
    schedule = optax.cosine_decay_schedule(args.learning_rate, decay_steps=args.epochs)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)
    source["pde_normalization"] = float(abs(source["amplitude"]) * source["spatial_normalization"])
    weights = {
        "data": args.lambda_data,
        "velocity": args.lambda_velocity,
        "medium": args.lambda_medium,
        "pde": args.lambda_pde,
        "ic": args.lambda_ic,
        "bc": args.lambda_bc,
    }
    _, train_step, pde_residual_fn = make_loss_and_step(model, optimizer, source, weights)

    val_coords, val_targets = heldout_subset(rng, wavefield, times, heldout_frames, args.val_samples)
    best_val = float("inf")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    print(f"JAX devices: {jax.devices()} backend={jax.default_backend()}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Training frames: {train_frames[0]}..{train_frames[-1]} held-out: {heldout_frames[0]}..{heldout_frames[-1]}")

    for epoch in range(1, args.epochs + 1):
        batch_np = sample_batch(rng, wavefield, medium, times, train_frames, x_grid, y_grid, args)
        batch = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in batch_np.items()}
        params, opt_state, parts = train_step(params, opt_state, batch)

        if epoch % args.log_every == 0 or epoch == 1:
            row = {k: float(jax.device_get(v)) for k, v in parts.items()}
            row["epoch"] = int(epoch)
            row["elapsed_seconds"] = float(time.time() - start_time)
            history.append(row)
            print(
                "epoch={epoch:06d} total={loss_total:.6e} data={loss_data:.6e} "
                "vel={loss_velocity:.6e} medium={loss_medium:.6e} pde={loss_pde:.6e} "
                "ic={loss_ic:.6e} bc={loss_bc:.6e} "
                "grad={grad_norm:.6e} elapsed={elapsed_seconds:.1f}s".format(**row),
                flush=True,
            )

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_pred = predict_points(model, params, val_coords)
            val_nmse = nmse(val_pred, val_targets)
            val_psnr = psnr_db(val_pred, val_targets)
            print(f"validation epoch={epoch:06d} heldout_subset_nmse={val_nmse:.6e} psnr={val_psnr:.3f}dB", flush=True)
            if val_nmse < best_val:
                best_val = val_nmse
                best_payload = {
                    "framework": "jax_haiku",
                    "params": params,
                    "opt_state": opt_state,
                    "epoch": epoch,
                    "best_heldout_subset_nmse": best_val,
                    "best_heldout_subset_psnr_db": val_psnr,
                    "model_config": model_config,
                    "training_config": vars(args),
                    "loss_weights": weights,
                    "train_frames": train_frames,
                    "heldout_frames": heldout_frames,
                    "loss_history": history,
                    "parameter_count": parameter_count,
                    "reference_paths": {
                        "wavefield": str(REFERENCE_DIR / "reference_wavefield.npy"),
                        "medium": str(REFERENCE_DIR / "reference_medium.npy"),
                        "times": str(REFERENCE_DIR / "times.npy"),
                    },
                    "source": source,
                }
                if not args.smoke_only:
                    save_checkpoint(CHECKPOINT_DIR / "project02_wavepinn_nif_best.pkl", best_payload)

    if args.smoke_only:
        print("smoke-only run completed before checkpoint/export")
        return

    final_payload = {
        "framework": "jax_haiku",
        "params": params,
        "opt_state": opt_state,
        "epoch": args.epochs,
        "best_heldout_subset_nmse": best_val,
        "model_config": model_config,
        "training_config": vars(args),
        "loss_weights": weights,
        "train_frames": train_frames,
        "heldout_frames": heldout_frames,
        "loss_history": history,
        "parameter_count": parameter_count,
        "source": source,
    }
    save_checkpoint(CHECKPOINT_DIR / "project02_wavepinn_nif_final.pkl", final_payload)
    if best_payload is not None:
        params_for_eval = best_payload["params"]
        best_epoch = int(best_payload["epoch"])
    else:
        params_for_eval = params
        best_epoch = args.epochs

    print("Predicting full fixed grid...", flush=True)
    learned = predict_full_grid(model, params_for_eval, times, nx, ny)
    xxyy = np.stack(np.meshgrid(x_grid, y_grid), axis=-1).reshape(-1, 2).astype(np.float32)
    medium_pred = predict_medium_points(model, params_for_eval, xxyy).reshape(ny, nx).astype(np.float32)

    if np.array_equal(wavefield, learned):
        raise SystemExit("REJECT: learned arrays are byte-identical to reference")
    if np.allclose(wavefield, learned, atol=1.0e-6):
        raise SystemExit("REJECT: learned arrays match reference to 1e-6")
    if float(np.max(np.abs(learned))) <= 0.01:
        raise SystemExit("REJECT: learned field is essentially zero")
    learned_motion = float(np.mean(np.diff(learned, axis=0) ** 2))
    if learned_motion <= 5.0e-5:
        raise SystemExit(f"REJECT: learned field has negligible temporal motion: {learned_motion}")
    ref_energy = float(np.mean(wavefield ** 2))
    learned_energy = float(np.mean(learned ** 2))
    energy_ratio = learned_energy / max(ref_energy, 1.0e-30)
    if energy_ratio == 1.0:
        raise SystemExit("REJECT: energy ratio is exactly 1.0")
    if not (0.3 <= energy_ratio <= 3.0):
        raise SystemExit(f"REJECT: energy ratio {energy_ratio} outside plausible range")
    heldout_nmse = nmse(learned[heldout_frames], wavefield[heldout_frames])
    if heldout_nmse <= 1.0e-8:
        raise SystemExit("REJECT: held-out NMSE below 1e-8; investigate leakage")

    heldout_pred = learned[heldout_frames]
    heldout_ref = wavefield[heldout_frames]
    frame_nmse = np.asarray([nmse(heldout_pred[i], heldout_ref[i]) for i in range(len(heldout_frames))], dtype=np.float64)
    frame_psnr = np.asarray([psnr_db(heldout_pred[i], heldout_ref[i]) for i in range(len(heldout_frames))], dtype=np.float64)
    pass_mask = (frame_nmse <= 1.0e-3) & (frame_psnr >= 30.0)
    trust_start_rel, trust_end_rel, trust_fraction = contiguous_pass_fraction(pass_mask)
    trust_start_frame = int(heldout_frames[trust_start_rel])
    trust_end_frame = int(heldout_frames[trust_end_rel])

    off_coords = np.concatenate(
        [
            rng.uniform(0.0, 1.0, size=(1000, 2)),
            rng.uniform(float(times[heldout_frames[0]]), float(times[-1]), size=(1000, 1)),
        ],
        axis=-1,
    ).astype(np.float32)
    off_ref = interpolate_wavefield(wavefield, times, off_coords)
    off_pred = predict_points(model, params_for_eval, off_coords)

    pde_points = np.concatenate(
        [
            rng.uniform(0.04, 0.96, size=(args.pde_eval_samples, 2)),
            rng.uniform(float(times[heldout_frames[0]]), float(times[-1]), size=(args.pde_eval_samples, 1)),
        ],
        axis=-1,
    ).astype(np.float32)
    pde_rms, pde_norm = evaluate_pde_residual(pde_residual_fn, params_for_eval, pde_points)

    medium_nmse_value = nmse(medium_pred, medium)
    medium_corr = correlation(medium_pred, medium)
    medium_off_xy = rng.uniform(0.0, 1.0, size=(2000, 2)).astype(np.float32)
    medium_off_ref = interpolate_medium(medium, medium_off_xy)
    medium_off_pred = predict_medium_points(model, params_for_eval, medium_off_xy)

    source_x = int(round(float(source["position"][0]) * (nx - 1)))
    source_y = int(round(float(source["position"][1]) * (ny - 1)))
    ref_series = wavefield[:, source_y, source_x]
    pred_series = learned[:, source_y, source_x]
    ref_peak_frame = int(np.argmax(np.abs(ref_series)))
    pred_peak_frame = int(np.argmax(np.abs(pred_series)))
    source_peak_ratio = float(np.max(np.abs(pred_series)) / (np.max(np.abs(ref_series)) + 1.0e-30))
    source_visible = bool(abs(pred_peak_frame - ref_peak_frame) <= 4 and source_peak_ratio >= 0.2)

    metrics = {
        "framework": "jax_haiku",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_best": str(CHECKPOINT_DIR / "project02_wavepinn_nif_best.pkl"),
        "checkpoint_final": str(CHECKPOINT_DIR / "project02_wavepinn_nif_final.pkl"),
        "parameter_count": int(parameter_count),
        "best_epoch": int(best_epoch),
        "train_frame_start": int(train_frames[0]),
        "train_frame_end": int(train_frames[-1]),
        "heldout_frame_start": int(heldout_frames[0]),
        "heldout_frame_end": int(heldout_frames[-1]),
        "time_heldout_nmse": float(heldout_nmse),
        "time_heldout_psnr_db": psnr_db(heldout_pred, heldout_ref),
        "heldout_frame_nmse": frame_nmse.tolist(),
        "heldout_frame_psnr_db": frame_psnr.tolist(),
        "energy_ratio": float(energy_ratio),
        "dynamic_range_ratio": float((np.max(learned) - np.min(learned)) / max(float(np.max(wavefield) - np.min(wavefield)), 1.0e-30)),
        "reference_energy": float(ref_energy),
        "learned_energy": float(learned_energy),
        "reference_temporal_motion_variance": float(np.mean(np.diff(wavefield, axis=0) ** 2)),
        "learned_temporal_motion_variance": float(learned_motion),
        "off_lattice_sample_count": 1000,
        "off_lattice_nmse": nmse(off_pred, off_ref),
        "off_lattice_psnr_db": psnr_db(off_pred, off_ref),
        "off_lattice_mae": float(np.mean(np.abs(off_pred - off_ref))),
        "heldout_pde_residual_sample_count": int(args.pde_eval_samples),
        "heldout_pde_residual_rms": float(pde_rms),
        "heldout_pde_residual_normalized_rms": float(pde_norm),
        "medium_nmse": float(medium_nmse_value),
        "medium_correlation": float(medium_corr),
        "medium_min_pred": float(np.min(medium_pred)),
        "medium_max_pred": float(np.max(medium_pred)),
        "medium_min_reference": float(np.min(medium)),
        "medium_max_reference": float(np.max(medium)),
        "medium_off_lattice_nmse": nmse(medium_off_pred, medium_off_ref),
        "medium_off_lattice_psnr_db": psnr_db(medium_off_pred, medium_off_ref),
        "medium_off_lattice_sample_count": int(len(medium_off_xy)),
        "source_event_visible": source_visible,
        "source_ref_peak_frame": ref_peak_frame,
        "source_pred_peak_frame": pred_peak_frame,
        "source_peak_amplitude_ratio": source_peak_ratio,
        "trust_window_start_frame": trust_start_frame,
        "trust_window_end_frame": trust_end_frame,
        "trust_window_fraction": float(trust_fraction),
        "trust_window_threshold": "per-held-out-frame NMSE <= 1e-3 and PSNR >= 30 dB",
        "anti_tautology": {
            "byte_identical": bool(np.array_equal(wavefield, learned)),
            "allclose_at_1e_6": bool(np.allclose(wavefield, learned, atol=1.0e-6)),
            "max_abs_reference_minus_learned": float(np.max(np.abs(wavefield - learned))),
            "energy_ratio_not_exactly_one": bool(energy_ratio != 1.0),
            "energy_ratio_in_plausible_range": bool(0.3 <= energy_ratio <= 3.0),
            "heldout_nmse_above_leakage_floor": bool(heldout_nmse > 1.0e-8),
            "zero_field_rejected": bool(np.max(np.abs(learned)) > 0.01 and learned_motion > 5.0e-5),
        },
        "training_wall_clock_minutes": float((time.time() - start_time) / 60.0),
        "loss_history": history,
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VALIDATION_DIR / "wavefield_sequence.npy", learned)
    np.save(VALIDATION_DIR / "medium_pred.npy", medium_pred)
    np.save(VALIDATION_DIR / "times.npy", times)
    write_json(VALIDATION_DIR / "metrics.json", metrics)
    save_snapshots(learned, times, VALIDATION_DIR / "snapshots.png", title="Project 02 Learned JAX/Haiku Surrogate")
    save_slice_xt(learned, times, source_y, VALIDATION_DIR / "slice_xt.png")
    save_medium(medium_pred, VALIDATION_DIR / "medium_pred.png")
    write_metrics_summary(PRIVATE_DIR / "project02_rebuild_metrics_summary.md", metrics, CHECKPOINT_DIR / "project02_wavepinn_nif_best.pkl", parameter_count)

    runlog = PRIVATE_DIR / "train_runlog_project02.md"
    runlog.parent.mkdir(parents=True, exist_ok=True)
    with runlog.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"## {datetime.now(timezone.utc).isoformat()} JAX/Haiku rebuild\n\n")
        f.write(f"- Epochs requested: `{args.epochs}`\n")
        f.write(f"- Best epoch: `{best_epoch}`\n")
        f.write(f"- Parameter count: `{parameter_count:,}`\n")
        f.write(f"- Wall-clock minutes: `{metrics['training_wall_clock_minutes']:.2f}`\n")
        f.write(f"- Held-out NMSE / PSNR: `{metrics['time_heldout_nmse']:.6e}` / `{metrics['time_heldout_psnr_db']:.3f} dB`\n")
        f.write(f"- Checkpoint: `{CHECKPOINT_DIR / 'project02_wavepinn_nif_best.pkl'}`\n")

    print(json.dumps({k: metrics[k] for k in (
        "time_heldout_nmse",
        "time_heldout_psnr_db",
        "off_lattice_nmse",
        "heldout_pde_residual_rms",
        "medium_nmse",
        "trust_window_fraction",
        "energy_ratio",
        "learned_temporal_motion_variance",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
