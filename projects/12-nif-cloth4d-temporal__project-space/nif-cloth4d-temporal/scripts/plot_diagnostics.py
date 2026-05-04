#!/usr/bin/env python3
"""
Diagnostic plots for the Prompt-3 media bundle.

- ``loss-curve.png`` — epoch vs train total / train MSE / train eikonal / val
  MSE for the hero checkpoint, with the ablation overlaid for cross-check.
- ``metrics-over-time.png`` — per-frame Chamfer + Hausdorff + NMSE from
  ``outputs/validation_model/metrics.json`` vs the playbook thresholds, with
  the ablation overlaid as dashed lines.

Both PNGs land in ``artifacts/taichi_final/`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_history(path: Path) -> tuple[np.ndarray, dict]:
    with path.open() as fh:
        data = json.load(fh)
    hist = data.get("history", [])
    epochs = np.array([h["epoch"] for h in hist], dtype=int)
    columns = {
        "train_total": np.array([h["train_total"] for h in hist], dtype=float),
        "train_mse":   np.array([h["train_mse"]   for h in hist], dtype=float),
        "train_eik":   np.array([h["train_eik"]   for h in hist], dtype=float),
        "val":         np.array([h["val"]         for h in hist], dtype=float),
    }
    return epochs, columns


def plot_loss_curve(hero_path: Path, ablation_path: Path, out_path: Path) -> None:
    eh, hh = load_history(hero_path)
    ea, ha = load_history(ablation_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, name, epochs, cols in (
        (axs[0], "Hero (GRU off)", eh, hh),
        (axs[1], "Ablation (GRU on)", ea, ha),
    ):
        ax.plot(epochs, cols["train_total"], label="train total",  lw=1.6)
        ax.plot(epochs, cols["train_mse"],   label="train mse",    lw=1.0, alpha=0.9)
        ax.plot(epochs, cols["train_eik"],   label="train eikonal", lw=1.0, alpha=0.6)
        ax.plot(epochs, cols["val"],         label="val mse",      lw=1.6, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(name, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    axs[0].set_ylabel("loss (log)")
    fig.suptitle("Project 12 — training loss over epochs", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_metrics_over_time(
    hero_metrics: Path,
    ablation_metrics: Path,
    out_path: Path,
) -> None:
    with hero_metrics.open() as fh:
        hm = json.load(fh)
    with ablation_metrics.open() as fh:
        am = json.load(fh)

    n = len(hm["heightfield_chamfer_per_frame"])
    t = np.linspace(0.0, 1.0, n, dtype=float)

    hf_c_h = np.asarray(hm["heightfield_chamfer_per_frame"])
    hf_c_a = np.asarray(am["heightfield_chamfer_per_frame"])
    hf_h_h = np.asarray(hm["heightfield_hausdorff_per_frame"])
    hf_h_a = np.asarray(am["heightfield_hausdorff_per_frame"])
    nmse_h = np.asarray(hm["nmse_per_frame"])
    nmse_a = np.asarray(am["nmse_per_frame"])

    thresh = hm["thresholds"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs[0].plot(t, hf_c_h, label="hero", lw=1.5)
    axs[0].plot(t, hf_c_a, label="ablation (GRU on)", lw=1.2, ls="--")
    axs[0].axhline(thresh["heightfield_chamfer_mean_max"], color="k", lw=0.8, ls=":",
                   label=f"threshold {thresh['heightfield_chamfer_mean_max']:.0e}")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("heightfield chamfer")
    axs[0].grid(True, which="both", alpha=0.25)
    axs[0].legend(loc="upper left", frameon=False, fontsize=9)

    axs[1].plot(t, hf_h_h, label="hero", lw=1.5)
    axs[1].plot(t, hf_h_a, label="ablation (GRU on)", lw=1.2, ls="--")
    axs[1].axhline(thresh["heightfield_hausdorff_mean_max"], color="k", lw=0.8, ls=":",
                   label=f"threshold {thresh['heightfield_hausdorff_mean_max']:.2f}")
    axs[1].set_ylabel("heightfield hausdorff")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper left", frameon=False, fontsize=9)

    axs[2].plot(t, nmse_h, label="hero", lw=1.5)
    axs[2].plot(t, nmse_a, label="ablation (GRU on)", lw=1.2, ls="--")
    axs[2].axhline(thresh["nmse_mean_max"], color="k", lw=0.8, ls=":",
                   label=f"threshold {thresh['nmse_mean_max']:.0e}")
    axs[2].set_yscale("log")
    axs[2].set_ylabel("volumetric nmse")
    axs[2].set_xlabel("t (normalized source time)")
    axs[2].grid(True, which="both", alpha=0.25)
    axs[2].legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle("Project 12 — per-timestep metrics vs playbook thresholds", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hero_summary", default=str(PROJECT_ROOT / "checkpoints" / "hero" / "training_summary.json"))
    ap.add_argument("--ablation_summary", default=str(PROJECT_ROOT / "checkpoints" / "ablation" / "training_summary.json"))
    ap.add_argument("--hero_metrics", default=str(PROJECT_ROOT / "outputs" / "validation_model" / "metrics.json"))
    ap.add_argument("--ablation_metrics", default=str(PROJECT_ROOT / "outputs" / "validation_model_temporal" / "metrics.json"))
    ap.add_argument("--output_dir", default=str(PROJECT_ROOT / "artifacts" / "taichi_final"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    plot_loss_curve(
        Path(args.hero_summary),
        Path(args.ablation_summary),
        out_dir / "loss-curve.png",
    )
    plot_metrics_over_time(
        Path(args.hero_metrics),
        Path(args.ablation_metrics),
        out_dir / "metrics-over-time.png",
    )


if __name__ == "__main__":
    main()
