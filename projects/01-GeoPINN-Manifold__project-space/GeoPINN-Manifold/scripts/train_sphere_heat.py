#!/usr/bin/env python3
"""
Train GeoPINN on the heat equation on the unit sphere.

PDE:  du/dt = Delta_S u   (heat/diffusion equation on sphere)

Analytical solution for spherical harmonic initial conditions:
    u_0(x) = x*y + 0.5*y*z   (mix of l=2 modes)
    u(x,t) = exp(-6t) * (x*y + 0.5*y*z)

The eigenvalue for l=2 spherical harmonics is -l(l+1) = -6.

This script:
  1. Trains the PINN with PDE residual + IC + supervision losses
  2. Saves checkpoint to checkpoints/geopinn_sphere_trained.pt
  3. Exports dense predictions at multiple timesteps
  4. Exports analytical baseline
  5. Computes comparison metrics
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = dict(
    # Model
    hidden_dim=128,
    num_layers=4,
    activation="tanh",

    # Training
    n_collocation=2000,
    n_ic=500,
    n_supervision=300,
    n_epochs=3000,
    lr=1e-3,
    lr_decay_step=1000,
    lr_decay_gamma=0.5,

    # Loss weights
    w_pde=1.0,
    w_ic=10.0,
    w_supervision=5.0,

    # Physics
    T_max=0.2,  # time domain [0, T_max]  (e^{-6*0.2}=0.30, still visible)

    # Export
    n_dense=15000,
    n_timesteps=12,

    # Paths (relative to project root)
    checkpoint_path="checkpoints/geopinn_sphere_trained.pt",
    pred_dir="outputs/geopinn_prediction",
    baseline_dir="outputs/analytical_baseline",
    metrics_path="outputs/metrics.json",

    log_every=100,
    checkpoint_every=500,
    seed=42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fibonacci_sphere(n, device="cpu"):
    """Quasi-uniform points on the unit sphere via Fibonacci spiral."""
    idx = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    phi = torch.acos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------
def analytical_u(xyz, t):
    """u(x,t) = exp(-6t) * (xy + 0.5*yz)"""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    return torch.exp(-6.0 * t) * (x * y + 0.5 * y * z)


def analytical_u_np(xyz, t):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.exp(-6.0 * t) * (x * y + 0.5 * y * z)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SphereHeatMLP(nn.Module):
    """MLP: (x, y, z, t) -> u scalar."""

    def __init__(self, hidden_dim=128, num_layers=4, activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU}[activation]
        layers = [nn.Linear(4, hidden_dim), act()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyzt):
        return self.net(xyzt)  # [N, 1]


# ---------------------------------------------------------------------------
# Laplace-Beltrami on unit sphere (autograd, same formula as sphere_trainer)
# ---------------------------------------------------------------------------
def laplace_beltrami_sphere(u_val, xyz):
    """
    Given u values and xyz points (both with grad graph), compute Delta_S u.

    Uses: Delta_S u = Delta u - n . grad(n . grad u) - n . grad u
    where n = xyz on unit sphere.
    """
    grad_u = torch.autograd.grad(
        u_val, xyz, torch.ones_like(u_val), create_graph=True, retain_graph=True
    )[0]  # [N, 3]

    # Euclidean Laplacian
    lap = torch.zeros_like(u_val)
    for i in range(3):
        g2 = torch.autograd.grad(
            grad_u[:, i:i+1], xyz, torch.ones_like(u_val),
            create_graph=True, retain_graph=True
        )[0]
        lap = lap + g2[:, i:i+1]

    n = xyz
    ndgu = (n * grad_u).sum(dim=1, keepdim=True)

    grad_ndgu = torch.autograd.grad(
        ndgu, xyz, torch.ones_like(ndgu), create_graph=True, retain_graph=True
    )[0]
    n_grad_ndgu = (n * grad_ndgu).sum(dim=1, keepdim=True)

    return lap - n_grad_ndgu - ndgu


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg, project_root):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    model = SphereHeatMLP(
        cfg["hidden_dim"], cfg["num_layers"], cfg["activation"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["lr_decay_step"], gamma=cfg["lr_decay_gamma"]
    )

    # --- Sample fixed collocation points ---
    xyz_col = fibonacci_sphere(cfg["n_collocation"], device)
    xyz_ic = fibonacci_sphere(cfg["n_ic"], device)
    xyz_sup = fibonacci_sphere(cfg["n_supervision"], device)

    history = {"loss": [], "pde_loss": [], "ic_loss": [], "sup_loss": [], "l2_error": []}

    print(f"\nTraining for {cfg['n_epochs']} epochs ...")
    print("-" * 60)
    t0_total = time.time()

    for epoch in range(1, cfg["n_epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        # -- PDE residual loss: du/dt - Delta_S u = 0 --
        # Random times in (0, T_max] for collocation
        t_col = torch.rand(cfg["n_collocation"], 1, device=device) * cfg["T_max"]
        xyz_c = xyz_col.clone().requires_grad_(True)
        t_c = t_col.clone().requires_grad_(True)
        inp_c = torch.cat([xyz_c, t_c], dim=1)  # [N, 4]
        u_c = model(inp_c)  # [N, 1]

        # du/dt via autograd
        du_dt = torch.autograd.grad(
            u_c, t_c, torch.ones_like(u_c), create_graph=True, retain_graph=True
        )[0]  # [N, 1]

        # Delta_S u via autograd (need grads w.r.t. xyz_c)
        lb_u = laplace_beltrami_sphere(u_c, xyz_c)  # [N, 1]

        pde_residual = du_dt - lb_u  # should be 0
        pde_loss = torch.mean(pde_residual ** 2)

        # -- IC loss: u(x, 0) = x*y + 0.5*y*z --
        t_zero = torch.zeros(cfg["n_ic"], 1, device=device)
        inp_ic = torch.cat([xyz_ic, t_zero], dim=1)
        u_ic_pred = model(inp_ic)
        u_ic_true = analytical_u(xyz_ic, torch.tensor(0.0, device=device)).unsqueeze(-1)
        ic_loss = torch.mean((u_ic_pred - u_ic_true) ** 2)

        # -- Supervision loss: scattered (x, t) pairs with known u --
        t_sup = torch.rand(cfg["n_supervision"], 1, device=device) * cfg["T_max"]
        inp_sup = torch.cat([xyz_sup, t_sup], dim=1)
        u_sup_pred = model(inp_sup)
        u_sup_true = analytical_u(xyz_sup, t_sup).unsqueeze(-1)
        sup_loss = torch.mean((u_sup_pred - u_sup_true) ** 2)

        # -- Total loss --
        loss = (
            cfg["w_pde"] * pde_loss
            + cfg["w_ic"] * ic_loss
            + cfg["w_supervision"] * sup_loss
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        # -- L2 error on collocation points at t=T_max/2 --
        with torch.no_grad():
            t_eval = torch.full((cfg["n_collocation"], 1), cfg["T_max"] / 2, device=device)
            inp_eval = torch.cat([xyz_col, t_eval], dim=1)
            u_pred_eval = model(inp_eval).squeeze(-1)
            u_true_eval = analytical_u(xyz_col, cfg["T_max"] / 2)
            l2_err = torch.sqrt(torch.mean((u_pred_eval - u_true_eval) ** 2)).item()

        history["loss"].append(loss.item())
        history["pde_loss"].append(pde_loss.item())
        history["ic_loss"].append(ic_loss.item())
        history["sup_loss"].append(sup_loss.item())
        history["l2_error"].append(l2_err)

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:5d}/{cfg['n_epochs']} | "
                f"loss {loss.item():.6f} | pde {pde_loss.item():.6f} | "
                f"ic {ic_loss.item():.6f} | sup {sup_loss.item():.6f} | "
                f"L2 {l2_err:.6f} | lr {lr_now:.1e}"
            )

        if epoch % cfg["checkpoint_every"] == 0:
            cp_path = os.path.join(project_root, cfg["checkpoint_path"])
            os.makedirs(os.path.dirname(cp_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "config": cfg,
            }, cp_path)
            print(f"  -> checkpoint saved ({cp_path})")

    elapsed = time.time() - t0_total
    print("-" * 60)
    print(f"Training done in {elapsed:.1f}s  ({elapsed/cfg['n_epochs']*1000:.1f} ms/epoch)")
    print(f"Final loss: {history['loss'][-1]:.6f}  |  Final L2: {history['l2_error'][-1]:.6f}")

    # Final checkpoint
    cp_path = os.path.join(project_root, cfg["checkpoint_path"])
    os.makedirs(os.path.dirname(cp_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": cfg["n_epochs"],
        "history": history,
        "config": cfg,
    }, cp_path)
    print(f"Final checkpoint: {cp_path}")

    return model, history


# ---------------------------------------------------------------------------
# Export predictions + baseline + metrics
# ---------------------------------------------------------------------------
def export(model, cfg, project_root):
    device = next(model.parameters()).device
    model.eval()

    # Dense sphere mesh
    xyz_dense = fibonacci_sphere(cfg["n_dense"], device)  # [N, 3]
    coords_np = xyz_dense.cpu().numpy()

    timesteps = np.linspace(0, cfg["T_max"], cfg["n_timesteps"])

    pred_fields = []  # [T, N]
    anal_fields = []  # [T, N]

    with torch.no_grad():
        for t_val in timesteps:
            t_tensor = torch.full((cfg["n_dense"], 1), t_val, device=device)
            inp = torch.cat([xyz_dense, t_tensor], dim=1)
            u_pred = model(inp).squeeze(-1).cpu().numpy()
            u_anal = analytical_u_np(coords_np, t_val)
            pred_fields.append(u_pred)
            anal_fields.append(u_anal)

    pred_fields = np.stack(pred_fields)  # [T, N]
    anal_fields = np.stack(anal_fields)  # [T, N]

    # Save predictions
    pred_dir = os.path.join(project_root, cfg["pred_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, "field_values.npy"), pred_fields)
    np.save(os.path.join(pred_dir, "sphere_coords.npy"), coords_np)
    np.save(os.path.join(pred_dir, "timesteps.npy"), timesteps)
    print(f"Predictions saved: {pred_dir}  (field {pred_fields.shape}, coords {coords_np.shape})")

    # Save analytical baseline
    base_dir = os.path.join(project_root, cfg["baseline_dir"])
    os.makedirs(base_dir, exist_ok=True)
    np.save(os.path.join(base_dir, "field_values.npy"), anal_fields)
    np.save(os.path.join(base_dir, "sphere_coords.npy"), coords_np)
    np.save(os.path.join(base_dir, "timesteps.npy"), timesteps)
    print(f"Analytical baseline saved: {base_dir}")

    # Metrics
    l2_per_t = np.sqrt(np.mean((pred_fields - anal_fields) ** 2, axis=1))  # [T]
    max_err_per_t = np.max(np.abs(pred_fields - anal_fields), axis=1)
    rel_l2_per_t = l2_per_t / (np.sqrt(np.mean(anal_fields ** 2, axis=1)) + 1e-10)

    # PSNR (using max of analytical range)
    data_range = anal_fields.max() - anal_fields.min()
    mse_per_t = np.mean((pred_fields - anal_fields) ** 2, axis=1)
    psnr_per_t = 10 * np.log10(data_range ** 2 / (mse_per_t + 1e-15))

    metrics = {
        "timesteps": timesteps.tolist(),
        "l2_error_per_timestep": l2_per_t.tolist(),
        "max_error_per_timestep": max_err_per_t.tolist(),
        "relative_l2_per_timestep": rel_l2_per_t.tolist(),
        "psnr_per_timestep": psnr_per_t.tolist(),
        "mean_l2": float(l2_per_t.mean()),
        "mean_psnr": float(psnr_per_t.mean()),
        "n_dense_points": cfg["n_dense"],
        "n_timesteps": cfg["n_timesteps"],
    }

    metrics_path = os.path.join(project_root, cfg["metrics_path"])
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    print("\n=== Per-Timestep Metrics ===")
    print(f"{'t':>6s}  {'L2':>10s}  {'RelL2':>10s}  {'MaxErr':>10s}  {'PSNR':>8s}")
    for i, t_val in enumerate(timesteps):
        print(
            f"{t_val:6.4f}  {l2_per_t[i]:10.6f}  {rel_l2_per_t[i]:10.6f}  "
            f"{max_err_per_t[i]:10.6f}  {psnr_per_t[i]:8.2f}"
        )
    print(f"\nMean L2: {metrics['mean_l2']:.6f}  |  Mean PSNR: {metrics['mean_psnr']:.2f} dB")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Determine project root (script is in scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("=" * 60)
    print("GeoPINN: Heat Equation on Unit Sphere")
    print("  PDE: du/dt = Delta_S u")
    print("  IC:  u(x,0) = xy + 0.5*yz  (l=2 spherical harmonics)")
    print("  Analytical: u(x,t) = exp(-6t) * (xy + 0.5*yz)")
    print("=" * 60)
    print(f"Project root: {project_root}\n")

    model, history = train(CFG, project_root)
    metrics = export(model, CFG, project_root)

    print("\nDone.")
