"""
Cost-Benefit Analysis for CoastFlow-GNN

Compares:
- Runtime: CFD simulation vs. GNN surrogate inference
- Energy consumption: HPC cluster vs. single GPU
- Accuracy tradeoffs at different fidelity levels

Generates comparison reports and visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SimulationMetrics:
    """Metrics for a CFD simulation run."""
    name: str
    runtime_seconds: float
    num_nodes: int
    num_timesteps: int
    cpu_hours: float
    gpu_hours: float
    energy_kwh: float
    cost_usd: float
    accuracy_mse: Optional[float] = None


@dataclass
class SurrogateMetrics:
    """Metrics for GNN surrogate inference."""
    name: str
    inference_time_ms: float
    num_nodes: int
    gpu_memory_mb: float
    energy_per_inference_wh: float
    training_time_hours: float
    training_energy_kwh: float
    accuracy_mse: float


def estimate_cfd_metrics(
    num_nodes: int = 100000,
    num_timesteps: int = 1000,
    solver: str = "openfoam",
    hardware: str = "hpc_cluster",
) -> SimulationMetrics:
    """
    Estimate CFD simulation cost and runtime.
    
    Based on typical OpenFOAM performance benchmarks.
    
    Args:
        num_nodes: Mesh nodes
        num_timesteps: Simulation timesteps
        solver: CFD solver (openfoam, fluent, etc.)
        hardware: Hardware type (hpc_cluster, workstation, cloud)
        
    Returns:
        SimulationMetrics with estimates
    """
    # Typical scaling factors (empirical estimates)
    # OpenFOAM: ~1e-6 seconds per cell per timestep on modern CPU
    base_time_per_cell_per_step = {
        "openfoam": 1e-6,
        "fluent": 8e-7,
        "su2": 1.2e-6,
    }.get(solver, 1e-6)
    
    # Hardware multipliers
    hardware_factor = {
        "hpc_cluster": 0.1,  # 10 nodes parallel
        "workstation": 1.0,
        "cloud_gpu": 0.3,
    }.get(hardware, 1.0)
    
    # Compute runtime
    runtime_seconds = (
        base_time_per_cell_per_step 
        * num_nodes 
        * num_timesteps 
        * hardware_factor
    )
    
    # CPU hours (assuming 32 cores for HPC, 8 for workstation)
    cores = {"hpc_cluster": 32, "workstation": 8, "cloud_gpu": 16}.get(hardware, 8)
    cpu_hours = runtime_seconds * cores / 3600
    
    # GPU hours (if applicable)
    gpu_hours = runtime_seconds / 3600 if "gpu" in hardware else 0
    
    # Energy consumption (kWh)
    # Typical: HPC node ~500W, workstation ~300W, cloud GPU ~400W
    power_watts = {"hpc_cluster": 500, "workstation": 300, "cloud_gpu": 400}.get(hardware, 300)
    energy_kwh = power_watts * runtime_seconds / 3600 / 1000
    
    # Cost (USD)
    # HPC: ~$0.05/core-hour, cloud: ~$3/GPU-hour
    if hardware == "hpc_cluster":
        cost_usd = cpu_hours * 0.05
    elif hardware == "cloud_gpu":
        cost_usd = gpu_hours * 3.0
    else:
        cost_usd = energy_kwh * 0.12  # Electricity cost
    
    return SimulationMetrics(
        name=f"{solver}_{hardware}_{num_nodes//1000}k",
        runtime_seconds=runtime_seconds,
        num_nodes=num_nodes,
        num_timesteps=num_timesteps,
        cpu_hours=cpu_hours,
        gpu_hours=gpu_hours,
        energy_kwh=energy_kwh,
        cost_usd=cost_usd,
    )


def estimate_surrogate_metrics(
    num_nodes: int = 1000,
    inference_time_ms: float = 10.0,
    model_params: int = 500000,
    training_epochs: int = 50,
    training_samples: int = 1000,
    accuracy_mse: float = 0.005,
) -> SurrogateMetrics:
    """
    Estimate GNN surrogate costs.
    
    Args:
        num_nodes: Nodes per mesh
        inference_time_ms: Inference time per sample
        model_params: Number of model parameters
        training_epochs: Training epochs
        training_samples: Number of training samples
        accuracy_mse: Achieved MSE on test set
        
    Returns:
        SurrogateMetrics with estimates
    """
    # GPU memory estimate (rough: 4 bytes per param + activations)
    gpu_memory_mb = model_params * 4 / 1e6 * 3  # 3x for gradients + activations
    
    # Energy per inference (GPU at ~300W, utilization ~50%)
    power_watts = 300 * 0.5
    energy_per_inference_wh = power_watts * inference_time_ms / 1000 / 3600
    
    # Training time estimate
    # Assume 100ms per batch, 4 samples per batch
    batch_time_s = 0.1
    batches_per_epoch = training_samples // 4
    training_time_hours = training_epochs * batches_per_epoch * batch_time_s / 3600
    
    # Training energy
    training_power_w = 300  # Full GPU utilization during training
    training_energy_kwh = training_power_w * training_time_hours / 1000
    
    return SurrogateMetrics(
        name=f"coastflow_gnn_{num_nodes}n",
        inference_time_ms=inference_time_ms,
        num_nodes=num_nodes,
        gpu_memory_mb=gpu_memory_mb,
        energy_per_inference_wh=energy_per_inference_wh,
        training_time_hours=training_time_hours,
        training_energy_kwh=training_energy_kwh,
        accuracy_mse=accuracy_mse,
    )


def compare_runtime(
    cfd_metrics: SimulationMetrics,
    surrogate_metrics: SurrogateMetrics,
    num_scenarios: int = 100,
) -> Dict[str, float]:
    """
    Compare runtime between CFD and surrogate.
    
    Args:
        cfd_metrics: CFD simulation metrics
        surrogate_metrics: GNN surrogate metrics
        num_scenarios: Number of scenarios to evaluate
        
    Returns:
        Comparison metrics
    """
    # CFD: each scenario requires full simulation
    cfd_total_time = cfd_metrics.runtime_seconds * num_scenarios
    
    # Surrogate: just inference (training is amortized)
    surrogate_total_time = surrogate_metrics.inference_time_ms * num_scenarios / 1000
    
    speedup = cfd_total_time / surrogate_total_time
    
    return {
        'cfd_runtime_seconds': cfd_total_time,
        'surrogate_runtime_seconds': surrogate_total_time,
        'speedup_factor': speedup,
        'cfd_per_scenario_s': cfd_metrics.runtime_seconds,
        'surrogate_per_scenario_ms': surrogate_metrics.inference_time_ms,
    }


def compare_energy(
    cfd_metrics: SimulationMetrics,
    surrogate_metrics: SurrogateMetrics,
    num_scenarios: int = 100,
    include_training: bool = True,
) -> Dict[str, float]:
    """
    Compare energy consumption between CFD and surrogate.
    
    Args:
        cfd_metrics: CFD simulation metrics
        surrogate_metrics: GNN surrogate metrics
        num_scenarios: Number of scenarios to evaluate
        include_training: Include training energy for surrogate
        
    Returns:
        Energy comparison metrics
    """
    # CFD energy
    cfd_energy = cfd_metrics.energy_kwh * num_scenarios
    
    # Surrogate energy
    inference_energy = surrogate_metrics.energy_per_inference_wh * num_scenarios / 1000
    training_energy = surrogate_metrics.training_energy_kwh if include_training else 0
    surrogate_energy = inference_energy + training_energy
    
    energy_savings = cfd_energy - surrogate_energy
    savings_percent = (energy_savings / cfd_energy) * 100 if cfd_energy > 0 else 0
    
    # CO2 equivalent (0.4 kg CO2 per kWh average)
    co2_factor = 0.4
    cfd_co2_kg = cfd_energy * co2_factor
    surrogate_co2_kg = surrogate_energy * co2_factor
    
    return {
        'cfd_energy_kwh': cfd_energy,
        'surrogate_energy_kwh': surrogate_energy,
        'training_energy_kwh': training_energy,
        'inference_energy_kwh': inference_energy,
        'energy_savings_kwh': energy_savings,
        'savings_percent': savings_percent,
        'cfd_co2_kg': cfd_co2_kg,
        'surrogate_co2_kg': surrogate_co2_kg,
    }


def generate_report(
    cfd_metrics: SimulationMetrics,
    surrogate_metrics: SurrogateMetrics,
    num_scenarios: int = 100,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Generate comprehensive cost-benefit report.
    
    Args:
        cfd_metrics: CFD simulation metrics
        surrogate_metrics: GNN surrogate metrics
        num_scenarios: Number of scenarios for comparison
        output_path: Path to save JSON report
        
    Returns:
        Full report dictionary
    """
    runtime_comparison = compare_runtime(cfd_metrics, surrogate_metrics, num_scenarios)
    energy_comparison = compare_energy(cfd_metrics, surrogate_metrics, num_scenarios)
    
    # Break-even analysis: how many scenarios until surrogate is cheaper?
    training_cost = surrogate_metrics.training_energy_kwh * 0.12  # $0.12/kWh
    per_scenario_savings = cfd_metrics.cost_usd - (
        surrogate_metrics.energy_per_inference_wh / 1000 * 0.12
    )
    
    if per_scenario_savings > 0:
        break_even_scenarios = training_cost / per_scenario_savings
    else:
        break_even_scenarios = float('inf')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_scenarios': num_scenarios,
        'cfd_metrics': asdict(cfd_metrics),
        'surrogate_metrics': asdict(surrogate_metrics),
        'runtime_comparison': runtime_comparison,
        'energy_comparison': energy_comparison,
        'break_even_scenarios': break_even_scenarios,
        'summary': {
            'speedup': f"{runtime_comparison['speedup_factor']:.1f}x",
            'energy_savings': f"{energy_comparison['savings_percent']:.1f}%",
            'accuracy_mse': surrogate_metrics.accuracy_mse,
            'recommendation': (
                'Surrogate recommended' if break_even_scenarios < num_scenarios 
                else 'CFD recommended for small scenario counts'
            ),
        }
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")
    
    return report


def plot_comparison(
    report: Dict,
    output_dir: str = "./outputs",
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generate comparison visualizations.
    
    Args:
        report: Cost-benefit report dictionary
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (runtime_fig, energy_fig)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Runtime comparison
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    runtime = report['runtime_comparison']
    methods = ['CFD Simulation', 'GNN Surrogate']
    times = [runtime['cfd_runtime_seconds'] / 3600, 
             runtime['surrogate_runtime_seconds'] / 3600]
    
    bars = ax.bar(methods, times, color=['#e74c3c', '#2ecc71'], edgecolor='black')
    ax.set_ylabel('Total Runtime (hours)')
    ax.set_title(f'Runtime Comparison ({report["num_scenarios"]} scenarios)\n'
                 f'Speedup: {runtime["speedup_factor"]:.0f}x')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, time in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{time:.2f}h', ha='center', va='bottom', fontsize=12)
    
    ax.grid(True, alpha=0.3, axis='y')
    fig1.tight_layout()
    fig1.savefig(output_dir / 'runtime_comparison.png', dpi=150)
    
    # Energy comparison
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    energy = report['energy_comparison']
    
    # Bar chart
    ax = axes[0]
    methods = ['CFD\nSimulation', 'GNN Surrogate\n(Training)', 'GNN Surrogate\n(Inference)']
    energies = [energy['cfd_energy_kwh'], 
                energy['training_energy_kwh'],
                energy['inference_energy_kwh']]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax.bar(methods, energies, color=colors, edgecolor='black')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy Consumption Breakdown')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{e:.2f}', ha='center', va='bottom', fontsize=10)
    
    # CO2 emissions
    ax = axes[1]
    co2 = [energy['cfd_co2_kg'], energy['surrogate_co2_kg']]
    methods = ['CFD', 'Surrogate']
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(methods, co2, color=colors, edgecolor='black')
    ax.set_ylabel('CO₂ Emissions (kg)')
    ax.set_title(f'Carbon Footprint\n({energy["savings_percent"]:.0f}% reduction)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, c in zip(bars, co2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{c:.2f}', ha='center', va='bottom', fontsize=10)
    
    fig2.tight_layout()
    fig2.savefig(output_dir / 'energy_comparison.png', dpi=150)
    
    print(f"Plots saved to: {output_dir}")
    
    return fig1, fig2


def print_summary(report: Dict) -> None:
    """Print formatted summary of cost-benefit analysis."""
    print("\n" + "=" * 60)
    print("CoastFlow-GNN Cost-Benefit Analysis")
    print("=" * 60)
    
    summary = report['summary']
    runtime = report['runtime_comparison']
    energy = report['energy_comparison']
    
    print(f"\n📊 Runtime:")
    print(f"   CFD: {runtime['cfd_runtime_seconds']/3600:.2f} hours "
          f"({runtime['cfd_per_scenario_s']:.1f}s per scenario)")
    print(f"   GNN: {runtime['surrogate_runtime_seconds']:.2f} seconds "
          f"({runtime['surrogate_per_scenario_ms']:.1f}ms per scenario)")
    print(f"   Speedup: {summary['speedup']}")
    
    print(f"\n⚡ Energy:")
    print(f"   CFD: {energy['cfd_energy_kwh']:.2f} kWh")
    print(f"   GNN: {energy['surrogate_energy_kwh']:.2f} kWh "
          f"(training: {energy['training_energy_kwh']:.2f}, "
          f"inference: {energy['inference_energy_kwh']:.4f})")
    print(f"   Savings: {summary['energy_savings']}")
    
    print(f"\n🎯 Accuracy:")
    print(f"   MSE: {summary['accuracy_mse']:.6f}")
    
    print(f"\n💰 Break-even:")
    print(f"   Scenarios needed: {report['break_even_scenarios']:.0f}")
    
    print(f"\n📋 Recommendation: {summary['recommendation']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Running cost-benefit analysis...")
    
    # Estimate metrics
    cfd = estimate_cfd_metrics(
        num_nodes=50000,
        num_timesteps=500,
        solver="openfoam",
        hardware="hpc_cluster",
    )
    
    surrogate = estimate_surrogate_metrics(
        num_nodes=1000,
        inference_time_ms=15.0,
        model_params=500000,
        training_epochs=50,
        training_samples=140,
        accuracy_mse=0.005,
    )
    
    # Generate report
    report = generate_report(
        cfd, surrogate,
        num_scenarios=100,
        output_path="./cost_benefit_report.json"
    )
    
    # Print summary
    print_summary(report)
    
    # Generate plots
    plot_comparison(report, output_dir="./outputs")
    
    print("✓ Cost-benefit analysis complete!")
