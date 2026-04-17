"""
Report generation utilities.

Generates JSON results, markdown tables, and HTML dashboards.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .core import BenchmarkResult


def save_json_result(
    result: BenchmarkResult,
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save a benchmark result to JSON.
    
    Args:
        result: BenchmarkResult to save
        output_dir: Directory to save to
        filename: Optional filename (default: auto-generated)
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.project_id}_{result.project_name.replace(' ', '_')}_{timestamp}.json"
    
    filepath = output_dir / filename
    result.save(filepath)
    return filepath


def save_aggregated_results(
    results: list[BenchmarkResult],
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save all benchmark results to a single aggregated JSON.
    
    Args:
        results: List of BenchmarkResults
        output_dir: Directory to save to
        filename: Optional filename
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"benchmark_comparison_{timestamp}.json"
    
    # Build aggregated structure
    aggregated = {
        "benchmark_run": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "harness_version": "1.0.0",
            "num_projects": len(results),
            "successful": sum(1 for r in results if r.status == "success"),
            "failed": sum(1 for r in results if r.status == "failed"),
        },
        "projects": [r.to_dict() for r in results],
        "summary": _compute_summary(results),
    }
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    return filepath


def _compute_summary(results: list[BenchmarkResult]) -> dict:
    """Compute summary statistics across all results."""
    successful = [r for r in results if r.status == "success"]
    
    if not successful:
        return {"note": "No successful benchmarks to summarize"}
    
    summary = {}
    
    # Fastest inference
    with_timing = [r for r in successful if r.inference_time_gpu]
    if with_timing:
        fastest = min(with_timing, key=lambda r: r.inference_time_gpu.mean_ms)
        summary["fastest_inference"] = {
            "project": fastest.project_id,
            "name": fastest.project_name,
            "time_ms": fastest.inference_time_gpu.mean_ms,
        }
    
    # Smallest model
    with_params = [r for r in successful if r.model.parameter_count > 0]
    if with_params:
        smallest = min(with_params, key=lambda r: r.model.parameter_count)
        summary["smallest_model"] = {
            "project": smallest.project_id,
            "name": smallest.project_name,
            "parameters": smallest.model.parameter_count,
        }
    
    # Lowest L2 error
    with_l2 = [r for r in successful if r.accuracy.l2_relative_error is not None]
    if with_l2:
        best_accuracy = min(with_l2, key=lambda r: r.accuracy.l2_relative_error)
        summary["lowest_l2_error"] = {
            "project": best_accuracy.project_id,
            "name": best_accuracy.project_name,
            "error": best_accuracy.accuracy.l2_relative_error,
        }
    
    # Highest throughput
    with_throughput = [r for r in successful if r.throughput_fps]
    if with_throughput:
        highest = max(with_throughput, key=lambda r: r.throughput_fps)
        summary["highest_throughput"] = {
            "project": highest.project_id,
            "name": highest.project_name,
            "fps": highest.throughput_fps,
        }
    
    return summary


def format_number(value: float, precision: int = 2) -> str:
    """Format a number for human readability."""
    if value is None:
        return "N/A"
    
    if abs(value) >= 1e9:
        return f"{value/1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    elif abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms is None:
        return "N/A"
    
    if ms >= 3600000:  # >= 1 hour
        return f"{ms/3600000:.1f} hr"
    elif ms >= 60000:  # >= 1 minute
        return f"{ms/60000:.1f} min"
    elif ms >= 1000:  # >= 1 second
        return f"{ms/1000:.1f} sec"
    elif ms >= 1:
        return f"{ms:.1f} ms"
    else:
        return f"{ms*1000:.1f} µs"


def format_params(count: int) -> str:
    """Format parameter count."""
    if count is None:
        return "N/A"
    
    if count >= 1e9:
        return f"{count/1e9:.1f}B"
    elif count >= 1e6:
        return f"{count/1e6:.1f}M"
    elif count >= 1e3:
        return f"{count/1e3:.0f}K"
    else:
        return str(count)


def generate_markdown_tables(
    results: list[BenchmarkResult],
    output_dir: Path,
) -> list[Path]:
    """
    Generate markdown comparison tables.
    
    Args:
        results: List of BenchmarkResults
        output_dir: Directory to save tables
    
    Returns:
        List of paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    paths = []
    
    # Sort results by project ID
    results = sorted(results, key=lambda r: r.project_id)
    
    # Table 1: Performance Comparison
    perf_table = _generate_performance_table(results)
    perf_path = output_dir / f"performance_comparison_{timestamp}.md"
    with open(perf_path, 'w') as f:
        f.write(perf_table)
    paths.append(perf_path)
    
    # Table 2: Accuracy Comparison
    acc_table = _generate_accuracy_table(results)
    acc_path = output_dir / f"accuracy_comparison_{timestamp}.md"
    with open(acc_path, 'w') as f:
        f.write(acc_table)
    paths.append(acc_path)
    
    # Table 3: Pareto Summary
    pareto_table = _generate_pareto_table(results)
    pareto_path = output_dir / f"pareto_summary_{timestamp}.md"
    with open(pareto_path, 'w') as f:
        f.write(pareto_table)
    paths.append(pareto_path)
    
    # Combined table
    combined = f"# Benchmark Results\n\n"
    combined += f"Generated: {datetime.utcnow().isoformat()}Z\n\n"
    combined += perf_table + "\n\n"
    combined += acc_table + "\n\n"
    combined += pareto_table
    
    combined_path = output_dir / f"benchmark_tables_{timestamp}.md"
    with open(combined_path, 'w') as f:
        f.write(combined)
    paths.append(combined_path)
    
    return paths


def _generate_performance_table(results: list[BenchmarkResult]) -> str:
    """Generate performance comparison markdown table."""
    lines = [
        "## Performance Comparison\n",
        "| # | Project | Params | Size (MB) | Inference (ms) | Memory (GB) | FPS | Training | Framework |",
        "|---|---------|--------|-----------|----------------|-------------|-----|----------|-----------|",
    ]
    
    for r in results:
        pid = r.project_id
        name = r.project_name[:20]  # Truncate long names
        params = format_params(r.model.parameter_count)
        size = f"{r.model.model_size_mb:.2f}" if r.model.model_size_mb else "N/A"
        
        if r.inference_time_gpu:
            timing = f"{r.inference_time_gpu.mean_ms:.1f} ± {r.inference_time_gpu.std_ms:.1f}"
        else:
            timing = "N/A"
        
        memory = f"{r.memory.peak_gpu_gb:.3f}" if r.memory.peak_gpu_gb else "N/A"
        fps = f"{r.throughput_fps:.0f}" if r.throughput_fps else "N/A"
        
        if r.training.total_time_seconds:
            training = format_time(r.training.total_time_seconds * 1000)
        else:
            training = "N/A"
        
        framework = r.framework.capitalize()
        
        status_icon = "✅" if r.status == "success" else "❌"
        
        lines.append(f"| {pid} | {name} {status_icon} | {params} | {size} | {timing} | {memory} | {fps} | {training} | {framework} |")
    
    return "\n".join(lines)


def _generate_accuracy_table(results: list[BenchmarkResult]) -> str:
    """Generate accuracy comparison markdown table."""
    lines = [
        "## Accuracy Comparison\n",
        "| # | Project | Domain | RMSE | NRMSE | L2 Rel. Error | Key Metric | Value |",
        "|---|---------|--------|------|-------|---------------|------------|-------|",
    ]
    
    for r in results:
        if r.status != "success":
            continue
        
        pid = r.project_id
        name = r.project_name[:15]
        
        # Infer domain from project name
        domain = _infer_domain(r.project_name)
        
        rmse = f"{r.accuracy.rmse:.4f}" if r.accuracy.rmse else "N/A"
        nrmse = f"{r.accuracy.nrmse:.4f}" if r.accuracy.nrmse else "N/A"
        l2_err = f"{r.accuracy.l2_relative_error:.4f}" if r.accuracy.l2_relative_error else "N/A"
        
        # Get primary domain metric
        key_metric, key_value = _get_primary_domain_metric(r.domain_metrics)
        
        lines.append(f"| {pid} | {name} | {domain} | {rmse} | {nrmse} | {l2_err} | {key_metric} | {key_value} |")
    
    return "\n".join(lines)


def _generate_pareto_table(results: list[BenchmarkResult]) -> str:
    """Generate Pareto efficiency summary table."""
    lines = [
        "## Speed vs. Accuracy Tiers\n",
        "| Tier | Projects | Inference | L2 Error | Use Case |",
        "|------|----------|-----------|----------|----------|",
    ]
    
    # Categorize projects into tiers
    ultra_fast = []
    fast = []
    standard = []
    heavy = []
    
    for r in results:
        if r.status != "success" or not r.inference_time_gpu:
            continue
        
        t = r.inference_time_gpu.mean_ms
        if t < 1:
            ultra_fast.append(r.project_id)
        elif t < 5:
            fast.append(r.project_id)
        elif t < 50:
            standard.append(r.project_id)
        else:
            heavy.append(r.project_id)
    
    if ultra_fast:
        lines.append(f"| Ultra-fast | {', '.join(ultra_fast)} | < 1 ms | varies | Edge/mobile/real-time |")
    if fast:
        lines.append(f"| Fast | {', '.join(fast)} | < 5 ms | varies | Interactive/desktop |")
    if standard:
        lines.append(f"| Standard | {', '.join(standard)} | < 50 ms | varies | Near-real-time |")
    if heavy:
        lines.append(f"| Heavy | {', '.join(heavy)} | ≥ 50 ms | varies | Offline/batch |")
    
    if len(lines) == 3:
        lines.append("| - | No successful benchmarks | - | - | - |")
    
    return "\n".join(lines)


def _infer_domain(project_name: str) -> str:
    """Infer physics domain from project name."""
    name_lower = project_name.lower()
    
    if "cloth" in name_lower:
        return "Cloth"
    elif "wave" in name_lower:
        return "Wave/Acoustic"
    elif "maxwell" in name_lower:
        return "EM Fields"
    elif "coast" in name_lower or "flow" in name_lower or "surf" in name_lower:
        return "Fluid/Flow"
    elif "foil" in name_lower:
        return "Aerodynamics"
    elif "motion" in name_lower:
        return "Motion"
    elif "deform" in name_lower:
        return "Deformation"
    elif "cell" in name_lower or "path" in name_lower:
        return "Biology"
    elif "geo" in name_lower or "manifold" in name_lower:
        return "PDE/Manifold"
    else:
        return "Physics"


def _get_primary_domain_metric(domain_metrics: dict) -> tuple[str, str]:
    """Get the most important domain-specific metric."""
    if not domain_metrics:
        return "N/A", "N/A"
    
    # Priority order for different metric types
    priority_keywords = [
        "l2", "rmse", "error", "loss", "chamfer", "residual", 
        "violation", "preservation", "divergence"
    ]
    
    for keyword in priority_keywords:
        for name, value in domain_metrics.items():
            if keyword in name.lower():
                formatted = format_number(value, 4) if isinstance(value, (int, float)) else str(value)
                return name, formatted
    
    # Return first metric if no priority match
    name, value = next(iter(domain_metrics.items()))
    formatted = format_number(value, 4) if isinstance(value, (int, float)) else str(value)
    return name, formatted


def generate_html_dashboard(
    results: list[BenchmarkResult],
    output_dir: Path,
    filename: str = "dashboard.html",
) -> Path:
    """
    Generate an interactive HTML dashboard with charts.
    
    Uses Chart.js via CDN for visualizations.
    
    Args:
        results: List of BenchmarkResults
        output_dir: Directory to save dashboard
        filename: Output filename
    
    Returns:
        Path to generated HTML file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for charts
    successful = [r for r in results if r.status == "success"]
    
    labels = [r.project_id for r in successful]
    params = [r.model.parameter_count for r in successful]
    inference_times = [
        r.inference_time_gpu.mean_ms if r.inference_time_gpu else 0 
        for r in successful
    ]
    l2_errors = [
        r.accuracy.l2_relative_error if r.accuracy.l2_relative_error else 0
        for r in successful
    ]
    fps_values = [r.throughput_fps if r.throughput_fps else 0 for r in successful]
    
    # Generate scatter data for Pareto chart
    scatter_data = []
    for r in successful:
        if r.inference_time_gpu and r.accuracy.l2_relative_error:
            scatter_data.append({
                "x": r.inference_time_gpu.mean_ms,
                "y": r.accuracy.l2_relative_error,
                "label": r.project_id,
            })
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PINN Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0; color: #333; }}
        .header p {{ color: #666; margin-top: 5px; }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin: 0 0 15px 0;
            font-size: 16px;
            color: #333;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f9f9f9; font-weight: 600; }}
        .status-success {{ color: #4CAF50; }}
        .status-failed {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 PINN Benchmark Dashboard</h1>
        <p>Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC | {len(successful)}/{len(results)} successful</p>
    </div>
    
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{len(results)}</div>
            <div class="stat-label">Total Projects</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(successful)}</div>
            <div class="stat-label">Successful</div>
        </div>
        <div class="stat">
            <div class="stat-value">{min(inference_times) if inference_times else 'N/A':.2f} ms</div>
            <div class="stat-label">Fastest Inference</div>
        </div>
        <div class="stat">
            <div class="stat-value">{min(l2_errors) if [e for e in l2_errors if e > 0] else 'N/A':.4f}</div>
            <div class="stat-label">Best L2 Error</div>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h2>📊 Parameter Count by Project</h2>
            <div class="chart-container">
                <canvas id="paramsChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>⚡ Inference Time (ms)</h2>
            <div class="chart-container">
                <canvas id="inferenceChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 Speed vs Accuracy (Pareto)</h2>
            <div class="chart-container">
                <canvas id="paretoChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>🚀 Throughput (FPS)</h2>
            <div class="chart-container">
                <canvas id="fpsChart"></canvas>
            </div>
        </div>
    </div>
    
    <div style="max-width: 1400px; margin: 20px auto;">
        <div class="card">
            <h2>📋 Full Results Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Project</th>
                        <th>Framework</th>
                        <th>Parameters</th>
                        <th>Inference (ms)</th>
                        <th>L2 Error</th>
                        <th>FPS</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(_generate_table_rows(results))}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Parameter count chart
        new Chart(document.getElementById('paramsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Parameters',
                    data: {json.dumps(params)},
                    backgroundColor: 'rgba(33, 150, 243, 0.7)',
                    borderColor: 'rgba(33, 150, 243, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'Parameters (log scale)' }}
                    }}
                }}
            }}
        }});
        
        // Inference time chart
        new Chart(document.getElementById('inferenceChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Inference Time (ms)',
                    data: {json.dumps(inference_times)},
                    backgroundColor: 'rgba(76, 175, 80, 0.7)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'Time (ms, log scale)' }}
                    }}
                }}
            }}
        }});
        
        // Pareto chart (scatter)
        new Chart(document.getElementById('paretoChart'), {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Projects',
                    data: {json.dumps(scatter_data)},
                    backgroundColor: 'rgba(156, 39, 176, 0.7)',
                    borderColor: 'rgba(156, 39, 176, 1)',
                    pointRadius: 8,
                    pointHoverRadius: 10
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'Inference Time (ms)' }}
                    }},
                    y: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'L2 Relative Error' }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.raw.label + ': ' + context.raw.x.toFixed(2) + 'ms, ' + context.raw.y.toFixed(4);
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // FPS chart
        new Chart(document.getElementById('fpsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Throughput (FPS)',
                    data: {json.dumps(fps_values)},
                    backgroundColor: 'rgba(255, 152, 0, 0.7)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        title: {{ display: true, text: 'FPS' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    return filepath


def _generate_table_rows(results: list[BenchmarkResult]) -> list[str]:
    """Generate HTML table rows for results."""
    rows = []
    for r in sorted(results, key=lambda x: x.project_id):
        status_class = "status-success" if r.status == "success" else "status-failed"
        status_icon = "✅" if r.status == "success" else "❌"
        
        inference = (
            f"{r.inference_time_gpu.mean_ms:.2f}" 
            if r.inference_time_gpu else "N/A"
        )
        l2_err = (
            f"{r.accuracy.l2_relative_error:.4f}"
            if r.accuracy.l2_relative_error else "N/A"
        )
        fps = f"{r.throughput_fps:.0f}" if r.throughput_fps else "N/A"
        
        rows.append(f'''
            <tr>
                <td>{r.project_id}</td>
                <td>{r.project_name}</td>
                <td>{r.framework}</td>
                <td>{format_params(r.model.parameter_count)}</td>
                <td>{inference}</td>
                <td>{l2_err}</td>
                <td>{fps}</td>
                <td class="{status_class}">{status_icon} {r.status}</td>
            </tr>
        ''')
    
    return rows
