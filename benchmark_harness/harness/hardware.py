"""
Hardware and software environment detection.

Collects system information for reproducibility.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional

from .core import HardwareInfo, SoftwareInfo


def get_hardware_info() -> HardwareInfo:
    """
    Detect and return hardware environment information.
    
    Returns:
        HardwareInfo dataclass with GPU, CPU, RAM, and OS info
    """
    # OS
    os_name = f"{platform.system()} {platform.release()}"
    
    # CPU
    cpu_name = _get_cpu_name()
    
    # RAM
    ram_gb = _get_ram_gb()
    
    # GPU
    gpu_name, gpu_memory_gb, cuda_version = _get_gpu_info()
    
    return HardwareInfo(
        gpu=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cuda_version=cuda_version,
        cpu=cpu_name,
        ram_gb=ram_gb,
        os=os_name,
    )


def _get_cpu_name() -> str:
    """Get CPU model name."""
    try:
        if platform.system() == "Darwin":
            # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif platform.system() == "Linux":
            # Linux
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        elif platform.system() == "Windows":
            # Windows
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return cpu_name
    except Exception:
        pass
    
    return platform.processor() or "Unknown"


def _get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except Exception:
        pass
    
    return 0.0


def _get_gpu_info() -> tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Get GPU information via nvidia-smi.
    
    Returns:
        Tuple of (gpu_name, gpu_memory_gb, cuda_version)
    """
    gpu_name = None
    gpu_memory_gb = None
    cuda_version = None
    
    try:
        # Get GPU name
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split('\n')[0]
        
        # Get GPU memory
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_memory_mb = float(result.stdout.strip().split('\n')[0])
            gpu_memory_gb = gpu_memory_mb / 1024
        
        # Get CUDA version
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip().split('\n')[0]
            # Also try to get CUDA version
            result2 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=5
            )
            if result2.returncode == 0:
                # Parse CUDA version from nvidia-smi output
                for line in result2.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        parts = line.split('CUDA Version:')
                        if len(parts) > 1:
                            cuda_version = parts[1].strip().split()[0]
                            break
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return gpu_name, gpu_memory_gb, cuda_version


def get_software_info(framework: str = "pytorch") -> SoftwareInfo:
    """
    Get software version information.
    
    Args:
        framework: Primary framework ("pytorch", "jax", "tensorflow", "onnx")
    
    Returns:
        SoftwareInfo dataclass
    """
    python_version = platform.python_version()
    framework_version = "Unknown"
    cuda_toolkit = None
    
    if framework == "pytorch":
        try:
            import torch
            framework_version = torch.__version__
            if torch.cuda.is_available():
                cuda_toolkit = torch.version.cuda
        except ImportError:
            pass
    
    elif framework == "jax":
        try:
            import jax
            framework_version = jax.__version__
            # JAX CUDA version is harder to get
        except ImportError:
            pass
    
    elif framework == "tensorflow":
        try:
            import tensorflow as tf
            framework_version = tf.__version__
        except ImportError:
            pass
    
    elif framework == "onnx":
        try:
            import onnxruntime as ort
            framework_version = ort.__version__
        except ImportError:
            pass
    
    return SoftwareInfo(
        python=python_version,
        framework=framework,
        framework_version=framework_version,
        cuda_toolkit=cuda_toolkit,
    )


def get_git_hash(repo_path: Optional[str] = None) -> Optional[str]:
    """
    Get current git commit hash.
    
    Args:
        repo_path: Path to git repository (default: current directory)
    
    Returns:
        Git commit hash (short) or None if not a git repo
    """
    try:
        cwd = repo_path or os.getcwd()
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_environment_summary() -> dict:
    """
    Get complete environment summary for benchmark reproducibility.
    
    Returns:
        Dict with all environment information
    """
    hardware = get_hardware_info()
    
    # Try to detect installed frameworks
    frameworks = {}
    
    try:
        import torch
        frameworks["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            frameworks["pytorch_cuda"] = torch.version.cuda
    except ImportError:
        pass
    
    try:
        import jax
        frameworks["jax"] = jax.__version__
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        frameworks["tensorflow"] = tf.__version__
    except ImportError:
        pass
    
    try:
        import onnxruntime as ort
        frameworks["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    
    try:
        import torch_geometric
        frameworks["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    
    return {
        "hardware": hardware.to_dict(),
        "python_version": platform.python_version(),
        "frameworks": frameworks,
        "git_hash": get_git_hash(),
    }


def check_dependencies() -> dict[str, bool]:
    """
    Check which optional dependencies are available.
    
    Returns:
        Dict mapping dependency name to availability
    """
    dependencies = {
        "torch": False,
        "torch_cuda": False,
        "jax": False,
        "jax_gpu": False,
        "tensorflow": False,
        "onnxruntime": False,
        "torch_geometric": False,
        "torch_cluster": False,
        "torch_scatter": False,
        "torch_sparse": False,
        "psutil": False,
        "pyyaml": False,
    }
    
    try:
        import torch
        dependencies["torch"] = True
        dependencies["torch_cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import jax
        dependencies["jax"] = True
        devices = jax.devices()
        dependencies["jax_gpu"] = any(d.platform == "gpu" for d in devices)
    except ImportError:
        pass
    
    try:
        import tensorflow
        dependencies["tensorflow"] = True
    except ImportError:
        pass
    
    try:
        import onnxruntime
        dependencies["onnxruntime"] = True
    except ImportError:
        pass
    
    try:
        import torch_geometric
        dependencies["torch_geometric"] = True
    except ImportError:
        pass
    
    try:
        import torch_cluster
        dependencies["torch_cluster"] = True
    except ImportError:
        pass
    
    try:
        import torch_scatter
        dependencies["torch_scatter"] = True
    except ImportError:
        pass
    
    try:
        import torch_sparse
        dependencies["torch_sparse"] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        pass
    
    try:
        import yaml
        dependencies["pyyaml"] = True
    except ImportError:
        pass
    
    return dependencies
