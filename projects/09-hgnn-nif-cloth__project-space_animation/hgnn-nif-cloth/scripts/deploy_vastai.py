"""
Vast.ai Deployment Script for HGNN-NIF-Cloth

Automates:
1. Search for available L40S GPU instances
2. Create and provision instance
3. Upload project code
4. Install dependencies
5. Run validation tests

Usage:
    pip install vastai
    python scripts/deploy_vastai.py
    
Requirements:
    - Vast.ai CLI installed: pip install vastai
    - API key configured
"""

import os
import subprocess
import json
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import re

# Configuration
VASTAI_API_KEY = "REDACTED_VASTAI_API_KEY"
TARGET_GPU = "L40S"
MIN_VRAM_GB = 45
MIN_DISK_GB = 50
DOCKER_IMAGE = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"

# Set API key in environment
os.environ['VASTAI_API_KEY'] = VASTAI_API_KEY


def run_command(cmd: List[str], capture: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def check_vastai_installed() -> bool:
    """Check if vastai CLI is installed."""
    code, _, _ = run_command(['which', 'vastai'])
    return code == 0


def search_gpu_instances(
    gpu_name: str = TARGET_GPU,
    min_vram: int = MIN_VRAM_GB
) -> List[Dict]:
    """
    Search for available GPU instances with fallback options.
    
    Returns list of available offers sorted by price.
    """
    # GPU fallback priority: L40S > L40 > A100 > RTX_4090 > RTX_3090
    gpu_fallbacks = [gpu_name, 'L40', 'A100', 'RTX_4090', 'RTX_3090']
    
    for gpu in gpu_fallbacks:
        print(f"Searching for {gpu} instances...")
        
        # Build search query
        query = f"gpu_name={gpu} rentable=true verified=true disk_space>={MIN_DISK_GB}"
        
        cmd = [
            'vastai', 'search', 'offers',
            query,
            '--raw'
        ]
        
        code, stdout, stderr = run_command(cmd)
        
        if code != 0:
            print(f"  Error: {stderr.strip() if stderr else 'search failed'}")
            continue
        
        try:
            offers = json.loads(stdout)
        except json.JSONDecodeError:
            print(f"  Warning: Could not parse response")
            continue
        
        if not offers:
            print(f"  No {gpu} instances available")
            continue
        
        # Adjust min_vram for smaller GPUs
        effective_min_vram = min_vram if gpu in ['L40S', 'L40', 'A100'] else 20
        
        # Filter and sort by price
        valid_offers = [
            o for o in offers 
            if o.get('gpu_ram', 0) >= effective_min_vram * 1024  # Convert GB to MB
            and o.get('rentable', False)
        ]
        
        if valid_offers:
            print(f"  Found {len(valid_offers)} {gpu} instances!")
            # Sort by price (dph_total)
            valid_offers.sort(key=lambda x: x.get('dph_total', float('inf')))
            return valid_offers[:10]  # Return top 10
        else:
            print(f"  No {gpu} with >= {effective_min_vram}GB VRAM")
    
    print("\nNo suitable GPU instances found in any fallback category")
    return []


def display_offers(offers: List[Dict]) -> None:
    """Display available offers in a table format."""
    print("\nAvailable instances:")
    print("-" * 80)
    print(f"{'ID':>8} | {'GPU':>10} | {'VRAM':>6} | {'CPU':>4} | {'RAM':>6} | {'Disk':>6} | {'$/hr':>8}")
    print("-" * 80)
    
    for offer in offers:
        print(f"{offer.get('id', 'N/A'):>8} | "
              f"{offer.get('gpu_name', 'N/A')[:10]:>10} | "
              f"{offer.get('gpu_ram', 0) / 1024:>5.0f}G | "
              f"{offer.get('cpu_cores_effective', 0):>4.0f} | "
              f"{offer.get('cpu_ram', 0) / 1024:>5.0f}G | "
              f"{offer.get('disk_space', 0):>5.0f}G | "
              f"${offer.get('dph_total', 0):>7.3f}")
    print("-" * 80)


def create_instance(offer_id: int) -> Optional[int]:
    """
    Create a new instance from an offer.
    
    Returns instance ID on success.
    """
    print(f"\nCreating instance from offer {offer_id}...")
    
    cmd = [
        'vastai', 'create', 'instance', str(offer_id),
        '--image', DOCKER_IMAGE,
        '--disk', str(MIN_DISK_GB),
        '--raw'
    ]
    
    code, stdout, stderr = run_command(cmd)
    
    if code != 0:
        print(f"Error creating instance: {stderr}")
        return None
    
    try:
        result = json.loads(stdout)
        instance_id = result.get('new_contract')
        print(f"Created instance: {instance_id}")
        return instance_id
    except (json.JSONDecodeError, KeyError):
        # Try to parse instance ID from stdout
        match = re.search(r'new_contract.*?(\d+)', stdout)
        if match:
            return int(match.group(1))
        print(f"Failed to get instance ID: {stdout}")
        return None


def get_instance_status(instance_id: int) -> Dict:
    """Get instance status and connection info."""
    cmd = ['vastai', 'show', 'instance', str(instance_id), '--raw']
    code, stdout, stderr = run_command(cmd)
    
    if code != 0:
        return {}
    
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {}


def wait_for_instance(instance_id: int, timeout: int = 300) -> bool:
    """
    Wait for instance to be running.
    
    Returns True if instance is ready, False on timeout.
    """
    print(f"Waiting for instance {instance_id} to start...")
    
    start_time = time.time()
    last_status = ""
    
    while time.time() - start_time < timeout:
        status_info = get_instance_status(instance_id)
        
        if isinstance(status_info, list) and len(status_info) > 0:
            status_info = status_info[0]
        
        current_status = status_info.get('actual_status', 'unknown')
        
        if current_status != last_status:
            print(f"  Status: {current_status}")
            last_status = current_status
        
        if current_status == 'running':
            print(f"\n✓ Instance {instance_id} is running!")
            return True
        
        time.sleep(10)
    
    print(f"\n✗ Timeout waiting for instance")
    return False


def get_ssh_info(instance_id: int) -> Tuple[Optional[str], Optional[int]]:
    """Get SSH connection info for an instance."""
    status = get_instance_status(instance_id)
    
    if isinstance(status, list) and len(status) > 0:
        status = status[0]
    
    ssh_host = status.get('ssh_host')
    ssh_port = status.get('ssh_port')
    
    return ssh_host, ssh_port


def setup_ssh_key() -> str:
    """Ensure SSH key is available and return path."""
    ssh_key_path = Path.home() / '.ssh' / 'id_rsa'
    
    if not ssh_key_path.exists():
        print("Generating SSH key...")
        run_command([
            'ssh-keygen', '-t', 'rsa', '-b', '4096',
            '-f', str(ssh_key_path), '-N', ''
        ])
    
    return str(ssh_key_path)


def upload_project(ssh_host: str, ssh_port: int) -> bool:
    """Upload project files to instance."""
    print("\nUploading project files...")
    
    project_root = Path(__file__).parent.parent
    
    # Create directory on remote
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host}"
    run_command([
        'bash', '-c',
        f'{ssh_cmd} "mkdir -p /workspace/hgnn-nif-cloth"'
    ])
    
    # Rsync project files
    cmd = [
        'rsync', '-avz', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -p {ssh_port}',
        '--exclude', '__pycache__',
        '--exclude', '*.pyc',
        '--exclude', '.git',
        '--exclude', 'outputs',
        '--exclude', 'data/*.h5',
        f'{project_root}/',
        f'root@{ssh_host}:/workspace/hgnn-nif-cloth/'
    ]
    
    code, stdout, stderr = run_command(cmd, capture=False)
    
    if code != 0:
        print(f"Upload failed: {stderr}")
        return False
    
    print("✓ Project uploaded successfully")
    return True


def install_dependencies(ssh_host: str, ssh_port: int) -> bool:
    """Install Python dependencies on remote instance."""
    print("\nInstalling dependencies...")
    
    install_script = """
cd /workspace/hgnn-nif-cloth && \
pip install --no-cache-dir numpy scipy h5py matplotlib pytest tensorboard tqdm pyyaml && \
pip list | grep -E "torch|numpy|scipy"
"""
    
    cmd = [
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-p', str(ssh_port),
        f'root@{ssh_host}',
        install_script
    ]
    
    code, stdout, stderr = run_command(cmd, capture=False)
    
    if code != 0:
        print(f"Install failed")
        return False
    
    print("✓ Dependencies installed")
    return True


def run_remote_test(ssh_host: str, ssh_port: int) -> bool:
    """Run validation test on remote instance."""
    print("\nRunning validation tests...")
    
    test_script = """
cd /workspace/hgnn-nif-cloth && \
echo "=== GPU Info ===" && \
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" && \
echo "" && \
echo "=== Running Tests ===" && \
python -m pytest tests/test_forward.py -v --tb=short
"""
    
    cmd = [
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-p', str(ssh_port),
        f'root@{ssh_host}',
        test_script
    ]
    
    code, _, _ = run_command(cmd, capture=False)
    return code == 0


def run_training_test(ssh_host: str, ssh_port: int) -> bool:
    """Run quick training test on remote instance."""
    print("\nRunning training test (3 epochs)...")
    
    train_script = """
cd /workspace/hgnn-nif-cloth && \
python scripts/train_local.py --test_mode --output_dir outputs/test_run
"""
    
    cmd = [
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-p', str(ssh_port),
        f'root@{ssh_host}',
        train_script
    ]
    
    code, _, _ = run_command(cmd, capture=False)
    return code == 0


def destroy_instance(instance_id: int) -> bool:
    """Destroy an instance."""
    print(f"\nDestroying instance {instance_id}...")
    cmd = ['vastai', 'destroy', 'instance', str(instance_id)]
    code, _, _ = run_command(cmd)
    return code == 0


def main():
    print("=" * 60)
    print("HGNN-NIF-Cloth Vast.ai Deployment")
    print("=" * 60)
    
    # Check vastai CLI
    if not check_vastai_installed():
        print("ERROR: vastai CLI not found. Install with: pip install vastai")
        sys.exit(1)
    
    # Step 1: Search for instances
    print(f"\n[1/6] Searching for {TARGET_GPU} instances...")
    offers = search_gpu_instances()
    
    if not offers:
        print("No suitable instances found!")
        sys.exit(1)
    
    display_offers(offers)
    
    # Step 2: Select instance
    cheapest = offers[0]
    print(f"\nCheapest option: ID {cheapest['id']} at ${cheapest.get('dph_total', 0):.3f}/hr")
    
    user_input = input("\nEnter offer ID (or press Enter for cheapest): ").strip()
    offer_id = int(user_input) if user_input else cheapest['id']
    
    # Step 3: Create instance
    print(f"\n[2/6] Creating instance from offer {offer_id}...")
    instance_id = create_instance(offer_id)
    
    if instance_id is None:
        print("Failed to create instance!")
        sys.exit(1)
    
    # Step 4: Wait for instance
    print(f"\n[3/6] Waiting for instance {instance_id} to start...")
    if not wait_for_instance(instance_id, timeout=300):
        print("Instance failed to start. Cleaning up...")
        destroy_instance(instance_id)
        sys.exit(1)
    
    # Get SSH info
    time.sleep(10)  # Wait for SSH to be ready
    ssh_host, ssh_port = get_ssh_info(instance_id)
    
    if not ssh_host or not ssh_port:
        print("Failed to get SSH info")
        sys.exit(1)
    
    print(f"SSH: ssh -p {ssh_port} root@{ssh_host}")
    
    # Step 5: Setup instance
    print(f"\n[4/6] Setting up instance...")
    
    if not upload_project(ssh_host, ssh_port):
        print("Failed to upload project")
        sys.exit(1)
    
    if not install_dependencies(ssh_host, ssh_port):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Step 6: Run tests
    print(f"\n[5/6] Running validation tests...")
    
    if not run_remote_test(ssh_host, ssh_port):
        print("Validation tests failed!")
    else:
        print("✓ Validation tests passed!")
    
    # Step 7: Training test
    print(f"\n[6/6] Running training test...")
    
    if run_training_test(ssh_host, ssh_port):
        print("✓ Training test passed!")
    else:
        print("Training test had issues")
    
    # Summary
    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print(f"\nInstance ID: {instance_id}")
    print(f"SSH Command: ssh -p {ssh_port} root@{ssh_host}")
    print(f"\nTo connect:  vastai ssh {instance_id}")
    print(f"To destroy:  vastai destroy instance {instance_id}")
    print(f"\nEstimated cost: ${cheapest.get('dph_total', 0):.3f}/hour")
    
    # Ask about keeping instance
    keep = input("\nKeep instance running? (y/n): ").strip().lower()
    if keep != 'y':
        destroy_instance(instance_id)
        print("Instance destroyed.")


if __name__ == '__main__':
    main()
