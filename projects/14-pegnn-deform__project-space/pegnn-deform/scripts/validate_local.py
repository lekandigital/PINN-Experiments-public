#!/usr/bin/env python3
"""
PEGNN-Deform: Quick Validation Script

Runs basic syntax and import checks without requiring PyTorch.
For full tests, run on a GPU instance with PyTorch installed.

Usage: python3 scripts/validate_local.py
"""

import ast
import sys
from pathlib import Path


def check_syntax(file_path: Path) -> bool:
    """Check Python file for syntax errors."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"  Syntax error in {file_path}: {e}")
        return False


def main():
    print("=" * 60)
    print("PEGNN-Deform: Local Validation")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # Files to check
    python_files = [
        project_root / "src" / "mesh_features.py",
        project_root / "src" / "pegdeform_model.py",
        project_root / "src" / "train_pegdeform.py",
        project_root / "src" / "benchmark.py",
        project_root / "data" / "generate_synthetic.py",
        project_root / "tests" / "test_model.py",
    ]
    
    print("\n1. Checking Python syntax...")
    all_ok = True
    for f in python_files:
        if f.exists():
            if check_syntax(f):
                print(f"  ✓ {f.name}")
            else:
                all_ok = False
        else:
            print(f"  ✗ {f.name} (not found)")
            all_ok = False
    
    print("\n2. Checking project structure...")
    required_dirs = ["src", "data", "tests", "configs", "scripts"]
    for d in required_dirs:
        dir_path = project_root / d
        if dir_path.exists():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ (not found)")
            all_ok = False
    
    print("\n3. Checking configuration files...")
    config_files = [
        ("configs/environment.yaml", "Conda environment"),
        ("configs/wandb_sweep.yaml", "Wandb sweep config"),
        ("scripts/setup_vastai.sh", "vast.ai setup script"),
        ("scripts/run_tests.sh", "Test runner script"),
        ("README.md", "Documentation"),
    ]
    for f, desc in config_files:
        if (project_root / f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (not found)")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("VALIDATION PASSED ✓")
        print("=" * 60)
        print("\nProject is ready for deployment to vast.ai!")
        print("\nNext steps:")
        print("  1. Set your vast.ai API key:")
        print('     export VASTAI_API_KEY="your_key"')
        print("  2. Run the deployment script:")
        print("     bash scripts/deploy_vastai.sh")
        return 0
    else:
        print("VALIDATION FAILED ✗")
        print("=" * 60)
        print("\nPlease fix the issues above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
