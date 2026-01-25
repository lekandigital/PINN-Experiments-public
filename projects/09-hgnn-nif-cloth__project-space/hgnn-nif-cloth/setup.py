"""
HGNN-NIF-Cloth: Hybrid Graph Neural Network + Neural Implicit Field for Cloth Simulation

A hybrid framework combining hierarchical GNNs with SIREN-based neural implicit fields.
"""

from setuptools import setup, find_packages

setup(
    name="hgnn-nif-cloth",
    version="0.1.0",
    description="Hybrid HGNN + Neural Implicit Field for Cloth Simulation",
    author="HGNN-NIF-Cloth Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black",
            "flake8",
        ],
        "deploy": [
            "vastai",
        ],
    },
    entry_points={
        "console_scripts": [
            "hgnn-train=scripts.train_local:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
