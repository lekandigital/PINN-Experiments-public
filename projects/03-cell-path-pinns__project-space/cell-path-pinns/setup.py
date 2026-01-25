from setuptools import setup, find_packages

setup(
    name='cell-path-pinns',
    version='0.1.0',
    author='Research Team',
    description='Physics-informed neural networks for microbe trajectory prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
    ],
    extras_require={
        'dev': ['pytest>=7.0.0', 'jupyter>=1.0.0'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
