"""
Backend adapters package.

Each adapter wraps a specific project's model and translates between
the abstract ModelBackend interface and the project's actual API.
"""

# Adapters are discovered dynamically by the BackendManager
# to avoid import errors when dependencies are missing.
