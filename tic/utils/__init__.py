# tic/utils/__init__.py
"""
tic.utils: Utility functions for logging and experiment management.
"""

from .logging import get_logger, save_experiment_config
__all__ = [
    "get_logger",
    "save_experiment_config",
]