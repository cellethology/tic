# tic/utils/__init__.py
"""
tic.utils: Utility functions for logging and experiment management.
"""

from .logging import get_logger, save_experiment_config
from .seed import set_random_seed
__all__ = [
    "get_logger",
    "save_experiment_config",
    "set_random_seed",
]