# tic/metrics/__init__.py
"""
tic.metrics: Trend and monotonicity analysis.
"""

from .api import (
    calculate_monotonicity,
    calculate_trend,
    rank_by_monotonicity,
    rank_by_trend,
    compute_variability
)


__all__ = [
    "calculate_monotonicity",
    "calculate_trend",
    "rank_by_monotonicity",
    "rank_by_trend",
    "compute_variability"
]