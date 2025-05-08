# tic/__init__.py
"""
TIC: Temporal Inference of Cells.
Main package initializer, exposing public API and version.
"""

__version__ = "2.0.0"

# Annotation
from .annotation import LLMPredictor, get_llm, assign_cell_clusters, return_top_genes

# Configuration & constants
from .config import GraphConfig, PseudotimeConfig, FeatureConfig

# Core pipelines
from .pipeline import (
    pseudotime as pipeline_pseudotime,
)

# Data handling
from .data import (
    io as data_io,
)

# Feature extraction
from .features import (
    base as features_base,
    registry as features_registry,
)

# Causal inference
from .causal import (
    base as causal_base,
    factory as causal_factory,
)

# Metrics
from .metrics import (
    calculate_monotonicity,
    calculate_trend,
    rank_by_monotonicity,
    rank_by_trend,
)

__all__ = [
    "__version__",
    "GraphConfig",
    "PseudotimeConfig",
    "FeatureConfig",
    "pipeline_pseudotime",
    "data_io",
    "features_base",
    "features_registry",
    "causal_base",
    "causal_factory",
    "calculate_monotonicity",
    "calculate_trend",
    "rank_by_monotonicity",
    "rank_by_trend",
    "assign_cell_clusters",
    "return_top_genes",
    "LLMPredictor",
    "get_llm",
]