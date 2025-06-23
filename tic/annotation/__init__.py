
"""
The `tic.annotation` package.
Provides functions for single-cell clustering and LLM-based cluster annotation.
"""

from .annotation import assign_cell_clusters, return_top_genes
from .llm_caller import LLMPredictor
from .llm_backends import get_llm
from .api import annotate_adata, annotate_emt_state

__all__ = [
    "assign_cell_clusters",
    "return_top_genes",
    "LLMPredictor",
    "get_llm",
    "annotate_adata",
    "annotate_emt_state",
]
