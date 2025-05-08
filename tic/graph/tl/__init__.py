# file: tic/graph/tl/__init__.py
"""Tools built *on top* of global graphs (subgraph extraction, metrics…)."""
from __future__ import annotations

from .subgraph import extract_subgraph  # noqa: F401
from .metrics import local_degree  # noqa: F401
__all__ = [
    "extract_subgraph",
    "local_degree",
]