# file: tic/pseudotime/__init__.py
"""Pseudotime inference utilities.

The public API is re-exported here so users can simply::

    import tic.pseudotime as tp

and access all functions.
"""
from __future__ import annotations

from . import pp, tl  # noqa: F401 – sub‑packages as attributes
from .pp.dimensionality import DimensionalityReduction  # noqa: F401
from .pp.clustering import Clustering  # noqa: F401
from .tl.slingshot import SlingshotMethod  # noqa: F401
from .tl.base import PseudotimeMethod  # noqa: F401

__all__ = [
    "pp",
    "tl",
    "DimensionalityReduction",
    "Clustering",
    "PseudotimeMethod",
    "SlingshotMethod",
]