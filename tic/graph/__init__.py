# file: tic/graph/__init__.py
"""Top-level namespace for ``tic.graph``.

This sub-package mirrors Scanpy's design:
* ``pp`` - pre-processing utilities that build and cache global graphs in
  ``AnnData``.
* ``tl`` - downstream tools (sub-graph extraction, metrics)…
* ``io`` - conversions to other graph formats.

Only *public* symbols are re-exported here to keep the external API tidy.
"""
from __future__ import annotations

from importlib import metadata as _metadata

from . import pp, tl  # re‑export sub‑packages
from .io import to_networkx, to_pyg
from .utils import get_connectivities_key, estimate_radius

__all__ = [
    "pp",
    "tl",
    "to_networkx",
    "to_pyg",
    "get_connectivities_key",
    "estimate_radius",
]

# Package version (falls back to "0.0.0" if not installed via pip)
try:
    __version__: str = _metadata.version(__name__.replace(".", "-"))
except _metadata.PackageNotFoundError:  # pragma: no cover – editable install
    __version__ = "0.0.0"