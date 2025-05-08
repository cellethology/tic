# file: tic/graph/_typing.py
"""Central place for custom typing aliases used across ``tic.graph``."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import scipy.sparse as sp

# -----------------------------------------------------------------------------
# ``AnnData`` is *optional* at *import* time to keep stub files importable
# even if ``anndata`` is not installed in every environment (e.g. docs).
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover – type‑checking only
    from anndata import AnnData  # noqa: F401

ArrayLike = np.ndarray | Sequence[float] | Sequence[int]
SparseMatrix = sp.spmatrix

__all__ = [
    "ArrayLike",
    "SparseMatrix",
]