# file: tic/graph/tl/metrics.py
"""Local graph statistics (stub)."""
from __future__ import annotations

import scipy.sparse as sp

from ..utils import get_connectivities_key

try:
    from anndata import AnnData
except ImportError as exc:  # pragma: no cover
    raise ImportError("local_degree requires the `anndata` package") from exc


def local_degree(
    adata: "AnnData",
    center: int | str,
    *,
    graph_key: str | None = None,
) -> int:
    """Return the degree of ``center`` in the stored graph (stub)."""
    key = get_connectivities_key(graph_key)
    mat: sp.spmatrix = adata.obsp[key]
    idx: int = adata.obs_names.get_loc(center) if isinstance(center, str) else center
    return int(mat[idx].getnnz())
