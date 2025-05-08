"""Build neighbor graphs and store in ``AnnData``."""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Voronoi
from sklearn.neighbors import BallTree, KDTree

from ..utils import get_connectivities_key

try:
    from anndata import AnnData
except ImportError as exc:  # pragma: no cover – optional at import time
    raise ImportError("compute_neighbors requires the `anndata` package") from exc


def compute_neighbors(
    adata: "AnnData",
    *,
    method: Literal["knn", "radius", "voronoi"] = "knn",
    k: int = 6,
    radius: float | None = None,
    metric: str = "euclidean",
    key_added: str | None = None,
    copy: bool = False,
) -> "AnnData" | None:
    """Populate `adata.obsp[key_connectivities]` with a sparse adjacency matrix.

    Parameters
    ----------
    adata
        Annotated data matrix with `obsm['spatial']` coordinates.
    method
        'knn' for k-nearest neighbors, 'radius' for fixed-distance neighbors, 'voronoi' for Voronoi-based adjacency.
    k
        Number of neighbors for 'knn'.
    radius
        Distance threshold for 'radius' (required if method='radius').
    metric
        Distance metric for KDTree/BallTree ('knn' or 'radius').
    key_added
        Custom key to store adjacency in `adata.obsp`.
    copy
        If True, returns a new AnnData with adjacency; otherwise modifies in place.

    Returns
    -------
    AnnData or None
        Updated AnnData if `copy=True`, else None.
    """
    if copy:
        adata = adata.copy()

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    n_cells = coords.shape[0]

    key = get_connectivities_key(key_added)

    # Build adjacency
    if method == "knn":
        tree = KDTree(coords, metric=metric)
        idxs = tree.query(coords, k=min(k + 1, n_cells), return_distance=False)
        nbrs = idxs[:, 1:k+1]  # Remove self (first column)
        rows = np.repeat(np.arange(n_cells), nbrs.shape[1])
        cols = nbrs.ravel()
        data = np.ones_like(rows, dtype=np.float32)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    elif method == "radius":
        if radius is None:
            raise ValueError("`radius` must be provided for method='radius'.")
        tree = BallTree(coords, metric=metric)
        idxs = tree.query_radius(coords, r=radius)
        rows, cols = [], []
        for i, nbr in enumerate(idxs):
            nbr = nbr[nbr != i]  # Exclude self
            rows.extend([i] * len(nbr))
            cols.extend(nbr)
        data = np.ones(len(rows), dtype=np.float32)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    elif method == "voronoi":
        vor = Voronoi(coords)
        edges = set()
        for i, j in vor.ridge_points:
            edges.add((min(i, j), max(i, j)))
        if edges:
            rows, cols = zip(*edges)
            data = np.ones(len(rows), dtype=np.float32)
            mat = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        else:
            mat = sp.csr_matrix((n_cells, n_cells), dtype=np.float32)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Symmetrize adjacency (undirected)
    adj = (mat + mat.T).maximum(1).tocsr()

    adata.obsp[key] = adj

    # Record parameters
    graph_params = dict(method=method)
    if method == "knn":
        graph_params.update(k=k, metric=metric)
    elif method == "radius":
        graph_params.update(radius=radius, metric=metric)
    adata.uns.setdefault("graph_params", {})[key] = graph_params

    return adata if copy else None