from __future__ import annotations
"""
Spatial spot assignment and aggregation utilities.

This module provides functions to partition single-cell AnnData into spatial spots
using k-means clustering or a regular grid, and to aggregate expression and metadata
at the spot level.
"""
import pandas as pd
from scipy.sparse import issparse

from typing import Literal, Optional
import numpy as np
from sklearn.cluster import KMeans

try:
    from anndata import AnnData
except ImportError as exc:
    raise ImportError("assign_spots requires the `anndata` package") from exc

# ------------------------------------------------------------------------------
# assign spots
# ------------------------------------------------------------------------------

def assign_spots(
    adata: AnnData,
    *,
    method: Literal["kmeans", "grid"] = "kmeans",
    cells_per_spot: int = 100,  # Used for 'kmeans'
    grid_x: int | None = None,  # Used for 'grid'
    grid_y: int | None = None,  # Used for 'grid'
    spot_id_key: str = "spot_id",
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Assign spatial spots to cells based on either KMeans clustering or grid tiling.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with spatial coordinates in `adata.obsm['spatial']`.
    method : {'kmeans', 'grid'}, optional
        Method for assigning spatial spots. 'kmeans' applies KMeans clustering,
        'grid' divides the spatial domain into a regular grid.
    cells_per_spot : int, optional
        Target number of cells per spot (used for 'kmeans').
    grid_x : int, optional
        Number of grid divisions along the X axis (used for 'grid').
    grid_y : int, optional
        Number of grid divisions along the Y axis (used for 'grid').
    spot_id_key : str, optional
        Key under which spot IDs will be stored in `adata.obs`.
    copy : bool, optional
        If True, operate on a copy and return it; otherwise modify in place.

    Returns
    -------
    Optional[AnnData]
        The annotated AnnData if copy=True, else None.
        added_key: str, default="spot_id"
            The key in `.obs` to store the spot IDs.

    Examples
    --------
    >>> from tic.graph.pp import assign_spots
    >>> adata = assign_spots(adata, method='kmeans', cells_per_spot=50)
    >>> adata = assign_spots(adata, method='grid', grid_x=10, grid_y=10)
    """
    if copy:
        adata = adata.copy()

    if "spatial" not in adata.obsm:
        raise KeyError("Spatial coordinates missing: adata.obsm['spatial'] is required.")

    coords = adata.obsm["spatial"]

    if method == "kmeans":
        # KMeans clustering for spot assignment
        n_cells = adata.n_obs
        n_clusters = max(1, n_cells // cells_per_spot)

        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(coords)
        adata.obs[spot_id_key] = labels.astype(str)

    elif method == "grid":
        # Grid-based spot assignment
        if grid_x is None or grid_y is None:
            raise ValueError("grid_x and grid_y must be provided for grid-based spot assignment.")

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        x_edges = np.linspace(x_min, x_max, grid_x + 1)
        y_edges = np.linspace(y_min, y_max, grid_y + 1)

        x_idx = np.digitize(coords[:, 0], x_edges) - 1
        y_idx = np.digitize(coords[:, 1], y_edges) - 1

        labels = x_idx + y_idx * grid_x
        adata.obs[spot_id_key] = labels.astype(str)

    else:
        raise ValueError(f"Invalid method: {method!r}. Choose 'kmeans' or 'grid'.")

    return adata if copy else None

# ------------------------------------------------------------------------------
# aggregate by spot
# ------------------------------------------------------------------------------

def aggregate_by_spot(
    adata: AnnData,
    *,
    spot_id_key: str = "spot_id",
    cell_type_key: str = "cell_type",
) -> AnnData:
    """
    Aggregate single-cell data to spot-level AnnData.

    Parameters
    ----------
    adata : AnnData
        Cell-level AnnData with `obs[spot_id_key]` indicating spot assignments.
    spot_id_key : str, optional
        Column in `adata.obs` defining spot membership.

    Returns
    -------
    AnnData
        Spot-level AnnData with:
          - X: (n_spots, n_genes) summed expression per spot.
          - obs['cell_count']: number of cells per spot.
          - obs_names: spot identifiers prefixed with 'spot_'.
          - var: copied from input AnnData.
          - obsm['spatial']: mean coordinates per spot.
          - obsm['cell_fraction']: fractions per cell type (if `adata.obs['cell_type']` exists).
          - uns['cell_types']: list of cell type categories.
    """
    if spot_id_key not in adata.obs:
        raise KeyError(f"Missing spot assignment: adata.obs[{spot_id_key!r}] is required.")

    spot_ids = adata.obs[spot_id_key].astype(str)
    sorted_spot_ids = sorted(spot_ids.unique())

    # Aggregate expression
    if issparse(adata.X):
        expr_list = [
            adata[spot_ids == sid].X.sum(axis=0)
            for sid in sorted_spot_ids
        ]
        X_spot = np.vstack(expr_list)
    else:
        df_expr = pd.DataFrame(
            adata.X, index=spot_ids, columns=adata.var_names
        )
        X_spot = df_expr.groupby(level=0).sum().loc[sorted_spot_ids].values

    # Cell counts per spot
    counts = spot_ids.value_counts().loc[sorted_spot_ids]
    obs = pd.DataFrame(
        {"cell_count": counts.values},
        index=[f"spot_{sid}" for sid in sorted_spot_ids]
    )

    # Compute spot center coordinates
    df_coords = pd.DataFrame(
        adata.obsm["spatial"],
        index=spot_ids,
        columns=("x", "y")
    )
    spatial_means = df_coords.groupby(level=0).mean().loc[sorted_spot_ids].values
    obsm: dict[str, np.ndarray] = {"spatial": spatial_means}

    # Cell-type fractions
    cell_types: list[str] = []
    if cell_type_key in adata.obs:
        ct_series = adata.obs[cell_type_key].astype(str)
        dummies = pd.get_dummies(ct_series)
        dummies[spot_id_key] = spot_ids.values
        frac_df = (
            dummies.groupby(spot_id_key)
            .sum()
            .div(counts, axis=0)
            .reindex(sorted_spot_ids, fill_value=0)
        )
        obsm["cell_fraction"] = frac_df.values
        cell_types = sorted(ct_series.unique())

    # Build spot-level AnnData
    spot_adata = AnnData(
        X=X_spot,
        obs=obs,
        var=adata.var.copy(),
        obsm=obsm,
    )
    if cell_types:
        spot_adata.uns["cell_types"] = cell_types

    return spot_adata
