"""
Module: tic.data.utils

Provides utility functions for building standardized AnnData objects and
validating AnnData structures for Tissue reconstruction.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import anndata


def build_ann_data(
    cells: List[Any],
    X: np.ndarray,
    extra_obs: List[Dict[str, Any]],
    uns: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> anndata.AnnData:
    """
    Build a standardized AnnData object from a list of cell-like objects.

    The resulting AnnData has:
      - obs: a DataFrame of cell metadata with index set to cell IDs.
      - var: a DataFrame of feature names.
      - obsm: a dictionary containing spatial coordinates under "spatial".
      - uns: a dictionary with additional information.

    Parameters
    ----------
    cells : List[Any]
        A list of cell-like objects that have attributes 'cell_id' and 'pos'.
    X : np.ndarray
        A 2D array of shape (n_cells, n_features).
    extra_obs : List[Dict[str, Any]]
        A list of metadata dictionaries for each cell.
    uns : Dict[str, Any]
        A dictionary to store extra information (e.g., 'data_level', 'tissue_id').
    feature_names : Optional[List[str]]
        A list of feature names for var.index. If None, default names are generated.

    Returns
    -------
    anndata.AnnData
        The constructed AnnData object.
    """
    n_cells = len(cells)
    if X.shape[0] != n_cells:
        raise ValueError("X must have the same number of rows as there are cells.")

    obs_df = pd.DataFrame(extra_obs)
    obs_df.index = [c.cell_id for c in cells]

    if feature_names is None:
        n_features = X.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]
    var_df = pd.DataFrame(index=feature_names)

    obsm = {}
    if all(hasattr(c, "pos") for c in cells):
        obsm["spatial"] = np.array([list(c.pos) for c in cells])

    adata = anndata.AnnData(X=X, obs=obs_df, var=var_df, obsm=obsm)
    for key, value in uns.items():
        adata.uns[key] = value

    return adata


def check_anndata_for_tissue(adata: anndata.AnnData) -> None:
    """
    Validate that the provided AnnData contains the required keys and structure
    for Tissue.from_anndata().

    Required:
      - adata.obsm contains a "spatial" key with a 2D array.
      - adata.obs includes 'cell_type' and 'size' columns.
      - adata.var.index is not empty.
      - Dimensions of adata.X match those of adata.obs and adata.var.

    Raises
    ------
    ValueError
        If any required component is missing or mismatched.
    """
    if "spatial" not in adata.obsm:
        raise ValueError(
            "AnnData.obsm is missing the 'spatial' key, which is required for Tissue.from_anndata()."
        )

    required_obs_cols = {"cell_type", "size"}
    missing_obs = required_obs_cols - set(adata.obs.columns)
    if missing_obs:
        raise ValueError(
            f"AnnData.obs is missing required columns: {missing_obs}. "
            "Please include 'cell_type' and 'size'."
        )

    if adata.var.index.empty:
        raise ValueError("AnnData.var.index is empty. No biomarker/gene names were found.")

    n_cells, n_features = adata.X.shape
    if n_cells != adata.obs.shape[0]:
        raise ValueError(
            f"Number of rows in AnnData.X ({n_cells}) does not match number of observations ({adata.obs.shape[0]})."
        )
    if n_features != adata.var.shape[0]:
        raise ValueError(
            f"Number of columns in AnnData.X ({n_features}) does not match number of variables ({adata.var.shape[0]})."
        )

    spatial = adata.obsm["spatial"]
    if spatial.ndim != 2:
        raise ValueError(
            f"Spatial coordinates (adata.obsm['spatial']) should be a 2D array, got shape {spatial.shape}."
        )