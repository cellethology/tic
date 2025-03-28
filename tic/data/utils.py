import numpy as np
import pandas as pd
import anndata
from typing import Any, Dict, List

def build_ann_data(
    cells: List[Any],                
    X: np.ndarray,                   
    extra_obs: List[Dict[str, Any]], 
    uns: Dict[str, Any],             
    feature_names: List[str] = None  
) -> anndata.AnnData:
    """
    Build a standardized AnnData object from a list of cell-like objects.
    
    The output AnnData object has:
      - obs: a DataFrame with cell metadata. Its index is set to each cell's cell_id 
      - var: a DataFrame with feature names (if not provided, default names are generated).
      - obsm: a dictionary storing spatial coordinates under the key "spatial".
      - uns: a dictionary updated with additional info.
    
    :param cells: A list of cell-like objects with attributes 'cell_id' and 'pos'.
    :param X: A 2D NumPy array with shape (n_cells, n_features).
    :param extra_obs: A list of dictionaries containing metadata for each cell.
    :param uns: A dictionary to store additional information (like 'data_level', 'tissue_id', etc.).
    :param feature_names: Optional list of feature names for var.index. If None, default names are generated.
    :return: An AnnData object with consistent obs, var, obsm, and uns structure.
    """
    n_cells = len(cells)
    if X.shape[0] != n_cells:
        raise ValueError("X must have the same number of rows as there are cells.")
    
    # Create obs DataFrame from extra_obs.
    obs_df = pd.DataFrame(extra_obs)

    obs_df.index = [c.cell_id for c in cells]
    
    # Build var from feature_names (or generate default names if None).
    if feature_names is None:
        n_features = X.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]
    var_df = pd.DataFrame(index=feature_names)
    
    # Build obsm: store spatial coordinates from each cell's 'pos' attribute.
    obsm = {}
    if all(hasattr(c, "pos") for c in cells):
        obsm["spatial"] = np.array([list(c.pos) for c in cells])
    
    # Create the AnnData object.
    adata = anndata.AnnData(X=X, obs=obs_df, var=var_df, obsm=obsm)
    
    # Update uns with additional information.
    for key, value in uns.items():
        adata.uns[key] = value
    return adata

def check_anndata_for_tissue(adata: anndata.AnnData) -> None:
    """
    Check if the provided AnnData object contains the required keys and structures for Tissue.from_anndata().

    Required:
      - adata.obsm contains a "spatial" key with a 2D array of spatial coordinates.
      - adata.obs must include 'cell_type' and 'size' columns.
      - adata.var.index must not be empty (i.e., there should be at least one biomarker/gene).
      - The dimensions of adata.X must match those of adata.obs and adata.var.

    Raises:
      ValueError: if any of the required components is missing or mismatched.
    """
    # Check for spatial coordinates
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData.obsm is missing the 'spatial' key, which is required for Tissue.from_anndata().")
    
    # Check that obs contains the required columns
    required_obs_cols = {"cell_type", "size"}
    missing_obs = required_obs_cols - set(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"AnnData.obs is missing required columns: {missing_obs}. Please include 'cell_type' and 'size'.")
    
    # Check that var.index is not empty
    if adata.var.index.empty:
        raise ValueError("AnnData.var.index is empty. No biomarker/gene names were found.")
    
    # Verify that X dimensions are consistent with obs and var
    n_cells, n_features = adata.X.shape
    if n_cells != adata.obs.shape[0]:
        raise ValueError(f"The number of rows in AnnData.X ({n_cells}) does not match the number of observations ({adata.obs.shape[0]}).")
    if n_features != adata.var.shape[0]:
        raise ValueError(f"The number of columns in AnnData.X ({n_features}) does not match the number of variables ({adata.var.shape[0]}).")
    
    # ensure the spatial coordinates are 2D
    spatial = adata.obsm["spatial"]
    if spatial.ndim != 2:
        raise ValueError(f"Spatial coordinates (adata.obsm['spatial']) should be a 2D array, but got shape {spatial.shape}.")