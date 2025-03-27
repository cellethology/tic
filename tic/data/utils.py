import numpy as np
import pandas as pd
import anndata
from typing import Any, Dict, List

def build_ann_data(
    cells: List[Any],                # List of Cell-like objects with at least .cell_id and .pos
    X: np.ndarray,                   # Expression (or representation) matrix (n_cells x n_features)
    extra_obs: List[Dict[str, Any]], # Per-cell metadata dictionaries
    uns: Dict[str, Any],             # Dictionary to store additional info (e.g., data_level, tissue_id)
    feature_names: List[str] = None  # Optional list of variable/feature names
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